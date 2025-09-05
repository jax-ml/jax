# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import enum
import sys
from collections.abc import Sequence
import dataclasses
import itertools
import logging
import os.path
import re
from typing import Any, Callable, NamedTuple, Type, Union
from functools import partial

import numpy as np

from jax._src import config
from jax._src import dtypes
from jax._src import path
from jax._src import tree_util
from jax._src.util import safe_map

from jax._src.repro.tracker import (
  _thread_local_state, flags_override, Call, Func,
)

def maybe_singleton(x: list[Any]) -> Any | tuple[Any, ...]:
  return x[0] if len(x) == 1 else tuple(x)

_flat_index_re = re.compile(r"<flat index (\d+)>")
def keystr_no_flat_index(p: tree_util.KeyPath) -> str:
  return _flat_index_re.sub("\\1", tree_util.keystr(p))


# Maps types to source code emitters
_operand_emitter: dict[Type, Callable[["EmitFuncContext", Any], str]] = {}
_operand_emitter_initialized = False  # For lazy initialization, for circular imports

def register_emitter(typ, emitter: Callable[["EmitFuncContext", Any], str]) -> None:
  """Registers `emitter` to use to emit operands of type `typ`"""
  _operand_emitter[typ] = emitter


class EmitLiterally:
  # These are used as replacements for arguments to be emitted literally.
  def __init__(self, literal: str):
    self.literal = literal


def initialize_operand_emitter():
  # We wrap this in a function that is called late so that we can import here
  # other JAX modules
  global _operand_emitter_initialized
  _operand_emitter_initialized = True

  def emit_enum(enum_name: str) -> Callable[["EmitFuncContext", enum.Enum], str]:
    # For classes that derive from enum.Enum
    def emitter(ctx: "EmitFuncContext", v: enum.Enum):
      return f"{enum_name}.{v.name}"
    return emitter

  def emit_namedtuple(named_tuple_name: str) -> Callable[["EmitFuncContext", NamedTuple], str]:
    def emitter(ctx: "EmitFuncContext", v: NamedTuple):
      return f"{named_tuple_name}({ctx.traverse_sequence_value(v)})"  # type: ignore
    return emitter

  _operand_emitter[set] = lambda ctx, v: repr(v)
  _operand_emitter[frozenset] = lambda ctx, v: repr(v)
  _operand_emitter[EmitLiterally] = lambda ctx, v: v.literal

  _operand_emitter[slice] = lambda ctx, v: repr(v)
  _operand_emitter[type(...)] = lambda ctx, v: "..."

  from jax._src import literals
  _operand_emitter[literals.TypedNdArray] = (
      lambda ctx, v: f"literals.TypedNdArray({ctx.traverse_value_atom(v.val)}, weak_type={v.weak_type})")

  from jax._src import core  # type: ignore
  _operand_emitter[core.Primitive] = (
      lambda ctx, v: f"jax_primitive_bind(\"{v}\")")

  def emit_np_array(ctx: EmitFuncContext, v) -> str:
    if v.size <= _thread_local_state.flags.fake_array_threshold:
      res = repr(v)
    else:
      res = f"np.ones({v.shape}, dtype={ctx.traverse_value_atom(v.dtype)})"
    return ctx.named_value(res, prefix="arr")

  _operand_emitter[np.ndarray] = emit_np_array
  _operand_emitter[np.generic] = emit_np_array

  def emit_dtype(ctx: "EmitFuncContext", dt):
    return f"dtypes.dtype(\"{dt.name}\")"
  for t in dtypes._jax_types:
    _operand_emitter[type(t)] = emit_dtype

  from jax._src.array import ArrayImpl  # type: ignore
  @partial(register_emitter, ArrayImpl)
  def emit_jax_array(ctx: EmitFuncContext, v) -> str:
    # TODO: emit it as a real jax.Array
    return emit_np_array(ctx, np.array(v))

  @partial(register_emitter, core.ShapeDtypeStruct)
  def emit_ShapeDtypeStruct(ctx: "EmitFuncContext", v: core.ShapeDtypeStruct) -> str:
    dtype = ctx.traverse_value_atom(v.dtype)
    sharding = ctx.traverse_value(v.sharding)
    res = (f"jax.ShapeDtypeStruct({v.shape}, {dtype}, sharding={sharding}, "
           f"weak_type={v.weak_type}, vma={v.vma}, is_ref={v.is_ref})")
    return ctx.named_value(res, prefix="sds")

  @partial(register_emitter, core.ShapedArray)
  def emit_ShapedArray(ctx: "EmitFuncContext", v: core.ShapedArray) -> str:
    dtype = ctx.traverse_value(v.dtype)
    vma = ctx.traverse_value(v.vma)
    memory_space = ctx.traverse_value_atom(v.memory_space)
    return f"core.ShapedArray({ctx.traverse_value(v.shape)}, dtype={dtype}, vma={vma}, memory_space={memory_space})"

  from jax import sharding  # type: ignore

  @partial(register_emitter, sharding.PartitionSpec)
  def emit_PartitionSpec(ctx: "EmitFuncContext", v: sharding.PartitionSpec) -> str:
    partitions = ctx.traverse_sequence_value(v._partitions)
    if partitions:
      partitions += ", "
    res = f"sharding.PartitionSpec({partitions}unreduced={v.unreduced}, reduced={v.reduced})"
    return ctx.named_value(res, prefix="ps")

  from jax._src.lib import _jax
  _operand_emitter[_jax.UnconstrainedSingleton] = (
      lambda ctx, v: "sharding.PartitionSpec.UNCONSTRAINED")

  _operand_emitter[sharding.AxisType] = emit_enum("sharding.AxisType")
  _operand_emitter[core.MemorySpace] = emit_enum("core.MemorySpace")

  @partial(register_emitter, sharding.AbstractMesh)
  def emit_AbstractMesh(ctx: "EmitFuncContext", v: sharding.AbstractMesh) -> str:
    res = (f"sharding.AbstractMesh({v.axis_sizes}, {ctx.traverse_value(v.axis_names)}, "
           f"axis_types={ctx.traverse_value(v.axis_types)}, "
           f"abstract_device={ctx.traverse_value(v.abstract_device)})")
    return ctx.named_value(res, prefix="am")

  from jax._src.lib import xla_client  # type: ignore
  @partial(register_emitter, xla_client.Device)
  def emit_Device(ctx: "EmitFuncContext", v: sharding.Mesh) -> str:
    return f"jax_get_device(\"{v.platform}\", {v.id})"

  from jax._src.lib import xla_client  # type: ignore
  @partial(register_emitter, sharding.AbstractDevice)
  def emit_AbstractDevice(ctx: "EmitFuncContext", v: sharding.AbstractDevice) -> str:
    return f"sharding.AbstractDevice(\"{v.device_kind}\", {v.num_cores})"

  @partial(register_emitter, sharding.Mesh)
  def emit_Mesh(ctx: "EmitFuncContext", v: sharding.Mesh) -> str:
    devices_list = v.devices.tolist()
    devices_str = ctx.traverse_value(devices_list)
    res = (f"sharding.Mesh(np.array({devices_str}), "
           f"axis_names={ctx.traverse_value(v.axis_names)}, "
           f"axis_types={ctx.traverse_value(v.axis_types)})")
    return ctx.named_value(res, prefix="mesh")

  @partial(register_emitter, sharding.NamedSharding)
  def emit_NamedSharding(ctx: "EmitFuncContext", v: sharding.NamedSharding) -> str:
    mesh = ctx.traverse_value_atom(v.mesh)
    spec = ctx.traverse_value_atom(v.spec)
    memory_kind = ctx.traverse_value_atom(v.memory_kind)
    res = f"sharding.NamedSharding({mesh}, {spec}, memory_kind={memory_kind})"
    return ctx.named_value(res, prefix="ns")

  from jax import lax  # type: ignore
  register_emitter(lax.AccuracyMode, emit_enum("lax.AccuracyMode"))
  _operand_emitter[lax.ConvDimensionNumbers] = emit_namedtuple("lax.ConvDimensionNumbers")
  _operand_emitter[lax.DotAlgorithm] = emit_namedtuple("lax.DotAlgorithm")
  _operand_emitter[lax.DotAlgorithmPreset] = emit_enum("lax.DotAlgorithmPreset")
  _operand_emitter[lax.FftType] = emit_namedtuple("lax.FftType")
  _operand_emitter[lax.GatherDimensionNumbers] = emit_namedtuple("lax.GatherDimensionNumbers")
  _operand_emitter[lax.GatherScatterMode] = emit_enum("lax.GatherScatterMode")
  _operand_emitter[lax.Precision] = emit_enum("lax.Precision")
  _operand_emitter[lax.RandomAlgorithm] = emit_enum("lax.RandomAlgorithm")
  _operand_emitter[lax.RoundingMethod] = emit_enum("lax.RoundingMethod")
  _operand_emitter[lax.ScatterDimensionNumbers] = emit_namedtuple("lax.ScatterDimensionNumbers")
  def emit_Tolerance(ctx: "EmitFuncContext", v: lax.Tolerance) -> str:
    return f"lax.Tolerance({v.atol}, {v.rtol}, {v.ulps})"
  _operand_emitter[lax.Tolerance] = emit_Tolerance

  from jax._src import random  # type: ignore
  def emit_PRNGImpl(ctx: "EmitFuncContext", v: random.PRNGImpl) -> str:
    return f"resolve_prng_impl(\"{v.name}\")"
  _operand_emitter[random.PRNGImpl] = emit_PRNGImpl

  from jax._src.prng import PRNGKeyArray  # type: ignore
  @partial(register_emitter, PRNGKeyArray)
  def emit_PRNGKeyArray(ctx: "EmitFuncContext", v: PRNGKeyArray) -> str:
    res = f"prng.PRNGKeyArray({emit_PRNGImpl(ctx, v._impl)}, {ctx.traverse_value_atom(v._base_array)})"
    return ctx.named_value(res, prefix="key")

  _operand_emitter[xla_client.ArrayCopySemantics] = emit_enum("xla_client.ArrayCopySemantics")

  from jax._src.state import indexing  # type: ignore
  @partial(register_emitter, indexing.Slice)
  def emit_Slice(ctx: "EmitFuncContext", v: indexing.Slice) -> str:
    start = ctx.traverse_value_atom(v.start)
    size = ctx.traverse_value_atom(v.size)
    stride = ctx.traverse_value_atom(v.stride)
    return f"indexing.Slice({start}, {size}, {stride})"

  from jax._src.pallas import core as pallas_core  # type: ignore
  from jax.experimental.pallas import tpu as pltpu  # type: ignore
  @partial(register_emitter, pallas_core.GridSpec)
  def emit_GridSpec(ctx: "EmitFuncContext", v: pallas_core.GridSpec) -> str:
    grid = ctx.traverse_value(v.grid)
    in_specs = ctx.traverse_value(v.in_specs)
    out_specs = ctx.traverse_value(v.out_specs)
    scratch_shapes = ctx.traverse_value(v.scratch_shapes)
    if type(v) is pallas_core.GridSpec:
      grid_spec_type = "pallas_core.GridSpec"
      rest = ""
    elif type(v) is pltpu.PrefetchScalarGridSpec:
      grid_spec_type = "pltpu.PrefetchScalarGridSpec"
      rest = f", num_scalar_prefetch={v.num_scalar_prefetch}"
    else:
      assert False
    res = (f"{grid_spec_type}(grid={grid}, "
           f"in_specs={in_specs}, out_specs={out_specs}, "
           f"scratch_shapes={scratch_shapes}{rest})")
    return ctx.named_value(res, prefix="gs")

  register_emitter(pltpu.PrefetchScalarGridSpec, emit_GridSpec)

  register_emitter(pallas_core.NoBlockSpec, lambda ctx, v: "pallas_core.no_block_spec")

  @partial(register_emitter, pallas_core.BlockSpec)
  def emit_BlockSpec(ctx: "EmitFuncContext", v: pallas_core.BlockSpec) -> str:
    index_map = ctx.traverse_value_atom(v.index_map)
    res = (f"pallas_core.BlockSpec(block_shape={v.block_shape}, "
           f"index_map={index_map}, memory_space={v.memory_space}, "
           f"pipeline_mode={v.pipeline_mode})")
    return ctx.named_value(res, prefix="bs")

  @partial(register_emitter, pallas_core._IndexMapFunc)
  def emit_IndexMapFunc(ctx: "EmitFuncContext", v: pallas_core._IndexMapFunc) -> str:
    # The _IndexMapFunc is added upon construction of BlockSpec, it is
    # ok to strip it when emitting
    return ctx.traverse_value_atom(v.index_map)

  from jax.experimental.pallas import tpu as pltpu  # type: ignore
  @partial(register_emitter, pltpu.CompilerParams)
  def emit_CompilerParams(ctx: "EmitFuncContext", v: pltpu.CompilerParams) -> str:
    dimension_semantics = ctx.traverse_value(v.dimension_semantics)
    allow_input_fusion = ctx.traverse_value(v.allow_input_fusion)
    flags = ctx.traverse_value(v.flags)
    kernel_type = ctx.traverse_value_atom(v.kernel_type)
    res = (f"pltpu.CompilerParams(dimension_semantics={dimension_semantics}, "
           f"allow_input_fusion={allow_input_fusion}, vmem_limit_bytes={v.vmem_limit_bytes}, "
           f"collective_id={v.collective_id}, has_side_effects={v.has_side_effects}, "
           f"flags={flags}, internal_scratch_in_bytes={v.internal_scratch_in_bytes}, "
           f"serialization_format={v.serialization_format}, kernel_type={kernel_type}, "
           f"disable_bounds_checks={v.disable_bounds_checks}, skip_device_barrier={v.skip_device_barrier}, "
           f"allow_collective_id_without_custom_barrier={v.allow_collective_id_without_custom_barrier})")
    return ctx.named_value(res, prefix="cp")

  _operand_emitter[pltpu.KernelType] = emit_enum("pltpu.KernelType")
  _operand_emitter[pltpu.GridDimensionSemantics] = emit_enum("pltpu.GridDimensionSemantics")
  _operand_emitter[pltpu.MemorySpace] = emit_enum("pltpu.MemorySpace")
  _operand_emitter[pltpu.SemaphoreType] = emit_enum("pltpu.SemaphoreType")

  @partial(register_emitter, pallas_core.MemoryRef)
  def emit_MemoryRef(ctx: "EmitFuncContext", v: pallas_core.MemoryRef) -> str:
    inner_aval = ctx.traverse_value_atom(v.inner_aval)
    memory_space = ctx.traverse_value_atom(v.memory_space)
    res = f"pallas_core.MemoryRef({inner_aval}, {memory_space})"
    return ctx.named_value(res, prefix="mr")


@dataclasses.dataclass
class EmittedFunction:
  invocation: Call

  lines: list[str]
  # The immediate externals are those referenced in the `lines`, using
  # global variable names
  immediate_externals: dict[int, Any]
  # All externals include the `immediate_externals` and also those referenced
  # in functions that are recursively referenced by the `immediate_externals`.
  all_externals: dict[int, Any]


class EmitGlobalContext:
  _var_for_val: dict[int, str] # id(val) -> var_name

  def __init__(self):
    self._var_for_val = {}
    # Initialize the emitter rules
    if not _operand_emitter_initialized:
      initialize_operand_emitter()  # type: ignore

  def var_for_val(self, for_v: Any, *, prefix="v") -> str:
    v_id = id(for_v)
    vn = self._var_for_val.get(v_id, None)
    if vn is None:
      if isinstance(for_v, Func):
        vn = for_v.python_name()
      else:
        vn = f"{prefix}_{_thread_local_state.small_id(v_id)}"
      self._var_for_val[v_id] = vn
    return vn


  def emit_function(self, invocation: Call,
                    parent_ctx: Union["EmitFuncContext", None]
                    ) -> EmittedFunction:
    if invocation.emitted_function is None:
      invocation.emitted_function = EmitFuncContext(self, invocation, parent_ctx).emit_body()
    return invocation.emitted_function

# A Path is a string representation of an indexer into a pytree value.
Path = str

class EmitFuncContext:

  def __init__(self, global_ctx: EmitGlobalContext, invocation: Call,
               parent_ctx: Union["EmitFuncContext", None]):
    self.global_ctx: EmitGlobalContext = global_ctx
    self.parent_ctx = parent_ctx
    self.invocation = invocation  # The body we are emitting
    self.lines: list[str] = []
    self.local_index = itertools.count()
    # For all the values defined here, their local name (by id)
    self.definitions: dict[int, str] = {}
    # We map some value definition to a value name
    self.named_values: dict[str, str] = {}

    self.externals: dict[int, Any] = {}  # direct externals, by id
    self.externals_from_nested: dict[int, Any] = {}  # externals from nested
    self.indent: int = 0
    self.current_traceback: Union["Traceback", None] = None  # type: ignore  # noqa: F821

  def new_local_name(self, *, prefix="v") -> str:
    return f"{prefix}_{next(self.local_index)}"

  def define_value(self, v: Any, vn: str) -> None:
    v_id = id(v)
    if v_id not in self.definitions:
      # Sometimes we see a function invoked with the same value for multiple
      # args. Keep the first definition.
      self.definitions[v_id] = vn

  def named_value(self, val_str: str, *, prefix="v") -> str:
    vn = self.named_values.get(val_str)
    if vn is None:
      vn = self.new_local_name(prefix=prefix)
      self.named_values[val_str] = vn
      self.emit_line(f"{vn} = {val_str}")
    return vn

  def use_value(self, v: Any) -> str:
    v_id = id(v)
    frame = self
    while True:
      if (vn := frame.definitions.get(v_id)) is not None:
        if frame is self:
          return vn
        else:
          self.externals[v_id] = v
          return self.global_ctx.var_for_val(v, prefix="g")
      frame = frame.parent_ctx  # type: ignore
      if not frame: break

    if _thread_local_state.undefined_value_handler:
      return _thread_local_state.undefined_value_handler(self, v)

    vn = self.global_ctx.var_for_val(v, prefix="g")
    from jax._src import core  # type: ignore
    if not isinstance(v, (core.Tracer, Func, tree_util.Partial)):
      _thread_local_state.warn_or_error(
          f"Found non-tracer without custom emitter: {v} of type {type(v)}. Using global name {vn}.",
          traceback=self.current_traceback)
    else:
      _thread_local_state.warn_or_error(f"Using undefined value {v}. Using global name {vn}.",
                                        traceback=self.current_traceback)
    return vn

  def emit_line(self, l: str):
    self.lines.append(" " * self.indent + l)

  def traverse_value(
      self, a, path: Path = "", paths: list[tuple[Path, Any]] | None = None) -> str:
    """We traverse values to eliminate custom pytree nodes (which should not
    be in the repro). Those nodes are turned into tuples of leaves. The top-level
    standard Python containers (lists, tuples, dicts, None) are kept. We also
    keep those containers for which we have special `_operand_emitter` registered.

    We use this traversal in two cases: (1) to emit a string representation
    that can be used when the value is passed as argument to a call, or in a
    return statement. In this case `path` is `None` and the function returns a
    string. And, (2) to construct a list of paths, e.g., `[0]['field']` to
    access components of the rendered value when accessing a function definition
    arguments the values returned from a call. In this case, the returned
    string is not used.
    """
    if isinstance(a, tuple) and not hasattr(a, "_fields"):  # skip NamedTuple
      return f"({self.traverse_sequence_value(a, path, paths)}{',' if a else ''})"
    if isinstance(a, list):
      return f"[{self.traverse_sequence_value(a, path, paths)}]"
    if isinstance(a, dict):
      return f"{{{self.traverse_key_value_sequence(a, path, paths,
                                                   keys_are_identifiers=False)}}}"
    return str(self.traverse_value_atom(a, path, paths))

  def traverse_sequence_value(
      self, a: Sequence, path: Path = "", paths: list[tuple[Path, Any]] | None = None) -> str:
    return ", ".join(self.traverse_value(v, path + f"[{i}]", paths)
                     for i, v in enumerate(a))

  def traverse_key_value_sequence(
      self, a, path: Path = "", paths: list[tuple[Path, Any]] | None = None,
      keys_are_identifiers=True) -> str:
    if keys_are_identifiers:
      return ", ".join(f"{k}={self.traverse_value(v, path + f'[{repr(str(k))}]', paths)}"
                       for k, v in a.items())
    else:
      return ", ".join(f"'{k}': {self.traverse_value(v, path + f'[{repr(str(k))}]', paths)}"
                       for k, v in a.items())

  def traverse_value_atom(
      self, v: Any, path: Path = "", paths: list[tuple[Path, Any]] | None = None) -> str:
    # Emit first the data types for which we have special rules, because they
    # may also be pytrees and we don't want to loose their type.
    def found_leaf():
      paths.append((path, v))
      return ""

    if v is None:
      if paths is not None: return ""
      return repr(v)
    if isinstance(v, (int, float, bool, str, complex)):
      if not isinstance(v, enum.IntEnum):
        if paths is not None: return found_leaf()
        return repr(v)
    for t in type(v).__mro__:
      if (emitter := _operand_emitter.get(t)) is not None:
        # If 'v' has an emitter, it is a leaf. Don't call the emitter if
        # we are only collecting paths to leaves
        if paths is not None: return found_leaf()
        return emitter(self, v)

    v_type_str = str(type(v))
    if "flax.core.axes_scan._Broadcast" in v_type_str:  # TODO: flax
      if paths is not None: return found_leaf()
      return "flax.core.axes_scan.broadcast"

    if isinstance(v, Func):
      if v.api_name:
        if paths is not None: return found_leaf()
        return v.api_name
      if not v.is_jax:
        # USER functions, emit them to collect the externals
        if paths is not None: return found_leaf()
        self.emit_operand_user_func(v)
        return v.python_name()
      if paths is not None: return found_leaf()
      return self.use_value(v)

    # Perhaps this is a user-defined pytree node, we flatten it but only
    # one level. We need to keep structure underneath, for APIs like
    # vmap in_axes that use tree prefix matching.
    try:
      vs = tree_util.flatten_one_level(v)[0]
    except ValueError:
      pass  # v is not an actual tree
    else:
      vs = maybe_singleton(vs)
      if vs is not v:
        return self.traverse_value(vs, path, paths)

    if paths is not None: return found_leaf()
    return self.use_value(v)

  def emit_operand_user_func(self, f: Func) -> None:
    if not f.invocation:
      # A function that was never invoked is emitted in the main function
      if self.invocation.func.is_main:
        self.emit_line(f"def {self.global_ctx.var_for_val(f, prefix='fun')}(*args, **kwargs):")
        self.emit_line("  pass")
        self.emit_line("")
      else:
        self.externals[id(f)] = f
      return
    f_emitted = self.global_ctx.emit_function(f.invocation, self)
    # We emit the externals here if we are in the main function, or if
    # any of the externals are defined here. Otherwise, we will emit these
    # externals in some enclosing function.
    if (self.invocation.func.is_main or
        any(ae in self.definitions for ae in f_emitted.all_externals)):
      # Emit (or mark as externals) the external functions
      for e in f_emitted.immediate_externals.values():
        if isinstance(e, Func) and not e.is_jax:
          self.emit_operand_user_func(e)
        elif id(e) in self.definitions:  # Defined here
          self.emit_line(f"{self.global_ctx.var_for_val(e)} = {self.definitions[id(e)]}")
        else:  # External for us too
          self.externals[id(e)] = e
      for bl in f_emitted.lines:
        self.emit_line(bl)
      return

    # We make this function all external
    self.externals[id(f)] = f
    for ae in f_emitted.all_externals.values():
      self.externals_from_nested[id(ae)] = ae

  @staticmethod
  def filter_statics(args, kwargs,
                     static_argnums: tuple[int, ...]=(),
                     static_argnames: tuple[str, ...]=(),
                     ) -> tuple[tuple[Any, ...], dict[str, Any]]:
    dyn_args = []
    dyn_kwargs = {}
    # TODO: we should try to get this directly from JAX, there is a slight
    # chance we'll do something different otherwise. This is in C++ in JAX.
    static_argnums = {i if i >= 0 else len(args) - i for i in static_argnums}
    for i, a in enumerate(args):
      if i not in static_argnums:
        dyn_args.append(a)
    for k, a in sorted(kwargs.items()):
      if k not in static_argnames:
        dyn_kwargs[k] = a
    return tuple(dyn_args), dyn_kwargs

  @staticmethod
  def jax_jit_call_filter_statics(jitted_fun: Func,
                                  jit_kwargs: dict[str, Any],
                                  args: tuple[Any, ...],
                                  kwargs: dict[str, Any],
                                  ) -> tuple[tuple[Any, ...],
                                             dict[str, Any],
                                             dict[str, Any]]:
    """Dynamic args and kwargs, and also new jit_kwargs, given a jax_jit_call."""
    from jax._src import api_util  # type: ignore
    # Drop the static_argnums and static_argnames from the jit_kwargs
    new_jit_kwargs = dict(jit_kwargs)
    if "static_argnums" in jit_kwargs: del new_jit_kwargs["static_argnums"]
    if "static_argnames" in jit_kwargs: del new_jit_kwargs["static_argnames"]

    jitted_fun_sig = api_util.fun_signature(jitted_fun.fun)
    if jitted_fun_sig is None: # E.g., for built-in functions, or np.dot
      static_argnums: tuple[int, ...] = ()
      static_argnames: tuple[str, ...] = ()
    else:
      static_argnums, static_argnames = api_util.infer_argnums_and_argnames(
          jitted_fun_sig, jit_kwargs.get("static_argnums"),
          jit_kwargs.get("static_argnames"))
    dyn_args, dyn_kwargs = EmitFuncContext.filter_statics(
        args, kwargs,
        static_argnums=static_argnums, static_argnames=static_argnames)
    return dyn_args, dyn_kwargs, new_jit_kwargs

  def preprocess_args_for_call(self, call: Call) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """We are emitting a function call to `call`."""
    if call.func.api_name in ("jax_jit_call", "jax_jit_aot_trace_call",
                              "jax_jit_aot_lower_call",
                              "jax_pjit_call", "jax_pjit_aot_trace_call",
                              "jax_pjit_aot_lower_call"):
      jitted_fun, jit_ctx_mesh, jit_args, jit_kwargs, *rest_args = call.args
      # TODO: this is a hack: we save the jit_kwargs into the invocation of
      # the jitted function, so that we can process the static arguments
      if jitted_fun.invocation:  # invocation may be missing in case of errors
        jitted_fun.invocation.jit_kwargs = jit_kwargs
      dyn_args, dyn_kwargs, new_jit_call_kwargs = EmitFuncContext.jax_jit_call_filter_statics(
          jitted_fun, jit_kwargs, tuple(rest_args), call.kwargs)
      all_args = (jitted_fun, jit_ctx_mesh, jit_args, new_jit_call_kwargs, *dyn_args)
      return all_args, dyn_kwargs

    if call.func.api_name == "jax_checkpoint_call":
      from jax.ad_checkpoint import checkpoint_policies  # type: ignore
      f, t_args, t_kwargs, *rest_args = call.args
      new_t_kwargs = dict(t_kwargs)
      if (policy := new_t_kwargs.get("policy")) is not None:
        matching = [(policy_name, getattr(checkpoint_policies, policy_name))
                     for policy_name in dir(checkpoint_policies)
                     if policy is getattr(checkpoint_policies, policy_name)]
        if not matching:
          _thread_local_state.warn_or_error(
            f"Unrecognized jax.checkpoint policy: {policy}. Using 'dots_saveable'",
            traceback=self.current_traceback)
          matching = [("dots_saveable", checkpoint_policies.dots_saveable)]
        new_t_kwargs["policy"] = EmitLiterally(f"jax.checkpoint_policies.{matching[0][0]}")
        new_args = (f, t_args, new_t_kwargs, *rest_args)
        return (new_args, call.kwargs)

    return (call.args, call.kwargs)


  def emit_body(self) -> EmittedFunction:
    """Emits the body of the function.
    This is called starting with the main function.
    """
    invocation = self.invocation
    assert not invocation.func.is_jax

    self.current_traceback = invocation.traceback
    self.emit_line("")
    self.emit_line(f"# body from invocation {invocation}")
    if hasattr(invocation, "jit_kwargs"):
      inv_args, inv_kwargs, _ = EmitFuncContext.jax_jit_call_filter_statics(
          invocation.func, invocation.jit_kwargs, invocation.args, invocation.kwargs,
      )
    else:
      inv_args, inv_kwargs = invocation.args, invocation.kwargs

    arg_names = [self.new_local_name() for _ in inv_args]
    args_str = ", ".join(arg_names)
    if inv_kwargs:
      args_str += ", *, " + ", ".join(inv_kwargs.keys())
    self.emit_line(f"def {self.global_ctx.var_for_val(invocation.func)}({args_str}):")
    self.indent += 2
    paths: list[tuple[Path, Any]] = []
    safe_map(lambda an, a: self.traverse_value(a, an, paths), arg_names, inv_args)
    safe_map(lambda ka: self.traverse_value(ka[1], ka[0], paths), inv_kwargs.items())
    for pth, a in paths:
      self.define_value(a, pth)
    del inv_args, inv_kwargs

    last_res_str = None  # The last result
    for c in invocation.body:  # For each call statement in the body
      self.current_traceback = c.traceback
      last_res_str = self.emit_call(c)

    # The end of the function
    self.current_traceback = invocation.traceback
    if invocation.func.is_main:
      # Return the last result
      assert invocation.result is None
      self.emit_line(f"return {last_res_str}")
    else:
      self.emit_line(f"return {self.traverse_value(invocation.result)}")
    all_externals = dict(self.externals)  # copy
    for e in self.externals_from_nested.values():
      all_externals[id(e)] = e
    return EmittedFunction(self.invocation, self.lines, self.externals, all_externals)

  def emit_call(self, c: Call) -> str:
    if isinstance(c.func, Func):
      args, kwargs = self.preprocess_args_for_call(c)
    else:
      args, kwargs = c.args, c.kwargs
    func = c.func
    result = c.result

    result_paths: list[tuple[Path, Any]] = []
    self.traverse_value(result, "", result_paths)

    pre_res = _thread_local_state.emit_call_preprocessor(
        self, func, args, kwargs, result_paths)
    if pre_res is None: return "_"
    func, args, kwargs, result_path = pre_res

    callee_name_str = self.traverse_value_atom(func)
    call_args_str = self.traverse_sequence_value(args)
    if kwargs:
      call_args_str += (", " if args else "") + self.traverse_key_value_sequence(kwargs,
                                                                                 keys_are_identifiers=True)

    overall_res_name = self.new_local_name()
    self.emit_line(f"{overall_res_name} = {callee_name_str}({call_args_str})  # {c}")
    for pth, r in result_paths:
      self.define_value(r, overall_res_name + pth)

    return overall_res_name or "_"


def emit_repro(top_entry: Call,
               extra_message: str,
               repro_name_prefix: str = "jax_repro"):
  np.set_printoptions(threshold=sys.maxsize)  # Do not summarize arrays

  assert config.repro_dir.value
  dump_to = config.repro_dir.value
  if not (out_dir := path.make_jax_dump_dir(dump_to)):
    return None
  fresh_id = itertools.count()
  while True:
    repro_path = out_dir / f"{repro_name_prefix}_{next(fresh_id)}.py"
    if not os.path.exists(repro_path):
      break
  assert not top_entry.func.is_jax
  # Collect the relevant functions, functions earlier in the list may depend
  # on function later in the list. We will emit the code starting with the
  # last function.
  global_ctx = EmitGlobalContext()
  main_invocation = top_entry
  if not top_entry.func.is_main:
    assert top_entry.parent is not None, top_entry
    main_invocation = top_entry.parent.parent  # type: ignore
  assert main_invocation.parent is None

  assert not _thread_local_state.emitting_repro
  _thread_local_state.emitting_repro = True
  _thread_local_state.emitting_repro_had_errors = False
  try:
    main_emitted = global_ctx.emit_function(main_invocation, None)
    preamble = f"""
# This file was generated by JAX repro extractor.

import jax
from jax._src import config

from jax.experimental.repro_runtime import *

if config.enable_x64.value != {config.enable_x64.value}:
  raise ValueError("This repro was saved with JAX_ENABLE_X64={config.enable_x64.value}."
                   "You should run with with the same value of the flag.")

# Repro was saved due to:
"""
    preamble += "\n".join([f"  # {l}" for l in extra_message.split("\n")]) + "\n\n"
    post_amble = """

if __name__ == "__main__":
  main_repro()
"""
    repro_source = preamble + "\n".join(main_emitted.lines) + post_amble
    logging.warning(f"Dumped JAX repro at {repro_path} due to {extra_message[:128]}...")
    repro_path.write_text(repro_source)
  finally:
    _thread_local_state.emitting_repro = False

  _thread_local_state.last_repro = (str(repro_path), repro_source)
  if (_thread_local_state.emitting_repro_had_errors and
      _thread_local_state.flags.enable_checks_as_errors):
    _thread_local_state.warn_or_error(
        "There were errors during repro emitting. See the error log."
        f"\nThe repro has been saved to {repro_path}.")
  if main_emitted.all_externals:
    msg = ("Got undefined symbols: " +
           ", ".join(f"{e} = {global_ctx.var_for_val(e)}"
                     for e in main_emitted.all_externals.values()) +
           f"\nThe repro has been saved to {repro_path}.")
    _thread_local_state.warn_or_error(msg)


  return str(repro_path), repro_source


def save_repro(func: Callable,
               repro_name_prefix: str = "jax_repro",
               ) -> tuple[Any, str, str]:
  """Executes `func` and saves a reproducer. The reproducer is saved
  whether the execution raises an uncaught exception or not.

  You can access the last saved repro using `last_repro`.
  """
  logging.info(f"Found jax_repro_dir: {config.repro_dir.value}")
  if not config.repro_dir.value:
    raise ValueError("You must set JAX_REPRO_DIR=something, when using save_repro")
  _thread_local_state.last_repro = None
  try:
    with flags_override("collect_repro_on_success", True):
      res = func()
  except Exception as e:
    raise
  else:
    assert _thread_local_state.last_repro is None
    repro_stack_entry = _thread_local_state.call_stack[-1]
    assert repro_stack_entry.func.is_main
    emit_repro(repro_stack_entry, "save_repro success",
               repro_name_prefix=repro_name_prefix)
    return res


def last_repro() -> tuple[str, str] | None:
  """The last repro (path and source) that was saved."""
  return _thread_local_state.last_repro


def eval_repro(repro_path: str, repro_source: str) -> Any:
  postamble = """
_repro_result = main_repro()
"""
  compiled = compile(repro_source + postamble, repro_path, "exec")
  custom_namespace = {}
  custom_namespace['__builtins__'] = __builtins__
  exec(compiled, custom_namespace, custom_namespace)
  return custom_namespace["_repro_result"]
