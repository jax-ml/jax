# Copyright 2022 The JAX Authors.
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
"""
Interfaces to JAX's compilation steps, and utilities for conforming to them.

This module defines a set of public-facing types that reflect the output of
intermediate stages in the process of compilation. Currently there are two
stages modeled: lowering (which produces compiler input), and compilation
(which produces compiler output).

It also defines some internal-facing types to guide what JAX can present in
this common form: an internal ``Lowering`` suffices to back a public-facing
``Lowered`` and an internal ``Executable`` suffices to back a public-facing
``Compiled``.

Finally, this module defines a couple more classes to commonly adapt our
various internal XLA-backed lowerings and executables into the lowering and
executable protocols described above.
"""
from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
import enum
import itertools as it
from typing import Any, NamedTuple, Protocol, runtime_checkable

from jax._src import config
from jax._src import core
from jax._src import literals
from jax._src import sharding as sharding_lib
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import tree_util
from jax._src import util
from jax._src.core import typeof
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.layout import AutoLayoutSingleton, Format, Layout
from jax._src.lib import _jax
from jax._src.lib import hlo
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jax._src.sharding_impls import UnspecifiedValue
from jax._src.mesh import AbstractMesh
from jax._src import flattree as ft
from jax._src.tree_util import tree_unflatten
from jax._src.typing import ArrayLike

source_info_util.register_exclusion(__file__)
traceback_util.register_exclusion(__file__)

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

CompilerOptions = dict[str, Any]


# -- Internal types


class Executable:

  def xla_extension_executable(self) -> xc.LoadedExecutable:
    raise NotImplementedError(
        "compiled executable carries no loaded XLA executable. It may be "
        f"that {type(self)} defines an incomplete implementation.")

  def call(self, *args_flat) -> Sequence[Any]:
    """Execute on the flat list of arguments, returning flat outputs."""
    raise NotImplementedError("compiled executable does not support invocation")

  def create_cpp_call(self, params: CompiledCallParams) -> Any:
    """Optionally constructs a fast c++ dispatcher."""
    return None

  def shard_const_args(self, const_args: Sequence[ArrayLike]) -> list[ArrayLike]:
    raise NotImplementedError(
        f"sharding const_args unsupported on {type(self)}")

  def input_shardings(self) -> Sequence[sharding_lib.Sharding]:
    """Flat sequence of input shardings.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend,
    compiler, or runtime.
    """
    raise NotImplementedError(
        "compiled executable carries no input sharding information")

  def output_shardings(self) -> Sequence[sharding_lib.Sharding]:
    """Flat sequence of output shardings.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend,
    compiler, or runtime.
    """
    raise NotImplementedError(
        "compiled executable carries no output sharding information")

  def input_formats(self):
    raise NotImplementedError(
        "compiled executable carries no input layout information")

  def output_formats(self):
    raise NotImplementedError(
        "compiled executable carries no output layout information")

  def as_text(self) -> str:
    """A human-readable text representation of this executable.

    Intended for visualization and debugging purposes. This need not be a valid
    nor reliable serialization. It is relayed directly to external callers.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend,
    compiler, or runtime.
    """
    xla_ext_exe = self.xla_extension_executable()
    err_msg = ("text view unsupported on current XLA backend: "
               f"{type(xla_ext_exe)}")

    if hasattr(xla_ext_exe, "get_hlo_text"):
      try:
        return xla_ext_exe.get_hlo_text()
      except _jax.JaxRuntimeError as e:
        msg, *_ = e.args
        if type(msg) is str and msg.startswith("UNIMPLEMENTED"):
          raise NotImplementedError(err_msg) from e
        else:
          raise
    else:
      if not hasattr(xla_ext_exe, "hlo_modules"):
        raise NotImplementedError(err_msg)
      try:
        return "\n\n".join([m.to_string() for m in xla_ext_exe.hlo_modules()])
      except _jax.JaxRuntimeError as e:
        msg, *_ = e.args
        if type(msg) is str and msg.startswith("UNIMPLEMENTED"):
          raise NotImplementedError(err_msg) from e
        else:
          raise

  def cost_analysis(self) -> Any:
    """A summary of execution cost estimates.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it need not be consistent across versions of JAX
    and jaxlib, or even across invocations. It is relayed directly to external
    callers.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend,
    compiler, or runtime.
    """
    xla_ext_exe = self.xla_extension_executable()

    if hasattr(xla_ext_exe, "cost_analysis"):
      try:
        return xla_ext_exe.cost_analysis()
      except _jax.JaxRuntimeError as e:
        msg, *_ = e.args
        if not (type(msg) is str and msg.startswith("UNIMPLEMENTED")):
          raise

    if (
        xla_ext_exe is None
        and hasattr(self, "unsafe_call")
        and hasattr(self.unsafe_call, "compiled")
        and hasattr(self.unsafe_call.compiled, "cost_analysis")
    ):
      return self.unsafe_call.compiled.cost_analysis()

    raise NotImplementedError(
        f"cost analysis unsupported on current XLA backend: {type(xla_ext_exe)}"
    )

  def memory_analysis(self) -> Any:
    """A summary of estimated memory requirements.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it need not be consistent across versions of JAX
    and jaxlib, or even across invocations. It is relayed directly to external
    callers.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend,
    compiler, or runtime.
    """
    xla_ext_exe = self.xla_extension_executable()
    err_msg = ("memory analysis unsupported on current XLA backend: "
               f"{type(xla_ext_exe)}")
    if not hasattr(xla_ext_exe, "get_compiled_memory_stats"):
      raise NotImplementedError(err_msg)
    try:
      return xla_ext_exe.get_compiled_memory_stats()
    except _jax.JaxRuntimeError as e:
      msg, *_ = e.args
      if type(msg) is str and msg.startswith("UNIMPLEMENTED"):
        raise NotImplementedError(err_msg) from e
      else:
        raise

  def runtime_executable(self) -> Any:
    """An arbitrary object representation of this executable.

    Intended for debugging purposes. This need not be a valid nor reliable
    serialization. It is relayed directly to external callers, with no
    guarantee on type, structure, or consistency across invocations.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend or
    compiler.
    """
    return self.xla_extension_executable()


class Lowering:

  compile_args: dict[str, Any]
  # the constants that have been hoisted out and must be passed as first args,
  # after the tokens.
  # See https://docs.jax.dev/en/latest/internals/constants.html.
  const_args: list[ArrayLike]

  def hlo(self) -> xc.XlaComputation:
    """Return an HLO representation of this computation."""
    hlo = self.stablehlo()
    m: str | bytes
    m = mlir.module_to_bytecode(hlo)
    return _jax.mlir.mlir_module_to_xla_computation(
        m, use_tuple_args=self.compile_args["tuple_args"])

  def stablehlo(self) -> ir.Module:
    """Return a StableHLO representation of this computation."""
    raise NotImplementedError(
        f"cost analysis unsupported on XLA computation: {type(self)}")

  def compile(
      self, compiler_options: CompilerOptions | None = None, *,
      device_assignment: tuple[xc.Device, ...] | None = None) -> Executable:
    """Compile and return a corresponding ``Executable``."""
    raise NotImplementedError(
        f"cost analysis unsupported on XLA computation: {type(self)}")

  def as_text(self, dialect: str | None = None,
              *,
              debug_info: bool = False) -> str:
    """A human-readable text representation of this lowering.

    Intended for visualization and debugging purposes. This need not be a valid
    nor reliable serialization. It is relayed directly to external callers.
    """
    if dialect is None:
      dialect = "stablehlo"
    if dialect == "stablehlo":
      return mlir.module_to_string(self.stablehlo(),
                                   enable_debug_info=debug_info)
    elif dialect == "hlo":
      print_opts = hlo.HloPrintOptions.short_parsable()
      print_opts.print_metadata = debug_info
      return self.hlo().as_hlo_module().to_string(print_opts)
    else:
      raise ValueError(f"unknown dialect: {dialect}")

  def compiler_ir(self, dialect: str | None = None) -> Any:
    """An arbitrary object representation of this lowering.

    Intended for debugging purposes. This need not be a valid nor reliable
    serialization. It is relayed directly to external callers, with no
    guarantee on type, structure, or consistency across invocations.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend or
    compiler.

    Args:
      dialect: Optional string specifying a representation dialect
      (e.g. "stablehlo")
    """
    if dialect is None:
      dialect = "stablehlo"
    if dialect == "stablehlo":
      return self.stablehlo()
    elif dialect == "hlo":
      return self.hlo()
    else:
      raise ValueError(f"unknown dialect: {dialect}")

  def cost_analysis(self) -> Any:
    """A summary of execution cost estimates.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it need not be consistent across versions of JAX
    and jaxlib, or even across invocations. It is relayed directly to external
    callers.

    This function estimates execution cost in the absence of compiler
    optimizations, which may drastically affect the cost. For execution cost
    estimates after optimizations, compile this lowering and see
    ``Compiled.cost_analysis``.

    May raise ``NotImplementedError`` if unavailable, e.g. based on backend,
    compiler, or runtime.
    """
    raise NotImplementedError(
        f"cost analysis unsupported on XLA computation: {type(self)}")


# -- Public-facing API, plus helpers

@dataclass(frozen=True, slots=True)
class ArgInfo:
  _aval: core.AbstractValue
  donated: bool

  @property
  def shape(self):
    if not hasattr(self._aval, "shape"):
      raise TypeError(f"No shape attribute with aval of type {type(self._aval)}")
    return self._aval.shape

  @property
  def dtype(self):
    if not hasattr(self._aval, "dtype"):
      raise TypeError(f"No dtype attribute with aval of type {type(self._aval)}")
    return self._aval.dtype


class Stage:
  args_info: Any  # PyTree of ArgInfo

  @property
  def in_tree(self) -> tree_util.PyTreeDef:
    """Tree structure of the pair (positional arguments, keyword arguments)."""
    return tree_util.tracing_registry.flatten(self.args_info)[1]

  @property
  def in_avals(self):
    """Tree of input avals."""
    return tree_util.tree_map(lambda x: x._aval, self.args_info)

  @property
  def donate_argnums(self):
    """Flat tuple of donated argument indices."""
    return tuple(
        i for i, x in enumerate(tree_util.tree_leaves(self.args_info))
        if x.donated)

  def specialize(self, *args, **kwargs) -> Stage:
    """Specializes a shape-polymorphic stage to concrete argument shapes."""
    raise NotImplementedError(f"Specialization not supported for {type(self)}")


def make_args_info(in_tree, in_avals, donate_argnums):
  donate_argnums = frozenset(donate_argnums)
  flat_avals, _ = tree_util.tree_flatten(in_avals)  # todo: remove
  return in_tree.unflatten([
      ArgInfo(aval, i in donate_argnums)
      for i, aval in enumerate(flat_avals)])


class CompiledCallParams(NamedTuple):
  executable: Executable
  no_kwargs: bool
  in_tree: tree_util.PyTreeDef  # lo tree
  out_tree: tree_util.PyTreeDef  # lo tree
  const_args: list[ArrayLike]  # https://docs.jax.dev/en/latest/internals/constants.html
  in_types: tuple[tree_util.PyTreeDef, list[core.AbstractValue]] | None
  out_types: tuple[tree_util.PyTreeDef, list[core.AbstractValue]] | None

  @property
  def is_high(self):
    return self.in_types and self.out_types and any(
        a.is_high for a in it.chain(self.in_types[1], self.out_types[1]))


def _traced_args_info(self):
  don8_rgn = tuple(i for i, d in enumerate(self._params['donated_invars']) if d)
  arg_avals = self.jaxpr.in_avals[self._num_consts:]
  return make_args_info(self._in_tree, arg_avals, don8_rgn)

def _traced_out_info(self):
  out_shardings = [None if isinstance(s, UnspecifiedValue) else s
                   for s in self._params['out_shardings']]
  out_layouts = [None if isinstance(l, AutoLayoutSingleton) else l
                 for l in self._params['out_layouts']]
  out = []
  for a, out_s, out_l in zip(self.jaxpr.out_avals, out_shardings, out_layouts):
    if isinstance(a, core.ShapedArray):
      s = ((a.sharding if a.sharding.mesh.are_all_axes_explicit_or_manual
            else out_s) if out_s is None else out_s)
      out.append(
          core.ShapeDtypeStruct(
              a.shape, a.dtype, sharding=Format(out_l, s),
              weak_type=a.weak_type,
              manual_axis_type=(a.mat if config._check_vma.value else None),
              _memory_space=a.memory_space))
    else:
      out.append(a)
  return tree_util.tree_unflatten(self.out_tree, out)

# Cached on the jaxpr, which jit's tracing cache shares among Traceds of the same
# function and argument types, so that those Traceds get the same `fun`, and
# thus tracing cache hits when `fun` is traced (e.g. in `with_consts_as_arg`).
@util.weakref_lru_cache
def _flat_closure_converted_fun(jaxpr):
  jaxpr = jaxpr.replace(consts=None)
  return lambda consts, *args: core.eval_jaxpr_p.bind(*consts, *args, call_jaxpr=jaxpr)

@util.weakref_lru_cache
def _closure_converted_fun(jaxpr, consts_tree, in_tree, out_tree):
  jaxpr = jaxpr.replace(consts=None)
  in_tree = tree_util.treedef_tuple_tracing_registry(
      (consts_tree, *in_tree.children()))
  def fun(consts, *args, **kwargs):
    args_flat = tree_util.tree_leaves_checked(in_tree, (consts, args, kwargs))
    out_flat = core.eval_jaxpr_p.bind(*args_flat, call_jaxpr=jaxpr)
    return tree_unflatten(out_tree, out_flat)
  return fun


def _eval_dim(d: Any, dim_values: dict[str, int]) -> int | Any:
  if isinstance(d, int):
    return d
  if hasattr(d, "_evaluate"):
    return int(d._evaluate(dim_values))
  try:
    return int(d)
  except (TypeError, ValueError):
    return d


def _eval_shape(
    shape: tuple[Any, ...], dim_values: dict[str, int]
) -> tuple[int, ...]:
  return tuple(_eval_dim(d, dim_values) for d in shape)


def _specialize_aval(
    aval: core.AbstractValue, dim_values: dict[str, int]
) -> core.AbstractValue:
  if isinstance(aval, core.ShapedArray):
    new_shape = _eval_shape(aval.shape, dim_values)
    return aval.update(shape=new_shape)
  from jax._src.state import types as state_types  # pyrefly: ignore[missing-import]
  if isinstance(aval, state_types.AbstractRef):
    return aval.update(
        inner_aval=_specialize_aval(aval.inner_aval, dim_values)
    )
  if isinstance(aval, state_types.AbstractLinVal):
    return dataclasses.replace(
        aval, inner_aval=_specialize_aval(aval.inner_aval, dim_values)
    )
  if isinstance(aval, state_types.TransformedRef) and isinstance(
      aval.ref, core.AbstractValue
  ):
    return dataclasses.replace(
        aval, ref=_specialize_aval(aval.ref, dim_values)
    )
  return aval


def _specialize_jaxpr(
    jaxpr: core.Jaxpr, dim_values: dict[str, int]
) -> core.Jaxpr:
  import numpy as np

  var_map: dict[core.Var, core.Atom] = {}

  def _subst_effects(effs):
    if not effs:
      return effs
    out_effs = set()
    for e in effs:
      if isinstance(e, core.effects.JaxprInputEffect):
        mapped = var_map.get(e.input, e.input)
        if isinstance(mapped, core.Literal):
          continue
        e = e.replace(mapped)
      out_effs.add(e)
    return out_effs

  new_constvars = []
  for v in jaxpr.constvars:
    new_v = core.Var(_specialize_aval(v.aval, dim_values))
    var_map[v] = new_v
    new_constvars.append(new_v)

  new_invars = []
  for v in jaxpr.invars:
    new_v = core.Var(_specialize_aval(v.aval, dim_values))
    var_map[v] = new_v
    new_invars.append(new_v)

  new_eqns = []
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == "dim_as_value":
      dim = eqn.params.get("dim")
      val = _eval_dim(dim, dim_values)
      out_aval = _specialize_aval(eqn.outvars[0].aval, dim_values)
      out_dtype = (
          out_aval.dtype
          if isinstance(out_aval, core.ShapedArray)
          else core.dim_value_dtype()
      )
      val_arr = np.asarray(val, dtype=out_dtype)
      lit = core.Literal(val_arr, out_aval)
      var_map[eqn.outvars[0]] = lit
      continue

    new_in_atoms = [
        var_map.get(v, v) if isinstance(v, core.Var) else v for v in eqn.invars
    ]
    if (
        eqn.primitive.name == "convert_element_type"
        and not eqn.effects
        and len(new_in_atoms) == 1
        and isinstance(new_in_atoms[0], core.Literal)
    ):
      out_aval = _specialize_aval(eqn.outvars[0].aval, dim_values)
      if isinstance(out_aval, core.ShapedArray):
        val_arr = np.asarray(new_in_atoms[0].val, dtype=out_aval.dtype)
        var_map[eqn.outvars[0]] = core.Literal(val_arr, out_aval)
        continue

    new_outvars = []
    for v in eqn.outvars:
      new_v = core.Var(_specialize_aval(v.aval, dim_values))
      var_map[v] = new_v
      new_outvars.append(new_v)

    new_params = dict(eqn.params)
    if (
        eqn.primitive.name in ("get", "swap", "addupdate")
        and "tree" in new_params
    ):
      from jax._src import indexing as jax_indexing  # pyrefly: ignore[missing-import]
      from jax._src.state import indexing as state_indexing  # pyrefly: ignore[missing-import]
      from jax._src.state import types as state_types  # pyrefly: ignore[missing-import]

      idx_offset = 1 if eqn.primitive.name == "get" else 2
      flat_idx = new_in_atoms[idx_offset:]
      transforms = tree_util.tree_unflatten(new_params["tree"], flat_idx)

      def _spec_transform(t):
        if isinstance(t, state_indexing.NDIndexer):
          new_indices = []
          for idx in t.indices:
            if isinstance(idx, jax_indexing.Slice):
              new_indices.append(
                  jax_indexing.Slice(
                      _eval_dim(idx.start, dim_values),
                      _eval_dim(idx.size, dim_values),
                      _eval_dim(idx.stride, dim_values),
                  )
              )
            else:
              new_indices.append(_eval_dim(idx, dim_values))
          return state_indexing.NDIndexer(
              tuple(new_indices),
              _eval_shape(t.shape, dim_values),
              _eval_shape(t.int_indexer_shape, dim_values),
          )
        if isinstance(t, state_types.ReshapeTransform):
          return dataclasses.replace(
              t, shape=_eval_shape(t.shape, dim_values)
          )
        return t

      new_transforms = tuple(_spec_transform(t) for t in transforms)
      new_flat_idx, new_tree = tree_util.tree_flatten(new_transforms)
      new_in_atoms = [*new_in_atoms[:idx_offset], *new_flat_idx]
      new_params["tree"] = new_tree

    for jaxpr_key in ("jaxpr", "call_jaxpr", "cond_jaxpr", "body_jaxpr"):
      if jaxpr_key in new_params:
        sub_j = new_params[jaxpr_key]
        if isinstance(sub_j, core.Jaxpr):
          new_params[jaxpr_key] = _specialize_jaxpr(sub_j, dim_values)
        elif hasattr(sub_j, "jaxpr") and isinstance(sub_j.jaxpr, core.Jaxpr):
          new_sub_j = _specialize_jaxpr(sub_j.jaxpr, dim_values)
          new_params[jaxpr_key] = (
              core.ClosedJaxpr(new_sub_j, sub_j.consts)
              if hasattr(core, "ClosedJaxpr")
              else new_sub_j
          )

    if "branches" in new_params and isinstance(
        new_params["branches"], (list, tuple)
    ):
      new_branches = []
      for b in new_params["branches"]:
        if isinstance(b, core.Jaxpr):
          new_branches.append(_specialize_jaxpr(b, dim_values))
        elif hasattr(b, "jaxpr") and isinstance(b.jaxpr, core.Jaxpr):
          new_sub_j = _specialize_jaxpr(b.jaxpr, dim_values)
          new_branches.append(
              core.ClosedJaxpr(new_sub_j, b.consts)
              if hasattr(core, "ClosedJaxpr")
              else new_sub_j
          )
        else:
          new_branches.append(b)
      new_params["branches"] = tuple(new_branches)

    for k in (
        "shape",
        "new_sizes",
        "dimensions",
        "start_indices",
        "limit_indices",
        "strides",
        "padding",
        "window_strides",
        "lhs_dilation",
        "rhs_dilation",
    ):
      if k in new_params and isinstance(new_params[k], (list, tuple)):
        new_params[k] = tuple(_eval_dim(d, dim_values) for d in new_params[k])

    if eqn.primitive.name == "pallas_call":
      if "grid_mapping" in new_params:
        gm = new_params["grid_mapping"]
        gm_kwargs = {}
        if hasattr(gm, "grid") and gm.grid is not None:
          gm_kwargs["grid"] = tuple(_eval_dim(d, dim_values) for d in gm.grid)
        if hasattr(gm, "index_map_avals") and gm.index_map_avals is not None:
          gm_kwargs["index_map_avals"] = tuple(
              _specialize_aval(a, dim_values) for a in gm.index_map_avals
          )
        if hasattr(gm, "scratch_avals") and gm.scratch_avals is not None:
          gm_kwargs["scratch_avals"] = tuple(
              _specialize_aval(a, dim_values) for a in gm.scratch_avals
          )
        if hasattr(gm, "block_mappings") and gm.block_mappings is not None:
          new_bms = []
          for bm in gm.block_mappings:
            new_bs = None
            if hasattr(bm, "block_shape") and bm.block_shape is not None:
              def _eval_block_dim(bd):
                if hasattr(bd, "block_size"):
                  return dataclasses.replace(
                      bd, block_size=_eval_dim(bd.block_size, dim_values)
                  )
                return _eval_dim(bd, dim_values)
              new_bs = tuple(_eval_block_dim(d) for d in bm.block_shape)

            new_tba = None
            if (
                hasattr(bm, "transformed_block_aval")
                and bm.transformed_block_aval is not None
            ):
              new_tba = _specialize_aval(bm.transformed_block_aval, dim_values)

            new_aa = None
            if hasattr(bm, "array_aval") and bm.array_aval is not None:
              new_aa = _specialize_aval(bm.array_aval, dim_values)

            new_im_jaxpr = None
            if hasattr(bm, "index_map_jaxpr") and bm.index_map_jaxpr is not None:
              new_im_jaxpr = _specialize_jaxpr(bm.index_map_jaxpr, dim_values)

            kwargs = {}
            if new_bs is not None:
              kwargs["block_shape"] = new_bs
            if new_tba is not None:
              kwargs["transformed_block_aval"] = new_tba
            if new_aa is not None:
              kwargs["array_aval"] = new_aa
            if new_im_jaxpr is not None:
              kwargs["index_map_jaxpr"] = new_im_jaxpr

            if kwargs and hasattr(bm, "replace"):
              bm = bm.replace(**kwargs)
            new_bms.append(bm)
          gm_kwargs["block_mappings"] = tuple(new_bms)
        if gm_kwargs and hasattr(gm, "replace"):
          gm = gm.replace(**gm_kwargs)
        new_params["grid_mapping"] = gm

      if "out_avals" in new_params and new_params["out_avals"] is not None:
        new_params["out_avals"] = tuple(
            _specialize_aval(a, dim_values) for a in new_params["out_avals"]
        )

      if (
          "cost_estimate" in new_params
          and new_params["cost_estimate"] is not None
      ):
        ce = new_params["cost_estimate"]
        if hasattr(ce, "flops"):
          def _eval_opt(v):
            return int(_eval_dim(v, dim_values)) if v is not None else None

          new_params["cost_estimate"] = dataclasses.replace(
              ce,
              flops=_eval_opt(ce.flops),
              transcendentals=_eval_opt(ce.transcendentals),
              bytes_accessed=_eval_opt(ce.bytes_accessed),
              remote_bytes_transferred=_eval_opt(ce.remote_bytes_transferred),
          )
    new_eqn = eqn.replace(
        invars=new_in_atoms,
        outvars=new_outvars,
        params=new_params,
        effects=_subst_effects(eqn.effects),
    )
    new_eqns.append(new_eqn)

  new_out_atoms = [
      var_map.get(v, v) if isinstance(v, core.Var) else v for v in jaxpr.outvars
  ]

  specialized_jaxpr = core.Jaxpr(
      new_constvars,
      new_invars,
      new_out_atoms,
      new_eqns,
      effects=_subst_effects(jaxpr.effects),
      debug_info=jaxpr.debug_info,
      is_high=jaxpr.is_high,
      consts=jaxpr.consts,
  )
  if config.enable_checks.value:
    core.check_jaxpr(specialized_jaxpr)
  return specialized_jaxpr


class Traced(Stage):
  """Traced form of a function specialized to argument types and values.

  A traced computation is ready for lowering. This class carries the
  traced representation with the remaining information needed to later
  lower, compile, and execute it.

  Provides access to both the hijax (high-level) and lojax (low-level)
  representations via `.jaxpr` and `.lojax` properties respectively.
  """
  __slots__ = ['_meta_tys_flat', '_params', '_in_tree', 'out_tree', '_consts',
               '_fun_sourceinfo', '_lojax', '_closure_converted']

  def __init__(self, meta_tys_flat, params, in_tree, out_tree, consts,
               fun_sourceinfo):
    self._meta_tys_flat = meta_tys_flat
    self._params = params
    self._in_tree = in_tree
    self.out_tree = out_tree
    self._consts = consts
    self._fun_sourceinfo = fun_sourceinfo
    self._lojax = None
    self._closure_converted = None

  jaxpr = property(lambda self: self._params['jaxpr'])
  fun_name = property(lambda self: self._params['name'])
  args_info = property(_traced_args_info)  # pyrefly: ignore[bad-override]
  out_info = property(_traced_out_info)
  _num_consts = property(lambda self: len(self._consts))

  @property
  def out_avals(self):
    return tree_unflatten(self.out_tree, self.jaxpr.out_avals)

  @property
  def effects(self) -> frozenset[core.Effect]:
    return frozenset(core.positional_effects(self.jaxpr))

  def __call__(self, *args, **kwargs):
    args_flat = tree_util.tree_leaves_checked(self.in_tree, (args, kwargs))
    out_flat = core.eval_jaxpr_p.bind(*args_flat, call_jaxpr=self.jaxpr)
    out_flat = [type(x).__base__(x) if type(x) in literals.typed_scalar_types  # pyrefly: ignore[not-callable]
                else x for x in out_flat]
    return tree_unflatten(self.out_tree, out_flat)

  def closure_convert(self):
    """Closure conversion: makes this Traced's captured constants explicit.

    Returns a pair ``(consts, fun)``, where ``consts`` are the values this
    Traced captured from its function's closure during tracing (not Python
    ``__closure__`` cells: any values encountered during tracing that
    determine the output), and ``fun`` is a closed function such that
    ``fun(consts, *args, **kwargs)`` computes the same results as this Traced
    applied to ``args`` and ``kwargs``. The environment ``consts`` is passed
    as a single leading argument, and may be replaced by any pytree of values
    of the same types.
    """
    if self._closure_converted is None:
      consts = [*self.jaxpr.consts, *self._consts]
      _, consts_tree = tree_util.tracing_registry.flatten(consts)
      fun = _closure_converted_fun(self.jaxpr, consts_tree, self.in_tree,
                                   self.out_tree)
      self._closure_converted = (consts, fun)
    return self._closure_converted

  def with_consts_as_arg(self) -> tuple[list[Any], Traced]:
    """Returns consts and an equivalent flat Traced taking them as one leading argument.

    The returned Traced takes ``(consts, *args_flat)`` and returns a flat list,
    where ``args_flat`` are this Traced's flattened arguments, so that argument
    and result pytrees are never rebuilt around types. Its trace stages a single
    call eqn, not a re-trace of the original function.
    """
    from jax._src.api import jit  # type: ignore
    consts = [*self.jaxpr.consts, *self._consts]
    arg_avals = self.jaxpr.in_avals[len(self._consts):]
    traced = jit(_flat_closure_converted_fun(self.jaxpr)).trace(consts, *arg_avals)
    jaxpr = traced.jaxpr
    if (not jaxpr.consts and len(jaxpr.eqns) == 1 and
        (eqn := jaxpr.eqns[0]).primitive is core.eval_jaxpr_p and
        list(eqn.invars) == list(jaxpr.invars) and
        list(eqn.outvars) == list(jaxpr.outvars)):
      traced._params = dict(traced._params, jaxpr=eqn.params['call_jaxpr'])
    traced._params = dict(traced._params, name=self.fun_name)
    traced._fun_sourceinfo = self._fun_sourceinfo
    return consts, traced

  def _batch(self, axis_data, in_axes) -> tuple[Traced, tuple[Any, ...]]:
    """Returns this Traced batched along `axis_data`, with its flattened
    arguments batched along `in_axes` (one int or None each), and the batch
    axes of its flattened results."""
    in_axes = (None,) * len(self._consts) + tuple(in_axes)
    new_jaxpr, out_axes = batching.batch_jaxpr2(self.jaxpr, axis_data, in_axes)
    new_params = dict(self._params, jaxpr=new_jaxpr)
    return Traced(list(new_jaxpr.in_avals), new_params, self._in_tree,
                  self.out_tree, self._consts, self._fun_sourceinfo), out_axes

  def physicalize(self, ctx) -> Traced:
    new_jaxpr = ctx.physicalize_closed_jaxpr(self.jaxpr)
    new_params = dict(self._params, jaxpr=new_jaxpr)
    return Traced(
        list(new_jaxpr.in_avals),
        new_params,
        self._in_tree,
        self.out_tree,
        self._consts,
        self._fun_sourceinfo,
    )

  @property
  def lojax(self) -> LoJax:
    if self._lojax is not None:
      return self._lojax

    if not self.jaxpr.is_high:
      self._lojax = LoJax(
          self._meta_tys_flat, self._params, self._in_tree, self.out_tree,
          (self._in_tree, self.jaxpr.in_avals),
          (self.out_tree, self.jaxpr.out_avals),
          self._consts, self._fun_sourceinfo)
      return self._lojax

    # TODO(mattjj): when pmap is deleted, merge with pjit.py BUILD rule
    from jax._src.pjit import _lojax_expand_params  # pyrefly: ignore[missing-import]
    hi_jaxpr = self.jaxpr
    in_avals = ft.flatten(([a.lo_ty() for a in hi_jaxpr.in_avals], {}))
    lo_jaxpr, out_avals = pe.lower_jaxpr(hi_jaxpr, in_avals)
    params = dict(_lojax_expand_params(in_avals, out_avals, **self._params), jaxpr=lo_jaxpr)
    if any(a.is_high for a in hi_jaxpr.in_avals):
      in_tree = lojax_pytree(hi_jaxpr.in_avals, self._in_tree)
    else:
      in_tree = self._in_tree
    if any(a.is_high for a in hi_jaxpr.out_avals):
      out_tree = lojax_pytree(hi_jaxpr.out_avals, self.out_tree)
    else:
      out_tree = self.out_tree
    lo_meta_tys = [mty.replace(aval=lo_ty)
                   for mty in self._meta_tys_flat
                   for lo_ty in mty.aval.lo_ty()]
    self._lojax = LoJax(
        lo_meta_tys, params, in_tree, out_tree,
        (self._in_tree, hi_jaxpr.in_avals),
        (self.out_tree, hi_jaxpr.out_avals),
        self._consts, self._fun_sourceinfo)
    return self._lojax

  def lower(self, *, lowering_platforms: tuple[str, ...] | None = None,
            _private_parameters: mlir.LoweringParameters | None = None):
    """Lower to compiler input, returning a ``Lowered`` instance."""
    from jax._src.pjit import _resolve_and_lower  # pyrefly: ignore[missing-import]
    lo = self.lojax
    if _private_parameters is None:
      _private_parameters = mlir.LoweringParameters()
    try:
      import contextlib
      from jax._src.export import shape_poly  # pyrefly: ignore[missing-import]
      has_sym = any(
          hasattr(m.aval, 'shape')
          and any(shape_poly.is_symbolic_dim(d) for d in m.aval.shape)
          for m in lo._meta_tys_flat
      )
      if has_sym:
        try:
          from jax._src.pallas import core as pallas_core  # pyrefly: ignore[missing-import]
          pallas_ctx = pallas_core.pallas_export_experimental(dynamic_shapes=True)
        except Exception:
          pallas_ctx = contextlib.nullcontext()
      else:
        pallas_ctx = contextlib.nullcontext()

      ctx_mesh = lo._params['ctx_mesh']
      if (lowering_platforms is None and isinstance(ctx_mesh, AbstractMesh)
          and (abd := ctx_mesh.abstract_device) is not None):
        lowering_platforms = (abd.platform,)
      with pallas_ctx:
        lowering = _resolve_and_lower(
            lo._meta_tys_flat, **lo._params, lowering_platforms=lowering_platforms,
            lowering_parameters=_private_parameters, pgle_profiler=None)
    except DeviceAssignmentMismatchError as e:
      fails, = e.args
      msg = _device_assignment_mismatch_error(
          lo._params['name'], fails, lo._meta_tys_flat, 'jit',
          lo.jaxpr.debug_info.safe_arg_names(len(lo.jaxpr.in_avals)))
      raise ValueError(msg) from None
    return Lowered(lowering, lo.args_info, lo.out_tree,
                   in_types=lo._in_types, out_types=lo._out_types)

  def specialize(self, *args, **kwargs) -> Traced:
    """Specializes a shape-polymorphic Traced stage to concrete argument shapes.

    Substitutes concrete dimension values into the underlying ClosedJaxpr,
    evaluating all symbolic dimensions to concrete integers, replacing
    `dim_as_value` equations with integer literals, and specializing
    embedded Pallas/Mosaic custom calls and sub-jaxprs.

    Returns a new `Traced` stage with 100% static shapes, ready for lowering
    and compilation without dynamic-shape overhead.
    """
    from jax._src.export import shape_poly  # pyrefly: ignore[missing-import]
    from jax._src.pjit import convert_to_metaty, create_meta_ty  # pyrefly: ignore[missing-import]

    args_flat, in_tree = tree_util.tree_flatten((args, kwargs))
    if in_tree != self.in_tree:
      raise ValueError(
          f"Tree structure mismatch in specialize: expected {self.in_tree}, got"
          f" {in_tree}"
      )

    concrete_avals = []
    for a in args_flat:
      if isinstance(a, core.AbstractValue):
        concrete_avals.append(a)
      elif hasattr(a, "shape") and hasattr(a, "dtype"):
        sharding = getattr(a, "sharding", None)
        try:
          c_aval = core.ShapedArray(a.shape, a.dtype, sharding=sharding)
        except Exception:
          c_aval = core.ShapedArray(a.shape, a.dtype)
        concrete_avals.append(c_aval)
      else:
        concrete_avals.append(typeof(a))

    user_in_avals = self.jaxpr.in_avals[self._num_consts :]
    has_sym = any(
        hasattr(a, "shape")
        and any(shape_poly.is_symbolic_dim(d) for d in a.shape)
        for a in user_in_avals
    )
    if not has_sym:
      return self

    solution, constraints, synth_vars = shape_poly.solve_dim_vars(
        tuple(user_in_avals), self.in_tree
    )
    env = {}
    for synth_name, arg_idx, dim_idx in synth_vars:
      actual_dim = concrete_avals[arg_idx].shape[dim_idx]
      env[synth_name] = actual_dim
    evaluator = shape_poly.ShapeEvaluator(env)
    constraints.check_statically(evaluator)
    dim_values = {v: int(evaluator.evaluate(solution[v])) for v in solution}

    new_jaxpr = _specialize_jaxpr(self.jaxpr, dim_values)
    new_params = dict(self._params, jaxpr=new_jaxpr)

    new_meta_tys = list(self._meta_tys_flat[: self._num_consts])
    for orig_mty, a, c_aval in zip(
        self._meta_tys_flat[self._num_consts :], args_flat, concrete_avals
    ):
      in_sharding = getattr(a, "sharding", None)
      if in_sharding is None:
        in_sharding = orig_mty.sharding
      new_meta_tys.append(
          orig_mty.replace(aval=c_aval, sharding=in_sharding)
      )

    return Traced(
        new_meta_tys,
        new_params,
        self._in_tree,
        self.out_tree,
        self._consts,
        self._fun_sourceinfo,
    )


def lojax_pytree(hi_avals, tree):
  lo_avals = [t.lo_ty() for t in hi_avals]
  return tree_util.tracing_registry.flatten(tree_unflatten(tree, lo_avals))[1]


class LoJax:
  __slots__ = ['_meta_tys_flat', '_params', '_in_tree', 'out_tree',
               '_consts', '_fun_sourceinfo', '_in_types', '_out_types']

  def __init__(self, meta_tys_flat, params, in_tree, out_tree, in_types, out_types,
               consts, fun_sourceinfo):
    self._meta_tys_flat = meta_tys_flat
    self._params = params
    self._in_tree = in_tree
    self.out_tree = out_tree
    self._consts = consts
    self._fun_sourceinfo = fun_sourceinfo
    self._in_types = in_types  # hi types
    self._out_types = out_types

  jaxpr = property(lambda self: self._params['jaxpr'])
  fun_name = property(lambda self: self._params['name'])
  args_info = property(_traced_args_info)
  out_info = property(_traced_out_info)
  _num_consts = property(lambda self: len(self._consts))


class Lowered(Stage):
  """Lowering of a function specialized to argument types and values.

  A lowering is a computation ready for compilation. This class
  carries a lowering together with the remaining information needed to
  later compile and execute it. It also provides a common API for
  querying properties of lowered computations across JAX's various
  lowering paths (:func:`~jax.jit`, :func:`~jax.pmap`, etc.).
  """
  __slots__ = ["_lowering", "args_info", "out_tree", "_no_kwargs",
               "_in_types", "_out_types"]

  _lowering: Lowering
  args_info: Any  # PyTree of ArgInfo, not including the const_args
  out_tree: tree_util.PyTreeDef
  _no_kwargs: bool
  _in_types: tuple[tree_util.PyTreeDef, list[core.AbstractValue]] | None
  _out_types: list[core.AbstractValue] | None

  def __init__(self, lowering: Lowering, args_info,
               out_tree: tree_util.PyTreeDef, no_kwargs: bool = False,
               in_types=None, out_types=None):

    self._lowering = lowering
    self.args_info = args_info
    self.out_tree = out_tree
    self._no_kwargs = no_kwargs
    self._in_types = in_types
    self._out_types = out_types

  @property
  def in_avals(self):
    in_avals_ = self._lowering.compile_args["global_in_avals"]
    kept_var_idx = self._lowering.compile_args["kept_var_idx"]
    non_dce_avals = self._lowering.compile_args["all_args_info"].in_avals
    if self.in_tree.num_leaves > len(in_avals_):
      iter_in_avals = iter(in_avals_)
      in_avals_ = [
          next(iter_in_avals) if i in kept_var_idx
          else a for i, a in zip(range(self.in_tree.num_leaves), non_dce_avals)]
    return self.in_tree.unflatten(in_avals_)

  @property
  def out_info(self):  # PyTree of OutInfo
    out_avals = self._lowering.compile_args["global_out_avals"]
    out_shardings = self._lowering.compile_args["out_shardings"]
    out_layouts = self._lowering.compile_args["out_layouts"]
    outs = []
    for o, l, s in zip(out_avals, out_layouts, out_shardings):
      s = None if isinstance(s, UnspecifiedValue) else s
      l = None if isinstance(l, AutoLayoutSingleton) else l
      format = Format(l, s)
      outs.append(core.ShapeDtypeStruct(o.shape, o.dtype, sharding=format))
    return self.out_tree.unflatten(outs)

  def compile(
      self, compiler_options: CompilerOptions | None = None, *,
      device_assignment: tuple[xc.Device, ...] | None = None) -> Compiled:
    """Compile, returning a corresponding ``Compiled`` instance."""
    kw: dict[str, Any] = {"compiler_options": compiler_options,
                          "device_assignment": device_assignment}
    return Compiled(self._lowering.compile(**kw), self._lowering.const_args,
                    self.args_info, self.out_tree, self._no_kwargs,
                    self._in_types, self._out_types)

  def specialize(self, *args, **kwargs) -> Lowered:
    """Specializes a shape-polymorphic Lowered stage to concrete argument shapes.

    Refines polymorphic StableHLO types in the module to static shapes using
    concrete dimension values.
    """
    args_flat, in_tree = tree_util.tree_flatten((args, kwargs))
    if in_tree != self.in_tree:
      raise ValueError(
          f"Tree structure mismatch in specialize: expected {self.in_tree}, got"
          f" {in_tree}"
      )

    from jax._src.interpreters import pxla  # pyrefly: ignore[missing-module-attribute,missing-import]

    if isinstance(self._lowering, pxla.MeshComputation):
      new_lowering = pxla.specialize_mesh_computation(
          self._lowering, args_flat, in_tree
      )
      return Lowered(
          new_lowering,
          self.args_info,
          self.out_tree,
          self._no_kwargs,
          in_types=self._in_types,
          out_types=self._out_types,
      )
    raise NotImplementedError(
        "Lowered.specialize not supported for lowering type"
        f" {type(self._lowering)}"
    )

  def as_text(self, dialect: str | None = None, *,
              debug_info: bool = False) -> str:
    """A human-readable text representation of this lowering.

    Intended for visualization and debugging purposes. This need not be a valid
    nor reliable serialization.
    Use `jax.export` if you want reliable and portable serialization.

    Args:
      dialect: Optional string specifying a lowering dialect (e.g. "stablehlo",
        or "hlo").
      debug_info: Whether to include debugging information,
        e.g., source location.
    """
    return self._lowering.as_text(dialect, debug_info=debug_info)

  def compiler_ir(self, dialect: str | None = None) -> Any | None:
    """An arbitrary object representation of this lowering.

    Intended for debugging purposes. This is not a valid nor reliable
    serialization. The output has no guarantee of consistency across
    invocations.
    Use `jax.export` if you want reliable and portable serialization.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.

    Args:
      dialect: Optional string specifying a lowering dialect (e.g. "stablehlo",
        or "hlo").
    """
    try:
      return self._lowering.compiler_ir(dialect)
    except NotImplementedError:
      return None

  def cost_analysis(self) -> Any | None:
    """A summary of execution cost estimates.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it may be inconsistent across versions of JAX
    and jaxlib, or even across invocations.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.
    """
    # TODO(frostig): improve annotation (basic pytree of arbitrary structure)
    try:
      return self._lowering.cost_analysis()
    except NotImplementedError:
      return None


class Compiled(Stage):
  """Compiled representation of a function specialized to types/values.

  A compiled computation is associated with an executable and the
  remaining information needed to execute it. It also provides a
  common API for querying properties of compiled computations across
  JAX's various compilation paths and backends.
  """
  __slots__ = ["args_info", "out_tree", "_executable", "_no_kwargs", "_params"]

  # PyTree of ArgInfo, including dead args, but not const_args
  args_info: Any
  out_tree: tree_util.PyTreeDef
  _executable: Executable
  _no_kwargs: bool
  _params: CompiledCallParams

  def __init__(self, executable, const_args: list[ArrayLike],
               args_info, out_tree, no_kwargs=False, in_types=None, out_types=None):
    self._executable = executable
    self._no_kwargs = no_kwargs
    self.args_info = args_info
    self.out_tree = out_tree
    self._params = CompiledCallParams(
        self._executable, self._no_kwargs, self.in_tree, self.out_tree,
        const_args, in_types, out_types)
    self._call = None

  def as_text(self) -> str | None:
    """A human-readable text representation of this executable.

    Intended for visualization and debugging purposes. This is not a valid nor
    reliable serialization.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.
    """
    try:
      return self._executable.as_text()
    except NotImplementedError:
      return None

  def cost_analysis(self) -> Any | None:
    """A summary of execution cost estimates.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it may be inconsistent across versions of JAX
    and jaxlib, or even across invocations.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.
    """
    # TODO(frostig): improve annotation (basic pytree of arbitrary structure)
    try:
      return self._executable.cost_analysis()
    except NotImplementedError:
      return None

  def memory_analysis(self) -> Any | None:
    """A summary of estimated memory requirements.

    Intended for visualization and debugging purposes. The object output by
    this is some simple data structure that can easily be printed or serialized
    (e.g. nested dicts, lists, and tuples with numeric leaves). However, its
    structure can be arbitrary: it may be inconsistent across versions of JAX
    and jaxlib, or even across invocations.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.
    """
    # TODO(frostig): improve annotation (basic pytree of arbitrary structure)
    try:
      return self._executable.memory_analysis()
    except NotImplementedError:
      return None

  @property
  def in_avals(self):
    # Including dead args, but not const_args
    nr_const_args = len(self._params.const_args)
    in_avals_ = self._executable.in_avals[nr_const_args:]  # pyrefly: ignore[missing-attribute]
    if self.in_tree.num_leaves > len(in_avals_):
      iter_in_avals = iter(in_avals_)
      non_dce_avals = self._executable._all_args_info.in_avals[nr_const_args:]  # pyrefly: ignore[missing-attribute]
      in_avals_ = [
          next(iter_in_avals) if i + nr_const_args in self._executable._kept_var_idx # pyrefly: ignore[missing-attribute]
          else a for i, a in zip(range(self.in_tree.num_leaves), non_dce_avals)]
    return self.in_tree.unflatten(in_avals_)

  @property
  def out_info(self):  # PyTree of jax.ShapeDtypeStruct
    out_avals = self._executable.out_avals  # pyrefly: ignore[missing-attribute]
    out_formats_flat = self._output_formats_flat
    return self.out_tree.unflatten(
        [core.ShapeDtypeStruct(o.shape, o.dtype, sharding=f)
         for o, f in zip(out_avals, out_formats_flat)])

  def runtime_executable(self) -> Any | None:
    """An arbitrary object representation of this executable.

    Intended for debugging purposes. This is not valid nor reliable
    serialization. The output has no guarantee of consistency across
    invocations.

    Returns ``None`` if unavailable, e.g. based on backend, compiler, or
    runtime.
    """
    return self._executable.runtime_executable()

  def _input_shardings_flat(self):
    nr_const_args = len(self._params.const_args)
    shardings_flat = self._executable._in_shardings[nr_const_args:]  # pyrefly: ignore[missing-attribute]
    # Some input shardings got DCE'd
    if self.in_tree.num_leaves > len(shardings_flat):
      iter_shardings_flat = iter(shardings_flat)
      shardings_flat = [next(iter_shardings_flat) if i + nr_const_args in self._executable._kept_var_idx  # pyrefly: ignore[missing-attribute]
                        else None for i in range(self.in_tree.num_leaves)]
    return shardings_flat

  @property
  def input_shardings(self):  # -> PyTree[sharding.Sharding]
    # Including dead args, but not const_args
    shardings_flat = self._input_shardings_flat()
    return tree_util.tree_unflatten(self.in_tree, shardings_flat)

  @property
  def output_shardings(self):  # -> PyTree[sharding.Sharding]
    shardings_flat = self._executable._out_shardings  # pyrefly: ignore[missing-attribute]
    return tree_util.tree_unflatten(self.out_tree, shardings_flat)

  def _input_layouts_flat(self):
    nr_const_args = len(self._params.const_args)
    layouts_flat = self._executable._xla_in_layouts[nr_const_args:]  # pyrefly: ignore[missing-attribute]
    # Some input layouts got DCE'd
    if self.in_tree.num_leaves > len(layouts_flat):
      iter_layouts_flat = iter(layouts_flat)
      layouts_flat = [next(iter_layouts_flat) if i + nr_const_args in self._executable._kept_var_idx  # pyrefly: ignore[missing-attribute]
                      else None for i in range(self.in_tree.num_leaves)]
    return layouts_flat

  @property
  def input_formats(self):
    # Including dead args, but not const_args
    layouts_flat = self._input_layouts_flat()
    shardings_flat = self._input_shardings_flat()
    formats_flat = [Format(l, s) for l, s in zip(layouts_flat, shardings_flat)]
    return tree_util.tree_unflatten(self.in_tree, formats_flat)

  @property
  def _output_formats_flat(self):
    layouts_flat = self._executable._xla_out_layouts  # pyrefly: ignore[missing-attribute]
    shardings_flat = self._executable._out_shardings  # pyrefly: ignore[missing-attribute]
    assert all(isinstance(l, Layout) for l in layouts_flat)
    return [Format(l, s) for l, s in zip(layouts_flat, shardings_flat)]

  @property
  def output_formats(self):
    formats_flat = self._output_formats_flat
    return tree_util.tree_unflatten(self.out_tree, formats_flat)

  @staticmethod
  def call(*args, **kwargs):
    util.test_event("stages_compiled_call")
    # This is because `__call__` passes in `self._params` as the first argument.
    # Instead of making the call signature `call(params, *args, **kwargs)`
    # extract it from args because `params` can be passed as a kwarg by users
    # which might conflict here.
    params = args[0]
    args = args[1:]  # Not including const_args
    if params.no_kwargs and kwargs:
      kws = ', '.join(kwargs.keys())
      raise NotImplementedError(
          "function was compiled by a transformation that does not support "
          f"keyword arguments, but called with keyword arguments: {kws}")

    if params.is_high:
      hi_args_flat, hi_tree = tree_util.tracing_registry.flatten((args, kwargs))
      args_flat = [typeof(x).lower_val(x) for x in hi_args_flat]
      args_flat, in_tree = tree_util.tracing_registry.flatten(
          tree_util.tree_unflatten(hi_tree, args_flat))
    else:
      args_flat, in_tree = tree_util.tracing_registry.flatten((args, kwargs))

    # TODO(mattjj): improve wrong-number-of-args error
    if in_tree != params.in_tree:
      errs = list(tree_util.equality_errors_pytreedef(in_tree, params.in_tree))
      msg = []
      msg.append(
          "Function compiled with input pytree does not match the input pytree"
          f" it was called with. There are {len(errs)} mismatches, including:")
      for path, thing1, thing2, explanation in errs:
        fst, *rest = path
        base = ['args', 'kwargs'][fst.idx]
        msg.append(
            f"    * at {base}{tree_util.keystr(tuple(rest))}, seen {thing2} but now"
            f" given {thing1}, so {explanation}")
      raise TypeError('\n'.join(msg))

    if not core.trace_state_clean():
      # We check for tracers when we are under a transformation, and skip the
      # check in the common path. We can't transform ahead-of-time compiled
      # calls, since we've lowered and compiled for a fixed function signature,
      # and JAX transformations change signatures.
      for arg in args_flat:
        if isinstance(arg, core.Tracer):
          raise TypeError(
              "Cannot apply JAX transformations to a function lowered and "
              "compiled for a particular signature. Detected argument of "
              f"Tracer type {type(arg)}.")
    lo_outs = params.executable.call(*params.const_args, *args_flat)

    if params.is_high:
      out_hi_tree, out_hi_types = params.out_types
      out_flat = raise_lo_outs(out_hi_types, lo_outs)
      outs = tree_util.tree_unflatten(out_hi_tree, out_flat)
    else:
      out_flat = lo_outs
      outs = tree_util.tree_unflatten(params.out_tree, out_flat)

    return outs, out_flat, args_flat

  def __call__(self, *args, **kwargs):
    if self._call is None:
      self._call = self._executable.create_cpp_call(self._params)
      if self._call is None:
        params = self._params._replace(
            const_args=list(self._executable.shard_const_args(self._params.const_args)))
        def cpp_call_fallback(*args, **kwargs):
          outs, _, _ = Compiled.call(params, *args, **kwargs)
          return outs
        self._call = cpp_call_fallback
    return self._call(*args, **kwargs)

# TODO(mattjj): de-dup with partial_eval.py
def raise_lo_outs(hi_avals, lo_outs):
  lo_outs_ = iter(lo_outs)
  hi_outs = [t.raise_val(*it.islice(lo_outs_, len(t.lo_ty()))) for t in hi_avals]
  assert next(lo_outs_, None) is None
  return hi_outs

@runtime_checkable
class Wrapped(Protocol):
  """A function ready to be traced, lowered, and compiled.

  This protocol reflects the output of functions such as
  ``jax.jit``. Calling it results in JIT (just-in-time) lowering,
  compilation, and execution. It can also be explicitly lowered prior
  to compilation, and the result compiled prior to execution.
  """

  def __call__(self, *args, **kwargs):
    """Executes the wrapped function, lowering and compiling as needed."""
    raise NotImplementedError

  def trace(self, *args, **kwargs) -> Traced:
    """Trace this function explicitly for the given arguments.

    A traced function is staged out of Python and translated to a jaxpr. It is
    ready for lowering but not yet lowered.

    Returns:
      A ``Traced`` instance representing the tracing.
    """
    raise NotImplementedError

  def lower(self, *args, **kwargs) -> Lowered:
    """Lower this function explicitly for the given arguments.

    This is a shortcut for ``self.trace(*args, **kwargs).lower()``.

    A lowered function is staged out of Python and translated to a
    compiler's input language, possibly in a backend-dependent
    manner. It is ready for compilation but not yet compiled.

    Returns:
      A ``Lowered`` instance representing the lowering.
    """
    raise NotImplementedError


class MismatchType(enum.Enum):
  ARG_SHARDING = enum.auto()
  CONST_SHARDING = enum.auto()
  OUT_SHARDING = enum.auto()
  SHARDING_INSIDE_COMPUTATION = enum.auto()
  CONTEXT_DEVICES = enum.auto()
  IN_SHARDING = enum.auto()

  def __str__(self):
    if self.name == 'IN_SHARDING':
      return 'explicit input sharding'
    if self.name == 'CONST_SHARDING':
      return 'closed over constant sharding'
    elif self.name == 'OUT_SHARDING':
      return 'explicit output sharding'
    elif self.name == 'CONTEXT_DEVICES':
      return 'context mesh'
    return f'{self.name}'


class SourceInfo(NamedTuple):
  source_info: source_info_util.SourceInfo
  eqn_name: str


@dataclasses.dataclass(slots=True)
class DeviceAssignmentMismatch:
  da: Sequence[xc.Device] | int
  m_type: MismatchType
  source_info: SourceInfo | None

  @property
  def device_ids(self) -> Sequence[int]:
    return [d.id for d in self.da]  # pyrefly: ignore[not-iterable]

  @property
  def platform(self) -> str:
    return self.da[0].platform.upper()  # pyrefly: ignore[bad-index]

  def _maybe_api_name(self, api_name) -> str:
    return f" {api_name}'s" if self.m_type == MismatchType.CONTEXT_DEVICES else ""

  @property
  def source_info_str(self):
    return (
        "" if self.source_info is None
        else f" at {source_info_util.summarize(self.source_info.source_info)}"
    )

  @property
  def _dev_ids_plat_str(self):
    if isinstance(self.da, int):
      return f"as AbstractMesh of size {self.da}"
    else:
      return f"with device ids {self.device_ids} on platform {self.platform}"

  def m_type_str(self, api_name):
    return (f'{self.source_info and self.source_info.eqn_name} inside {api_name}'
            if self.m_type == MismatchType.SHARDING_INSIDE_COMPUTATION else self.m_type)

  def _str(self, api_name):
    return (f"{self._maybe_api_name(api_name)} {self.m_type_str(api_name)} "
            f"{self._dev_ids_plat_str}{self.source_info_str}")


class DeviceAssignmentMismatchError(Exception):
  pass


def _find_arg_mismatch(arg_list, fails, fun_name):
  mismatched_args_msg = []
  def mismatch(err):
    for name, inp_da, aval in arg_list:
      if err.m_type == MismatchType.ARG_SHARDING and err.da == inp_da:
        mismatched_args_msg.append(
            f"argument {name} of {fun_name} with shape {aval.str_short()} and "
            f"{err._dev_ids_plat_str}")
        break
  first_err, second_err = fails
  mismatch(first_err)
  mismatch(second_err)
  return mismatched_args_msg


def _device_assignment_mismatch_error(fun_name, fails, args_flat, api_name,
                                      arg_names):
  arg_list = []
  if arg_names is None:
    arg_names = [''] * len(args_flat)
  for a, n in zip(args_flat, arg_names):
    da = a.sharding._device_assignment if a.sharding is not None else None
    arg_list.append((n, da, a.aval))

  mismatched_args_msg = _find_arg_mismatch(arg_list, fails, fun_name)

  if len(mismatched_args_msg) == 2:
    first, second = mismatched_args_msg
    extra_msg = f" Got {first} and {second}"
  elif len(mismatched_args_msg) == 1:
    first, second = fails
    # Choose the failure left which is not already covered by ARG_SHARDING.
    left = second if first.m_type == MismatchType.ARG_SHARDING else first
    extra_msg = f" Got {mismatched_args_msg[0]} and{left._str(api_name)}"
  else:
    first, second = fails
    extra_msg = f" Got{first._str(api_name)} and{second._str(api_name)}"
  msg = (f"Received incompatible devices for {api_name}ted computation.{extra_msg}")
  return msg
