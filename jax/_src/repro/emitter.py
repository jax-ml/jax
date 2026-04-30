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
from collections.abc import Sequence, Iterator
import dataclasses
import itertools
import functools
from functools import partial
import logging
import os.path
import pathlib
import re
import sys
import threading
from typing import Any, Callable, Generic, NamedTuple, Type, TypeVar, Union


import numpy as np

from jax._src import config
from jax._src import path
from jax._src.util import safe_map
from jax._src import traceback_util
from jax._src import tree_util

from jax._src.repro import tracker
from jax._src.repro.tracker import (
  Call, Statement, FunctionDef, Func,
)
from jax._src.repro import repro_primitives

def maybe_singleton(x: list[Any]) -> Any | tuple[Any, ...]:
  return x[0] if len(x) == 1 else tuple(x)


# Maps types to source code emitters
_operand_emitter_by_type: dict[Type, Callable[["EmitFunctionDefContext", Any], str]] = {}


def register_emitter_by_type(typ, emitter: Callable[["EmitFunctionDefContext", Any], str]) -> Callable[["EmitFunctionDefContext", Any], str]:
  """Registers `emitter` to use to emit operands of type `typ`"""
  _operand_emitter_by_type[typ] = emitter
  return emitter


_operand_emitter_by_val: dict[Any, str] = {}

def register_emitter_by_val(val, val_str: str) -> None:
  """Registers `emitter` for a given value"""
  _operand_emitter_by_val[val] = val_str


def emit_enum(enum_name: str) -> Callable[["EmitFunctionDefContext", enum.Enum], str]:
  # For classes that derive from enum.Enum
  def emitter(ctx: "EmitFunctionDefContext", v: enum.Enum):
    return f"{enum_name}.{v.name}"
  return emitter

def emit_namedtuple(named_tuple_name: str) -> Callable[["EmitFunctionDefContext", NamedTuple], str]:
  def emitter(ctx: "EmitFunctionDefContext", v: NamedTuple):
    return f"{named_tuple_name}({ctx.traverse_sequence_value(v)})"  # type: ignore
  return emitter


def initialize_operand_emitter():
  repro_primitives.populate()

  # We wrap this in a function that is called late so that we can import here
  # other JAX modules
  _operand_emitter_by_type[set] = lambda ctx, v: repr(v)
  _operand_emitter_by_type[frozenset] = lambda ctx, v: repr(v)
  _operand_emitter_by_type[tracker.EmitLiterally] = lambda ctx, v: v.literal
  _operand_emitter_by_type[slice] = lambda ctx, v: repr(v)
  _operand_emitter_by_type[type(...)] = lambda ctx, v: "..."

  from jax._src import literals  # type: ignore
  # TODO: even better would be to fix the repr of these types
  _operand_emitter_by_type[literals.TypedNdArray] = (
      lambda ctx, v: f"literals.TypedNdArray({ctx.traverse_value_atom(v.val)})")

  _operand_emitter_by_type[literals.TypedFloat] = (
      lambda ctx, v: f"literals.TypedFloat({float(v)}, dtype={ctx.traverse_value_atom(v.dtype)})")

  _operand_emitter_by_type[literals.TypedInt] = (
      lambda ctx, v: f"literals.TypedInt({int(v)}, dtype={ctx.traverse_value_atom(v.dtype)})")

  _operand_emitter_by_type[literals.TypedComplex] = (
      lambda ctx, v: f"literals.TypedComplex({complex(v)}, dtype={ctx.traverse_value_atom(v.dtype)})")

  from jax._src import core  # type: ignore
  @partial(register_emitter_by_type, core.Primitive)
  def emit_primitive(ctx, v):
    repro_name = repro_primitives.primitives[v]
    return f"jax_primitive_bind(\"{repro_name}\")"

  from jax._src import dtypes  # type: ignore
  def emit_dtype(dt):
    if hasattr(np, dt.name):
      return f"np.{dt.name}"  # e.g., np.float32
    if hasattr(dtypes, dt.name):
      return f"dtypes.{dt.name}"  # e.g., dtypes.bfloat16
    else:
      return f"dtypes.dtype(\"{dt.name}\")"

  for t in dtypes._jax_types:
    _operand_emitter_by_val[t] = emit_dtype(t)
  for t in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64,
            np.float16, np.float32, np.float64, np.bool_,
            np.complex64, np.complex128]:
    _operand_emitter_by_val[t] = f"np.{t.__name__}"
  for t in dtypes._custom_float_scalar_types:
    _operand_emitter_by_val[t] = f"dtypes.{t.__name__}"
  for t in dtypes._intn_dtypes:
    _operand_emitter_by_val[t] = f"dtypes.{t.name}"

  def emit_np_array(ctx: EmitFunctionDefContext, v) -> str:
    if v.size <= tracker._thread_local_state.flags.fake_array_threshold:
      if not v.shape:  # For some dtypes, `repr` does not print the dtype, e.g., bfloat16
        if isinstance(v, np.ndarray):
          res = f"array({v}, dtype={emit_dtype(v.dtype)})"
        else:  # Use np scalar types when possible, they are hashable
          res = f"{emit_dtype(v.dtype)}({v})"

      else:
        res = repr(v)
    else:
      res = f"np.ones({v.shape}, dtype={ctx.traverse_value_atom(v.dtype)})"
    if v.size <= 8:
      return res
    return ctx.named_value(res, prefix="arr")

  _operand_emitter_by_type[np.ndarray] = emit_np_array
  _operand_emitter_by_type[np.generic] = emit_np_array

  from jax._src.array import ArrayImpl  # type: ignore
  @partial(register_emitter_by_type, ArrayImpl)
  def emit_jax_array(ctx: EmitFunctionDefContext, v) -> str:
    # TODO: emit it as a real jax.Array
    return emit_np_array(ctx, np.array(v))

  @partial(register_emitter_by_type, core.ShapeDtypeStruct)
  def emit_ShapeDtypeStruct(ctx: "EmitFunctionDefContext", v: core.ShapeDtypeStruct) -> str:
    dtype = ctx.traverse_value_atom(v.dtype)
    sharding = ctx.traverse_value(v.sharding)
    manual_axis_type = ctx.traverse_value(v.manual_axis_type)
    shape = ctx.traverse_value(v.shape)  # may contain DimExpr
    res = (f"jax.ShapeDtypeStruct({shape}, {dtype}, sharding={sharding}, "
           f"weak_type={v.weak_type}, manual_axis_type={manual_axis_type}, "
           f"is_ref={v.is_ref})")
    return ctx.named_value(res, prefix="sds")

  from jax._src.export.shape_poly import _DimExpr, _DimTerm, _DimFactor
  @partial(register_emitter_by_type, _DimExpr)
  def emit_DimExpr(ctx: "EmitFunctionDefContext", v: _DimExpr):
    # Almost like _DimExpr.__str__
    def _term_with_coeff(t: _DimTerm, t_k: int) -> str:
      abs_t_k = abs(t_k)
      sgn_t_k = "+" if t_k > 0 else "-"
      if t.is_constant:
        return f"{sgn_t_k} {abs_t_k}" if abs_t_k != 0 else "0"
      if abs_t_k == 1:
        return f"{sgn_t_k} {_term(t)}"
      return f"{sgn_t_k} {abs_t_k}*{_term(t)}"

    def _term(t: _DimTerm) -> str:
      return "*".join(f"{_factor(fact)}^{exponent}" if exponent != 1 else _factor(fact)
                      for fact, exponent in sorted(t._factors))

    def _factor(f: _DimFactor) -> str:
      if f.var is not None:
        # Make a _DimExpr from this var, and see if we have one that is equal
        # in the context. We use that because traverse_value_atom will lookup
        # by id()
        var = _DimExpr._from_var(f.var, v.scope)
        def lookup(l_ctx: EmitFunctionDefContext) -> str:
          for _, def_v in l_ctx.definitions_by_name.items():
            if isinstance(def_v, _DimExpr) and def_v == var:
              return ctx.traverse_value_atom(def_v)
          if l_ctx.parent_ctx is not None:
            return lookup(l_ctx.parent_ctx)
          return f.var  # This will result in error
        return lookup(ctx)

      v1, v2 = f.operands
      v1_str, v2_str = emit_DimExpr(ctx, v1), emit_DimExpr(ctx, v2)
      if f.operation == _DimFactor.MAX:
        op_str = "core.max_dim"
      elif f.operation == _DimFactor.MIN:
        op_str = "core.min_dim"
      else:
        op_str = f.operation

      return f"{op_str}({v1_str}, {v2_str})"

    # We print first the "larger" terms, so that the constant is last.
    res = " ".join(_term_with_coeff(t, t_k)
                   for t, t_k in v._sorted_terms)
    if res.startswith("+ "):
      res = res[2:]
    return res

  from jax._src.typing import DeprecatedArg  # type: ignore
  @partial(register_emitter_by_type, DeprecatedArg)
  def emit_DeprecatedArg(ctx: "EmitFunctionDefContext", v: DeprecatedArg):
    return "DeprecatedArg()"

  @partial(register_emitter_by_type, core.ShapedArray)
  def emit_ShapedArray(ctx: "EmitFunctionDefContext", v: core.ShapedArray) -> str:
    dtype = ctx.traverse_value(v.dtype)
    weak_type = ctx.traverse_value(v.weak_type)
    sharding = ctx.traverse_value(v.sharding)
    manual_axis_type = ctx.traverse_value(v.manual_axis_type)
    memory_space = ctx.traverse_value_atom(v.memory_space)
    res = (f"core.ShapedArray({ctx.traverse_value(v.shape)}, dtype={dtype}, "
           f"weak_type={weak_type}, sharding={sharding}, "
          f"manual_axis_type={manual_axis_type}, memory_space={memory_space})")
    return ctx.named_value(res, prefix="sha")

  @partial(register_emitter_by_type, core.ManualAxisType)
  def emit_ManualAxisType(ctx: "EmitFunctionDefContext", v: core.ManualAxisType):
    varying = ctx.traverse_value(v.varying)
    unreduced = ctx.traverse_value(v.unreduced)
    reduced = ctx.traverse_value(v.reduced)
    return (f"core.ManualAxisType(varying={varying}, unreduced={unreduced}, "
           f"reduced={reduced})")

  from jax._src import frozen_dict
  @partial(register_emitter_by_type, frozen_dict.FrozenDict)
  def emit_FrozenDict(ctx: "EmitFunctionDefContext", v: frozen_dict.FrozenDict) -> str:
    return f"FrozenDict({ctx.traverse_value(v._d)})"

  from jax import sharding  # type: ignore

  @partial(register_emitter_by_type, sharding.PartitionSpec)
  def emit_PartitionSpec(ctx: "EmitFunctionDefContext", v: sharding.PartitionSpec) -> str:
    partitions = ctx.traverse_sequence_value(v._partitions)
    if partitions:
      partitions += ", "
    res = f"PartitionSpec({partitions}unreduced={v.unreduced}, reduced={v.reduced})"
    return ctx.named_value(res, prefix="ps")

  from jax._src.lib import jaxlib_extension_version  # type: ignore
  if jaxlib_extension_version >= 446:
    from jax._src import partition_spec  # type: ignore
    @partial(register_emitter_by_type, partition_spec.UnconstrainedSingleton)  # type: ignore
    def emit_UnconstrainedSingleton(ctx: "EmitFunctionDefContext", v):
      return "PartitionSpec.UNCONSTRAINED"
  else:
    from jax._src import lib as jaxlib  # type: ignore
    @partial(register_emitter_by_type, jaxlib._jax.UnconstrainedSingleton)  # type: ignore
    def emit_UnconstrainedSingleton(ctx: "EmitFunctionDefContext", v):
      return "PartitionSpec.UNCONSTRAINED"

  _operand_emitter_by_type[sharding.AxisType] = emit_enum("AxisType")
  _operand_emitter_by_type[core.MemorySpace] = emit_enum("core.MemorySpace")

  @partial(register_emitter_by_type, sharding.AbstractMesh)
  def emit_AbstractMesh(ctx: "EmitFunctionDefContext", v: sharding.AbstractMesh) -> str:
    res = (f"AbstractMesh({v.axis_sizes}, {ctx.traverse_value(v.axis_names)}, "
           f"axis_types={ctx.traverse_value(v.axis_types)}, "
           f"abstract_device={ctx.traverse_value(v.abstract_device)})")
    return ctx.named_value(res, prefix="am")

  from jax._src.lib import xla_client  # type: ignore
  @partial(register_emitter_by_type, xla_client.Device)
  def emit_Device(ctx: "EmitFunctionDefContext", v: Any) -> str:
    return f"jax_get_device(\"{v.platform}\", {v.id})"

  from jax._src.lib import xla_client  # type: ignore
  @partial(register_emitter_by_type, sharding.AbstractDevice)
  def emit_AbstractDevice(ctx: "EmitFunctionDefContext", v: sharding.AbstractDevice) -> str:
    res = f"AbstractDevice(\"{v.device_kind}\", {v.num_cores}, \"{v.platform}\")"
    return ctx.named_value(res, prefix="ad")

  @partial(register_emitter_by_type, sharding.Mesh)
  def emit_Mesh(ctx: "EmitFunctionDefContext", v: sharding.Mesh) -> str:
    devices_list = v.devices.tolist()
    devices_str = ctx.traverse_value(devices_list)
    res = (f"Mesh(np.array({devices_str}), "
           f"axis_names={ctx.traverse_value(v.axis_names)}, "
           f"axis_types={ctx.traverse_value(v.axis_types)})")
    return ctx.named_value(res, prefix="mesh")

  @partial(register_emitter_by_type, sharding.NamedSharding)
  def emit_NamedSharding(ctx: "EmitFunctionDefContext", v: sharding.NamedSharding) -> str:
    mesh = ctx.traverse_value_atom(v.mesh)
    spec = ctx.traverse_value_atom(v.spec)
    memory_kind = ctx.traverse_value_atom(v.memory_kind)
    res = f"NamedSharding({mesh}, {spec}, memory_kind={memory_kind})"
    return ctx.named_value(res, prefix="ns")

  @partial(register_emitter_by_type, sharding.SingleDeviceSharding)
  def emit_SingleDeviceSharding(ctx: "EmitFunctionDefContext", v: sharding.SingleDeviceSharding) -> str:
    device = ctx.traverse_value_atom(v._device)
    memory_kind = ctx.traverse_value_atom(v.memory_kind)
    return f"make_single_device_sharding({device}, memory_kind={memory_kind})"

  from jax._src import shard_map  # type: ignore
  @partial(register_emitter_by_type, shard_map.InferFromArgs)
  def emit_InferFromArgs(ctx: "EmitFunctionDefContext",
                         v: shard_map.InferFromArgs) -> str:
    return "shard_map.Infer"

  from jax import lax  # type: ignore
  register_emitter_by_type(lax.AccuracyMode, emit_enum("lax.AccuracyMode"))
  _operand_emitter_by_type[lax.ConvDimensionNumbers] = emit_namedtuple("lax.ConvDimensionNumbers")
  _operand_emitter_by_type[lax.DotAlgorithm] = emit_namedtuple("lax.DotAlgorithm")
  _operand_emitter_by_type[lax.DotAlgorithmPreset] = emit_enum("lax.DotAlgorithmPreset")
  _operand_emitter_by_type[lax.FftType] = emit_namedtuple("lax.FftType")
  _operand_emitter_by_type[lax.GatherDimensionNumbers] = emit_namedtuple("lax.GatherDimensionNumbers")
  _operand_emitter_by_type[lax.GatherScatterMode] = emit_enum("lax.GatherScatterMode")
  _operand_emitter_by_type[lax.Precision] = emit_enum("lax.Precision")
  _operand_emitter_by_type[lax.RandomAlgorithm] = emit_enum("lax.RandomAlgorithm")
  _operand_emitter_by_type[lax.RoundingMethod] = emit_enum("lax.RoundingMethod")
  _operand_emitter_by_type[lax.ScatterDimensionNumbers] = emit_namedtuple("lax.ScatterDimensionNumbers")

  from jax._src.lax.lax import RaggedDotDimensionNumbers  # type: ignore
  @partial(register_emitter_by_type, RaggedDotDimensionNumbers)
  def emit_RaggedDotDimensionNumbers(ctx: "EmitFunctionDefContext",
                                     v) -> str:
    ddn = ctx.traverse_value(v.dot_dimension_numbers)
    lrd = ctx.traverse_value(v.lhs_ragged_dimensions)
    rgd = ctx.traverse_value(v.rhs_group_dimensions)
    return (f"RaggedDotDimensionNumbers("
            f"dot_dimension_numbers={ddn}, "
            f"lhs_ragged_dimensions={lrd}, "
            f"rhs_group_dimensions={rgd})")

  @partial(register_emitter_by_type, lax.Tolerance)
  def emit_Tolerance(ctx: "EmitFunctionDefContext", v: lax.Tolerance) -> str:
    return f"lax.Tolerance({v.atol}, {v.rtol}, {v.ulps})"

  from jax._src import layout  # type: ignore
  @partial(register_emitter_by_type, layout.Layout)
  def emit_Layout(ctx: "EmitFunctionDefContext", v: layout.Layout) -> str:
    return f"layout.Layout({v.major_to_minor}, {v.tiling}, sub_byte_element_size_in_bits={v.sub_byte_element_size_in_bits})"

  from jax._src.random import prng  # type: ignore
  @partial(register_emitter_by_type, prng.PRNGImpl)
  def emit_PRNGImpl(ctx: "EmitFunctionDefContext", v: prng.PRNGImpl) -> str:
    return f"resolve_prng_impl(\"{v.name}\")"

  @partial(register_emitter_by_type, prng.KeyTy)
  def emit_KeyTy(ctx: "EmitFunctionDefContext", v: prng.KeyTy) -> str:
    return f"prng.KeyTy({emit_PRNGImpl(ctx, v._impl)})"

  @partial(register_emitter_by_type, prng.PRNGKeyArray)
  def emit_PRNGKeyArray(ctx: "EmitFunctionDefContext", v: prng.PRNGKeyArray) -> str:
    res = f"prng.PRNGKeyArray({emit_PRNGImpl(ctx, v._impl)}, {ctx.traverse_value_atom(v._base_array)})"
    return ctx.named_value(res, prefix="key")

  _operand_emitter_by_type[xla_client.ArrayCopySemantics] = \
      emit_enum("xla_client.ArrayCopySemantics")

  from jax._src.xla_metadata_lib import XlaMetadata  # type: ignore
  @partial(register_emitter_by_type, XlaMetadata)
  def emit_XlaMetadata(ctx: "EmitFunctionDefContext", v: XlaMetadata) -> str:
    return f"XlaMetadata({ctx.traverse_value(v.val)})"

  _operand_emitter_by_type[config.ExplicitX64Mode] = \
      emit_enum("config.ExplicitX64Mode")
  _operand_emitter_by_type[config.NumpyDtypePromotion] = \
      emit_enum("config.NumpyDtypePromotion")

  from jax import export  # type: ignore
  @partial(register_emitter_by_type, export.DisabledSafetyCheck)
  def emit_DisabledSafetyCheck(ctx: "EmitFunctionDefContext", v: export.DisabledSafetyCheck):
    if v._impl == export.DisabledSafetyCheck.platform()._impl:
      return "export.DisabledSafetyCheck.platform()"
    if (cc := v.is_custom_call()) is not None:
      return f"export.DisabledSafetyCheck.custom_call(\"{cc}\")"
    raise NotImplementedError(v)

  from jax._src.state import indexing  # type: ignore
  @partial(register_emitter_by_type, indexing.Slice)
  def emit_Slice(ctx: "EmitFunctionDefContext", v: indexing.Slice) -> str:
    start = ctx.traverse_value_atom(v.start)
    size = ctx.traverse_value_atom(v.size)
    stride = ctx.traverse_value_atom(v.stride)
    return f"indexing.Slice({start}, {size}, {stride})"

  @partial(register_emitter_by_type, indexing.NDIndexer)
  def emit_NDIndexer(ctx: "EmitFunctionDefContext", v: indexing.NDIndexer) -> str:
    indices = ctx.traverse_value(v.indices)
    shape = ctx.traverse_value(v.shape)
    int_indexer_shape = ctx.traverse_value(v.int_indexer_shape)
    return f"indexing.NDIndexer({indices}, {shape}, {int_indexer_shape})"

  from jax._src.state import types as state_types  # type: ignore
  @partial(register_emitter_by_type, state_types.TransformedRef)
  def emit_TransformedRef(ctx: "EmitFunctionDefContext", v: state_types.TransformedRef) -> str:
    ref = ctx.traverse_value_atom(v.ref)
    transforms = ctx.traverse_value(v.transforms)
    res = f"state_types.TransformedRef({ref}, {transforms})"
    return ctx.named_value(res, prefix="tref")

  @partial(register_emitter_by_type, state_types.ReshapeTransform)
  def emit_ReshapeTransform(ctx: "EmitFunctionDefContext", v: state_types.ReshapeTransform) -> str:
    return f"state_types.ReshapeTransform({ctx.traverse_value(v.shape)})"

  @partial(register_emitter_by_type, state_types.BitcastTransform)
  def emit_BitcastTransform(ctx: "EmitFunctionDefContext", v: state_types.BitcastTransform) -> str:
    return f"state_types.BitcastTransform({ctx.traverse_value(v.dtype)})"

  @partial(register_emitter_by_type, state_types.TransposeTransform)
  def emit_TransposeTransform(ctx: "EmitFunctionDefContext", v: state_types.TransposeTransform) -> str:
    return f"state_types.TransposeTransform({ctx.traverse_value(v.permutation)})"
  @partial(register_emitter_by_type, state_types.Uninitialized)
  def emit_Uninitialized(ctx: "EmitFunctionDefContext", v: state_types.Uninitialized) -> str:
    return "state_types.uninitialized"  # Code uses id checks, use singleton

  initialize_operand_emitter_pallas_core()
  initialize_operand_emitter_pallas_tpu()
  initialize_operand_emitter_pallas_gpu()
  initialize_operand_emitter_hijax()


def initialize_operand_emitter_pallas_core():
  try:
    from jax._src.pallas import core as pallas_core  # type: ignore
  except ImportError:
    return

  @partial(register_emitter_by_type, pallas_core.GridSpec)
  def emit_GridSpec(ctx: "EmitFunctionDefContext", v: pallas_core.GridSpec) -> str:
    grid = ctx.traverse_value(v.grid)
    in_specs = ctx.traverse_value(v.in_specs)
    out_specs = ctx.traverse_value(v.out_specs)
    scratch_shapes = ctx.traverse_value(v.scratch_shapes)
    res = (f"pallas_core.GridSpec(grid={grid}, "
           f"in_specs={in_specs}, out_specs={out_specs}, "
           f"scratch_shapes={scratch_shapes})")
    return ctx.named_value(res, prefix="gs")

  register_emitter_by_type(pallas_core.NoBlockSpec, lambda ctx, v: "pallas_core.no_block_spec")

  @partial(register_emitter_by_type, pallas_core.BlockSpec)
  def emit_BlockSpec(ctx: "EmitFunctionDefContext", v: pallas_core.BlockSpec) -> str:
    index_map = ctx.traverse_value_atom(v.index_map)
    res = (f"pallas_core.BlockSpec(block_shape={v.block_shape}, "
           f"index_map={index_map}, "
           f"memory_space={ctx.traverse_value(v.memory_space)}, "
           f"pipeline_mode={v.pipeline_mode})")
    return ctx.named_value(res, prefix="bs")

  @partial(register_emitter_by_type, pallas_core._IndexMapFunc)
  def emit_IndexMapFunc(ctx: "EmitFunctionDefContext", v: pallas_core._IndexMapFunc) -> str:
    # The _IndexMapFunc is added upon construction of BlockSpec, it is
    # ok to strip it when emitting
    return ctx.traverse_value_atom(v.index_map)

  _operand_emitter_by_type[pallas_core.MemorySpace] = emit_enum("pallas_core.MemorySpace")

  @partial(register_emitter_by_type, pallas_core.CostEstimate)
  def emit_CostEstimate(ctx: "EmitFunctionDefContext", v: pallas_core.CostEstimate) -> str:
    res = f"pallas_core.CostEstimate({v.flops}, {v.transcendentals}, {v.bytes_accessed}, {v.remote_bytes_transferred})"
    return ctx.named_value(res, prefix="ce")

  @partial(register_emitter_by_type, pallas_core.MemoryRef)
  def emit_MemoryRef(ctx: "EmitFunctionDefContext", v: pallas_core.MemoryRef) -> str:
    inner_aval = ctx.traverse_value_atom(v.inner_aval)
    memory_space = ctx.traverse_value_atom(v.memory_space)
    res = f"pallas_core.MemoryRef({inner_aval}, {memory_space})"
    return ctx.named_value(res, prefix="mr")

  @partial(register_emitter_by_type, pallas_core.Semaphore)
  def emit_Semaphore(ctx: "EmitFunctionDefContext", v: pallas_core.Semaphore) -> str:
    return "pallas_core.Semaphore()"

  @partial(register_emitter_by_type, pallas_core.BarrierSemaphore)
  def emit_BarrierSemaphore(ctx: "EmitFunctionDefContext",
                            v: pallas_core.BarrierSemaphore) -> str:
    return "pallas_core.BarrierSemaphore()"

  @partial(register_emitter_by_type, pallas_core.CoreMemorySpace)
  def emit_CoreMemorySpace(ctx: "EmitFunctionDefContext",
                           v: pallas_core.CoreMemorySpace) -> str:
    memory_space = ctx.traverse_value_atom(v.memory_space)
    mesh = ctx.traverse_value_atom(v.mesh)
    res = f"pallas_core.CoreMemorySpace(memory_space={memory_space}, mesh={mesh})"
    return ctx.named_value(res, prefix="cms")


def initialize_operand_emitter_pallas_tpu():
  try:
    from jax.experimental.pallas import tpu as pltpu  # type: ignore
    from jax._src.pallas.mosaic import tpu_info  # type: ignore
    from jax._src.pallas.mosaic import pipeline as tpu_pipeline  # type: ignore
  except ImportError:
    return

  @partial(register_emitter_by_type, pltpu.PrefetchScalarGridSpec)
  def emit_PrefetchScalarGridSpec(ctx: "EmitFunctionDefContext", v: pltpu.PrefetchScalarGridSpec) -> str:
    grid = ctx.traverse_value(v.grid)
    in_specs = ctx.traverse_value(v.in_specs)
    out_specs = ctx.traverse_value(v.out_specs)
    scratch_shapes = ctx.traverse_value(v.scratch_shapes)
    res = (f"pltpu.PrefetchScalarGridSpec(grid={grid}, "
           f"in_specs={in_specs}, out_specs={out_specs}, "
           f"scratch_shapes={scratch_shapes}, num_scalar_prefetch={v.num_scalar_prefetch})")
    return ctx.named_value(res, prefix="gs")

  @partial(register_emitter_by_type, pltpu.CompilerParams)
  def emit_CompilerParams(ctx: "EmitFunctionDefContext", v: pltpu.CompilerParams) -> str:
    dimension_semantics = ctx.traverse_value(v.dimension_semantics)
    allow_input_fusion = ctx.traverse_value(v.allow_input_fusion)
    flags = ctx.traverse_value(v.flags)
    res = (f"pltpu.CompilerParams(dimension_semantics={dimension_semantics}, "
           f"allow_input_fusion={allow_input_fusion}, vmem_limit_bytes={v.vmem_limit_bytes}, "
           f"collective_id={v.collective_id}, has_side_effects={v.has_side_effects}, "
           f"flags={flags}, internal_scratch_in_bytes={v.internal_scratch_in_bytes}, "
           f"serialization_format={v.serialization_format}, "
           f"disable_bounds_checks={v.disable_bounds_checks}, skip_device_barrier={v.skip_device_barrier}, "
           f"allow_collective_id_without_custom_barrier={v.allow_collective_id_without_custom_barrier})")
    return ctx.named_value(res, prefix="cp")

  _operand_emitter_by_type[pltpu.CoreType] = emit_enum("pltpu.CoreType")
  _operand_emitter_by_type[pltpu.GridDimensionSemantics] = emit_enum("pltpu.GridDimensionSemantics")
  _operand_emitter_by_type[pltpu.MemorySpace] = emit_enum("pltpu.MemorySpace")
  _operand_emitter_by_type[pltpu.SemaphoreType] = emit_enum("pltpu.SemaphoreType")
  _operand_emitter_by_type[tpu_info.Tiling] = emit_enum("pltpu.Tiling")

  from jax._src.pallas.mosaic.core import DMASemaphore  # type: ignore
  _operand_emitter_by_type[DMASemaphore] = (
      lambda ctx, v: "pltpu.SemaphoreType.DMA.dtype")

  from jax._src.pallas.primitives import DeviceIdType  # type: ignore
  _operand_emitter_by_type[DeviceIdType] = emit_enum("jax.experimental.pallas.DeviceIdType")

  from jax._src.pallas.mosaic.core import TensorCoreMesh  # type: ignore
  @partial(register_emitter_by_type, TensorCoreMesh)
  def emit_TensorCoreMesh(ctx: "EmitFunctionDefContext", v: TensorCoreMesh) -> str:
    # Emit using the factory function create_tensorcore_mesh
    axis_names = v.axis_names
    num_cores = len(v.devices)
    if len(axis_names) == 1:
      res = f"pltpu.create_tensorcore_mesh({axis_names[0]!r}, num_cores={num_cores})"
    else:
      raise NotImplementedError(f"TensorCoreMesh with multiple axes: {axis_names}")
    return ctx.named_value(res, prefix="mesh")

  from jax._src.pallas.mosaic.interpret.params import InterpretParams  # type: ignore
  @partial(register_emitter_by_type, InterpretParams)
  def emit_InterpretParams(ctx: "EmitFunctionDefContext", v: InterpretParams) -> str:
    # Emit only non-default fields
    defaults = InterpretParams()
    kwargs = []
    for f in dataclasses.fields(v):
      val = getattr(v, f.name)
      default_val = getattr(defaults, f.name)
      if val != default_val:
        kwargs.append(f"{f.name}={val!r}")
    args_str = ", ".join(kwargs)
    res = f"pltpu.InterpretParams({args_str})"
    return ctx.named_value(res, prefix="ip")

  from jax._src.pallas.mosaic.sc_core import VectorSubcoreMesh, ScalarSubcoreMesh  # type: ignore
  @partial(register_emitter_by_type, VectorSubcoreMesh)
  def emit_VectorSubcoreMesh(ctx: "EmitFunctionDefContext", v: VectorSubcoreMesh) -> str:
    res = (f"plsc.VectorSubcoreMesh(core_axis_name={v.core_axis_name!r}, "
           f"subcore_axis_name={v.subcore_axis_name!r}, "
           f"num_cores={v.num_cores}, num_subcores={v.num_subcores})")
    return ctx.named_value(res, prefix="mesh")

  @partial(register_emitter_by_type, ScalarSubcoreMesh)
  def emit_ScalarSubcoreMesh(ctx: "EmitFunctionDefContext", v: ScalarSubcoreMesh) -> str:
    res = (f"plsc.ScalarSubcoreMesh(axis_name={v.axis_name!r}, "
           f"num_cores={v.num_cores})")
    return ctx.named_value(res, prefix="mesh")

  from jax._src.pallas.mosaic.sc_core import Indices  # type: ignore
  @partial(register_emitter_by_type, Indices)
  def emit_Indices(ctx: "EmitFunctionDefContext", v: Indices) -> str:
    values = ctx.traverse_value(v.values)
    ignored_value = ctx.traverse_value_atom(v.ignored_value)
    return f"plsc.Indices({values}, ignored_value={ignored_value})"

  @partial(register_emitter_by_type, tpu_pipeline.BufferedRef)
  def emit_BufferedRef(ctx: "EmitFunctionDefContext", v: tpu_pipeline.BufferedRef) -> str:
    # We only register an emitter for BufferedRef sp that we don't normalize it.
    # These are never created by user code, so we will not need to emit them.
    return "tpu_pipeline.BufferedRef(unimplemented)"


def initialize_operand_emitter_pallas_gpu():
  try:
    from jax._src.pallas.mosaic_gpu import core as plgpu_core  # type: ignore
    from jax.experimental.pallas import mosaic_gpu as plgpu  # type: ignore
    from jax._src.pallas.mosaic_gpu import pipeline as plgpu_pipeline  # type: ignore
  except ImportError:
    return

  @partial(register_emitter_by_type, plgpu_core.TilingTransform)
  def emit_TilingTransform(ctx: "EmitFunctionDefContext", v: plgpu_core.TilingTransform) -> str:
    return f"plgpu_core.TilingTransform({v.tiling})"

  @partial(register_emitter_by_type, plgpu_core.SwizzleTransform)
  def emit_SwizzleTransform(ctx: "EmitFunctionDefContext", v: plgpu_core.SwizzleTransform) -> str:
    return f"plgpu_core.SwizzleTransform({v.swizzle})"

  @partial(register_emitter_by_type, plgpu_core.UntilingTransform)
  def emit_UntilingTransform(ctx: "EmitFunctionDefContext", v: plgpu_core.UntilingTransform) -> str:
    return f"plgpu_core.UntilingTransform({v.tiling})"

  @partial(register_emitter_by_type, plgpu_core.UnswizzleRef)
  def emit_UnswizzleRef(ctx: "EmitFunctionDefContext", v: plgpu_core.UnswizzleRef) -> str:
    return f"plgpu_core.UnswizzleRef({v.swizzle})"

  @partial(register_emitter_by_type, plgpu_core.GPUMemoryRef)
  def emit_GPUMemoryRef(ctx: "EmitFunctionDefContext", v: plgpu_core.GPUMemoryRef) -> str:
    inner_aval = ctx.traverse_value_atom(v.inner_aval)
    memory_space = ctx.traverse_value_atom(v.memory_space)
    transforms = ctx.traverse_value(v.transforms)
    layout = ctx.traverse_value(v.layout)
    collective = ctx.traverse_value(v.collective)
    res = (f"plgpu_core.GPUMemoryRef({inner_aval}, {memory_space}, "
           f"transforms={transforms}, layout={layout}, collective={collective})")
    return ctx.named_value(res, prefix="mr")

  @partial(register_emitter_by_type, plgpu_core.WGMMAAccumulatorRef)
  def emit_WGMMAAccumulatorRef(ctx: "EmitFunctionDefContext", v: plgpu_core.WGMMAAccumulatorRef) -> str:
    dtype = ctx.traverse_value(v.dtype)
    _init = ctx.traverse_value(v._init)
    res = f"plgpu_core.WGMMAAccumulatorRef(shape={v.shape}, dtype={dtype}, _init={_init})"
    return ctx.named_value(res, prefix="mr")

  @partial(register_emitter_by_type, plgpu_core.CompilerParams)
  def emit_CompilerParams(ctx: "EmitFunctionDefContext", v: plgpu_core.CompilerParams) -> str:
    dimension_semantics = ctx.traverse_value(v.dimension_semantics)
    profile_trace_scope = ctx.traverse_value(v.profile_trace_scope)
    lowering_semantics = ctx.traverse_value(v.lowering_semantics)
    res = (f"plgpu_core.CompilerParams(approx_math={v.approx_math}, "
           f"dimension_semantics={dimension_semantics}, max_concurrent_steps={v.max_concurrent_steps}, "
           f"unsafe_no_auto_barriers={v.unsafe_no_auto_barriers}, reduction_scratch_bytes={v.reduction_scratch_bytes}, "
           f"profile_space={v.profile_space}, profile_dir={v.profile_dir!r}, "
           f"profile_trace_scope={profile_trace_scope}, lowering_semantics={lowering_semantics})")
    return ctx.named_value(res, prefix="cp")

  _operand_emitter_by_type[plgpu_core.TraceScope] = emit_enum("plgpu_core.TraceScope")
  _operand_emitter_by_type[plgpu_core.MemorySpace] = emit_enum("plgpu_core.MemorySpace")
  _operand_emitter_by_type[plgpu_core.SemaphoreType] = emit_enum("plgpu_core.SemaphoreType")
  _operand_emitter_by_type[plgpu.LoweringSemantics] = emit_enum("plgpu.LoweringSemantics")
  _operand_emitter_by_type[plgpu_core.TMEMLayout] = emit_enum("plgpu_core.TMEMLayout")
  _operand_emitter_by_type[plgpu.Layout] = emit_enum("plgpu.Layout")

  @partial(register_emitter_by_type, plgpu_core.Barrier)
  def emit_Barrier_gpu(ctx: "EmitFunctionDefContext", v: plgpu_core.Barrier) -> str:
    res = f"plgpu_core.Barrier(num_arrivals={v.num_arrivals}, num_barriers={v.num_barriers}, orders_tensor_core={v.orders_tensor_core})"
    return ctx.named_value(res, prefix="bar")

  @partial(register_emitter_by_type, plgpu_pipeline.BufferedRef)
  def emit_BufferedRef_gpu(ctx: "EmitFunctionDefContext", v: plgpu_pipeline.BufferedRef) -> str:
    spec = ctx.traverse_value(v.spec)
    gmem_ref = ctx.traverse_value(v.gmem_ref)
    smem_ref = ctx.traverse_value(v.smem_ref)
    res = f"plgpu_pipeline.BufferedRef(spec={spec}, is_index_invariant={v.is_index_invariant}, gmem_ref={gmem_ref}, smem_ref={smem_ref})"
    return ctx.named_value(res, prefix="bref")

  @partial(register_emitter_by_type, plgpu_core.BlockSpec)
  def emit_BlockSpec_gpu(ctx: "EmitFunctionDefContext", v: plgpu_core.BlockSpec) -> str:
    index_map = ctx.traverse_value_atom(v.index_map)
    memory_space = ctx.traverse_value(v.memory_space)
    transforms = ctx.traverse_value(v.transforms)
    collective_axes = ctx.traverse_value(v.collective_axes)
    res = (f"plgpu_core.BlockSpec(block_shape={v.block_shape}, "
           f"index_map={index_map}, "
           f"memory_space={memory_space}, "
           f"pipeline_mode={v.pipeline_mode}, "
           f"transforms={transforms}, "
           f"delay_release={v.delay_release}, "
           f"collective_axes={collective_axes})")
    return ctx.named_value(res, prefix="bs")


def emit_hival(ctx: "EmitFunctionDefContext", v: object) -> str:
  from jax._src import core  # type: ignore
  t = core.typeof(v)
  lo_vals = t.lower_val(v)
  t_str = ctx.traverse_value_atom(t)
  lo_vals_str = ctx.traverse_sequence_value(lo_vals)
  return f"ReproHiVal({t_str}, ({lo_vals_str},))"


# TODO: hack + not threadsafe + need weakrefs
_hival_classes_lock = threading.Lock()
_hival_classes: list[type] = []


def register_hival_class(cls: type) -> None:
  with _hival_classes_lock:
    _hival_classes.append(cls)
    if tracker.lazy_initializers is None:
      register_emitter_by_type(cls, emit_hival)


def initialize_operand_emitter_hijax():
  # Already holds the tracker._lazy_init_lock
  try:
    from jax._src import hijax  # type: ignore
  except ImportError:
    return

  idx = 0
  while idx < len(_hival_classes):
    register_emitter_by_type(_hival_classes[idx], emit_hival)
    idx += 1

  @partial(register_emitter_by_type, hijax.HiType)
  def emit_hitype(ctx: "EmitFunctionDefContext", v: hijax.HiType) -> str:
    from jax._src.repro import repro_api  # type: ignore
    if isinstance(v, repro_api.ReproHiType):
      to_tangent_aval = ctx.traverse_value(v._to_tangent_aval)  # type: ignore
      name = v.name
      unique_id = v.unique_id
    else:
      name = type(v).__name__
      unique_id = repro_api.hitype_memo[v]
      to_tangent_aval = v.to_tangent_aval()
      if isinstance(to_tangent_aval, hijax.HiType):
        if to_tangent_aval == v:
          to_tangent_aval = "self"
        else:
          raise NotImplementedError

    to_tangent_aval_str = ctx.traverse_value(to_tangent_aval)
    return f"ReproHiType(\"{name}\", {unique_id}, tangent_aval={to_tangent_aval_str})"

  @partial(register_emitter_by_type, hijax.MappingSpec)
  def emit_mappingspec(ctx: "EmitFunctionDefContext", v: hijax.MappingSpec) -> str:
    from jax._src.repro import repro_api  # type: ignore
    if isinstance(v, repro_api.ReproMappingSpec):
      return f"ReproMappingSpec(\"{v.name}\")"
    return f"ReproMappingSpec(\"{str(v)}\")"


tracker.lazy_initializers.append(initialize_operand_emitter)  # type: ignore

@tracker.lazy_initializers.append  # type: ignore
def _():
  def save_statement(stmt: Statement, *,
                     extra_comment: str = "",
                     repro_name_prefix: str):
    source = to_source(stmt, extra_comment=extra_comment)
    save(source, repro_name_prefix=repro_name_prefix)

  tracker.save_statement_handler = save_statement


@dataclasses.dataclass
class EmittedFunction:

  # The source, including the def line.
  lines: list[str]
  # The immediate externals are those referenced in the `lines`, using
  # global variable names
  immediate_externals: dict[int, Any]
  # All externals include the `immediate_externals` and also those referenced
  # in functions that are recursively referenced by the `immediate_externals`.
  all_externals: dict[int, Any]


EmitReductionCandidate = TypeVar("EmitReductionCandidate")

def get_emitter(v: Any) -> Callable[["EmitFunctionDefContext", Any], str] | None:
  try:
    vn = _operand_emitter_by_val.get(v)
  except TypeError:
    vn = None
  if vn is not None:
    return lambda _, v: vn

  for t in type(v).__mro__:
    if (emitter := _operand_emitter_by_type.get(t)) is not None:
      return emitter
  return None


class EmitReductionStrategy(Generic[EmitReductionCandidate]):
  # TODO: document the interface
  def keep_call(self, c: "Statement") -> bool:
    # TODO: merge into rewrite_statement
    return True

  def keep_expression(self, c: "Call", for_args: bool, idx: int, v: Any) -> bool:
    return True

  def rewrite_statement(self, s: "Statement") -> tuple[Any, tuple[Any, ...], dict[str, Any], Any]:
    return s.func, s.args, s.kwargs, s.result  # type: ignore

  def undefined_value(self, v: Any) -> str:
    from jax._src.random import prng  # type: ignore
    def fake_array(a) -> str:
      if isinstance(a.dtype, prng.KeyTy):
        return f"fake_prng_key(resolve_prng_impl(\"{a.dtype._impl.name}\"), {a.shape})"
      else:
        return f"np.ones({a.shape}, dtype={_operand_emitter_by_val[a.dtype]})"

    if hasattr(v, "shape") and hasattr(v, "dtype"):
      return fake_array(v)

    raise tracker.ReproError(f"undefined value handler for {type(v)}: {v}")


class EmitGlobalContext:
  var_for_val: dict[int, str] # id(val) -> var_name
  var_index: Iterator[int]

  # Cache here the emitted function body, for user functions, by fun.id
  emitted_functions: dict[int, EmittedFunction]

  # Dumped functions. TODO: emit above means that we generated the source
  # but we did not necessarily dumped it
  dumped_functions: dict[str, str]  # by `f.python_name` -> `f.fun_info`

  # Which Call are we traversing values for, and whether it is for args or
  # for results. For USER functions we generate the body, and for JAX functions
  # we generate the calls.
  current_traverse_value_context: tuple[Call, bool, Iterator[int]] | None = None
  emit_reduction_strategy: EmitReductionStrategy | None = None

  def __init__(self, *,
               strategy: EmitReductionStrategy | None = None):
    self.var_name_for_val_dict: dict[int, str] = {}
    self.var_name_index = itertools.count()
    self.emit_reduction_strategy = strategy
    self.emitted_functions: dict[int, EmittedFunction] = {}  # By fun_def.id
    self.dumped_functions = {}
    self.current_traverse_value_context = None

  def set_current_traverse_value_context(self, c: Call, for_args: bool):
    # TODO: rename maybe, make it clear what this is for
    if self.emit_reduction_strategy:
      self.current_traverse_value_context = (c, for_args, itertools.count())

  def keep_expression(self, v: Any) -> bool:
    if self.current_traverse_value_context is not None:
      c, for_args, iter = self.current_traverse_value_context
      return self.emit_reduction_strategy.keep_expression(  # type: ignore
          c, for_args, next(iter), v)
    else:
      return True

  def var_name_for_val(self, for_v: Any, *, prefix="g") -> str:
    v_id = id(for_v)
    vn = self.var_name_for_val_dict.get(v_id, None)
    if vn is None:
      if isinstance(for_v, Func):
        vn = for_v.python_name()
      else:
        vn = f"{prefix}_{next(self.var_name_index)}"
      self.var_name_for_val_dict[v_id] = vn
    return vn


# A Path is a string representation of an indexer into a pytree value.
Path = str


class EmitFunctionDefContext:

  def __init__(self, debug_name: str,
               global_ctx: EmitGlobalContext,
               parent_ctx: Union["EmitFunctionDefContext", None]):
    self.debug_name = debug_name
    self.global_ctx: EmitGlobalContext = global_ctx
    self.parent_ctx = parent_ctx
    self.emitted_function = EmittedFunction([], {}, {})
    self.local_name_index = itertools.count()
    # For all the values defined here, their local name (by id)
    self.definitions: dict[int, str] = {}
    # The following is not needed for functionality, but has helped debug
    # undefined values.
    self.definitions_by_name: dict[str, Any] = {}

    # We map some value definitions to a value name
    self.named_values: dict[str, str] = {}
    self.current_traceback: Union["Traceback", None] = None  # type: ignore  # noqa: F821

    self.entry_state_context: tracker.StateContext = tracker.StateContext(
      default_lowering_platform=None, trace_context=())
    # the state_context for the last statement emitted
    self.current_state_context: tracker.StateContext | None = None  # None on entry to top-level function

    # Not empty where we are in one or more "with ..." blocks
    self.current_context_indent: str = ""

  def emit_line(self, l: str, indent: bool = True):
    if indent:
      l = "  " + l
    self.emitted_function.lines.append(l)

  def emit_newline(self):
    self.emitted_function.lines.append("")

  def new_local_name(self, *, prefix="v") -> str:
    return f"{prefix}_{next(self.local_name_index)}"

  def define_value(self, v: Any, vn: str) -> None:
    v_id = id(v)
    if v_id not in self.definitions:
      # TODO: we may need some way to register definitions, just like we
      # register emitters
      # Sometimes we see a function invoked with the same value for multiple
      # args. Keep the first definition.
      from jax._src.state import types as state_types  # type: ignore
      from jax._src.state import indexing  # type: ignore
      from jax._src.pallas import core as pallas_core  # type: ignore
      if isinstance(v, state_types.TransformedRef):
        # For some functions, like pipeline, the arguments are
        # v=TransformedRef(r, transform=t1) and we could apply transformations
        # before using `v`. Unlike other JAX operations, ref transformations
        # are not primitive. In order for the emitter to be able to construct
        # transformed ref values, we must decompose and expose accessors to
        # the components.
        self.define_value(v.ref, f"{vn}.ref")
        for i, t in enumerate(v.transforms):
          if isinstance(t, indexing.NDIndexer):
            for j, idx in enumerate(t.indices):
              self.define_value(idx, f"{vn}.transforms[{i}].indices[{j}]")

      elif isinstance(v, pallas_core.BlockSpec):
        # Some JAX functions return BlockSpec, e.g., get_fusible_values,
        # and we must expose the index_map inside as defined values, because
        # they may be referenced.
        assert isinstance(v.index_map, pallas_core._IndexMapFunc)
        f = v.index_map.index_map
        self.define_value(f, f"{vn}.index_map")

      elif type(v) in _hival_classes:
        from jax._src import core  # type: ignore
        t = core.typeof(v)
        lo_vals = t.lower_val(v)
        for i, v in enumerate(lo_vals):
          self.define_value(v, f"{vn}.lo_vals[{i}]")

      self.definitions[v_id] = vn
      self.definitions_by_name[vn] = v

  def named_value(self, val_str: str, *, prefix="v") -> str:
    vn = self.named_values.get(val_str)
    if vn is None:
      vn = self.new_local_name(prefix=prefix)
      self.named_values[val_str] = vn
      self.emit_line(f"{vn} = {val_str}")
    return vn

  def lookup_value(self, v: Any) -> str | None:
    v_id = id(v)
    frame = self
    while True:
      if (vn := frame.definitions.get(v_id)) is not None:
        if frame is self:
          return vn
        else:
          self.emitted_function.immediate_externals[v_id] = v
          self.emitted_function.all_externals[v_id] = v
          return self.global_ctx.var_name_for_val(v, prefix="g")
      frame = frame.parent_ctx  # type: ignore
      if not frame: break
    return None

  def use_value(self, v: Any) -> str:
    if (vn := self.lookup_value(v)) is not None:
      return vn

    if self.global_ctx.emit_reduction_strategy:
      return self.global_ctx.emit_reduction_strategy.undefined_value(v)

    # An undefined value, log details in the emitted repro.
    vn = self.global_ctx.var_name_for_val(v, prefix="g")
    msg = f"Undefined {vn} = {v} of type {type(v)}"
    from jax._src import core  # type: ignore
    if not isinstance(v, (core.Tracer, Func, tree_util.Partial)):
      msg += " (non-tracer without custom emitter)"
    for m in msg.splitlines():
      self.emit_line(f"# {m}")

    tracker._thread_local_state.warn_or_error(
        msg, traceback=self.current_traceback,
        during_tracking=False)
    return vn

  def traverse_value(
      self, a, path: Path = "", paths: list[tuple[Path, Any]] | None = None) -> str | None:
    """We traverse normalized values (see tracker.normalize_value).

    We use this traversal in two cases: (1) to emit a string representation
    that can be used when the value is passed as argument to a call, or in a
    return statement. In this case `paths` is `None` and the function returns a
    string. And, (2) to construct a list of paths, e.g., `[0]['field']` to
    access components of the rendered value when accessing a function definition
    arguments the values returned from a call. In this case, the returned
    string is not used.
    """
    if not self.global_ctx.keep_expression(a):
      return None
    if isinstance(a, tuple) and not hasattr(a, "_fields"):  # skip NamedTuple
      return f"({self.traverse_sequence_value(a, path, paths)}{',' if a else ''})"
    if isinstance(a, list):
      return f"[{self.traverse_sequence_value(a, path, paths)}]"
    if isinstance(a, dict):
      kv = self.traverse_key_value_sequence(a, path, paths,
                                            keys_are_identifiers=False)
      return f"{{{kv}}}"
    return str(self.traverse_value_atom(a, path, paths))

  def traverse_sequence_value(
      self, a: Sequence,
      path: Path = "", paths: list[tuple[Path, Any]] | None = None) -> str:
    acc = []
    for i, v in enumerate(a):
      if (vn := self.traverse_value(v, path + f"[{i}]", paths)) is not None:
        acc.append(vn)
    return ", ".join(acc)

  def traverse_key_value_sequence(
      self, a, path: Path = "", paths: list[tuple[Path, Any]] | None = None,
      keys_are_identifiers=True) -> str:
    acc = []
    for k, v in sorted(a.items()):
      if (vn := self.traverse_value(v, path + f'[{repr(str(k))}]', paths)) is not None:
        kn = f"{k}=" if keys_are_identifiers else f"'{k}': "
        acc.append(f"{kn}{vn}")
    return ", ".join(acc)

  def traverse_value_atom(
      self, v: Any, path: Path = "", paths: list[tuple[Path, Any]] | None = None) -> str:
    """The return value can be None if we use an ReduceEmitStrategy."""
    def v_is_leaf(vn: str | Callable[[], str]) -> str:
      if paths is not None:
        paths.append((path, v))
        return ""
      return vn() if callable(vn) else vn

    # Emit first the data types for which we have special rules, because they
    # may also be pytrees and we don't want to loose their type.
    if v is None:
      return v_is_leaf(repr(v))

    from jax._src import literals  # type: ignore
    if isinstance(v, (int, float, bool, str, complex)):
      if not isinstance(v, (enum.IntEnum, literals.TypedFloat,
                            literals.TypedComplex, literals.TypedInt)):
        if isinstance(v, enum.Enum) and isinstance(v, str):
          v = v.value  # Otherwise repr(v) may be "<MyType>: 'x'>"
        return v_is_leaf(repr(v))

    # Look up the exact value in the definitions. This is useful, e.g., in
    # eager mode for a jax.Array to use the name of the computed value,
    # instead of just the value. It is also useful to reproduce constant
    # sharing behavior.
    if (vn := self.lookup_value(v)) is not None:
      return v_is_leaf(vn)

    if (do_emit := get_emitter(v)) is not None:
      # If 'v' has an emitter, it is a leaf. Don't call the emitter if
      # we are only collecting paths to leaves
      return v_is_leaf(lambda: do_emit(self, v))

    v_type_str = str(type(v))
    if "flax.core.axes_scan._Broadcast" in v_type_str:  # TODO: flax
      return v_is_leaf("flax.core.axes_scan.broadcast")

    if isinstance(v, Func):
      if v.api_name:
        return v_is_leaf(v.api_name)
      if v.is_user:
        # USER functions, emit them to collect the externals
        # But we must be careful to restore the current_traverse_value_context
        def emit_func():
          prev_ctx = self.global_ctx.current_traverse_value_context
          try:
            if v.duplicate_of is None:
              self.emit_operand_user_func(v)
            return v.python_name()
          finally:
            self.global_ctx.current_traverse_value_context = prev_ctx

        return v_is_leaf(emit_func)

      return v_is_leaf(lambda: self.use_value(v))

    return v_is_leaf(lambda: self.use_value(v))

  def emit_operand_user_func(self, f: Func) -> None:
    assert f.duplicate_of is None, f
    if not f.function_def:
      # A function that was never invoked is emitted in the main function
      if not self.parent_ctx:
        self.emit_line(f"def {self.global_ctx.var_name_for_val(f, prefix='fun')}(*args, **kwargs):")
        self.emit_line("  pass  # Never called")
        self.emit_newline()
      else:
        self.emitted_function.immediate_externals[id(f)] = f
        self.emitted_function.all_externals[id(f)] = f
      return

    f_emitted = self.global_ctx.emitted_functions.get(f.function_def.id)
    if f_emitted is None:
      emit_ctx = EmitFunctionDefContext(str(f.function_def), self.global_ctx,
                                        self)
      f_emitted = emit_ctx.emitted_function
      self.global_ctx.emitted_functions[f.function_def.id] = f_emitted
      emit_ctx.emit_function_def(f.function_def)

    # We emit f here if we are in the main function, or if
    # any of f's externals are defined here. Otherwise, we will emit these
    # externals in some enclosing function.
    if (not self.parent_ctx or
        any(ae in self.definitions for ae in f_emitted.all_externals)):
      # Emit (or mark as externals) the external functions
      for id_e, e in f_emitted.immediate_externals.items():
        if isinstance(e, Func) and e.is_user:
          assert e.duplicate_of is None
          self.emit_operand_user_func(e)
        elif id_e in self.definitions:  # Defined here
          self.emit_line(f"{self.global_ctx.var_name_for_val(e)} = {self.definitions[id_e]}")
        else:  # External for us too
          self.emitted_function.immediate_externals[id_e] = e
          self.emitted_function.all_externals[id_e] = e
      if (prev_dumped := self.global_ctx.dumped_functions.get(f.python_name())) is not None:
        tracker._thread_local_state.warn_or_error(
            f"Duplicate function definition {f.python_name} ({f.fun_info}). "
            f"Previously emitted for {prev_dumped}.",
            warning_only=True, during_tracking=False
        )
      else:
        self.global_ctx.dumped_functions[f.python_name()] = f.fun_info
      self.emit_newline()
      for bl in f_emitted.lines:
        self.emit_line(bl)
      return

    # We make this function all external
    self.emitted_function.immediate_externals[id(f)] = f
    self.emitted_function.all_externals[id(f)] = f
    for id_ae, ae in f_emitted.all_externals.items():
      self.emitted_function.all_externals[id_ae] = ae

  def emit_function_def(self, fun_def: FunctionDef) -> None:
    """Emits the body of the function.
    This is called starting with the main function.
    """
    self.entry_state_context = self.current_state_context = fun_def.state_context

    self.global_ctx.set_current_traverse_value_context(fun_def, True)
    self.current_traceback = fun_def.traceback
    assert not self.emitted_function.lines
    comment = f" from invocation {fun_def}"
    self.emit_line(f"# body{comment} for {fun_def.func.fun_info}", indent=False)

    arg_names = [self.new_local_name() for _ in fun_def.args]  # type: ignore
    args_str = ", ".join(arg_names)
    if fun_def.kwargs:
      if fun_def.args:
        args_str += ", "
      args_str += "*, " + ", ".join(fun_def.kwargs.keys())

    func_name = self.global_ctx.var_name_for_val(fun_def.func)
    self.emit_line(f"def {func_name}({args_str}):", indent=False)

    paths: list[tuple[Path, Any]] = []
    safe_map(lambda an, a: self.traverse_value(a, an, paths), arg_names, fun_def.args)  # type: ignore
    safe_map(lambda key_a: self.traverse_value(key_a[1], key_a[0], paths), fun_def.kwargs.items())
    for pth, a in paths:
      self.define_value(a, pth)

    last_result_str = "_"
    for s in fun_def.body:
      if self.global_ctx.emit_reduction_strategy is not None:
        if not self.global_ctx.emit_reduction_strategy.keep_call(s):
          last_result_str = "_"
          continue
      last_result_str = self.emit_statement(s)

    self.global_ctx.set_current_traverse_value_context(fun_def, False)
    self.current_traceback = fun_def.traceback
    result_str = self.traverse_value(fun_def.result)

    if fun_def.uncaught_exception and last_result_str != "_":
      # Even for uncaught exception we return the last_result_str, otherwise
      # if we just return None the repro can be DCEed and then it ends empty.
      actual_return = last_result_str
    else:
      actual_return = result_str if fun_def.level > 0 else last_result_str

    comment = ""
    if fun_def.uncaught_exception is not None:
      comment = f" # uncaught exception {type(fun_def.uncaught_exception).__name__}: {fun_def.uncaught_exception}"
    self.emit_line(f"return {actual_return}{comment}")

  def emit_statement(self, stmt: Statement) -> str:
    if self.global_ctx.emit_reduction_strategy:
      func, args, kwargs, result = \
          self.global_ctx.emit_reduction_strategy.rewrite_statement(stmt)
    else:
      func, args, kwargs, result = \
          stmt.func, stmt.args, stmt.kwargs, stmt.result  # type: ignore
    self.global_ctx.set_current_traverse_value_context(stmt, True)

    callee_str = self.traverse_value(func)  # type: ignore
    args_str = self.traverse_sequence_value(args)  # pyrefly: ignore[bad-argument-type]
    if kwargs:
      args_str += (", " if args else "") + self.traverse_key_value_sequence(
        kwargs, keys_are_identifiers=True)

    self.global_ctx.set_current_traverse_value_context(stmt, False)
    result_str = self.new_local_name()
    result_paths: list[tuple[Path, Any]] = []
    self.traverse_value(result, "", result_paths)

    self.emit_context_for_statement(stmt)
    comment = f"  # {stmt}"
    self.emit_line(f"{self.current_context_indent}{result_str} = "
                   f"{callee_str}({args_str}){comment}")
    for pth, r in result_paths:
      self.define_value(r, result_str + pth)
    return result_str

  def emit_context_for_statement(self, stmt: Statement):
    current = self.current_state_context  # None on entry to top-level func
    if stmt.state_context == current:
      return
    if current is not None:
      if stmt.state_context == self.entry_state_context:
        self.current_context_indent = ""  # exit the context manager
        return
      # Some context states are different than current and than entry.
      # Keep it simple and pop the context manager
      current = self.entry_state_context
      self.current_context_indent = ""

    trace_context_keys = [s.name
                          for s in config.config_ext._trace_context_elements]
    # TODO: It seems that the axis_env is in the trace_context, at the end,
    # but not in names?
    assert len(stmt.state_context.trace_context) == 1 + len(trace_context_keys), (
      len(stmt.state_context.trace_context),
      len(trace_context_keys), trace_context_keys)
    state_keys = ("default_lowering_platform", *trace_context_keys)
    stmt_states = (stmt.state_context.default_lowering_platform,
                   *stmt.state_context.trace_context[:-1])

    if current is not None:
      current_states = (current.default_lowering_platform,
                        *current.trace_context[:-1])
    else:
      current_states = (None,) * len(state_keys)

    changed_states = {}
    from jax._src import xla_bridge  # type: ignore
    for name, curr_state, stmt_state in zip(state_keys, current_states, stmt_states):
      if current is None and stmt_state is None:
        if name == "abstract_mesh_context_manager":
          continue
        elif name == "default_lowering_platform":
          stmt_state = xla_bridge.local_devices()[0].platform

      if current is None or stmt_state != curr_state:
        changed_states[name] = stmt_state

    changed_kwargs_str = self.traverse_key_value_sequence(
        changed_states, keys_are_identifiers=True)
    self.emit_line(
        f"{self.current_context_indent}"
        f"with state_context({changed_kwargs_str}):")
    self.current_context_indent += "  "
    self.current_state_context = stmt.state_context

  def abstract_mesh_for_top_statement(self, stmt: Statement):
    import jax  # type: ignore
    from jax import sharding  # type: ignore

    # We emit an abstract mesh even if we don't have one in the context,
    # to make it easy to repro on a machine without the current accelerator
    # type.
    am = None
    for a in tree_util.tree_leaves((stmt.args, stmt.kwargs)):
      if ((shard := getattr(a, "sharding", None)) is not None and
            isinstance(shard, sharding.NamedSharding)):
        shard_am = shard.mesh.abstract_mesh
        if am is None:
          am = shard_am
        elif am != shard_am:
          raise NotImplementedError(
            f"Found multiple abstract meshes: {am} and {shard_am}")
    if am is None:  # Make a mesh
      # TODO: here we make a empty mesh, but the function may have
      # meshes inside!!
      dev0 = jax.local_devices()[0]
      ad = sharding.AbstractDevice(device_kind=dev0.device_kind,
                                   num_cores=getattr(dev0, "num_cores", None),
                                   platform=dev0.platform)
      am = sharding.AbstractMesh((), (), abstract_device=ad)
    return am


class collector:
  """Repro collector for a nullary function.

  This is a callable that will track the underlying function and collect
  data for emitting reproducers. A collector can be called only once, but
  the source can be emitted multiple times (e.g., with different emitting
  strategies.)

  For usage see: https://docs.jax.dev/en/latest/debugging/repro.html
  """
  def __init__(self,
               func: Callable[[], Any],
               static_argnums=(),
               static_argnames=()):
    self._func = func
    self._static_argnums = static_argnums
    self._static_argnames = static_argnames
    self._collection_point = tracker._CollectionPoint(
        stmt=None,
        stmt_level=len(tracker._thread_local_state.call_stack),
    )

  def __call__(self, *args, **kwargs):
    if self._collection_point.stmt is not None:
      raise ValueError("This repro collector was already invoked once.")
    if not traceback_util.repro_is_enabled:  # type: ignore
      raise ValueError(
        "You must set JAX_REPRO_DIR=something, when using repro.collector")
    self._collection_point.stmt_level = len(tracker._thread_local_state.call_stack)
    # The collection_point will be filled-in by the tracker at the end
    # of the call.
    tracker._thread_local_state.collection_points.append(self._collection_point)
    if self._collection_point.stmt_level == 0:
      tracker._thread_local_state.reset_counters()
    tracker._thread_local_state.source_info_mapping = \
        getattr(self._func, "_source_info_mapping", {})
    with tracker.flags_override(
        save_repro_on_uncaught_exception=False):
      from jax._src.repro import repro_api  # type: ignore
      # Wrap it with a JAX API call, because the top-level function call
      # must be a JAX API. We'll drop this during emitting.
      return repro_api.jax_repro_collect(
          self._func, *args, **kwargs,
          collect_static_argnums=self._static_argnums,
          collect_static_argnames=self._static_argnames)

  @property
  def deferred_error(self) -> Exception | None:
    """A deferred error, when error_mode="defer"."""
    return self._collection_point.deferred_error

  def to_source(self, *,
                extra_comment: str = "",
                strategy: EmitReductionStrategy | None = None) -> str:
    """Generates the repro source from a collector
    Args:
      extra_comment: text to be added in comments at the top of the
        repro source. Can contain multiple lines.
      strategy: apply reductions during emit (for use with the reducer).

    Returns:
      the source of the repro.
    """
    if not self._collection_point.stmt:
      raise ValueError("Invoking the `collector.to_source` without a "
                       "successful invocation of the collector. Eiher you "
                       "forgot to invoke the collector, or there was an "
                       "internal error.")
    src = to_source(self._collection_point.stmt, extra_comment=extra_comment,
                    strategy=strategy)
    if self._collection_point.deferred_error is None:
      self._collection_point.deferred_error = tracker._thread_local_state.deferred_error
    return src


def to_source(stmt: Statement, *,
              extra_comment: str = "",
              strategy: EmitReductionStrategy | None = None
              ) -> str:
  import jax  # type: ignore
  from jax._src import xla_bridge  # type: ignore
  """Generates the repro source from a collector or on uncaught exception.

  Args:
    stmt: the top-level statement
    extra_comment: text to be added in comments at the top of the
      repro source. Can contain multiple lines.
    strategy: apply reductions during emit (for use with the reducer).

  `stmt` is either:
     * jax_repro_collect(fn, *args) or
     * jax_<api>(*args) otherwise (implicit mode, uncaught exception)

  To simplify the rest of the source generation,
  if we are not using a collector already
  we add an extra call to jax_repro_collect.
  We rewrite `jax.<api>(*args)` as:

    def array_repro(*array_args):
      return api(*args)  # Use all args here including Func
    jax_repro_collect(array_repro, *array_args)

  Then, for a call `jax_repro_collect(fn, *array_args)` we emit

    def main_repro_func(*array_args):
      ... definitions for fn and functions it references ...
      return fn(*array_args)

    def main_repro_metadata():
      ... definitions for avals, shardings ...
      return dict(
        func: main_repro_func,
        top_level_inputs: array_args,
        avals: aval(array_args),
        shardings: shardings(array_args),
        uncaught_exception_type_str: ...,
        uncaught_exception_str: ...,
      }
    main_repro = main_repro_metadata()

  Returns:
    the source of the repro.
  """
  preamble = _source_preamble(stmt, extra_comment=extra_comment)
  preamble += """
def main_repro_func(*main_repro_args, **main_repro_kwargs):
"""
  if tracker.func_api_name(stmt.func) != "jax_repro_collect":
    stmt = _wrap_with_repro_collector(stmt)

  global_ctx = EmitGlobalContext(strategy=strategy)

  func_ctx = EmitFunctionDefContext("main_repro_func", global_ctx, None)
  func_ctx.entry_state_context = stmt.state_context
  func_ctx.current_state_context = None  # Force-emit the entire context
  callee_str = func_ctx.traverse_value(stmt.args[0])  # pyrefly: ignore [unsupported-operation]
  func_ctx.emit_newline()
  func_ctx.emit_context_for_statement(stmt)
  func_ctx.emit_line(f"{func_ctx.current_context_indent}return "
                     f"{callee_str}(*main_repro_args, **main_repro_kwargs)")

  # result_str = func_ctx.emit_statement(stmt)
  # TODO: really wasteful to materialize the whole repro_source
  repro_source = (preamble +
                  "\n".join(func_ctx.emitted_function.lines) + "\n\n")

  # Emit a function to compute the metadata
  metadata_ctx = EmitFunctionDefContext("main_repro_metadata",
                                        global_ctx, None)
  # We are now emitting code for a jax_repro_collect. All arguments are arrays,
  # except the first one which is the function being collected.abs
  # That function was emitted by emit_statement above
  args = stmt.args[1:]  # type: ignore
  kwargs = stmt.kwargs
  if stmt.level == 0:
    metadata_top_level_inputs_str = metadata_ctx.traverse_value((args, kwargs))
  else:
    metadata_top_level_inputs_str = "None"
  metadata_ctx.emit_newline()
  metadata_ctx.emit_line(
      f"main_repro_top_level_inputs = {metadata_top_level_inputs_str}")

  flat_inputs, inputs_tree = tree_util.tree_flatten((args, kwargs))
  def abstractify_val(v):
    try:
      return jax.typeof(v)
    except TypeError:
      return None
  aval_tree = inputs_tree.unflatten(abstractify_val(v) for v in flat_inputs)
  metadata_avals_str = metadata_ctx.traverse_value(aval_tree)
  metadata_ctx.emit_line(f"main_repro_avals = {metadata_avals_str}")

  sharding_tree = inputs_tree.unflatten(getattr(v, "sharding", None)
                                        for v in flat_inputs)
  metadata_shardings_str = metadata_ctx.traverse_value(sharding_tree)
  metadata_ctx.emit_line(f"main_repro_shardings = {metadata_shardings_str}")

  repro_source += ("def main_repro_metadata():\n" +
                  "\n".join(metadata_ctx.emitted_function.lines) + "\n")

  uncaught_exception_type_str = "None"
  uncaught_exception_str = "None"
  if stmt.uncaught_exception:
    uncaught_exception_type_str = f"\"{type(stmt.uncaught_exception).__name__}\""
    uncaught_exception_str = f"\"{stmt.uncaught_exception}\""

  default_device = jax.local_devices()[0]


  repro_source += f"""
  return dict(
    func=main_repro_func,
    top_level_inputs=main_repro_top_level_inputs,
    avals=main_repro_avals,
    shardings=main_repro_shardings,
    nr_devices={xla_bridge.local_device_count()},
    default_device_platform="{default_device.platform}",
    default_device_kind="{default_device.device_kind}",
    default_device_raw_platform="{getattr(default_device, "_raw_platform", default_device.platform)}",
    uncaught_exception_type_str={uncaught_exception_type_str},
    uncaught_exception_str={uncaught_exception_str},
  )

main_repro = main_repro_metadata()


if __name__ == "__main__":
  main_repro_args, main_repro_kwargs = main_repro["top_level_inputs"]
  main_repro["func"](*main_repro_args, **main_repro_kwargs)
"""

  return repro_source


def _source_preamble(stmt: Statement, *, extra_comment: str = "") -> str:
  from jax._src import xla_bridge  # type: ignore
  np.set_printoptions(threshold=sys.maxsize)  # Do not summarize arrays

  if stmt.uncaught_exception_traceback:
    if extra_comment:
      extra_comment += "\n"
    extra_comment += stmt.uncaught_exception_traceback
  comment = "\n".join([f"# {l}" for l in extra_comment.split("\n")]) + "\n\n"

  preamble = """
# This file was generated by JAX repro extractor.

import jax
from jax._src import config
"""
  if tracker._thread_local_state.flags.inline_runtime:
    this_src_dir = os.path.dirname(__file__)
    preamble += """

######## Start inlined repro_runtime.py:
"""
    with open(os.path.join(this_src_dir, "repro_runtime.py")) as f:
      preamble += f.read()
    preamble += """
######## End inlined repro_runtime.py

######## Start inlined repro_api.py:
"""
    with open(os.path.join(this_src_dir, "repro_api.py")) as f:
      preamble += f.read()
    preamble += """
######## End inlined repro_api.py

"""
  else:
    preamble += """
from jax._src.repro.repro_runtime import *  # type: ignore  # noqa: F401,F403
from jax._src.repro.repro_api import *  # type: ignore  # noqa: F401,F403

"""

  preamble += f"""
# Use the same number of devices as in the repro collection context
request_cpu_devices({xla_bridge.local_device_count()})


# TODO: for now there are some bespoke pieces that handle Flax, so some
# reproducers may include Flax.

try:
  import flax  # type: ignore
except ImportError:
  flax = None

{comment}
"""
  return preamble


def _wrap_with_repro_collector(stmt: Statement):
  """See comments in `to_source`."""
  import jax  # type: ignore
  from jax._src.repro import repro_api  # type: ignore
  flat_args, inputs_tree = tree_util.tree_flatten((stmt.args, stmt.kwargs))
  def get_arrays(v):
    try:
      jax.typeof(v)
      return v
    except TypeError:
      return None
  flat_array_args = map(get_arrays, flat_args)
  array_args, array_kwargs = inputs_tree.unflatten(flat_array_args)

  assert isinstance(repro_api.jax_repro_collect, Func)
  new_collect = tracker.Statement(None, repro_api.jax_repro_collect)
  def array_repro(*_, **__): return None
  array_repro_fun = tracker.boundary(array_repro, is_user=True)
  assert isinstance(array_repro_fun, Func)

  new_collect.set_args((array_repro_fun, *array_args), array_kwargs)
  new_collect.result = stmt.result
  new_collect.state_context = stmt.state_context
  new_collect.uncaught_exception = stmt.uncaught_exception

  array_repro_fundef = tracker.FunctionDef(new_collect, array_repro_fun)
  array_repro_fundef.set_args(array_args, array_kwargs)
  array_repro_fundef.body = [stmt]  # with all the args
  array_repro_fundef.result = stmt.result
  array_repro_fundef.state_context = stmt.state_context
  array_repro_fundef.uncaught_exception = stmt.uncaught_exception
  return new_collect


_last_saved_idx_for_repro_name_prefix: dict[str, int] = {}


def save(repro_source: str,
         repro_name_prefix: str = "jax_repro") -> pathlib.Path:
  """Saves the `repro_source` in a file."""
  assert config.repro_dir.value
  dump_to = config.repro_dir.value
  out_dir: pathlib.Path = path.make_jax_dump_dir(dump_to)  # type: ignore
  last_id = _last_saved_idx_for_repro_name_prefix.get(repro_name_prefix, 0)
  fresh_id = itertools.count(last_id + 1)
  while True:
    next_id = next(fresh_id)
    repro_path = out_dir / f"{repro_name_prefix}_{next_id}.py"
    if not os.path.exists(repro_path):  # This can race
      _last_saved_idx_for_repro_name_prefix[repro_name_prefix] = next_id
      break
  logging.warning(f"Saved JAX repro at {repro_path}")
  repro_path.write_text(repro_source)
  tracker._thread_local_state.last_saved_repro = (repro_path, repro_source)
  return repro_path


def collect_and_save(fn: Callable, *,
                     repro_name_prefix: str | None = None) -> Callable:
  """A function wrapper that adds reproducer saving.

  Usage:
    @partial(repro.collect_and_save, repro_name_prefix="my_repro")
    def my_fun(*args, **kwargs):
      ...

    my_fun(...)  # will save a repro to ${JAX_REPRO_DIR}/{repro_name_prefix}.py

  """
  if repro_name_prefix is None:
    repro_name_prefix = "jax_repro"
  else:
    repro_name_prefix = re.sub(r"[^\w\s-]", "_", repro_name_prefix.lower())
    repro_name_prefix = re.sub(r"_+", "_", repro_name_prefix).strip("_")
    repro_name_prefix = repro_name_prefix[:256]

  @functools.wraps(fn)
  def _fn_with_repro_collection(*args, **kwargs):
    col = collector(lambda: fn(*args, **kwargs))
    try:
      return col()
    finally:
      source = col.to_source()
      save(source, repro_name_prefix)
      if col.deferred_error is not None:
        raise col.deferred_error

  return _fn_with_repro_collection


# TODO: fix this for dedup
_loc_re = re.compile(r"# body from invocation .* USER\[(?P<name>.*?)\] for (?P<func_info>.*)")


@dataclasses.dataclass
class LoadedRepro:
  func: Callable
  # If this is a top-level function, then the inputs (args, kwargs), otherwise
  # None
  top_level_inputs: tuple[tuple[Any, ...], dict[str, Any]] | None
  # The abstract values of the inputs to the function (args, kwargs), even for
  # non-top-level functions.
  avals: tuple[tuple[Any, ...], dict[str, Any]]
  # The shardings of the inputs to the function (args, kwargs)
  shardings: tuple[tuple[Any, ...], dict[str, Any]]
  # Mapping of source code locations to function info
  source_info_mapping: dict[str, str] | None

  uncaught_exception_type_str: str | None  # e.g., "ValueError"
  uncaught_exception_str: str | None  # e.g., "got incompatible shapes"

  nr_devices: int | None
  default_device_platform: str | None
  default_device_kind: str | None
  default_device_raw_platform: str | None

  def __post_init__(self):
    # Store the source info mapping, in case we want to track repros for
    # the repros themselves. This propagates source info to reduced repros, e.g.
    setattr(self.func, "_source_info_mapping", self.source_info_mapping)
    setattr(self.run.__func__, "_source_info_mapping", self.source_info_mapping)

  def run(self):
    """Run the function.

    Run the function with concrete arguments, which are the actual arguments
    saved in the LoadedRepro for top-level functions, or synthetic arguments
    for lower-level functions. The arguments are sharded appropriately.

    The repros have a default_lowering_platform, so in some cases we
    can reproduce the uncaught exception even on a platform that does not
    have the exact same set of accelerators.
    """
    import jax  # type: ignore
    from jax._src import core  # type: ignore
    def make_arg(v, shard):
      if isinstance(v, core.ShapedArray):
        v = np.ones(v.shape, dtype=v.dtype)
      if shard is None:
        return v
      return jax.device_put(v, shard)
    if self.top_level_inputs is not None:
      args, kwargs = tree_util.tree_map(make_arg, self.top_level_inputs,
                                        self.shardings)
    else:
      args, kwargs = tree_util.tree_map(make_arg, self.avals, self.shardings)

    return self.func(*args, **kwargs)


def load(repro_source: str, repro_path: pathlib.Path | str) -> LoadedRepro:
  """Loads a repro created by `collector.to_source`.

  Returns a function that was saved in the repro source.
  """
  if isinstance(repro_path, pathlib.Path):
    repro_path, repro_path_str = repro_path, str(repro_path)
  else:
    repro_path, repro_path_str = pathlib.Path(repro_path), repro_path
  source_info_mapping = {}
  for i, line in enumerate(repro_source.splitlines()):
    if m := _loc_re.search(line):
      source_info_mapping[f"{m.group('name')} at {repro_path_str}:{i + 2}"] = m.group("func_info")

  compiled = compile(repro_source, repro_path_str, "exec")
  custom_namespace = {}
  custom_namespace['__builtins__'] = __builtins__
  exec(compiled, custom_namespace, custom_namespace)
  main_repro = custom_namespace["main_repro"]  # type: ignore
  return LoadedRepro(
    func=main_repro["func"],
    top_level_inputs=main_repro.get("top_level_inputs", ((), {})),
    avals=main_repro.get("avals", ((), {})),
    shardings=main_repro.get("shardings", ((), {})),
    source_info_mapping=source_info_mapping,
    uncaught_exception_type_str=main_repro.get("uncaught_exception_type_str", None),
    uncaught_exception_str=main_repro.get("uncaught_exception_str", None),
    nr_devices=main_repro.get("nr_devices", None),
    default_device_platform=main_repro.get("default_device_platform", None),
    default_device_kind=main_repro.get("default_device_kind", None),
    default_device_raw_platform=main_repro.get("default_device_raw_platform", None),
  )
