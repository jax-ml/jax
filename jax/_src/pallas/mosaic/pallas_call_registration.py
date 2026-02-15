# Copyright 2023 The JAX Authors.
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

"""Contains registrations for pallas_call on TPU."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
from typing import cast

import jax
from jax import dtypes
from jax._src import core as jax_core
from jax._src import frozen_dict
from jax._src import sharding_impls
from jax._src import tpu_custom_call
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir import passmanager
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import lowering
from jax._src.pallas.mosaic import sc_lowering
from jax._src.state import types as state_types
from jax.experimental import mosaic
from jax.experimental.mosaic.dialects import tpu


def _maybe_cast_to_int(x: jax.Array | jax_core.AbstractValue):
  """Casts boolean values to integers.

  We perform this cast because Mosaic does not directly support bool values
  for Memrefs. Instead, we load bools as integers and cast them to bools
  after loading from a memref inside of the kernel.
  """
  assert isinstance(
      x, (jax.Array, jax_core.ShapedArray, state_types.AbstractLinVal)
  ) or (
      isinstance(x, jax_core.Tracer)
      and isinstance(x.aval, state_types.AbstractLinVal)
  ), type(x)
  if isinstance(x, jax.Array):
    if dtypes.issubdtype(x.dtype, jax.numpy.bool_):
      return x.astype(lowering.BOOL_MEMREF_TYPE)
    return x
  else:
    if dtypes.issubdtype(x.dtype, jax.numpy.bool_):
      if isinstance(x, state_types.AbstractLinVal):
        raise NotImplementedError  # TODO(mattjj,sharadmv)
      return jax_core.ShapedArray(x.shape, lowering.BOOL_MEMREF_TYPE)
    return x


def _get_memory_space_from_aval(
    out_aval: jax_core.AbstractValue, kernel_type: tpu_core.KernelType
) -> tpu_custom_call.MemorySpace | None:
  if not isinstance(out_aval, jax_core.ShapedArray):
    raise ValueError("Memory spaces not defined for non-ShapedArrays")
  if not isinstance(out_aval, pallas_core.ShapedArrayWithMemorySpace):
    # If we are passed a regular old ShapedArray, we don't constrain the
    # memory space
    return None
  # If we are passed an aval with an explicit memory space tag, we use it
  # to constrain the memory space.
  match out_aval.memory_space:
    case tpu_core.MemorySpace.HBM:
      return tpu_custom_call.MemorySpace.HBM
    case tpu_core.MemorySpace.VMEM:
      return tpu_custom_call.MemorySpace.VMEM
    case tpu_core.MemorySpace.SMEM:
      return tpu_custom_call.MemorySpace.SMEM
    case tpu_core.MemorySpace.SEMAPHORE:
      match kernel_type:
        case tpu_core.KernelType.SC_SCALAR_SUBCORE:
          return tpu_custom_call.MemorySpace.SC_SCALAR_SEMAPHORE_MEM
        case tpu_core.KernelType.TC:
          return tpu_custom_call.MemorySpace.SEMAPHORE_MEM
        case _:
          raise ValueError(f"Invalid kernel type for semaphore: {kernel_type}")
    case tpu_core.MemorySpace.HOST:
      return tpu_custom_call.MemorySpace.HOST
  return None


def _get_memory_spaces_from_avals(
    avals: Sequence[jax_core.AbstractValue], kernel_type: tpu_core.KernelType
) -> tuple[tpu_custom_call.MemorySpace | None, ...] | None:
  memory_spaces = None
  if any(
      isinstance(aval, pallas_core.ShapedArrayWithMemorySpace) for aval in avals
  ):
    memory_spaces = tuple(
        _get_memory_space_from_aval(aval, kernel_type=kernel_type)
        for aval in avals
    )
  return memory_spaces


def _resolve_memory_spaces(
    in_avals: Sequence[jax_core.AbstractValue],
    out_avals: Sequence[jax_core.AbstractValue],
    *,
    input_output_aliases: tuple[tuple[int, int], ...],
    kernel_type: tpu_core.KernelType,
) -> tuple[
    tuple[tpu_custom_call.MemorySpace | None, ...] | None,
    tuple[tpu_custom_call.MemorySpace | None, ...] | None,
]:
  output_memory_spaces = _get_memory_spaces_from_avals(
      out_avals, kernel_type=kernel_type
  )
  input_memory_spaces = None
  if any(
      isinstance(aval, pallas_core.ShapedArrayWithMemorySpace)
      for aval in in_avals
  ):
    input_memory_spaces = _get_memory_spaces_from_avals(
        in_avals, kernel_type=kernel_type
    )
    # ShapedArrayWithMemorySpace wasn't allowed to escape the Pallas kernel, so
    # it was stripped from the original out_avals. We need to resolve this for
    # outputs aliased to inputs with pltpu.with_memory_space_constraint.
    if input_memory_spaces is not None:
      output_memory_spaces_list: list[tpu_custom_call.MemorySpace | None]
      if output_memory_spaces is None:
        output_memory_spaces_list = [None] * len(out_avals)
      else:
        output_memory_spaces_list = list(output_memory_spaces)
      modified = False
      for in_idx, out_idx in input_output_aliases:
        if (ms := input_memory_spaces[in_idx]) is not None:
          output_memory_spaces_list[out_idx] = ms
          modified = True
      if modified:
        output_memory_spaces = tuple(output_memory_spaces_list)
  if input_memory_spaces is None and output_memory_spaces is not None:
    input_memory_spaces_list: list[tpu_custom_call.MemorySpace | None] = [
        None,
    ] * len(in_avals)
    for input_output_alias in input_output_aliases:
      input_memory_spaces_list[input_output_alias[0]] = output_memory_spaces[
          input_output_alias[1]
      ]
    input_memory_spaces = tuple(input_memory_spaces_list)
  if input_memory_spaces is not None:
    input_memory_spaces = tuple(
        i
        if i
        in {  # pylint: disable=g-long-ternary
            tpu_custom_call.MemorySpace.HBM,
            tpu_custom_call.MemorySpace.VMEM,
            tpu_custom_call.MemorySpace.SMEM,
        }
        else None
        for i in input_memory_spaces
    )
  return input_memory_spaces, output_memory_spaces


def _resolve_side_effect_type(
    has_side_effects: bool | tpu_core.SideEffectType,
) -> bool | tpu_custom_call.TpuSideEffectType:
  match has_side_effects:
    case bool():
      return has_side_effects
    case tpu_core.SideEffectType.PURE:
      return tpu_custom_call.TpuSideEffectType.PURE
    case tpu_core.SideEffectType.DATAFLOW_SIDE_EFFECTING:
      return tpu_custom_call.TpuSideEffectType.DATAFLOW_SIDE_EFFECTING
    case tpu_core.SideEffectType.SIDE_EFFECTING:
      return tpu_custom_call.TpuSideEffectType.SIDE_EFFECTING
    case _:
      raise ValueError(f"Invalid side effect type: {has_side_effects}")


def _resolve_tiling(
    mosaic_params: tpu_core.CompilerParams,
) -> tpu_custom_call.Tiling | None:
  if mosaic_params.use_tc_tiling_on_sc is None:
    return None
  if mosaic_params.kernel_type not in (
      tpu_core.KernelType.SC_SCALAR_SUBCORE,
      tpu_core.KernelType.SC_VECTOR_SUBCORE,
  ):
    raise ValueError(
        "use_tc_tiling_on_sc= is only supported for SC_*_SUBCORE kernels"
    )

  return (
      tpu_custom_call.Tiling.COMPACT
      if mosaic_params.use_tc_tiling_on_sc
      else tpu_custom_call.Tiling.SPARSE_CORE
  )


def _lower_to_custom_call(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    mosaic_module: ir.Module,
    mosaic_params: tpu_core.CompilerParams,
    num_dynamic_grid_bounds: int,
    input_output_aliases: tuple[tuple[int, int], ...],
    cost_estimate: pallas_core.CostEstimate | None,
    out_avals: tuple[jax_core.AbstractValue, ...],
    effects: jax_core.Effects,
    metadata: frozen_dict.FrozenDict[str, str] | None,
    name: str,
    jax_mesh,
):
  kernel_type = mosaic_params.kernel_type
  input_output_aliases = tuple(
      (a[0] + num_dynamic_grid_bounds, a[1]) for a in input_output_aliases
  )

  # Replace in_avals to physical avals.
  # This step is required for mapping logical types to physical types.
  # (e.g. PRNG key -> uint32[2])
  physical_avals = [jax_core.physical_aval(aval) for aval in ctx.avals_in]
  ctx = ctx.replace(avals_in=physical_avals)

  # Booleans are loaded into the kernel as integers.
  def _maybe_cast_inputs(*args):
    args = [_maybe_cast_to_int(x) for x in args]
    return args

  kernel_in_avals = [_maybe_cast_to_int(x) for x in ctx.avals_in]
  kernel_out_avals = [_maybe_cast_to_int(x) for x in ctx.avals_out]
  cast_ctx = ctx.replace(avals_out=kernel_in_avals)
  in_nodes = mlir.lower_fun(_maybe_cast_inputs)(cast_ctx, *in_nodes)

  # Dynamic grid bounds have to go at the front.
  dynamic_grid_args, args = (
      in_nodes[:num_dynamic_grid_bounds],
      in_nodes[num_dynamic_grid_bounds:],
  )
  kernel_ctx = ctx.replace(avals_in=kernel_in_avals, avals_out=kernel_out_avals)
  input_memory_spaces, output_memory_spaces = _resolve_memory_spaces(
      ctx.avals_in,
      out_avals,
      input_output_aliases=input_output_aliases,
      kernel_type=kernel_type,
  )

  if cost_estimate is not None:
    mosaic_cost_estimate = cast(
        tpu_custom_call.CostEstimate, dataclasses.asdict(cost_estimate)
    )
  else:
    mosaic_cost_estimate = None

  dict_metadata = dict(metadata) if metadata is not None else {}
  del metadata
  if jax_mesh is not None:
    mesh_axes = {
        e.name
        for e in effects
        if isinstance(e, jax_core.NamedAxisEffect)
        # Filter for only device mesh axis name effects
        and e.name in jax_mesh.axis_names
    }
    # Only put mesh axes in metadata if there are any.
    if mesh_axes:
      if "mesh_axes" in dict_metadata:
        raise ValueError("Metadata already contains mesh axes.")
      mesh_axes_list = list(mesh_axes)
      if all(isinstance(a, str) for a in mesh_axes):
        mesh_axes_list = sorted(mesh_axes)  # type: ignore
      dict_metadata["mesh_axes"] = json.dumps(mesh_axes_list)
  out_nodes = mosaic.lower_module_to_custom_call(
      kernel_ctx,
      *dynamic_grid_args,
      *args,
      module=mosaic_module,
      out_type=kernel_out_avals,
      kernel_name=mlir.sanitize_name(name),
      cost_estimate=mosaic_cost_estimate,
      vmem_limit_bytes=mosaic_params.vmem_limit_bytes,
      flags=mosaic_params.flags,
      allow_input_fusion=mosaic_params.allow_input_fusion,
      input_output_aliases=input_output_aliases,
      serialization_format=mosaic_params.serialization_format,
      internal_scratch_in_bytes=mosaic_params.internal_scratch_in_bytes,
      collective_id=mosaic_params.collective_id,
      has_side_effects=_resolve_side_effect_type(
          mosaic_params.has_side_effects
      ),
      output_memory_spaces=output_memory_spaces,
      disable_bounds_checks=mosaic_params.disable_bounds_checks,
      input_memory_spaces=input_memory_spaces,
      metadata=dict_metadata,
      skip_device_barrier=mosaic_params.skip_device_barrier,
      allow_collective_id_without_custom_barrier=mosaic_params.allow_collective_id_without_custom_barrier,
      shape_invariant_numerics=mosaic_params.shape_invariant_numerics,
      tiling=_resolve_tiling(mosaic_params),
  )
  _maybe_cast_to_bool = (
      lambda x, aval: x.astype(jax.numpy.bool_)
      if aval.dtype == jax.numpy.bool_
      else x
  )

  def _maybe_cast_outputs(*args):
    args = [_maybe_cast_to_bool(x, aval) for x, aval in zip(args, out_avals)]
    return args

  cast_ctx = ctx.replace(avals_in=kernel_out_avals)
  return mlir.lower_fun(_maybe_cast_outputs)(cast_ctx, *out_nodes)


def pallas_call_tpu_lowering_rule(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    jaxpr: jax_core.Jaxpr,
    grid_mapping: pallas_core.GridMapping,
    mesh: pallas_core.Mesh | None,
    input_output_aliases: tuple[tuple[int, int], ...],
    debug: bool,
    interpret: bool,
    compiler_params: pallas_core.CompilerParams | None,
    cost_estimate: pallas_core.CostEstimate | None,
    out_avals: tuple[jax_core.AbstractValue, ...],
    metadata: frozen_dict.FrozenDict[str, str] | None,
    name: str | None,
):
  """Lowers a pallas_call to a Mosaic TPU custom call."""
  del interpret  # Unused.

  debug_info = jaxpr.debug_info
  if debug:
    print(f"\nThe kernel jaxpr for pallas_call {debug_info.func_src_info}:")
    print(jaxpr)

  if compiler_params is None:
    mosaic_params = tpu_core.CompilerParams()
  else:
    assert isinstance(compiler_params, tpu_core.CompilerParams)
    mosaic_params = compiler_params  # type: ignore[assignment]

  del mesh
  jax_mesh = None
  axis_context = ctx.module_context.axis_context
  if axis_context is not None:
    if isinstance(axis_context, sharding_impls.SPMDAxisContext):
      jax_mesh = axis_context.mesh
  mlir_ctx = mlir.JaxIrContext()
  mlir_ctx.append_dialect_registry(mlir.upstream_dialects)
  mlir_ctx.load_all_available_dialects()
  tpu.register_dialect(mlir_ctx)

  match (kernel_type := mosaic_params.kernel_type):
    case tpu_core.KernelType.TC:
      lower_jaxpr_to_module = lowering.lower_jaxpr_to_module
    case (
        tpu_core.KernelType.SC_SCALAR_SUBCORE
        | tpu_core.KernelType.SC_VECTOR_SUBCORE
    ):
      lower_jaxpr_to_module = sc_lowering.lower_jaxpr_to_module
    case _:
      raise ValueError(f"Unsupported kernel type: {mosaic_params.kernel_type}")

  with mlir_ctx, ir.Location.unknown(mlir_ctx):
    mosaic_module = lower_jaxpr_to_module(
        ctx,
        grid_mapping,
        jaxpr,
        dimension_semantics=mosaic_params.dimension_semantics,
        kernel_type=kernel_type,
        mesh=jax_mesh,
        dynamic_shape_replacement_enabled=pallas_core.dynamic_shapes_export_enabled(),
    )

  if debug:
    pm = passmanager.PassManager.parse("builtin.module(canonicalize)", mlir_ctx)
    pm.run(mosaic_module.operation)
    print(f"\nThe Mosaic module for pallas_call {debug_info.func_src_info}:")
    print(mosaic_module)

  return _lower_to_custom_call(
      ctx,
      *in_nodes,
      mosaic_module=mosaic_module,
      mosaic_params=mosaic_params,
      num_dynamic_grid_bounds=grid_mapping.num_dynamic_grid_bounds,
      input_output_aliases=input_output_aliases,
      cost_estimate=cost_estimate,
      out_avals=out_avals,
      effects=jaxpr.effects,
      metadata=metadata,
      name=name or debug_info.func_name,
      jax_mesh=jax_mesh,
  )
