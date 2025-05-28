# Copyright 2024 The JAX Authors.
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

"""Module registering a lowering rule for pallas_call on GPU."""

from __future__ import annotations

import io
from typing import cast
import zlib

import jax
import jax._src.core as jax_core
from jax._src.interpreters import mlir
from jax._src.lib import gpu_triton as triton_kernel_call_lib
from jax._src.lib import triton
from jax._src.lib.mlir import ir
from jax._src.pallas import core as pallas_core
from jax._src.pallas.triton import core as triton_core
from jax._src.pallas.triton import lowering


def normalize_grid(grid: pallas_core.StaticGrid) -> tuple[int, int, int]:
  if isinstance(grid, int):
    grid = (grid,)
  elif len(grid) > 3:
    raise ValueError("`grid` should have three or fewer dimensions.")
  return tuple(grid) + (1,) * (3 - len(grid))  # type: ignore


def avals_to_layouts(avals):
  return [list(reversed(range(aval.ndim))) for aval in avals]  # pytype: disable=attribute-error


def pallas_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    jaxpr: jax_core.Jaxpr,
    interpret: bool,
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: pallas_core.GridMapping,
    mesh: pallas_core.Mesh | None,
    compiler_params: dict[str, pallas_core.CompilerParams],
    cost_estimate: pallas_core.CostEstimate | None,
    out_avals: tuple[jax_core.AbstractValue, ...],
):
  del interpret, out_avals, cost_estimate
  debug_info = jaxpr.debug_info
  if grid_mapping.num_dynamic_grid_bounds:
    raise NotImplementedError(
        "dynamic grid bounds not supported in the Triton backend"
    )
  if grid_mapping.num_index_operands:
    raise NotImplementedError(
        "scalar prefetch not implemented in the Triton backend"
    )
  if mesh is not None:
    raise NotImplementedError("mesh is not supported in the Triton backend")

  [lowering_platform] = ctx.platforms or ctx.module_context.platforms

  if "triton" in compiler_params:
    params = cast(triton_core.CompilerParams, compiler_params["triton"])
  else:
    params = triton_core.CompilerParams()
  num_warps = 4 if params.num_warps is None else params.num_warps
  num_stages = params.num_stages
  if num_stages is None:
    num_stages = 1 if lowering_platform == "rocm" else 3

  if debug:
    print(f"\nThe kernel jaxpr for pallas_call {debug_info.func_src_info}:")
    print(jaxpr)
    print(f"The grid mapping for pallas_call {debug_info.func_src_info}:")
    print(grid_mapping)

  # Sanitize the name to conform to NVPTX requirements. We do this here
  # to avoid the need to fetch the new name from PTX post compilation.
  name = mlir.sanitize_name(debug_info.func_name)
  lowering_result = lowering.lower_jaxpr_to_triton_module(
      jaxpr, grid_mapping, lowering_platform
  )
  module_op = lowering_result.module.operation
  if debug:
    print(f"\nThe Triton module for pallas_call {debug_info.func_src_info}:")
    print(module_op.get_asm(enable_debug_info=True, pretty_debug_info=True))

  grid_x, grid_y, grid_z = normalize_grid(lowering_result.grid)
  buf = io.BytesIO()
  module_op.write_bytecode(buf)

  # TODO(b/394629193): Remove True once the bug is fixed.
  if True:
    # AOT Triton compilation is only available on jaxlib 0.5.1+.
    out_types = [
      ir.RankedTensorType.get(bm.array_shape_dtype.shape,
                              mlir.dtype_to_ir_type(bm.array_shape_dtype.dtype))
      for bm in grid_mapping.block_mappings_output
    ]
    backend_config = dict(
        name=ir.StringAttr.get(name),
        ir=ir.StringAttr.get(buf.getvalue()),
        num_stages=mlir.i32_attr(num_stages),
        num_warps=mlir.i32_attr(num_warps),
        grid_x=mlir.i32_attr(grid_x),
        grid_y=mlir.i32_attr(grid_y),
        grid_z=mlir.i32_attr(grid_z),
        debug=ir.BoolAttr.get(debug),
    )
    if params.serialized_metadata is not None:
      # This field is unstable and may be removed in the future.
      backend_config["serialized_metadata"] = ir.StringAttr.get(
          params.serialized_metadata
      )
    return mlir.custom_call(
        call_target_name="__gpu$xla.gpu.triton",
        result_types=out_types,
        operands=in_nodes,
        backend_config=backend_config,
        api_version=4,
        operand_layouts=avals_to_layouts(ctx.avals_in),
        result_layouts=avals_to_layouts(ctx.avals_out),
        operand_output_aliases=dict(input_output_aliases),
    ).results

  # TODO(slebedev): Make this work for ROCm.
  try:
    gpu_device, *_ = jax.local_devices(backend="gpu")
  except RuntimeError:
    # GPU device is not available. Fall back to the minimum CC supported by Triton.
    # TODO(slebedev): Make the fallback CC configurable.
    arch_name = "8.0"
    cc = 80
  else:
    arch_name = str(gpu_device.compute_capability)
    cc = int(arch_name.replace(".", ""))

  compilation_result = triton.compile(
      lowering_platform,
      buf.getvalue(),
      arch_name,
      num_warps=num_warps,
      num_ctas=1,
      num_stages=num_stages,
  )
  kernel = triton_kernel_call_lib.TritonKernel(
      debug_info.func_name,
      num_warps,
      compilation_result.smem_bytes,
      compilation_result.asm,
      module_op.get_asm(enable_debug_info=True, pretty_debug_info=True),
      cc,
      compilation_result.cluster_dim_x,
      compilation_result.cluster_dim_y,
      compilation_result.cluster_dim_z,
  )
  kernel_call = triton_kernel_call_lib.TritonKernelCall(
      kernel,
      grid_x,
      grid_y,
      grid_z,
      [triton_kernel_call_lib.create_array_parameter(0, 16)]
      * (len(ctx.avals_in) + len(ctx.avals_out)),
  )
  # TODO(b/392558289): Migrate to ``jax.ffi``.
  return mlir.custom_call(
      call_target_name="triton_kernel_call",
      result_types=[*map(mlir.aval_to_ir_type, ctx.avals_out)],
      operands=in_nodes,
      backend_config=zlib.compress(
          kernel_call.to_proto(
              debug_info.func_name,
              params.serialized_metadata or b"",
          )
      ),
      operand_layouts=avals_to_layouts(ctx.avals_in),
      result_layouts=avals_to_layouts(ctx.avals_out),
      operand_output_aliases=dict(input_output_aliases),
  ).results
