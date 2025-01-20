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
import re
from typing import Any
import zlib

import jax
import jax._src.core as jax_core
from jax._src.interpreters import mlir
from jax._src.lib import gpu_triton as triton_kernel_call_lib
from jax._src.lib import triton
from jax._src.pallas import core as pallas_core
from jax._src.pallas.triton import lowering


def normalize_grid(grid: pallas_core.StaticGrid) -> tuple[int, int, int]:
  if isinstance(grid, int):
    grid = (grid,)
  elif len(grid) > 3:
    raise ValueError("`grid` should have three or fewer dimensions.")
  return tuple(grid) + (1,) * (3 - len(grid))  # type: ignore


def avals_to_layouts(avals):
  return [list(reversed(range(aval.ndim))) for aval in avals]


def pallas_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    jaxpr: jax_core.Jaxpr,
    name_and_src_info: pallas_core.NameAndSrcInfo,
    interpret: bool,
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: pallas_core.GridMapping,
    compiler_params: dict[str, Any],
    cost_estimate: pallas_core.CostEstimate | None,
    out_avals: tuple[jax_core.AbstractValue, ...],
):
  del interpret, cost_estimate, out_avals
  if grid_mapping.num_dynamic_grid_bounds:
    raise NotImplementedError(
        "dynamic grid bounds not supported in the Triton backend"
    )
  if grid_mapping.num_index_operands:
    raise NotImplementedError(
        "scalar prefetch not implemented in the Triton backend"
    )
  triton_params = compiler_params.get("triton", compiler_params)
  num_warps = triton_params.get("num_warps", 4)
  num_warps = 4 if num_warps is None else num_warps
  [lowering_platform] = ctx.platforms or ctx.module_context.platforms
  if lowering_platform == "rocm":
    num_stages = triton_params.get("num_stages", 1)
    num_stages = 1 if num_stages is None else num_stages
  else:
    num_stages = triton_params.get("num_stages", 3)
    num_stages = 3 if num_stages is None else num_stages

  if debug:
    print(f"\nThe kernel jaxpr for pallas_call {name_and_src_info}:")
    print(jaxpr)
    print("The grid mapping for pallas_call {name_and_src_info}:")
    print(grid_mapping)

  # Sanitize the name to conform to NVPTX requirements. We do this here
  # to avoid the need to fetch the new name from PTX post compilation.
  name_and_src_info = name_and_src_info.replace(
      name=re.sub(r"[^a-zA-Z0-9_$]", "_", name_and_src_info.name)
  )
  lowering_result = lowering.lower_jaxpr_to_triton_module(
      jaxpr, grid_mapping, name_and_src_info, lowering_platform
  )
  module_op = lowering_result.module.operation
  if debug:
    print(f"\nThe Triton module for pallas_call {name_and_src_info}:")
    print(module_op.get_asm(enable_debug_info=True, pretty_debug_info=True))

  grid_x, grid_y, grid_z = normalize_grid(lowering_result.grid)
  buf = io.BytesIO()
  module_op.write_bytecode(buf)
  gpu_device, *_ = jax.local_devices(backend="gpu")
  compilation_result = triton.compile(
      lowering_platform.upper(),
      buf.getvalue(),
      str(gpu_device.compute_capability),  # e.g. 7.0
      num_warps=num_warps,
      num_ctas=1,
      num_stages=num_stages,
  )
  kernel = triton_kernel_call_lib.TritonKernel(
      name_and_src_info.name,
      num_warps,
      compilation_result.smem_bytes,
      compilation_result.asm,
      module_op.get_asm(enable_debug_info=True, pretty_debug_info=True),
      triton_kernel_call_lib.get_compute_capability(0),
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
  # TODO(slebedev): Migrate to ``jax.ffi``.
  return mlir.custom_call(
      call_target_name="triton_kernel_call",
      result_types=[*map(mlir.aval_to_ir_type, ctx.avals_out)],  # type: ignore[list-item]
      operands=in_nodes,
      backend_config=zlib.compress(
          kernel_call.to_proto(
              name_and_src_info.name,
              triton_params.get("serialized_metadata") or b"",
          )
      ),
      operand_layouts=avals_to_layouts(ctx.avals_in),
      result_layouts=avals_to_layouts(ctx.avals_out),
      operand_output_aliases=dict(input_output_aliases),
  ).results
