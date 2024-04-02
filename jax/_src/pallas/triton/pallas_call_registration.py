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

# TODO(sharadmv): Enable type checking.
# mypy: ignore-errors

from __future__ import annotations

import dataclasses
import io
from typing import Any
import zlib

import jax
from jax import core as jax_core
from jax._src import config
from jax._src.interpreters import mlir
from jax._src.lib import gpu_triton as triton_kernel_call_lib
from jax._src.lib.mlir import ir
from jax._src.pallas import core as pallas_core
from jax._src.pallas.pallas_call import pallas_call_p
from jax._src.pallas.triton import lowering
from jax._src import util


@dataclasses.dataclass
class CompilationResult:
  kernel_name: str
  ttir: str
  ptx: str
  shared_mem_bytes: int
  compute_capability: int
  lowering_result: lowering.LoweringResult


@util.weakref_lru_cache
def compile_jaxpr(
    jaxpr: jax_core.Jaxpr,
    in_shapes,
    grid_mapping: pallas_core.GridMapping,
    name: str,
    num_warps: int,
    num_stages: int,
    debug: bool,
) -> CompilationResult:
  from jax_triton.triton_lib import compile_ttir_to_ptx_inplace  # type: ignore
  import triton.backends.nvidia.compiler as cb  # type: ignore

  # TODO(sharadmv): handle multiple devices, right now we assume device 0
  # which is fine when we have multiple of the same GPU but this won't work in
  # general.
  device = 0
  compute_capability = triton_kernel_call_lib.get_compute_capability(device)
  target = ("cuda", compute_capability)
  cuda_backend = cb.CUDABackend(target)
  cuda_options = cuda_backend.parse_options(
      dict(
          num_warps=num_warps,
          num_stages=num_stages,
          debug=debug,
      )
  )
  lowering_result = lowering.lower_jaxpr_to_triton_module(
      jaxpr, in_shapes, grid_mapping, name, cuda_options
  )

  ttir = str(lowering_result.module)
  ptx, name, shared_mem_bytes, _ = compile_ttir_to_ptx_inplace(
      lowering_result.module,
      cuda_backend,
      cuda_options,
      compute_capability,
  )
  return CompilationResult(
      name, ttir, ptx, shared_mem_bytes, compute_capability, lowering_result
  )


def normalize_grid(grid: pallas_core.StaticGrid) -> tuple[int, int, int]:
  if isinstance(grid, int):
    grid = (grid,)
  elif len(grid) > 3:
    raise ValueError("`grid` should have three or fewer dimensions.")
  return tuple(grid) + (1,) * (3 - len(grid))


def avals_to_layouts(avals):
  return [list(reversed(range(aval.ndim))) for aval in avals]


def _pallas_call_ptx_lowering(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    jaxpr: jax_core.Jaxpr,
    name: str,
    in_shapes: tuple[jax.ShapeDtypeStruct, ...],
    out_shapes: tuple[jax.ShapeDtypeStruct, ...],
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: pallas_core.GridMapping,
    triton_params: dict[str, Any],
    num_warps: int,
    num_stages: int,
):
  compilation_result = compile_jaxpr(
      jaxpr,
      (*in_shapes, *out_shapes),
      grid_mapping,
      name,
      num_warps,
      num_stages,
      debug=debug,
  )
  # Triton returns a tuple for ROCm. We just want file path to be passed
  if ctx.module_context.platforms[0] == 'rocm':
    compilation_result.ptx = compilation_result.ptx[1]

  if debug:
    compilation_result.lowering_result.module.dump()

  kernel = triton_kernel_call_lib.TritonKernel(
      compilation_result.kernel_name,
      num_warps,
      compilation_result.shared_mem_bytes,
      compilation_result.ptx,
      compilation_result.ttir,
      compilation_result.compute_capability,
      1,
      1,
      1,  # TODO(giorgioa): Add support for clustering on H100s on Pallas.
  )

  grid = normalize_grid(compilation_result.lowering_result.grid)

  kernel_params = []
  for _ in range(len(in_shapes) + len(out_shapes)):
    kernel_params.append(
        triton_kernel_call_lib.create_array_parameter(
            0,  # bytes to zero  # TODO(cjfj): Expose through user API.
            16,  # divisible by 16
        )
    )

  kernel_call = triton_kernel_call_lib.TritonKernelCall(
      kernel, grid[0], grid[1], grid[2], kernel_params
  )

  out_types = [
      ir.RankedTensorType.get(shape.shape, mlir.dtype_to_ir_type(shape.dtype))
      for shape in out_shapes
  ]

  serialized_metadata = triton_params.get("serialized_metadata", b"")
  kernel_call_proto = kernel_call.to_proto(name, serialized_metadata)
  return mlir.custom_call(
      call_target_name="triton_kernel_call",
      result_types=out_types,
      operands=in_nodes,
      backend_config=zlib.compress(kernel_call_proto),
      operand_layouts=avals_to_layouts(ctx.avals_in),
      result_layouts=avals_to_layouts(ctx.avals_out),
      operand_output_aliases=dict(input_output_aliases),
  ).results


def _pallas_call_ttir_lowering(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    jaxpr: jax_core.Jaxpr,
    name: str,
    in_shapes: tuple[jax.ShapeDtypeStruct, ...],
    out_shapes: tuple[jax.ShapeDtypeStruct, ...],
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: pallas_core.GridMapping,
    triton_params: dict[str, Any] | None = None,
    num_warps: int,
    num_stages: int,
):
  # TODO(sharadmv): handle multiple devices, right now we assume device 0
  # which is fine when we have multiple of the same GPU but this won't work in
  # general.
  device = 0
  compute_capability = triton_kernel_call_lib.get_compute_capability(device)
  cuda_options = dict(
      compute_capability=compute_capability,
      num_warps=num_warps,
      num_stages=num_stages,
      debug=debug,
  )

  lowering_result = lowering.lower_jaxpr_to_triton_module(
      jaxpr, (*in_shapes, *out_shapes), grid_mapping, name, cuda_options
  )
  module_op = lowering_result.module.operation
  if debug:
    print(module_op.get_asm(enable_debug_info=True, pretty_debug_info=True))

  grid_x, grid_y, grid_z = normalize_grid(lowering_result.grid)
  out_types = [
      ir.RankedTensorType.get(shape.shape, mlir.dtype_to_ir_type(shape.dtype))
      for shape in out_shapes
  ]
  buf = io.BytesIO()
  module_op.write_bytecode(buf)
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
  if "serialized_metadata" in (triton_params or {}):
    # This field is unstable and may be removed in the future.
    backend_config["serialized_metadata"] = ir.StringAttr.get(
        triton_params["serialized_metadata"]
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


_TRITON_COMPILE_VIA_XLA = config.DEFINE_bool(
    "jax_triton_compile_via_xla",
    default=config.bool_env("JAX_TRITON_COMPILE_VIA_XLA", True),
    help="If True, Pallas delegates Triton kernel compilation to XLA.",
)


def pallas_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    jaxpr: jax_core.Jaxpr,
    name: str,
    in_shapes: tuple[jax.ShapeDtypeStruct, ...],
    out_shapes: tuple[jax.ShapeDtypeStruct, ...],
    which_linear: tuple[bool, ...],
    interpret: bool,
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: pallas_core.GridMapping,
    compiler_params: dict[str, Any],
):
  if interpret:
    return mlir.lower_fun(pallas_call_p.impl, multiple_results=True)(
        ctx,
        *in_nodes,
        jaxpr=jaxpr,
        name=name,
        out_shapes=out_shapes,
        in_shapes=in_shapes,
        which_linear=which_linear,
        interpret=interpret,
        debug=debug,
        input_output_aliases=input_output_aliases,
        grid_mapping=grid_mapping,
        compiler_params=compiler_params,
    )

  if grid_mapping.num_dynamic_grid_bounds:
    raise NotImplementedError(
        "dynamic grid bounds not supported in the Triton backend"
    )
  triton_compiler_params = compiler_params.get("triton", compiler_params)
  triton_params = compiler_params.get("triton_params", {})
  num_warps = triton_compiler_params.pop("num_warps", 4)
  if len(ctx.module_context.platforms) > 1:
    raise NotImplementedError("multi-platform lowering for Pallas kernels")
  if ctx.module_context.platforms[0] == "rocm":
    num_stages = triton_compiler_params.pop("num_stages", 1)
  else:
    num_stages = triton_compiler_params.pop("num_stages", 3)

  if debug:
    print(jaxpr)
    print(grid_mapping)

  if _TRITON_COMPILE_VIA_XLA.value:
    lowering_fn = _pallas_call_ttir_lowering
  else:
    lowering_fn = _pallas_call_ptx_lowering

  return lowering_fn(
        ctx,
        *in_nodes,
        jaxpr=jaxpr,
        name=name,
        in_shapes=in_shapes,
        out_shapes=out_shapes,
        debug=debug,
        input_output_aliases=input_output_aliases,
        grid_mapping=grid_mapping,
        triton_params=triton_params,
        num_warps=num_warps,
        num_stages=num_stages,
    )
