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

from collections.abc import Sequence
import os
import time
from typing import cast
import warnings

import jax
from jax._src import core as jax_core
from jax._src import sharding_impls
from jax._src.interpreters import mlir
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import lowering
from jax.experimental.mosaic import gpu as mgpu
import jax.numpy as jnp
import numpy as np


def _emit_mosaic_gpu_custom_call(
    ctx: mlir.LoweringRuleContext,
    args,
    lowering_result: lowering.LoweringResult,
    input_output_aliases: tuple[tuple[int, int], ...],
    debug_info,
    skip_device_barrier: bool = False,
):
  module = lowering_result.module
  new_avals_in = list(ctx.avals_in)
  new_avals_out = list(map(_as_shaped_array, lowering_result.new_out_shapes))
  scratch_args = ()
  if lowering_result.gmem_scratch_shapes:
    # The new_out_shapes contain the original outputs first, followed by the
    # GMEM scratch shapes, and optionally the profiler buffer.
    input_output_aliases += tuple(
        (len(ctx.avals_in) + i, len(ctx.avals_out) + i)
        for i in range(len(lowering_result.gmem_scratch_shapes))
    )
    # The GMEM scratch is an aliased kernel input/output.
    new_avals_in.extend(map(_as_shaped_array, lowering_result.gmem_scratch_shapes))
    # We guarantee zero-initialization of the GMEM scratch at the moment, which
    # is important for semaphores.
    def zero_init_gmem_scratch():
      return [jnp.zeros_like(s) for s in lowering_result.gmem_scratch_shapes]
    scratch_args = mlir.lower_fun(
        zero_init_gmem_scratch, multiple_results=True
    )(ctx.replace(avals_in=()))
  outs = mgpu.core._mosaic_gpu_lowering_rule(
      ctx.replace(avals_in=new_avals_in, avals_out=new_avals_out),
      *args,
      *scratch_args,
      module=module,
      out_types=lowering_result.new_out_shapes,
      inout_types=(),
      input_output_aliases=input_output_aliases,
      # False until we add get_barrier_semaphore() feature.
      use_custom_barrier=False,
      skip_device_barrier=skip_device_barrier,
  )
  if (prof_spec := lowering_result.profiler_spec) is not None:
    *outs, prof_buffer = outs
    out_file = os.path.join(
        prof_spec.dump_path,
        f"{mlir.sanitize_name(debug_info.func_name)}-{time.time_ns()}-trace.json",
    )
    def dump_profile(prof_buffer):
      assert prof_spec is not None  # pyrefly#40
      try:
        with open(out_file, "x") as f:
          prof_spec.dump(
              prof_buffer,
              f,
              grid=lowering_result.grid,
              block=lowering_result.block,
          )
      except FileExistsError:
        warnings.warn(
            f"Failed to dump profile for {debug_info.func_src_info}, profile"
            f" already exists at {out_file}"
        )
    def do_callback(prof_buffer):
      jax.debug.callback(dump_profile, prof_buffer)
      return ()
    mlir.lower_fun(do_callback, multiple_results=True)(
        ctx.replace(avals_in=(new_avals_out[-1],)), prof_buffer
    )
  if lowering_result.gmem_scratch_shapes:  # Drop the GMEM scratch.
    outs = outs[:-len(lowering_result.gmem_scratch_shapes)]
  return outs


def _as_shaped_array(t: jax.ShapeDtypeStruct) -> jax_core.ShapedArray:
  return jax_core.ShapedArray(t.shape, np.dtype(t.dtype))


def mpmd_map_mgpu_lowering_rule(
    ctx: mlir.LoweringRuleContext,
    *args,
    meshes,
    jaxprs,
    out_avals,
    input_output_aliases,
    compiler_params,
    interpret,
    debug,
    cost_estimate,
    metadata,
    name,
    external_meshes,
):
  del interpret, cost_estimate, metadata, name, out_avals  # Unused.

  if len(jaxprs) != 1:
    raise NotImplementedError(
        "Lowering multiple mesh/function pairs is not supported by the Mosaic"
        " GPU backend"
    )
  if external_meshes:
    raise NotImplementedError(
        "External meshes are not supported by the Mosaic GPU backend"
    )
  [jaxpr] = jaxprs
  [mesh] = meshes
  if not isinstance(mesh, gpu_core.Mesh):
    raise NotImplementedError(
        f"Mesh {mesh} is not supported by the Mosaic GPU backend"
    )
  # On GPU ``mpmd_map`` kernels never carry scratch operands -- scratch is
  # handled separately by ``plgpu.kernel``. So the jaxpr invars are exactly the
  # inputs followed by the outputs.
  if len(jaxpr.invars) != len(args) + len(ctx.avals_out):
    raise NotImplementedError(
        "Scratch operands are not supported by the Mosaic GPU mpmd_map lowering"
    )

  if debug:
    print(f"\nThe kernel jaxpr for mpmd_map {jaxpr.debug_info.func_src_info}:")
    print(jaxpr)

  mgpu.dialect.register_dialect(ctx.module_context.context)

  if compiler_params is None:
    gpu_params = gpu_core.CompilerParams()
  else:
    assert isinstance(compiler_params, gpu_core.CompilerParams)
    gpu_params = compiler_params

  jax_mesh = None
  axis_context = ctx.module_context.axis_context
  if axis_context is not None:
    if isinstance(axis_context, sharding_impls.SPMDAxisContext):
      jax_mesh = axis_context.mesh

  # ``axis_index``/``program_id`` inside the kernel (including nested
  # ``run_scoped`` traces during lowering) resolve mesh axes from the JAX core
  # axis environment. The pipelined path binds them via ``grid_mapping.trace_env``;
  # here we bind them straight from the mesh.
  from jax._src.pallas import mpmd  # pyrefly: ignore[import-cycle]
  with mpmd.mpmd_map_tracing_context(mesh, (*meshes, *external_meshes)):
    lowering_result = lowering.lower_unpipelined_jaxpr_to_module(
        mesh,
        jax_mesh,
        jaxpr,
        cast(Sequence[jax_core.ShapedArray], ctx.avals_in),
        cast(Sequence[jax_core.ShapedArray], ctx.avals_out),
        gpu_params,
        outer_traceback=ctx.traceback,
    )
  if debug:
    print(f"\nThe Mosaic GPU module for mpmd_map {jaxpr.debug_info.func_src_info}:")
    print(lowering_result.module.operation)

  return _emit_mosaic_gpu_custom_call(
      ctx,
      args,
      lowering_result,
      tuple(input_output_aliases.items()),
      jaxpr.debug_info,
      skip_device_barrier=gpu_params.skip_device_barrier,
  )
