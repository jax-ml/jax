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

import os
import time
import warnings

import jax
from jax._src import config
from jax._src import core as jax_core
from jax._src import frozen_dict
from jax._src import sharding_impls
from jax._src.interpreters import mlir
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import lowering
from jax.experimental.mosaic import gpu as mgpu
import jax.numpy as jnp
import numpy as np


def pallas_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *args,
    jaxpr: jax_core.Jaxpr,
    interpret: bool,
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: pallas_core.GridMapping,
    mesh: pallas_core.Mesh | None,
    compiler_params: pallas_core.CompilerParams | None,
    cost_estimate: pallas_core.CostEstimate | None,
    out_avals: tuple[jax_core.AbstractValue, ...],
    metadata: frozen_dict.FrozenDict[str, str] | None,
    name: str | None,
):
  del metadata, name  # TODO(sharadmv): Add metadata to HLO.
  debug_info = jaxpr.debug_info
  del interpret, out_avals
  if grid_mapping.num_dynamic_grid_bounds:
    raise NotImplementedError(
        "dynamic grid bounds not supported in the Mosaic GPU backend"
    )

  if debug:
    print(f"\nThe kernel jaxpr for pallas_call {debug_info.func_src_info}:")
    print(jaxpr)
    print(f"The grid mapping for pallas_call {debug_info.func_src_info}:")
    print(grid_mapping)

  mgpu.dialect.register_dialect(ctx.module_context.context)  # pytype: disable=attribute-error

  if compiler_params is None:
    gpu_params = gpu_core.CompilerParams()
  else:
    assert isinstance(compiler_params, gpu_core.CompilerParams)
    gpu_params = compiler_params  # type: ignore[assignment]


  jax_mesh = None
  axis_context = ctx.module_context.axis_context
  if axis_context is not None:
    if isinstance(axis_context, sharding_impls.SPMDAxisContext):
      jax_mesh = axis_context.mesh

  # TODO(slebedev): Remove this once the ensure-debug-info-scope-on-llvm-func
  # pass correctly handles full tracebacks.
  with config.include_full_tracebacks_in_locations(False):
    lowering_result = lowering.lower_pipelined_jaxpr_to_module(
        grid_mapping, mesh, jax_mesh, jaxpr, gpu_params, cost_estimate
    )
  if debug:
    print(f"\nThe Mosaic GPU module for pallas_call {debug_info.func_src_info}:")
    print(lowering_result.module.operation)

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
      *args, *scratch_args,
      module=module,
      out_types=lowering_result.new_out_shapes,
      inout_types=(),
      input_output_aliases=input_output_aliases,
      # False until we add get_barrier_semaphore() feature.
      use_custom_barrier=False,
  )
  if (prof_spec := lowering_result.profiler_spec) is not None:
    *outs, prof_buffer = outs
    out_file = os.path.join(
        prof_spec.dump_path,
        f"{mlir.sanitize_name(debug_info.func_name)}-{time.time_ns()}-trace.json",
    )
    def dump_profile(prof_buffer):
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
            f"Failed to dump profile for pallas_call {debug_info.func_src_info}, "
            f"profile already exists at {out_file}"
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
