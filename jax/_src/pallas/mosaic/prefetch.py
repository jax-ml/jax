# Copyright 2026 The JAX Authors.
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

"""Prefetching functionality for Pallas pipelines."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses

import jax
from jax._src import state
from jax._src.pallas import core as pallas_core
from jax._src.pallas import helpers
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import pipeline
from jax._src.pallas.mosaic import tpu_info


def _allocate_bref(
    bref: pipeline.BufferedRef, mesh: pallas_core.Mesh, input_val,
) -> pipeline.BufferedRef:
  if not bref.is_buffered:
    return bref
  if bref.is_trivial_windowing:
    buf_shape = input_val.shape
  else:
    block_shape = pipeline._get_block_shape(bref.spec)
    buf_shape = (
        (bref.buffer_count * block_shape[0],)
        if len(block_shape) == 1
        and bref.tiling is not tpu_info.Tiling.SPARSE_CORE
        else (bref.buffer_count, *block_shape)
    )
  core_type = getattr(mesh, "core_type", tpu_core.CoreType.TC)
  mem_space = bref.spec.memory_space or tpu_core.MemorySpace.VMEM
  mem_space = tpu_core.memory_space_to_tpu_memory_space(mem_space, core_type)
  target_mem_space = pallas_core.CoreMemorySpace(mem_space, mesh)
  window_out = jax.empty_ref(
      jax.ShapeDtypeStruct(buf_shape, input_val.dtype),
      memory_space=target_mem_space,
  )
  sem_out = jax.empty_ref(
      jax.ShapeDtypeStruct((bref.buffer_count,), tpu_core.DMASemaphore()),
      memory_space=pallas_core.CoreMemorySpace(
          tpu_core.MemorySpace.SEMAPHORE, mesh
      ),
  )
  return dataclasses.replace(
      bref,
      window_ref=window_out,
      sem_recvs=sem_out if bref.buffer_type.is_input else None,
      sem_sends=sem_out if bref.buffer_type.is_output else None,
      await_prefetch=bref.prefetched_count > 0,
  )


def emit_pipeline_with_async_prefetch(
    body,
    *,
    grid: tuple[int | jax.Array, ...],
    in_specs=(),
    out_specs=(),
    mesh: pallas_core.Mesh,
    tiling: tpu_info.Tiling | None = None,
    core_axis: tuple[int, ...] | int | None = None,
    core_axis_name: tuple[str, ...] | str | None = None,
    dimension_semantics: tuple[tpu_core.GridDimensionSemantics, ...] | None = None,
    trace_scopes: bool = True,
    no_pipelining: bool = False,
    _explicit_indices: bool = False,
) -> tuple[Callable, Callable]:

  pipeline_fn = pipeline.emit_pipeline(
      body,
      grid=grid,
      in_specs=in_specs,
      out_specs=out_specs,
      tiling=tiling,
      core_axis=core_axis,
      core_axis_name=core_axis_name,
      dimension_semantics=dimension_semantics,
      trace_scopes=trace_scopes,
      no_pipelining=no_pipelining,
      _explicit_indices=_explicit_indices,
  )

  def async_prefetch(*inputs):
    in_specs_norm = pipeline._normalize_specs(in_specs)
    filt_in_specs, filt_inputs = pipeline._filter_specs_and_refs(in_specs_norm, inputs)
    inputs_flat = [
        x
        if isinstance(x, (state.AbstractRef, state.TransformedRef))
        else jax.new_ref(x)
        for x in jax.tree.leaves(filt_inputs)
    ]
    in_specs_flat = jax.tree.leaves(filt_in_specs)
    raw_allocs = pipeline._make_pipeline_allocations(
        *inputs_flat, in_specs=in_specs_flat, out_specs=(), grid=grid
    )
    allocs_flat = [
        _allocate_bref(b, mesh, x) for b, x in zip(raw_allocs, inputs_flat)
    ]
    allocations = jax.tree.unflatten(jax.tree.structure(filt_in_specs), allocs_flat)

    def _prefetch_kernel(*in_refs):
      in_refs_list = jax.tree.leaves(in_refs)
      core_axis_ = core_axis_name if core_axis is None else core_axis
      num_cores, core_id = pipeline._resolve_core_info(core_axis_)
      partitioned_grid, grid_offsets = pipeline._partition_grid(
          grid, dimension_semantics, num_cores, core_id
      )

      max_buffer_count = max(
          (2, *(b.buffer_count for b in allocs_flat if b.is_buffered)),
          default=2,
      )
      scheduler = pipeline.Scheduler(
          0,
          (0,) * len(partitioned_grid),
          partitioned_grid,
          grid_offsets=grid_offsets,
          num_stages=max_buffer_count,
          trace_scopes=trace_scopes,
          _explicit_indices=_explicit_indices,
      )
      brefs = [
          dataclasses.replace(b, prefetched_count=0).initialize_slots()
          if b.is_buffered
          else b
          for b in allocs_flat
      ]
      max_prefetch = max(
          (b.prefetched_count for b in allocs_flat if b.is_buffered),
          default=0,
      )
      with scheduler.grid_env():
        for step in range(max_prefetch):
          for i, (b, orig_b, in_ref) in enumerate(
              zip(brefs, allocs_flat, in_refs_list, strict=True)
          ):
            if orig_b.buffer_type.is_input and orig_b.prefetched_count > step:
              brefs[i] = scheduler.initialize_step(
                  b, in_ref, step=step, init_limit=orig_b.prefetched_count
              )

    helpers.kernel(_prefetch_kernel, mesh=mesh)(*inputs_flat)
    return allocations

  return pipeline_fn, async_prefetch
