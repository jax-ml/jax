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

"""Module for emitting custom GPU pipelines within a Pallas kernel."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import dataclasses
import functools
import itertools as it
from typing import Any

import jax
from jax import lax
from jax._src import util
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import primitives as gpu_primitives
from jax.experimental import pallas as pl


map = util.safe_map


@dataclasses.dataclass(frozen=True)
class BufferedRef:
  spec: pallas_core.BlockSpec
  gmem_ref: pallas_core.AbstractMemoryRef
  smem_ref: pallas_core.AbstractMemoryRef

  def compute_gmem_slice(self, grid_indices) -> tuple[Any, ...]:
    return tuple(
        pl.ds(idx * size, size)
        for idx, size in zip(
            self.spec.index_map(*grid_indices), self.spec.block_shape
        )
    )

  def copy_in(self, slot, grid_indices, barrier_ref):
    gmem_slices = self.compute_gmem_slice(grid_indices)
    gpu_primitives.copy_gmem_to_smem(
        self.gmem_ref.at[gmem_slices],  # pytype: disable=unsupported-operands
        self.smem_ref.at[slot],
        barrier=barrier_ref.at[slot],
    )

  def copy_out(self, slot, grid_indices):
    gmem_slices = self.compute_gmem_slice(grid_indices)
    gpu_primitives.copy_smem_to_gmem(
        self.smem_ref.at[slot], self.gmem_ref.at[gmem_slices]  # pytype: disable=unsupported-operands
    )


jax.tree_util.register_dataclass(
    BufferedRef, data_fields=["gmem_ref", "smem_ref"], meta_fields=["spec"]
)


def foreach_bref(
    brefs: Iterable[BufferedRef], fn: Callable[[BufferedRef], Any]
) -> None:
  for bref in brefs:
    fn(bref)


def emit_pipeline(
    body,
    *,
    grid: pallas_core.StaticGrid,
    in_specs: Sequence[pallas_core.BlockSpec] = (),
    out_specs: Sequence[pallas_core.BlockSpec] = (),
    max_concurrent_steps: int = 1,
    dimension_semantics: tuple[gpu_core.DimensionSemantics, ...] | None = None,
):
  """Creates a function to emit a manual pipeline within a Pallas kernel."""
  sequential_axes = tuple(
      i for i, s in enumerate(dimension_semantics) if s == "sequential"
  )
  if len(sequential_axes) != 1:
    raise ValueError(
        "Exactly one sequential axis is required, got: {sequential_axes}"
    )
  [sequential_axis] = sequential_axes
  num_steps = grid[sequential_axis]

  # Shrink ``max_concurrent_steps`` if the total number of steps is lower to
  # reduce the size of the allocated buffers below.
  if max_concurrent_steps > num_steps:
    max_concurrent_steps = num_steps

  def pipeline(*gmem_refs: pallas_core.AbstractMemoryRef):
    in_gmem_refs, out_gmem_refs = util.split_list(gmem_refs, [len(in_specs)])
    in_smem_refs = map(
        lambda spec, ref: gpu_core.SMEM(
            (max_concurrent_steps, *spec.block_shape), ref.dtype
        ),
        in_specs,
        in_gmem_refs,
    )
    out_smem_refs = map(
        lambda spec, ref: gpu_core.SMEM(
            (max_concurrent_steps, *spec.block_shape), ref.dtype
        ),
        out_specs,
        out_gmem_refs,
    )
    return pl.run_scoped(
        functools.partial(
            scoped_pipeline,
            in_gmem_refs=in_gmem_refs,
            out_gmem_refs=out_gmem_refs,
        ),
        in_smem_refs=in_smem_refs,
        out_smem_refs=out_smem_refs,
        barrier_ref=gpu_core.Barrier(1, num_barriers=max_concurrent_steps),
    )

  def make_grid_indices(step):
    parallel_count = it.count()
    return [
        pl.program_id(next(parallel_count))
        if axis not in sequential_axes
        else step
        for axis in range(len(grid))
    ]

  def scoped_pipeline(
      *, in_gmem_refs, out_gmem_refs, in_smem_refs, out_smem_refs, barrier_ref
  ):

    in_brefs: Sequence[BufferedRef] = map(
        BufferedRef, in_specs, in_gmem_refs, in_smem_refs
    )
    out_brefs: Sequence[BufferedRef] = map(
        BufferedRef, out_specs, out_gmem_refs, out_smem_refs
    )

    def loop_body(step, _):
      slot = step % max_concurrent_steps
      grid_indices = make_grid_indices(step)

      # Wait for the current GMEM->SMEM copy to complete.
      gpu_primitives.barrier_wait(barrier_ref.at[slot])
      # Wait for the previous output SMEM->GMEM copy to complete.
      gpu_primitives.wait_smem_to_gmem(max_concurrent_steps - 1)

      body(*(bref.smem_ref.at[slot] for bref in it.chain(in_brefs, out_brefs)))

      # Copy the output from SMEM to GMEM.
      foreach_bref(out_brefs, lambda bref: bref.copy_out(slot, grid_indices))

      fetch_step = step + max_concurrent_steps
      fetch_slot = slot  # (x + y) % y == x % y
      jax.lax.cond(
          fetch_step < num_steps,
          lambda: foreach_bref(
              in_brefs,
              lambda bref: bref.copy_in(
                  fetch_slot, make_grid_indices(fetch_step), barrier_ref
              ),
          ),
          lambda: None,
      )

      return ()

    for step in range(min(max_concurrent_steps, num_steps)):
      grid_indices = make_grid_indices(step)
      foreach_bref(
          in_brefs, lambda bref: bref.copy_in(step, grid_indices, barrier_ref)
      )

    lax.fori_loop(0, num_steps, loop_body, ())

    # Finalize the pipeline.
    gpu_primitives.wait_smem_to_gmem(0)

  return pipeline
