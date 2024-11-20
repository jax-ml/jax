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

from collections.abc import Callable, Sequence
import dataclasses
import functools
import itertools as it
import math
from typing import Any

import jax
from jax import lax
from jax._src import core
from jax._src import linear_util as lu
from jax._src import util
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import primitives as gpu_primitives
from jax.experimental import pallas as pl
import jax.numpy as jnp


map = util.safe_map
zip = util.safe_zip


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BufferedRef:
  spec: pallas_core.BlockSpec = dataclasses.field(metadata={"static": True})
  is_index_invariant: bool = dataclasses.field(metadata={"static": True})
  gmem_ref: pallas_core.AbstractMemoryRef
  smem_ref: pallas_core.AbstractMemoryRef  # [num_slots, *spec.block_shape]

  def compute_gmem_slice(self, grid_indices) -> tuple[pl.Slice, ...]:
    index_map = self.spec.index_map
    assert index_map is not None
    return tuple(
        pl.Slice(idx * size, size)  # type: ignore[arg-type]
        for idx, size in zip(
            index_map(*grid_indices), self.spec.block_shape  # type: ignore[arg-type]
        )
    )

  def copy_in(self, slot, grid_indices, barrier_ref):
    gmem_slices = self.compute_gmem_slice(grid_indices)
    gpu_primitives.copy_gmem_to_smem(
        self.gmem_ref.at[gmem_slices],  # pytype: disable=unsupported-operands
        self.smem_ref.at[slot],
        barrier_ref.at[slot],
    )

  def copy_out(self, slot, grid_indices, predicate=None):
    gmem_slices = self.compute_gmem_slice(grid_indices)
    gpu_primitives.copy_smem_to_gmem(
        self.smem_ref.at[slot],
        self.gmem_ref.at[gmem_slices],  # pytype: disable=unsupported-operands
        predicate=predicate,
    )


def _uses_arguments(
    index_map: Callable[..., Any], num_args: int
) -> Sequence[bool]:
  jaxpr, _, _, () = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(index_map), (core.ShapedArray((), jnp.int32),) * num_args
  )
  _, used_inputs = pe.dce_jaxpr(jaxpr, used_outputs=[True] * len(jaxpr.outvars))
  return used_inputs


def _is_index_invariant(
    spec: pallas_core.BlockSpec, grid: pallas_core.StaticGrid
) -> bool:
  index_map = spec.index_map
  assert index_map is not None
  return not any(_uses_arguments(index_map, len(grid)))


def _inc_grid_by_1(
    indices: tuple[jax.Array, ...], grid: Sequence[int]
) -> tuple[jax.Array, ...]:
  next_indices = []
  carry: bool | jax.Array = True
  for idx, size in reversed(list(zip(indices, grid))):
    next_idx = lax.select(carry, idx + 1, idx)
    carry = next_idx == size
    next_indices.append(lax.select(carry, 0, next_idx).astype(idx.dtype))
  return tuple(reversed(next_indices))


# ``pl.Slice`` uses a different pytree encoding, depending on whether the
# start/size are static or dynamic. This leads to pytree structure mismatch
# in the pipeline body. So, we define a different ``Slice`` class below.


@dataclasses.dataclass(frozen=True)
class _Slice:
  start: int | jax.Array
  size: int | jax.Array

  def __eq__(self, other: _Slice) -> jax.Array:  # type: ignore
    return lax.bitwise_and(self.start == other.start, self.size == other.size)


jax.tree_util.register_dataclass(
    _Slice, data_fields=["start", "size"], meta_fields=[]
)


def emit_pipeline(
    body,
    *,
    grid: pallas_core.StaticGrid,
    in_specs: Sequence[pallas_core.BlockSpec] = (),
    out_specs: Sequence[pallas_core.BlockSpec] = (),
    max_concurrent_steps: int = 1,
):
  """Creates a function to emit a manual pipeline within a Pallas kernel."""
  num_steps = math.prod(grid)

  # Shrink ``max_concurrent_steps`` if the total number of steps is lower to
  # reduce the size of the allocated buffers below.
  if max_concurrent_steps > num_steps:
    max_concurrent_steps = num_steps

  def pipeline(*gmem_refs: pallas_core.AbstractMemoryRef):
    for gmem_ref, spec in zip(gmem_refs, it.chain(in_specs, out_specs)):
      if any(
          spec.block_shape[-idx] * grid[-idx] != gmem_ref.shape[-idx]  # type: ignore
          for idx in range(1, len(grid) + 1)
      ):
        raise NotImplementedError(
            f"Cannot emit a pipeline over the {grid=} for {gmem_ref} with block"
            f" shape {spec.block_shape}."
        )

    in_gmem_refs, out_gmem_refs = util.split_list(gmem_refs, [len(in_specs)])
    in_smem_refs, out_smem_refs = util.split_list(
        map(
            lambda spec, ref: gpu_core.SMEM(
                (max_concurrent_steps, *spec.block_shape),  # type: ignore
                ref.dtype,
            ),
            it.chain(in_specs, out_specs),
            gmem_refs,
        ),
        [len(in_specs)],
    )
    return pl.run_scoped(
        functools.partial(
            scoped_pipeline,
            in_gmem_refs=in_gmem_refs,
            out_gmem_refs=out_gmem_refs,
        ),
        in_smem_refs=in_smem_refs,
        out_smem_refs=out_smem_refs,
        barrier_ref=gpu_core.Barrier(
            # TODO(slebedev): Change this to arrive only once.
            len(in_specs),
            num_barriers=max_concurrent_steps,
        ),
    )

  def scoped_pipeline(
      *, in_gmem_refs, out_gmem_refs, in_smem_refs, out_smem_refs, barrier_ref
  ):
    in_brefs: Sequence[BufferedRef] = [
        BufferedRef(spec, _is_index_invariant(spec, grid), gmem_ref, smem_ref)
        for spec, gmem_ref, smem_ref in zip(
            in_specs, in_gmem_refs, in_smem_refs
        )
    ]
    out_brefs: Sequence[BufferedRef] = [
        BufferedRef(spec, _is_index_invariant(spec, grid), gmem_ref, smem_ref)
        for spec, gmem_ref, smem_ref in zip(
            out_specs, out_gmem_refs, out_smem_refs
        )
    ]

    for step, indices in enumerate(
        it.islice(it.product(*map(range, grid)), max_concurrent_steps)
    ):
      map(lambda bref: bref.copy_in(step, indices, barrier_ref), in_brefs)

    def loop_body(step, carry):
      slot = step % max_concurrent_steps
      indices, fetch_indices, last_store_slices = carry

      if in_specs:
        # Wait for the current GMEM->SMEM copy to complete.
        gpu_primitives.barrier_wait(barrier_ref.at[slot])
      # Wait for the previous output SMEM->GMEM copy to complete.
      gpu_primitives.wait_smem_to_gmem(
          max_concurrent_steps - 1, wait_read_only=True
      )

      with pallas_core.grid_env(map(pallas_core.GridAxis, indices, grid)):
        body(
            *(bref.smem_ref.at[slot] for bref in it.chain(in_brefs, out_brefs))
        )

      if not all(bref.is_index_invariant for bref in out_brefs):
        gpu_primitives.commit_smem()

      # Copy the output from SMEM to GMEM.
      new_store_slices = last_store_slices[:]
      for idx, bref in enumerate(out_brefs):
        if bref.is_index_invariant:
          assert last_store_slices[idx] is None
          continue
        assert last_store_slices[idx] is not None
        new_store_slices[idx] = tuple(
            _Slice(s.start, s.size) for s in bref.compute_gmem_slice(indices)
        )
        are_same_slices = map(
            lambda old, new: old == new,
            last_store_slices[idx],
            new_store_slices[idx],
        )
        slices_changed = ~functools.reduce(lax.bitwise_and, are_same_slices)
        is_last_step = step == num_steps - 1
        # TODO(apaszke,slebedev): This still diverges significantly from the
        # TPU semantics in that it will move on to the next SMEM output slice
        # even if it's not storing the previous one.
        bref.copy_out(
            slot,
            indices,
            predicate=lax.bitwise_or(slices_changed, is_last_step),
        )

      fetch_step = step + max_concurrent_steps
      fetch_slot = slot  # (x + y) % y == x % y
      jax.lax.cond(
          fetch_step < num_steps,
          lambda: map(
              lambda bref: bref.copy_in(fetch_slot, fetch_indices, barrier_ref),
              in_brefs,
          ),
          lambda: [None] * len(in_brefs),
      )

      return (
          _inc_grid_by_1(indices, grid),
          _inc_grid_by_1(fetch_indices, grid),
          new_store_slices,
      )

    indices = (jnp.asarray(0, dtype=lax.dtype(0)),) * len(grid)
    fetch_indices = indices
    for _ in range(max_concurrent_steps):
      fetch_indices = _inc_grid_by_1(fetch_indices, grid)
    last_store_slices = [
        None
        if bref.is_index_invariant
        else (_Slice(-1, -1),) * len(bref.spec.block_shape)
        for bref in out_brefs
    ]
    last_indices, _, _ = lax.fori_loop(
        0, num_steps, loop_body, (indices, fetch_indices, last_store_slices)
    )

    # Outputs invariant to the sequential axis are never written from inside the
    # loop. This is the only place where we store them.
    if all(bref.is_index_invariant for bref in out_brefs):
      gpu_primitives.commit_smem()
    last_slot = (num_steps - 1) % max_concurrent_steps
    for bref in out_brefs:
      if bref.is_index_invariant:
        bref.copy_out(last_slot, last_indices, predicate=None)

    # Finalize the pipeline.
    gpu_primitives.wait_smem_to_gmem(0)

  return pipeline
