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
from typing import Any, Protocol, TypeVar, cast

import jax
from jax import api_util
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
T = TypeVar('T')

def _get_block_size(
    bd: pl.Blocked | pl.Element | pl.Squeezed | pl.BoundedSlice | int | None,
) -> int:
  match bd:
    case int():
      return bd
    case pl.Blocked(block_size):
      return block_size
    case _:
      raise NotImplementedError(f"Unsupported block size type: {type(bd)}")

def _get_block_shape(spec: pallas_core.BlockSpec):
  assert spec.block_shape is not None
  return tuple(_get_block_size(bd) for bd in spec.block_shape)

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BufferedRef:
  spec: pallas_core.BlockSpec = dataclasses.field(metadata={"static": True})
  is_index_invariant: bool = dataclasses.field(metadata={"static": True})
  gmem_ref: pallas_core.AbstractMemoryRef
  # ``None`` if the ref is pinned to GMEM; otherwise, has shape
  # [num_slots, *spec.block_shape].
  smem_ref: pallas_core.AbstractMemoryRef | None

  def get_ref_for_slot(
      self, slot: int | jax.Array
  ) -> pallas_core.AbstractMemoryRef:
    if self.smem_ref is None:
      return self.gmem_ref
    return self.smem_ref.at[slot]

  def compute_gmem_slice(self, grid_indices) -> tuple[pl.Slice, ...]:
    index_map = self.spec.index_map
    assert index_map is not None
    # We don't allow Python scalars here, because they are interpreted
    # differently depending on the x32/x64 mode.
    assert all(i.dtype == jnp.dtype(jnp.int32) for i in grid_indices)
    sizes = _get_block_shape(self.spec)
    return tuple(
        pl.Slice(idx * size, size)  # type: ignore[arg-type]
        for idx, size in zip(
            index_map(*grid_indices), sizes  # type: ignore[arg-type]
        )
    )

  def copy_in(self, slot, grid_indices, barrier_ref):
    if not _in_smem(self.spec):
      return
    assert self.smem_ref is not None
    gmem_slices = self.compute_gmem_slice(grid_indices)
    gpu_primitives.copy_gmem_to_smem(
        self.gmem_ref.at[gmem_slices],  # pytype: disable=unsupported-operands
        self.smem_ref.at[slot],  # pytype: disable=unsupported-operands
        barrier_ref.at[slot],
    )

  def copy_out(self, slot, grid_indices, predicate=None):
    if not _in_smem(self.spec):
      return
    assert self.smem_ref is not None
    gmem_slices = self.compute_gmem_slice(grid_indices)
    gpu_primitives.copy_smem_to_gmem(
        self.smem_ref.at[slot],  # pytype: disable=unsupported-operands
        self.gmem_ref.at[gmem_slices],  # pytype: disable=unsupported-operands
        predicate=predicate,
        commit_group=False,
    )


def _uses_arguments(
    index_map: Callable[..., Any], num_args: int
) -> Sequence[bool]:
  if not num_args:
    return ()

  jaxpr, _, _, () = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(
          index_map,
          debug_info=api_util.debug_info("pallas index_map",
                                         index_map,
                                         (0,) * num_args, {})),
      (core.ShapedArray((), jnp.int32),) * num_args
  )
  _, used_inputs = pe.dce_jaxpr(jaxpr, used_outputs=[True] * len(jaxpr.outvars))
  return used_inputs


def _is_index_invariant(
    spec: pallas_core.BlockSpec, grid: pallas_core.TupleGrid
) -> bool:
  if (index_map := spec.index_map) is None:
    return True
  return not any(_uses_arguments(index_map, len(grid)))


def _inc_grid_by_1(
    indices: tuple[jax.Array, ...], grid: pallas_core.TupleGrid
) -> tuple[jax.Array, ...]:
  next_indices = []
  carry: bool | jax.Array = True
  for idx, size in reversed(list(zip(indices, grid))):
    next_idx = lax.select(carry, idx + 1, idx)
    carry = next_idx == size
    next_indices.append(
        lax.select(carry, jnp.asarray(0, dtype=idx.dtype), next_idx)
    )
  return tuple(reversed(next_indices))


def _in_smem(spec: pallas_core.BlockSpec) -> bool:
  return spec.memory_space in (None, gpu_core.SMEM)


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
    body: Callable[..., T],
    *,
    grid: pallas_core.TupleGrid,
    in_specs: Sequence[pallas_core.BlockSpec] = (),
    out_specs: Sequence[pallas_core.BlockSpec] = (),
    max_concurrent_steps: int = 1,
    delay_release: int = 0,
    init_carry: T | None = None,
):
  r"""Creates a function to emit a manual pipeline within a Pallas kernel.

  Args:
    body: The pipeline body function, which is called with

      - ``indices``: Tuple of current loop indices.
      - ``*input_refs``: SMEM refs for inputs.
      - ``*output_refs``: SMEM refs for outputs.

      If ``init_carry`` is provided, ``body`` receives an additional argument
      ``carry`` -- the carry from the previous iteration. It must then return
      the next carry value.
    grid: The grid dimensions for the pipeline.
    in_specs: A sequence of :class:`~jax.experimental.pallas.BlockSpec`\s
      for inputs.
    out_specs: A sequence of :class:`~jax.experimental.pallas.BlockSpec`\s
      for outputs.
    max_concurrent_steps: Maximum concurrently active pipeline stages.
    delay_release: Number of steps to delay before reusing input/output
      references. Must be ``< max_concurrent_steps``. Useful for hiding WGMMA
      latency (typically set to 1).
    init_carry: Optional initial carry. If provided, ``body`` handles
      carry-over state between iterations, and the pipeline returns the
      final carry.

  Returns:
    A function that, when called with GMEM input and output refs, executes the
    pipeline and returns the final carry value (if ``init_carry`` was used),
    otherwise it returns None.
  """
  if max_concurrent_steps <= delay_release:
    raise ValueError(
        "max_concurrent_steps must be greater than delay_release, but"
        f" {max_concurrent_steps=}, {delay_release=}"
    )

  num_steps = math.prod(grid)
  has_dynamic_grid = not isinstance(num_steps, int)

  # Shrink ``max_concurrent_steps`` if the total number of steps is lower to
  # reduce the size of the refs allocated in SMEM.
  if not has_dynamic_grid and max_concurrent_steps > num_steps:
    max_concurrent_steps = cast(int, num_steps)

  def pipeline(*gmem_refs: pallas_core.AbstractMemoryRef):
    in_gmem_refs, out_gmem_refs = util.split_list(gmem_refs, [len(in_specs)])
    in_smem_refs, out_smem_refs = util.split_list(
        [
            gpu_core.SMEM(
                (max_concurrent_steps, *_get_block_shape(spec)),  # type: ignore
                ref.dtype,
                transforms=tuple(
                    t.batch(1) for t in getattr(spec, "transforms", ())
                ),
            )
            if _in_smem(spec)
            else None
            for spec, ref in zip(it.chain(in_specs, out_specs), gmem_refs)
        ],
        [len(in_specs)],
    )
    num_arrivals = sum(map(_in_smem, in_specs))
    return pl.run_scoped(
        functools.partial(
            scoped_pipeline,
            in_gmem_refs=in_gmem_refs,
            out_gmem_refs=out_gmem_refs,
        ),
        in_smem_refs=in_smem_refs,
        out_smem_refs=out_smem_refs,
        barrier_ref=None
        if num_arrivals == 0
        else gpu_core.Barrier(
            # TODO(slebedev): Change this to arrive only once.
            num_arrivals=num_arrivals,
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

    # Initialize the pipeline.
    indices = (jnp.asarray(0, dtype=jnp.int32),) * len(grid)
    fetch_indices = indices
    for step in range(max_concurrent_steps):
      for bref in in_brefs:
        bref.copy_in(step, fetch_indices, barrier_ref)
      fetch_indices = _inc_grid_by_1(fetch_indices, grid)
    del fetch_indices

    # This is true if any of the outputs need to be transferred inside the loop.
    copies_out_in_loop = not all(bref.is_index_invariant for bref in out_brefs)

    def loop_body(step, carry):
      slot = lax.rem(step, max_concurrent_steps)
      indices, fetch_indices, last_store_slices, prev_body_carry = carry

      if barrier_ref is not None:
        # Wait for the current GMEM->SMEM copy to complete, if any.
        gpu_primitives.barrier_wait(barrier_ref.at[slot])
      # Wait for the previous output SMEM->GMEM copy to complete.
      if copies_out_in_loop:
        gpu_primitives.wait_smem_to_gmem(
            max_concurrent_steps - (1 + delay_release), wait_read_only=True
        )

      next_body_carry = body(
          indices,
          *(
              bref.get_ref_for_slot(slot)
              for bref in it.chain(in_brefs, out_brefs)
          ),
          *(prev_body_carry,) if init_carry is not None else (),
      )

      if copies_out_in_loop:
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

      if copies_out_in_loop:
        gpu_primitives.commit_smem_to_gmem_group()

      fetch_step = step + (max_concurrent_steps - delay_release)
      fetch_slot = lax.rem(fetch_step, max_concurrent_steps)

      def do_fetch():
        for bref in in_brefs:
          bref.copy_in(fetch_slot, fetch_indices, barrier_ref)

      jax.lax.cond(
          lax.bitwise_and(step >= delay_release, fetch_step < num_steps),
          do_fetch,
          lambda: None,
      )

      return (
          _inc_grid_by_1(indices, grid),
          _inc_grid_by_1(fetch_indices, grid),
          new_store_slices,
          next_body_carry if init_carry is not None else None,
      )

    # Invariant: ``indices`` and ``fetch_indices`` are always
    # ``max_concurrent_steps-delay_release`` apart.
    fetch_indices = indices
    for _ in range(max_concurrent_steps-delay_release):
      fetch_indices = _inc_grid_by_1(fetch_indices, grid)
    # TODO(justinfu): Only store base pointer instead of all indices.
    last_store_slices = [
        None
        if bref.is_index_invariant
        else (_Slice(-1, -1),) * len(bref.spec.block_shape)
        for bref in out_brefs
    ]
    last_indices, _, _, final_carry = lax.fori_loop(
        0,
        num_steps,
        loop_body,
        (indices, fetch_indices, last_store_slices, init_carry),
    )

    # Outputs invariant to the sequential axis are never written from inside the
    # loop. This is the only place where we store them.
    if not copies_out_in_loop:
      gpu_primitives.commit_smem()

    last_slot = lax.rem(num_steps - 1, max_concurrent_steps)
    for bref in out_brefs:
      if bref.is_index_invariant:
        bref.copy_out(last_slot, last_indices, predicate=None)

    gpu_primitives.commit_smem_to_gmem_group()

    # Finalize the pipeline.
    gpu_primitives.wait_smem_to_gmem(0)
    return final_carry if init_carry is not None else None

  return pipeline


class ComputeContext(Protocol):
  """Protocol for a compute context for the warp specialized pipeline.

  The ComputeContext is run exclusively in the compute thread and allows
  the user to set up a prologue to initialize a pipeline carry and an epilogue
  to consume the final carry.

  All values allocated in the ComputeContext will only be allocated in the
  compute thread and not the memory thread. This can potentially reduce
  register pressure if certain values are only consumed by the compute threads.

  Usage will usually follow this structure:

  ```
  def compute_context(pipeline):
    # Perform prologue work and compute the initial carry.
    initial_carry = ...
    # Run the pipeline.
    final_carry = pipeline(*initial_carry)
    # Perform epilogue work using the final carry.
    do_work(final_carry)
  ```

  """
  def __call__(self, pipeline: Callable[[T], T]) -> None:
    ...


def emit_pipeline_warp_specialized(
    body: Callable[..., None],
    *,
    grid: pallas_core.TupleGrid,
    memory_registers: int,
    in_specs: Sequence[pl.BlockSpec] = (),
    out_specs: Sequence[pl.BlockSpec] = (),
    max_concurrent_steps: int = 2,
    wg_axis: str,
    num_compute_wgs: int,
    manual_consumed_barriers: bool = False,
    compute_context: ComputeContext | None = None,
    memory_thread_idx: int | None = None,
):
  """Creates a function to emit a warp-specialized pipeline.

  The ``body`` function should have the following signature (without carry).
  ``consumed_barriers`` is an optional argument that is only passed if the
  ``manual_consumed_barriers`` argument is True.

  ```
  def body(indices, *input_refs, *output_refs, [consumed_barriers]) -> None:
  ```

  or with a carries enabled (enabled via the ``compute_context`` argument),
  where the body returns the next carry:

  ```
  def body(
      indices, *input_refs, *output_refs, [consumed_barriers], carry
  ) -> Carry:
  ```

  Args:
    body: The pipeline body.
    grid: The grid to use for the pipeline.
    memory_registers: The number of registers to reserve for the memory thread.
      For H100 GPUs, 40 is a reasonable value.
    in_specs: The block specs for the inputs.
    out_specs: The block specs for the outputs.
    max_concurrent_steps: The maximum number of sequential stages that are
      active concurrently. Defaults to 2.
    wg_axis: The axis name for the warp group axis.
    num_compute_wgs: The number of compute warpgroups
    manual_consumed_barriers: If True, consumed barriers will be
      passed into the body function after the output refs. There will be one
      barrier per input and will be passed in the same order.
    compute_context: If specified, enables carries in the pipeline and allows
      a user-specified prologue/epilogue that is only executed in the compute
      thread. The signature of the pipeline body function will be modified
      such that the last argument will be the current carry and it must
      return the next carry.
      The compute_context itself should follow the signature of `ComputeContext`
      and take a pipeline function as its sole argument. Calling the
      pipeline with the initial carry will run the pipeline and return the
      final carry.
    memory_thread_idx: The index of the memory thread. If not specified,
      defaults to the last thread.
  """
  # TODO(justinfu): Factor out common code between warp-specialized and
  # normal pipelines.

  if memory_thread_idx is None:
    memory_thread_idx = num_compute_wgs
  if memory_thread_idx != num_compute_wgs:
    # TODO(justinfu): Indexing calculations for buffers assume the memory
    # thread is the last thread.
    raise NotImplementedError("Memory thread must be the last thread.")

  has_carry = compute_context is not None

  # Trace the index maps to determine if they depend on the grid.
  # Grid-independent values will not be multiple-buffered.
  in_spec_has_seq_axis = [
      not _is_index_invariant(spec, grid) for spec in in_specs]
  out_spec_has_seq_axis = [
      not _is_index_invariant(spec, grid) for spec in out_specs]
  spec_has_seq_axis = [*in_spec_has_seq_axis, *out_spec_has_seq_axis]

  num_steps = math.prod(grid)
  has_dynamic_grid = not isinstance(num_steps, int)

  def _get_slot(step, has_seq_dim):
    """Returns the buffer slot given the pipeline step."""
    if has_seq_dim:
      return step
    else:
      return 0

  # Shrink ``max_concurrent_steps`` if the total number of steps is lower to
  # reduce the size of the refs allocated in SMEM.
  if not has_dynamic_grid and max_concurrent_steps > num_steps:
    max_concurrent_steps = cast(int, num_steps)

  def pipeline(*gmem_refs: pallas_core.AbstractMemoryRef):
    in_gmem_refs, out_gmem_refs = util.split_list(gmem_refs, [len(in_specs)])
    if len(out_gmem_refs) != len(out_specs):
      raise ValueError(
          "Number of output refs does not match number of output specs."
      )
    smem_allocs = []
    for spec, has_seq_dim, gmem_ref in zip(
        it.chain(in_specs, out_specs),
        spec_has_seq_axis,
        gmem_refs):
      slots = max_concurrent_steps if has_seq_dim else 1
      smem_allocs.append(
          gpu_core.SMEM(
              (slots, *spec.block_shape),   # type: ignore
              gmem_ref.dtype,
              transforms=getattr(spec, "transforms", ()),
          )
      )
    in_smem_refs, out_smem_refs = util.split_list(
        smem_allocs, [len(in_specs)])

    in_smem_barriers = []
    consumed_barriers = []
    for has_seq_dim in in_spec_has_seq_axis:
      num_barriers = max_concurrent_steps if has_seq_dim else 1
      in_smem_barriers.append(
          gpu_core.Barrier(
          num_arrivals=1,
          num_barriers=num_barriers))
      if manual_consumed_barriers:
        consumed_barriers.append(
            gpu_core.Barrier(
                num_arrivals=num_compute_wgs,
                num_barriers=max_concurrent_steps,
            )
        )
    if not manual_consumed_barriers:
      # We only allocated one consumed barrier for all inputs when using
      # automatic consumed barriers.
      consumed_barriers = [
          gpu_core.Barrier(
              num_arrivals=num_compute_wgs,
              num_barriers=max_concurrent_steps,
          )
      ]
    return pl.run_scoped(
        functools.partial(
            scoped_pipeline,
            in_gmem_refs=in_gmem_refs,
            out_gmem_refs=out_gmem_refs,
        ),
        in_smem_refs=in_smem_refs,
        out_smem_refs=out_smem_refs,
        in_smem_barrier_refs=in_smem_barriers,
        consumed_barrier_refs=consumed_barriers,
        collective_axes=wg_axis,
    )

  def scoped_pipeline(
      *,
      in_gmem_refs,
      out_gmem_refs,
      in_smem_refs,
      out_smem_refs,
      in_smem_barrier_refs,
      consumed_barrier_refs,
  ):
    in_brefs: Sequence[BufferedRef] = [
        BufferedRef(spec, not has_seq_axis, gmem_ref, smem_ref)
        for spec, has_seq_axis, gmem_ref, smem_ref in zip(
            in_specs, in_spec_has_seq_axis, in_gmem_refs, in_smem_refs
        )
    ]
    out_brefs: Sequence[BufferedRef] = [
        BufferedRef(spec, not has_seq_axis, gmem_ref, smem_ref)
        for spec, has_seq_axis, gmem_ref, smem_ref in zip(
            out_specs, out_spec_has_seq_axis, out_gmem_refs, out_smem_refs
        )
    ]

    def compute_block():
      gpu_primitives.set_max_registers(
          _compute_registers(memory_registers, num_compute_wgs),
          action="increase")

      # This is true if any of the outputs need to be transferred inside the loop.
      copies_out_in_loop = not all(bref.is_index_invariant for bref in out_brefs)

      def compute_loop_body(step, carry):
        indices, last_store_slices, prev_body_carry = carry
        slot = lax.rem(step, max_concurrent_steps)
        # Wait for the current GMEM->SMEM copies to complete.
        for in_barrier, has_seq_dim in zip(
            in_smem_barrier_refs, in_spec_has_seq_axis):
          # TODO(justinfu): Use a single barrier with
          # num_arrivals=len(in_smem_barrier_refs)
          gpu_primitives.barrier_wait(
              in_barrier.at[_get_slot(slot, has_seq_dim)])

        # Wait for the previous output SMEM->GMEM copy to complete.
        if copies_out_in_loop:
          gpu_primitives.wait_smem_to_gmem(max_concurrent_steps - 1)

        body_refs = []
        for bref in it.chain(in_brefs, out_brefs):
          buf_slot = _get_slot(slot, not bref.is_index_invariant)
          body_refs.append(bref.get_ref_for_slot(buf_slot))

        body_args = body_refs
        if manual_consumed_barriers:
          body_args += [consumed_barrier_ref.at[slot] for consumed_barrier_ref in consumed_barrier_refs]
        if has_carry:
          body_args += [prev_body_carry]
        next_body_carry = body(indices, *body_args)

        if not manual_consumed_barriers:
          [consumed_barrier_ref] = consumed_barrier_refs
          gpu_primitives.barrier_arrive(consumed_barrier_ref.at[slot])
        # TODO(justinfu,apaszke): This should probably be done by the memory WG.
        # Copy the output from SMEM to GMEM.
        if copies_out_in_loop:
          gpu_primitives.commit_smem()

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
          bref.copy_out(_get_slot(slot, not bref.is_index_invariant),
                        indices,
                        predicate=slices_changed)
        gpu_primitives.commit_smem_to_gmem_group()
        next_indices = _inc_grid_by_1(indices, grid)
        return (next_indices, new_store_slices, next_body_carry)
      init_indices = (jnp.asarray(0, dtype=jnp.int32),) * len(grid)
      # TODO(justinfu): Only store base pointer instead of all indices.
      last_store_slices = [
          None
          if bref.is_index_invariant
          else (_Slice(-1, -1),) * len(bref.spec.block_shape)
          for bref in out_brefs
      ]

      if has_carry:
        last_indices = None
        def pipeline_callback(user_init_carry):
          nonlocal last_indices
          if last_indices is not None:
            raise ValueError(
              "Cannot call pipeline more than once in `compute_context`")
          init_loop_carry = (init_indices, last_store_slices, user_init_carry)
          last_indices, _, final_body_carry = lax.fori_loop(0,
                        num_steps,
                        compute_loop_body,
                        init_loop_carry)
          return final_body_carry
        compute_context(pipeline_callback)
        if last_indices is None:
          raise ValueError("Pipeline was not called in `compute_context`")
      else:
        assert compute_context is None
        last_indices, _, _ = lax.fori_loop(
            0, num_steps, compute_loop_body,
            (init_indices, last_store_slices, None)
        )

      # Handle index_invariant outputs after the loop. They are not
      # written in the main pipeline loop.
      if not copies_out_in_loop:
        gpu_primitives.commit_smem()
      last_slot = lax.rem(num_steps - 1, max_concurrent_steps)
      for bref in out_brefs:
        if bref.is_index_invariant:
          bref.copy_out(_get_slot(last_slot, has_seq_dim=False),
                        last_indices, predicate=None)

      gpu_primitives.commit_smem_to_gmem_group()

      # Finalize the pipeline.
      gpu_primitives.wait_smem_to_gmem(0)

    # The memory thread executes this block which issues all pipelined DMAs.
    def memory_block():
      gpu_primitives.set_max_registers(memory_registers, action="decrease")
      indices = (jnp.asarray(0, dtype=jnp.int32),) * len(grid)
      if has_dynamic_grid:
        prologue_steps = lax.min(max_concurrent_steps, num_steps)
      else:
        assert max_concurrent_steps <= num_steps
        prologue_steps = max_concurrent_steps

      # Begin initial copies.
      def _init_step(step, indices):
        for bref, barrier in zip(in_brefs, in_smem_barrier_refs):
          buf_slot = _get_slot(step, not bref.is_index_invariant)
          bref.copy_in(buf_slot, indices, barrier)
        return _inc_grid_by_1(indices, grid)

      indices = jax.lax.fori_loop(
          0, prologue_steps, _init_step, indices, unroll=not has_dynamic_grid
      )

      def memory_loop_body(step, carry):
        indices, = carry
        slot = lax.rem(step, max_concurrent_steps)
        fetch_slot = slot  # (x + y) % y == x % y

        if not manual_consumed_barriers:
          # We only have one consumed barrier when using automatic consumed
          # barrier management.
          [consumed_barrier_ref] = consumed_barrier_refs
          gpu_primitives.barrier_wait(consumed_barrier_ref.at[slot])
          consumed_barrier_it = [None] * len(in_brefs)
        else:
          consumed_barrier_it = consumed_barrier_refs

        for bref, barrier, consumed_barrier in zip(
            in_brefs, in_smem_barrier_refs, consumed_barrier_it):
          if manual_consumed_barriers:
            gpu_primitives.barrier_wait(consumed_barrier.at[slot])  # pytype: disable=attribute-error
          bref.copy_in(
              _get_slot(fetch_slot, not bref.is_index_invariant), indices, barrier)
        next_indices = _inc_grid_by_1(indices, grid)
        return (next_indices,)
      lax.fori_loop(0, num_steps - max_concurrent_steps,
                    memory_loop_body, (indices,))
      # Await all the arrivals to not leave barriers in a bad state.
      # We only need to account for the prologue steps.
      @pl.loop(0, prologue_steps, unroll=not has_dynamic_grid)
      def _epi_step(step):
        for barrier in consumed_barrier_refs:
          gpu_primitives.barrier_wait(barrier.at[step])

    wg_idx = lax.axis_index(wg_axis)
    lax.cond(
        wg_idx != memory_thread_idx,
        compute_block,
        memory_block
    )
  return pipeline

def _compute_registers(
    memory_registers: int,
    num_compute_wgs: int,
) -> int:
  """Returns the max number of registers to use in compute threads.

  We start with the theoretical max registers per thread if one wargroup
  (128 threads) used the entire SM's 64k register file (64k / 128 = 512).
  Then reserve `memory_registers` for the producer warpgroup and distribute
  the remaining registers evenly among the compute warpgroups.

  Note: The maximum number of registers per thread is 255, so we clamp
  the value.
  """
  n_registers = min(256, (512 - memory_registers) / num_compute_wgs)
  # Round down to the nearest multiple of 8.
  return int((n_registers // 8) * 8)
