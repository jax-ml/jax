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
import enum
import functools
import itertools as it
import math
from typing import Any, Protocol, TypeVar, Union, cast

import jax
from jax import api_util
from jax import lax
from jax._src import core
from jax._src import linear_util as lu
from jax._src import state
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


BlockSpecPytree = Sequence[Union[pl.BlockSpec, "BlockSpecPytree"]]
AbstractRefPytree = Sequence[Union[state.AbstractRef, "AbstractRefPytree"]]

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


map_brefs = functools.partial(
    jax.tree.map, is_leaf=lambda x: isinstance(x, BufferedRef)
)

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BufferedRef:
  spec: gpu_core.BlockSpec = dataclasses.field(metadata={"static": True})
  is_index_invariant: bool = dataclasses.field(metadata={"static": True})
  gmem_ref: state.AbstractRef
  # ``None`` if the ref is pinned to GMEM; otherwise, has shape
  # [num_slots, *spec.block_shape].
  smem_ref: state.AbstractRef | None

  def get_ref_for_slot(
      self, slot: int | jax.Array
  ) -> state.AbstractRef:
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

  def copy_in(self, slot, grid_indices, barrier_ref, barrier_slot=None):
    if not _in_smem(self.spec):
      return
    assert self.smem_ref is not None
    gmem_slices = self.compute_gmem_slice(grid_indices)
    gpu_primitives.copy_gmem_to_smem(
        self.gmem_ref.at[gmem_slices],  # pytype: disable=unsupported-operands
        self.smem_ref.at[slot],  # pytype: disable=unsupported-operands
        barrier_ref.at[barrier_slot if barrier_slot is not None else slot],
        collective_axes=getattr(self.spec, "collective_axes", ()),
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

  jaxpr, _, _ = pe.trace_to_jaxpr_dynamic(
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


def _downcast_spec(
    spec: gpu_core.BlockSpec | pallas_core.BlockSpec,
) -> gpu_core.BlockSpec:
  if isinstance(spec, gpu_core.BlockSpec):
    return spec

  return gpu_core.BlockSpec(
      block_shape=spec.block_shape,
      index_map=spec.index_map,
      memory_space=spec.memory_space,
      pipeline_mode=spec.pipeline_mode,
  )


def emit_pipeline(
    body: Callable[..., T],
    *,
    grid: pallas_core.TupleGrid,
    in_specs: Sequence[pallas_core.BlockSpec] = (),
    out_specs: Sequence[pallas_core.BlockSpec] = (),
    max_concurrent_steps: int = 1,
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
    init_carry: Optional initial carry. If provided, ``body`` handles
      carry-over state between iterations, and the pipeline returns the
      final carry.

  Returns:
    A function that, when called with GMEM input and output refs, executes the
    pipeline and returns the final carry value (if ``init_carry`` was used),
    otherwise it returns None.
  """

  in_specs = tuple(map(_downcast_spec, in_specs))
  out_specs = tuple(map(_downcast_spec, out_specs))
  for spec in in_specs:
    if spec.collective_axes:
      raise NotImplementedError(
          "BlockSpecs with collective_axes are not supported in emit_pipeline"
      )
  for spec in out_specs:
    if spec.collective_axes:
      raise ValueError("Output BlockSpecs cannot have collective_axes")
  # TODO(justinfu): Factor out common code between warp-specialized and
  # normal pipelines.
  delay_release_levels = sorted({s.delay_release for s in in_specs}) or [0]
  if delay_release_levels and max_concurrent_steps <= delay_release_levels[0]:
    raise ValueError(
        "max_concurrent_steps must be greater than all delay_release values,"
        f" but {max_concurrent_steps=} and {delay_release_levels=}."
    )

  num_steps = math.prod(grid)
  has_dynamic_grid = not isinstance(num_steps, int)

  # Shrink ``max_concurrent_steps`` if the total number of steps is lower to
  # reduce the size of the refs allocated in SMEM.
  if not has_dynamic_grid and max_concurrent_steps > num_steps:
    max_concurrent_steps = cast(int, num_steps)

  def pipeline(*gmem_refs: state.AbstractRef):
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
    smem_out_brefs = [bref for bref in out_brefs if _in_smem(bref.spec)]
    copies_out_in_loop = not all(bref.is_index_invariant for bref in smem_out_brefs)
    needs_epilogue = any(bref.is_index_invariant for bref in smem_out_brefs)

    def loop_body(step, carry):
      slot = lax.rem(step, max_concurrent_steps)
      indices, fetch_index_levels, last_store_slices, prev_body_carry = carry

      if barrier_ref is not None:
        # Wait for the current GMEM->SMEM copy to complete, if any.
        gpu_primitives.barrier_wait(barrier_ref.at[slot])
      # Wait for the previous output SMEM->GMEM copy to complete.
      if copies_out_in_loop:
        gpu_primitives.wait_smem_to_gmem(
            max_concurrent_steps - 1, wait_read_only=True
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

      for delay_release, fetch_indices in zip(
          delay_release_levels, fetch_index_levels
      ):
        fetch_step = step + (max_concurrent_steps - delay_release)
        fetch_slot = lax.rem(fetch_step, max_concurrent_steps)

        # pylint: disable=cell-var-from-loop
        def do_fetch():
          for bref in in_brefs:
            if bref.spec.delay_release == delay_release:
              bref.copy_in(fetch_slot, fetch_indices, barrier_ref)
        # pylint: enable=cell-var-from-loop

        jax.lax.cond(
            lax.bitwise_and(step >= delay_release, fetch_step < num_steps),
            do_fetch,
            lambda: None,
        )

      next_fetch_indices_levels = [
          _inc_grid_by_1(fetch_indices, grid)
          for fetch_indices in fetch_index_levels
      ]
      return (
          _inc_grid_by_1(indices, grid),
          next_fetch_indices_levels,
          new_store_slices,
          next_body_carry if init_carry is not None else None,
      )

    fetch_index_levels = []
    for delay_release in delay_release_levels:
      fetch_indices = indices
      for _ in range(max_concurrent_steps - delay_release):
        fetch_indices = _inc_grid_by_1(fetch_indices, grid)
      fetch_index_levels.append(fetch_indices)

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
        (indices, fetch_index_levels, last_store_slices, init_carry),
    )

    # Outputs invariant to the sequential axis are never written from inside the
    # loop. This is the only place where we store them.
    if not copies_out_in_loop and needs_epilogue:
      gpu_primitives.commit_smem()

    if needs_epilogue:
      last_slot = lax.rem(num_steps - 1, max_concurrent_steps)
      for bref in out_brefs:
        if bref.is_index_invariant:
          bref.copy_out(last_slot, last_indices, predicate=None)

      gpu_primitives.commit_smem_to_gmem_group()

    if smem_out_brefs:
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


class PipelinePipeline(enum.IntEnum):
  START = 0
  STEADY = 1
  STOP = 2


class WarpSpecializedPipeline(Protocol):
  """Protocol for a warp specialized pipeline."""
  def __call__(
      self, *gmem_refs: Any, allocations: Any | None = None,
  ) -> None:
    ...

  def get_allocations(self, *gmem_refs: Any) -> Any:
    ...


def emit_pipeline_warp_specialized(
    body: Callable[..., None],
    *,
    grid: pallas_core.TupleGrid,
    memory_registers: int,
    in_specs: BlockSpecPytree = (),
    out_specs: BlockSpecPytree = (),
    max_concurrent_steps: int = 2,
    wg_axis: str,
    num_compute_wgs: int,
    pipeline_state: jax.Array | PipelinePipeline | None = None,
    manual_consumed_barriers: bool = False,
    compute_context: ComputeContext | None = None,
    memory_thread_idx: int | None = None,
) -> WarpSpecializedPipeline:
  """Creates a function to emit a warp-specialized pipeline.

  The ``body`` function should have the following signature (without carry).
  ``consumed_barriers`` is an optional argument that is only passed if the
  ``manual_consumed_barriers`` argument is True::

    def body(indices, *input_refs, *output_refs, *consumed_barriers) -> None:

  or with a carries enabled (enabled via the ``compute_context`` argument),
  where the body returns the next carry::

    def body(
        indices, *input_refs, *output_refs, *consumed_barriers, carry
    ) -> Carry:

  When ``manual_consumed_barriers`` is True, the user must arrive on all the
  consumed barriers from all compute warpgroups at each pipeline step.

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
    pipeline_state: If multiple pipelines that have almost the same parameters
      (only in/out_specs and body can differ) are going to be evaluated
      in sequence, this argument can be used to avoid pipeline bubbles between
      their invocations. The first pipeline in the sequence should use the
      ``START`` state, followed by an arbitrary number of ``STEADY`` states,
      followed by a single ``STOP`` state. Note that until the pipeline with
      ``STOP`` is done, the memory thread will not wait for the compute threads
      to complete and fully consume their work. Any modification of their
      operands other than invoking another pipeline is disallowed.

      Important: To achieve bubble-free execution, it is important to also use
      the manual allocation mode by calling ``get_allocations`` on the returned
      function, passing the result to ``pl.run_scoped`` and the provided results
      to the returned function as an ``allocations`` keyword argument.
      Otherwise, the pipeline function will perform the scoped allocation itself
      which can lead to synchronization that can still cause pipeline bubbles.
  """

  # TODO(justinfu): Factor out common code between warp-specialized and
  # normal pipelines.
  if not isinstance(in_specs, (list, tuple)):
    in_specs = (in_specs,)
  if not isinstance(out_specs, (list, tuple)):
    out_specs = (out_specs,)
  if isinstance(in_specs, list):
    in_specs = tuple(in_specs)
  if isinstance(out_specs, list):
    out_specs = tuple(out_specs)

  flat_in_specs, in_specs_treedef = jax.tree.flatten(in_specs)
  flat_in_specs = tuple(map(_downcast_spec, flat_in_specs))
  for spec in flat_in_specs:
    if len(spec.collective_axes) > 1:
      raise ValueError(
          "Only a single collective axis supported in input BlockSpecs, but"
          f" got {spec.collective_axes}"
      )
  collective_axes = tuple(frozenset(
      a for spec in flat_in_specs for a in spec.collective_axes
  ))
  flat_out_specs, out_specs_treedef = jax.tree.flatten(out_specs)
  flat_out_specs = tuple(map(_downcast_spec, flat_out_specs))
  for spec in flat_out_specs:
    if spec.collective_axes:
      raise ValueError("Output BlockSpecs cannot have collective_axes")
  delay_release = None
  for in_spec in in_specs:
    if not isinstance(in_spec, gpu_core.BlockSpec):
      delay_release = 0
      continue
    delay_release = in_spec.delay_release
    if in_spec.delay_release != delay_release:
      raise NotImplementedError(
          "All inputs must have the same delay_release, but"
          f" {in_spec.delay_release=} != {delay_release=}"
      )

  delay_release = delay_release or 0
  if max_concurrent_steps <= delay_release:
    raise ValueError(
        "max_concurrent_steps must be greater than delay_release, but"
        f" {max_concurrent_steps=}, {delay_release=}"
    )

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
      not _is_index_invariant(spec, grid) for spec in flat_in_specs]
  out_spec_has_seq_axis = [
      not _is_index_invariant(spec, grid) for spec in flat_out_specs]
  spec_has_seq_axis = [*in_spec_has_seq_axis, *out_spec_has_seq_axis]
  if not all(in_spec_has_seq_axis):
    raise NotImplementedError("Only inputs with a dependency on the grid are supported.")

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

  def _get_scoped_allocs(*gmem_refs: AbstractRefPytree):
    in_gmem_refs = gmem_refs[:len(in_specs)]
    out_gmem_refs = gmem_refs[len(in_specs):]
    flat_in_gmem_refs, in_gmem_refs_treedef = jax.tree.flatten(in_gmem_refs)
    flat_out_gmem_refs, out_gmem_refs_treedef = jax.tree.flatten(out_gmem_refs)
    if in_specs_treedef != in_gmem_refs_treedef:
      raise ValueError(
          "Input specs and input gmem refs must have the same pytree structure."
          f" {in_specs_treedef} != {in_gmem_refs_treedef}"
      )
    if out_specs_treedef != out_gmem_refs_treedef:
      raise ValueError(
          "Output specs and output gmem refs must have the same pytree structure."
          f" {out_specs_treedef} != {out_gmem_refs_treedef}"
      )
    flat_gmem_refs = [*flat_in_gmem_refs, *flat_out_gmem_refs]
    smem_allocs = []
    for spec, has_seq_dim, gmem_ref in zip(
        it.chain(flat_in_specs, flat_out_specs),
        spec_has_seq_axis,
        flat_gmem_refs):
      slots = max_concurrent_steps if has_seq_dim else 1
      smem_allocs.append(
          gpu_core.SMEM(
              (slots, *spec.block_shape),   # type: ignore
              gmem_ref.dtype,
              transforms=getattr(spec, "transforms", ()),
          )
      )
    flat_in_smem_refs, flat_out_smem_refs = util.split_list(
        smem_allocs, [len(flat_in_specs)])
    in_smem_barrier = gpu_core.Barrier(num_arrivals=len(flat_in_specs), num_barriers=max_concurrent_steps)
    flat_consumed_barriers = []
    consumed_barrier_type: Any
    if collective_axes:
      consumed_barrier_type = functools.partial(
          gpu_core.ClusterBarrier, collective_axes=collective_axes  # type: ignore
      )
    else:
      consumed_barrier_type = gpu_core.Barrier
    for _ in flat_in_specs:
      if manual_consumed_barriers:
        flat_consumed_barriers.append(
            consumed_barrier_type(
                num_arrivals=num_compute_wgs,
                num_barriers=max_concurrent_steps,
            )
        )
    if not manual_consumed_barriers:
      # We only allocated one consumed barrier for all inputs when using
      # automatic consumed barriers.
      flat_consumed_barriers = [
          consumed_barrier_type(
              num_arrivals=num_compute_wgs,
              num_barriers=max_concurrent_steps,
          )
      ]
    return dict(
        flat_in_smem_refs=flat_in_smem_refs,
        flat_out_smem_refs=flat_out_smem_refs,
        in_smem_barrier_ref=in_smem_barrier,
        flat_consumed_barrier_refs=flat_consumed_barriers,
    )

  def pipeline(*gmem_refs: AbstractRefPytree, allocations: Any | None = None):
    """
    Run the pipeline.

    Args:
      *gmem_refs: A list of pytrees of pallas refs
      allocations: The allocation provided by ``pl.run_scoped`` when the result
        of calling ``get_allocations(*gmem_refs)`` is passed to
        ``pl.run_scoped``.
    """
    in_gmem_refs = gmem_refs[:len(in_specs)]
    out_gmem_refs = gmem_refs[len(in_specs):]
    flat_in_gmem_refs, in_gmem_refs_treedef = jax.tree.flatten(in_gmem_refs)
    flat_out_gmem_refs, out_gmem_refs_treedef = jax.tree.flatten(out_gmem_refs)
    if in_specs_treedef != in_gmem_refs_treedef:
      raise ValueError(
          "Input specs and input gmem refs must have the same pytree structure."
          f" {in_specs_treedef} != {in_gmem_refs_treedef}"
      )
    if out_specs_treedef != out_gmem_refs_treedef:
      raise ValueError(
          "Output specs and output gmem refs must have the same pytree structure."
          f" {out_specs_treedef} != {out_gmem_refs_treedef}"
      )

    if allocations is None:
      if pipeline_state is not None:
        raise ValueError(
            "Pipeline state should not be set when using automatic allocation."
        )
      return pl.run_scoped(
          functools.partial(
              scoped_pipeline,
              flat_in_gmem_refs=flat_in_gmem_refs,
              flat_out_gmem_refs=flat_out_gmem_refs,
          ),
          **_get_scoped_allocs(*gmem_refs),
          collective_axes=wg_axis,
      )
    else:
      scoped_pipeline(
          flat_in_gmem_refs=flat_in_gmem_refs,
          flat_out_gmem_refs=flat_out_gmem_refs,
          **allocations,
      )

  pipeline.get_allocations = _get_scoped_allocs

  def scoped_pipeline(
      *,
      flat_in_gmem_refs,
      flat_out_gmem_refs,
      flat_in_smem_refs,
      flat_out_smem_refs,
      in_smem_barrier_ref,
      flat_consumed_barrier_refs,
  ):
    flat_in_brefs: Sequence[BufferedRef] = [
        BufferedRef(spec, not has_seq_axis, gmem_ref, smem_ref)
        for spec, has_seq_axis, gmem_ref, smem_ref in zip(
            flat_in_specs, in_spec_has_seq_axis, flat_in_gmem_refs, flat_in_smem_refs
        )
    ]
    flat_out_brefs: Sequence[BufferedRef] = [
        BufferedRef(spec, not has_seq_axis, gmem_ref, smem_ref)
        for spec, has_seq_axis, gmem_ref, smem_ref in zip(
            flat_out_specs, out_spec_has_seq_axis, flat_out_gmem_refs, flat_out_smem_refs
        )
    ]

    def compute_block():
      gpu_primitives.set_max_registers(
          _compute_registers(memory_registers, num_compute_wgs),
          action="increase")

      # This is true if any of the outputs need to be transferred inside the loop.
      smem_out_brefs = [bref for bref in flat_out_brefs if _in_smem(bref.spec)]
      # The implementation below has races when we have multiple compute WGs.
      # The problem is that we expect the compute WGs to deal with issuing the
      # SMEM->GMEM copies, but (1) we never predicate them, so we repeat the
      # same copy multiple times, and (2) we don't synchronize the compute WGs
      # in any way. In the unlikely event that one of the compute WGs runs 2
      # steps ahead, it might start overwriting the output buffer before the
      # other WG has issued its copy.
      #
      # The best fix here would be to move the SMEM->GMEM copies into the memory
      # WG and use proper barriers (with arrival_count=2) to ensure all WGs have
      # produced their outputs before it is sent out to GMEM.
      if smem_out_brefs and num_compute_wgs > 1:
        raise NotImplementedError(
            "SMEM outputs are not supported with multiple compute warpgroups"
        )
      copies_out_in_loop = not all(bref.is_index_invariant for bref in smem_out_brefs)
      needs_epilogue = any(bref.is_index_invariant for bref in smem_out_brefs)

      def compute_loop_body(step, carry):
        indices, last_store_slices, prev_body_carry = carry
        slot = lax.rem(step, max_concurrent_steps)
        consumed_slot = lax.rem(step - delay_release, max_concurrent_steps)
        # Wait for the current GMEM->SMEM copies to complete.
        gpu_primitives.barrier_wait(in_smem_barrier_ref.at[_get_slot(slot, True)])

        # Wait for the previous output SMEM->GMEM copy to complete.
        if copies_out_in_loop:
          gpu_primitives.wait_smem_to_gmem(
              max_concurrent_steps - 1, wait_read_only=True
          )

        in_brefs = jax.tree.unflatten(in_specs_treedef, flat_in_brefs)
        out_brefs = jax.tree.unflatten(out_specs_treedef, flat_out_brefs)
        all_brefs = (*in_brefs, *out_brefs)
        body_args = map_brefs(
            lambda bref: bref.get_ref_for_slot(
                _get_slot(slot, not bref.is_index_invariant)
            ),
            all_brefs,
        )

        if manual_consumed_barriers:
          barriers = jax.tree.unflatten(
              in_specs_treedef,
              [barrier.at[consumed_slot] for barrier in flat_consumed_barrier_refs],
          )
          body_args = (*body_args, *barriers)
        if has_carry:
          body_args = (*body_args, prev_body_carry)
        next_body_carry = body(indices, *body_args)

        if not manual_consumed_barriers:
          [consumed_barrier_ref] = flat_consumed_barrier_refs
          if delay_release > 0:
            lax.cond(
                step < delay_release,
                lambda: None,
                lambda: gpu_primitives.barrier_arrive(consumed_barrier_ref.at[consumed_slot]),
            )
          else:
            gpu_primitives.barrier_arrive(consumed_barrier_ref.at[consumed_slot])
        # TODO(justinfu,apaszke): This should probably be done by the memory WG.
        # Copy the output from SMEM to GMEM.
        if copies_out_in_loop:
          gpu_primitives.commit_smem()

        new_store_slices = last_store_slices[:]
        for idx, bref in enumerate(flat_out_brefs):
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
          for bref in flat_out_brefs
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
      if not copies_out_in_loop and needs_epilogue:
        gpu_primitives.commit_smem()

      if needs_epilogue:
        last_slot = lax.rem(num_steps - 1, max_concurrent_steps)
        for bref in flat_out_brefs:
          if bref.is_index_invariant:
            bref.copy_out(_get_slot(last_slot, has_seq_dim=False),
                          last_indices, predicate=None)

        gpu_primitives.commit_smem_to_gmem_group()

      if smem_out_brefs:
        # Finalize the pipeline.
        gpu_primitives.wait_smem_to_gmem(0)

    # The memory thread executes this block which issues all pipelined DMAs.
    # TODO(apaszke,justinfu): Use a single arrive_expect_tx for all transfers.
    def memory_block():
      gpu_primitives.set_max_registers(memory_registers, action="decrease")
      indices = (jnp.asarray(0, dtype=jnp.int32),) * len(grid)
      if has_dynamic_grid:
        prologue_steps = lax.min(max_concurrent_steps, num_steps)
      else:
        assert max_concurrent_steps <= num_steps
        prologue_steps = max_concurrent_steps
      pipeline_init_prologue_steps = prologue_steps
      if pipeline_state is not None:
        if has_dynamic_grid:
          raise NotImplementedError(
              "A pipeline of pipelines is not supported with dynamic grids"
          )
        if num_steps % max_concurrent_steps:
          raise NotImplementedError(
              "A pipeline of pipelines is only allowed when the number of steps"
              f" (product of grid, here {num_steps}) is divisible by"
              f" {max_concurrent_steps=}"
          )
        if delay_release:
          raise NotImplementedError(
              "A pipeline of pipelines is not supported with delay_release"
          )
        if isinstance(pipeline_state, PipelinePipeline):
          prologue_steps = prologue_steps if pipeline_state == PipelinePipeline.START else 0
        else:
          prologue_steps = jnp.where(pipeline_state == PipelinePipeline.START, prologue_steps, 0)

      # Begin initial copies.
      def _init_step(step, indices):
        for bref in flat_in_brefs:
          buf_slot = _get_slot(step, not bref.is_index_invariant)
          barrier_slot = _get_slot(step, True)
          bref.copy_in(buf_slot, indices, in_smem_barrier_ref, barrier_slot)
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
          [consumed_barrier_ref] = flat_consumed_barrier_refs
          gpu_primitives.barrier_wait(consumed_barrier_ref.at[slot])
          consumed_barrier_it = [None] * len(flat_in_brefs)
        else:
          consumed_barrier_it = flat_consumed_barrier_refs

        for bref, consumed_barrier in zip(flat_in_brefs, consumed_barrier_it):
          if manual_consumed_barriers:
            gpu_primitives.barrier_wait(consumed_barrier.at[slot])  # pytype: disable=attribute-error
          buf_slot = _get_slot(fetch_slot, not bref.is_index_invariant)
          barrier_slot = _get_slot(fetch_slot, True)
          bref.copy_in(buf_slot, indices, in_smem_barrier_ref, barrier_slot)
        next_indices = _inc_grid_by_1(indices, grid)
        return (next_indices,)
      lax.fori_loop(0, num_steps - prologue_steps, memory_loop_body, (indices,))
      # Await all the arrivals to not leave barriers in a bad state.
      # We only need to account for the prologue steps, only the first
      # delay_release of them skip arrivals, so we subtract them.
      @pl.when(pipeline_state is None or pipeline_state == PipelinePipeline.STOP)
      def _quiesce():
        @pl.loop(
            num_steps - pipeline_init_prologue_steps,
            num_steps - delay_release,
            unroll=not has_dynamic_grid,
        )
        def _epi_step(step):
          consumed_slot = lax.rem(step, max_concurrent_steps)
          for barrier in flat_consumed_barrier_refs:
            gpu_primitives.barrier_wait(barrier.at[consumed_slot])

    wg_idx = lax.axis_index(wg_axis)
    lax.cond(
        wg_idx != memory_thread_idx,
        compute_block,
        memory_block
    )
  # Mypy doesn't notice the .get_allocations assignment above.
  return pipeline  # type: ignore

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
