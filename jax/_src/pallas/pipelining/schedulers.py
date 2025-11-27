# Copyright 2025 The JAX Authors.
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
"""Pipeline scheduler implementations."""
# mypy: ignore-errors
# pytype: disable=invalid-annotation
# pytype: disable=wrong-arg-types
# pytype: disable=bad-return-type
# pylint: disable=missing-function-docstring
# pylint: disable=g-doc-args

import collections
from collections.abc import Callable, Mapping, Sequence
import copy
import dataclasses
import functools
import math
import operator
from typing import Any, cast, Protocol

import jax
from jax import lax
from jax import numpy as jnp
from jax._src import core as jax_core
import numpy as np

from jax._src.pallas.pipelining import internal


PipelineState = Any
PipelineScheduler = Callable[
    [internal.NDLoopStruct, Sequence[Any], Any, Any], None]


def compute_grid_indices(linear_index: jax.Array, grid_size: Sequence[int]):
  """Computes the grid indices for a given linear index."""
  indices = []
  for i, _ in enumerate(grid_size):
    rest_size = math.prod(grid_size[i+1:])
    axis_index = linear_index // rest_size
    indices.append(axis_index)
    linear_index = lax.rem(linear_index, rest_size)
  return indices


def increment_grid(indices: Sequence[int | jax.Array],
                   grid: Sequence[int],
                   dynamic: bool = False):
  """Increments the grid indices by 1."""
  next_indices = []
  carry: bool | jax.Array = True
  for idx, size in reversed(list(zip(indices, grid, strict=True))):
    if dynamic:
      idx = cast(jax.Array, idx)
      next_idx = lax.select(carry, idx + 1, idx)
      carry = next_idx == size
      next_indices.append(
          lax.select(carry, jnp.asarray(0, dtype=idx.dtype), next_idx)
      )
    else:
      next_idx = idx + 1 if carry else idx
      carry = next_idx == size
      next_indices.append(0 if carry else next_idx)
  return tuple(reversed(next_indices)), carry


@functools.partial(jax.tree_util.register_dataclass,
                   data_fields=["loop_index",
                                "linearized_index",
                                "pipeline_state"],
                   meta_fields=[])
@dataclasses.dataclass(frozen=True)
class PipelineContext:
  """Container class containing pipeline state information.

  Attributes:
    loop_index: The current grid indices to run for the current stage.
    linearized_index: The linearized ``loop_index``.
    pipeline_state: The global pipeline carry state.
  """
  loop_index: tuple[jax.Array, ...]
  linearized_index: jax.Array
  pipeline_state: PipelineState

  @classmethod
  def aval_pytree(cls, grid, state_avals) -> "PipelineContext":
    return PipelineContext(
        loop_index=(jax_core.ShapedArray((), jnp.int32),) * len(grid),
        linearized_index=jax_core.ShapedArray((), jnp.int32),
        pipeline_state=state_avals)


def check_pipeline(stages: Sequence[internal.PipelineStage]):
  """Runs sanity checks on the pipeline."""
  last_write = collections.defaultdict(lambda: None)
  last_read = collections.defaultdict(lambda: None)
  for i, stage in enumerate(stages):
    for read_idx in stage.get_read_idxs():
      if last_write[read_idx] is None:
        raise ValueError(
            f"Read before write. {stage} attempted to read ref {read_idx}"
            " without a prior stage writing to it.")
      last_read[read_idx] = i
    for write_idx in stage.get_write_idxs():
      if last_write[write_idx] is not None:
        raise ValueError(
            f"Write conflict. {stage} writes to ref {write_idx} but it was"
            f" already written to by stage {stages[last_write[write_idx]]}."
            " The current scheduler only allows one stage to write to each"
            " buffer.")
      last_write[write_idx] = i
  all_idxs = last_write.keys() | last_read.keys()
  for i in all_idxs:
    if last_write[i] > last_read[i]:
      raise ValueError(f"Ref {i} is written to after its final read.")


@functools.partial(jax.tree_util.register_dataclass,
                   data_fields=["stage_counters"],
                   meta_fields=["which_stage_writes", "which_stages_read"])
@dataclasses.dataclass(frozen=True)
class Scoreboard:
  """A scoreboard used to book-keep data dependencies.

  Attributes:
    which_stage_writes: A mapping from buffer index to the stage index that
      writes to it.
    which_stages_read: A mapping from buffer index to the stages that read
      from it.
    stage_counters: A list of length num_stages that tracks the number of times
      each stage has run.
  """
  which_stage_writes: Mapping[internal.BufferIndex, int]
  which_stages_read: Mapping[internal.BufferIndex, Sequence[int]]
  stage_counters: list[jax.Array | int]

  @classmethod
  def create(cls, stages: Sequence[internal.PipelineStage]):
    which_stage_writes = collections.defaultdict(lambda: None)
    which_stage_reads = collections.defaultdict(set)
    stage_counters = [0] * len(stages)
    for i, stage in enumerate(stages):
      for write_idx in stage.get_write_idxs():
        which_stage_writes[write_idx] = i
      for read_idx in stage.get_read_idxs():
        which_stage_reads[read_idx].add(i)
    return cls(which_stage_writes, which_stage_reads, stage_counters)

  def get_stage_counter(self, stage_idx: int) -> jax.Array | int:
    """Returns the current stage counter for the given stage index."""
    return self.stage_counters[stage_idx]

  def get_writing_stage(self, buffer_idx: internal.BufferIndex) -> int:
    """Returns the stage index that writes to the given buffer index."""
    return self.which_stage_writes[buffer_idx]

  def increment_stage_counter(self, stage_idx: int) -> None:
    """Increments the stage counter for the given stage index."""
    self.stage_counters[stage_idx] += 1

  def copy(self) -> "Scoreboard":
    """Returns a deep copy of the scoreboard."""
    new_stage_counters = copy.copy(self.stage_counters)
    return Scoreboard(self.which_stage_writes, self.which_stages_read,
                      new_stage_counters)


@functools.partial(jax.tree_util.register_dataclass,
                   data_fields=["indices"],
                   meta_fields=["grid", "offsets", "dynamic"])
@dataclasses.dataclass(frozen=True)
class GridCarry:
  """Helper class for managing the pipeline grid indices.

  Attributes:
    grid: The size of the grid.
    offsets: A mapping from the stage index to the integer offset from the
      slowest scheduled stage.
    dynamic: Whether grid indices should be calculated dynamically.
    indices: A mapping from offset to the grid indices.
  """
  grid: Sequence[int]
  offsets: Sequence[int]
  dynamic: bool
  indices: Sequence[Sequence[int | jax.Array]]

  @classmethod
  def init(cls, grid, offsets, dynamic=False) -> 'GridCarry':
    max_offset = max(offsets)
    cur_indices = tuple([0] * len(grid))
    indices = [cur_indices]
    for _ in range(1, max_offset + 1):
      next_indices, _ = increment_grid(cur_indices, grid)
      indices.append(next_indices)
      cur_indices = next_indices
    return cls(grid, offsets, dynamic, tuple(indices))

  def next(self) -> "GridCarry":
    next_indices, _ = increment_grid(
        self.indices[-1], self.grid, dynamic=self.dynamic
    )
    new_indices = (*self.indices[1:], next_indices)
    return GridCarry(self.grid, self.offsets, self.dynamic, new_indices)

  def get_indices_for_stage(self, stage_idx: int) -> Sequence[int | jax.Array]:
    return self.indices[self.offsets[stage_idx]]


def check_args_ready(
    stage: internal.PipelineStage,
    scoreboard: Scoreboard,
    new_scoreboard: Scoreboard,
    current_stage_counter: int | jax.Array,
    dynamic=False,
) -> bool | jax.Array:
  """Returns whether all arguments to the stage have already been computed."""
  all_read_stages = []
  for arg_idx in stage.get_read_idxs():
    if stage.properties.is_async_start:
      # Async start stages can start immediately after the preceding
      # stage, so we use new_scoreboard instead of scoreboard.
      arg_stage_idx = new_scoreboard.get_writing_stage(arg_idx)
      arg_stage_ctr = new_scoreboard.get_stage_counter(arg_stage_idx)
    else:
      arg_stage_idx = scoreboard.get_writing_stage(arg_idx)
      arg_stage_ctr = scoreboard.get_stage_counter(arg_stage_idx)
    all_read_stages.append(arg_stage_ctr > current_stage_counter)
  op = jnp.logical_and if dynamic else operator.and_
  args_ready = functools.reduce(op, all_read_stages, True)
  return args_ready


def check_async_done(stage: internal.PipelineStage,
                     scoreboard: Scoreboard,
                     num_itrs: int | jax.Array,
                     current_stage_counter: int | jax.Array,
                     dynamic=False) -> bool | jax.Array:
  """Returns whether the async done stage can run."""
  and_op = jnp.logical_and if dynamic else operator.and_
  # For async done stages, we need to insert delays so that they
  # happen as late as possible.
  # First condition is that there are a full number of async starts
  # in flight.
  max_in_flight = stage.properties.max_in_flight
  can_run = True
  token_read_effs = internal.filter_tokens(
      internal.filter_read_effects(stage.effects))
  read_tokens = {effect.input_index for effect in token_read_effs}
  assert len(read_tokens) == 1, stage.effects
  read_token = tuple(read_tokens)[0]
  async_start_stage_idx = scoreboard.which_stage_writes[read_token]
  async_start_counter = scoreboard.get_stage_counter(
      async_start_stage_idx)
  async_done_counter = current_stage_counter
  min_op = jnp.minimum if dynamic else min
  start_full = (async_start_counter >=
                min_op(async_done_counter + max_in_flight, num_itrs))
  can_run = and_op(can_run, start_full)
  # Second condition - the consumers of this stage's outputs will
  # actually need the results on the next iteration.
  for write_idx in stage.get_write_idxs():
    which_stages_read = scoreboard.which_stages_read[write_idx]
    for read_stage_idx in which_stages_read:
      read_itr = scoreboard.stage_counters[read_stage_idx]
      can_run = and_op(can_run, (current_stage_counter <= read_itr))
  return can_run


def check_async_start(
    stage: internal.PipelineStage,
    scoreboard: Scoreboard,
    current_stage_counter: int | jax.Array,
    dynamic=False,
) -> bool | jax.Array:
  """Returns whether the async start stage can run."""
  token_write_effs = internal.filter_tokens(
      internal.filter_write_effects(stage.effects)
  )
  assert len(token_write_effs) == 1, stage.effects
  token_write_idx = tuple(token_write_effs)[0].input_index
  dependent_stages = scoreboard.which_stages_read[token_write_idx]

  dependents_ready = []
  max_in_flight = stage.properties.max_in_flight
  for dependent_stage_idx in dependent_stages:
    check_itr = scoreboard.stage_counters[dependent_stage_idx]
    # Do not issue more async_starts than max_in_flight.
    dependents_ready.append(
        current_stage_counter < check_itr + max_in_flight)
  op = jnp.logical_and if dynamic else operator.and_
  dependents_ready = functools.reduce(op, dependents_ready, True)
  return dependents_ready


class EvalStageFunc(Protocol):
  def __call__(
      self,
      ctx: PipelineContext,
      stage: internal.PipelineStage,
      args: Sequence[Any],
  ) -> PipelineState:
    ...


def eval_stage(ctx: PipelineContext, stage: internal.PipelineStage, args
               ) -> PipelineState:
  """Evaluates a single stage."""
  flat_ctx = jax.tree.leaves(ctx)
  state_tree = jax.tree.structure(ctx.pipeline_state)
  next_state = jax_core.eval_jaxpr(
      stage.jaxpr.jaxpr, stage.jaxpr.consts, *flat_ctx, *args
  )
  if next_state:
    return jax.tree.unflatten(state_tree, next_state)
  return ctx.pipeline_state


def linearize_stages(stages: Sequence[internal.PipelineStage]
                     ) -> Sequence[internal.PipelineStage]:
  """Computes a linearization of the pipeline stages."""
  linearized_stages = []
  outputs_written = set()
  available_stages = stages
  while available_stages:
    stage_added = False
    new_available_stages = list(available_stages)
    for stage in available_stages:
      if all(read_idx in outputs_written for read_idx in stage.get_read_idxs()):
        linearized_stages.append(stage)
        outputs_written.update(stage.get_write_idxs())
        stage_added = True
        new_available_stages.remove(stage)
    available_stages = new_available_stages
    if not stage_added:
      raise ValueError(
          "Failed to linearize pipeline stages. Could not linearize"
          f" {available_stages=}")
  return linearized_stages


def make_ctx(stage: internal.PipelineStage,
             stage_idx: int,
             scoreboard: Scoreboard,
             pipeline_state: PipelineState,
             grid_carry: GridCarry | None = None,
             grid: Sequence[int] | None = None,
             offset: int | jax.Array = 0) -> PipelineContext:
  del stage
  step = scoreboard.stage_counters[stage_idx] + offset
  if grid_carry is not None:
    loop_index = grid_carry.get_indices_for_stage(stage_idx)
  else:
    loop_index = compute_grid_indices(step, grid)
  return PipelineContext(loop_index=loop_index,
                         linearized_index=step,
                         pipeline_state=pipeline_state)


# TODO(justinfu): Implement a second version that rolls more of the pipeline
# into the loop body to reduce code size.
def static_nd_loop_scheduler(
    nd_loop: internal.NDLoopStruct,
    args: Sequence[Any],
    initial_state: PipelineState | None = None,
    eval_fn: EvalStageFunc | None = None,
):
  """Schedules and emits the pipeline into a single instruction stream.

  This scheduler is static in the sense that most of the control logic is
  implemented in Python and run at JAX tracing time. This reduce scalar
  core pressure as the scoreboarding logic does not have to be computed
  at runtime.
  """
  if eval_fn is None:
    eval_fn = eval_stage

  stages = linearize_stages(nd_loop.stages)
  num_stages = len(stages)
  num_itrs = np.prod(nd_loop.grid)
  check_pipeline(stages)
  scoreboard = Scoreboard.create(stages)

  def can_run_stage(
      stage: internal.PipelineStage,
      scoreboard: Scoreboard,
      new_scoreboard: Scoreboard,
      current_stage_counter: int | jax.Array,
  ) -> bool | jax.Array:
    can_run = True
    # Check args ready.
    can_run = can_run & check_args_ready(
        stage, scoreboard, new_scoreboard, current_stage_counter)
    # Check dependents
    if stage.properties.is_async_start:
      can_run = can_run & check_async_start(
          stage, scoreboard, current_stage_counter,
      )
    if stage.properties.is_async_done:
      can_run = can_run & check_async_done(
          stage, scoreboard, num_itrs, current_stage_counter)
    return can_run

  def compute_offsets(scoreboard: Scoreboard) -> Sequence[int] | None:
    while any(scoreboard.stage_counters[i] < 1 for i in range(num_stages)):
      new_scoreboard = scoreboard.copy()
      for stage_idx, stage in enumerate(stages):
        current_stage_counter = scoreboard.stage_counters[stage_idx]
        can_run = can_run_stage(
            stage, scoreboard, new_scoreboard, current_stage_counter
        )
        if can_run:
          new_scoreboard.increment_stage_counter(stage_idx)
      if scoreboard.stage_counters == new_scoreboard.stage_counters:
        raise ValueError("Scheduling error. No stages ran.")
      scoreboard = new_scoreboard
    min_stage = min(scoreboard.stage_counters)
    offsets = [
        scoreboard.stage_counters[i] - min_stage for i in range(num_stages)
    ]
    if max(offsets) > num_itrs:
      # Bail out, since we won't be running the main loop.
      return None
    return offsets

  # Main loop stage iteration offsets.
  # This is a list of integers containing the number of iterations each
  # stage is ahead of the slowest stage.
  offsets = compute_offsets(scoreboard)

  # Static prologue
  # This runs the pipeline up until the steady state.
  pipeline_state = initial_state
  with jax.named_scope("pipeline_prologue"):
    while any(
        scoreboard.stage_counters[i] < (offsets[i] if offsets else 1)
        for i in range(num_stages)
    ):
      new_scoreboard = scoreboard.copy()
      for stage_idx, stage in enumerate(stages):
        current_stage_counter = scoreboard.stage_counters[stage_idx]
        if offsets:
          can_run = current_stage_counter < offsets[stage_idx]
        else:
          can_run = current_stage_counter < num_itrs
        can_run = can_run & can_run_stage(
            stage, scoreboard, new_scoreboard, current_stage_counter
        )
        if can_run:
          pipeline_state = eval_fn(
              make_ctx(
                  stage, stage_idx, scoreboard, pipeline_state,
                  grid=nd_loop.grid,
              ),
              stage,
              args,
          )
          new_scoreboard.increment_stage_counter(stage_idx)
      if scoreboard.stage_counters == new_scoreboard.stage_counters:
        raise ValueError("Scheduling error. No stages ran.")
      scoreboard = new_scoreboard

  if offsets:
    assert all(
        scoreboard.stage_counters[i] == offsets[i] for i in range(num_stages)
    ), (
        f"Scheduling error. Scoreboard {scoreboard.stage_counters} does not"
        f" match computed offsets {offsets}"
    )

  # Dynamic loop body.
  # This runs the steady state of the pipeline where all stages run with
  # no control flow.
  @jax.named_scope("pipeline_steady_state")
  def loop_body(itr: jax.Array, carry: tuple[PipelineState, GridCarry]):
    pipeline_state, grid_carry = carry
    stages_left = list(stages)
    old_scoreboard = scoreboard.copy()
    while any(stages_left):
      new_scoreboard = old_scoreboard.copy()
      for stage_idx, stage in enumerate(stages_left):
        if stage is None:
          continue
        current_stage_counter = old_scoreboard.stage_counters[stage_idx]
        can_run = can_run_stage(
            stage, old_scoreboard, new_scoreboard, current_stage_counter
        )
        if can_run:
          pipeline_state = eval_fn(
              make_ctx(
                  stage,
                  stage_idx,
                  old_scoreboard,
                  pipeline_state,
                  grid_carry=grid_carry,
                  offset=itr,
              ),
              stage,
              args,
          )
          new_scoreboard.increment_stage_counter(stage_idx)
          stages_left[stage_idx] = None
      old_scoreboard = new_scoreboard
    return (pipeline_state, grid_carry.next())

  num_loop_itrs = int(max(num_itrs - max(scoreboard.stage_counters), 0))
  if offsets:
    grid_carry = GridCarry.init(
        offsets=offsets, grid=nd_loop.grid, dynamic=True)
    init_carry = (pipeline_state, grid_carry)
    final_carry = jax.lax.fori_loop(0, num_loop_itrs, loop_body, init_carry)
    (pipeline_state, _) = final_carry

  # Update the static scoreboard to reflect the fact that each stage ran
  # num_loop_itrs times.
  for stage_idx in range(len(stages)):
    scoreboard.stage_counters[stage_idx] += num_loop_itrs

  # Static epilogue
  with jax.named_scope("pipeline_epilogue"):
    while any(
        scoreboard.stage_counters[i] < num_itrs for i in range(num_stages)
    ):
      new_scoreboard = scoreboard.copy()
      for stage_idx, stage in enumerate(stages):
        current_stage_counter = scoreboard.stage_counters[stage_idx]
        can_run = current_stage_counter < num_itrs
        can_run = can_run & can_run_stage(
            stage, scoreboard, new_scoreboard, current_stage_counter
        )
        if can_run:
          pipeline_state = eval_fn(
              make_ctx(
                  stage, stage_idx, scoreboard, pipeline_state,
                  grid=nd_loop.grid,
              ),
              stage,
              args,
          )
          new_scoreboard.increment_stage_counter(stage_idx)
      if scoreboard.stage_counters == new_scoreboard.stage_counters:
        raise ValueError("Scheduling error. No stages ran.")
      scoreboard = new_scoreboard
