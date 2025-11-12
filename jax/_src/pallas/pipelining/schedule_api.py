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
"""Internal API for the Pallas pipelining scheduler."""
# mypy: ignore-errors
# pylint: disable=missing-function-docstring
# pylint: disable=g-doc-args
# pytype: disable=wrong-keyword-args
import dataclasses
from typing import Any, Optional, Sequence

import jax
from jax._src import api_util
from jax._src import core as jax_core
from jax._src import linear_util as lu
from jax._src.interpreters import partial_eval as pe
from jax._src.state import types as state_types
from jax._src.pallas.pipelining import schedulers
from jax._src.pallas.pipelining import internal


PipelineContext = schedulers.PipelineContext


def stage(max_in_flight: int):
  """Wrapper for creating a pipeline stage."""
  def wrapper(func) -> SyncStage:
    return SyncStage(func, max_in_flight)
  return wrapper


class SyncStage:
  """Constructs a synchronous pipeline stage."""

  def __init__(self, func, max_in_flight: int):
    self.func = func
    self.max_in_flight = max_in_flight

  def trace(
      self, abstract_refs, state_avals, grid
  ) -> internal.PipelineStage:
    jaxpr, effs = trace_fun(
        self.func, abstract_refs, state_avals, grid
    )
    name = getattr(self.func, "__name__", str(self.func))
    return internal.PipelineStage(
        jaxpr=jaxpr,
        effects=set(effs),
        properties=internal.SchedulingProperties(
            max_in_flight=self.max_in_flight,
            is_async_start=False,
            is_async_done=False,
        ),
        name=name,
    )


class AsyncStage:
  """Constructs an asynchronous pipeline stage."""

  def __init__(self, max_in_flight: int):
    self.start_func = None
    self.end_func = None
    self.max_in_flight = max_in_flight

  def def_start(self, func):
    self.start_func = func
    return self

  def def_end(self, func):
    self.end_func = func
    return self

  def trace(
      self, abstract_refs, state_avals, grid
  ) -> tuple[internal.PipelineStage, internal.PipelineStage]:
    start_jaxpr, start_effs = trace_fun(
        self.start_func, abstract_refs, state_avals, grid
    )
    end_jaxpr, end_effs = trace_fun(
        self.end_func, abstract_refs, state_avals, grid
    )
    token = internal.make_token(self)
    start_effs = {*start_effs, internal.WriteEffect(token)}
    end_effs = {*end_effs, internal.ReadEffect(token)}
    name = getattr(self.start_func, "__name__", str(self.start_func))
    start_stage = internal.PipelineStage(
        jaxpr=start_jaxpr,
        effects=start_effs,
        properties=internal.SchedulingProperties(
            max_in_flight=self.max_in_flight,
            is_async_start=True,
            is_async_done=False,
        ),
        name=name,
    )
    name = getattr(self.end_func, "__name__", str(self.end_func))
    end_stage = internal.PipelineStage(
        jaxpr=end_jaxpr,
        effects=end_effs,
        properties=internal.SchedulingProperties(
            max_in_flight=self.max_in_flight,
            is_async_start=False,
            is_async_done=True,
        ),
        name=name,
    )
    return start_stage, end_stage


Stage = SyncStage | AsyncStage


def trace_fun(
    fun, ref_avals, state_avals, grid
) -> tuple[jax_core.ClosedJaxpr, Sequence[internal.RefEffect]]:
  """Trace a stage body function to a Jaxpr."""
  ctx_aval = PipelineContext.aval_pytree(grid, state_avals)
  num_ctx_avals = len(jax.tree.leaves(ctx_aval))
  flat_avals, in_tree = jax.tree.flatten((ctx_aval, *ref_avals))
  debug_info = api_util.debug_info("trace_fun", fun, flat_avals, {})
  flat_fn, out_tree_thunk = api_util.flatten_fun_nokwargs(
      lu.wrap_init(fun, debug_info=debug_info), in_tree
  )
  del out_tree_thunk
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fn, flat_avals)
  ref_effects = [
      eff for eff in jaxpr.effects if isinstance(eff, state_types.RefEffect)
  ]
  # Subtract off the consts and state_avals, since this is variable per stage.
  n_const = len(consts)
  ref_effects = [
      type(eff)(input_index=eff.input_index - n_const - num_ctx_avals)
      for eff in ref_effects
  ]
  return jax_core.ClosedJaxpr(jaxpr, consts), ref_effects

def apply_ref_filter(
    stages: Sequence[internal.PipelineStage],
    ref_filter: Any,
    grid, state_avals
) -> Sequence[internal.PipelineStage]:
  """Removes any effects belonging to Refs that do not pass the filter."""
  if ref_filter is None:
    return stages
  ctx_aval = PipelineContext.aval_pytree(grid, state_avals)
  num_ctx_avals = len(jax.tree.leaves(ctx_aval))
  new_stages = []
  for stage_ in stages:
    jaxpr = stage_.jaxpr.jaxpr
    ref_effects = stage_.effects
    token_effects = list(internal.filter_tokens(ref_effects))
    refs_to_keep = {
        i - num_ctx_avals
        for i, aval in enumerate(jaxpr.in_avals)
        if ref_filter(aval)
    }
    new_effects = [
        eff for eff in ref_effects if eff.input_index in refs_to_keep
    ] + token_effects
    new_stages.append(dataclasses.replace(stage_, effects=set(new_effects)))
  return new_stages

def convert_accum_effects_to_writes(stages: Sequence[internal.PipelineStage]
                                    ) -> Sequence[internal.PipelineStage]:
  """Replaces all accumulate effects with simple writes."""
  # After tracing, an accumulation such as ref[...] += y
  # will result in both a ReadEffect and a WriteEffect into `ref`.
  new_stages = []
  for stage_ in stages:
    read_effs = internal.filter_read_effects(stage_.effects)
    write_effs = internal.filter_write_effects(stage_.effects)
    new_read_effs = (
        eff
        for eff in read_effs
        if state_types.WriteEffect(eff.input_index) not in write_effs
    )
    effs = (*new_read_effs, *write_effs)
    new_stages.append(dataclasses.replace(stage_, effects=set(effs)))
  return new_stages


def remove_duplicate_writes_between_async_stages(
    stages: Sequence[internal.PipelineStage],
) -> Sequence[internal.PipelineStage]:
  """Removes duplicate writes between the async start and done stages.

  This is done because the scheduler doesn't support multiple writes to
  the same Ref in different stages. We instead write to a token in the
  async_start stage that's read by the async_done and all direct consumers.
  """
  new_stages = []
  for stage_ in stages:
    if stage_.properties.is_async_start:
      start_read_effs = internal.filter_read_effects(stage_.effects)
      start_write_effs = internal.filter_write_effects(stage_.effects)
      write_token = internal.filter_tokens(start_write_effs)
      assert len(write_token) == 1, stage_.effects
      write_token = tuple(write_token)[0]
      read_token = state_types.ReadEffect(write_token.input_index)

      done_stage = [
          x
          for x in stages
          if x.properties.is_async_done and read_token in x.effects
      ]
      assert len(done_stage) == 1
      done_stage = done_stage[0]
      end_write_effs = internal.filter_write_effects(done_stage.effects)
      start_write_effs = start_write_effs - end_write_effs
      start_effs = (*start_read_effs, *start_write_effs)
      new_stages.append(dataclasses.replace(stage_, effects=set(start_effs)))
    else:
      new_stages.append(stage_)
  return new_stages


def thread_token_deps_to_consumers(stages: Sequence[internal.PipelineStage]
                                   ) -> Sequence[internal.PipelineStage]:
  """Threads the async token to consumers of async op.

  This ensures that the async_start op does not start too soon and potentially
  clobber buffers that the consumers are reading from.
  """
  effects = [stage_.effects for stage_ in stages]
  for stage_ in stages:
    if stage_.properties.is_async_done:
      write_tokens = internal.filter_tokens(
          internal.filter_write_effects(stage_.effects)
      )
      read_tokens = internal.filter_tokens(
          internal.filter_read_effects(stage_.effects)
      )
      assert not write_tokens, stage_.effects
      assert len(read_tokens) == 1, stage_.effects
      read_token_effect = tuple(read_tokens)[0]
      write_idxs = stage_.get_write_idxs()
      for i, other_stage in enumerate(stages):
        if any(
            write_idx in other_stage.get_read_idxs() for write_idx in write_idxs
        ):
          effects[i].add(read_token_effect)
  return [dataclasses.replace(stage_, effects=set(effects[i])
                             ) for i, stage_ in enumerate(stages)]


def schedule_pipeline(
    stages: Sequence[Stage],
    grid: Sequence[int],
    args: Sequence[Any],
    ref_filter: Optional[Any] = None,
    initial_state: schedulers.PipelineState | None = None,
    scheduler: schedulers.PipelineScheduler = schedulers.static_nd_loop_scheduler,
    **scheduler_kwargs,
):
  """Schedules stages and emits the code for a pipeline.

  Args:
    stages: A sequence of pipeline stages.
    grid: The loop grid size.
    args: A sequence of arguments to the pipeline. These will be passed
      directly to each stage.
    ref_filter: An optional function to filter out Refs during tracing so
      that they do not affect the pipeline schedule.
    initial_state: An optional pipeline state that will be passed as a
      carry into each stage.
    scheduler: Which scheduling function to use.
    **scheduler_kwargs: Additional arguments to pass to the scheduler.

  Returns:
    A function that can be called with ``args`` and runs the pipeline.
  """
  _, ref_tree = jax.tree.flatten(args)
  def _get_aval(x):
    if hasattr(x, "get_ref_aval"):
      return x.get_ref_aval()
    return jax_core.get_aval(x)
  avals = jax.tree.map(_get_aval, args)

  # Make state avals.
  state_avals = jax.tree.map(_get_aval, initial_state)

  traced_stages = []
  for stage in stages:
    if isinstance(stage, SyncStage):
      traced_stages.append(stage.trace(avals, state_avals, grid))
    elif isinstance(stage, AsyncStage):
      start_stage, end_stage = stage.trace(avals, state_avals, grid)
      traced_stages.append(start_stage)
      traced_stages.append(end_stage)
    else:
      raise ValueError(f"Unsupported stage type: {type(stage)}")

  # Run several "passes" to clean up effects before scheduling.
  traced_stages = apply_ref_filter(traced_stages, ref_filter, grid, state_avals)
  traced_stages = convert_accum_effects_to_writes(traced_stages)
  traced_stages = remove_duplicate_writes_between_async_stages(traced_stages)
  traced_stages = thread_token_deps_to_consumers(traced_stages)

  loop_struct = internal.NDLoopStruct(stages=traced_stages, grid=grid)

  def pipeline(*args):
    flat_args, args_tree = jax.tree.flatten(args)
    if args_tree != ref_tree:
      raise ValueError(
          f"Args tree and ref tree do not match.\n{args_tree=}\n{ref_tree=}"
      )
    scheduler(
        loop_struct,
        args=flat_args,
        initial_state=initial_state,
        **scheduler_kwargs,
    )

  return pipeline
