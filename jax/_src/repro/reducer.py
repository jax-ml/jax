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
import dataclasses
from collections.abc import Sequence
import logging
import pathlib
import re

from typing import Any, Callable, Generic, Type, TypeVar, Union

from jax._src.repro import tracker
from jax._src.repro.tracker import Call
from jax._src.repro import emitter

###
### Delta debugging repro reduction implementation
###
# See paper: https://www.cs.purdue.edu/homes/xyzhang/fall07/Papers/delta-debugging.pdf

Candidate = TypeVar("Candidate")

class DDRepro(Generic[Candidate]):
  """Interface to the reproducer, for delta debugging."""
  def __init__(self, repro_path: pathlib.Path):
    self.repro_path = repro_path

  def get_all_candidates(self) -> list[Candidate]:
    """All the reduction candidates."""
    raise NotImplementedError

  def is_reduced_repro(self, reduce_candidates: list[Candidate]) -> Union["DDRepro", None]:
    """Check if we still have a reproducer after applying some reductions."""
    raise NotImplementedError


@dataclasses.dataclass
class DDState(Generic[Candidate]):
  """Delta debugging state"""
  all_candidates: list[Candidate]
  granularity: int  # Work in chunks of size CEIL(len(all_candidates) / granularity)
  chunk_size: int = 0  # We apply the reduction candidates in sequential chunks
  start: int = 0  # The first reduction candidate to consider
  complements: bool = False  # After we finish the chunks at a given chunk size,
                             # we try the complements

  def __post_init__(self):
    self.chunk_size = (len(self.all_candidates) + self.granularity - 1 ) // self.granularity

  def __repr__(self):
    return f"DDState(g={self.granularity}, chunk_size={self.chunk_size}/{len(self.all_candidates)}, start={self.start})"

  def advance(self, steps: int = 1) -> Union["DDState[Candidate]", None]:
    # Advances the state by a certain number of steps, possibly decreasing
    # chunk size. Returns None when we are at the end of the search.
    state = self
    while True:
      new_start = state.start + steps * state.chunk_size
      if new_start < len(state.all_candidates):
        return dataclasses.replace(state, start=new_start)
      # Steps remaining after we move to the next smaller chunk size
      steps = (new_start - len(state.all_candidates)) // state.chunk_size
      # Disable complements for now, do not seem to help
      # if not state.complements and self.chunk_size < len(self.all_candidates):
      #   state = dataclasses.replace(state, start=0, complements=True)
      #   continue
      # Decrease chunk size
      if state.chunk_size == 1:
        return None
      state = dataclasses.replace(
          state, start=0,
          granularity=min(len(state.all_candidates), 2 * state.granularity),
          complements=False)

  def select_candidates(self) -> list[Candidate]:
    if self.complements:
      return self.all_candidates[:self.start] + self.all_candidates[self.start + self.chunk_size:]
    else:
      return self.all_candidates[self.start:self.start + self.chunk_size]


@dataclasses.dataclass()
class DDStats:
  total_steps: int = 0
  steps_since_last_repro: int = 0
  total_size: int = 0  # The sum of the (len(all_candidates) - len(selected_candidates))


def next_smaller_repro(repro: DDRepro[Candidate],
                       state: DDState[Candidate],
                       stats: DDStats
                       ) -> tuple[DDRepro[Candidate] | None,
                                  DDState[Candidate], DDStats]:
  while True:
    logging.info(f"Repro reducer trying {state} with stats={stats} starting from {repro.repro_path}")
    reduction_candidates: list[Candidate] = state.select_candidates()
    stats = dataclasses.replace(
        stats,
        total_size=stats.total_size + (len(state.all_candidates) - len(reduction_candidates)),
        total_steps=stats.total_steps + 1,
        steps_since_last_repro=stats.steps_since_last_repro + 1)
    if (result := repro.is_reduced_repro(reduction_candidates)) is not None:
      return result, state, stats
    if (state := state.advance(1)) is not None:  # type: ignore
      continue
    return None, state, stats  # type: ignore


def ddmin(r: DDRepro, *,
          stats: DDStats | None = None,
          granularity: int = 2,
          start_index: int = 0,
          ) -> tuple[DDRepro, DDStats]:
  stats = stats or DDStats()
  while True:
    candidates = r.get_all_candidates()
    state = DDState(all_candidates=candidates, granularity=granularity,
                    start=start_index)
    stats = dataclasses.replace(stats, steps_since_last_repro=0)
    if (smaller_repro := next_smaller_repro(r, state, stats))[0] is None:
      logging.info(f"ddmin finished with repro {r} with stats={smaller_repro[1]}")
      return r, smaller_repro[1]  # type: ignore
    r, prev_state, stats = smaller_repro  # type: ignore
    granularity = max(prev_state.granularity - 1, 2)

###

Strategy = Callable[[list[Candidate] | None], emitter.ReduceEmitStrategy]

class DropFunctionCallsStrategy(emitter.ReduceEmitStrategy):
  def __init__(self, reduce_candidates: Sequence[emitter.Call] | None):
    self.all_candidates: list[emitter.Call] = []
    self.reduce_candidates = reduce_candidates
    self.reduce_candidates_ids = {c.id for c in reduce_candidates} if reduce_candidates else set()

  def keep_call(self, c: "Call") -> bool:
    self.all_candidates.append(c)
    if self.reduce_candidates:
      return (c.id not in self.reduce_candidates_ids)
    return True


class DropExpressionsStrategy(emitter.ReduceEmitStrategy):
  def __init__(self, reduce_candidates: list[tuple[emitter.Call, bool, int]] | None):
    self.all_candidates: list[tuple[emitter.Call, bool, int]] = []
    self.reduce_candidates = reduce_candidates
    self.reduce_candidates_set = {(c[0].id, c[1], c[2])
                                  for c in reduce_candidates} if reduce_candidates else set()

  def keep_expression(self, c: "Call", for_args: bool, atom_idx: int, v: Any) -> bool:
    # Do not drop the args for user functions nor the results for the JAX
    # functions, because they are only used to define values, and if the
    # values are not used then there is no real reduction.
    is_jax = tracker.is_primitive(c.func) or not c.func.is_user
    if is_jax == (not for_args):
      return True
    if self.reduce_candidates:
      return (c.id, for_args, atom_idx) not in self.reduce_candidates_set
    self.all_candidates.append((c, for_args, atom_idx))
    return True


class Repro(DDRepro[Candidate]):
  def __init__(self,
               repro_path: pathlib.Path,
               repro_source: str,
               expect_error: tuple[Type, str],
               strategy: Strategy,
               all_candidates: list[Candidate],
               collector: emitter.collector,
               ):
    super().__init__(repro_path)
    self.repro_source = repro_source
    self.expect_error = expect_error
    self.all_candidates = all_candidates
    self.collector = collector
    self.strategy = strategy

  def __repr__(self):
    return f"Repro(path={self.repro_path}, nr_candidates={len(self.all_candidates)})"

  @staticmethod
  def make(repro_path: pathlib.Path,
           repro_source: str,
           expect_error: tuple[Type, str],
           strategy: Strategy) -> "Repro":
    res: tuple[emitter.collector, list[Candidate]] = \
      Repro.check_repro(repro_source, expect_error, strategy,  # type: ignore
                        repro_path=repro_path,
                        raise_on_failure=True)
    col, all_candidates = res
    return Repro(repro_path, repro_source, expect_error, strategy, all_candidates, col)

  def get_all_candidates(self) -> list[Candidate]:
    return self.all_candidates

  def is_reduced_repro(self, reduce_candidates: list[Candidate]) -> Union["Repro", None]:
    assert reduce_candidates
    drop_emit_config = self.strategy(reduce_candidates)
    new_source = self.collector.to_source(strategy=drop_emit_config)

    res: str | tuple[emitter.collector, list[Candidate]] = \
      Repro.check_repro(new_source, self.expect_error, self.strategy, raise_on_failure=False)
    if isinstance(res, str):
      logging.info(f"Not a repro: {res}")
      return None

    col, all_candidates = res
    repro_path = emitter.save(new_source)
    logging.info(f"** Found a repro with {len(all_candidates)} candidates at {repro_path}.")

    assert len(all_candidates) < len(self.all_candidates), (len(all_candidates), len(self.all_candidates))
    return Repro(repro_path, new_source, self.expect_error, self.strategy,
                 all_candidates, col)


  @staticmethod
  def check_repro(repro_source: str,
                  expect_error: tuple[Type, str],
                  strategy: Strategy, *,
                  repro_path: pathlib.Path = pathlib.Path("<memory>"),
                  raise_on_failure: bool = True) -> str | tuple[emitter.collector, list[Candidate]]:
    # Don't need to save reproducers on error; there will be plenty of those
    col = emitter.collector(lambda: emitter.load(repro_path, repro_source)(),
                            allow_save_on_error=False)
    collect_emit_config = strategy(None)
    try:
      col()
    except Exception as e:
      if isinstance(e, expect_error[0]) and re.search(expect_error[1], str(e)):
        col.to_source(strategy=collect_emit_config)  # Populate collect_emit_config
        return col, collect_emit_config.all_candidates
      else:
        msg = f"Not a reproducer, it raises: {e} or {type(e)}."
        if raise_on_failure:
          raise ValueError(msg) from e
        return msg
    else:
      msg = "Not a reproducer, it returns successfully."
      if raise_on_failure:
        raise ValueError(msg)
      return msg
