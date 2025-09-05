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

"""
Delta debugging repro reduction implementation

See paper: https://www.cs.purdue.edu/homes/xyzhang/fall07/Papers/delta-debugging.pdf
"""

import dataclasses
import logging

from typing import Generic, TypeVar, Union


Candidate = TypeVar("Candidate")

class Repro(Generic[Candidate]):
  """Abstract base class for a reproducer, for delta debugging."""

  def progress_msg(self, state: "State[Candidate]") -> str:
    """Returns a custom progress message for the current state."""
    raise NotImplementedError

  def get_all_candidates(self) -> list[Candidate]:
    raise NotImplementedError

  def get_reduced_repro(self, reduction_candidates: list[Candidate]) -> Union["Repro", None]:
    """Checks if the `reduction_candidates` subset still reproduces the issue.

    Args:
      reduction_candidates: A subset of candidates from `get_all_candidates()`.

    Returns:
      A new, smaller `Repro` object if the candidates still reproduce the issue,
      otherwise `None`.
    """
    raise NotImplementedError


@dataclasses.dataclass
class State(Generic[Candidate]):
  """Delta debugging state.

  Attributes:
    all_candidates: All the reduction candidates.
    granularity: The number of chunks to split candidates into.
    chunk_size: The size of each chunk.
    start_index: The index of the first candidate in the current chunk.
  """
  all_candidates: list[Candidate]
  granularity: int
  chunk_size: int = 0
  start_index: int = 0

  def __post_init__(self):
    self.chunk_size = (len(self.all_candidates) + self.granularity - 1) // self.granularity

  def __repr__(self):
    return f"State(g={self.granularity}, chunk_size={self.chunk_size}/{len(self.all_candidates)}, start={self.start_index})"

  def advance(self, steps: int = 1) -> Union["State[Candidate]", None]:
    # Advances the state by a certain number of steps, possibly decreasing
    # chunk size. Returns None when we are at the end of the search.
    state = self
    while True:
      new_start = state.start_index + steps * state.chunk_size
      if new_start < len(state.all_candidates):
        return dataclasses.replace(state, start_index=new_start)
      # Steps remaining after we move to the next smaller chunk size
      steps = (new_start - len(state.all_candidates)) // state.chunk_size
      # Decrease chunk size
      if state.chunk_size == 1:
        return None
      state = dataclasses.replace(
          state, start_index=0,
          granularity=min(len(state.all_candidates), 2 * state.granularity))

  def select_candidates(self) -> list[Candidate]:
    return self.all_candidates[self.start_index:self.start_index + self.chunk_size]


@dataclasses.dataclass()
class Stats:
  total_steps: int = 0
  steps_since_last_repro: int = 0
  total_size: int = 0  # The sum of the (len(all_candidates) - len(selected_candidates))


def next_smaller_repro(repro: Repro[Candidate],
                       state: State[Candidate],
                       stats: Stats
                       ) -> tuple[Union[Repro[Candidate], None],
                                  State[Candidate],
                                  Stats]:
  """Searches for the next smaller repro by iterating through states.
  Returns:
    A tuple containing:
    * The found smaller repro, or None if the search finished without finding one.
    * The state at which the repro was found (or the last state).
    * Updated stats.
  """
  while True:
    logging.info(f"Repro reducer trying {state} with stats={stats}: {repro.progress_msg(state)}")
    reduction_candidates: list[Candidate] = state.select_candidates()
    if not reduction_candidates:
      return None, state, stats
    stats = dataclasses.replace(
        stats,
        total_size=stats.total_size + (len(state.all_candidates) - len(reduction_candidates)),
        total_steps=stats.total_steps + 1,
        steps_since_last_repro=stats.steps_since_last_repro + 1)
    if (result := repro.get_reduced_repro(reduction_candidates)) is not None:
      logging.info(f"Found smaller repro in {state} with stats={stats}: {repro.progress_msg(state)}")
      logging.info("")
      return result, state, stats
    new_state = state.advance(1)
    if new_state is None:
      return None, state, stats
    state = new_state


def ddmin(r: Repro[Candidate], *,
          stats: Union[Stats, None] = None,
          granularity: int = 2,
          start_index: int = 0,
          ) -> tuple[Repro[Candidate], Stats]:
  """Minimizes a reproducer using the delta debugging algorithm.

  Args:
    r: The initial reproducer.
    stats: Optional stats object to track progress.
    granularity: Initial granularity for splitting candidates.
    start_index: Initial start index for candidates.

  Returns:
    A tuple containing the minimized reproducer and the final stats.
  """
  stats = stats or Stats()
  while True:
    candidates = r.get_all_candidates()
    state = State(all_candidates=candidates, granularity=granularity,
                  start_index=start_index)
    stats = dataclasses.replace(stats, steps_since_last_repro=0)
    new_r, prev_state, new_stats = next_smaller_repro(r, state, stats)
    if new_r is None:
      logging.info(f"ddmin finished with repro {r} with stats={new_stats}")
      return r, new_stats
    r, stats = new_r, new_stats
    granularity = max(prev_state.granularity - 1, 2)
