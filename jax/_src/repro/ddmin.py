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

  all_candidates: list[Candidate]

  def progress_msg(self, state: "State[Candidate]") -> str:
    """Returns a custom progress message for the current state."""
    raise NotImplementedError

  def get_reduced_repro(self, reduction_candidates: list[Candidate]) -> Union["Repro", None]:
    """Checks if the `reduction_candidates` subset still reproduces the issue.

    Args:
      reduction_candidates: A subset of candidates from `all_candidates`.

    Returns:
      A new, smaller `Repro` object if the candidates still reproduce the issue,
      otherwise `None`.
    """
    raise NotImplementedError


@dataclasses.dataclass
class State(Generic[Candidate]):
  """Delta debugging state.

  Attributes:
    all_candidates: All the reduction candidates, as returned by `Repro.all_candidates`.
    chunk_size: The size of each chunk of reductions candidates to try.
    start_index: The index in `all_candidates` of the first candidate in
      the current chunk, after considering the `offset`.
  """
  all_candidates: list[Candidate]
  chunk_size: int = 0
  start_index: int = 0

  def __repr__(self):
    return f"State(chunk_size={self.chunk_size}/{len(self.all_candidates)}, start={self.start_index})"

  def advance(self) -> Union["State[Candidate]", None]:
    # Advances the state possibly decreasing
    # chunk size. Returns None when we are at the end of the search.
    new_start = self.start_index + self.chunk_size
    if new_start < len(self.all_candidates):
      return dataclasses.replace(self, start_index=new_start)
    # Decrease chunk size
    if self.chunk_size == 1:
      return None
    next_chunk_size = self.chunk_size // 2
    if next_chunk_size == 2:  # at this point, accelerate the search
      next_chunk_size = 1
    return State(all_candidates=self.all_candidates, chunk_size=next_chunk_size,
                 start_index=0)

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
    assert reduction_candidates, (state, len(repro.all_candidates))
    stats = dataclasses.replace(
        stats,
        total_size=stats.total_size + (len(state.all_candidates) - len(reduction_candidates)),
        total_steps=stats.total_steps + 1,
        steps_since_last_repro=stats.steps_since_last_repro + 1)
    if (result := repro.get_reduced_repro(reduction_candidates)) is not None:
      logging.info(f"Found smaller repro in {state} with stats={stats}: {repro.progress_msg(state)}")
      logging.info("")
      return result, state, stats
    new_state = state.advance()
    if new_state is None:
      return None, state, stats
    state = new_state


def ddmin(r: Repro[Candidate], *,
          chunk_size: int,
          stats: Union[Stats, None] = None,
          start_index: int = 0,
          offset: int = 0,
          ) -> tuple[Repro[Candidate], Stats]:
  """Minimizes a reproducer using the delta debugging algorithm.

  Args:
    r: The initial reproducer.
    chunk_size: the size of a chunk of candidates to try to reduce.
    stats: Optional stats object to track progress.
    start_index: See `State.start_index`.
    offset: See `State.offset`.

  Returns:
    A tuple containing the minimized reproducer and the final stats.
  """
  stats = stats or Stats()
  if start_index >= len(r.all_candidates):
    raise ValueError(f"start_index must be within range [0, {len(r.all_candidates)})")
  if chunk_size > len(r.all_candidates):
    raise ValueError(f"chunk_size must be within range [0, {len(r.all_candidates)}]")

  while True:
    candidates = r.all_candidates
    if not candidates:
      logging.info(f"ddmin finished with repro {r} with stats={stats}; no more candidates.")
      return r, stats
    # Now we know that the state contains some reduction candidates
    state = State(all_candidates=candidates, chunk_size=chunk_size, start_index=start_index)
    stats = dataclasses.replace(stats, steps_since_last_repro=0)
    new_r, new_r_state, new_stats = next_smaller_repro(r, state, stats)
    if new_r is None:
      logging.info(f"ddmin finished with repro {r} with stats={new_stats}")
      return r, new_stats
    r, stats = new_r, new_stats
    # When we find a repro after reducing a given chunk, it is often the case that
    # the new `r.all_candidates` is like the previous one, with the given chunk removed.
    # In that case, we want to continue with the same `start_index` to avoid trying
    # out the same chunks again. In one experiment this reduced the number of steps
    # from 5900 to 700 (and from 14 minutes to 3 minutes).
    # However, this can lead to sub-optimal results. E.g., in the above-mentioned
    # experiment, the final repro was 280 vs. 263 without this optimization.
    nr_candidates = len(r.all_candidates)
    chunk_size = min(new_r_state.chunk_size, nr_candidates)
    start_index = min(new_r_state.start_index, nr_candidates - chunk_size)
