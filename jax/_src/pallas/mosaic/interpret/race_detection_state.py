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

import collections
import dataclasses
import itertools
import logging
import threading
from typing import Literal

from jax._src import source_info_util
from jax._src.pallas.mosaic.interpret import vector_clock as vc

logger = logging.getLogger(__name__)


def _is_empty_slice(slice_or_idx: slice | int):
  if isinstance(slice_or_idx, int) or (slice_or_idx == slice(None)):
    return False

  # NOTE: All slices here will have known size.
  start = int(slice_or_idx.start) if slice_or_idx.start is not None else 0
  stop = int(slice_or_idx.stop)
  return start < stop


def _slices_overlap(slice_or_idx1: slice | int, slice_or_idx2: slice | int):
  if isinstance(slice_or_idx1, int):
    slice_or_idx1 = slice(slice_or_idx1, slice_or_idx1 + 1)
  if isinstance(slice_or_idx2, int):
    slice_or_idx2 = slice(slice_or_idx2, slice_or_idx2 + 1)

  if slice_or_idx1 == slice(None):
    return _is_empty_slice(slice_or_idx2)
  if slice_or_idx2 == slice(None):
    return _is_empty_slice(slice_or_idx1)

  # TODO(jburnim): Handle non-zero steps.
  assert (slice_or_idx1.step == 1) or (slice_or_idx1.step is None)
  assert (slice_or_idx2.step == 1) or (slice_or_idx2.step is None)

  assert slice_or_idx1.start is not None
  assert slice_or_idx1.stop is not None
  assert slice_or_idx2.start is not None
  assert slice_or_idx2.stop is not None

  # NOTE: We are only comparing slices with known stops (and sizes).
  # Do we need to handle zero-length slices?
  return (slice_or_idx1.start <= slice_or_idx2.start < slice_or_idx1.stop) | (
      slice_or_idx2.start <= slice_or_idx1.start < slice_or_idx2.stop
  )


def _ranges_overlap(
    range1: tuple[slice | int, ...], range2: tuple[slice | int, ...]
) -> bool:
  return all(
      _slices_overlap(r1, r2)
      for r1, r2 in itertools.zip_longest(range1, range2, fillvalue=slice(None))
  )


@dataclasses.dataclass
class RaceDetectionState[ThreadKey]:
  on_race: Literal["raise", "warn"] = "raise"

  # buffer_key -> thread_key -> (monotone_chain, unordered)
  # Each entry is (seq, VectorClock, range, source_info).
  reads: dict = dataclasses.field(
      default_factory=lambda: collections.defaultdict(
          lambda: collections.defaultdict(lambda: ([], []))
      )
  )

  # buffer_key -> thread_key -> (monotone_chain, unordered)
  # Each entry is (seq, VectorClock, range, source_info).
  writes: dict = dataclasses.field(
      default_factory=lambda: collections.defaultdict(
          lambda: collections.defaultdict(lambda: ([], []))
      )
  )

  lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

  races_found: bool = False

  next_seq: int = 0

  @staticmethod
  def _format_source_info(source_info) -> str:
    if source_info is not None:
      return source_info_util.summarize(source_info)
    return 'pallas_call'

  @staticmethod
  def _record(by_thread, thread, seq, clock, rnge, source_info):
    chain, unordered = by_thread[thread]
    if not chain or chain[-1][1].lt(clock):
      chain.append((seq, clock, rnge, source_info))
    else:
      unordered.append((seq, clock, rnge, source_info))

  @staticmethod
  def _find_race(by_thread, clock, rnge):
    best = None
    for other_thread, (chain, unordered) in by_thread.items():
      if chain and not chain[-1][1].lt(clock):
        # Binary search for the earliest entry in `chain` not < `clock`.
        # Since `chain` is strictly happens-before-increasing, all earlier
        # entries satisfy `other_clock.lt(clock)` (ordered), and by
        # transitivity no entry at or after `lo` can satisfy `lt(clock)`.
        lo = 0
        hi = len(chain)
        while lo < hi:
          mid = (lo + hi) // 2
          if chain[mid][1].lt(clock):
            lo = mid + 1
          else:
            hi = mid
        for i in range(lo, len(chain)):
          seq, other_clock, other_range, other_source_info = chain[i]
          if best is not None and seq >= best[0]:
            break
          if clock.lt(other_clock):
            continue
          if _ranges_overlap(rnge, other_range):
            best = (
                seq, other_thread, other_clock, other_range, other_source_info
            )
            break
      for seq, other_clock, other_range, other_source_info in unordered:
        if best is not None and seq >= best[0]:
          break
        if other_clock.ordered(clock):
          continue
        if _ranges_overlap(rnge, other_range):
          best = (
              seq, other_thread, other_clock, other_range, other_source_info
          )
          break
    if best is None:
      return None
    return best[1:]

  def check_read(
      self,
      thread: ThreadKey,
      clock: vc.NpVectorClock,
      buffer_key,
      rnge,
      source_info=None,
  ):
    with self.lock:
      conflict = self._find_race(self.writes[buffer_key], clock, rnge)
      seq = self.next_seq
      self.next_seq = seq + 1
      self._record(
          self.reads[buffer_key], thread, seq, clock, rnge, source_info
      )

    if conflict is not None:
      write_thread, write_clock, write_range, write_source_info = conflict
      user_frame = self._format_source_info(source_info)
      write_frame = self._format_source_info(write_source_info)
      # TODO(jburnim): When printing device IDs for reads/writes, distinguish
      # between real device IDs vs. DMA IDs.
      msg = (f'RACE DETECTED\n  read of {buffer_key}[{rnge}] from {thread},'
             f' {user_frame}\n  clock: {clock}\n  write of'
             f' {buffer_key}[{write_range}] from {write_thread},'
             f' {write_frame}\n  clock: {write_clock}\n')
      if self.on_race == "raise":
        raise RuntimeError(msg)
      else:
        logger.warning(msg)
      with self.lock:
        self.races_found = True

  def check_write(
      self,
      thread: ThreadKey,
      clock: vc.NpVectorClock,
      buffer_key,
      rnge,
      source_info=None,
  ):
    with self.lock:
      writes_by_thread = self.writes[buffer_key]
      reads_by_thread = self.reads[buffer_key]
      write_conflict = self._find_race(writes_by_thread, clock, rnge)
      read_conflict = None
      if write_conflict is None or self.on_race != "raise":
        read_conflict = self._find_race(reads_by_thread, clock, rnge)
      seq = self.next_seq
      self.next_seq = seq + 1
      self._record(writes_by_thread, thread, seq, clock, rnge, source_info)

    if write_conflict is not None:
      write_thread, write_clock, write_range, write_source_info = (
          write_conflict
      )
      user_frame = self._format_source_info(source_info)
      write_frame = self._format_source_info(write_source_info)
      # TODO(jburnim): When printing device IDs for reads/writes, distinguish
      # between real device IDs vs. DMA IDs.
      msg = (f'RACE DETECTED\n  write of {buffer_key}[{rnge}] from {thread},'
             f' {user_frame}\n  clock: {clock}\n  write of'
             f' {buffer_key}[{write_range}] from {write_thread},'
             f' {write_frame}\n  clock: {write_clock}\n')
      if self.on_race == "raise":
        raise RuntimeError(msg)
      else:
        logging.warning(msg)
      with self.lock:
        self.races_found = True

    if read_conflict is not None:
      read_thread, read_clock, read_range, read_source_info = read_conflict
      user_frame = self._format_source_info(source_info)
      read_frame = self._format_source_info(read_source_info)
      # TODO(jburnim): When printing device IDs for reads/writes, distinguish
      # between real device IDs vs. DMA IDs.
      msg = (f'RACE DETECTED\n  write of {buffer_key}[{rnge}] from {thread},'
             f' {user_frame}\n  clock: {clock}\n  read of'
             f' {buffer_key}[{read_range}] from {read_thread},'
             f' {read_frame}\n  clock: {read_clock}\n')
      if self.on_race == "raise":
        raise RuntimeError(msg)
      else:
        logging.warning(msg)
      with self.lock:
        self.races_found = True
