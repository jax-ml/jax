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
import math
import threading
from typing import Any

from jax._src import source_info_util
from jax._src.pallas.mosaic.interpret import vector_clock as vc
import numpy as np


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


def _shaped_ranges_overlap(range1: Any, range2: Any) -> bool:
  if getattr(range1, "layout", None) is not None or getattr(range2, "layout", None) is not None:
    cols1 = range1.shape[1] if len(range1.shape) > 1 else range1.shape[0]
    cols2 = range2.shape[1] if len(range2.shape) > 1 else range2.shape[0]
    extent = max(range1.offset + cols1, range2.offset + cols2)
    r1 = np.zeros((128, extent), dtype=np.int8)
    if range1.indices:
      r1[:, range1.offset : range1.offset + cols1][range1.indices] += 1
    else:
      r1[:, range1.offset : range1.offset + cols1] += 1

    r2 = np.zeros((128, extent), dtype=np.int8)
    if range2.indices:
      r2[:, range2.offset : range2.offset + cols2][range2.indices] += 1
    else:
      r2[:, range2.offset : range2.offset + cols2] += 1

    return bool(np.any((r1 + r2) == 2))

  # range1
  range1_itemsize = np.dtype(range1.dtype).itemsize
  range1_byte_len = math.prod(range1.shape) * range1_itemsize
  range1_byte_slice = slice(range1.offset, range1.offset + range1_byte_len, 1)
  range1_word = np.ones((range1_itemsize,), dtype=np.int8).view(range1.dtype).item()

  range2_itemsize = np.dtype(range2.dtype).itemsize
  range2_byte_len = math.prod(range2.shape) * range2_itemsize
  range2_byte_slice = slice(range2.offset, range2.offset + range2_byte_len, 1)
  range2_word = np.ones((range2_itemsize,), dtype=np.int8).view(range2.dtype).item()

  extent = max(range1_byte_slice.stop, range2_byte_slice.stop)

  r1 = np.zeros((extent,), dtype=np.int8)
  a = np.full(range1.shape, range1_word)
  a_view_of_r = r1[range1_byte_slice].view(range1.dtype).reshape(range1.shape, copy=False)
  if range1.indices:
    a_view_of_r[range1.indices] = a_view_of_r[range1.indices] + a[range1.indices]
  else:
    a_view_of_r[:] = a_view_of_r + a

  r2 = np.zeros((extent,), dtype=np.int8)
  b = np.full(range2.shape, range2_word)
  b_view_of_r = r2[range2_byte_slice].view(range2.dtype).reshape(range2.shape, copy=False)
  if range2.indices:
    b_view_of_r[range2.indices] = b_view_of_r[range2.indices] + b[range2.indices]
  else:
    b_view_of_r[:] = b_view_of_r + b

  r = r1 + r2

  if np.any(r == 2):
    return True
  return False


def _ranges_overlap(
    range1: Any | tuple[slice | int, ...], range2: Any | tuple[slice | int, ...]
) -> bool:
  if isinstance(range1, tuple) and isinstance(range2, tuple):
    return all(
        _slices_overlap(r1, r2)
        for r1, r2 in itertools.zip_longest(range1, range2, fillvalue=slice(None))
    )
  elif not isinstance(range1, tuple) and not isinstance(range2, tuple):
    return _shaped_ranges_overlap(range1, range2)
  else:
    raise ValueError(f'Either both allocations should be refunion or neither: {range1}, {range2}')


@dataclasses.dataclass
class RaceDetectionState[ThreadKey]:
  # TODO(nrink): Remove this field; it seems to be unused.
  num_cores: int

  # (memory_space, buffer_id, thread_key) -> [(device_id, local_core_id, VectorClock, range)]
  reads: dict = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list)
  )

  # (memory_space, buffer_id, thread_key) -> [(device_id, local_core_id, VectorClock, range)]
  writes: dict = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list)
  )

  lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

  races_found: bool = False

  def check_read(
      self,
      thread: ThreadKey,
      clock: vc.NpVectorClock,
      buffer_key,
      rnge,
      source_info=None,
  ):

    if source_info is not None:
      user_frame = source_info_util.summarize(source_info)
    else:
      user_frame = 'pallas_call'

    with self.lock:
      writes = self.writes[buffer_key]
      num_writes = len(writes)
      self.reads[buffer_key].append(
          (thread, clock, rnge, user_frame)
      )

    for i in range(num_writes):
      (
          write_thread,
          write_clock,
          write_range,
          write_frame,
      ) = writes[i]
      if write_clock.ordered(clock):
        continue
      if not _ranges_overlap(rnge, write_range):
        continue
      # TODO(jburnim): When printing device IDs for reads/writes, distinguish
      # between real device IDs vs. DMA IDs.
      print(
          f'RACE DETECTED\n  read of {buffer_key}[{rnge}] from {thread},'
          f' {user_frame}\n  clock: {clock}\n  write of'
          f' {buffer_key}[{write_range}] from {write_thread},'
          f' {write_frame}\n  clock: {write_clock}\n'
      )
      with self.lock:
        self.races_found = True
      return

  def check_write(
      self,
      thread: ThreadKey,
      clock: vc.NpVectorClock,
      buffer_key,
      rnge,
      source_info=None,
  ):

    if source_info is not None:
      user_frame = source_info_util.summarize(source_info)
    else:
      user_frame = 'pallas_call'

    with self.lock:
      writes = self.writes[buffer_key]
      reads = self.reads[buffer_key]
      num_writes = len(writes)
      num_reads = len(reads)
      self.writes[buffer_key].append((thread, clock, rnge, user_frame))

    # TODO(jburnim): For performance, we should also probably remove any
    # conflicting reads and writes that happened-before the current write.

    for i in range(num_writes):
      (
          write_thread,
          write_clock,
          write_range,
          write_frame,
      ) = writes[i]
      if write_clock.ordered(clock):
        continue
      if not _ranges_overlap(rnge, write_range):
        continue
      # TODO(jburnim): When printing device IDs for reads/writes, distinguish
      # between real device IDs vs. DMA IDs.
      print(
          f'RACE DETECTED\n  write of {buffer_key}[{rnge}] from {thread},'
          f' {user_frame}\n  clock: {clock}\n  write of'
          f' {buffer_key}[{write_range}] from {write_thread},'
          f' {write_frame}\n  clock: {write_clock}\n'
      )
      with self.lock:
        self.races_found = True
      break

    for i in range(num_reads):
      read_thread, read_clock, read_range, read_frame = (
          reads[i]
      )
      if read_clock.ordered(clock):
        continue
      if not _ranges_overlap(rnge, read_range):
        continue
      # TODO(jburnim): When printing device IDs for reads/writes, distinguish
      # between real device IDs vs. DMA IDs.
      print(
          f'RACE DETECTED\n  write of {buffer_key}[{rnge}] from {thread},'
          f' {user_frame}\n  clock: {clock}\n  read of'
          f' {buffer_key}[{read_range}] from {read_thread},'
          f' {read_frame}\n  clock: {read_clock}\n'
      )
      with self.lock:
        self.races_found = True
      return
