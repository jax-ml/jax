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
import threading

from jax._src import source_info_util
from jax._src.pallas.mosaic.interpret import vector_clock as vc


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
class RaceDetectionState:
  num_cores: int

  # (memory_space, buffer_id, device_id, local_core_id) -> [(device_id, local_core_id, VectorClock, range)]
  reads: dict = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list)
  )

  # (memory_space, buffer_id, device_id, local_core_id) -> [(device_id, local_core_id, VectorClock, range)]
  writes: dict = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list)
  )

  lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

  races_found: bool = False

  def check_read(
      self, device_id, local_core_id, clock, buffer_key, rnge, source_info=None
  ):
    if source_info is not None:
      user_frame = source_info_util.summarize(source_info)
    else:
      user_frame = 'pallas_call'

    with self.lock:
      writes = self.writes[buffer_key]
      num_writes = len(writes)
      self.reads[buffer_key].append(
          (device_id, local_core_id, clock, rnge, user_frame)
      )

    for i in range(num_writes):
      (
          write_device_id,
          write_local_core_id,
          write_clock,
          write_range,
          write_frame,
      ) = writes[i]
      if vc.ordered(write_clock, clock):
        continue
      if not _ranges_overlap(rnge, write_range):
        continue
      # TODO(jburnim): When printing device IDs for reads/writes, distinguish
      # between real device IDs vs. DMA IDs.
      print(
          f'RACE DETECTED\n  read of {buffer_key}[{rnge}] from {device_id},'
          f' {local_core_id}, {user_frame}\n  clock: {clock}\n  write of'
          f' {buffer_key}[{write_range}] from {write_device_id},'
          f' {write_local_core_id} {write_frame}\n  clock: {write_clock}\n'
      )
      with self.lock:
        self.races_found = True
      return

  def check_write(
      self, device_id, local_core_id, clock, buffer_key, rnge, source_info=None
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
      self.writes[buffer_key].append((device_id, local_core_id, clock, rnge, user_frame))

    # TODO(jburnim): For performance, we should also probably remove any
    # conflicting reads and writes that happened-before the current write.

    for i in range(num_writes):
      (
          write_device_id,
          write_local_core_id,
          write_clock,
          write_range,
          write_frame,
      ) = writes[i]
      if vc.ordered(write_clock, clock):
        continue
      if not _ranges_overlap(rnge, write_range):
        continue
      # TODO(jburnim): When printing device IDs for reads/writes, distinguish
      # between real device IDs vs. DMA IDs.
      print(
          f'RACE DETECTED\n  write of {buffer_key}[{rnge}] from {device_id},'
          f' {local_core_id}, {user_frame}\n  clock: {clock}\n  write of'
          f' {buffer_key}[{write_range}] from {write_device_id},'
          f' {write_local_core_id}, {write_frame}\n  clock: {write_clock}\n'
      )
      with self.lock:
        self.races_found = True
      break

    for i in range(num_reads):
      read_device_id, read_local_core_id, read_clock, read_range, read_frame = (
          reads[i]
      )
      if vc.ordered(read_clock, clock):
        continue
      if not _ranges_overlap(rnge, read_range):
        continue
      # TODO(jburnim): When printing device IDs for reads/writes, distinguish
      # between real device IDs vs. DMA IDs.
      print(
          f'RACE DETECTED\n  write of {buffer_key}[{rnge}] from {device_id},'
          f' {local_core_id}, {user_frame}\n  clock: {clock}\n  read of'
          f' {buffer_key}[{read_range}] from {read_device_id},'
          f' {read_local_core_id}, {read_frame}\n  clock: {read_clock}\n'
      )
      with self.lock:
        self.races_found = True
      return
