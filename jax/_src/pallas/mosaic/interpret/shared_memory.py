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

from __future__ import annotations

import collections
from collections.abc import Callable, Sequence
import contextlib
import dataclasses
import enum
import gc
import itertools
import math
import threading
from typing import Any

import jax
from jax._src import source_info_util
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic import core as mosaic_core
from jax._src.pallas.mosaic.interpret import vector_clock as vc
from jax._src.pallas.mosaic.interpret.race_detection_state import RaceDetectionState
from jax._src.typing import Array
import numpy as np


class Semaphore:

  def __init__(
      self,
      shared_memory: SharedMemory,
      semaphore_id: int,
  ):
    self.shared_memory = shared_memory
    self.id: int = semaphore_id

    # TODO(jburnim): Use one Condition variable per device.  (Which will be
    # easier to do when we're using single integer device IDs.)
    self.cv = threading.Condition()

    self.count_by_core = np.zeros(self.shared_memory.num_cores, dtype=np.int32)

    if self.shared_memory.detect_races:
      # We associate a vector clock with each count in self.counts.  Whenever
      # self.count_by_core[i] is signaled, self.clocks[i] is updated with the
      # vector clock of the signaling core.  Whenever core i successfully waits
      # on self.count_by_core[i], the vector clock of core i is updated with
      # self.clocks[i].
      #
      # TODO(jburnim): Model happens-before more precisely for the case where
      # semaphores are over-signaled.
      self.clocks: list[vc.VectorClock | None] = [
          None
      ] * self.shared_memory.num_cores

  @property
  def num_cores(self) -> int:
    return self.shared_memory.num_cores

  @property
  def detect_races(self) -> bool:
    return self.shared_memory.detect_races

  @property
  def dma_execution_mode(self) -> str:
    return self.shared_memory.dma_execution_mode

  def get_random_virtual_device_id(self) -> int:
    # Virtual device IDs are needed for DMA semaphores. Conceptually, each DMA
    # runs on its own, independent device. Representing this precisely would
    # require vector clocks to have sizes linear in the number of DMAs.
    #
    # Instead, we use approximate vector clocks of fixed size. We assign each
    # DMA a virtual core ID in the range
    #
    #   [num_devices*num_cores_per_device, shared_memory.vector_clock_size - 1],
    #
    # and each operation of a DMA increments the corresponding coordinate in its
    # vector clock. (So the "virtual" part of a vector clock is effectively
    # counting, for each virtual core, the number of DMAs that happened-before
    # the vector clock and were assigned to that virtual core.)
    #
    # If two approximate clocks are unordered, then their corresponding events
    # are not ordered by the happens-before relation. So this approximation will
    # not introduce any false positives in detecting data races. But we may fail
    # to detect some true data races because there can be cases where two
    # approximate clocks are ordered, and we will treat the corresponding events
    # as ordered by the happens-before relation, but the corresponding events
    # are not actually ordered.
    return np.random.randint(
        self.num_cores, self.shared_memory.vector_clock_size
    )

  def signal(self, inc, global_core_id, clock):
    """Signal the semaphore on `(device_id, core_id)` by `inc`.

    Args:
      inc: A positive integer.  The amount by which to increment the semaphore
        on the target device.
      global_core_id: The ID of the target core.
      clock: The vector clock of the signaling device at the time of the signal.
    """
    global_core_id = int(global_core_id)
    with self.cv:
      self.count_by_core[global_core_id] += inc
      if self.shared_memory.detect_races:
        if self.clocks[global_core_id] is None:
          self.clocks[global_core_id] = vc.copy_vector_clock(clock)
        else:
          vc.update_vector_clock(self.clocks[global_core_id], clock)
      self.cv.notify_all()

  def read(self, global_core_id):
    with self.cv:
      return self.count_by_core[global_core_id]

  def wait(self, value, global_core_id, *, is_dma=False):
    global_core_id = int(global_core_id)

    # TODO(jburnim):
    #  - If the count is larger than value, raise an error?
    #  - If the count is equal to value, but there DMAs waiting to signal us,
    #    raise an error?

    # Simple implementation for non-DMA semaphores.
    clock = None
    if not is_dma or (self.dma_execution_mode == 'eager'):
      with self.cv:
        while self.count_by_core[global_core_id] < value:
          self.cv.wait()
        self.count_by_core[global_core_id] -= value
        if self.detect_races:
          assert self.clocks[global_core_id] is not None
          clock = vc.copy_vector_clock(self.clocks[global_core_id])
      if self.detect_races:
        with self.shared_memory.lock:
          assert clock is not None
          vc.update_vector_clock(
              self.shared_memory.clocks[global_core_id], clock
          )
      return

    # For DMA semaphores (when shared_memory.dma_execution_mode=='on_wait'),
    # while our count is not large enough we will select and partially execute
    # pending DMAs until our count is large enough.
    #
    # This approach will tend to run DMAs as late as possible, as well as
    # out-of-order.  This approach also lets us avoid the complexity of spinning
    # up separate threads to handle executing DMAs.
    while True:
      clock = None
      with self.cv:
        if self.count_by_core[global_core_id] >= value:
          self.count_by_core[global_core_id] -= value
          if self.detect_races:
            assert self.clocks[global_core_id] is not None
            clock = vc.copy_vector_clock(self.clocks[global_core_id])
          else:
            return
      if clock is not None:
        with self.shared_memory.lock:
          vc.update_vector_clock(
              self.shared_memory.clocks[global_core_id], clock
          )
        return

      with self.shared_memory.lock:
        dma_queue = self.shared_memory.dmas_by_sem[self.id]
        if len(dma_queue) > 0:
          dma = dma_queue.pop()
        else:
          continue

      # Only execute the DMA as far as necessary to signal us.
      assert (dma.src_sem is self) or (dma.dst_sem is self)
      with dma.lock:
        if dma.virtual_device_id is None:
          dma.virtual_device_id = self.get_random_virtual_device_id()

        if dma.state == DmaState.STARTED:
          # Do the read.
          if self.detect_races:
            vc.inc_vector_clock(dma.clock, dma.virtual_device_id)
          dma.data = get(
              dma.src_device_id,
              dma.src_local_core_id,
              dma.src_memory_space,
              dma.src_buffer_id,
              dma.src_transforms,
              clock=vc.copy_vector_clock(dma.clock),
              src_device_id=dma.id,
              src_local_core_id=0,
              source_info=dma.source_info,
          )
          if self.detect_races:
            vc.inc_vector_clock(dma.clock, dma.virtual_device_id)
          if dma.src_sem is not None:
            data_size = dma.data.itemsize * dma.data.size
            dma.src_sem.signal(
                data_size,
                global_core_id=self.shared_memory.get_global_core_id(
                    dma.src_device_id, dma.src_local_core_id
                ),
                clock=dma.clock,
            )
          dma.state = DmaState.READ

        if dma.src_sem is self:
          # We were only waiting for the DMA read (i.e., we're the send
          # semaphore), so leave the DMA write for later.
          continue
        assert dma.state == DmaState.READ

        # Do the write.
        assert dma.dst_sem is self
        if self.detect_races:
          vc.inc_vector_clock(dma.clock, dma.virtual_device_id)
        store(
            dma.dst_device_id,
            dma.dst_local_core_id,
            dma.dst_memory_space,
            dma.dst_buffer_id,
            dma.dst_transforms,
            dma.data,
            clock=vc.copy_vector_clock(dma.clock),
            src_device_id=dma.id,
            src_local_core_id=0,
            source_info=dma.source_info,
        )
        if self.detect_races:
          vc.inc_vector_clock(dma.clock, dma.virtual_device_id)
        assert type(dma.data) is np.ndarray
        data_size = dma.data.itemsize * dma.data.size
        dma.dst_sem.signal(
            data_size,
            global_core_id=self.shared_memory.get_global_core_id(
                dma.dst_device_id, dma.dst_local_core_id
            ),
            clock=dma.clock,
        )

        dma.data = None
        dma.state = DmaState.COMPLETED


class DmaState(enum.Enum):
  STARTED = 0
  READ = 1
  COMPLETED = 2


@dataclasses.dataclass
class DMA:
  id: int

  src_device_id: int
  src_local_core_id: int
  src_memory_space: int
  src_buffer_id: int
  src_transforms: tuple[Any, ...]
  dst_device_id: int
  dst_local_core_id: int
  dst_memory_space: int
  dst_buffer_id: int
  dst_transforms: tuple[Any, ...]
  src_sem: Semaphore
  dst_sem: Semaphore

  clock: vc.VectorClock

  source_info: source_info_util.SourceInfo | None = None

  state: DmaState = DmaState.STARTED
  data: np.ndarray | None = None
  virtual_device_id: int | None = None
  lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


@dataclasses.dataclass
class Buffer:
  content: np.ndarray
  _: dataclasses.KW_ONLY
  ref_count: int = 1

  def decrease_ref_count(self):
    # We should never decrease the `ref_count` to below zero.
    assert self.ref_count > 0
    self.ref_count -= 1

  def has_zero_ref_count(self) -> bool:
    return self.ref_count == 0

  def size(self) -> int:
    return self.content.itemsize * self.content.size


@dataclasses.dataclass
class SharedMemory:
  num_devices: int
  num_cores_per_device: int
  out_of_bounds_reads: str
  dma_execution_mode: str
  detect_races: bool
  races: RaceDetectionState | None
  vector_clock_size: int | None

  clocks: list[vc.VectorClock]
  barrier: threading.Barrier
  clean_up_barrier: threading.Barrier

  # dtype -> value
  uninitialized_value_callback: Callable[[Any], Any]

  # (memory_space, buffer_id, device_id, local_core_id) -> NumPy array
  mem: dict[tuple[str, int, int, int], Buffer] = dataclasses.field(
      default_factory=dict
  )

  # semaphore_id -> Semaphore
  sem: dict[int, Semaphore] = dataclasses.field(default_factory=dict)

  # (semaphore_id, device_id)
  #   -> list of DMAs that will signal the semaphore on the given device
  # TODO(jburnim): Fix uses of `dmas_by_sem` to align with the two lines of
  # documentation above, i.e. index `dmas_by_sem` with
  # `(semaphore_id, device_id)` (currently indexed with `semaphore_id only).
  dmas_by_sem: dict[int, list[DMA]] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list)
  )

  lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

  # (device_id, local_core_id) -> next buffer ID
  next_buffer_id: dict[tuple[int, int], int] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(lambda: 100)
  )
  # global_core_id -> next semaphore ID
  next_semaphore_id: dict[int, int] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(lambda: 2000)
  )

  next_dma_id: int = 100

  deallocated_bytes: int = 0

  # (device_id, local_core_id) -> [(grid_index, [range])]
  output_ranges: dict[tuple[int, int], list] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list)
  )

  @property
  def num_cores(self) -> int:
    return self.num_devices * self.num_cores_per_device

  def get_global_core_id(self, device_id: int, local_core_id: int) -> int:
    """Computes the global core ID from the given device and local core ID."""
    return device_id * self.num_cores_per_device + local_core_id

  def get_global_core_ids(self, device_id: int) -> Sequence[int]:
    """Computes the global core IDs for all cores in the given device."""
    return tuple(
        self.get_global_core_id(device_id, core_id)
        for core_id in range(self.num_cores_per_device)
    )


# TODO(jburnim): Do we want to support multiple instances of SharedMemory?
# Maybe for running multiple distinct interpreted computations in parallel?
_shared_memory: SharedMemory | None = None


def get_shared_memory() -> SharedMemory:
  assert _shared_memory is not None
  return _shared_memory

def reset_shared_memory():
  global _shared_memory
  _shared_memory = None

def is_shared_memory_initialized() -> bool:
  return _shared_memory is not None

def set_shared_memory(shared_memory: SharedMemory):
  global _shared_memory
  assert _shared_memory is None
  _shared_memory = shared_memory

def _allocate_buffer(
    device_id: Array,
    local_core_id: Array | None,
    memory_space: Array,
    val: Array,
):
  """Allocates a memory buffer on the device with id `device_id` and core with id `local_core_id`.

  Args:
    device_id: Singleton array holding the device id where the buffer will be
      allocated.
    local_core_id: None or singleton array holding the core id where the buffer
      will be allocated. If None, a buffer will be allocated on each cores on
      the device.
    memory_space: Singleton array indicating the memory space to allocate the
      buffer in. If the corresponding memory space is "any" (i.e. HBM), at most
      one buffer will be allocated and it will belong to (local) core id 0.
    val: Array of values to initialize the allocated buffer with.

  Returns:
    Integer id for the allocated buffer.
  """
  device_id = int(device_id)
  memory_space_str = TPU_MEMORY_SPACE_NAMES[int(memory_space)]
  del memory_space
  val = np.array(val)

  shared_memory = get_shared_memory()

  if local_core_id is None:
    local_core_id_int = 0
    local_core_ids = tuple(range(shared_memory.num_cores_per_device))
  else:
    local_core_id_int = int(local_core_id)
    local_core_ids = (local_core_id_int,)
  del local_core_id

  local_core_id_to_buffer_id: dict[int, int] = {}
  with shared_memory.lock:
    for lci in local_core_ids:
      buffer_id = shared_memory.next_buffer_id[(device_id, lci)]
      shared_memory.next_buffer_id[(device_id, lci)] = buffer_id + 1
      if memory_space_str in ['any', 'hbm']:
        # If allocating in HBM, only actually allocate a buffer once.
        # The first local core (i.e. thread) that gets here allocates the
        # buffer, but the buffer is still keyed in the shared memory with core
        # id 0. However, since the buffer is shared across all cores, we
        # initialize the buffer's `ref_count` with the number of cores per
        # device. This ensures that the buffer is not deallocated until all
        # cores have exited the scope of the allocation (e.g. have exited the
        # body of a `run_scoped`).
        key = (memory_space_str, buffer_id, device_id, 0)
        if key not in shared_memory.mem:
          shared_memory.mem[key] = Buffer(
              val, ref_count=shared_memory.num_cores_per_device
          )
      else:
        # If we are allocating more than one buffer, we must make additional
        # copies of `val` so that each buffer is a distinct ndarray.
        if len(local_core_id_to_buffer_id) > 0:
          val = val.copy()
        shared_memory.mem[(memory_space_str, buffer_id, device_id, lci)] = (
            Buffer(val)
        )

      local_core_id_to_buffer_id[lci] = buffer_id

  # The buffer ids should always be kept in sync across all cores.
  assert all(
      buffer_id == local_core_id_to_buffer_id[local_core_id_int]
      for buffer_id in local_core_id_to_buffer_id.values()
  )

  # TODO(jburnim): Raise an error if buffer_id is too big for int16.
  return np.int16(local_core_id_to_buffer_id[local_core_id_int])


def _local_core_id_or_zero_if_hbm(local_core_id: int, memory_space: str) -> int:
  if memory_space in ['any', 'hbm']:
    return 0
  return local_core_id


def _deallocate_buffer(device_id, local_core_id, memory_space, buffer_id):
  device_id = int(device_id)
  local_core_id = int(local_core_id)
  memory_space = TPU_MEMORY_SPACE_NAMES[int(memory_space)]
  buffer_id = int(buffer_id)

  local_core_id = _local_core_id_or_zero_if_hbm(local_core_id, memory_space)

  shared_memory = get_shared_memory()
  with shared_memory.lock:
    key = (memory_space, buffer_id, device_id, local_core_id)
    buff = shared_memory.mem[key]
    buff.decrease_ref_count()
    if buff.has_zero_ref_count():
      shared_memory.mem.pop(key)
      shared_memory.deallocated_bytes += buff.size()
      del buff

    should_collect = shared_memory.deallocated_bytes > 100_000_000
    if should_collect:
      shared_memory.deallocated_bytes = 0

  if should_collect:
    # Periodic garbage collection here prevents OOMs -- although it's not clear
    # why arrays are not getting freed without this.
    gc.collect()


def _allocate_semaphores(
    device_id: Array, local_core_id: Array | None, shape: Array
):
  """Allocates semaphores on the device with id `device_id` and core with id `local_core_id`.

  The number of semaphores allocated is given by the product of the entries in
  `shape`.

  Since for each semaphore id there is really only one global `Semaphore`
  object, 'allocation' of semaphores per device and core here means that the
  internal counter of semaphore ids that is held by `SharedMemory` is
  incremented for each the device and core (or for all cores on the dive if
  argument `local_core_id` is None, see below).

  Args:
    device_id: Singleton array holding the id for the device where the
      semaphores will be allocated.
    local_core_id: None or singleton array holding the id for the core where the
      semaphores will be allocated. If None, semaphores will be allocated on all
      cores on the device.
    shape: Shape of the semaphore array to allocate.

  Returns:
    Array of semaphore ids.
  """
  device_id = int(device_id)
  shape = tuple(map(int, shape))
  num_semaphores = math.prod(shape)

  shared_memory = get_shared_memory()

  if local_core_id is None:
    local_core_id_int = 0
    global_core_ids = shared_memory.get_global_core_ids(device_id)
  else:
    local_core_id_int = int(local_core_id)
    global_core_ids = (
        shared_memory.get_global_core_id(device_id, local_core_id_int),
    )
  del local_core_id

  global_core_id_to_semaphore_id = {}
  with shared_memory.lock:
    for gci in global_core_ids:
      semaphore_id = shared_memory.next_semaphore_id[gci]
      shared_memory.next_semaphore_id[gci] = semaphore_id + num_semaphores

      # Ensure that only one global `Semaphore` object is allocated for each
      # `semaphore_id`.
      for i in range(semaphore_id, semaphore_id + num_semaphores):
        if i not in shared_memory.sem:
          shared_memory.sem[i] = Semaphore(
              shared_memory=shared_memory,
              semaphore_id=i,
          )

      global_core_id_to_semaphore_id[gci] = semaphore_id

  global_core_id = shared_memory.get_global_core_id(
      device_id, local_core_id_int
  )
  # The semaphore ids should always be kept in sync across all cores.
  assert all(
      semaphore_id == global_core_id_to_semaphore_id[global_core_id]
      for semaphore_id in global_core_id_to_semaphore_id.values()
  )

  # NOTE: For now, we use a relatively uncommon datatype (int16) for
  # semaphore (and buffer) IDs, so these values are more easily identifiable
  # in kernels.
  #
  # TODO(jburnim): Raise an error if any IDs are too big for int16.
  semaphore_id = global_core_id_to_semaphore_id[global_core_id]
  return np.arange(
      semaphore_id, semaphore_id + num_semaphores, dtype=np.int16
  ).reshape(shape)


TPU_MEMORY_SPACE_IDXS: dict[
    mosaic_core.MemorySpace | pallas_core.MemorySpace | None, int
] = {v: i for i, v in enumerate(mosaic_core.MemorySpace)}
TPU_MEMORY_SPACE_IDXS[pallas_core.MemorySpace.ANY] = TPU_MEMORY_SPACE_IDXS[
    mosaic_core.MemorySpace.ANY
]
TPU_MEMORY_SPACE_NAMES = {
    i: v.value for i, v in enumerate(mosaic_core.MemorySpace)
}

# Default to VMEM when no memory space is specified.
TPU_MEMORY_SPACE_IDXS[None] = TPU_MEMORY_SPACE_IDXS[
    mosaic_core.MemorySpace.VMEM
]


def get_barrier_semaphore(device_id, collective_id):
  del device_id
  collective_id = int(collective_id)

  # TODO(jburnim): Check/fix so that IDs for barrier semaphores do not conflict
  # with IDs for regular or DMA semaphores.  (For example, store them in a
  # different table.)
  shared_memory = get_shared_memory()
  with shared_memory.lock:
    semaphore_id = collective_id
    if semaphore_id not in shared_memory.sem:
      shared_memory.sem[semaphore_id] = Semaphore(
          semaphore_id=semaphore_id, shared_memory=shared_memory
      )

  return np.int16(semaphore_id)


def _transform_slice_or_index(slice_or_idx):
  if isinstance(slice_or_idx, int):
    return slice_or_idx
  else:
    start = int(slice_or_idx.start)
    size = int(slice_or_idx.size)
    stride = int(slice_or_idx.stride)
    return slice(start, start + size * stride, stride)


def _compose_slice_or_index(slice_or_idx1, slice_or_idx2):
  ret = []
  i = 0
  j = 0
  while True:
    if i == len(slice_or_idx1):
      ret.extend(slice_or_idx2[j:])
      return tuple(ret)
    elif j == len(slice_or_idx2):
      ret.extend(slice_or_idx1[i:])
      return tuple(ret)
    elif isinstance(slice_or_idx1[i], int):
      ret.append(slice_or_idx1[i])
      i += 1
    elif isinstance(slice_or_idx2[j], int):
      ret.append(
          slice_or_idx1[i].start + slice_or_idx2[j] * slice_or_idx1[i].step
      )
      i += 1
      j += 1
    else:
      ret.append(
          slice(
              slice_or_idx1[i].start
              + slice_or_idx2[j].start * slice_or_idx1[i].step,
              slice_or_idx1[i].start
              + slice_or_idx2[j].stop * slice_or_idx1[i].step,
              slice_or_idx1[i].step * slice_or_idx2[j].step,
          )
      )
      i += 1
      j += 1


def _to_range(transforms) -> tuple[slice | int, ...]:
  ret = ()
  for transform in transforms:
    # For now, assume only NDIndexer transforms.
    ret = _compose_slice_or_index(
        ret, tuple(_transform_slice_or_index(i) for i in transform.indices)
    )
  return ret


def _to_int(x: int | Array | None) -> int | None:
  """Converts a value to an integer, or returns None if the value is None."""
  if x is None:
    return None
  return int(x)


def get(
    device_id,
    local_core_id,
    memory_space,
    buffer_id,
    transforms,
    block_indices=None,
    grid_loop_idx=None,
    *,
    src_device_id=None,
    src_local_core_id=None,
    clock=None,
    source_info=None,
    input_name=None,
) -> np.ndarray:
  device_id = int(device_id)
  local_core_id = int(local_core_id)
  memory_space = TPU_MEMORY_SPACE_NAMES[int(memory_space)]
  buffer_id = int(buffer_id)
  try:
    transforms = jax.tree.map(int, transforms)
  except:
    raise ValueError('Advanced indexers are not supported on TPU')
  src_device_id = _to_int(src_device_id)
  src_local_core_id = _to_int(src_local_core_id)
  if input_name is not None:
    # NOTE: input_name, block_indices, and grid_loop_idx are set only if this
    # function is being called to read a block from a pallas_call input (at the
    # start of one iteration of the kernel body).
    block_indices = tuple(int(x) for x in block_indices)
    grid_loop_idx = tuple(int(x) for x in tuple(grid_loop_idx))

  shared_memory = get_shared_memory()

  local_core_id_for_buffer = _local_core_id_or_zero_if_hbm(
      local_core_id, memory_space
  )
  global_core_id = shared_memory.get_global_core_id(device_id, local_core_id)

  with shared_memory.lock:
    read_range = _to_range(transforms)
    if shared_memory.detect_races:
      vc.inc_vector_clock(shared_memory.clocks[global_core_id], global_core_id)
      if clock is None:
        clock = vc.copy_vector_clock(shared_memory.clocks[global_core_id])
    array = shared_memory.mem[
        (memory_space, buffer_id, device_id, local_core_id_for_buffer)
    ].content
    try:
      ret = array[read_range].copy()
    except:
      ret = None

    # Compute the shape of the read value, assuming the read is fully in-bounds.
    # TODO(jburnim): We already know this shape in the Jaxpr where we insert a
    # callback to `get`.  Should we just pass the shape to `get`?
    # TODO(jburnim): Move to a helper function?
    full_read_shape = []
    assert len(read_range) <= len(array.shape)
    for dim_size, idx_or_slice in itertools.zip_longest(
        array.shape, read_range, fillvalue=None
    ):
      assert isinstance(dim_size, int)
      if idx_or_slice is None:
        full_read_shape.append(dim_size)
      elif isinstance(idx_or_slice, int):
        continue
      else:
        dim_size = (idx_or_slice.stop - idx_or_slice.start) // idx_or_slice.step
        assert isinstance(dim_size, int)
        full_read_shape.append(dim_size)
    full_read_shape = tuple(full_read_shape)

    if (ret is None) or (full_read_shape != ret.shape):
      if shared_memory.out_of_bounds_reads == 'raise':
        if source_info is None:
          ctx = contextlib.nullcontext()
        else:
          ctx = source_info_util.user_context(
              traceback=source_info.traceback, name_stack=source_info.name_stack
          )  # type: ignore[assignment]
        with ctx:
          if input_name is None:
            raise IndexError(
                'Out-of-bounds read of'
                f' ({device_id} {local_core_id} {memory_space} {buffer_id}):'
                f' reading [{read_range}] but buffer has shape {array.shape}.'
            )
          else:
            # Different error message when we are reading a block of an input,
            # to copy it to a buffer before invoking the kernel body.
            raise IndexError(
                f'Out-of-bounds block index {block_indices} for'
                f' input "{input_name}" in iteration {grid_loop_idx}'
                f' on device {device_id} (core {local_core_id}):'
                f' reading [{read_range}] but input has shape {array.shape}.'
            )
      # out_of_bounds_reads == "uninitialized"
      uninit_array = np.full(
          full_read_shape,
          shared_memory.uninitialized_value_callback(array.dtype),
          dtype=array.dtype,
      )
      if ret is None:
        ret = uninit_array
      else:
        uninit_array[tuple(slice(s) for s in ret.shape)] = ret
        ret = uninit_array

  if shared_memory.detect_races:
    if src_device_id is None:
      src_device_id = device_id
    if src_local_core_id is None:
      src_local_core_id = local_core_id
    assert shared_memory.races is not None
    shared_memory.races.check_read(
        src_device_id,
        src_local_core_id,
        clock,
        (memory_space, buffer_id, device_id, local_core_id_for_buffer),
        read_range,
        source_info=source_info,
    )

  return ret


def store(
    device_id,
    local_core_id,
    memory_space,
    buffer_id,
    transforms,
    val,
    block_indices=None,
    grid_loop_idx=None,
    *,
    src_device_id=None,
    src_local_core_id=None,
    clock=None,
    source_info=None,
    output_name=None,
):
  device_id = int(device_id)
  local_core_id = int(local_core_id)
  memory_space = TPU_MEMORY_SPACE_NAMES[int(memory_space)]
  buffer_id = int(buffer_id)
  try:
    transforms = jax.tree.map(int, transforms)
  except:
    raise ValueError('Advanced indexers are not supported on TPU')
  val = np.array(val)
  src_device_id = _to_int(src_device_id)
  src_local_core_id = _to_int(src_local_core_id)
  if output_name is not None:
    # NOTE: output_name, block_indices, and grid_loop_idx are set only if this
    # function is being called to store a block into a pallas_call output (at
    # the end of one iteration of the kernel body).
    block_indices = tuple(int(x) for x in block_indices)
    grid_loop_idx = tuple(int(x) for x in tuple(grid_loop_idx))

  shared_memory = get_shared_memory()

  local_core_id_for_buffer = _local_core_id_or_zero_if_hbm(
      local_core_id, memory_space
  )
  global_core_id = shared_memory.get_global_core_id(device_id, local_core_id)

  with shared_memory.lock:
    if shared_memory.detect_races:
      vc.inc_vector_clock(shared_memory.clocks[global_core_id], global_core_id)
      if clock is None:
        clock = vc.copy_vector_clock(shared_memory.clocks[global_core_id])

    array = shared_memory.mem[
        (memory_space, buffer_id, device_id, local_core_id_for_buffer)
    ].content
    assert array.dtype == val.dtype  # TODO(jburnim): Catch this statically.
    write_range = _to_range(transforms)
    # TODO(jburnim): Better error message if this raises?
    in_bounds_shape = array[write_range].shape
    if in_bounds_shape != val.shape:
      if output_name is None:
        raise ValueError(
            'Out-of-bounds write of'
            f' ({device_id} {local_core_id} {memory_space} {buffer_id}):'
            f' writing [{write_range}] but buffer has shape {array.shape} .'
        )
      else:
        # Different error message when we are copying a kernel buffer to a
        # block of an output (just after a kernel invocation).
        raise IndexError(
            f'Out-of-bounds block index {block_indices} for'
            f' output "{output_name}" in iteration {grid_loop_idx}'
            f' on device {device_id} (core {local_core_id}):'
            f' reading [{write_range}] but output has shape {array.shape}.'
        )
    array[write_range] = val

  if shared_memory.detect_races:
    if src_device_id is None:
      src_device_id = device_id
    if src_local_core_id is None:
      src_local_core_id = local_core_id
    assert shared_memory.races is not None
    shared_memory.races.check_write(
        src_device_id,
        src_local_core_id,
        clock,
        (memory_space, buffer_id, device_id, local_core_id_for_buffer),
        write_range,
        source_info=source_info,
    )


def swap(
    device_id,
    local_core_id,
    memory_space,
    buffer_id,
    transforms,
    val,
    mask,
    *,
    source_info=None,
):
  device_id = int(device_id)
  local_core_id = int(local_core_id)
  memory_space = TPU_MEMORY_SPACE_NAMES[int(memory_space)]
  buffer_id = int(buffer_id)
  try:
    transforms = jax.tree.map(int, transforms)
  except:
    raise ValueError('Advanced indexers are not supported on TPU')
  val = np.array(val)
  mask = np.array(mask) if mask is not None else None
  if mask is not None:
    assert mask.shape == val.shape

  shared_memory = get_shared_memory()

  local_core_id_for_buffer = _local_core_id_or_zero_if_hbm(
      local_core_id, memory_space
  )
  global_core_id = shared_memory.get_global_core_id(device_id, local_core_id)

  with shared_memory.lock:
    if shared_memory.detect_races:
      vc.inc_vector_clock(shared_memory.clocks[global_core_id], global_core_id)
      clock = vc.copy_vector_clock(shared_memory.clocks[global_core_id])
    array = shared_memory.mem[
        (memory_space, buffer_id, device_id, local_core_id_for_buffer)
    ].content
    assert array.dtype == val.dtype  # TODO(jburnim): Catch this statically.
    read_write_range = _to_range(transforms)
    # TODO(jburnim): Better error message if this raises?
    raw_result = array[read_write_range]
    in_bounds_shape = raw_result.shape
    if mask is None:
      if in_bounds_shape != val.shape:
        raise ValueError(
            'Out-of-bounds swap of'
            f' ({device_id} {local_core_id} {memory_space} {buffer_id}):'
            f' swapping [{read_write_range}] but buffer has shape'
            f' {array.shape} .'
        )
      array[read_write_range] = val
      return raw_result.copy()

    in_bounds_mask = np.full(mask.shape, True)
    for i in range(len(in_bounds_shape)):
      in_bounds_mask[in_bounds_shape[i] :] = False
    if (~in_bounds_mask & mask).any():
      # TODO(jburnim): Include indices of out-of-bounds locations where mask
      # is True.
      raise ValueError(
          'Out-of-bounds masked swap of'
          f' ({device_id} {local_core_id} {memory_space} {buffer_id}): swapping'
          f' [{read_write_range}] but buffer has shape {array.shape} . '
      )

    in_bounds_idx = tuple(slice(i) for i in in_bounds_shape)
    result = val.copy()
    result[in_bounds_idx] = np.where(
        mask[in_bounds_idx], raw_result, val[in_bounds_idx]
    )
    array[read_write_range] = np.where(
        mask[in_bounds_idx], val[in_bounds_idx], raw_result
    )

  if shared_memory.detect_races:
    assert shared_memory.races is not None
    shared_memory.races.check_write(
        device_id,
        local_core_id,
        clock,
        (memory_space, buffer_id, device_id, local_core_id_for_buffer),
        read_write_range,
        source_info=source_info,
    )
  return result


def execute_dma(dma):
  # TODO(jburnim) Eliminate duplicate code here and in Semaphore.wait.
  shared_memory = get_shared_memory()
  with dma.lock:
    assert dma.state == DmaState.STARTED

    if dma.virtual_device_id is None:
      # See comment in Semaphore.wait .
      dma.virtual_device_id = np.random.randint(
          shared_memory.num_cores, shared_memory.vector_clock_size
      )

    # Do the read.
    if shared_memory.detect_races:
      vc.inc_vector_clock(dma.clock, dma.virtual_device_id)
    dma.data = get(
        dma.src_device_id,
        dma.src_local_core_id,
        dma.src_memory_space,
        dma.src_buffer_id,
        dma.src_transforms,
        clock=vc.copy_vector_clock(dma.clock),
        src_device_id=dma.id,
        src_local_core_id=0,
        source_info=dma.source_info,
    )
    data_size = dma.data.itemsize * dma.data.size

    # Signal the send semaphore.
    if shared_memory.detect_races:
      vc.inc_vector_clock(dma.clock, dma.virtual_device_id)
    if dma.src_sem is not None:
      dma.src_sem.signal(
          data_size,
          global_core_id=shared_memory.get_global_core_id(
              dma.src_device_id, dma.src_local_core_id
          ),
          clock=dma.clock,
      )
    dma.state = DmaState.READ

    # Do the write.
    if shared_memory.detect_races:
      vc.inc_vector_clock(dma.clock, dma.virtual_device_id)
    store(
        dma.dst_device_id,
        dma.dst_local_core_id,
        dma.dst_memory_space,
        dma.dst_buffer_id,
        dma.dst_transforms,
        dma.data,
        clock=vc.copy_vector_clock(dma.clock),
        src_device_id=dma.id,
        src_local_core_id=0,
        source_info=dma.source_info,
    )

    # Signal the receive semaphore.
    if shared_memory.detect_races:
      vc.inc_vector_clock(dma.clock, dma.virtual_device_id)
    if dma.dst_sem is not None:
      dma.dst_sem.signal(
          data_size,
          global_core_id=shared_memory.get_global_core_id(
              dma.dst_device_id, dma.dst_local_core_id
          ),
          clock=dma.clock,
      )

    dma.data = None
    dma.state = DmaState.COMPLETED


def print_memory(device_id):
  device_id = int(device_id)
  if device_id == 0:
    shared_memory = get_shared_memory()
    with shared_memory.lock:
      print(shared_memory.mem)


def dma_start(
    device_id,
    src_local_core_id,
    src_memory_space,
    src_id,
    src_transforms,
    dst_memory_space,
    dst_id,
    dst_transforms,
    dst_sem_id,
    src_sem_id,
    dst_device_id,
    source_info=None,
):
  shared_memory = get_shared_memory()
  device_id = int(device_id)
  src_local_core_id = int(src_local_core_id)
  src_global_core_id = shared_memory.get_global_core_id(
      device_id, src_local_core_id
  )
  src_memory_space, src_id = int(src_memory_space), int(src_id)
  src_transforms = jax.tree.map(int, src_transforms)
  dst_memory_space, dst_id = int(dst_memory_space), int(dst_id)
  dst_transforms = jax.tree.map(int, dst_transforms)
  dst_sem_id = int(dst_sem_id)
  src_sem_id = int(src_sem_id) if src_sem_id is not None else None
  if dst_device_id is not None:
    dst_device_id = int(dst_device_id)
  else:
    dst_device_id = device_id

  with shared_memory.lock:
    dst_sem = shared_memory.sem[dst_sem_id]
    src_sem = shared_memory.sem[src_sem_id] if src_sem_id is not None else None

    clock = None
    if shared_memory.detect_races:
      vc.inc_vector_clock(
          shared_memory.clocks[src_global_core_id], src_global_core_id
      )
      clock = vc.copy_vector_clock(shared_memory.clocks[src_global_core_id])
    dma_id = shared_memory.next_dma_id
    shared_memory.next_dma_id += 1

    dma = DMA(
        dma_id,
        device_id,
        src_local_core_id,
        src_memory_space,
        src_id,
        src_transforms,
        dst_device_id,
        src_local_core_id,  # Same core on destination device as on source.
        dst_memory_space,
        dst_id,
        dst_transforms,
        src_sem,
        dst_sem,
        clock=clock,
        source_info=source_info,
    )

    if shared_memory.dma_execution_mode == 'on_wait':
      shared_memory.dmas_by_sem[dst_sem_id].append(dma)
      if src_sem_id is not None:
        shared_memory.dmas_by_sem[src_sem_id].append(dma)
      return

  assert shared_memory.dma_execution_mode == 'eager'
  execute_dma(dma)


def dma_wait(device_id, local_core_id, sem_id, size):
  shared_memory = get_shared_memory()

  device_id = int(device_id)
  local_core_id = int(local_core_id)
  sem_id = int(sem_id)
  size = int(size)

  global_core_id = shared_memory.get_global_core_id(device_id, local_core_id)

  with shared_memory.lock:
    if shared_memory.detect_races:
      vc.inc_vector_clock(shared_memory.clocks[global_core_id], global_core_id)
    sem = shared_memory.sem[sem_id]
  sem.wait(size, global_core_id, is_dma=True)


def semaphore_signal(
    device_id,
    local_core_id,
    sem_id,
    inc,
    target_device_id,
    target_local_core_id,
):
  shared_memory = get_shared_memory()

  device_id = int(device_id)
  local_core_id = int(local_core_id)
  sem_id = int(sem_id)
  inc = int(inc)
  src_global_core_id = shared_memory.get_global_core_id(
      device_id, local_core_id
  )
  if target_device_id is None:
    target_device_id = device_id
  else:
    target_device_id = int(target_device_id)
  if target_local_core_id is None:
    target_local_core_id = 0

  with shared_memory.lock:
    clock = None
    if shared_memory.detect_races:
      vc.inc_vector_clock(
          shared_memory.clocks[src_global_core_id], src_global_core_id
      )
      clock = vc.copy_vector_clock(shared_memory.clocks[src_global_core_id])
    sem = shared_memory.sem[sem_id]

  sem.signal(
      inc,
      shared_memory.get_global_core_id(target_device_id, target_local_core_id),
      clock,
  )


def semaphore_wait(device_id, local_core_id, sem_id, value):
  shared_memory = get_shared_memory()

  device_id = int(device_id)
  local_core_id = int(local_core_id)
  sem_id = int(sem_id)
  value = int(value)
  global_core_id = shared_memory.get_global_core_id(device_id, local_core_id)

  with shared_memory.lock:
    if shared_memory.detect_races:
      vc.inc_vector_clock(shared_memory.clocks[global_core_id], global_core_id)
    sem = shared_memory.sem[sem_id]

  sem.wait(value, global_core_id)


def _check_for_revisiting(device_id, local_core_id, loop_idx, output_blocks):
  device_id = int(device_id)
  local_core_id = int(local_core_id)
  loop_idx = tuple(int(x) for x in loop_idx)
  try:
    output_blocks = jax.tree.map(int, output_blocks)
  except:
    raise ValueError('Advanced indexers are not supported on TPU')
  output_ranges = [
      _to_range(b) if b is not None else None for b in output_blocks
  ]

  shared_memory = get_shared_memory()
  past_output_ranges = shared_memory.output_ranges[(device_id, local_core_id)]
  if not past_output_ranges:
    past_output_ranges.append((loop_idx, output_ranges))
    return

  for i in range(len(output_ranges)):
    if output_ranges[i] is None:
      continue
    if past_output_ranges[-1][1][i] == output_ranges[i]:
      continue
    # TODO(jburnim): Do something constant time instead of linear here.
    past_idxs = [
        j
        for j, ors in enumerate(past_output_ranges)
        if ors[1][i] == output_ranges[i]
    ]
    if past_idxs:
      raise RuntimeError(
          f'Revisited block {output_ranges[i]} of output {i} in iteration '
          f'{loop_idx}. The block was previously visited in iterations '
          f'{past_output_ranges[past_idxs[0]][0]} through '
          f'{past_output_ranges[past_idxs[-1]][0]} .'
      )

  past_output_ranges.append((loop_idx, output_ranges))


def _validate(device_id):
  device_id = int(device_id)

  shared_memory = get_shared_memory()
  local_core_ids = tuple(range(shared_memory.num_cores_per_device))
  with shared_memory.lock:
    for sem in shared_memory.sem.values():
      with sem.cv:
        for lci in local_core_ids:
          global_core_id = shared_memory.get_global_core_id(device_id, lci)
          if sem.count_by_core[global_core_id] != 0:
            # TODO(jburnim): Make this raise an error, but in a way that doesn't
            # cause other devices to hang later in `_clean_up_shared_memory`.
            print(
                f'Semaphore {sem.id} has non-zero count for {device_id} (core'
                f' {lci}) at kernel exit: {sem.count_by_core[global_core_id]}'
            )
