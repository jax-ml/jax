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
from collections.abc import Sequence
import dataclasses
import gc
import threading
from typing import Any, Callable, Literal

from jax._src.pallas.mosaic.interpret import vector_clock as vc
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

  def get_global_core_id(self, device_id: int, local_core_id: int) -> int:
    return self.shared_memory.get_global_core_id(device_id, local_core_id)

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
      if self.detect_races:
        if self.clocks[global_core_id] is None:
          self.clocks[global_core_id] = vc.copy_vector_clock(clock)
        else:
          vc.update_vector_clock(self.clocks[global_core_id], clock)
      self.cv.notify_all()

  def read(self, global_core_id):
    with self.cv:
      return self.count_by_core[global_core_id]

  def wait(self, value, global_core_id, *, has_tasks=False):
    global_core_id = int(global_core_id)

    # TODO(jburnim):
    #  - If the count is larger than value, raise an error?
    #  - If the count is equal to value, but there DMAs waiting to signal us,
    #    raise an error?

    # Simple implementation for semaphores that have no tasks that can signal
    # them.
    clock = None
    if not has_tasks:
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

    # TODO(nrink): Update the comment below to generalize from DMAs and DMA
    # semaphores. We now have the concept of 'tasks' that can signal a
    # semaphore. At the moment, DMAs are the only tasks that occur; and what is
    # allowed to be a task may still change (because it should probably be more
    # restricted than allowing tasks to be arbitrary callables, as is currently
    # done).
    #
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
        task_queue = self.shared_memory.tasks_by_sem[(self.id, global_core_id)]
        if len(task_queue) > 0:
          task = task_queue.pop()
        else:
          continue

      task()


# A `SemaphoreTask` is called when a semaphore is waiting to be signalled on a
# specific core. A `SemaphoreTask` will typically capture the `Semaphore` object
# that is waiting, so that when the task is called, it can signal the semaphore
# (by calling `Semaphore.signal` from within the task). When a `SemaphoreTask`
# object is called, it can be assumed that the call stack of the task will
# *not* hold the lock on the shared memory in the captured `Semaphore` object.
# This allows the task to use methods from `SharedMemory` to access and modify
# the global shared memory object.
SemaphoreTask = Callable[[], None]


@dataclasses.dataclass(init=False)
class Allocation:
  ...


@dataclasses.dataclass
class Buffer(Allocation):
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


@dataclasses.dataclass(frozen=True)
class ShapeAndDtype:
  shape: Sequence[int]
  dtype: np.dtype

  def __iter__(self):
    return iter((self.shape, self.dtype))


@dataclasses.dataclass
class SharedMemory:
  num_devices: int
  num_cores_per_device: int
  out_of_bounds_reads: str
  dma_execution_mode: str
  uninitialized_memory: Literal["nan", "zero"]
  detect_races: bool
  vector_clock_size: int

  clocks: list[vc.VectorClock]
  barrier: threading.Barrier
  clean_up_barrier: threading.Barrier

  # (memory_space, buffer_id, device_id, local_core_id) -> Allocation
  mem: dict[tuple[str, int, int, int], Allocation] = dataclasses.field(
      default_factory=dict
  )

  # semaphore_id -> Semaphore
  sem: dict[int, Semaphore] = dataclasses.field(default_factory=dict)

  # (semaphore_id, global_core_id)
  #   -> tasks that will signal the semaphore on the core with the given ID and
  #      that should therefore be considered for execution when the semaphore is
  #      waiting (to be signalled).
  tasks_by_sem: dict[tuple[int, int], list[SemaphoreTask]] = dataclasses.field(
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

  deallocated_bytes: int = 0

  # (device_id, local_core_id) -> [(grid_index, [range])]
  output_ranges: dict[tuple[int, int], list] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list)
  )

  # semaphore_id -> Semaphore, where the semaphore_id is user-specified.
  fixed_id_sem: dict[int, Semaphore] = dataclasses.field(
      default_factory=dict
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

  def append_semaphore_task(
      self,
      semaphore_id: int,
      global_core_id: int,
      task: SemaphoreTask,
  ):
    """Appends a task to be executed if the semaphore with the given sempahore ID is waiting to be signalled on the core with the given global core ID."""
    with self.lock:
      self.tasks_by_sem[(semaphore_id, global_core_id)].append(task)

  def get_random_virtual_device_id(self) -> int:
    # Virtual device IDs are needed for DMAs. Conceptually, each DMA runs on its
    # own, independent device. Representing this precisely would require vector
    # clocks to have sizes linear in the number of DMAs.
    #
    # Instead, we use approximate vector clocks of fixed size. We assign each
    # DMA a virtual core ID in the range
    #
    #   [num_cores, self.vector_clock_size - 1],
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
    return np.random.randint(self.num_cores, self.vector_clock_size)

  def print(self, device_id: int):
    device_id = int(device_id)
    if device_id == 0:
      with self.lock:
        print(self.mem)

  def get_semaphores_and_increment_clock(
      self, sem_ids: Sequence[int | None], global_core_id: int
  ) -> tuple[list[Semaphore | None], vc.VectorClock | None]:
    """Returns the semaphores with the given `sem_ids` and increments the vector clock for the core with `global_core_id`.

    If race detection is enabled, this method increments the vector clock for
    the core with the given `global_core_id` (while holding the lock on `self`).
    We do this so that we can associate a (vector clock) time with the shared
    memory operation of looking up the semaphores, which in turn can be used as
    a proxy for the time when the returned semaphores are used by the client of
    the `SharedMemory` class without acquiring the lock on `self`. (For the
    purpose of encapsulation, we prefer to think of `self.lock` as a private
    attribute of the `SharedMemory` class; hence clients of the class should not
    attempt to acquire this lock explicitly.)

    Args:
      sem_ids: The IDs of the semaphores to return or None.
      global_core_id: The ID of the core whose vector clock should be
        incremented (if race detection is enabled).

    Returns:
      - The semaphores with the given `sem_ids` or None if the corresponding
        entry in `sem_ids` is None.
      - The incremented vector clock for the core with the given
        `global_core_id`, or None if race detection is not enabled.
    """
    clock = None
    with self.lock:
      if self.detect_races:
        vc.inc_vector_clock(self.clocks[global_core_id], global_core_id)
        clock = vc.copy_vector_clock(self.clocks[global_core_id])

      sems = []
      for sem_id in sem_ids:
        if sem_id is None:
          sem = None
        elif sem_id in self.fixed_id_sem:
          if sem_id in self.sem:
            # TODO(nrink): For now we make it the responsibility of the client to
            # ensure that fixed-ID semaphores do not collide with internal
            # semaphore IDs.
            raise ValueError(
                f'Semaphore {sem_id} occurs as both fixed-id and internal.'
            )
          sem = self.fixed_id_sem[sem_id]
        else:
          sem = self.sem[sem_id]
        sems.append(sem)

    return sems, clock

  def get_sempahores_with_nonzero_count(
      self, device_id: int
  ) -> list[tuple[Semaphore, int]]:
    """Returns tuples (semaphore, global_core_id) for all semaphores with a nonzero count for the core with `global_core_id`."""
    result = []
    with self.lock:
      for _, sem in self.sem.items() | self.fixed_id_sem.items():
        with sem.cv:
          for gci in self.get_global_core_ids(device_id):
            if sem.count_by_core[gci] != 0:
              result.append((sem, gci))
    return result

  def get_next_buffer_id(self, device_id: int, local_core_id: int) -> int:
    """Returns the next buffer ID for the given device and local core ID."""
    with self.lock:
      buffer_id = self.next_buffer_id[(device_id, local_core_id)]
      self.next_buffer_id[(device_id, local_core_id)] = buffer_id + 1
      return buffer_id

  def allocate_buffer(
      self,
      key: Any,
      ref_count: int,
      value: np.ndarray,
  ):
    """Allocates a memory buffer with the given key unless it already exists."""
    with self.lock:
      if key not in self.mem:
        self.mem[key] = Buffer(value, ref_count=ref_count)

  def deallocate_buffer(self, key: Any):
    """Decreases the ref count for the buffer with `key` and deallocates the buffer if the ref count is zero."""
    with self.lock:
      buff = self.mem[key]
      if not isinstance(buff, Buffer):
        raise ValueError(
            f"Attempting to deallocate allocation with key `{key}` that is not"
            " a `Buffer`."
        )

      buff.decrease_ref_count()
      if buff.has_zero_ref_count():
        self.mem.pop(key)
        self.deallocated_bytes += buff.size()
        del buff

      should_collect = self.deallocated_bytes > 100_000_000
      if should_collect:
        self.deallocated_bytes = 0

    if should_collect:
      # Periodic garbage collection here prevents OOMs -- although it's not clear
      # why arrays are not getting freed without this.
      gc.collect()

  def allocate_semaphores(self, key: Any, num_semaphores: int) -> int:
    """Returns the next semaphore ID and ensures that the next `num_semaphores` are allocated."""
    with self.lock:
      semaphore_id = self.next_semaphore_id[key]
      self.next_semaphore_id[key] = semaphore_id + num_semaphores

      for i in range(semaphore_id, semaphore_id + num_semaphores):
        if i not in self.sem:
          self.sem[i] = Semaphore(shared_memory=self, semaphore_id=i)

    return semaphore_id

  def guarantee_semaphore_with_fixed_id(self, semaphore_id: int):
    """Ensures that a semaphore with the given `semaphore_id` exists.

    If the semaphore with the given ID does not exist, it is allocated. Note
    that semaphores that are allocated with this method live in their own
    address space (internally, they are mapped in a separate dictionary) from
    the sempahores allocated with the `allocate_sempahores` method above.

    This methods is intended to be used for barrier semaphores, where the
    _collective_ semaphore ID is specified by the interpreter (i.e. by the
    client of the `SharedMemory` class). This simulates sempahores that exist
    prior to any Pallas kernels being run.

    Args:
      semaphore_id: The ID of the semaphore to ensure exists, i.e. is allocated.
    """
    with self.lock:
      if semaphore_id not in self.fixed_id_sem:
        self.fixed_id_sem[semaphore_id] = Semaphore(
            semaphore_id=semaphore_id, shared_memory=self
        )

  def get_buffer_content(
      self, key: Any, rnge: tuple[slice | int, ...], global_core_id: int
  ) -> tuple[np.ndarray | None, ShapeAndDtype, vc.VectorClock | None]:
    """Reads contents of a memory buffer.

    Args:
      key: The key of the buffer to read.
      rnge: The range to read within the buffer.
      global_core_id: The global core ID of the core reading the buffer.

    Returns:
      - The contents of the read range of the buffer, or None if reading out of
        bounds.
      - The shape and dtype of the full content array of the buffer.
      - The incremented vector clock for the core with the given global core ID,
        or None if race detection is not enabled.
    """
    clock = None
    with self.lock:
      if self.detect_races:
        vc.inc_vector_clock(self.clocks[global_core_id], global_core_id)
        clock = vc.copy_vector_clock(self.clocks[global_core_id])

      buff = self.mem[key]
      if not isinstance(buff, Buffer):
        raise ValueError(
            f"Attempting to get contents of allocation with key `{key}` that is"
            " not a `Buffer`."
        )
      array = buff.content

      try:
        result = array[rnge].copy()
      except:
        result = None

    shape_and_dtype = ShapeAndDtype(array.shape, array.dtype)
    return result, shape_and_dtype, clock

  def store_buffer_content(
      self,
      key: Any,
      rnge: tuple[slice | int, ...],
      value: np.ndarray,
      global_core_id: int,
  ) -> tuple[bool, ShapeAndDtype, vc.VectorClock | None]:
    """Stores contents into a memory buffer.

    Args:
      key: The key of the buffer to store into.
      rnge: The range within the buffer contents that `value` is written to.
      value: The array to store into the buffer.
      global_core_id: The global core ID of the core writing into the buffer.

    Returns:
      - True of the store was in bounds, False otherwise.
      - The shape and dtype of the full content array of the buffer.
      - The incremented vector clock for the core with the given global core ID,
        or None if race detection is not enabled.
    """
    clock = None
    with self.lock:
      if self.detect_races:
        vc.inc_vector_clock(self.clocks[global_core_id], global_core_id)
        clock = vc.copy_vector_clock(self.clocks[global_core_id])

      buff = self.mem[key]
      if not isinstance(buff, Buffer):
        raise ValueError(
            f"Attempting to store into allocation with key `{key}` that is not"
            " a `Buffer`."
        )
      array = buff.content
      shape_and_dtype = ShapeAndDtype(array.shape, array.dtype)

      assert array.dtype == value.dtype  # TODO(jburnim): Catch this statically.
      # TODO(jburnim): Better error message if this raises?
      in_bounds_shape = array[rnge].shape
      if in_bounds_shape == value.shape:
        is_in_bounds = True
        array[rnge] = value
      else:
        is_in_bounds = False

      return is_in_bounds, shape_and_dtype, clock

  def swap_buffer_content(
      self,
      key: Any,
      rnge: tuple[slice | int, ...],
      value: np.ndarray,
      mask: np.ndarray | None,
      global_core_id: int,
  ) -> tuple[np.ndarray | None, ShapeAndDtype, vc.VectorClock | None]:
    """Swaps contents of a memory buffer.

    Args:
      key: The key of the buffer to swap into.
      rnge: The range within the buffer contents that `value` is swapped into.
      value: The array to be written into the buffer.
      mask: The mask to apply to the swap operation.
      global_core_id: The global core ID of the core writing into the buffer.

    Returns:
      - The contents of the range of the buffer (prior to the swap), or None if
        accessing buffer contents bounds.
      - The shape and dtype of the full content array of the buffer.
      - The incremented vector clock for the core with the given global core ID,
        or None if race detection is not enabled.
    """
    clock = None
    with self.lock:
      if self.detect_races:
        vc.inc_vector_clock(self.clocks[global_core_id], global_core_id)
        clock = vc.copy_vector_clock(self.clocks[global_core_id])

      buff = self.mem[key]
      if not isinstance(buff, Buffer):
        raise ValueError(
            f"Attempting to swap into allocation with `key` {key} that is not a"
            " `Buffer`."
        )

      array = buff.content
      shape_and_dtype = ShapeAndDtype(array.shape, array.dtype)

      assert array.dtype == value.dtype  # TODO(jburnim): Catch this statically.
      # TODO(jburnim): Better error message if this raises?
      raw_result = array[rnge]
      in_bounds_shape = raw_result.shape

      if mask is None:
        if in_bounds_shape == value.shape:
          array[rnge] = value
          return raw_result.copy(), shape_and_dtype, clock
        else:
          return None, shape_and_dtype, clock
      else:
        in_bounds_mask = np.full(mask.shape, True)
        for i in range(len(in_bounds_shape)):
          in_bounds_mask[in_bounds_shape[i] :] = False
        if (~in_bounds_mask & mask).any():
          return None, shape_and_dtype, clock
        else:
          in_bounds_idx = tuple(slice(i) for i in in_bounds_shape)
          result = value.copy()
          result[in_bounds_idx] = np.where(
              mask[in_bounds_idx], raw_result, value[in_bounds_idx]
          )
          array[rnge] = np.where(
              mask[in_bounds_idx], value[in_bounds_idx], raw_result
          )
          return result.copy(), shape_and_dtype, clock

  def update_clocks(self, low_global_core_id, high_global_core_id):
    """Synchronizes the vector clocks for the cores with ids in the range between the two arguments."""
    # Despite only updating the vector clocks for some cores, we still need to
    # hold the global lock to ensure that no other devices are concurrently
    # accessing the same vector clocks.
    with self.lock:
      for c in self.clocks[low_global_core_id + 1 : high_global_core_id]:
        vc.update_vector_clock(self.clocks[low_global_core_id], c)
      for c in self.clocks[low_global_core_id + 1 : high_global_core_id]:
        vc.update_vector_clock(c, self.clocks[low_global_core_id])

  def update_clocks_for_device_barrier(self, device_id):
    """Synchronizes the vector clocks for the cores on the given device."""
    low_core_id = device_id * self.num_cores_per_device
    high_core_id = (device_id + 1) * self.num_cores_per_device
    self.update_clocks(low_core_id, high_core_id)
