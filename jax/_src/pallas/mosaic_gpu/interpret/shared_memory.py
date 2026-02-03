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

import dataclasses
import threading
from typing import Any

from absl import logging
from jax._src.pallas.mosaic.interpret import shared_memory as memory
from jax._src.pallas.mosaic.interpret import utils as interpret_utils
from jax._src.pallas.mosaic.interpret import vector_clock as vc


class Barrier(memory.Allocation):

  # A `Barrier` is very similar to a `Semaphore` (as defined in
  # `shared_memory.py`). Two key differences are:
  #   - A `Barrier` is allocated with a fixed `num_arrivals`, whereas for a
  #     `Semaphore` a thread/core waiting on the `Semaphore` can freely choose
  #     which value the `Semaphore` must have for the waiting to complete (in
  #     this thread/on this core).
  #   - Unlike a `Semaphore`, a `Barrier` cannot be used to signal threads/cores
  #     on arbitrary devices (in a mesh). The `Barrier` _lives_ in `SMEM` on one
  #     (GPU) device, and can therefore be arrived at or waited on only by the
  #     threads that are running on this device.
  # As a consequence of the second point, a `Barrier` stores only a single
  # vector clock, which is updated when threads arrive at the `Barrier`. When
  # a thread completes waiting on the `Barrier` the thread's vector clock is
  # updated with the clock value at which the `Barrier` was last arrived at.
  #
  # Internally the implementation of a `Barrier` relies on a condition variable
  # `self.cv`. Waiting on a `Barrier` is internally implemented as waiting on
  # the condition variable (until a barrier arrival has been completed). To
  # complement this, when a thread arrives at a `Barrier`, we notify all threads
  # that are currently waiting by notifying `self.cv` internally. We also use
  # the lock on `self.cv` to protect internal state of the `Barrier` that can be
  # modified by multiple threads. Internal state that can be modified by
  # multiple threads must then also only be read under the protection of the
  # lock on `self.cv`. Attributes that are protected in this way must only be
  # read or modified while holding the lock on `self.cv`. The attributes this
  # applies to are annoted with comments "Protected by `self.cv`'s lock" below.
  def __init__(
      self,
      shared_memory: GPUSharedMemory,
      ref_count: int,
      num_arrivals: int,
      enable_logging: bool = False,
  ):
    self.shared_memory = shared_memory
    self.ref_count: int = ref_count  # Protected by `self.cv`'s lock.
    self.num_arrivals: int = num_arrivals  # Protected by `self.cv`'s lock.
    self.arrivals_count: int = 0  # Protected by `self.cv`'s lock.
    self.enable_logging: bool = enable_logging

    # We model the `Barrier`'s phase as an integer and, consequently,
    # the 'next awaited phase by thread' as an array of integers. Note that on
    # real GPU hardware, a barrier's phase is a single bit/boolean that is
    # flipped when advancing to the next phase (i.e. when an arrival at the
    # barrier has been completed). In the `Barrier` implementation here, we
    # increment `self.phase` (by one) when a barrier is completed. Using an
    # integer for the `Barrier`s phase (and incrementing it instead of flipping
    # a bit) can be helpful for debugging.
    self.phase: int = 0  # Protected by `self.cv`'s lock.
    self.next_awaited_phase_by_thread: list[int] = [  # Protected by `self.cv`'s lock.
        1
    ] * shared_memory.num_threads_per_device
    # Initialize `self.phase_change_observed` to `True` so that the first
    # arrival (more precisely, the first time we have arrived
    # `self.num_arrivals` times) at the `Barrier` does not raise an error due to
    # an unobserved phase change.
    self.phase_change_observed: bool = True  # Protected by `self.cv`'s lock.

    # Invariant: We allow the lock on `self.cv` to be acquired and held in a
    # scope where `self.shared_memory.lock` is already held, but *not* the other
    # way around. The reasons for this are:
    #   - From code that holds `self.shared_memory.lock` we need to able to call
    #     methods of `Barrier` that then acquire the lock on `self.cv`
    #     internally (to modify internal state of `self` in a thread-safe way).
    #     This is needed, for example, when `self.shared_memory` deallocates a
    #     barrier (or at least decreases a barrier's ref count).
    #   - If we allowed the scopes during which `self.shared_memory.lock` and
    #     `self.cv` are both held to be nested in both ways, this can lead to
    #     deadlock.
    self.cv = threading.Condition()

    if self.shared_memory.detect_races:
      self.clock: vc.VectorClock | None = None  # Protected by `self.cv`'s lock.

  def __repr__(self) -> str:
    return (
        f"Barrier(num_arrivals={self.num_arrivals},"
        f" arrivals_count={self.arrivals_count})"
    )

  def _log(self, message: str):
    if self.enable_logging:
      logging.info(message)

  @property
  def detect_races(self) -> bool:
    return self.shared_memory.detect_races

  def has_zero_ref_count(self) -> bool:
    with self.cv:
      return self.ref_count == 0

  def deallocate(self):
    """Deallocates the `Barrier`."""
    with self.cv:
      self.ref_count -= 1
      if self.ref_count > 0:
        return

      passed_waits_by_thread = [
          p - 1 for p in self.next_awaited_phase_by_thread
      ]
      for tid, x in enumerate(passed_waits_by_thread):
        # Note that `self.phase` counts the number of completed arrivals.
        if 0 < x < self.phase:
          raise ValueError(
              f"Thread {tid} did not observe all phases ({self.phase}) for"
              f" barrier (but observed {x} {'phases' if x > 1 else 'phase'})."
          )

  def arrive(self, device_id: int, local_thread_id: int, clock):
    with self.cv:
      self.arrivals_count += 1
      if self.arrivals_count == self.num_arrivals:
        if not self.phase_change_observed:
          raise ValueError(
              "Barrier arrival was completed again before previous completion"
              " was observed by a thread."
          )
        self.phase += 1
        self.arrivals_count = 0
        self.phase_change_observed = False

        self._log(
            f"Device {device_id}, thread {local_thread_id}: Barrier {id(self)}"
            f" has completed arrival. Phase is now {self.phase}."
        )

      if self.detect_races:
        if self.clock is None:
          self.clock = vc.copy_vector_clock(clock)
        else:
          vc.update_vector_clock(self.clock, clock)

      self.cv.notify_all()

  def wait(self, device_id: int, local_thread_id: int):
    with self.cv:
      # We are waiting for the barrier to reach exactly the phase that this
      # thread is waiting for. This could lead to deadlock (see the comment in
      # the body of the `while` loop below). One way to avoid deadlock would be
      # to replace `!=` with `>`, which would allow the barrier's phase to run
      # ahead without this thread observing exactly the phase it is waiting for
      # (but only a later one). Here, we choose to compare with `!=` and avoid
      # deadlock by raising an exception inside the `while` loop.
      #
      # Note also that if instead of modelling the barrier's phase as an
      # integer, we had used a boolean (which would be closer to real GPU
      # hardware), we would be forced to use `!=` here (since `>` would not be
      # an option).
      while self.next_awaited_phase_by_thread[local_thread_id] != self.phase:
        # If `self.phase` is already past the phase that this thread is waiting
        # for, this thread will wait forever. This is because `self.phase` never
        # decreases and the only way for
        # `self.next_awaited_phase_by_thread[local_thread_id]` to increase is by
        # exiting this `while` loop.
        if self.next_awaited_phase_by_thread[local_thread_id] < self.phase:
          raise ValueError(
              f"Thread {local_thread_id} is awaiting phase"
              f" {self.next_awaited_phase_by_thread[local_thread_id]}, but"
              f" barrier is already at phase {self.phase}. (This means that"
              f" Thread {local_thread_id} has not participated in all"
              " completions of the barrier.)"
          )

        self._log(
            f"Device {device_id}, thread {local_thread_id}: Waiting for barrier"
            f" {id(self)} to reach phase"
            f" {self.next_awaited_phase_by_thread[local_thread_id]}. (Current"
            f" phase: {self.phase})"
        )
        self.cv.wait()

      self.phase_change_observed = True
      self.next_awaited_phase_by_thread[local_thread_id] += 1

      self._log(
          f"Device {device_id}, thread {local_thread_id}: Finished waiting for"
          f" phase {self.phase} of barrier {id(self)}."
      )

      # Read `self.clock` while still holding the lock on `self.cv`. (If race
      # detection is enabled, the clock is needed below to update a vector clock
      # that is managed by `self.shared_memory`.)
      clock = self.clock if self.detect_races else None

    # Note that this block cannot be nested under the `with self.cv` block
    # immediately above since this would violate the invariant that
    # `self.shared_memory.lock` *cannot* be acquired when `self.cv`'s lock is
    # already held. (See the documentation of `self.cv` above.)
    if self.detect_races:
      global_thread_id = self.shared_memory.get_global_thread_id(
          device_id, local_thread_id
      )
      # Assert before acquiring the lock on `self.shared_memory`.
      assert clock is not None
      with self.shared_memory.lock:
        vc.update_vector_clock(
            self.shared_memory.clocks[global_thread_id], clock
        )


@dataclasses.dataclass
class GPUSharedMemory(memory.SharedMemory):

  logging_mode: interpret_utils.LoggingMode | None = None

  def _log(self, message: str):
    if (
        self.logging_mode is not None
        and interpret_utils.LoggingMode.SHARED_MEMORY in self.logging_mode
    ):
      logging.info(message)

  @property
  def num_threads_per_device(self) -> int:
    return self.num_cores_per_device

  @property
  def num_global_threads(self) -> int:
    return self.num_cores

  def get_global_thread_id(self, device_id: int, local_thread_id: int) -> int:
    """Computes the global thread ID from the given device and local thread ID."""
    return self.get_global_core_id(device_id, local_thread_id)

  def allocate_barrier(
      self,
      device_id: int,
      thread_id: int,
      key: Any,
      ref_count: int,
      num_arrivals: int,
  ):
    """Allocates a barrier with the given key unless it already exists."""
    with self.lock:
      if key not in self.mem:
        barrier = Barrier(
            self,
            ref_count=ref_count,
            num_arrivals=num_arrivals,
            enable_logging=(
                self.logging_mode is not None
                and interpret_utils.LoggingMode.BARRIER in self.logging_mode
            ),
        )
        self.mem[key] = barrier
        self._log(
            f"Device {device_id}, thread {thread_id}: Allocated barrier"
            f" {id(barrier)} ({barrier}) with key {key}."
        )

  def get_barrier_and_increment_clock(
        self, key: Any, device_id: int, thread_id: int
  ) -> tuple[Barrier, vc.VectorClock | None]:
    clock = None
    with self.lock:
      if self.detect_races:
        global_thread_id = self.get_global_thread_id(
            device_id, thread_id
        )
        vc.inc_vector_clock(self.clocks[global_thread_id], global_thread_id)
        clock = vc.copy_vector_clock(self.clocks[global_thread_id])

      barrier = self.mem[key]

    if not isinstance(barrier, Barrier):
      raise ValueError(
          f"Attempting to get barrier from allocation with {key} that is not a"
          " `Barrier`."
      )

    return barrier, clock

  def deallocate_barrier(self, device_id: int, thread_id: int, key: Any):
    with self.lock:
      barrier = self.mem[key]
      if not isinstance(barrier, Barrier):
        raise ValueError(
            f"Attempting to get barrier from allocation with {key} that is not"
            " a `Barrier`."
        )

      self._log(
          f"Device {device_id}, thread {thread_id}: Decreasing ref count of"
          f" barrier {id(barrier)} with key {key}."
      )
      barrier.deallocate()

      if barrier.has_zero_ref_count():
        self._log(
            f"Device {device_id}, thread {thread_id}: Deallocating barrier"
            f" {id(barrier)} with key {key}."
        )
        self.mem.pop(key)

  def assert_no_barriers_allocated(self):
    for key, alloc in self.mem.items():
      assert not isinstance(
          alloc, Barrier
      ), f"Barrier remains allocated at key `{key}`."
