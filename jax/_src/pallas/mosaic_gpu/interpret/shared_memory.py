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
import dataclasses
import logging
import math
import threading
from typing import Protocol, Literal, Self

import jax
from jax import numpy as jnp
from jax._src import source_info_util
from jax._src.pallas.mosaic.interpret import shared_memory as memory
from jax._src.pallas.mosaic.interpret import utils as interpret_utils
from jax._src.pallas.mosaic.interpret import vector_clock as vc
from jax._src.pallas.mosaic_gpu.interpret import params as params
import numpy as np

logger = logging.getLogger(__name__)


def _to_int(x):
  """Normalizes a jax scalar array to a python int, passing through tracers

  Useful for classes like MeshLocation that can be called in 3 contexts:
  1. By the user at trace time with tuples of jax tracers
  2. By io_callback at runtime with tuples of jax array scalars
  3. By the user at runtime with python integer tuples
  """
  if isinstance(x, jax.core.Tracer):
    return x
  if isinstance(x, jax.Array):
    return x.item()
  if isinstance(x, int):
    return x
  raise ValueError(f"Unsupported type: {type(x)}")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(init=False)
class MeshLocation:
  """A location in an MGPU mesh"""
  # The location of a device within the mesh
  device_coords: tuple[int, ...]
  # The location of a cluster within the grid
  cluster_coords: tuple[int, ...]
  # The location of a block within the cluster
  block_coords: tuple[int, ...]
  # The ID of a thread within the block
  thread_id: int
  # The warp ID within the thread (warpgroup), if core-mapped over a WarpMesh
  warp_id: None | int = None

  def __init__(self,
      device_coords: tuple[int, ...] | tuple[jax.Array, ...],
      cluster_coords: tuple[int, ...] | tuple[jax.Array, ...],
      block_coords: tuple[int, ...] | tuple[jax.Array, ...],
      thread_id: int | jax.Array,
      warp_id: None | int | jax.Array = None,
  ):
    self.device_coords = tuple(_to_int(x) for x in device_coords)
    self.cluster_coords = tuple(_to_int(x) for x in cluster_coords)
    self.block_coords = tuple(_to_int(x) for x in block_coords)
    self.thread_id = _to_int(thread_id)
    self.warp_id = _to_int(warp_id) if warp_id is not None else None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(init=False)
class Device:
  """A device in an MGPU mesh."""
  device_id: int

  def __init__(self, device_id: int | jax.Array):
    self.device_id = _to_int(device_id)

  def __repr__(self) -> str:
    return f"Device({self.device_id})"

  def __hash__(self) -> int:
    return hash((self.device_id,))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(init=False)
class Warpgroup(Device):
  """
  A physical warpgroup in GPU interpret mode with flattened IDs.

  "physical" refers to the fact that this represents a group of threads that
  will run concurrently on different CPU threads. Since we don't run multiple
  clusters concurrently, cluster_id will always be 0.
  """
  cluster_id: int
  block_id: int
  warpgroup_id: int

  def __init__(
      self,
      device_id: int | jax.Array,
      cluster_id: int | jax.Array,
      block_id: int | jax.Array,
      warpgroup_id: int | jax.Array,
  ):
    super().__init__(device_id=device_id)
    self.cluster_id = _to_int(cluster_id)
    self.block_id = _to_int(block_id)
    self.warpgroup_id = _to_int(warpgroup_id)

  def __repr__(self) -> str:
    return (
        f"Warpgroup(device_id={self.device_id}, cluster_id={self.cluster_id},"
        f" block_id={self.block_id}, warpgroup_id={self.warpgroup_id})"
    )

  def __hash__(self) -> int:
    return hash((self.device_id, self.cluster_id, self.block_id, self.warpgroup_id))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(init=False)
class Warp(Warpgroup):
  """A physical warp in GPU interpret mode with flattened IDs"""
  warp_id: int

  def __init__(
      self,
      device_id: int | jax.Array,
      cluster_id: int | jax.Array,
      block_id: int | jax.Array,
      warpgroup_id: int | jax.Array,
      warp_id: int | jax.Array,
  ):
    super().__init__(
        device_id=device_id,
        cluster_id=cluster_id,
        block_id=block_id,
        warpgroup_id=warpgroup_id,
    )
    self.warp_id = _to_int(warp_id)

  def __repr__(self) -> str:
    return (
        f"Warp(device_id={self.device_id}, cluster_id={self.cluster_id},"
        f" block_id={self.block_id}, warpgroup_id={self.warpgroup_id})"
        f" warp_id={self.warp_id})"
    )

  def __hash__(self) -> int:
    return hash((
        self.device_id,
        self.cluster_id,
        self.block_id,
        self.warpgroup_id,
        self.warp_id,
    ))


Thread = Warpgroup | Warp


class GPULoggingInfo(interpret_utils.LoggingInfo):
  """Logging info for GPU interpret mode."""

  # The absolute location of this thread in the mesh, if available.
  mesh_location: MeshLocation | None
  # The physical thread that this logical thread is running on.
  thread: Thread | None

  def __init__(
      self,
      mesh_location: MeshLocation | None,
      thread: Thread | None,
      source_info: source_info_util.SourceInfo | None = None,
  ):
    device_id = thread.device_id if thread is not None else -1
    super().__init__(source_info=source_info, device_id=device_id)
    self.mesh_location = mesh_location
    self.thread = thread

  def get_location_str(self) -> str:
    return f"Mesh location: {self.mesh_location}, Thread: {self.thread}"


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
      *,
      num_pallas_threads_per_block: int,
      ref_count: int,
      num_arrivals: int,
      enable_logging: bool = False,
  ):
    self.shared_memory = shared_memory
    self.ref_count: int = ref_count  # Protected by `self.cv`'s lock.
    self.num_pallas_threads_per_block: int = num_pallas_threads_per_block
    self.num_arrivals: int = num_arrivals  # Protected by `self.cv`'s lock.
    self.arrivals_count: int = 0  # Protected by `self.cv`'s lock.
    self.enable_logging: bool = enable_logging

    # The next phase of the barrier that will complete.
    #
    # NOTE: the underlying hardware uses a single bit to track only the polarity
    # of the barrier. We track the full phase number in order to catch violations
    # of Pallas-specific invariants:
    # 1. A thread that waits on any phase must wait on all phases.
    # 2. At least one thread must observe each barrier completion.
    self.phase: int = 0  # Protected by `self.cv`'s lock.
    # Last observed phase by each thread. Note that not every thread has to
    # participate in the barrier, and we don't know ahead of time which ones
    # will, so we lazily initialize this dict the first time a thread waits on
    # the barrier.
    self.last_observed_phase_by_thread: dict[Thread, int] = (
        {}
    )  # Protected by `self.cv`'s lock.

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
      self.smem_commit_clock: vc.VectorClock | None = (
          None  # Protected by `self.cv`'s lock.
      )

  def __repr__(self) -> str:
    return (
        f"Barrier(num_arrivals={self.num_arrivals},"
        f" arrivals_count={self.arrivals_count})"
    )

  def _log(self, message: str):
    # Log every line separately to make sure `absl.logging` adds the correct
    # prefix (i.e. I*** <time> ... <source.py>:<line_number>) to each line in
    # `message`. This should not lead to mangled output within the logging for
    # `self` since the lock on `self.cv` is expected to be held whenever this
    # method is called. However, nothing keeps logged output from being
    # interleaved with logging from other barriers or from the global
    # `SharedMemory` object.
    for msg in message.split("\n"):
      logger.info(msg)

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

      if self.arrivals_count != 0:
        raise ValueError(
            f"Barrier deallocated with {self.arrivals_count} arrivals pending."
        )

      for tid, x in self.last_observed_phase_by_thread.items():
        if x != self.phase:
          raise ValueError(
              f"Thread {tid} only observed barrier up to phase {x-1}, but"
              f" barrier completed up to phase {self.phase - 1}."
          )

  def arrive(
      self,
      *,
      clock: vc.VectorClock | None = None,
      smem_commit_clock: vc.VectorClock | None = None,
      logging_info: GPULoggingInfo | None = None,
  ):
    with self.cv:
      self.arrivals_count += 1
      if self.arrivals_count == self.num_arrivals:
        if self.enable_logging and logging_info is not None:
          self._log(
              logging_info.format(
                  f"Barrier {id(self)} has completed phase {self.phase}.",
                  line_prefix="`arrive`",
              )
          )

        self.phase += 1
        self.arrivals_count = 0

        if self.phase == 2 and len(self.last_observed_phase_by_thread) == 0:
          raise ValueError(
              "Barrier completed phase 1, but no threads observed phase 0."
          )

        for (
            i,
            last_observed_phase,
        ) in self.last_observed_phase_by_thread.items():
          if last_observed_phase < self.phase - 2:
            raise ValueError(
                f"Thread {i} only observed barrier up to phase"
                f" {last_observed_phase}, but barrier completed up to phase"
                f" {self.phase - 1}, meaning thread {i} missed observing phase"
                f" {last_observed_phase + 1}."
            )

      if self.detect_races:
        assert clock is not None
        if self.clock is None:
          self.clock = vc.copy_vector_clock(clock)
        else:
          vc.update_vector_clock(self.clock, clock)

        assert smem_commit_clock is not None
        if self.smem_commit_clock is None:
          self.smem_commit_clock = vc.copy_vector_clock(smem_commit_clock)
        else:
          vc.update_vector_clock(self.smem_commit_clock, smem_commit_clock)

      self.cv.notify_all()

  def wait(
      self,
      thread: Thread,
      logging_info: GPULoggingInfo | None = None,
  ):

    with self.cv:
      last_observed_phase = self.last_observed_phase_by_thread.get(
          thread, None
      )

      # Since not all threads in a block may use the barrier, we must lazily
      # initialize the `last_observed_phase_by_thread` array the first time each
      # thread participates in the barrier.
      if last_observed_phase is None:
        if self.phase > 1:
          raise ValueError(
              f"Thread {thread} is waiting at barrier {id(self)} for"
              " the first time, but barrier is already at phase"
              f" {self.phase}. Any thread that participates in the barrier must"
              " do so in all phases."
          )
        last_observed_phase = 0

      # Suppose our last observed phase was phase `p`...
      if last_observed_phase <= self.phase - 2:
        # Case 1: the next phase to complete is `p+3` or more, meaning we missed
        # our chance to observe phase `p+1` and must raise an error.
        raise ValueError(
            f"Thread {thread} is awaiting phase"
            f" {last_observed_phase + 1}, but barrier is already at phase"
            f" {self.phase}, which violates the invariant that threads must"
            " observe each barrier completion before any arrivals in the next"
            " phase."
        )
      elif last_observed_phase == self.phase - 1:
        # Case 2: the next phase to complete is `p+2`, so we're attempting to observe
        # phase `p+1`, which has completed already, and can proceed immediately.
        pass
      elif last_observed_phase == self.phase:
        # Case 3: we're attempting to observe phase `p+1`, which has not completed yet.
        # We must wait.
        while last_observed_phase == self.phase:
          if self.enable_logging and logging_info is not None:
            self._log(
                logging_info.format(
                    f"Waiting for barrier {id(self)} to reach phase"
                    f" {self.phase}.",
                    line_prefix="`wait`",
                )
            )
          self.cv.wait()

        if self.enable_logging and logging_info is not None:
          self._log(
              logging_info.format(
                  f"Thread {thread}: Finished"
                  f" waiting for phase {self.phase-1} of barrier {id(self)}.",
                  line_prefix="`wait`",
              )
          )
        # It's possible for us to wake up and find that the barrier has
        # completed multiple phases while we slept. This is fine: if it caused
        # us to miss a phase we'll catch it the next time we wait or on deallocation.
      else:
        assert False, "Unreachable"

      self.last_observed_phase_by_thread[thread] = last_observed_phase + 1

      # Read `self.clock` while still holding the lock on `self.cv`. (If race
      # detection is enabled, the clock is needed below to update a vector clock
      # that is managed by `self.shared_memory`.)
      clock = self.clock if self.detect_races else None
      smem_commit_clock = self.smem_commit_clock if self.detect_races else None

    # Note that this block cannot be nested under the `with self.cv` block
    # immediately above since this would violate the invariant that
    # `self.shared_memory.lock` *cannot* be acquired when `self.cv`'s lock is
    # already held. (See the documentation of `self.cv` above.)
    if self.detect_races:
      assert clock is not None
      assert smem_commit_clock is not None
      with self.shared_memory.lock:
        vc.update_vector_clock(
            self.shared_memory.clocks[thread], clock
        )
        vc.update_vector_clock(
            self.shared_memory.smem_commit_clocks[thread],
            smem_commit_clock,
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class HostAllocationRequest:
  """Request for an allocation on a device/thread and in a memory space."""

  memory_space_id: int
  device_id: int
  block_id: int
  # Defaults to zero for `AllocationRequest`s that do not specify a thread ID.
  thread_id: int = 0
  # The reference count is needed only for allocations that are explicitly
  # deallocated (with _deallocate_buffer below). This currently only applies to
  # allocations made by a `run_scoped` primitive.
  initial_ref_count: int = 1

  def __iter__(self):
    # We make `self` iterable to ease conversion into arrays (cf. method
    # `as_jax_array` below). Note that for this purpose it would suffice to have
    # any method that return a suitable iterator, instead of implementing the
    # special `__iter__` method. Not implementing `__iter__` would mean that
    # objects of this class cannot (accidentally) be iterated over by clients of
    # the class.
    return iter((
        self.memory_space_id,
        self.device_id,
        self.block_id,
        self.thread_id,
        self.initial_ref_count,
    ))

  @classmethod
  def shape_and_dtype(cls) -> jax.ShapeDtypeStruct:
    num_fields = len(dataclasses.fields(cls))
    return jax.ShapeDtypeStruct((num_fields,), jnp.int32)

  @property
  def as_np_array(self) -> np.ndarray:
    return np.array(list(self), dtype=np.int32)

  @classmethod
  def from_array(cls, request: jax.Array | np.ndarray) -> Self:
    if request.shape != cls.shape_and_dtype().shape:
      raise ValueError(
          f"Expected shape {cls.shape_and_dtype().shape} but got"
          f" {request.shape}"
      )
    if not interpret_utils.is_int(request.dtype):
      raise ValueError(f"Expected integer dtype but got {request.dtype}")

    arg_names = [f.name for f in dataclasses.fields(cls)]
    values = map(int, np.asarray(request).tolist())
    return cls(**dict(zip(arg_names, values)))


@dataclasses.dataclass(frozen=True, kw_only=True)
class HostAllocationKey(HostAllocationRequest):
  """Key for an allocation in shared memory."""

  buffer_id: int

  def __iter__(self):
    # Note that implementing `__iter__` here affects the bahviour of the
    # `as_array` and `as_jax_array` methods of the base class. This is intended.
    yield from super().__iter__()
    yield self.buffer_id


class AsyncTask(Protocol):
  """Async task to be run on some non-main thread (e.g. TMA or TensorCore)"""

  def __call__(self, tma_thread_id: int) -> None:
    """Execute the async task on the given thread."""
    ...


class ClusterBarrier(memory.Allocation):

  def __init__(
      self,
      shared_memory: GPUSharedMemory,
      *,
      axes_dims: tuple[int, ...],
      is_axis_collective: tuple[bool, ...],
      ref_count: int,
      num_arrivals: int,
      enable_logging: bool = False,
  ):
    """Initializes the ClusterBarrier.

    Args:
      shared_memory: The GPUSharedMemory instance managing this cluster barrier.
      axes_dims: The dimensions of the cluster axes and the final thread-block
        axis.
      is_axis_collective: Whether each of the cluster axes (or the final
        thread-block axis) is collective.
      ref_count: The initial reference count of the cluster barrier. This is
        typically the number of CPU threads among which the cluster barrier is
        shared.
      num_arrivals: Number of arrivals expected per thread block.
      enable_logging: Whether to enable logging of cluster barrier operations.
    """
    self.axes_dims = axes_dims
    self.is_axis_collective = is_axis_collective
    self.ref_count = ref_count  # protected by `self.lock`
    self.num_arrivals = num_arrivals
    self.enable_logging = enable_logging

    self.lock = threading.Lock()

    assert not is_axis_collective[-1]

    # The number of blocks along the cluster axes; equals the number of
    # 'normal' `Barrier`s we need to allocate (in `self.barriers`, see below) to
    # implement this `ClusterBarrier`.
    num_blocks_in_cluster = math.prod(axes_dims[:-1])

    num_blocks_for_arrival = 1 + sum(
        axis_dim - 1
        for axis_dim, is_collective in zip(
            axes_dims[:-1], is_axis_collective[:-1], strict=True
        )
        if is_collective
    )

    # Protected by `self.lock` (to avoid races between accessing and
    # deallocating barriers).
    self.barriers = [
        Barrier(
            shared_memory,
            num_pallas_threads_per_block=axes_dims[-1],
            # This `ClusterBarrier` is considered the only reference for each of
            # the underlying barriers.
            ref_count=1,
            num_arrivals=num_arrivals * num_blocks_for_arrival,
            enable_logging=enable_logging,
        )
        for _ in range(num_blocks_in_cluster)
    ]

  def _log(self, message: str):
    # Log every line separately to make sure `absl.logging` adds the correct
    # prefix (i.e. I*** <time> ... <source.py>:<line_number>) to each line in
    # `message`. This should not lead to mangled output within the logging for
    # `self` since the lock on `self.lock` is expected to be held whenever this
    # method is called. However, nothing keeps logged output from being
    # interleaved with logging from other (cluster) barriers or from the global
    # `SharedMemory` object.
    for msg in message.split("\n"):
      logging.info(msg)

  def has_zero_ref_count(self) -> bool:
    with self.lock:
      return self.ref_count == 0

  def arrive(
      self,
      *,
      mesh_location: MeshLocation,
      thread: Thread,
      clock: vc.VectorClock | None = None,
      smem_commit_clock: vc.VectorClock | None = None,
      logging_info: GPULoggingInfo | None = None,
  ):
    if self.enable_logging and logging_info is not None:
      with self.lock:
        self._log(
            logging_info.format(
                f"Arriving at cluster barrier {id(self)}",
                line_prefix="`arrive`",
            )
        )

    # Arrive at the barrier for the block that `thread` belongs to. Note that
    # this is the barrier for the block whose coordinate do *not* differ (along
    # any collective axis) from `block_coords`.
    with self.lock:
      barrier = self.barriers[thread.block_id]
    barrier.arrive(
        clock=clock,
        smem_commit_clock=smem_commit_clock,
        logging_info=logging_info,
    )

    # Arrive at the barriers for those blocks whose coordinates differ from
    # `block_coords` along *exactly one* collective axis.
    for i, (axis_dim, is_collective) in enumerate(
        zip(
            self.axes_dims[:-1],
            self.is_axis_collective[:-1],
            strict=True,
        )
    ):
      if not is_collective:
        continue

      for j in range(axis_dim):
        if j == mesh_location.block_coords[i]:
          # The barrier for the block with coordinates identical to
          # `block_coords` has already been arrived at above.
          continue

        block_coords_to_arrive_at = list(mesh_location.block_coords)
        # Note that (because of the `if ... continue` above) we have
        # `j != block_coords[i]` here.
        block_coords_to_arrive_at[i] = j
        barrier_index = np.ravel_multi_index(
            block_coords_to_arrive_at, self.axes_dims[:-1]
        )
        with self.lock:
          barrier = self.barriers[barrier_index]
        barrier.arrive(
            clock=clock,
            smem_commit_clock=smem_commit_clock,
            logging_info=logging_info,
        )

  def wait(
      self,
      thread: Thread,
      logging_info: GPULoggingInfo | None = None,
  ):
    if self.enable_logging and logging_info is not None:
      with self.lock:
        self._log(
            logging_info.format(
                f"Waiting for cluster barrier {id(self)}",
                line_prefix="`wait`",
            )
        )

    with self.lock:
      barrier = self.barriers[thread.block_id]

    barrier.wait(thread, logging_info)

  def deallocate(self):
    """Deallocates the `ClusterBarrier`."""
    with self.lock:
      self.ref_count -= 1
      if self.ref_count > 0:
        return

      for barrier in self.barriers:
        barrier.deallocate()


class GPUSharedMemory(memory.GenericSharedMemory[HostAllocationKey, Thread]):

  # TODO(paulbib): Is there a way to assert these match the base class?
  MemKey = HostAllocationKey
  ThreadKey = Thread

  # All Pallas threads that are concurrently executed by interpret mode.
  all_concurrent_threads: frozenset[ThreadKey]

  num_blocks_per_cluster: int

  next_tma_thread_id: int

  num_pallas_threads_per_block: int

  # thread -> next available REGS buffer ID.
  #
  # NOTE: We use negative integers so that, when debugging, it is easy to
  # visually distinguish REGS IDs from other IDs.
  next_regs_id: dict[ThreadKey, int]

  # Vector clocks used for tracking the logical time of the last observed call
  # to `commit_smem` for each thread.
  smem_commit_clocks: dict[ThreadKey, vc.VectorClock]

  # For each thread, a queue of clocks used for the read/write side of pending
  # smem_to_gmem transfers.
  pending_smem_to_gmem_read_clocks: dict[ThreadKey, collections.deque[vc.VectorClock]]
  pending_smem_to_gmem_write_clocks: dict[ThreadKey, collections.deque[vc.VectorClock]]

  def __init__(
      self,
      *,
      num_devices: int,
      out_of_bounds_reads: Literal["raise", "uninitialized"],
      dma_execution_mode: str,
      uninitialized_memory: Literal["nan", "zero"],
      detect_races: bool,
      barrier: threading.Barrier,
      clean_up_barrier: threading.Barrier,
      buffer_bounds: Literal["logical", "padded"] | None = None,
      logging_mode: params.LoggingMode | None = None,
      num_threads_per_block: int,
      num_blocks_per_cluster: int,
      num_tma_threads_per_device: int,
  ):
    self.all_concurrent_threads = frozenset(
        Warpgroup(device, cluster, block, warpgroup)
        for device in range(num_devices)
        for cluster in [0]
        for block in range(num_blocks_per_cluster)
        for warpgroup in range(num_threads_per_block)
    )

    self.num_tma_threads = num_devices * num_tma_threads_per_device

    # For a hypothetical mesh with 2 devices, two blocks per cluster, and 3
    # threads per block, entries in each vector clock are organized as follows:
    #   d0b0t0, d0b0t1, d0b0t2
    #   d0b1t0, d0b1t1, d0b1t2
    #   d1b0t0, d1b0t1, d1b0t2
    #   d1b1t0, d1b1t1, d1b1t2
    #   tma0, tma1, tma2, ...
    # Where dMbNtP denotes the P-th thread in the M-th block on the N-th device.
    # and tmaX denotes the X-th TMA thread.
    # TMA threads do not correspond to actual CPU threads, but they are used as
    # pseudo-threads for async TMA ops. We allocate a finite number which are
    # reused.
    # Note that the cluster id does not appear since we do not execute multiple
    # clusters concurrently.
    vector_clock_size = len(self.all_concurrent_threads) + self.num_tma_threads
    clocks: dict[Thread, np.ndarray] = {
        thread: vc.make_vector_clock(vector_clock_size)
        for thread in self.all_concurrent_threads
    }
    self.smem_commit_clocks = {
        thread: vc.make_vector_clock(vector_clock_size)
        for thread in self.all_concurrent_threads
    }

    super().__init__(
        num_devices=num_devices,
        out_of_bounds_reads=out_of_bounds_reads,
        dma_execution_mode=dma_execution_mode,
        uninitialized_memory=uninitialized_memory,
        detect_races=detect_races,
        vector_clock_size=vector_clock_size,
        clocks=clocks,
        barrier=barrier,
        clean_up_barrier=clean_up_barrier,
        buffer_bounds=buffer_bounds,
        logging_mode=logging_mode,
    )

    if self.dma_execution_mode != "eager":
      raise NotImplementedError(
          "Currently only eager DMA execution mode is supported when"
          " interpreting GPU kernels."
      )

    self.num_blocks_per_cluster = num_blocks_per_cluster
    self.next_tma_thread_id = 0
    self.num_pallas_threads_per_block = num_threads_per_block
    self.next_regs_id = collections.defaultdict(lambda: -100)
    self.pending_smem_to_gmem_read_clocks = {
        thread: collections.deque()
        for thread in self.all_concurrent_threads
    }
    self.pending_smem_to_gmem_write_clocks = {
        thread: collections.deque()
        for thread in self.all_concurrent_threads
    }

  def thread_to_vc_position(self, thread: ThreadKey) -> int:
    if isinstance(thread, Warpgroup):
      # see comment in __init__ for explanation of this mapping
      return (
          thread.device_id * self.num_blocks_per_cluster * self.num_pallas_threads_per_block
          + thread.block_id * self.num_pallas_threads_per_block
          + thread.warpgroup_id
      )
    elif isinstance(thread, Warp):
      raise NotImplementedError(
          "Warp-level thread keys are not supported for GPU kernel"
          " interpretation yet."
      )
    else:
      raise ValueError(f"Unsupported thread type: {type(thread)}")

  def get_next_tma_thread_id(self) -> int:
    with self.lock:
      # TODO(nrink): Consider adding an option for selecting TMA thread IDs
      # randomly (similar to how 'virtual' device IDs are selected randomly for
      # DMAs in TPU kernel interpret mode).
      next_tma_thread_id = self.next_tma_thread_id
      self.next_tma_thread_id = (next_tma_thread_id + 1) % self.num_tma_threads
      return len(self.all_concurrent_threads) + next_tma_thread_id

  def get_next_wgmma_accumulator_id(self, thread: ThreadKey) -> int:
    with self.lock:
      regs_id = self.next_regs_id[thread]
      self.next_regs_id[thread] = regs_id - 1
      return regs_id

  def allocate_barrier(
      self,
      key: MemKey,
      ref_count: int,
      num_arrivals: int,
      logging_info: GPULoggingInfo | None = None,
  ):
    """Allocates a barrier with the given key unless it already exists."""
    with self.lock:
      if key not in self.mem:
        barrier = Barrier(
            self,
            num_pallas_threads_per_block=self.num_pallas_threads_per_block,
            ref_count=ref_count,
            num_arrivals=num_arrivals,
            enable_logging=(
                self.logging_mode is not None
                and params.LoggingMode.BARRIER in self.logging_mode
            ),
        )
        self.mem[key] = barrier

        if self.enable_logging and logging_info is not None:
          self._log(
              logging_info.format(
                  "Allocated barrier"
                  f" {id(barrier)} ({barrier}) with key {key}.",
                  line_prefix="`allocate_barrier`",
              )
          )

  def get_barrier_and_increment_clock(
        self, key: MemKey, thread: ThreadKey
  ) -> tuple[Barrier | ClusterBarrier, vc.VectorClock | None]:
    clock = None
    with self.lock:
      if self.detect_races:
        clock = self.incr_clock(thread, take_lock=False)

      barrier = self.mem[key]

    if not isinstance(barrier, Barrier) and not isinstance(
        barrier, ClusterBarrier
    ):
      raise ValueError(
          f"Attempting to get barrier from allocation with {key} that is not a"
          " `Barrier` or `ClusterBarrier`."
      )

    return barrier, clock

  def get_barrier(self, key: MemKey) -> Barrier:
    with self.lock:
      barrier = self.mem[key]
    if not isinstance(barrier, Barrier):
      raise ValueError(
          f"Attempting to get barrier from allocation with {key} that is not a"
          " `Barrier`."
      )
    return barrier

  def deallocate_barrier(
      self,
      key: MemKey,
      logging_info: GPULoggingInfo | None = None,
  ):
    with self.lock:
      barrier = self.mem[key]
      if not isinstance(barrier, Barrier):
        raise ValueError(
            f"Attempting to get barrier from allocation with {key} that is not"
            " a `Barrier`."
        )

      if self.enable_logging and logging_info is not None:
        self._log(
            logging_info.format(
                "Decreasing ref count of"
                f" barrier {id(barrier)} with key {key}.",
                line_prefix="`deallocate_barrier`",
            )
        )

      barrier.deallocate()

      if barrier.has_zero_ref_count():
        if self.enable_logging and logging_info is not None:
          self._log(
              logging_info.format(
                  f"Deallocating barrier {id(barrier)} with key {key}.",
                  line_prefix="`deallocate_barrier`",
              )
          )
        self.mem.pop(key)

  # TODO(nrink): Consider unifying this method with `allocate_barrier`.
  def allocate_cluster_barrier(
      self,
      key: MemKey,
      axes_dims: tuple[int, ...],
      is_axis_collective: tuple[bool, ...],
      ref_count: int,
      num_arrivals: int,
      logging_info: GPULoggingInfo | None = None,
  ):
    """Allocates a cluster barrier with the given key unless it already exists."""
    with self.lock:
      if key not in self.mem:
        barrier = ClusterBarrier(
            self,
            axes_dims=axes_dims,
            is_axis_collective=is_axis_collective,
            ref_count=ref_count,
            num_arrivals=num_arrivals,
            enable_logging=(
                self.logging_mode is not None
                and params.LoggingMode.BARRIER in self.logging_mode
            ),
        )
        self.mem[key] = barrier

        if self.enable_logging and logging_info is not None:
          self._log(
              logging_info.format(
                  "Allocated cluster barrier"
                  f" {id(barrier)} ({barrier}) with key {key}.",
                  line_prefix="`allocate_cluster_barrier`",
              )
          )

  # TODO(nrink): Consider unifying this method with `deallocate_barrier`.
  def deallocate_cluster_barrier(
      self,
      key: MemKey,
      logging_info: GPULoggingInfo | None = None,
  ):
    with self.lock:
      barrier = self.mem[key]
      if not isinstance(barrier, ClusterBarrier):
        raise ValueError(
            f"Attempting to get cluster barrier from allocation with {key} that"
            " is not a `ClusterBarrier`."
        )

      if self.enable_logging and logging_info is not None:
        self._log(
            logging_info.format(
                "Decreasing ref count of"
                f" cluster barrier {id(barrier)} with key {key}.",
                line_prefix="`deallocate_cluster_barrier`",
            )
        )

      barrier.deallocate()

      if barrier.has_zero_ref_count():
        if self.enable_logging and logging_info is not None:
          self._log(
              logging_info.format(
                  "Deallocating cluster barrier"
                  f" {id(barrier)} with key {key}.",
                  line_prefix="`deallocate_cluster_barrier`",
              )
          )
        self.mem.pop(key)

  def assert_no_barriers_allocated(self):
    for key, alloc in self.mem.items():
      assert not isinstance(
          alloc, Barrier
      ), f"Barrier remains allocated at key `{key}`."
      assert not isinstance(
          alloc, ClusterBarrier
      ), f"Cluster barrier remains allocated at key `{key}`."

  def update_smem_commit_clock(self, thread: ThreadKey):
    """Sets the smem commit clock for the given core to its current clock."""
    if not self.detect_races:
      return
    with self.lock:
      vc.update_vector_clock(
          self.smem_commit_clocks[thread], self.clocks[thread]
      )

  def get_smem_commit_clock(self, thread: ThreadKey) -> vc.VectorClock | None:
    if not self.detect_races:
      return None
    with self.lock:
      return vc.copy_vector_clock(self.smem_commit_clocks[thread])

  def execute_async_task(self, task: AsyncTask, thread: ThreadKey):
    """Executes an async task immediately (intiated by the given thread)."""
    self.incr_clock(thread)

    tma_thread_id = self.get_next_tma_thread_id()
    task(tma_thread_id)

  def add_copy_smem_to_gmem_clocks(
      self,
      thread: ThreadKey,
      read_clock: vc.VectorClock,
      write_clock: vc.VectorClock,
  ):
    """Records read and write clocks for a completed copy from SMEM to GMEM."""
    with self.lock:
      self.pending_smem_to_gmem_read_clocks[thread].append(read_clock)
      self.pending_smem_to_gmem_write_clocks[thread].append(
          write_clock
      )

  def wait_smem_to_gmem(
      self, thread: ThreadKey, n: int, wait_read_only: bool
  ):
    """Ensures no more than n SMEM to GMEM copies are outstanding."""
    # TODO(paulbib): if copies were actually async, they would be run here.
    with self.lock:
      self.incr_clock(thread, take_lock=False)
      while len(self.pending_smem_to_gmem_read_clocks[thread]) > n:
        vc.update_vector_clock(
            self.clocks[thread],
            self.pending_smem_to_gmem_read_clocks[thread].popleft(),
        )

      if not wait_read_only:
        while len(self.pending_smem_to_gmem_write_clocks[thread]) > n:
          vc.update_vector_clock(
              self.clocks[thread],
              self.pending_smem_to_gmem_write_clocks[
                  thread
              ].popleft(),
          )

  def kernel_thread_finished(self, thread: ThreadKey):
    """Called when a thread completes execution of a kernel."""
    with self.lock:
      if self.detect_races:
        # The PTX docs are not explicit about this, but we believe that it is
        # necessary and sufficient to make sure the read side of any async
        # smem to gmem copies have completed before the kernel finishes.
        if len(self.pending_smem_to_gmem_read_clocks[thread]) > 0:
          raise ValueError(
              "Not all copy_smem_to_gmem read-side operations completed before"
              f" kernel finished on thread {thread}."
          )

  def concurrent_threads(self, device: Device) -> list[Thread]:
    """Returns all concurrent threads on the given device."""
    return [
        thread
        for thread in self.all_concurrent_threads
        if thread.device_id == device.device_id
    ]

  def update_clocks_for_device_barrier(self, device: Device):
    """Synchronizes the vector clocks for the cores on the given device."""
    self.update_clocks(self.concurrent_threads(device))
