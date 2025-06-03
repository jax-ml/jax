# Copyright 2024 The JAX Authors.
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
import enum
import functools
import gc
import itertools
import math
import threading
from typing import Any, Callable, Literal, cast

import jax
from jax import lax
from jax._src import callback
from jax._src import core as jax_core
from jax._src.lax.control_flow import for_loop
from jax._src import linear_util as lu
from jax._src import source_info_util
from jax._src.pallas.mosaic import core as mosaic_core
from jax._src.pallas.mosaic import primitives as mosaic_primitives
from jax._src.pallas.mosaic import verification
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives
from jax._src import pjit
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.state import discharge as state_discharge
from jax._src.state import indexing
from jax._src.state import primitives as state_primitives
from jax._src.typing import Array
from jax._src.util import (
    safe_map,
    safe_zip,
    split_list
)
from jax.interpreters import partial_eval as pe
import jax.numpy as jnp
import numpy as np


map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

Grid = pallas_core.Grid
TupleGrid = pallas_core.TupleGrid
GridSpec = pallas_core.GridSpec
BlockMapping = pallas_core.BlockMapping
GridMapping = pallas_core.GridMapping
BlockSpec = pallas_core.BlockSpec
BlockSpecTree = pallas_core.BlockSpecTree
NoBlockSpec = pallas_core.NoBlockSpec
no_block_spec = pallas_core.no_block_spec
ScratchShapeTree = pallas_core.ScratchShapeTree
CostEstimate = pallas_core.CostEstimate


@dataclasses.dataclass(frozen=True)
class TPUInterpretParams:
  """Parameters for Mosaic TPU interpret mode.

  Attributes:
    dma_execution_mode:  If "eager", DMAs are executed as soon as they are
      issued.  If "on_wait", DMA reads or writes are only executed when a device
      is waiting on a DMA semaphore that will be signaled when the read or write
      is complete.
      Default: "on_wait".
    detect_races: If True, a dynamic, happens-before race detector will be used
      to detect data races during kernel interpretation.  If any races are
      detected, a message will be printed and `races.races_found` will be set to
      True.
      Default: False.
    skip_floating_point_ops: If True, operations that produce only floating
      point values will not be interpreted; instead, their results will be
      replaced with arrays all of `jnp.inf`. Additionaly any floating point
      operands to any operation will be replaced with (arrays of) `jnp.inf`.
      Default: False.
    uninitialized_memory: If "nan", allocated buffers are initialized to contain
      all NaNs (or to their maximum possible value for integers). If "zero",
      allocated buffers are initialized to all zeros.
      Default: "nan".
    random_seed: Seed for random number generator used during interpretation.
      Currently random numbers are used to randomize the grid coordinates along
      dimensions with 'parallel' semantics.
      Default: None.
    grid_point_recorder: Callback that is invoked by the interpreter for each
      grid point in the order in which the grid points are traversed. The
      callback is invoked with two arguments:
        - A tuple of grid coordinates.
        - The local core ID of the core that is processing the grid point.
      This callback is intended for inspecting
        - the randomization of coordinates along grid dimensions with 'parallel'
          semantics and
        - the mapping of grid points to local (i.e. per-device) cores.
      Default: None.
    num_cores_per_device: The number of cores per device.
      Default: 1.
  """
  dma_execution_mode: Literal["eager", "on_wait"] = "on_wait"
  detect_races: bool = False
  skip_floating_point_ops: bool = False
  uninitialized_memory: Literal["nan", "zero"] = "nan"
  random_seed: int | None = None
  grid_point_recorder: (
      Callable[[tuple[np.int32, ...], np.int32], None] | None
  ) = None
  num_cores_per_device: int = 1


VectorClock = np.ndarray

# Conceptually, each DMA runs on its own, independent device.  Representing
# this precisely would require vector clocks to have sizes linear in the number
# of DMAs.
#
# Instead, we use approximate vector clocks of fixed size.  We assign each DMA
# a virtual core ID in the range
#   [num_devices*num_cores_per_device + 1, NUM_VIRTUAL_CORES],
# and each operation of a DMA increments the corresponding coordinate in its
# vector clock. (So the "virtual" part of a vector clock is effectively
# counting, for each virtual core, the number of DMAs that happened-before
# the vector clock and were assigned to that virtual core.)
#
# If two approximate clocks are unordered, then their corresponding events are
# not ordered by the happens-before relation.  So this approximation will not
# introduce any false positives in detecting data races.  But we may fail to
# detect some true data races because there can be cases where two approximate
# clocks are ordered, and we will treat the corresponding events as ordered
# by the happens-before relation, but the corresponding events are not
# actually ordered.
NUM_VIRTUAL_CORES = 32

def make_vector_clock(_: int) -> VectorClock:
  del _
  return np.zeros(NUM_VIRTUAL_CORES, dtype=np.int32)

def copy_vector_clock(x: VectorClock) -> VectorClock:
  if x is None:
    return None
  return x.copy()

def update_vector_clock(x: VectorClock, y: VectorClock):
  x[:] = np.maximum(x[:], y[:])

def lt(x: VectorClock, y: VectorClock) -> bool:
  return bool((x <= y).all() & (x < y).any())

def ordered(x: VectorClock, y: VectorClock) -> bool:
  return lt(x, y) | lt(y, x)

def inc_vector_clock(x: VectorClock, global_core_id: int):
  if global_core_id >= len(x):
    raise ValueError(f'device_id={global_core_id} is out of range for x={x}')
  assert global_core_id < len(x)
  x[global_core_id] += 1

def _get_global_core_id(device_id, local_core_id):
  """Computes the global core ID from the given device and local core ID."""
  device_id = int(device_id)
  local_core_id = int(local_core_id)
  return device_id * _get_shared_memory().num_cores_per_device + local_core_id


class Semaphore:
  def __init__(self, semaphore_id=None):
    shared_memory = _get_shared_memory()

    self.id = semaphore_id

    # TODO(jburnim): Use one Condition variable per device.  (Which will be
    # easier to do when we're using single integer device IDs.)
    self.cv = threading.Condition()

    self.counts = np.zeros(shared_memory.num_cores, dtype=np.int32)

    self.interpret_params = shared_memory.interpret_params
    if self.interpret_params.detect_races:
      # We associate a vector clock with each count in self.counts.  Whenever
      # self.counts[i] is signaled, self.clocks[i] is updated with the vector
      # clock of the signaling core.  Whenever core i successfully waits on
      # self.counts[i], the vector clock of core i is updated with
      # self.clocks[i].
      #
      # TODO(jburnim): Model happens-before more precisely for the case where
      # semaphores are over-signaled.
      self.clocks = [None] * shared_memory.num_cores

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
      self.counts[global_core_id] += inc
      if self.interpret_params.detect_races:
        if self.clocks[global_core_id] is None:
          self.clocks[global_core_id] = copy_vector_clock(clock)
        else:
          update_vector_clock(self.clocks[global_core_id], clock)
      self.cv.notify_all()

  def read(self, global_core_id):
    with self.cv:
      return self.counts[global_core_id]

  def wait(self, value, global_core_id, *, is_dma=False):
    global_core_id = int(global_core_id)
    shared_memory = _get_shared_memory()

    # TODO(jburnim):
    #  - If the count is larger than value, raise an error?
    #  - If the count is equal to value, but there DMAs waiting to signal us,
    #    raise an error?

    # Simple implementation for non-DMA semaphores.
    if not is_dma or (self.interpret_params.dma_execution_mode == "eager"):
      with self.cv:
        while self.counts[global_core_id] < value:
          self.cv.wait()
        self.counts[global_core_id] -= value
        if self.interpret_params.detect_races:
          clock = copy_vector_clock(self.clocks[global_core_id])
      if self.interpret_params.detect_races:
        with shared_memory.lock:
          update_vector_clock(shared_memory.clocks[global_core_id], clock)
      return

    # For DMA semaphores (when dma_execution_mode=='on_wait'), while our count
    # is not large enough we will select and partially execute pending DMAs
    # until our count is large enough.
    #
    # This approach will tend to run DMAs as late as possible, as well as
    # out-of-order.  This approach also lets us avoid the complexity of spinning
    # up separate threads to handle executing DMAs.
    shared_memory = _get_shared_memory()
    while True:
      clock = None
      with self.cv:
        if self.counts[global_core_id] >= value:
          self.counts[global_core_id] -= value
          if self.interpret_params.detect_races:
            clock = copy_vector_clock(self.clocks[global_core_id])
          else:
            return
      if clock is not None:
        with shared_memory.lock:
          update_vector_clock(shared_memory.clocks[global_core_id], clock)
        return

      with shared_memory.lock:
        dma_queue = shared_memory.dmas_by_sem[self.id]
        if len(dma_queue) > 0:
          dma = dma_queue.pop()
        else:
          continue

      # Only execute the DMA as far as necessary to signal us.
      assert (dma.src_sem is self) or (dma.dst_sem is self)
      with dma.lock:
        if dma.virtual_device_id is None:
          dma.virtual_device_id = np.random.randint(
              shared_memory.num_devices, NUM_VIRTUAL_CORES)

        if dma.state == DmaState.STARTED:
          # Do the read.
          if self.interpret_params.detect_races:
            inc_vector_clock(dma.clock, dma.virtual_device_id)
          dma.data = get(dma.src_device_id,
                         dma.src_local_core_id,
                         dma.src_memory_space,
                         dma.src_buffer_id,
                         dma.src_transforms,
                         clock=copy_vector_clock(dma.clock),
                         src_device_id=dma.id,
                         src_local_core_id=0,
                         source_info=dma.source_info)
          if self.interpret_params.detect_races:
            inc_vector_clock(dma.clock, dma.virtual_device_id)
          if dma.src_sem is not None:
            data_size = dma.data.itemsize * dma.data.size
            dma.src_sem.signal(
                data_size,
                global_core_id=_get_global_core_id(
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
        if self.interpret_params.detect_races:
          inc_vector_clock(dma.clock, dma.virtual_device_id)
        store(dma.dst_device_id,
              dma.dst_local_core_id,
              dma.dst_memory_space,
              dma.dst_buffer_id,
              dma.dst_transforms,
              dma.data,
              clock=copy_vector_clock(dma.clock),
              src_device_id=dma.id,
              src_local_core_id=0,
              source_info=dma.source_info)
        if self.interpret_params.detect_races:
          inc_vector_clock(dma.clock, dma.virtual_device_id)
        data_size = dma.data.itemsize * dma.data.size
        dma.dst_sem.signal(
            data_size,
            global_core_id=_get_global_core_id(
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

  clock: VectorClock

  source_info: source_info_util.SourceInfo | None = None

  state: DmaState = DmaState.STARTED
  data: np.ndarray | None = None
  virtual_device_id: int | None = None
  lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


@dataclasses.dataclass
class RaceDetectionState:
  num_cores: int


  # (memory_space, buffer_id, device_id, local_core_id) -> [(device_id, local_core_id, VectorClock, range)]
  reads: dict = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list))

  # (memory_space, buffer_id, device_id, local_core_id) -> [(device_id, local_core_id, VectorClock, range)]
  writes: dict = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list))

  lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

  races_found: bool = False

def _is_empty_slice(slice_or_idx: slice | int):
  if isinstance(slice_or_idx, int) or (slice_or_idx == slice(None)):
    return False

  # NOTE: All slices here will have known size.
  start = int(slice_or_idx.start) if slice_or_idx.start is not None else 0
  stop = int(slice_or_idx.stop)
  return (start < stop)

def slices_overlap(slice_or_idx1: slice | int, slice_or_idx2: slice | int):
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

  # NOTE: We are only comparing slices with known stops (and sizes).
  # Do we need to handle zero-length slices?
  return ((slice_or_idx1.start <= slice_or_idx2.start < slice_or_idx1.stop)
          | (slice_or_idx2.start <= slice_or_idx1.start < slice_or_idx2.stop))

def ranges_overlap(range1: tuple[slice | int, ...],
                   range2: tuple[slice | int, ...]) -> bool:
  return all(slices_overlap(r1, r2) for r1, r2
             in itertools.zip_longest(range1, range2, fillvalue=slice(None)))


def check_read(
    device_id, local_core_id, clock, buffer_key, rnge, source_info=None
):
  if source_info is not None:
    user_frame = source_info_util.summarize(source_info)
  else:
    user_frame = 'pallas_call'

  with races.lock:
    writes = races.writes[buffer_key]
    num_writes = len(writes)
    races.reads[buffer_key].append(
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
    if ordered(write_clock, clock):
      continue
    if not ranges_overlap(rnge, write_range):
      continue
    # TODO(jburnim): When printing device IDs for reads/writes, distinguish
    # between real device IDs vs. DMA IDs.
    print(
        f'RACE DETECTED\n  read of {buffer_key}[{rnge}] from {device_id},'
        f' {local_core_id}, {user_frame}\n  write of'
        f' {buffer_key}[{write_range}] from {write_device_id},'
        f' {write_local_core_id} {write_frame}'
    )
    with races.lock:
      races.races_found = True
    return


def check_write(device_id, local_core_id, clock, buffer_key, rnge, source_info=None):
  if source_info is not None:
    user_frame = source_info_util.summarize(source_info)
  else:
    user_frame = 'pallas_call'

  with races.lock:
    writes = races.writes[buffer_key]
    reads = races.reads[buffer_key]
    num_writes = len(writes)
    num_reads = len(reads)
    races.writes[buffer_key].append((device_id, local_core_id, clock, rnge, user_frame))

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
    if ordered(write_clock, clock):
      continue
    if not ranges_overlap(rnge, write_range):
      continue
    # TODO(jburnim): When printing device IDs for reads/writes, distinguish
    # between real device IDs vs. DMA IDs.
    print(
        f'RACE DETECTED\n  write of {buffer_key}[{rnge}] from {device_id},'
        f' {local_core_id}, {user_frame}\n  write of'
        f' {buffer_key}[{write_range}] from {write_device_id},'
        f' {write_local_core_id}, {write_frame}'
    )
    with races.lock:
      races.races_found = True
    break

  for i in range(num_reads):
    read_device_id, read_local_core_id, read_clock, read_range, read_frame = (
        reads[i]
    )
    if ordered(read_clock, clock):
      continue
    if not ranges_overlap(rnge, read_range):
      continue
    # TODO(jburnim): When printing device IDs for reads/writes, distinguish
    # between real device IDs vs. DMA IDs.
    print(
        f'RACE DETECTED\n  write of {buffer_key}[{rnge}] from {device_id},'
        f' {local_core_id}, {user_frame}\n  read of {buffer_key}[{read_range}]'
        f' from {read_device_id}, {read_local_core_id}, {read_frame}'
    )
    with races.lock:
      races.races_found = True
    return


@dataclasses.dataclass
class SharedMemory:
  interpret_params: TPUInterpretParams
  num_devices: int
  num_cores_per_device: int
  clocks: list[VectorClock]
  barrier: threading.Barrier
  clean_up_barrier: threading.Barrier

  # (memory_space, buffer_id, device_id, local_core_id) -> NumPy array
  # TODO(jburnim): Handle Megacore.
  mem: dict[tuple[str, int, int, int], np.ndarray] = dataclasses.field(
      default_factory=dict)

  # semaphore_id -> Semaphore
  sem: dict[int, Semaphore] = dataclasses.field(default_factory=dict)

  # (semaphore_id, device_id)
  #   -> list of DMAs that will signal the semaphore on the given device
  # TODO(jburnim): Fix uses of `dmas_by_sem` to align with the two lines of
  # documentation above, i.e. index `dmas_by_sem` with
  # `(semaphore_id, device_id)` (currently indexed with `semaphore_id only).
  dmas_by_sem: dict[tuple[int, int], list[DMA]] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list))

  lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

  # (device_id, local_core_id) -> next buffer ID
  next_buffer_id: dict[tuple[int, int], int] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(lambda: 100))
  # global_core_id -> next semaphore ID
  next_semaphore_id: dict[int, int] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(lambda: 2000))

  next_dma_id: int = 100

  deallocated_bytes: int = 0

  @property
  def num_cores(self) -> int:
    return self.num_devices * self.num_cores_per_device


# TODO(jburnim): Do we want to support multiple instances of SharedMemory?
# Maybe for running multiple distinct interpreted computations in parallel?
_shared_memory : SharedMemory | None = None
_shared_memory_init_lock = threading.Lock()
races : RaceDetectionState | None = None

def _get_shared_memory() -> SharedMemory:
  assert _shared_memory is not None
  return _shared_memory

def _clear_shared_memory():
  global _shared_memory
  with _shared_memory_init_lock:
    _shared_memory = None


def _initialize_shared_memory(
    device_id, num_devices, num_cores_per_device, *, interpret_params
):
  global _shared_memory
  del device_id
  num_devices = int(num_devices)
  num_cores_per_device = int(num_cores_per_device)
  num_cores = num_devices * num_cores_per_device
  with _shared_memory_init_lock:
    if _shared_memory is None:
      _shared_memory = SharedMemory(
          interpret_params=interpret_params,
          num_devices=num_devices,
          num_cores_per_device=num_cores_per_device,
          clocks=[make_vector_clock(num_cores) for _ in range(num_cores)],
          barrier=threading.Barrier(
              num_devices, action=_update_clocks_for_global_barrier),
          clean_up_barrier=threading.Barrier(
              num_devices, action=_clear_shared_memory))
  assert _shared_memory.num_cores == num_cores

  global races
  races = RaceDetectionState(num_cores=num_cores)

def _update_clocks(low_global_core_id, high_global_core_id):
  """Synchronizes the vector clocks for the cores with ids in the range between the two arguments."""
  shared_memory = _get_shared_memory()
  # Despite only updating the vector clocks for some cores, we still need to
  # hold the global lock to ensure that no other devices are concurrently
  # accessing the same vector clocks.
  with shared_memory.lock:
    for c in shared_memory.clocks[low_global_core_id + 1 : high_global_core_id]:
      update_vector_clock(shared_memory.clocks[low_global_core_id], c)
    for c in shared_memory.clocks[low_global_core_id + 1 : high_global_core_id]:
      update_vector_clock(c, shared_memory.clocks[low_global_core_id])

def _update_clocks_for_device_barrier(device_id):
  """Synchronizes the vector clocks for the cores on the given device."""
  shared_memory = _get_shared_memory()
  low_core_id = device_id * shared_memory.num_cores_per_device
  high_core_id = (device_id + 1) * shared_memory.num_cores_per_device
  _update_clocks(low_core_id, high_core_id)

def _update_clocks_for_global_barrier():
  """Synchronizes all vector clocks."""
  shared_memory = _get_shared_memory()
  _update_clocks(0, shared_memory.num_cores)

def _barrier(device_id):
  device_id = int(device_id)
  shared_memory = _get_shared_memory()
  if shared_memory.num_devices > 1:
    shared_memory.barrier.wait()

def _clean_up_shared_memory(device_id):
  device_id = int(device_id)
  shared_memory = _get_shared_memory()
  shared_memory.clean_up_barrier.wait()

def _validate(device_id):
  device_id = int(device_id)

  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    for sem in shared_memory.sem.values():
      with sem.cv:
        if sem.counts[device_id] != 0:
          # TODO(jburnim): Make this raise an error, but in a way that doesn't
          # cause other devices to hang later in `_clean_up_shared_memory`.
          print(
              f'Semaphore {sem.id} has non-zero count for {device_id} at '
              f'kernel exit: {sem.counts[device_id]}')

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

  shared_memory = _get_shared_memory()

  if local_core_id is None:
    local_core_id_int = 0
    local_core_ids = tuple(range(shared_memory.num_cores_per_device))
  else:
    local_core_id_int = int(local_core_id)
    local_core_ids = (local_core_id_int,)
  del local_core_id

  local_core_id_to_buffer_id = {}
  with shared_memory.lock:
    for lci in local_core_ids:
      buffer_id = shared_memory.next_buffer_id[(device_id, lci)]
      shared_memory.next_buffer_id[(device_id, lci)] = buffer_id + 1
      if lci == 0 or memory_space_str != 'any':
        # If allocating in HBM, only actually allocate a buffer for local core
        # id 0.
        # TODO(jburnim): Add options for initializing memory (e.g., with NaNs,
        # with zeros, or with the buffer ID).
        shared_memory.mem[(memory_space_str, buffer_id, device_id, lci)] = val

      local_core_id_to_buffer_id[lci] = buffer_id

  # The buffer ids should always be kept in sync across all cores.
  assert all(
      buffer_id == local_core_id_to_buffer_id[local_core_id_int]
      for buffer_id in local_core_id_to_buffer_id.values()
  )

  # TODO(jburnim): Raise an error if buffer_id is too big for int16.
  return np.int16(local_core_id_to_buffer_id[local_core_id_int])

def _deallocate_buffer(device_id, local_core_id, memory_space, buffer_id):
  device_id = int(device_id)
  local_core_id = int(local_core_id)
  memory_space = TPU_MEMORY_SPACE_NAMES[int(memory_space)]
  buffer_id = int(buffer_id)

  if memory_space == 'any':
    local_core_id = 0

  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    buff = shared_memory.mem.pop(
        (memory_space, buffer_id, device_id, local_core_id)
    )
    shared_memory.deallocated_bytes += buff.size * buff.itemsize
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

  shared_memory = _get_shared_memory()

  if local_core_id is None:
    local_core_id_int = 0
    global_core_ids = tuple(
        _get_global_core_id(device_id, core_id)
        for core_id in range(shared_memory.num_cores_per_device)
    )
  else:
    local_core_id_int = int(local_core_id)
    global_core_ids = (_get_global_core_id(device_id, local_core_id_int),)
  del local_core_id

  global_core_id_to_semaphore_id = {}
  with shared_memory.lock:
    for gci in global_core_ids:
      semaphore_id = shared_memory.next_semaphore_id[gci]
      shared_memory.next_semaphore_id[gci] = (
          semaphore_id + num_semaphores
      )

      # Ensure that only one global `Semaphore` object is allocated for each
      # `semaphore_id`.
      for i in range(semaphore_id, semaphore_id + num_semaphores):
        if i not in shared_memory.sem:
          shared_memory.sem[i] = Semaphore(i)

      global_core_id_to_semaphore_id[gci] = semaphore_id

  global_core_id = _get_global_core_id(device_id, local_core_id_int)
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


TPU_MEMORY_SPACE_IDXS : dict[mosaic_core.MemorySpace | pallas_core.MemorySpace | None, int] = {
    v: i for i, v in enumerate(mosaic_core.MemorySpace)}
TPU_MEMORY_SPACE_IDXS[pallas_core.MemorySpace.ANY] = (
    TPU_MEMORY_SPACE_IDXS[mosaic_core.MemorySpace.ANY])
TPU_MEMORY_SPACE_NAMES = {
    i: v.value for i, v in enumerate(mosaic_core.MemorySpace)}

# Default to VMEM when no memory space is specified.
TPU_MEMORY_SPACE_IDXS[None] = (
    TPU_MEMORY_SPACE_IDXS[mosaic_core.MemorySpace.VMEM])

def get_barrier_semaphore(device_id, collective_id):
  del device_id
  collective_id = int(collective_id)

  # TODO(jburnim): Check/fix so that IDs for barrier semaphores do not conflict
  # with IDs for regular or DMA semaphores.  (For example, store them in a
  # different table.)
  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    semaphore_id = collective_id
    if semaphore_id not in shared_memory.sem:
      shared_memory.sem[semaphore_id] = Semaphore()

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
      ret.append(slice_or_idx1[i].start + slice_or_idx2[j] * slice_or_idx1[i].step)
      i += 1
      j += 1
    else:
      ret.append(slice(
          slice_or_idx1[i].start + slice_or_idx2[j].start * slice_or_idx1[i].step,
          slice_or_idx1[i].start + slice_or_idx2[j].stop * slice_or_idx1[i].step,
          slice_or_idx1[i].step * slice_or_idx2[j].step
      ))
      i += 1
      j += 1

def _to_range(transforms) -> tuple[slice | int, ...]:
  ret = ()
  for transform in transforms:
    # For now, assume only NDIndexer transforms.
    ret = _compose_slice_or_index(
        ret, tuple(_transform_slice_or_index(i) for i in transform.indices))
  return ret

def _to_int(x : int | Array | None) -> int | None:
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
    *,
    src_device_id=None,
    src_local_core_id=None,
    clock=None,
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
  src_device_id = _to_int(src_device_id)
  src_local_core_id = _to_int(src_local_core_id)

  local_core_id_for_buffer = 0 if memory_space == 'any' else local_core_id
  global_core_id = _get_global_core_id(device_id, local_core_id)

  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    read_range = _to_range(transforms)
    if shared_memory.interpret_params.detect_races:
      inc_vector_clock(shared_memory.clocks[global_core_id], global_core_id)
      if clock is None:
        clock = copy_vector_clock(shared_memory.clocks[global_core_id])
    buffer = shared_memory.mem[
        (memory_space, buffer_id, device_id, local_core_id_for_buffer)
    ]
    ret = buffer[read_range].copy()
    if transforms:
      # TODO(jburnim): Instead of using NDIndexer, do the computation ourselves
      # with buffer.shape and read_range?
      expected_shape = transforms[-1].get_indexer_shape()
      if expected_shape != ret.shape[:len(expected_shape)]:
        raise ValueError(
            'Out-of-bounds read of'
            f' ({device_id} {local_core_id} {memory_space} {buffer_id}):'
            f' reading [{read_range}] but bufer has shape {buffer.shape} .'
        )

  if shared_memory.interpret_params.detect_races:
    if src_device_id is None:
      src_device_id = device_id
    if src_local_core_id is None:
      src_local_core_id = local_core_id
    check_read(
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
    *,
    src_device_id=None,
    src_local_core_id=None,
    clock=None,
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
  src_device_id = _to_int(src_device_id)
  src_local_core_id = _to_int(src_local_core_id)

  local_core_id_for_buffer = 0 if memory_space == 'any' else local_core_id
  global_core_id = _get_global_core_id(device_id, local_core_id)

  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    if shared_memory.interpret_params.detect_races:
      inc_vector_clock(shared_memory.clocks[global_core_id], global_core_id)
      if clock is None:
        clock = copy_vector_clock(shared_memory.clocks[global_core_id])

    buff = shared_memory.mem[
        (memory_space, buffer_id, device_id, local_core_id_for_buffer)
    ]
    assert buff.dtype == val.dtype  # TODO(jburnim): Catch this statically.
    write_range = _to_range(transforms)
    # TODO(jburnim): Better error message if this raises?
    in_bounds_shape = buff[write_range].shape
    if in_bounds_shape != val.shape:
      raise ValueError(
          'Out-of-bounds write of'
          f' ({device_id} {local_core_id} {memory_space} {buffer_id}): writing'
          f' [{write_range}] but buffer has shape {buff.shape} .'
      )
    buff[write_range] = val

  if shared_memory.interpret_params.detect_races:
    if src_device_id is None:
      src_device_id = device_id
    if src_local_core_id is None:
      src_local_core_id = local_core_id
    check_write(
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
    # jax.debug.print(f'swap: {transforms}')
  except:
    raise ValueError('Advanced indexers are not supported on TPU')
  val = np.array(val)
  mask = np.array(mask) if mask is not None else None
  if mask is not None:
    assert mask.shape == val.shape

  local_core_id_for_buffer = 0 if memory_space == 'any' else local_core_id
  global_core_id = _get_global_core_id(device_id, local_core_id)

  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    if shared_memory.interpret_params.detect_races:
      inc_vector_clock(shared_memory.clocks[global_core_id], global_core_id)
      clock = copy_vector_clock(shared_memory.clocks[global_core_id])
    buff = shared_memory.mem[
        (memory_space, buffer_id, device_id, local_core_id_for_buffer)
    ]
    assert buff.dtype == val.dtype  # TODO(jburnim): Catch this statically.
    read_write_range = _to_range(transforms)
    # TODO(jburnim): Better error message if this raises?
    raw_result = buff[read_write_range]
    in_bounds_shape = raw_result.shape
    if mask is None:
      if in_bounds_shape != val.shape:
        raise ValueError(
            'Out-of-bounds swap of'
            f' ({device_id} {local_core_id} {memory_space} {buffer_id}):'
            f' swapping [{read_write_range}] but buffer has shape'
            f' {buff.shape} .'
        )
      buff[read_write_range] = val
      return raw_result.copy()

    in_bounds_mask = np.full(mask.shape, True)
    for i in range(len(in_bounds_shape)):
      in_bounds_mask[in_bounds_shape[i]:] = False
    if (~in_bounds_mask & mask).any():
      # TODO(jburnim): Include indices of out-of-bounds locations where mask
      # is True.
      raise ValueError(
          'Out-of-bounds masked swap of'
          f' ({device_id} {local_core_id} {memory_space} {buffer_id}): swapping'
          f' [{read_write_range}] but buffer has shape {buff.shape} . '
      )

    in_bounds_idx = tuple(slice(i) for i in in_bounds_shape)
    result = val.copy()
    result[in_bounds_idx] = np.where(
        mask[in_bounds_idx], raw_result, val[in_bounds_idx])
    buff[read_write_range] = np.where(
        mask[in_bounds_idx], val[in_bounds_idx], raw_result)

  if shared_memory.interpret_params.detect_races:
    check_write(
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
  shared_memory = _get_shared_memory()
  with dma.lock:
    assert dma.state == DmaState.STARTED

    if dma.virtual_device_id is None:
      # See comment in Semaphore.wait .
      dma.virtual_device_id = np.random.randint(
          shared_memory.num_cores, NUM_VIRTUAL_CORES)

    # Do the read.
    if shared_memory.interpret_params.detect_races:
      inc_vector_clock(dma.clock, dma.virtual_device_id)
    dma.data = get(dma.src_device_id,
                   dma.src_local_core_id,
                   dma.src_memory_space,
                   dma.src_buffer_id,
                   dma.src_transforms,
                   clock=copy_vector_clock(dma.clock),
                   src_device_id=dma.id,
                   src_local_core_id=0,
                   source_info=dma.source_info)
    data_size = dma.data.itemsize * dma.data.size

    # Signal the send semaphore.
    if shared_memory.interpret_params.detect_races:
      inc_vector_clock(dma.clock, dma.virtual_device_id)
    if dma.src_sem is not None:
      dma.src_sem.signal(
          data_size,
          global_core_id=_get_global_core_id(
              dma.src_device_id, dma.src_local_core_id
          ),
          clock=dma.clock,
      )
    dma.state = DmaState.READ

    # Do the write.
    if shared_memory.interpret_params.detect_races:
      inc_vector_clock(dma.clock, dma.virtual_device_id)
    store(dma.dst_device_id,
          dma.dst_local_core_id,
          dma.dst_memory_space,
          dma.dst_buffer_id,
          dma.dst_transforms,
          dma.data,
          clock=copy_vector_clock(dma.clock),
          src_device_id=dma.id,
          src_local_core_id=0,
          source_info=dma.source_info)

    # Signal the receive semaphore.
    if shared_memory.interpret_params.detect_races:
      inc_vector_clock(dma.clock, dma.virtual_device_id)
    if dma.dst_sem is not None:
      dma.dst_sem.signal(
          data_size,
          global_core_id=_get_global_core_id(
              dma.dst_device_id, dma.dst_local_core_id
          ),
          clock=dma.clock,
      )

    dma.data = None
    dma.state = DmaState.COMPLETED

def print_memory(device_id):
  device_id = int(device_id)
  if all(d == 0 for d in device_id):
    shared_memory = _get_shared_memory()
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
  device_id = int(device_id)
  src_local_core_id = int(src_local_core_id)
  src_global_core_id = _get_global_core_id(device_id, src_local_core_id)
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

  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    dst_sem = shared_memory.sem[dst_sem_id]
    src_sem = shared_memory.sem[src_sem_id] if src_sem_id is not None else None

    clock = None
    if shared_memory.interpret_params.detect_races:
      inc_vector_clock(
          shared_memory.clocks[src_global_core_id], src_global_core_id
      )
      clock = copy_vector_clock(shared_memory.clocks[src_global_core_id])
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

    if shared_memory.interpret_params.dma_execution_mode == 'on_wait':
      shared_memory.dmas_by_sem[dst_sem_id].append(dma)
      if src_sem_id is not None:
        shared_memory.dmas_by_sem[src_sem_id].append(dma)
      return

  assert shared_memory.interpret_params.dma_execution_mode == 'eager'
  execute_dma(dma)

def dma_wait(device_id, local_core_id, sem_id, size):
  device_id = int(device_id)
  local_core_id = int(local_core_id)
  sem_id = int(sem_id)
  size = int(size)
  global_core_id = _get_global_core_id(device_id, local_core_id)

  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    if shared_memory.interpret_params.detect_races:
      inc_vector_clock(shared_memory.clocks[global_core_id], global_core_id)
    sem = shared_memory.sem[sem_id]
  sem.wait(size, global_core_id, is_dma=True)

def semaphore_signal(device_id, local_core_id, sem_id, inc, target_device_id,
                     target_local_core_id):
  device_id = int(device_id)
  local_core_id = int(local_core_id)
  sem_id = int(sem_id)
  inc = int(inc)
  src_global_core_id = _get_global_core_id(device_id, local_core_id)
  if target_device_id is None:
    target_device_id = device_id
  else:
    target_device_id = int(target_device_id)

  if target_local_core_id is None:
    target_local_core_id = 0

  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    clock = None
    if shared_memory.interpret_params.detect_races:
      inc_vector_clock(
          shared_memory.clocks[src_global_core_id], src_global_core_id
      )
      clock = copy_vector_clock(shared_memory.clocks[src_global_core_id])
    sem = shared_memory.sem[sem_id]
  sem.signal(
      inc, _get_global_core_id(target_device_id, target_local_core_id), clock
  )

def semaphore_wait(device_id, local_core_id, sem_id, value):
  device_id = int(device_id)
  local_core_id = int(local_core_id)
  sem_id = int(sem_id)
  value = int(value)
  global_core_id = _get_global_core_id(device_id, local_core_id)

  shared_memory = _get_shared_memory()
  with shared_memory.lock:
    if shared_memory.interpret_params.detect_races:
      inc_vector_clock(shared_memory.clocks[global_core_id], global_core_id)
    sem = shared_memory.sem[sem_id]
  sem.wait(value, global_core_id)

def _compute_transformed_shape_and_dtype(shape, dtype, transforms):
  for transform in transforms:
    if transform is None:
      continue
    shape = transform.transform_shape(shape)
    dtype = transform.transform_dtype(dtype)
  return shape, dtype

def _device_coords_to_logical_id(device_coords, axis_sizes):
  if not isinstance(device_coords, tuple):
    device_coords = (device_coords,)
  assert len(device_coords) == len(axis_sizes)
  sizes = list(axis_sizes.values())
  ret = 0
  for i in range(len(device_coords)):
    ret += device_coords[i] * math.prod(sizes[i+1:])
  return ret

def _device_id_to_logical(device_id, device_id_type, axis_sizes):
  if device_id is None:
    return None
  if device_id_type == primitives.DeviceIdType.MESH:
    return _device_coords_to_logical_id(device_id, axis_sizes)
  elif device_id_type == primitives.DeviceIdType.LOGICAL:
    return device_id
  else:
    raise ValueError(f'Unsupported device ID type: {device_id_type}')

@lu.cache
def _to_jaxpr(flat_fun, in_avals):
  new_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
  new_jaxpr = jax_core.ClosedJaxpr(new_jaxpr, consts)
  return new_jaxpr

def _is_any(memory_space):
  return ((memory_space == mosaic_core.MemorySpace.ANY) or
          (memory_space == pallas_core.MemorySpace.ANY))

def _is_float(dtype):
  return jnp.issubdtype(dtype, jnp.floating)

_SENTINEL = jnp.inf

@dataclasses.dataclass(frozen=True)
class Placeholder:
  """Placeholder for use in `_interpret_jaxpr` below instead of putting a concrete value into `env`."""
  shape: tuple[int, ...]
  dtype: jnp.dtype


def _interpret_jaxpr(
    jaxpr, *args, mesh, local_core_id, compiler_params, interpret_params
):
  env = {}

  def read(var):
    if isinstance(var, jax_core.Literal):
      result = var.val
    else:
      result = env[var]
    if isinstance(result, Placeholder):
      result = jax.lax.full(result.shape, _SENTINEL, result.dtype)
    return result

  def write(var, value):
    if interpret_params.skip_floating_point_ops and _is_float(value.dtype):
      value = Placeholder(value.shape, value.dtype)
    env[var] = value

  jax._src.util.safe_map(write, jaxpr.constvars + jaxpr.invars, args)

  # Get the device ID.
  axis_sizes = jax_core.get_axis_env().axis_sizes
  device_id = _device_coords_to_logical_id(
      tuple(lax.axis_index(s) for s in axis_sizes.keys()),
      axis_sizes)
  # TODO(jburnim): Pass the device ID around, instead of re-fetching/computing
  # it for each sub-jaxpr.

  # TODO(jburnim): Clean up and finish this evaluation loop.  For example:
  #  - Replace the big if-statement with a dictionary of rules.
  #  - Handle other higher-order primitives?
  #  - Megacore.
  _interpret = functools.partial(
      _interpret_jaxpr,
      mesh=mesh,
      local_core_id=local_core_id,
      compiler_params=compiler_params,
      interpret_params=interpret_params,
  )
  for eqn in jaxpr.eqns:
    with source_info_util.user_context(
         eqn.source_info.traceback, name_stack=eqn.source_info.name_stack):
      prim = eqn.primitive
      # We defer reading the values for `eqn.invars` into each of the branches
      # of the if-elif-else statement below. This is because the else branch may
      # not need to do any reads if `interpret_params.skip_floating_point_ops`
      # is True. If this is the case, we want to avoid materializing the read
      # array into the jaxpr when this function is traced.
      deferred_invals = functools.partial(
          jax._src.util.safe_map, read, eqn.invars
      )

      if prim is primitives.load_p:
        (ref, transforms, mask, _) = jax.tree.unflatten(
            eqn.params['args_tree'], deferred_invals())
        if mask is not None:
          raise NotImplementedError('masked load_p')
        out = callback.io_callback(
            functools.partial(get, source_info=eqn.source_info),
            eqn.outvars[0].aval,
            device_id,
            local_core_id,
            TPU_MEMORY_SPACE_IDXS[eqn.invars[0].aval.memory_space],
            ref,
            transforms,
            ordered=True)

      elif prim is primitives.swap_p:
        (ref, transforms, val, mask) = jax.tree.unflatten(
            eqn.params['args_tree'], deferred_invals())
        out = callback.io_callback(
            functools.partial(swap, source_info=eqn.source_info),
            eqn.outvars[0].aval,
            device_id,
            local_core_id,
            TPU_MEMORY_SPACE_IDXS[eqn.invars[0].aval.memory_space],
            ref,
            transforms,
            val,
            mask,
            ordered=True)

      elif prim is mosaic_primitives.delay_p:
        # TODO(jburnim): Implement this properly?
        out = []

      elif prim is mosaic_primitives.prng_seed_p:
        # TODO(jburnim): Implement this properly?
        out = []

      elif prim is mosaic_primitives.prng_random_bits_p:
        # TODO(jburnim): Implement this properly?
        out = jnp.zeros(eqn.params['shape'], jnp.int32)

      elif prim is verification.assume_p:
        out = read(eqn.invars[0])

      elif prim is verification.pretend_p:
        out = []

      elif ((prim is lax.axis_index_p)
            and (mesh is not None) and (eqn.params['axis_name'] in mesh.shape)):
        # For now, there can only be one core.
        # TODO(jburnim): Support two Megacore cores.
        out = jnp.int32(0)

      elif prim is lax.cond_p:
        def _make_branch(jaxpr):
          return lambda *args: _interpret(jaxpr, *args)
        invals = deferred_invals()
        out = lax.switch(
            invals[0],
            [_make_branch(branch_jaxpr.jaxpr)
            for branch_jaxpr in eqn.params['branches']],
            *invals[1:])

      elif prim is lax.scan_p:
        consts, init_carry, xs = split_list(
            deferred_invals(),
            [eqn.params['num_consts'], eqn.params['num_carry']],
        )
        def _scan_body(c, a):
          return split_list(
              _interpret(eqn.params['jaxpr'].jaxpr, *consts, *c, *a),
              [eqn.params['num_carry']])
        carry, out = lax.scan(_scan_body, init_carry, xs=xs,
                              length=eqn.params.get('length', None))
        out = carry + out

      elif prim is lax.while_p:
        cond_consts, body_consts, init_vals = split_list(
            deferred_invals(),
            [eqn.params['cond_nconsts'], eqn.params['body_nconsts']],
        )
        out = lax.while_loop(
            lambda args: _interpret(
                eqn.params['cond_jaxpr'].jaxpr, *cond_consts, *args)[0],
            lambda args: _interpret(
                eqn.params['body_jaxpr'].jaxpr, *body_consts, *args),
            init_vals)

      elif prim is for_loop.for_p:
        raise NotImplementedError('for_p')

      elif prim is pjit.pjit_p:
        def f(*args, jaxpr):
          return _interpret(jaxpr.jaxpr, *jaxpr.consts, *args)
        invals = deferred_invals()
        in_avals = tuple(jax_core.shaped_abstractify(i) for i in invals)
        new_jaxpr = _to_jaxpr(
            lu.wrap_init(functools.partial(f, jaxpr=eqn.params['jaxpr']),
                        debug_info=eqn.params['jaxpr'].jaxpr.debug_info),
            in_avals)
        out = pjit.pjit_p.bind(*invals, **(eqn.params | {'jaxpr': new_jaxpr}))

      elif prim is primitives.run_scoped_p:
        if eqn.params['collective_axes']:
          raise NotImplementedError(
              'run_scoped_p with collective axes is not supported'
          )
        # Allocate a buffer or semaphore for each element of
        # eqn.params['jaxpr'].invars. It is assumed that each core
        # runs the same sequence of `run_scoped`s.
        allocs = []
        for v in eqn.params['jaxpr'].invars:
          if v.aval.memory_space == mosaic_core.MemorySpace.SEMAPHORE:
            allocs.append(callback.io_callback(
                _allocate_semaphores,
                jax.ShapeDtypeStruct(v.aval.shape, jnp.int16),
                device_id,
                local_core_id,
                v.aval.shape,
                ordered=True))
          else:
            allocs.append(callback.io_callback(
                _allocate_buffer,
                jax.ShapeDtypeStruct((), jnp.int16),
                device_id,
                local_core_id,
                TPU_MEMORY_SPACE_IDXS[v.aval.memory_space],
                _uninitialized_value(
                    v.aval.shape, v.aval.dtype, interpret_params),
                ordered=True))

        out = _interpret(eqn.params['jaxpr'], *deferred_invals(), *allocs)

        for a, v in zip(allocs, eqn.params['jaxpr'].invars):
          if v.aval.memory_space == mosaic_core.MemorySpace.SEMAPHORE:
            # TODO(jburnim): De-allocate semaphores.
            # callback.io_callback(
            #     _deallocate_semaphores,
            #     None,
            #     device_id,
            #     a,
            #     ordered=True)
            pass
          else:
            callback.io_callback(
                _deallocate_buffer,
                None,
                device_id,
                local_core_id,
                TPU_MEMORY_SPACE_IDXS[v.aval.memory_space],
                a,
                ordered=True)

      elif prim is state_primitives.get_p:
        invals = deferred_invals()
        out = callback.io_callback(
            functools.partial(get, source_info=eqn.source_info),
            eqn.outvars[0].aval,
            device_id,
            local_core_id,
            TPU_MEMORY_SPACE_IDXS[eqn.invars[0].aval.memory_space],
            invals[0],
            jax.tree.unflatten(eqn.params['tree'], invals[1:]),
            ordered=True)

      elif prim is state_primitives.swap_p:
        invals = deferred_invals()
        out = callback.io_callback(
            functools.partial(swap, source_info=eqn.source_info),
            eqn.outvars[0].aval,
            device_id,
            local_core_id,
            TPU_MEMORY_SPACE_IDXS[eqn.invars[0].aval.memory_space],
            invals[0],
            jax.tree.unflatten(eqn.params['tree'], invals[2:]),
            invals[1],
            None,
            ordered=True)

      elif prim is mosaic_primitives.dma_start_p:
        (
            src,
            src_transforms,
            dst,
            dst_transforms,
            dst_sem,
            dst_sem_transforms,
            src_sem,
            src_sem_transforms,
            target_device_id,
        ) = jax.tree.unflatten(eqn.params['tree'], deferred_invals())
        target_device_id = _device_id_to_logical(
            target_device_id, eqn.params['device_id_type'], axis_sizes)
        (orig_src_ref, _, orig_dst_ref, *_
        ) = jax.tree.unflatten(eqn.params['tree'], eqn.invars)
        callback.io_callback(
            functools.partial(dma_start, source_info=eqn.source_info),
            (),
            device_id,
            local_core_id,
            TPU_MEMORY_SPACE_IDXS[getattr(orig_src_ref.aval, 'memory_space', mosaic_core.MemorySpace.ANY)],
            src, src_transforms,
            TPU_MEMORY_SPACE_IDXS[getattr(orig_dst_ref.aval, 'memory_space', mosaic_core.MemorySpace.ANY)],
            dst, dst_transforms,
            state_discharge.transform_array(dst_sem, dst_sem_transforms),
            state_discharge.transform_array(src_sem, src_sem_transforms),
            target_device_id,
            ordered=True)
        out = []

      elif prim is mosaic_primitives.dma_wait_p:
        (
            src,
            src_transforms,
            dst,
            dst_transforms,
            dst_sem,
            dst_sem_transforms,
            src_sem,
            src_sem_transforms,
            target_device_id,
        ) = jax.tree.unflatten(eqn.params['tree'], deferred_invals())
        read_shape, read_dtype = _compute_transformed_shape_and_dtype(
            eqn.invars[0].aval.shape, eqn.invars[0].aval.dtype, src_transforms)
        callback.io_callback(
            dma_wait,
            (),
            device_id,
            local_core_id,
            state_discharge.transform_array(dst_sem, dst_sem_transforms),
            math.prod(read_shape) * read_dtype.itemsize,
            ordered=True)
        out = []

      elif prim is mosaic_primitives.get_barrier_semaphore_p:
        out = callback.io_callback(
            get_barrier_semaphore,
            jax.ShapeDtypeStruct((), jnp.int16),
            device_id,
            _get_mosaic_params(compiler_params).collective_id,
            ordered=True)

      elif prim is primitives.semaphore_signal_p:
        sem, sem_transforms, inc, target_device_id, core_index = (
            jax.tree.unflatten(eqn.params['args_tree'], deferred_invals()))
        target_device_id = _device_id_to_logical(
            target_device_id, eqn.params['device_id_type'], axis_sizes)
        callback.io_callback(
            semaphore_signal,
            (),
            device_id,
            local_core_id,
            state_discharge.transform_array(sem, sem_transforms),
            inc,
            target_device_id,
            core_index,
            ordered=True)
        out = []

      elif prim is primitives.semaphore_wait_p:
        sem, sem_transforms, value = (
            jax.tree.unflatten(eqn.params['args_tree'], deferred_invals()))
        callback.io_callback(
            semaphore_wait,
            (),
            device_id,
            local_core_id,
            state_discharge.transform_array(sem, sem_transforms),
            value,
            ordered=True)
        out = []

      elif prim is primitives.atomic_rmw_p:
        raise NotImplementedError('atomic_rmw_p')

      elif prim is primitives.atomic_cas_p:
        raise NotImplementedError('atomic_cas_p')

      else:
        if interpret_params.skip_floating_point_ops and all(
            _is_float(ovar.aval.dtype) for ovar in eqn.outvars
        ):
          # Skip `prim.bind` since `prim` only produces floating-point values.
          # It is safe to populate `out` with avals since mapping `write` over
          #  `out` below only relies on the shape and dtype (for writing
          # `Placeholder`s).
          out = [ovar.aval for ovar in eqn.outvars]
          if not prim.multiple_results:
            out = out[0]
        else:
          subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
          out = prim.bind(*subfuns, *deferred_invals(), **bind_params)

      out = out if prim.multiple_results else [out]
      jax._src.util.safe_map(write, eqn.outvars, out)

  return jax._src.util.safe_map(read, jaxpr.outvars)

def _compute_start_indices(
    block_mapping, loop_idx, *args, mesh, compiler_params, interpret_params):
  jaxpr = block_mapping.index_map_jaxpr
  block_indices = _interpret_jaxpr(
      jaxpr.jaxpr,
      *jaxpr.consts,
      *loop_idx,
      *args,
      mesh=mesh,
      local_core_id=0,
      compiler_params=compiler_params,
      interpret_params=interpret_params,
  )
  def _get_start_index(i, b):
    match b:
      case pallas_core.Squeezed():
        return i
      case pallas_core.Element():
        return i
      case pallas_core.Blocked():
        return i * b.block_size
      case _:
        raise ValueError(f"Unsupported block dim type: {type(b)}")
  ret = jnp.array(
      tuple(
          _get_start_index(i, b)
          for i, b in zip(block_indices, block_mapping.block_shape)
      ),
      dtype=jnp.int32,
  )
  return ret

def _get_next_indices(grid, indices):
  next_indices = []
  carry = True
  for dim_size, index in reversed(list(zip(grid, indices))):
    i = jnp.where(carry, index + 1, index)
    carry = dim_size == i
    next_indices.append(jnp.where(carry, 0, i))
  return tuple(reversed(next_indices))


def _get_mosaic_params(compiler_params: dict[str, pallas_core.CompilerParams]) -> tpu_core.CompilerParams:
  try:
    return cast(tpu_core.CompilerParams, compiler_params['mosaic_tpu'])
  except KeyError:
    return tpu_core.CompilerParams()


def _get_parallel_dim_semantics(
    compiler_params: dict[str, Any], num_dimensions_in_grid: int,
) -> tuple[bool, ...]:
  """Returns a tuple indicating which grid dimensions have parallel semantics.

  Args:
    compiler_params: Representation of a `mosaic_core.CompilerParams` object
      as a dictionary.
    num_dimensions_in_grid: The number of dimensions in the grid.

  Returns:
    A tuple of booleans where the entry at index `i` is `True` precisely if the
    `i`-th dimension in the grid has parallel semantics.

  Raises:
    ValueError: If the dimensions with parallel semantics do not form a prefix
      of the grid.
  """
  mosaic_params = _get_mosaic_params(compiler_params)
  if mosaic_params.dimension_semantics is None:
    return (False,) * num_dimensions_in_grid
  result = tuple(ds == 'parallel' for ds in mosaic_params.dimension_semantics)
  for ds0, ds1 in zip(result[:-1], result[1:]):
    if ds1 and not ds0:
      raise ValueError(
          'Dimensions with parallel semantics must form a prefix of the grid.'
      )
  return result


def _get_parallel_subgrid_size(
    parallel_semantics_per_dim: tuple[bool, ...], grid: tuple[int, ...]
) -> int:
  """Returns the size of the subgrid along the parallel dimensions."""
  return functools.reduce(
      lambda x, y: x * y,
      (
          dim_size if parallel_dim else 1
          for dim_size, parallel_dim in zip(grid, parallel_semantics_per_dim)
      ),
      1,
  )

_GridPointCoordinatesPerDim = tuple[Array, ...]

def _get_randomized_grid_coordinates(
    grid: tuple[int, ...],
    compiler_params: dict[str, Any],
    random_seed: int | None,
) -> _GridPointCoordinatesPerDim:
  """Returns a tuple of randomized coordinates for each 'parallel' dimension in `grid`.

  For a dimension with 'parallel' semantics at position `d` in the grid, the
  returned tuple contains a random permutation of the sequence `[0,...,
  grid[d] - 1]` at index `d`. For each dimension with 'arbitrary' semantics,
  the resulting tuple contains an empty array. (Inserting an empty arry for an
  'arbitrary' dimension at position `d` in the grid, instead of the sequence
  `[0,..., grid[d] - 1]`, allows `grid[d]` to be a dynamic value, i.e. a value
  not known at Jax trace time.)

  Args:
    grid: Tuple of sizes of the dimensions in the grid.
    compiler_params: Representation of a `mosaic_core.CompilerParams` object
      as a dictionary.
    parallel_semantics_per_dim: A tuple of booleans indicating whether the
      corresponding dimension in the grid has parallel semantics.
    random_seed: The seed to use for randomizing coordinates in parallel
      dimensions.
  """
  parallel_semantics_per_dim = _get_parallel_dim_semantics(
      compiler_params, len(grid)
  )

  key = jax.random.key(random_seed or 0)
  grid_point_coordinates = []
  for dim_size, parallel_dim in zip(grid, parallel_semantics_per_dim):
    if parallel_dim:
      # The size of a dimension with `parallel` semantics must be known at Jax
      # trace time. This ensures that the arguments to `jnp.arange` and
      # `jax.random.permutation` below are valid.
      dim_size = jax_core.concrete_or_error(None, dim_size)

      coordindates_along_dim = jnp.arange(dim_size, dtype=jnp.int32)
      key, subkey = jax.random.split(key)
      coordindates_along_dim = jax.random.permutation(
          subkey, coordindates_along_dim
      )
      grid_point_coordinates.append(coordindates_along_dim)
    else:
      grid_point_coordinates.append(jnp.array((), dtype=jnp.int32))

  return tuple(grid_point_coordinates)


def _get_grid_point(
    loop_indices: tuple[Array, ...],
    grid_point_coordinates: _GridPointCoordinatesPerDim,
) -> Array:
  """Indexes each entry in `grid_point_coordinates` with the corresponding entry in `loop_indices`.

  If an entry in `grid_point_coordinates` is an empty array, the corresponding
  entry in the returned array is the corresponding entry in `loop_indices`.
  Otherwise, the returned array contains the entry in `grid_point_coordinates`
  indexed with the corresponding entry in `loop_indices`.

  Args:
    loop_indices: A tuple of loop indices.
    grid_point_coordinates: A tuple of coordinate arrays for each dimension in
      the grid. Dimensions with 'arbitrary' semantics are represented by empty
      arrays. Dimensions with 'parallel' semantics are represented by arrays of
      randomized coordinates.

  Returns:
    A 1-dimensional array containing the coordinates for the grid point
    corresponding to the specified `loop_indices`.
  """
  grid_point = []
  for li, coords in zip(loop_indices, grid_point_coordinates):
    grid_point.append(li if jnp.size(coords) == 0 else coords[li])
  return jnp.array(grid_point, dtype=np.int32)

def _uninitialized_value(shape, dtype, interpret_params):
  if interpret_params.uninitialized_memory == 'nan':
    if jnp.issubdtype(dtype, jnp.floating):
      return jnp.full(shape, jnp.nan, dtype)
    elif jnp.issubdtype(dtype, jnp.integer):
      return jnp.full(shape, jnp.iinfo(dtype).max, dtype)
    elif jnp.issubdtype(dtype, jnp.bool):
      return jnp.full(shape, False, dtype)
  if interpret_params.uninitialized_memory == 'zero':
    return jnp.full(shape, 0, dtype)
  raise NotImplementedError(
      interpret_params.uninitialized_memory + ' + ' + str(dtype))

def _pad_to_block_dimension(value, block_shape, interpret_params):
  """Pads values so the shape evenly divides into block dimensions.

  For example, if values has a shape of (33, 2, 5) with a block_shape of
  (32, 2, 4), this function will pad the value of shape to (64, 2, 8).

  Args:
    value: Array to be padded.
    block_shape: Block shapes to use for padding. If None, no padding will
      be performed.

  Returns:
    A padded array.
  """
  padded_shape = tuple(
      ((v - 1) // b + 1) * b for v, b in zip(value.shape, block_shape)
  )
  if padded_shape != value.shape:
    pad_width = tuple((0, a-b) for a, b in zip(padded_shape, value.shape))
    pad_value = _uninitialized_value((), value.dtype, interpret_params)
    value = jnp.pad(value, pad_width, constant_values=pad_value)
  return value

def get_interpret_effects():
  return {callback._OrderedIOEffect}

def interpret_pallas_call(
    *args,
    jaxpr: jax_core.Jaxpr,
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: GridMapping,
    mesh: pallas_core.Mesh | None,
    compiler_params: dict[str, Any],
    cost_estimate: CostEstimate,
    out_avals: tuple[jax_core.AbstractValue, ...],
    interpret_params: TPUInterpretParams,
):
  del debug, cost_estimate, out_avals

  # args contains: *dynamic_grid_sizes, *index, *inputs.  (No consts?)
  dynamic_grid_args, scalars, input_args = split_list(
      args,
      [grid_mapping.num_dynamic_grid_bounds, grid_mapping.num_index_operands],
  )
  dynamic_grid_args_iter = iter(dynamic_grid_args)
  grid = tuple(
      a if a is not pallas_core.dynamic_grid_dim
      else next(dynamic_grid_args_iter)
      for a in grid_mapping.grid
  )
  assert next(dynamic_grid_args_iter, None) is None

  axis_sizes = jax_core.get_axis_env().axis_sizes
  num_devices = functools.reduce(
      jnp.multiply, axis_sizes.values(), jnp.int32(1))
  device_id = _device_coords_to_logical_id(
      tuple(lax.axis_index(s) for s in axis_sizes.keys()),
      axis_sizes)
  callback.io_callback(
      functools.partial(
          _initialize_shared_memory, interpret_params=interpret_params),
      (),
      device_id,
      num_devices,
      interpret_params.num_cores_per_device,
      ordered=True)

  # Pad input arguments.
  is_squeeze_dim = [
      tuple(isinstance(b, pallas_core.Squeezed) for b in bm.block_shape)
      for bm in grid_mapping.block_mappings
  ]
  block_shapes = [
      pallas_core._get_block_shape(bm.block_shape)
      for bm in grid_mapping.block_mappings
  ]
  num_inputs = grid_mapping.num_inputs
  input_args = [
      _pad_to_block_dimension(a, bs, interpret_params)
      for a, bs in zip(input_args, block_shapes[:num_inputs])
  ]

  # Allocate HBM buffers for pallas_call inputs.
  #
  # TODO(jburnim): As an optimization, skip allocating buffers for inputs that
  # are neither aliased nor passed to the kernel in HBM?
  input_buffer_ids = []
  for i, var in enumerate(
      jaxpr.invars[grid_mapping.num_index_operands:][:grid_mapping.num_inputs]):
    assert var.aval.dtype == input_args[i].dtype
    input_buffer_ids.append(callback.io_callback(
        _allocate_buffer,
        jax.ShapeDtypeStruct((), jnp.int16),
        device_id,
        None,  # local_core_id
        TPU_MEMORY_SPACE_IDXS[mosaic_core.MemorySpace.ANY],
        input_args[i],
        ordered=True))

  # Allocate buffers in HBM for pallas_call outputs.
  oi_alias_map = {v: k for k, v in input_output_aliases}
  output_buffer_ids = []
  output_buffer_shapes = []
  output_vals = []
  num_outputs = grid_mapping.num_outputs
  output_block_shapes = block_shapes[num_inputs : num_inputs + num_outputs]
  for i, bm in enumerate(grid_mapping.block_mappings_output):
    if i in oi_alias_map:
      # Re-use the HBM buffer for the aliased pallas_call input.
      output_buffer_ids.append(input_buffer_ids[oi_alias_map[i]])
      output_buffer_shapes.append(input_args[oi_alias_map[i]].shape)
      output_vals.append(input_args[oi_alias_map[i]])
    else:
      out_val = _uninitialized_value(bm.array_shape_dtype.shape,
                                     bm.array_shape_dtype.dtype,
                                     interpret_params)
      padded_val = _pad_to_block_dimension(
          out_val, output_block_shapes[i], interpret_params
      )
      output_buffer_ids.append(
          callback.io_callback(
              _allocate_buffer,
              jax.ShapeDtypeStruct((), jnp.int16),
              device_id,
              None,  # local_core_id
              TPU_MEMORY_SPACE_IDXS[mosaic_core.MemorySpace.ANY],
              padded_val,
              ordered=True,
          )
      )
      output_buffer_shapes.append(padded_val.shape)
      output_vals.append(out_val)

  # Allocate buffers for non-HBM kernel arguments (e.g., scalars, inputs,
  # outputs, scratch).
  scalar_buffer_ids = []
  for var, val in zip(jaxpr.invars[grid_mapping.slice_index_ops], scalars):
    assert var.aval.shape == val.shape
    assert var.aval.dtype == val.dtype
    scalar_buffer_ids.append(
        callback.io_callback(
            _allocate_buffer,
            jax.ShapeDtypeStruct((), jnp.int16),
            device_id,
            None,  # local_core_id,
            TPU_MEMORY_SPACE_IDXS[mosaic_core.MemorySpace.SMEM],
            val,
            ordered=True,
        )
    )

  kernel_buffer_ids = scalar_buffer_ids.copy()
  for i, var in enumerate(jaxpr.invars[grid_mapping.num_index_operands:]):
    output_idx = i - grid_mapping.num_inputs
    is_input = i < grid_mapping.num_inputs
    is_output = (output_idx >= 0) and (output_idx < grid_mapping.num_outputs)
    if var.aval.memory_space == mosaic_core.MemorySpace.SEMAPHORE:
      kernel_buffer_ids.append(
          callback.io_callback(
              _allocate_semaphores,
              jax.ShapeDtypeStruct(var.aval.shape, jnp.int16),
              device_id,
              None,  # local_core_id
              var.aval.shape,
              ordered=True,
          )
      )
    elif _is_any(var.aval.memory_space):
      # Use the already-allocated HBM input or output buffer.
      #
      # TODO(jburnim): For kernel args in HBM, check that block shape eqals the
      # shape of the corresponding pallas_call input, and that the index_map
      # is trivial.
      assert is_input ^ is_output
      if is_input:
        kernel_buffer_ids.append(input_buffer_ids[i])
      if is_output:
        kernel_buffer_ids.append(output_buffer_ids[output_idx])
    else:
      kernel_buffer_ids.append(
          callback.io_callback(
              _allocate_buffer,
              jax.ShapeDtypeStruct((), jnp.int16),
              device_id,
              None,  # local_core_id,
              TPU_MEMORY_SPACE_IDXS[var.aval.memory_space],
              _uninitialized_value(
                  var.aval.shape, var.aval.dtype, interpret_params
              ),
              ordered=True,
          )
      )

  if _get_mosaic_params(compiler_params).collective_id is None:
    # The kernel doesn't specify its own barrier semaphore, so we do a global
    # barrier before running the first iteration of the kernel.
    callback.io_callback(_barrier, (), device_id, ordered=True)

  _, input_ids, kernel_output_ids, _  = split_list(
      kernel_buffer_ids,
      [grid_mapping.num_index_operands, num_inputs, grid_mapping.num_outputs])
  input_vars, output_vars = split_list(
      jaxpr.invars[grid_mapping.slice_block_ops], [num_inputs])

  if grid:
    num_iterations = functools.reduce(jnp.multiply, grid)  # type: ignore[arg-type]
  else:
    # Base case is always one iteration when grid is ()
    num_iterations = 1

  randomized_grid_coordinates = _get_randomized_grid_coordinates(
      grid, compiler_params, interpret_params.random_seed  # type: ignore[arg-type]
  )

  parallel_dim_semantics = _get_parallel_dim_semantics(
      compiler_params, len(grid)
  )
  parallel_subgrid_size = _get_parallel_subgrid_size(
      parallel_dim_semantics, grid  # type: ignore[arg-type]
  )
  num_points_in_parallel_subgrid_per_core = (
      parallel_subgrid_size + interpret_params.num_cores_per_device - 1
  ) // interpret_params.num_cores_per_device  # We round up here.
  num_iterations_per_point_in_parallel_subgrid = (
      # This is evenly divisible.
      num_iterations // parallel_subgrid_size  # type: ignore[operator]
  )
  num_iterations_per_core = (
      num_points_in_parallel_subgrid_per_core
      * num_iterations_per_point_in_parallel_subgrid
  )

  def _get_local_grid_env(loop_idx):
    if grid_mapping.local_grid_env is not None:
      return grid_mapping.local_grid_env(loop_idx, grid)
    else:
      return tuple(
          pallas_core.GridAxis(idx, b)
          for dim, (idx, b) in enumerate(zip(loop_idx, grid))
          if dim not in grid_mapping.vmapped_dims
      )

  def body(
      carry: tuple[
          jnp.int32,
          tuple[jnp.int32, ...],
          jnp.ndarray,
          jnp.int32,
          jnp.int32,
          list[jnp.ndarray],
          list[jnp.ndarray],
      ],
  ) -> tuple[
      jnp.int32,
      tuple[jnp.int32, ...],
      jnp.ndarray,
      jnp.int32,
      jnp.int32,
      list[jnp.ndarray],
      list[jnp.ndarray],
  ]:
    """Performs a single iteration of `jaxpr` in the device grid.

    Execution of `jaxpr` is preceded by reading kernel input buffers and
    followed by writing kernel output buffers.

    Args:
      carry: (iteration_idx, loop_idx, grid_point, prev_local_core_id,
              cur_local_core_id, prev_start_indices, cur_start_indices).
        - iteration_idx is the interation index.
        - loop_idx are the program ids for each grid axis.
        - grid_point is the grid point for the current loop iteration.
        - prev_local_core_id is the (device-local) core id from the previous
          loop iteration.
        - cur_local_core_id is the (device-local) core id for the current loop
          iteration.
        - prev_start_indices is a rank-1 array that contains the start indices
          for the slices of inputs and outputs processed in the previous loop
          iteration.
        - cur_start_indices is a rank-1 array that contains the start indices
          for the slices of inputs and outputs processed in the current loop
          iteration.

        Note that by carrying the previous *and* current start indices between
        loop iterations, it suffices to compute only one list of start indices,
        i.e. `next_start_indices` (see below), per iteration.

    Returns:
      The carry for the next iteration.
    """
    (
        iteration_idx,
        loop_idx,
        grid_point,
        prev_local_core_id,
        cur_local_core_id,
        prev_start_indices,
        cur_start_indices,
    ) = carry
    if interpret_params.grid_point_recorder is not None:
      callback.io_callback(
          interpret_params.grid_point_recorder,
          (),
          grid_point,
          cur_local_core_id,
      )

    next_local_core_id = (iteration_idx + 1) // num_iterations_per_core

    with pallas_core.grid_env(_get_local_grid_env(loop_idx)):
      next_loop_idx = _get_next_indices(grid, loop_idx)
      next_grid_point = _get_grid_point(
          next_loop_idx, randomized_grid_coordinates
      )
      next_start_indices = [
          _compute_start_indices(
              bm,
              next_grid_point,
              *scalar_buffer_ids,
              mesh=mesh,
              compiler_params=compiler_params,
              interpret_params=interpret_params,
          )
          for bm in grid_mapping.block_mappings
      ]

      # Copy slices of the input to the kernel buffers.
      def _store_slice_to_kernel_input(index, input_var):
        # Copy from the HBM buffer for the pallas_call input to the kernel
        # input buffer.
        # TODO(jburnim): Just use input_args[j] when the input is not aliased?
        transform = indexing.NDIndexer(
            indices=tuple(
                indexing.ds(st, sz) if not iid else st
                for st, sz, iid in zip(
                    cur_start_indices[index],
                    block_shapes[index],
                    is_squeeze_dim[index],
                )
            ),
            shape=input_args[index].shape,
            int_indexer_shape=(),
        )
        sliced_val = callback.io_callback(
            # TODO(jburnim): Pass source_info from the pallas_call, in case this
            # read is involved in a data race.
            get,
            jax.ShapeDtypeStruct(input_var.aval.shape, input_var.aval.dtype),
            device_id,
            cur_local_core_id,
            TPU_MEMORY_SPACE_IDXS[mosaic_core.MemorySpace.ANY],
            input_buffer_ids[index],
            (transform,),
            ordered=True,
        )
        callback.io_callback(
            # TODO(jburnim): Pass source_info from the pallas_call, in case this
            # store is involved in a data race.
            store,
            (),
            device_id,
            cur_local_core_id,
            TPU_MEMORY_SPACE_IDXS[input_var.aval.memory_space],
            input_ids[index],
            (),
            sliced_val,
            ordered=True,
        )

      for j, var in enumerate(input_vars):
        if _is_any(var.aval.memory_space):
          continue
        assert len(cur_start_indices[j].shape) == 1
        assert len(prev_start_indices[j].shape) == 1
        jax.lax.cond(
            (iteration_idx == 0)
            | (cur_local_core_id != prev_local_core_id)
            | jax.lax.reduce_or(
                cur_start_indices[j] != prev_start_indices[j], axes=(0,)
            ),
            functools.partial(_store_slice_to_kernel_input, j, var),
            lambda: None,
        )

      # Invoke the kernel.
      _interpret_jaxpr(
          jaxpr,
          *kernel_buffer_ids,
          mesh=mesh,
          local_core_id=cur_local_core_id,
          compiler_params=compiler_params,
          interpret_params=interpret_params,
      )

      # Copy from the kernel buffers to slices of the output in HBM.
      def _store_to_output_buffer(index, output_var):
        kernel_output_val = callback.io_callback(
            # TODO(jburnim): Pass source_info from the pallas_call, in case this
            # get is involved in a data race.
            get,
            output_var.aval,
            device_id,
            cur_local_core_id,
            TPU_MEMORY_SPACE_IDXS[output_var.aval.memory_space],
            kernel_output_ids[j],
            (),
            ordered=True,
        )
        transform = indexing.NDIndexer(
            indices=tuple(
                indexing.ds(st, sz) if not iid else st
                for st, sz, iid in zip(
                    cur_start_indices[num_inputs + index],
                    block_shapes[num_inputs + index],
                    is_squeeze_dim[num_inputs + index],
                )
            ),
            shape=output_vals[index].shape,
            int_indexer_shape=(index),
        )
        callback.io_callback(
            # TODO(jburnim): Pass source_info from the pallas_call, in case this
            # store is involved in a data race.
            store,
            (),
            device_id,
            cur_local_core_id,
            TPU_MEMORY_SPACE_IDXS[mosaic_core.MemorySpace.ANY],
            output_buffer_ids[index],
            (transform,),
            kernel_output_val,
            ordered=True,
        )

      for j, var in enumerate(output_vars):
        if _is_any(var.aval.memory_space):
          continue
        assert len(cur_start_indices[num_inputs + j].shape) == 1
        assert len(next_start_indices[num_inputs + j].shape) == 1
        jax.lax.cond(
            (iteration_idx + 1 == num_iterations)
            | (cur_local_core_id != next_local_core_id)
            | jax.lax.reduce_or(
                cur_start_indices[num_inputs + j]
                != next_start_indices[num_inputs + j],
                axes=(0,),
            ),
            functools.partial(_store_to_output_buffer, j, var),
            lambda: None,
        )

      return (
          iteration_idx + 1,
          next_loop_idx,
          next_grid_point,
          cur_local_core_id,
          next_local_core_id,
          cur_start_indices,
          next_start_indices,
      )

  initial_loop_idx = (jnp.int32(0),) * len(grid)
  initial_grid_point = _get_grid_point(
      initial_loop_idx, randomized_grid_coordinates
  )
  with pallas_core.grid_env(_get_local_grid_env(initial_loop_idx)):
    initial_start_indices = [
        _compute_start_indices(
            bm,
            initial_grid_point,
            *scalar_buffer_ids,
            mesh=mesh,
            compiler_params=compiler_params,
            interpret_params=interpret_params,
        )
        for bm in grid_mapping.block_mappings
    ]
  # TODO(jburnim): Handle parallel grid dimensions + megacore.
  callback.io_callback(
      _update_clocks_for_device_barrier, (), device_id, ordered=True
  )
  _ = lax.while_loop(
      lambda carry: carry[0] < num_iterations,
      body,
      (
          jnp.int32(0),
          initial_loop_idx,
          initial_grid_point,
          jnp.int32(0),  # Previous core id is ignored on the first iteration.
          jnp.int32(0),  # Current core id is set to 0 for the first iteration.
          initial_start_indices,  # Previous start indices are ignored on the first iteration.
          initial_start_indices,
      ),
  )
  callback.io_callback(
      _update_clocks_for_device_barrier, (), device_id, ordered=True
  )

  # Read the output from the allocated output buffers.
  ret = [
      callback.io_callback(
          # TODO(jburnim): Pass source_info from the pallas_call, in case this
          # get is involved in a data race.
          get,
          val,
          device_id,
          0,  # local_core_id
          TPU_MEMORY_SPACE_IDXS[mosaic_core.MemorySpace.ANY],
          output_buffer_id,
          (indexing.NDIndexer.from_indices_shape(
              tuple(indexing.ds(0, s) for s in val.shape),
              output_buffer_shape),),
          ordered=True)
      for val, output_buffer_id, output_buffer_shape in zip(
          output_vals, output_buffer_ids, output_buffer_shapes)
  ]

  callback.io_callback(
      _validate,
      (),
      device_id,
      ordered=True)

  # For now, when we're done with a pallas_call, we delete the shared memory.
  # We use a barrier to ensure that all devices are done running the kernel.
  #
  # TODO(jburnim): Get rid of this barrier.  And figure out how this should
  # work if we want to invoke successive pallas_calls that use the same
  # shared memory.
  callback.io_callback(
      _clean_up_shared_memory,
      (),
      device_id,
      ordered=True)

  return ret
