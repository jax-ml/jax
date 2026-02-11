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

from collections.abc import Callable
import contextlib
import dataclasses
import enum
import functools
import itertools
import math
import threading
from typing import Any, Literal

import jax
from jax import lax
from jax._src import callback
from jax._src import config
from jax._src import core as jax_core
from jax._src import frozen_dict
from jax._src import pjit
from jax._src import source_info_util
from jax._src.interpreters import mlir
from jax._src.tree_util import FlatTree
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives
from jax._src.pallas.mosaic import core as mosaic_core
from jax._src.pallas.mosaic import primitives as mosaic_primitives
from jax._src.pallas.mosaic.interpret import shared_memory as memory
from jax._src.pallas.mosaic.interpret import vector_clock as vc
from jax._src.pallas.mosaic.interpret.race_detection_state import RaceDetectionState
from jax._src.pallas.mosaic.interpret.thread_map import thread_map
import jax._src.pallas.mosaic.interpret.utils as interpret_utils
from jax._src import state
from jax._src.state import discharge as state_discharge
from jax._src.state import indexing
from jax._src.state import primitives as state_primitives
from jax._src.typing import Array
from jax._src.util import (
    safe_map,
    safe_zip,
    split_list
)
from jax._src.interpreters import partial_eval as pe
import jax.numpy as jnp
import numpy as np


map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip


@dataclasses.dataclass(frozen=True, kw_only=True)
class InterpretParams(interpret_utils.InterpretParams):
  """Parameters for TPU interpret mode.

  TPU interpret mode is a way run Pallas TPU kernels on CPU, while simulating
  a TPU's shared memory (HBM, VMEM, etc.), communication (remote and local
  DMAs), and synchronization operations (semaphores, barriers, etc.).  This mode
  is intended for debugging and testing.

  To run a kernel under TPU interpret mode, pass an instance of
  ``InterpretParams`` as an argument for the ``interpret`` parameter of
  :func:`jax.experimental.pallas.pallas_call` or
  :func:`jax.experimental.pallas.core_map`.

  NOTE: If an exception is raised while interpreting a kernel, you must call
  :func:`reset_tpu_interpret_mode_state` before using TPU interpret mode
  again in the same process.

  Attributes:
    dma_execution_mode:  If "eager", DMAs are executed as soon as they are
      issued.  If "on_wait", DMA reads or writes are only executed when a device
      is waiting on a DMA semaphore that will be signaled when the read or write
      is complete.
      Default: "on_wait".
    random_seed: Seed for random number generator used during interpretation.
      Currently random numbers are used to randomize the grid coordinates along
      dimensions with 'parallel' semantics.
      Default: None.
    grid_point_recorder: Callback that is invoked by the interpreter for each
      grid point in the order in which the grid points are traversed. The
      callback is invoked with two arguments: - A tuple of grid coordinates. -
      The local core ID of the core that is processing the grid point. This
      callback is intended for inspecting - the randomization of coordinates
      along grid dimensions with 'parallel' semantics and - the mapping of grid
      points to local (i.e. per-device) cores.
      Default: None.
    allow_hbm_allocation_in_run_scoped: If `True`, allows the allocation of HBM
      buffers (which are then shared across the cores in a device) in
      `run_scoped`. While this behavior can be enabled in the interpreter,
      allocating HBM buffers with `run_scoped` is not supported when executing
      Pallas kernels on a real TPU.
      Default: `False`.
  """

  dma_execution_mode: Literal["eager", "on_wait"] = "on_wait"
  random_seed: int | None = None
  grid_point_recorder: (
      Callable[[tuple[np.int32, ...], np.int32], None] | None
  ) = None
  allow_hbm_allocation_in_run_scoped: bool = False

  @property
  def num_cores_per_device(self) -> int:
    return self.num_cores_or_threads


@contextlib.contextmanager
def force_tpu_interpret_mode(params: InterpretParams = InterpretParams()):
  """Context manager that forces TPU interpret mode under its dynamic context.

  TPU interpret mode is a way run Pallas TPU kernels on CPU, while simulating
  a TPU's shared memory (HBM, VMEM, etc.), communication (remote and local
  DMAs), and synchronization operations (semaphores, barriers, etc.).  This mode
  is intended for debugging and testing.  See :class:`InterpretParams` for
  additional information.

  Args:
    params: an instance of :class:`InterpretParams`.  Any call to
      :func:`jax.experimental.pallas.pallas_call` or
      :func:`jax.experimental.pallas.core_map` that is traced under this context
      manager will be run with ``interpret=params``.  When ``params`` is not
      ``None``, this will cause those calls to run with TPU interpret mode.
  """
  prev = config.pallas_tpu_interpret_mode_context_manager.swap_local(params)
  try:
    yield
  finally:
    config.pallas_tpu_interpret_mode_context_manager.set_local(prev)

def set_tpu_interpret_mode(params: InterpretParams = InterpretParams()):
  config.pallas_tpu_interpret_mode_context_manager.set_global(params)  # type: ignore[arg-type]


# TODO(jburnim): Do we want to support multiple instances of SharedMemory?
# Maybe for running multiple distinct interpreted computations in parallel?
_shared_memory: memory.SharedMemory | None = None
_shared_memory_init_lock = threading.Lock()
races: RaceDetectionState | None = None
dma_id_counter: interpret_utils.Counter | None = None

def reset_tpu_interpret_mode_state():
  """Resets all global, shared state used by TPU interpret mode.

  TPU interpret mode uses global, shared state for simulating memory buffers
  and semaphores, for race detection, etc., when interpreting a kernel.
  Normally, this shared state is cleaned up after a kernel is interpreted.

  But if an exception is thrown while interpreting a kernel, the shared state
  is not cleaned up, allowing the simulated TPU state to be examined for
  debugging purposes.  In this case, the shared state must be reset before
  any further kernels are interpreted.
  """
  global _shared_memory, races, dma_id_counter
  with _shared_memory_init_lock:
    _shared_memory = None
    races = None
    dma_id_counter = None


def _get_shared_memory() -> memory.SharedMemory:
  assert _shared_memory is not None
  return _shared_memory


def _clear_shared_memory():
  global _shared_memory
  with _shared_memory_init_lock:
    _shared_memory = None


def _initialize_shared_memory(
    device_id, num_devices, num_cores_per_device, *, interpret_params
):
  global _shared_memory, races, dma_id_counter
  del device_id

  num_devices = int(num_devices)
  num_cores_per_device = int(num_cores_per_device)
  num_cores = num_devices * num_cores_per_device

  with _shared_memory_init_lock:
    if _shared_memory is None:
      vector_clock_size = interpret_params.get_vector_clock_size(num_devices)
      races = RaceDetectionState(num_cores=num_cores)
      dma_id_counter = interpret_utils.Counter(100)
      _shared_memory = memory.SharedMemory(
          num_devices=num_devices,
          num_cores_per_device=num_cores_per_device,
          out_of_bounds_reads=interpret_params.out_of_bounds_reads,
          dma_execution_mode=interpret_params.dma_execution_mode,
          uninitialized_memory=interpret_params.uninitialized_memory,
          detect_races=interpret_params.detect_races,
          vector_clock_size=vector_clock_size,
          clocks=[
              vc.make_vector_clock(vector_clock_size) for _ in range(num_cores)
          ],
          barrier=threading.Barrier(
              num_devices, action=_update_clocks_for_global_barrier
          ),
          clean_up_barrier=threading.Barrier(
              num_devices, action=_clear_shared_memory
          ),
      )
  assert _shared_memory.num_cores == num_cores


def _update_clocks_for_device_barrier(device_id):
  """Synchronizes the vector clocks for the cores on the given device."""
  shared_memory = _get_shared_memory()
  shared_memory.update_clocks_for_device_barrier(device_id)


def _update_clocks_for_global_barrier():
  """Synchronizes all vector clocks."""
  shared_memory = _get_shared_memory()
  shared_memory.update_clocks(0, shared_memory.num_cores)


def _barrier(device_id):
  del device_id
  shared_memory = _get_shared_memory()
  if shared_memory.num_devices > 1:
    shared_memory.barrier.wait()


def _clean_up_shared_memory(device_id):
  del device_id
  shared_memory = _get_shared_memory()
  shared_memory.clean_up_barrier.wait()


def _check_for_revisiting(device_id, local_core_id, loop_idx, output_blocks):
  device_id = int(device_id)
  local_core_id = int(local_core_id)
  loop_idx = tuple(int(x) for x in loop_idx)
  try:
    output_blocks = jax.tree.map(int, output_blocks)
  except:
    raise ValueError('Advanced indexers are not supported on TPU')
  output_ranges = [
      interpret_utils.to_range(b) if b is not None else None
      for b in output_blocks
  ]

  shared_memory = _get_shared_memory()
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

  shared_memory = _get_shared_memory()
  semaphores = shared_memory.get_sempahores_with_nonzero_count(device_id)
  if semaphores:
    sem, global_core_id = semaphores[0]
    # TODO(jburnim): Make this raise an error, but in a way that doesn't
    # cause other devices to hang later in `_clean_up_shared_memory`.
    print(
        f'Semaphore {sem.id} has non-zero count for {device_id} (global core'
        f' {global_core_id}) at kernel exit:'
        f' {sem.count_by_core[global_core_id]}'
    )


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

  local_core_id_to_buffer_id: dict[int, int] = {}
  for lci in local_core_ids:
    buffer_id = shared_memory.get_next_buffer_id(device_id, lci)
    if memory_space_str in ['any', 'hbm']:
      # If allocating in HBM, only actually allocate a buffer once. The first
      # local core (i.e. thread) that gets here allocates the buffer, but the
      # buffer is still keyed in the shared memory with core ID 0. However,
      # since the buffer is shared across all cores, we initialize the buffer's
      # `ref_count` with the number of cores per device. This ensures that the
      # buffer is not deallocated until all cores have exited the scope of the
      # allocation (e.g. have exited the body of a `run_scoped`).
      key = (memory_space_str, buffer_id, device_id, 0)
      ref_count = shared_memory.num_cores_per_device
    else:
      key = (memory_space_str, buffer_id, device_id, lci)
      ref_count = 1
      if len(local_core_id_to_buffer_id) > 0:
        # If we are allocating more than one buffer, we must make additional
        # copies of `val` so that each buffer is a distinct ndarray.
        val = val.copy()

    shared_memory.allocate_buffer(key, ref_count=ref_count, value=val)
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

  shared_memory = _get_shared_memory()
  key = (memory_space, buffer_id, device_id, local_core_id)
  shared_memory.deallocate_buffer(key)


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
    global_core_ids = shared_memory.get_global_core_ids(device_id)
  else:
    local_core_id_int = int(local_core_id)
    global_core_ids = (
        shared_memory.get_global_core_id(device_id, local_core_id_int),
    )
  del local_core_id

  global_core_id_to_semaphore_id = {}
  for gci in global_core_ids:
    semaphore_id = shared_memory.allocate_semaphores(gci, num_semaphores)
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
TPU_MEMORY_SPACE_NAMES = {
    i: v.value for i, v in enumerate(mosaic_core.MemorySpace)
}

# Inject ANY as the last memory space.
TPU_MEMORY_SPACE_NAMES[len(TPU_MEMORY_SPACE_IDXS)] = (
    pallas_core.MemorySpace.ANY.value
)
TPU_MEMORY_SPACE_IDXS[pallas_core.MemorySpace.ANY] = len(TPU_MEMORY_SPACE_IDXS)

# Default to VMEM when no memory space is specified.
TPU_MEMORY_SPACE_IDXS[None] = TPU_MEMORY_SPACE_IDXS[
    mosaic_core.MemorySpace.VMEM
]


def get_barrier_semaphore(device_id, collective_id):
  del device_id
  collective_id = int(collective_id)
  shared_memory = _get_shared_memory()
  shared_memory.guarantee_semaphore_with_fixed_id(collective_id)
  return np.int16(collective_id)


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

  shared_memory = _get_shared_memory()

  local_core_id_for_buffer = _local_core_id_or_zero_if_hbm(
      local_core_id, memory_space
  )
  global_core_id = shared_memory.get_global_core_id(device_id, local_core_id)

  key = (memory_space, buffer_id, device_id, local_core_id_for_buffer)
  read_range = interpret_utils.to_range(transforms)
  ret, (shape, dtype), clock_ = shared_memory.get_buffer_content(
      key, read_range, global_core_id
  )
  clock = clock if clock is not None else clock_

  # Compute the shape of the read value, assuming the read is fully in-bounds.
  # TODO(jburnim): We already know this shape in the Jaxpr where we insert a
  # callback to `get`.  Should we just pass the shape to `get`?
  # TODO(jburnim): Move to a helper function?
  full_read_shape = []
  assert len(read_range) <= len(shape)
  for dim_size, idx_or_slice in itertools.zip_longest(
      shape, read_range, fillvalue=None
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
              f' reading [{read_range}] but buffer has shape {shape}.'
          )
        else:
          # Different error message when we are reading a block of an input,
          # to copy it to a buffer before invoking the kernel body.
          raise IndexError(
              f'Out-of-bounds block index {block_indices} for'
              f' input "{input_name}" in iteration {grid_loop_idx}'
              f' on device {device_id} (core {local_core_id}):'
              f' reading [{read_range}] but input has shape {shape}.'
          )
    # out_of_bounds_reads == "uninitialized"
    uninit_array = np.full(
        full_read_shape,
        interpret_utils.get_uninitialized_value(
            dtype, shared_memory.uninitialized_memory
        ),
        dtype=dtype,
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
    assert races is not None
    races.check_read(
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

  shared_memory = _get_shared_memory()

  local_core_id_for_buffer = _local_core_id_or_zero_if_hbm(
      local_core_id, memory_space
  )
  global_core_id = shared_memory.get_global_core_id(device_id, local_core_id)

  key = (memory_space, buffer_id, device_id, local_core_id_for_buffer)
  write_range = interpret_utils.to_range(transforms)
  in_bounds, (shape, _), clock_ = shared_memory.store_buffer_content(
      key, write_range, val, global_core_id
  )
  clock = clock if clock is not None else clock_

  if not in_bounds:
    if output_name is None:
      raise ValueError(
          'Out-of-bounds write of'
          f' ({device_id} {local_core_id} {memory_space} {buffer_id}):'
          f' writing [{write_range}] but buffer has shape {shape} .'
      )
    else:
      # Different error message when we are copying a kernel buffer to a
      # block of an output (just after a kernel invocation).
      raise IndexError(
          f'Out-of-bounds block index {block_indices} for'
          f' output "{output_name}" in iteration {grid_loop_idx}'
          f' on device {device_id} (core {local_core_id}):'
          f' reading [{write_range}] but output has shape {shape}.'
      )

  if shared_memory.detect_races:
    if src_device_id is None:
      src_device_id = device_id
    if src_local_core_id is None:
      src_local_core_id = local_core_id
    assert races is not None
    races.check_write(
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

  shared_memory = _get_shared_memory()

  local_core_id_for_buffer = _local_core_id_or_zero_if_hbm(
      local_core_id, memory_space
  )
  global_core_id = shared_memory.get_global_core_id(device_id, local_core_id)

  key = (memory_space, buffer_id, device_id, local_core_id_for_buffer)
  read_write_range = interpret_utils.to_range(transforms)
  ret, (shape, _), clock = shared_memory.swap_buffer_content(
      key, read_write_range, val, mask, global_core_id
  )

  if ret is None:
    if mask is None:
      raise ValueError(
          'Out-of-bounds swap of'
          f' ({device_id} {local_core_id} {memory_space} {buffer_id}):'
          f' swapping [{read_write_range}] but buffer has shape'
          f' {shape} .'
      )
    else:
      # TODO(jburnim): Include indices of out-of-bounds locations where mask
      # is True.
      raise ValueError(
          'Out-of-bounds masked swap of'
          f' ({device_id} {local_core_id} {memory_space} {buffer_id}): swapping'
          f' [{read_write_range}] but buffer has shape {shape} . '
      )

  if shared_memory.detect_races:
    assert races is not None
    races.check_write(
        device_id,
        local_core_id,
        clock,
        (memory_space, buffer_id, device_id, local_core_id_for_buffer),
        read_write_range,
        source_info=source_info,
    )
  return ret


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
  src_sem: memory.Semaphore | None
  dst_sem: memory.Semaphore
  virtual_device_id: int
  clock: vc.VectorClock

  source_info: source_info_util.SourceInfo | None = None

  state: DmaState = DmaState.STARTED
  data: np.ndarray | None = None
  lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

  @property
  def data_size(self) -> int:
    assert self.data is not None
    return self.data.itemsize * self.data.size

  @property
  def detect_races(self) -> bool:
    return self.dst_sem.detect_races

  @property
  def src_global_core_id(self) -> int:
    return self.dst_sem.get_global_core_id(
        self.src_device_id, self.src_local_core_id
    )

  @property
  def dst_global_core_id(self) -> int:
    return self.dst_sem.get_global_core_id(
        self.dst_device_id, self.dst_local_core_id
    )

  def execute_read(self):
    """Executes the reading part of this DMA.

    Note that the caller must not hold the lock on the shared memory (because
    `get` is called in this method).
    """
    # Must acquire the lock on `self` because:
    #   - `self.state` is inspected and modified in this method.
    #   - `self.data` is assigned in this method.
    with self.lock:
      if self.state != DmaState.STARTED:
        return

      if self.detect_races:
        vc.inc_vector_clock(self.clock, self.virtual_device_id)

      self.data = get(
          self.src_device_id,
          self.src_local_core_id,
          self.src_memory_space,
          self.src_buffer_id,
          self.src_transforms,
          clock=vc.copy_vector_clock(self.clock),
          src_device_id=self.id,
          src_local_core_id=0,
          source_info=self.source_info,
      )

      if self.detect_races:
        vc.inc_vector_clock(self.clock, self.virtual_device_id)

      # Signal the send semaphore.
      if self.src_sem is not None:
        self.src_sem.signal(
            self.data_size, self.src_global_core_id, clock=self.clock
        )

      self.state = DmaState.READ

  def execute_write(self):
    """Executes the writing part of this DMA.

    Note that the caller must not hold the lock on the shared memory (because
    `store` is called in this method).
    """
    # Must acquire the lock on `self` because:
    #   - `self.state` is inspected and modified in this method.
    #   - `self.data` is assigned in this method.
    with self.lock:
      assert self.state in (DmaState.READ, DmaState.COMPLETED)
      if self.state == DmaState.COMPLETED:
        return
      assert self.data is not None

      if self.detect_races:
        vc.inc_vector_clock(self.clock, self.virtual_device_id)

      store(
          self.dst_device_id,
          self.dst_local_core_id,
          self.dst_memory_space,
          self.dst_buffer_id,
          self.dst_transforms,
          self.data,
          clock=vc.copy_vector_clock(self.clock),
          src_device_id=self.id,
          src_local_core_id=0,
          source_info=self.source_info,
      )

      if self.detect_races:
        vc.inc_vector_clock(self.clock, self.virtual_device_id)

      self.dst_sem.signal(
          self.data_size, self.dst_global_core_id, clock=self.clock
      )

      self.data = None
      self.state = DmaState.COMPLETED

  def execute_read_and_write(self):
    """Executes this DMA, bot the reading and writing parts.

    Note that the caller must not hold the lock on the shared memory.
    """
    self.execute_read()
    self.execute_write()


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
  shared_memory = _get_shared_memory()
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
  dst_global_core_id = shared_memory.get_global_core_id(
      dst_device_id, src_local_core_id  # Same core on destination device as on source.
  )

  (src_sem, dst_sem), clock = shared_memory.get_semaphores_and_increment_clock(
      (src_sem_id, dst_sem_id), src_global_core_id
  )

  assert dma_id_counter is not None
  id = dma_id_counter.get_next()

  dma = DMA(
      id,
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
      virtual_device_id = shared_memory.get_random_virtual_device_id(),
      clock=clock,
      source_info=source_info,
  )

  if shared_memory.dma_execution_mode == 'on_wait':
    if src_sem_id is None:
      shared_memory.append_semaphore_task(
          dst_sem_id, dst_global_core_id, dma.execute_read_and_write
      )
    else:
      shared_memory.append_semaphore_task(
          src_sem_id, src_global_core_id, dma.execute_read
      )
      shared_memory.append_semaphore_task(
          dst_sem_id,
          dst_global_core_id,
          # This task for the waiting semaphore with ID `dst_sem_id` may be
          # executed before the corresponding DMA task for the sending semaphore
          # that does the DMA read. We therefore have to append a read-and-write
          # task here, instead of just a write task. If the reading for the DMA
          # has already been executed, the DMA's state will indicate this and
          # the read-write-task appended here will do the write only.
          # (Alternatively, we could have the DMA write task wait on the
          # `send_semphore`. This issue with this approach is that we do not
          # know the number of bytes transferred that `send_semaphore` should be
          # waiting for until after the reader task is done.)
          dma.execute_read_and_write,
      )
    return

  assert shared_memory.dma_execution_mode == 'eager'
  dma.execute_read_and_write()


def dma_wait(device_id, local_core_id, sem_id, size):
  shared_memory = _get_shared_memory()

  device_id = int(device_id)
  local_core_id = int(local_core_id)
  sem_id = int(sem_id)
  size = int(size)

  global_core_id = shared_memory.get_global_core_id(device_id, local_core_id)

  (sem,), _ = shared_memory.get_semaphores_and_increment_clock(
      {sem_id}, global_core_id
  )
  assert sem is not None
  sem.wait(size, global_core_id, has_tasks=True)


def semaphore_signal(
    device_id,
    local_core_id,
    sem_id,
    inc,
    target_device_id,
    target_local_core_id,
):
  shared_memory = _get_shared_memory()

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

  (sem,), clock = shared_memory.get_semaphores_and_increment_clock(
      {sem_id}, src_global_core_id
  )
  assert sem is not None
  sem.signal(
      inc,
      shared_memory.get_global_core_id(target_device_id, target_local_core_id),
      clock,
  )


def semaphore_wait(device_id, local_core_id, sem_id, value):
  shared_memory = _get_shared_memory()

  device_id = int(device_id)
  local_core_id = int(local_core_id)
  sem_id = int(sem_id)
  value = int(value)
  global_core_id = shared_memory.get_global_core_id(device_id, local_core_id)

  (sem,), _ = shared_memory.get_semaphores_and_increment_clock(
      {sem_id}, global_core_id
  )
  assert sem is not None
  sem.wait(value, global_core_id)


def _is_any(memory_space):
  return memory_space is pallas_core.MemorySpace.ANY


_SENTINEL = jnp.inf


def _get_memory_space_and_raise_if_hbm(aval, primitive_name, message=None):
  memory_space = aval.memory_space
  if memory_space in [mosaic_core.MemorySpace.HBM, pallas_core.MemorySpace.ANY]:
    if message is None:
      message = (
          f'{primitive_name}: Buffers with a memory space of HBM or ANY cannot'
          ' be referenced directly. Instead, use `pltpu.sync_copy` or'
          ' `pltpu.async_copy`.'
      )
    raise ValueError(message)
  return memory_space


def _interpret_jaxpr(
    jaxpr,
    *args,
    axis_sizes,
    mesh,
    axis_indices,
    device_id,
    local_core_id,
    mosaic_params,
    interpret_params
):
  sentinel_for_floating_point_values = (
      _SENTINEL if interpret_params.skip_floating_point_ops else None
  )
  env = interpret_utils.JaxprEnv(
      vars=jaxpr.constvars + jaxpr.invars,
      values=args,
      sentinel_for_floating_point_values=sentinel_for_floating_point_values,
  )

  # TODO(jburnim): Clean up and finish this evaluation loop.  For example:
  #  - Replace the big if-statement with a dictionary of rules.
  #  - Handle other higher-order primitives?
  _interpret = functools.partial(
      _interpret_jaxpr,
      axis_sizes=axis_sizes,
      mesh=mesh,
      axis_indices=axis_indices,
      device_id=device_id,
      local_core_id=local_core_id,
      mosaic_params=mosaic_params,
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
      deferred_invals = functools.partial(env.read_many, eqn.invars)

      if prim is primitives.load_p:
        (ref, transforms, mask, _) = jax.tree.unflatten(
            eqn.params['args_tree'], deferred_invals())
        if mask is not None:
          raise NotImplementedError('masked load_p')
        memory_space = _get_memory_space_and_raise_if_hbm(
            eqn.invars[0].aval, 'load_p'
        )
        out = callback.io_callback(
            functools.partial(get, source_info=eqn.source_info),
            eqn.outvars[0].aval,
            device_id,
            local_core_id,
            TPU_MEMORY_SPACE_IDXS[memory_space],
            ref,
            transforms,
            ordered=True,
        )

      elif prim is primitives.swap_p:
        (ref, transforms, val, mask) = jax.tree.unflatten(
            eqn.params['args_tree'], deferred_invals())
        memory_space = _get_memory_space_and_raise_if_hbm(
            eqn.invars[0].aval, 'swap_p'
        )
        out = callback.io_callback(
            functools.partial(swap, source_info=eqn.source_info),
            eqn.outvars[0].aval,
            device_id,
            local_core_id,
            TPU_MEMORY_SPACE_IDXS[memory_space],
            ref,
            transforms,
            val,
            mask,
            ordered=True,
        )

      elif prim is primitives.delay_p:
        # TODO(jburnim): Implement this properly?
        out = []

      elif prim is mosaic_primitives.prng_seed_p:
        # TODO(jburnim): Implement this properly?
        out = []

      elif prim is mosaic_primitives.prng_random_bits_p:
        # TODO(jburnim): Implement this properly?
        out = jnp.zeros(eqn.params['shape'], jnp.int32)

      elif ((prim is lax.axis_index_p)
            and (mesh is not None) and (eqn.params['axis_name'] in mesh.shape)):
        # We are interpreting a core_map, and this lax.axis_index call is
        # querying our index along the core axis, so return our core ID.
        out = local_core_id

      elif ((prim is lax.axis_index_p)
            and (eqn.params['axis_name'] in axis_indices)):
        # We replace lax.axis_index calls in the kernel body, so that the
        # kernel body jaxpr can be run on other threads (via an io_callback)
        # without having to recreate the axis environment in those threads.
        out = axis_indices[eqn.params['axis_name']]

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

      elif prim is pjit.jit_p:
        def f(*args, jaxpr):
          return _interpret(jaxpr.jaxpr, *jaxpr.consts, *args)
        invals = deferred_invals()
        args_ft = FlatTree.flatten((invals, {}))
        avals_ft = args_ft.map(jax_core.shaped_abstractify)
        new_jaxpr, _ = pe.trace_to_jaxpr(
            functools.partial(f, jaxpr=eqn.params['jaxpr']), avals_ft,
            eqn.params['jaxpr'].jaxpr.debug_info)
        out = pjit.jit_p.bind(*invals, **(eqn.params | {'jaxpr': new_jaxpr}))

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
            allocs.append(
                callback.io_callback(
                    _allocate_semaphores,
                    jax.ShapeDtypeStruct(v.aval.shape, jnp.int16),
                    device_id,
                    local_core_id,
                    v.aval.shape,
                    ordered=True,
                )
            )
          else:
            if not interpret_params.allow_hbm_allocation_in_run_scoped:
              memory_space = _get_memory_space_and_raise_if_hbm(
                v.aval, 'run_scoped_p', "Cannot allocate HBM in `run_scoped`."
              )
            else:
              memory_space = v.aval.memory_space
            allocs.append(
                callback.io_callback(
                    _allocate_buffer,
                    jax.ShapeDtypeStruct((), jnp.int16),
                    device_id,
                    local_core_id,
                    TPU_MEMORY_SPACE_IDXS[memory_space],
                    interpret_params.get_uninitialized_array(
                        v.aval.shape, v.aval.dtype
                    ),
                    ordered=True,
                )
            )

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
                ordered=True,
            )

      elif prim is state_primitives.get_p:
        memory_space = _get_memory_space_and_raise_if_hbm(
            eqn.invars[0].aval, 'get_p'
        )
        invals = deferred_invals()
        out = callback.io_callback(
            functools.partial(get, source_info=eqn.source_info),
            eqn.outvars[0].aval,
            device_id,
            local_core_id,
            TPU_MEMORY_SPACE_IDXS[memory_space],
            invals[0],
            jax.tree.unflatten(eqn.params['tree'], invals[1:]),
            ordered=True,
        )

      elif prim is state_primitives.swap_p:
        memory_space = _get_memory_space_and_raise_if_hbm(
            eqn.invars[0].aval, 'swap_p'
        )
        invals = deferred_invals()
        out = callback.io_callback(
            functools.partial(swap, source_info=eqn.source_info),
            eqn.outvars[0].aval,
            device_id,
            local_core_id,
            TPU_MEMORY_SPACE_IDXS[memory_space],
            invals[0],
            jax.tree.unflatten(eqn.params['tree'], invals[2:]),
            invals[1],
            None,
            ordered=True,
        )

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
        target_device_id = interpret_utils._device_id_to_logical(
            target_device_id, eqn.params['device_id_type'], axis_sizes,
            axis_indices)
        (orig_src_ref, _, orig_dst_ref, *_
        ) = jax.tree.unflatten(eqn.params['tree'], eqn.invars)
        src_memory_space = getattr(orig_src_ref.aval, 'memory_space', None)
        if src_memory_space is None:
          src_memory_space = pallas_core.MemorySpace.ANY
        dst_memory_space = getattr(orig_dst_ref.aval, 'memory_space', None)
        if dst_memory_space is None:
          dst_memory_space = pallas_core.MemorySpace.ANY
        callback.io_callback(
            functools.partial(dma_start, source_info=eqn.source_info),
            (),
            device_id,
            local_core_id,
            TPU_MEMORY_SPACE_IDXS[src_memory_space],
            src,
            src_transforms,
            TPU_MEMORY_SPACE_IDXS[dst_memory_space],
            dst,
            dst_transforms,
            state_discharge.transform_array(dst_sem, dst_sem_transforms),
            state_discharge.transform_array(src_sem, src_sem_transforms),
            target_device_id,
            ordered=True,
        )
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
        src_ref_aval = state.transform_type(src_transforms, eqn.invars[0].aval)
        assert isinstance(src_ref_aval, state.AbstractRef)
        read_shape = src_ref_aval.shape
        read_dtype = src_ref_aval.dtype
        callback.io_callback(
            dma_wait,
            (),
            device_id,
            local_core_id,
            state_discharge.transform_array(dst_sem, dst_sem_transforms),
            math.prod(read_shape) * read_dtype.itemsize,
            ordered=True,
        )
        out = []

      elif prim is mosaic_primitives.get_barrier_semaphore_p:
        out = callback.io_callback(
            get_barrier_semaphore,
            jax.ShapeDtypeStruct((), jnp.int16),
            device_id,
            mosaic_params.collective_id,
            ordered=True,
        )

      elif prim is primitives.semaphore_signal_p:
        sem, sem_transforms, inc, target_device_id, core_index = (
            jax.tree.unflatten(eqn.params['args_tree'], deferred_invals()))
        target_device_id = interpret_utils._device_id_to_logical(
            target_device_id, eqn.params['device_id_type'], axis_sizes,
            axis_indices)
        callback.io_callback(
            semaphore_signal,
            (),
            device_id,
            local_core_id,
            state_discharge.transform_array(sem, sem_transforms),
            inc,
            target_device_id,
            core_index,
            ordered=True,
        )
        out = []

      elif prim is primitives.semaphore_wait_p:
        sem, sem_transforms, value, decrement = (
            jax.tree.unflatten(eqn.params['args_tree'], deferred_invals()))
        if not decrement:
          raise NotImplementedError('Non-decrementing wait is not supported.')
        callback.io_callback(
            semaphore_wait,
            (),
            device_id,
            local_core_id,
            state_discharge.transform_array(sem, sem_transforms),
            value,
            ordered=True,
        )
        out = []

      elif prim is primitives.atomic_rmw_p:
        raise NotImplementedError('atomic_rmw_p')

      elif prim is primitives.atomic_cas_p:
        raise NotImplementedError('atomic_cas_p')

      else:
        if interpret_params.skip_floating_point_ops and all(
            interpret_utils.is_float(ovar.aval.dtype) for ovar in eqn.outvars
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
      env.write_many(eqn.outvars, out)

  return env.read_many(jaxpr.outvars)

def _compute_start_indices(
    block_mapping, loop_idx, *args,
    axis_sizes, mesh, axis_indices, device_id, local_core_id,
    mosaic_params, interpret_params):
  jaxpr = block_mapping.index_map_jaxpr
  block_indices = _interpret_jaxpr(
      jaxpr.jaxpr,
      *jaxpr.consts,
      *loop_idx,
      *args,
      axis_sizes=axis_sizes,
      mesh=mesh,
      axis_indices=axis_indices,
      device_id=device_id,
      local_core_id=local_core_id,
      mosaic_params=mosaic_params,
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
  return block_indices, ret


def _get_parallel_dim_semantics(
    mosaic_params: mosaic_core.CompilerParams, num_dimensions_in_grid: int,
) -> tuple[bool, ...]:
  """Returns a tuple indicating which grid dimensions have parallel semantics.

  Args:
    mosaic_params: The compiler params for the Mosaic TPU backend.
    num_dimensions_in_grid: The number of dimensions in the grid.

  Returns:
    A tuple of booleans where the entry at index `i` is `True` precisely if the
    `i`-th dimension in the grid has parallel semantics.

  Raises:
    ValueError: If the dimensions with parallel semantics do not form a prefix
      of the grid.
  """
  if mosaic_params.dimension_semantics is None:
    return (False,) * num_dimensions_in_grid
  result = tuple(ds in ('parallel', mosaic_core.PARALLEL)
                 for ds in mosaic_params.dimension_semantics)
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
  return math.prod(
      dim_size if parallel_dim else 1
      for dim_size, parallel_dim in zip(grid, parallel_semantics_per_dim)
  )

_GridPointCoordinatesPerDim = tuple[Array, ...]

def _get_randomized_grid_coordinates(
    grid: tuple[int, ...],
    mosaic_params: mosaic_core.CompilerParams,
    random_seed: int | None,
) -> _GridPointCoordinatesPerDim:
  """Returns a tuple of randomized coordinates for each 'parallel' dimension in `grid`.

  For a dimension with 'parallel' semantics at position `d` in the grid, the
  returned tuple contains a random permutation of the sequence `[0,...,
  grid[d] - 1]` at index `d`. For each dimension with 'arbitrary' semantics,
  the resulting tuple contains an empty array. (Inserting an empty array for an
  'arbitrary' dimension at position `d` in the grid, instead of the sequence
  `[0,..., grid[d] - 1]`, allows `grid[d]` to be a dynamic value, i.e. a value
  not known at Jax trace time.)

  Args:
    grid: Tuple of sizes of the dimensions in the grid.
    mosaic_params: The compiler params for the Mosaic TPU backend.
    parallel_semantics_per_dim: A tuple of booleans indicating whether the
      corresponding dimension in the grid has parallel semantics.
    random_seed: The seed to use for randomizing coordinates in parallel
      dimensions.
  """
  parallel_semantics_per_dim = _get_parallel_dim_semantics(
      mosaic_params, len(grid)
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

# TODO(sharadmv, jburnim): add support for memory space constraints
remove_memory_space_p = jax_core.Primitive('remove_memory_space')

@remove_memory_space_p.def_abstract_eval
def _remove_memory_space_abstract_eval(x):
  if isinstance(x, pallas_core.ShapedArrayWithMemorySpace):
    if (
        x.memory_space is None
        or x.memory_space is pallas_core.MemorySpace.ANY
        or x.memory_space is mosaic_core.MemorySpace.HBM
    ):
      return jax_core.ShapedArray(x.shape, x.dtype)
    raise NotImplementedError(f'Unsupported memory space: {x.memory_space}')
  return x

@remove_memory_space_p.def_impl
def _remove_memory_space_impl(x):
  return x

def _remove_memory_space_lowering(_, x):
  return [x]
mlir.register_lowering(remove_memory_space_p, _remove_memory_space_lowering)


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


def get_interpret_effects():
  return {callback._OrderedIOEffect}


def interpret_pallas_call(
    *args,
    jaxpr: jax_core.Jaxpr,
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: pallas_core.GridMapping,
    mesh: pallas_core.Mesh | None,
    compiler_params: pallas_core.CompilerParams | None,
    cost_estimate: pallas_core.CostEstimate,
    out_avals: tuple[jax_core.AbstractValue, ...],
    interpret_params: InterpretParams,
    metadata: frozen_dict.FrozenDict[str, str] | None,
    name: str | None,
):
  del debug, cost_estimate, out_avals, name
  del metadata  # TODO(sharadmv): Add metadata to HLO.

  if isinstance(mesh, mosaic_core.TensorCoreMesh):
    # As a convenience for users, if we are interpreting a pl.core_map over a
    # TensorCoreMesh, we automatically set the number of cores per device so
    # that users don't have to specify it in the InterpretParams.
    assert len(mesh.shape) == 1
    interpret_params = dataclasses.replace(
        interpret_params, num_cores_or_threads=mesh.devices.shape[0]
    )

  if compiler_params is None:
    mosaic_params = mosaic_core.CompilerParams()
  else:
    assert isinstance(compiler_params, mosaic_core.CompilerParams)
    mosaic_params = compiler_params  # type: ignore[assignment]
  del compiler_params

  args = [remove_memory_space_p.bind(a) for a in args]
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
  axis_indices = {k: lax.axis_index(k) for k in axis_sizes.keys()}
  device_id = interpret_utils.device_coords_to_logical_id(
      tuple(axis_indices.values()), axis_sizes, axis_indices
  )
  callback.io_callback(
      functools.partial(
          _initialize_shared_memory, interpret_params=interpret_params
      ),
      (),
      device_id,
      num_devices,
      interpret_params.num_cores_per_device,
      ordered=True,
  )

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
      interpret_params.pad_to_block_dimension(a, bs)
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
    input_buffer_ids.append(
        callback.io_callback(
            _allocate_buffer,
            jax.ShapeDtypeStruct((), jnp.int16),
            device_id,
            None,  # local_core_id
            TPU_MEMORY_SPACE_IDXS[pallas_core.MemorySpace.ANY],
            input_args[i],
            ordered=True,
        )
    )

  # Allocate buffers in HBM for pallas_call outputs.
  oi_alias_map = {v: k - len(scalars) for k, v in input_output_aliases}
  if any(i < 0 for i in oi_alias_map.keys()):
    raise ValueError('Aliasing of scalar prefetch arguments is not currently '
                     'supported in TPU interpret mode.')
  output_buffer_ids = []
  output_buffer_shapes = []
  output_vals = []
  num_outputs = grid_mapping.num_outputs
  output_block_shapes = block_shapes[num_inputs : num_inputs + num_outputs]
  for i, bm in enumerate(grid_mapping.block_mappings_output):
    if i in oi_alias_map:
      # Reuse the HBM buffer for the aliased pallas_call input.
      output_buffer_ids.append(input_buffer_ids[oi_alias_map[i]])
      output_buffer_shapes.append(input_args[oi_alias_map[i]].shape)
      output_vals.append(input_args[oi_alias_map[i]])
    else:
      out_val = interpret_params.get_uninitialized_array(
          bm.array_aval.shape, bm.array_aval.dtype
      )
      padded_val = interpret_params.pad_to_block_dimension(
          out_val, output_block_shapes[i]
      )
      output_buffer_ids.append(
          callback.io_callback(
              _allocate_buffer,
              jax.ShapeDtypeStruct((), jnp.int16),
              device_id,
              None,  # local_core_id
              TPU_MEMORY_SPACE_IDXS[pallas_core.MemorySpace.ANY],
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
      # TODO(jburnim): For kernel args in HBM, check that block shape equals the
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
              interpret_params.get_uninitialized_array(
                  var.aval.shape, var.aval.dtype
              ),
              ordered=True,
          )
      )

  if mosaic_params.collective_id is None:
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

  if isinstance(mesh, mosaic_core.TensorCoreMesh):
    # We are interpreting a pl.core_map over a TensorCoreMesh, so we use a
    # fixed division of the grid between cores, instead of a random division.
    randomized_grid_coordinates = (jnp.array((), dtype=jnp.int32),) * len(grid)
  else:
    randomized_grid_coordinates = _get_randomized_grid_coordinates(
        grid, mosaic_params, interpret_params.random_seed  # type: ignore[arg-type]
    )

  parallel_dim_semantics = _get_parallel_dim_semantics(
      mosaic_params, len(grid)
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
  def _get_local_grid_env(grid_point):
    if grid_mapping.local_grid_env is not None:
      return grid_mapping.local_grid_env(grid_point, grid)
    else:
      return tuple(
          pallas_core.GridAxis(idx, b)
          for dim, (idx, b) in enumerate(zip(grid_point, grid))
          if dim not in grid_mapping.vmapped_dims
      )

  def _execute_grid_for_core(core_index):
    # NOTE: We assume here that all parallel dimensions appear before all
    # arbitrary dimensions in the grid.  (We will have raised an error earlier
    # if this is not the case.)
    #
    # TODO(jburnim): Are we overusing nested local functions here?
    initial_iteration_idx = core_index * num_iterations_per_core
    loop_bound = jnp.minimum(
        (core_index + 1) * num_iterations_per_core, num_iterations)

    def _body(
        carry: tuple[
            jnp.int32,
            tuple[jnp.int32, ...],
            jnp.ndarray,
            list[jnp.ndarray],
            list[jnp.ndarray],
            list[jnp.ndarray],
        ],
    ) -> tuple[
        jnp.int32,
        tuple[jnp.int32, ...],
        jnp.ndarray,
        list[jnp.ndarray],
        list[jnp.ndarray],
        list[jnp.ndarray],
    ]:
      """Performs one execution of the kernel body.

      Execution of `jaxpr` is preceded by reading kernel input buffers and
      followed by writing kernel output buffers.

      Args:
        carry: (iteration_idx, loop_idx, grid_point, prev_start_indices,
                cur_start_indices).
          - iteration_idx: the iteration index.
          - loop_idx: internal indices for looping over the grid.
          - grid_point: the current positions along all axes of the grid.
          - prev_start_indices: a rank-1 array that contains the start indices
            for the slices of inputs and outputs processed in the previous loop
            iteration.
          - cur_start_indices: a rank-1 array that contains the start indices
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
          prev_start_indices,
          cur_block_indices,
          cur_start_indices,
      ) = carry
      if interpret_params.grid_point_recorder is not None:
        callback.io_callback(
            interpret_params.grid_point_recorder,
            (),
            grid_point,
            core_index,
        )

      with pallas_core.grid_env(_get_local_grid_env(grid_point)):
        next_loop_idx = interpret_utils.get_next_indices(grid, loop_idx)
        next_grid_point = _get_grid_point(
            next_loop_idx, randomized_grid_coordinates
        )
        next_block_indices, next_start_indices = zip(*[
            _compute_start_indices(
                bm,
                next_grid_point,
                *scalar_buffer_ids,
                axis_sizes=axis_sizes,
                mesh=mesh,
                axis_indices=axis_indices,
                device_id=device_id,
                local_core_id=core_index,
                mosaic_params=mosaic_params,
                interpret_params=interpret_params,
            )
            for bm in grid_mapping.block_mappings
        ])
        if jaxpr.debug_info.arg_names is not None:
          input_names, output_names = split_list(
            jaxpr.debug_info.arg_names[grid_mapping.slice_block_ops], [num_inputs])
        else:
          input_names = ["unknown",] * grid_mapping.num_inputs
          output_names = ["unknown",] * grid_mapping.num_outputs

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
              functools.partial(get, input_name=input_names[index]),
              jax.ShapeDtypeStruct(input_var.aval.shape, input_var.aval.dtype),
              device_id,
              core_index,
              TPU_MEMORY_SPACE_IDXS[pallas_core.MemorySpace.ANY],
              input_buffer_ids[index],
              (transform,),
              cur_block_indices[index],
              grid_point,
              ordered=True,
          )
          callback.io_callback(
              # TODO(jburnim): Pass source_info from the pallas_call, in case this
              # store is involved in a data race.
              store,
              (),
              device_id,
              core_index,
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
              (iteration_idx == initial_iteration_idx)
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
            axis_sizes=axis_sizes,
            mesh=mesh,
            axis_indices=axis_indices,
            device_id=device_id,
            local_core_id=core_index,
            mosaic_params=mosaic_params,
            interpret_params=interpret_params,
        )

        # Copy from the kernel buffers to slices of the output in HBM.
        def _store_to_output_buffer(index, output_var, transform):
          kernel_output_val = callback.io_callback(
              # TODO(jburnim): Pass source_info from the pallas_call, in case this
              # get is involved in a data race.
              get,
              output_var.aval,
              device_id,
              core_index,
              TPU_MEMORY_SPACE_IDXS[output_var.aval.memory_space],
              kernel_output_ids[index],
              (),
              ordered=True,
          )
          callback.io_callback(
              # TODO(jburnim): Pass source_info from the pallas_call, in case this
              # store is involved in a data race.
              functools.partial(store, output_name=output_names[index]),
              (),
              device_id,
              core_index,
              TPU_MEMORY_SPACE_IDXS[pallas_core.MemorySpace.ANY],
              output_buffer_ids[index],
              (transform,),
              kernel_output_val,
              cur_block_indices[num_inputs + index],
              grid_point,
              ordered=True,
          )

        output_slices : list[Any] = []
        for j, var in enumerate(output_vars):
          if _is_any(var.aval.memory_space):
            output_slices.append(None)
            continue
          assert len(cur_start_indices[num_inputs + j].shape) == 1
          assert len(next_start_indices[num_inputs + j].shape) == 1
          transform = indexing.NDIndexer(
              indices=tuple(
                  indexing.ds(st, sz) if not iid else st  # type: ignore[misc]
                  for st, sz, iid in zip(
                      cur_start_indices[num_inputs + j],
                      block_shapes[num_inputs + j],
                      is_squeeze_dim[num_inputs + j],
                  )
              ),
              shape=output_vals[j].shape,
              int_indexer_shape=(),
          )
          if j in oi_alias_map:
            # Suppress revisiting check for output buffers that are aliased to
            # input buffers.
            output_slices.append(None)
          else:
            output_slices.append((transform,))
          jax.lax.cond(
              (iteration_idx + 1 == loop_bound)
              | jax.lax.reduce_or(
                  cur_start_indices[num_inputs + j]
                  != next_start_indices[num_inputs + j],
                  axes=(0,),
              ),
              functools.partial(_store_to_output_buffer, j, var, transform),
              lambda: None,
          )
        callback.io_callback(
            _check_for_revisiting,
            (),
            device_id,
            core_index,
            loop_idx,
            output_slices,
            ordered=True,
        )

        return (
            iteration_idx + 1,
            next_loop_idx,
            next_grid_point,
            cur_start_indices,
            next_block_indices,
            next_start_indices,
        )

    initial_loop_idx = interpret_utils.get_indices(grid, initial_iteration_idx)
    initial_grid_point = _get_grid_point(
      initial_loop_idx, randomized_grid_coordinates)
    with pallas_core.grid_env(_get_local_grid_env(initial_grid_point)):
      initial_block_indices, initial_start_indices = zip(*[
          _compute_start_indices(
              bm,
              initial_grid_point,
              *scalar_buffer_ids,
              axis_sizes=axis_sizes,
              mesh=mesh,
              axis_indices=axis_indices,
              device_id=device_id,
              local_core_id=core_index,
              mosaic_params=mosaic_params,
              interpret_params=interpret_params,
          )
          for bm in grid_mapping.block_mappings
      ])

    _ = lax.while_loop(
        lambda carry: carry[0] < loop_bound,
        _body,
        (
            initial_iteration_idx,
            initial_loop_idx,
            initial_grid_point,
            initial_start_indices,  # Previous start indices are ignored on the first iteration.
            initial_block_indices,
            initial_start_indices,
        ),
    )

  # TODO(jburnim): Should we only create happens-before here from core 0 to
  # the other cores?
  callback.io_callback(
      _update_clocks_for_device_barrier, (), device_id, ordered=True
  )

  thread_map(_execute_grid_for_core, interpret_params.num_cores_per_device)

  # TODO(jburnim): Should we only create happens-before here from the other
  # # cores to core 0?
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
          TPU_MEMORY_SPACE_IDXS[pallas_core.MemorySpace.ANY],
          output_buffer_id,
          (
              indexing.NDIndexer.from_indices_shape(
                  tuple(indexing.ds(0, s) for s in val.shape),
                  output_buffer_shape,
              ),
          ),
          ordered=True,
      )
      for val, output_buffer_id, output_buffer_shape in zip(
          output_vals, output_buffer_ids, output_buffer_shapes
      )
  ]

  callback.io_callback(_validate, (), device_id, ordered=True)

  # For now, when we're done with a pallas_call, we delete the shared memory.
  # We use a barrier to ensure that all devices are done running the kernel.
  #
  # TODO(jburnim): Get rid of this barrier.  And figure out how this should
  # work if we want to invoke successive pallas_calls that use the same
  # shared memory.
  callback.io_callback(
      _clean_up_shared_memory, (), device_id, ordered=True
  )

  return ret
