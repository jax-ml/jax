# Copyright 2026 The JAX Authors.
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

from collections.abc import Mapping, Sequence
import contextlib
import dataclasses
import functools
import itertools
import threading
import types
from typing import Self

import jax
from jax import numpy as jnp
from jax._src import callback
from jax._src import source_info_util
from jax._src.pallas.mosaic.interpret import utils as interpret_utils
from jax._src.pallas.mosaic.interpret import vector_clock as vc
from jax._src.pallas.mosaic.interpret.race_detection_state import RaceDetectionState
from jax._src.pallas.mosaic_gpu import core as mosaic_gpu_core
from jax._src.pallas.mosaic_gpu.interpret import shared_memory as memory
from jax._src.state import indexing
import numpy as np


IDX_BY_GPU_MEMORY_SPACE: Mapping[mosaic_gpu_core.MemorySpace, int] = (
    types.MappingProxyType(
        {v: i for i, v in enumerate(mosaic_gpu_core.MemorySpace)}
    )
)


GPU_MEMORY_SPACE_BY_IDX = types.MappingProxyType(
    dict(enumerate(mosaic_gpu_core.MemorySpace))
)


def get_memory_space_idx(space: mosaic_gpu_core.MemorySpace | None) -> int:
  if space is None:
    return IDX_BY_GPU_MEMORY_SPACE[mosaic_gpu_core.MemorySpace.SMEM]
  return IDX_BY_GPU_MEMORY_SPACE[space]


def is_smem_memory_space(space: mosaic_gpu_core.MemorySpace | None) -> bool:
  if space is None:
    return True
  return space == mosaic_gpu_core.MemorySpace.SMEM


def is_gmem_memory_space(space: mosaic_gpu_core.MemorySpace | None) -> bool:
  return space == mosaic_gpu_core.MemorySpace.GMEM


_shared_memory: memory.GPUSharedMemory | None = None
_shared_memory_init_lock = threading.Lock()
_races: RaceDetectionState | None = None


def _get_shared_memory() -> memory.GPUSharedMemory:
  assert _shared_memory is not None
  return _shared_memory


def _clear_shared_memory():
  global _shared_memory
  with _shared_memory_init_lock:
    _shared_memory = None


def get_races() -> RaceDetectionState:
  assert _races is not None
  return _races


def reset_gpu_interpret_mode_state():
  """Resets all global, shared state used by GPU interpret mode.

  GPU interpret mode uses global, shared state for simulating memory buffers,
  for race detection, etc., when interpreting a kernel. Normally, this shared
  state is cleaned up after a kernel is interpreted.

  But if an exception is thrown while interpreting a kernel, the shared state
  is not cleaned up, allowing the simulated GPU state to be examined for
  debugging purposes. In this case, the shared state must be reset before
  any further kernels are interpreted.
  """
  global _shared_memory, _races
  with _shared_memory_init_lock:
    _shared_memory = None
    _races = None


# Below we define pairs of _callback_ functions. Each pair consists of
#
#   (1) a module-private function, e.g. `_initialize_shared_memory`, and
#   (2) a thin wrapper around the this module-private function, e.g.
#       `call_initialize_shared_memory`.
#
# The module-private function (1) runs in the Python ("host") process and
# manages interaction of the interpreted Pallas kernel with the memory system,
# represented by the module-global `SharedMemory` object `_shared_memory`.
#
# The wrapper function (2) is to be called from the interpreted Pallas kernel
# (that is simulating a "device", or thread). It serves as the interface between
# the "device" kernel and the "host" memory system and merely passes arguments
# on to the corresponding function (1). Importantly, when the wrapper receives
# an argument that is a Jax (device) array, this argument is received as a Numpy
# (host) array by the corresponding function (1), due to the
# `callback.io_callback` mechanism.


def _initialize_shared_memory(
    num_devices: jnp.ndarray,
    num_threads: jnp.ndarray,
    *,
    interpret_params: interpret_utils.InterpretGPUParams,
):
  global _shared_memory, _races

  num_devices = int(num_devices)
  num_threads = int(num_threads)
  num_total_threads = num_devices * num_threads

  with _shared_memory_init_lock:
    if _shared_memory is None:
      vector_clock_size = interpret_params.get_vector_clock_size(num_devices)
      _races = RaceDetectionState(num_cores=num_total_threads)
      _shared_memory = memory.GPUSharedMemory(
          num_devices=num_devices,
          # We re-use the `SharedMemory`'s capability to model multiple cores
          # per (TPU) device for modeling the  multiple threads on a single GPU
          # device.
          num_cores_per_device=num_threads,
          out_of_bounds_reads=interpret_params.out_of_bounds_reads,
          # TODO(nrink): Support different DMA execution modes on GPU.
          dma_execution_mode="eager",
          uninitialized_memory=interpret_params.uninitialized_memory,
          detect_races=interpret_params.detect_races,
          vector_clock_size=vector_clock_size,
          clocks=[
              vc.make_vector_clock(vector_clock_size)
              for _ in range(num_total_threads)
          ],
          barrier=threading.Barrier(num_devices, action=lambda: None),
          clean_up_barrier=threading.Barrier(
              num_devices, action=_clear_shared_memory
          ),
          logging_mode=interpret_params.logging_mode,
      )
  # The naming of the `num_cores` property of `SharedMemory` originates from the
  # support for multipl cores in a (Megacore) TPU device. As commented above, on
  # GPU we model multiple threads per device as _cores_ in the
  # (TPU-/Megacore-)inspired terminology of`SharedMemory`.
  assert _shared_memory.num_cores == num_total_threads


def call_initialize_shared_memory(
    *,
    num_devices: int,
    num_threads: int,
    interpret_params: interpret_utils.InterpretGPUParams,
):
  callback.io_callback(
      functools.partial(
          _initialize_shared_memory,
          interpret_params=interpret_params,
      ),
      (),
      num_devices,
      num_threads,
      ordered=True,
  )


def _clean_up_shared_memory():
  shared_memory = _get_shared_memory()
  shared_memory.clean_up_barrier.wait()


def call_clean_up_shared_memory():
  callback.io_callback(_clean_up_shared_memory, (), ordered=True)


def _update_clocks_for_device_barrier(device_id: int):
  shared_memory = _get_shared_memory()
  shared_memory.update_clocks_for_device_barrier(device_id)


def call_update_clocks_for_device_barrier(device_id: int):
  callback.io_callback(
      _update_clocks_for_device_barrier, (), device_id, ordered=True
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class HostAllocationRequest:
  """Request for an allocation on a device/thread and in a memory space."""

  memory_space_id: int
  device_id: int
  # Defaults to zero for `AllocationRequest`s that do not specify a thread ID.
  thread_id: int = 0
  # The reference count is needed only for allocations that are explicitly
  # deallocated (with _deallocate_buffer below). This currently only applies to
  # allocations made by a `run_scoped` primitive.
  initial_ref_count: int = 1

  def __iter__(self):
    # We make `self` iterable to ease conversion into Numpy and Jax arrays (cf.
    # methods `as_array` and `as_jax_array` below). Note that for this purpose
    # it would suffice to have any method that return a suitable iterator,
    # instead of implementing the special `__iter__` method. Not implementing
    # `__iter__` would mean that objects of this class cannot (accidentally) be
    # iterated over by clients of the class.
    return iter((
        self.memory_space_id,
        self.device_id,
        self.thread_id,
        self.initial_ref_count,
    ))

  @classmethod
  def shape_and_dtype(cls) -> jax.ShapeDtypeStruct:
    num_fields = len(dataclasses.fields(cls))
    return jax.ShapeDtypeStruct((num_fields,), jnp.int32)

  @property
  def as_array(self) -> np.ndarray:
    return np.array(list(self), dtype=np.int32)

  @property
  def as_jax_array(self) -> jnp.ndarray:
    return jnp.array(list(self), dtype=jnp.int32)

  @classmethod
  def from_array(cls, request: np.ndarray | jnp.ndarray) -> Self:
    if request.shape != cls.shape_and_dtype().shape:
      raise ValueError(
          f"Expected shape {cls.shape_and_dtype().shape} but got"
          f" {request.shape}"
      )
    if not interpret_utils.is_int(request.dtype):
      raise ValueError(f"Expected integer dtype but got {request.dtype}")

    arg_names = [f.name for f in dataclasses.fields(cls)]
    values = map(int, request)
    return cls(**dict(zip(arg_names, values)))


def make_allocation_request_array(
    *,
    memory_space_id: int,
    device_id: int,
    thread_id: int = 0,
    initial_ref_count: int = 1,
) -> jnp.ndarray:
  return HostAllocationRequest(
      memory_space_id=memory_space_id,
      device_id=device_id,
      thread_id=thread_id,
      initial_ref_count=initial_ref_count,
  ).as_jax_array


@dataclasses.dataclass(frozen=True, kw_only=True)
class HostAllocationKey(HostAllocationRequest):
  """Key for an allocation in shared memory."""

  buffer_id: int

  def __iter__(self):
    # Note that implementing `__iter__` here affects the bahviour of the
    # `as_array` and `as_jax_array` methods of the base class. This is intended.
    yield from super().__iter__()
    yield self.buffer_id


def _allocate_buffer_for_all_threads(
    device_id: np.ndarray,
    allocation_request: np.ndarray,
    value: np.ndarray,
) -> np.ndarray:
  """Allocates a buffer for the given `allocation_request`.

  While only a single buffer is allocated, we increment the next buffer ID on
  `_shared_memory` for all threads. (This is analogous to the behavior when
  interpreting TPU kernels with multiple cores per TPU device.)

  Args:
    allocation_request: Array that converts into an `HostAllocationRequest` with
      `thread_id` set to zero. This requirement can be thought of as associating
      the allocated buffer (that is shared across all threads) with the zeroth
      thread.
    value: Array of values to initialize the allocated buffer with.

  Returns:
    `AllocationKey` to refer to the allocated buffer.

  Raises:
    ValueError: If the `thread_id` in `allocation_request` is not zero.
  """
  device_id = int(device_id)
  allocation_request = HostAllocationRequest.from_array(allocation_request)
  if allocation_request.thread_id != 0:
    raise ValueError(
        "`thread_id` must be zero when allocating a buffer for all threads"
    )
  value = np.array(value)
  shared_memory = _get_shared_memory()

  key = None
  buffer_id = None
  for thread_id in range(shared_memory.num_cores_per_device):
    buffer_id_for_thread_id = shared_memory.get_next_buffer_id(
        device_id, thread_id
    )
    if not buffer_id:
      buffer_id = buffer_id_for_thread_id
    else:
      # We keep the buffer ids in sync across all threads. This implies, in
      # particular, that every instance of the assignment to `key` below assigns
      # an `AllocationKey` object with the same attributes.
      assert buffer_id == buffer_id_for_thread_id

    key = HostAllocationKey(
        memory_space_id=allocation_request.memory_space_id,
        device_id=allocation_request.device_id,
        thread_id=0,
        initial_ref_count=allocation_request.initial_ref_count,
        buffer_id=buffer_id,
    )
    ref_count = allocation_request.initial_ref_count
    # We rely on the fact that `allocate_buffer` will not allocate a new buffer
    # if one with the same key already exists.
    shared_memory.allocate_buffer(key, ref_count=ref_count, value=value)

  # We expect the `for`-loop above to have executed its body at least once.
  assert key is not None
  return key.as_array


def call_allocate_buffer_for_all_threads(
    device_id: int,
    allocation_request: jnp.ndarray,
    value: jnp.ndarray,
) -> jnp.ndarray:
  return callback.io_callback(
      _allocate_buffer_for_all_threads,
      HostAllocationKey.shape_and_dtype(),
      device_id,
      allocation_request,
      value,
      ordered=True,
  )


def _allocate_buffer(
    device_id: np.ndarray,
    thread_id: np.ndarray,
    allocation_request: np.ndarray,
    value: np.ndarray,
) -> np.ndarray:
  """Allocates a buffer for the given `allocation_request`.

  Args:
    allocation_request: Array that converts into a `HostAllocationRequest`.
    value: Array of values to initialize the allocated buffer with.

  Returns:
    `AllocationKey` to refer to the allocated buffer.
  """
  device_id = int(device_id)
  thread_id = int(thread_id)
  allocation_request = HostAllocationRequest.from_array(allocation_request)
  value = np.array(value)
  shared_memory = _get_shared_memory()

  buffer_id = shared_memory.get_next_buffer_id(device_id, thread_id)

  key = HostAllocationKey(
      memory_space_id=allocation_request.memory_space_id,
      device_id=allocation_request.device_id,
      thread_id=allocation_request.thread_id,
      initial_ref_count=allocation_request.initial_ref_count,
      buffer_id=buffer_id,
  )
  ref_count = allocation_request.initial_ref_count
  shared_memory.allocate_buffer(key, ref_count=ref_count, value=value)
  return key.as_array


def call_allocate_buffer(
    device_id: int,
    thread_id: int,
    allocation_request: jnp.ndarray,
    value: jnp.ndarray,
) -> jnp.ndarray:
  return callback.io_callback(
      _allocate_buffer,
      HostAllocationKey.shape_and_dtype(),
      device_id,
      thread_id,
      allocation_request,
      value,
      ordered=True,
  )


def _deallocate_buffer(allocation_key: np.ndarray):
  """Decreases the reference count of the buffer with `allocation_key` (Deallocates the buffer if its reference count becomes zero)."""
  allocation_key = HostAllocationKey.from_array(allocation_key)
  shared_memory = _get_shared_memory()
  shared_memory.deallocate_buffer(allocation_key)


def call_deallocate_buffer(allocation_key: jnp.ndarray):
  callback.io_callback(
      _deallocate_buffer,
      None,
      allocation_key,
      ordered=True,
  )


def _handle_out_of_bounds_read(
    ret: np.ndarray | None,
    full_read_shape: tuple[int, ...],
    shape: Sequence[int],
    dtype: np.dtype,
    allocation_key: HostAllocationKey,
    read_range: tuple[int | slice, ...],
    shared_memory: memory.GPUSharedMemory,
    source_info,
    input_name: str | None,
    block_indices: tuple[int, ...] | None,
    grid_loop_idx: tuple[int, ...] | None,
) -> np.ndarray:
  """Handles out-of-bounds read based on shared_memory configuration."""
  if shared_memory.out_of_bounds_reads == "raise":
    if source_info is None:
      ctx = contextlib.nullcontext()
    else:
      ctx = source_info_util.user_context(
          traceback=source_info.traceback, name_stack=source_info.name_stack
      )  # type: ignore[assignment]
    with ctx:
      if input_name is None:
        raise IndexError(
            f"Out-of-bounds read of {allocation_key}:"
            f" reading [{read_range}] but buffer has shape {shape}."
        )
      else:
        # Different error message when we are reading a block of an input,
        # to copy it to a buffer before invoking the kernel body.
        raise IndexError(
            f"Out-of-bounds block index {block_indices} for {allocation_key},"
            f' input "{input_name}" in iteration {grid_loop_idx}:'
            f" reading [{read_range}] but input has shape {shape}."
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
    return uninit_array
  else:
    uninit_array[tuple(slice(s) for s in ret.shape)] = ret
    return uninit_array


def _is_dynamic(indexer: indexing.NDIndexer) -> bool:
  return any(
      isinstance(idx, indexing.Slice)
      and (idx.is_dynamic_start or idx.is_dynamic_size)
      for idx in indexer.indices
  )


def _validate_transforms(transforms):
  for transform in transforms:
    match transform:
      case indexing.NDIndexer():
        if _is_dynamic(transform):
          raise ValueError(
              "Dynamic indexing not supported in GPU interpret mode"
          )
      case _:
        raise ValueError(f"Unsupported transform: {transform}")


def _get(
    device_id: np.ndarray,
    thread_id: np.ndarray,
    allocation_key: np.ndarray,
    transforms,
    block_indices=None,
    grid_loop_idx=None,
    clock=None,
    source_info=None,
    input_name=None,
) -> np.ndarray:
  """Performs a read from the buffer for `allocation_key_as_array` from the given device and thread."""
  device_id = int(device_id)
  thread_id = int(thread_id)
  allocation_key = HostAllocationKey.from_array(allocation_key)

  _validate_transforms(transforms)
  # TODO(nrink): Support tiling and swizzling transforms.
  transforms = jax.tree.map(int, transforms)

  if input_name is not None:
    # NOTE: input_name, block_indices, and grid_loop_idx are set only if this
    # function is being called to read a block from a pallas_call input (at the
    # start of one iteration of the kernel body).
    assert block_indices is not None
    block_indices = tuple(int(x) for x in block_indices)
    assert grid_loop_idx is not None
    grid_loop_idx = tuple(int(x) for x in grid_loop_idx)

  shared_memory = _get_shared_memory()

  global_core_id = shared_memory.get_global_core_id(device_id, thread_id)

  read_range = interpret_utils.to_range(transforms)
  ret, (shape, dtype), clock_ = shared_memory.get_buffer_content(
      allocation_key, read_range, global_core_id
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
    ret = _handle_out_of_bounds_read(
        ret,
        full_read_shape,
        shape,
        dtype,
        allocation_key,
        read_range,
        shared_memory,
        source_info,
        input_name,
        block_indices,
        grid_loop_idx,
    )

  if shared_memory.detect_races:
    get_races().check_read(
        device_id,
        thread_id,
        clock,
        allocation_key,
        read_range,
        source_info=source_info,
    )
  return ret


def call_get(
    *,
    result_shape_and_dtype,
    device_id: int,
    thread_id: int,
    allocation_key: jnp.ndarray,
    transforms,
    block_indices=None,
    grid_loop_idx=None,
    clock=None,
    source_info=None,
    input_name=None,
) -> jnp.ndarray:
  return callback.io_callback(
      functools.partial(_get, source_info=source_info, input_name=input_name),
      result_shape_and_dtype,
      device_id,
      thread_id,
      allocation_key,
      transforms,
      block_indices,
      grid_loop_idx,
      clock,
      ordered=True,
  )


def _swap(
    device_id: np.ndarray,
    thread_id: np.ndarray,
    allocation_key_as_array: np.ndarray,
    transforms,
    val,
    mask,
    *,
    source_info=None,
):
  """Performs a swap into the buffer for `allocation_key_as_array` from the given device and thread."""
  device_id = int(device_id)
  thread_id = int(thread_id)
  allocation_key = HostAllocationKey.from_array(allocation_key_as_array)

  _validate_transforms(transforms)
  # TODO(nrink): Support tiling and swizzling transforms.
  transforms = jax.tree.map(int, transforms)

  val = np.array(val)
  mask = np.array(mask) if mask is not None else None
  if mask is not None:
    assert mask.shape == val.shape

  shared_memory = _get_shared_memory()

  global_core_id = shared_memory.get_global_core_id(device_id, thread_id)

  read_write_range = interpret_utils.to_range(transforms)
  ret, (shape, _), clock = shared_memory.swap_buffer_content(
      allocation_key, read_write_range, val, mask, global_core_id
  )

  if ret is None:
    if mask is None:
      raise ValueError(
          f"Out-of-bounds swap of {allocation_key}:"
          f" swapping [{read_write_range}] but buffer has shape"
          f" {shape} ."
      )
    else:
      # TODO(jburnim): Include indices of out-of-bounds locations where mask
      # is True.
      raise ValueError(
          f"Out-of-bounds masked swap of {allocation_key}: swapping"
          f" [{read_write_range}] but buffer has shape {shape} . "
      )

  if shared_memory.detect_races:
    get_races().check_write(
        device_id,
        thread_id,
        clock,
        allocation_key,
        read_write_range,
        source_info=source_info,
    )
  return ret


def call_swap(
    *,
    result_shape_and_dtype,
    device_id: int,
    thread_id: int,
    allocation_key: jnp.ndarray,
    transforms,
    val,
    mask,
    source_info=None,
):
  return callback.io_callback(
      functools.partial(_swap, source_info=source_info),
      result_shape_and_dtype,
      device_id,
      thread_id,
      allocation_key,
      transforms,
      val,
      mask,
      ordered=True,
  )


def _allocate_barriers(
    device_id: np.ndarray,
    thread_id: np.ndarray,
    num_arrivals: np.ndarray,
    num_barriers: np.ndarray,
    ref_count: np.ndarray,
) -> np.ndarray:
  device_id = int(device_id)
  thread_id = int(thread_id)
  num_arrivals = int(num_arrivals)
  num_barriers = int(num_barriers)
  ref_count = int(ref_count)
  shared_memory = _get_shared_memory()

  keys = []
  for _ in range(num_barriers):
    # Advance `shared_memory`'s internal buffer id counter for all threads that
    # call into this function.
    barrier_id = shared_memory.get_next_buffer_id(device_id, thread_id)
    smem_space_id = IDX_BY_GPU_MEMORY_SPACE[mosaic_gpu_core.SMEM]
    key = HostAllocationKey(
        memory_space_id=smem_space_id,
        device_id=device_id,
        # Barriers are shared between threads. Hence we associate all
        # allocations for `Barrier`s with the 0th thread.
        thread_id=0,
        initial_ref_count=ref_count,
        buffer_id=barrier_id,
    )

    shared_memory.allocate_barrier(
        device_id,
        thread_id,
        key,
        ref_count=ref_count,
        num_arrivals=num_arrivals,
    )
    keys.append(key.as_array)

  assert len(keys) == num_barriers
  return np.array(keys, dtype=np.int32)


def call_allocate_barriers(
    device_id: int,
    thread_id: int,
    num_arrivals: int,
    num_barriers: int,
    ref_count: int,
) -> jnp.ndarray:
  shape_and_dtype = HostAllocationKey.shape_and_dtype()
  result_shape = (num_barriers, *shape_and_dtype.shape)
  result_shape_and_dtype = jax.ShapeDtypeStruct(
      result_shape, shape_and_dtype.dtype
  )
  return callback.io_callback(
      _allocate_barriers,
      result_shape_and_dtype,
      device_id,
      thread_id,
      num_arrivals,
      num_barriers,
      ref_count,
      ordered=True,
  )


def _deallocate_barrier(
    device_id: np.ndarray, thread_id: np.ndarray, allocation_key: np.ndarray
):
  device_id = int(device_id)
  thread_id = int(thread_id)

  assert len(allocation_key.shape) == 2
  num_barriers = allocation_key.shape[0]

  keys_to_deallocate = []
  for i in range(num_barriers):
    keys_to_deallocate.append(allocation_key[i, :])

  shared_memory = _get_shared_memory()

  for key in keys_to_deallocate:
    barrier_allocation_key = HostAllocationKey.from_array(key)
    shared_memory.deallocate_barrier(
        device_id, thread_id, barrier_allocation_key
    )


def call_deallocate_barrier(
    device_id: int, thread_id: int, allocation_key: jnp.ndarray
):
  callback.io_callback(
      _deallocate_barrier,
      None,
      device_id,
      thread_id,
      allocation_key,
      ordered=True,
  )


def _barrier_wait(device_id: int, thread_id: int, allocation_key: np.ndarray):
  device_id = int(device_id)
  thread_id = int(thread_id)
  barrier_key = HostAllocationKey.from_array(allocation_key)
  shared_memory = _get_shared_memory()

  barrier, _ = shared_memory.get_barrier_and_increment_clock(
      barrier_key, device_id, thread_id
  )
  barrier.wait(device_id, thread_id)


def call_barrier_wait(
    device_id: int, thread_id: int, allocation_key: jnp.ndarray
):
  callback.io_callback(
      _barrier_wait,
      None,
      device_id,
      thread_id,
      allocation_key,
      ordered=True,
  )


def _barrier_arrive(device_id: int, thread_id: int, allocation_key: np.ndarray):
  device_id = int(device_id)
  thread_id = int(thread_id)
  barrier_key = HostAllocationKey.from_array(allocation_key)
  shared_memory = _get_shared_memory()

  barrier, clock = shared_memory.get_barrier_and_increment_clock(
      barrier_key, device_id, thread_id
  )
  barrier.arrive(device_id, thread_id, clock)


def call_barrier_arrive(
    device_id: int, thread_id: int, allocation_key: jnp.ndarray
):
  callback.io_callback(
      _barrier_arrive,
      None,
      device_id,
      thread_id,
      allocation_key,
      ordered=True,
  )


def _assert_no_barriers_allocated():
  _get_shared_memory().assert_no_barriers_allocated()


def call_assert_no_barriers_allocated():
  callback.io_callback(_assert_no_barriers_allocated, (), ordered=True)
