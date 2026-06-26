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
from typing import Any, Self

import jax
from jax import numpy as jnp
from jax._src import callback
from jax._src import source_info_util
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic.interpret import utils as interpret_utils
from jax._src.pallas.mosaic.interpret import vector_clock as vc
from jax._src.pallas.mosaic.interpret.race_detection_state import RaceDetectionState
from jax._src.pallas.mosaic_gpu import core as mosaic_gpu_core
from jax._src.pallas.mosaic_gpu.interpret import shared_memory as memory
from jax._src.pallas.mosaic_gpu.interpret.params import InterpretGPUParams
from jax._src.state import indexing
from jax.experimental.mosaic import gpu as mgpu
import numpy as np


IDX_BY_GPU_MEMORY_SPACE: Mapping[mosaic_gpu_core.MemorySpace, int]
IDX_BY_GPU_MEMORY_SPACE = types.MappingProxyType(
    {v: i for i, v in enumerate(mosaic_gpu_core.MemorySpace)}
)


GPU_MEMORY_SPACE_BY_IDX: Mapping[int, mosaic_gpu_core.MemorySpace]
GPU_MEMORY_SPACE_BY_IDX = types.MappingProxyType(
    dict(enumerate(mosaic_gpu_core.MemorySpace))
)


def get_memory_space_idx(space: mosaic_gpu_core.MemorySpace) -> int:
  if space is pallas_core.MemorySpace.DEFAULT:
    return IDX_BY_GPU_MEMORY_SPACE[mosaic_gpu_core.MemorySpace.SMEM]
  return IDX_BY_GPU_MEMORY_SPACE[space]


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


TOKEN_SHAPE_DTYPE = jax.ShapeDtypeStruct((), jnp.int32)


def ordering_barrier(token):
  return token


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
# on to the corresponding function (1).


def _initialize_shared_memory(
    *,
    token: jax.Array,
    num_gpus: jax.Array,
    num_threads_per_block: jax.Array,
    num_blocks_per_cluster: jax.Array,
    interpret_params: InterpretGPUParams,
):
  global _shared_memory, _races

  num_gpus_as_int = int(num_gpus)
  num_threads_per_block_as_int = int(num_threads_per_block)
  num_blocks_per_cluster_as_int = int(num_blocks_per_cluster)
  del num_gpus, num_threads_per_block, num_blocks_per_cluster

  num_total_concurrent_threads = (
      num_gpus_as_int
      * num_threads_per_block_as_int
      * num_blocks_per_cluster_as_int
  )

  with _shared_memory_init_lock:
    if _shared_memory is None:
      _races = RaceDetectionState(num_cores=num_total_concurrent_threads)
      _shared_memory = memory.GPUSharedMemory(
          num_devices=num_gpus_as_int,
          num_threads_per_block=num_threads_per_block_as_int,
          num_blocks_per_cluster=num_blocks_per_cluster_as_int,
          num_tma_threads_per_device=interpret_params.num_tma_threads_per_device,
          out_of_bounds_reads=interpret_params.out_of_bounds_reads,
          # TODO(nrink): Support different DMA execution modes on GPU.
          dma_execution_mode="eager",
          uninitialized_memory=interpret_params.uninitialized_memory,
          detect_races=interpret_params.detect_races,
          barrier=threading.Barrier(num_gpus_as_int, action=lambda: None),
          clean_up_barrier=threading.Barrier(
              num_gpus_as_int, action=_clear_shared_memory
          ),
          logging_mode=interpret_params.logging_mode,
      )
  # The naming of the `num_cores` property of `SharedMemory` originates from the
  # support for multipl cores in a (Megacore) TPU device. As commented above, on
  # GPU we model multiple Pallas threads per device as _cores_ in the
  # (TPU-/Megacore-)inspired terminology of `SharedMemory`.
  assert _shared_memory.num_cores == num_total_concurrent_threads
  return token


def call_initialize_shared_memory(
    *,
    token: jax.Array,
    num_gpus: jax.Array,
    num_threads_per_block: jax.Array,
    num_blocks_per_cluster: jax.Array,
    interpret_params: InterpretGPUParams,
):
  return callback.io_callback(
      functools.partial(
          _initialize_shared_memory,
          interpret_params=interpret_params,
      ),
      TOKEN_SHAPE_DTYPE,
      token=token,
      num_gpus=num_gpus,
      num_threads_per_block=num_threads_per_block,
      num_blocks_per_cluster=num_blocks_per_cluster,
  )


def _clean_up_shared_memory(token):
  shared_memory = _get_shared_memory()
  shared_memory.clean_up_barrier.wait()
  return token


def call_clean_up_shared_memory(token):
  return callback.io_callback(
      _clean_up_shared_memory, TOKEN_SHAPE_DTYPE, token
  )


def _update_clocks_for_device_barrier(token, device_id: jax.Array):
  device_id_as_int = int(device_id)
  del device_id

  shared_memory = _get_shared_memory()
  shared_memory.update_clocks_for_device_barrier(device_id_as_int)
  return token


def call_update_clocks_for_device_barrier(token, device_id: jax.Array):
  return callback.io_callback(
      _update_clocks_for_device_barrier,
      TOKEN_SHAPE_DTYPE,
      token,
      device_id,
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
    # We make `self` iterable to ease conversion into arrays (cf. method
    # `as_jax_array` below). Note that for this purpose it would suffice to have
    # any method that return a suitable iterator, instead of implementing the
    # special `__iter__` method. Not implementing `__iter__` would mean that
    # objects of this class cannot (accidentally) be iterated over by clients of
    # the class.
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
  def as_np_array(self) -> np.ndarray:
    return np.array(list(self), dtype=np.int32)

  @classmethod
  def from_array(cls, request: jax.Array) -> Self:
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


def _make_allocation_request_array(
    *,
    token: jax.Array,
    memory_space_id: int,
    device_id: jax.Array,
    thread_id: jax.Array | None = None,
    initial_ref_count: int = 1,
) -> tuple[jax.Array, np.ndarray]:
  device_id_as_int = int(device_id)
  thread_id_as_int = int(thread_id) if thread_id is not None else 0
  del device_id, thread_id

  return token, HostAllocationRequest(
      memory_space_id=memory_space_id,
      device_id=device_id_as_int,
      thread_id=thread_id_as_int,
      initial_ref_count=initial_ref_count,
  ).as_np_array


def call_make_allocation_request_array(
    *,
    token: jax.Array,
    memory_space_id: int,
    device_id: jax.Array,
    thread_id: jax.Array | None = None,
    initial_ref_count: int = 1,
) -> tuple[jax.Array, jax.Array]:
  return callback.io_callback(
      _make_allocation_request_array,
      (TOKEN_SHAPE_DTYPE, HostAllocationRequest.shape_and_dtype()),
      token=token,
      device_id=device_id,
      memory_space_id=memory_space_id,
      thread_id=thread_id,
      initial_ref_count=initial_ref_count,
      # The callback has no side-effect, so we allow this to be reordered
      # relative to other callbacks.
      ordered=False,
  )


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
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array | None,
    allocation_request_as_array: jax.Array,
    value: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
) -> tuple[jax.Array, np.ndarray]:
  """Allocates a buffer for the given `allocation_request`.

  While only a single buffer is allocated, we increment the next buffer ID on
  `_shared_memory` for all threads. (This is analogous to the behavior when
  interpreting TPU kernels with multiple cores per TPU device.)

  Args:
    allocation_request_as_array: Array that converts into an
      `HostAllocationRequest` with `thread_id` set to zero. This requirement can
      be thought of as associating the allocated buffer (that is shared across
      all threads) with the zeroth thread.
    value: Array of values to initialize the allocated buffer with.

  Returns:
    `AllocationKey` to refer to the allocated buffer.

  Raises:
    ValueError: If the `thread_id` in `allocation_request` is not zero.
  """
  device_id_as_int = int(device_id)
  grid_point_coords_as_tuple = (
      tuple(int(x) for x in grid_point_coords)
      if grid_point_coords is not None
      else None
  )
  allocation_request = HostAllocationRequest.from_array(
      allocation_request_as_array
  )
  del device_id, grid_point_coords, allocation_request_as_array

  if allocation_request.thread_id != 0:
    raise ValueError(
        "`thread_id` must be zero when allocating a buffer for all threads"
    )
  assert allocation_request.memory_space_id != get_memory_space_idx(
      mosaic_gpu_core.MemorySpace.REGS
  )

  shared_memory = _get_shared_memory()

  key: HostAllocationKey | None = None
  buffer_id: int | None = None
  for thread_id in range(shared_memory.num_cores_per_device):
    buffer_id_for_thread_id = shared_memory.get_next_buffer_id(
        device_id_as_int, thread_id
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
    shared_memory.allocate_buffer(
        key,
        ref_count=ref_count,
        value=np.array(value),
        logging_info=interpret_utils.GPULoggingInfo(
            device_id=device_id_as_int,
            grid_point_coords=grid_point_coords_as_tuple,
            thread_id=0,
            source_info=source_info,
        ),
    )

  # We expect the `for`-loop above to have executed its body at least once.
  assert key is not None
  return token, key.as_np_array


def call_allocate_buffer_for_all_threads(
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array | None,
    allocation_request_as_array: jax.Array,
    value: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
) -> tuple[jax.Array, jax.Array]:
  return callback.io_callback(
      functools.partial(
          _allocate_buffer_for_all_threads, source_info=source_info
      ),
      (TOKEN_SHAPE_DTYPE, HostAllocationKey.shape_and_dtype()),
      token,
      device_id,
      grid_point_coords,
      allocation_request_as_array,
      value,
  )


def _allocate_buffer(
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    allocation_request_as_array: jax.Array,
    value: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
) -> tuple[jax.Array, np.ndarray]:
  """Allocates a buffer for the given `allocation_request`.

  Args:
    allocation_reques_as_array: Array that converts into a
      `HostAllocationRequest`.
    value: Array of values to initialize the allocated buffer with.
    source_info: Information about the source code location of the allocation.

  Returns:
    `AllocationKey` to refer to the allocated buffer.
  """
  device_id_as_int = int(device_id)
  grid_point_coords_as_tuple = tuple(int(x) for x in grid_point_coords)
  thread_id_as_int = int(thread_id)
  allocation_request = HostAllocationRequest.from_array(
      allocation_request_as_array
  )
  del device_id, grid_point_coords, thread_id, allocation_request_as_array

  shared_memory = _get_shared_memory()

  if (allocation_request.memory_space_id
      == get_memory_space_idx(mosaic_gpu_core.MemorySpace.REGS)):
    # For barrier and buffer identifiers to line up across threads, we rely on
    # each thread making the same sequence of allocations.  But threads are
    # permitted to make different REGS allocations, so we use a different
    # sequence of integer identifiers for REGS allocations.
    buffer_id = shared_memory.get_next_wgmma_accumulator_id(
        device_id_as_int, thread_id_as_int)
  else:
    buffer_id = shared_memory.get_next_buffer_id(
        device_id_as_int, thread_id_as_int)

  key = HostAllocationKey(
      memory_space_id=allocation_request.memory_space_id,
      device_id=allocation_request.device_id,
      thread_id=allocation_request.thread_id,
      initial_ref_count=allocation_request.initial_ref_count,
      buffer_id=buffer_id,
  )
  ref_count = allocation_request.initial_ref_count
  shared_memory.allocate_buffer(
      key,
      ref_count=ref_count,
      value=np.array(value),
      logging_info=interpret_utils.GPULoggingInfo(
          device_id=device_id_as_int,
          grid_point_coords=grid_point_coords_as_tuple,
          thread_id=thread_id_as_int,
          source_info=source_info,
      ),
  )
  return token, key.as_np_array


def call_allocate_buffer(
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    allocation_request_as_array: jax.Array,
    value: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
) -> tuple[jax.Array, jax.Array]:
  return callback.io_callback(
      functools.partial(_allocate_buffer, source_info=source_info),
      (TOKEN_SHAPE_DTYPE, HostAllocationKey.shape_and_dtype()),
      token,
      device_id,
      grid_point_coords,
      thread_id,
      allocation_request_as_array,
      value,
  )


def _deallocate_buffer(
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    allocation_key_as_array: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
):
  """Decreases the reference count of the buffer with `allocation_key` (Deallocates the buffer if its reference count becomes zero)."""
  device_id_as_int = int(device_id)
  grid_point_coords_as_tuple = tuple(int(x) for x in grid_point_coords)
  thread_id_as_int = int(thread_id)
  allocation_key = HostAllocationKey.from_array(allocation_key_as_array)
  del device_id, grid_point_coords, thread_id, allocation_key_as_array
  shared_memory = _get_shared_memory()

  shared_memory.deallocate_buffer(
      allocation_key,
      logging_info=interpret_utils.GPULoggingInfo(
          device_id=device_id_as_int,
          grid_point_coords=grid_point_coords_as_tuple,
          thread_id=thread_id_as_int,
          source_info=source_info,
      ),
  )
  return token


def call_deallocate_buffer(
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    allocation_key_as_array: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
):
  return callback.io_callback(
      functools.partial(_deallocate_buffer, source_info=source_info),
      TOKEN_SHAPE_DTYPE,
      token,
      device_id,
      grid_point_coords,
      thread_id,
      allocation_key_as_array,
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
      )
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
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array | None,
    thread_id: jax.Array,
    allocation_key_as_array: jax.Array,
    transforms,
    block_indices=None,
    grid_loop_idx=None,
    clock=None,
    increment_clock: bool = True,
    source_info=None,
    input_name=None,
) -> tuple[jax.Array, np.ndarray]:
  """Performs a read from the buffer for `allocation_key_as_array` from the given device and thread."""
  device_id_as_int = int(device_id)
  grid_point_coords_as_tuple = (
      tuple(int(x) for x in grid_point_coords)
      if grid_point_coords is not None
      else None
  )
  thread_id_as_int = int(thread_id)
  allocation_key = HostAllocationKey.from_array(allocation_key_as_array)
  del device_id, grid_point_coords, thread_id, allocation_key_as_array

  transforms = _remove_noop_transforms(transforms)
  _validate_transforms(transforms)
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

  global_thread_id = shared_memory.get_global_thread_id(
      device_id_as_int, thread_id_as_int
  )

  read_range = interpret_utils.to_range(transforms)
  ret, (shape, dtype), clock_ = shared_memory.get_buffer_content(
      allocation_key,
      read_range,
      global_thread_id,
      increment_clock=increment_clock,
      logging_info=interpret_utils.GPULoggingInfo(
          device_id=device_id_as_int,
          grid_point_coords=grid_point_coords_as_tuple,
          thread_id=thread_id_as_int,
          source_info=source_info,
      ),
  )
  clock = clock if clock is not None else clock_

  # Compute the shape of the read value, assuming the read is fully in-bounds.
  # TODO(jburnim): We already know this shape in the Jaxpr where we insert a
  # callback to `get`.  Should we just pass the shape to `get`?
  # TODO(jburnim): Move to a helper function?
  new_full_read_shape: list[int] = []
  assert len(read_range) <= len(shape)
  for dim_size, idx_or_slice in itertools.zip_longest(
      shape, read_range, fillvalue=None
  ):
    assert isinstance(dim_size, int)
    if idx_or_slice is None:
      new_full_read_shape.append(dim_size)
    elif isinstance(idx_or_slice, int):
      continue
    else:
      dim_size = (idx_or_slice.stop - idx_or_slice.start) // idx_or_slice.step
      assert isinstance(dim_size, int)
      new_full_read_shape.append(dim_size)
  full_read_shape = tuple(new_full_read_shape)
  del new_full_read_shape

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
        device_id_as_int,
        thread_id_as_int,
        clock,
        allocation_key,
        read_range,
        source_info=source_info,
    )
  return token, ret


def call_get(
    *,
    token: jax.Array,
    result_shape_and_dtype,
    device_id: jax.Array,
    grid_point_coords: jax.Array | None,
    thread_id: jax.Array,
    allocation_key_as_array: jax.Array,
    transforms,
    block_indices=None,
    grid_loop_idx=None,
    clock=None,
    source_info=None,
    input_name=None,
) -> tuple[jax.Array, jax.Array]:
  return callback.io_callback(
      functools.partial(_get, source_info=source_info, input_name=input_name),
      (TOKEN_SHAPE_DTYPE, result_shape_and_dtype),
      token,
      device_id,
      grid_point_coords,
      thread_id,
      allocation_key_as_array,
      transforms,
      block_indices,
      grid_loop_idx,
      clock,
  )


def _swap(
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    allocation_key_as_array: jax.Array,
    transforms,
    val: np.ndarray,
    mask: jax.Array | None,
    *,
    clock=None,
    increment_clock: bool = True,
    source_info=None,
) -> tuple[jax.Array, np.ndarray]:
  """Performs a swap into the buffer for `allocation_key` from the given device and thread."""
  device_id_as_int = int(device_id)
  grid_point_coords_as_tuple = tuple(int(x) for x in grid_point_coords)
  thread_id_as_int = int(thread_id)
  allocation_key = HostAllocationKey.from_array(allocation_key_as_array)
  del device_id, thread_id, allocation_key_as_array

  transforms = _remove_noop_transforms(transforms)
  _validate_transforms(transforms)
  transforms = jax.tree.map(int, transforms)

  if mask is not None:
    assert mask.shape == val.shape

  shared_memory = _get_shared_memory()

  global_thread_id = shared_memory.get_global_thread_id(
      device_id_as_int, thread_id_as_int
  )

  read_write_range = interpret_utils.to_range(transforms)
  ret, (shape, _), clock_ = shared_memory.swap_buffer_content(
      allocation_key,
      read_write_range,
      np.array(val),
      np.array(mask) if mask is not None else None,
      global_thread_id,
      increment_clock=increment_clock,
      logging_info=interpret_utils.GPULoggingInfo(
          device_id=device_id_as_int,
          grid_point_coords=grid_point_coords_as_tuple,
          thread_id=thread_id_as_int,
          source_info=source_info,
      ),
  )
  clock = clock if clock is not None else clock_

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
        device_id_as_int,
        thread_id_as_int,
        clock,
        allocation_key,
        read_write_range,
        source_info=source_info,
    )
  return token, ret


def call_swap(
    *,
    token: jax.Array,
    result_shape_and_dtype,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    allocation_key_as_array: jax.Array,
    transforms,
    val: jax.Array,
    mask: jax.Array | None,
    clock=None,
    source_info=None,
) -> tuple[jax.Array, jax.Array]:
  return callback.io_callback(
      functools.partial(_swap, source_info=source_info),
      (TOKEN_SHAPE_DTYPE, result_shape_and_dtype),
      token,
      device_id,
      grid_point_coords,
      thread_id,
      allocation_key_as_array,
      transforms,
      val,
      mask,
      clock=clock,
  )


def get_thread_id_for_collective_allocation_key(
    thread_id: int,
    axes_dims: tuple[int, ...],
    is_last_thread_axis_collective: bool,
) -> int:
  """Returns the thread ID to use for the allocation key in a collective allocation.

  Only the last thread coordinate (corresponding to the threads in a block) can
  be collective; whether this is the case is determined by
  `is_last_thread_axis_collective`.

  Args:
    thread_id: A 'flat' thread ID.
    axes_dims: The dimensions of the cluster axes and block (row-major order,
      where the last/minor-most dimension is the block dimension).
    is_last_thread_axis_collective: A boolean indicating whether the last thread
      axis (correspodning to the threads in a block) is collective.

  Returns:
    The thread ID to use for the allocation key in a collective allocation.
  """
  if is_last_thread_axis_collective:
    return thread_id // axes_dims[-1]
  else:
    return thread_id


def _allocate_barriers(
    *,
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    axes_dims: tuple[int, ...],
    num_arrivals: jax.Array,
    num_barriers: jax.Array,
    ref_count: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
) -> tuple[jax.Array, np.ndarray]:
  device_id_as_int = int(device_id)
  grid_point_coords_as_tuple = tuple(int(x) for x in grid_point_coords)
  thread_id_as_int = int(thread_id)
  axes_dims = tuple(int(x) for x in axes_dims)
  num_arrivals_as_int = int(num_arrivals)
  num_barriers_as_int = int(num_barriers)
  ref_count_as_int = int(ref_count)
  del device_id, grid_point_coords, thread_id, num_arrivals, num_barriers, ref_count

  shared_memory = _get_shared_memory()

  keys = []
  for _ in range(num_barriers_as_int):
    # Advance `shared_memory`'s internal buffer id counter for all threads that
    # call into this function.
    barrier_id = shared_memory.get_next_buffer_id(
        device_id_as_int, thread_id_as_int
    )
    smem_space_id = IDX_BY_GPU_MEMORY_SPACE[mosaic_gpu_core.SMEM]

    # Barriers are shared between threads. For each group of threads that share
    # a barrier, we compute the thread ID to be used for the allocation key.
    # Invariant: `thread_id_for_key` is the same for all threads in a group that
    # shares the barrier.
    thread_id_for_key = get_thread_id_for_collective_allocation_key(
        thread_id_as_int, axes_dims, is_last_thread_axis_collective=True
    )

    key = HostAllocationKey(
        memory_space_id=smem_space_id,
        device_id=device_id_as_int,
        thread_id=thread_id_for_key,
        initial_ref_count=ref_count_as_int,
        buffer_id=barrier_id,
    )

    shared_memory.allocate_barrier(
        key,
        ref_count=ref_count_as_int,
        num_arrivals=num_arrivals_as_int,
        logging_info=interpret_utils.GPULoggingInfo(
            device_id=device_id_as_int,
            grid_point_coords=grid_point_coords_as_tuple,
            thread_id=thread_id_as_int,
            source_info=source_info,
        ),
    )
    keys.append(key.as_np_array)

  assert len(keys) == num_barriers_as_int
  return token, np.array(keys, dtype=np.int32)


def call_allocate_barriers(
    *,
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    axes_dims: tuple[int, ...],
    num_arrivals: jax.Array,
    num_barriers: jax.Array,
    ref_count: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
) -> tuple[jax.Array, jax.Array]:
  shape_and_dtype = HostAllocationKey.shape_and_dtype()
  result_shape = (num_barriers, *shape_and_dtype.shape)
  result_shape_and_dtype = jax.ShapeDtypeStruct(
      result_shape, shape_and_dtype.dtype
  )
  return callback.io_callback(
      functools.partial(
          _allocate_barriers,
          source_info=source_info,
          axes_dims=axes_dims,
      ),
      (TOKEN_SHAPE_DTYPE, result_shape_and_dtype),
      token=token,
      device_id=device_id,
      grid_point_coords=grid_point_coords,
      thread_id=thread_id,
      num_arrivals=num_arrivals,
      num_barriers=num_barriers,
      ref_count=ref_count,
  )


def _deallocate_barrier(
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    allocation_key_as_array: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
):
  # TODO(paulbib): Add race-check validation on deallocation, i.e.: make sure
  # there are no outstanding async copies from or to the deallocated buffer.
  device_id_as_int = int(device_id)
  grid_point_coords_as_tuple = tuple(int(x) for x in grid_point_coords)
  thread_id_as_int = int(thread_id)
  del device_id, grid_point_coords, thread_id

  assert len(allocation_key_as_array.shape) == 2
  num_barriers = allocation_key_as_array.shape[0]

  keys_to_deallocate = []
  for i in range(num_barriers):
    keys_to_deallocate.append(allocation_key_as_array[i, :])

  shared_memory = _get_shared_memory()

  for key in keys_to_deallocate:
    barrier_allocation_key = HostAllocationKey.from_array(key)
    shared_memory.deallocate_barrier(
        barrier_allocation_key,
        logging_info=interpret_utils.GPULoggingInfo(
            device_id=device_id_as_int,
            grid_point_coords=grid_point_coords_as_tuple,
            thread_id=thread_id_as_int,
            source_info=source_info,
        ),
    )
  return token


def call_deallocate_barrier(
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    allocation_key_as_array: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
):
  return callback.io_callback(
      functools.partial(_deallocate_barrier, source_info=source_info),
      TOKEN_SHAPE_DTYPE,
      token,
      device_id,
      grid_point_coords,
      thread_id,
      allocation_key_as_array,
  )


def _barrier_wait(
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    allocation_key_as_array: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
):
  device_id_as_int = int(device_id)
  grid_point_coords_as_tuple = tuple(int(x) for x in grid_point_coords)
  thread_id_as_int = int(thread_id)
  barrier_key = HostAllocationKey.from_array(allocation_key_as_array)
  del device_id, thread_id, allocation_key_as_array

  shared_memory = _get_shared_memory()

  barrier, _ = shared_memory.get_barrier_and_increment_clock(
      barrier_key, device_id_as_int, thread_id_as_int
  )
  barrier.wait(
      device_id_as_int,
      thread_id_as_int,
      logging_info=interpret_utils.GPULoggingInfo(
          device_id=device_id_as_int,
          grid_point_coords=grid_point_coords_as_tuple,
          thread_id=thread_id_as_int,
          source_info=source_info,
      ),
  )
  return token


def call_barrier_wait(
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    allocation_key_as_array: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
):
  return callback.io_callback(
      functools.partial(_barrier_wait, source_info=source_info),
      TOKEN_SHAPE_DTYPE,
      token,
      device_id,
      grid_point_coords,
      thread_id,
      allocation_key_as_array,
  )


def _barrier_arrive(
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    allocation_key_as_array: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
):
  device_id_as_int = int(device_id)
  grid_point_coords_as_tuple = tuple(int(x) for x in grid_point_coords)
  thread_id_as_int = int(thread_id)
  barrier_key = HostAllocationKey.from_array(allocation_key_as_array)
  del device_id, grid_point_coords, thread_id, allocation_key_as_array

  shared_memory = _get_shared_memory()

  barrier, clock = shared_memory.get_barrier_and_increment_clock(
      barrier_key, device_id_as_int, thread_id_as_int
  )
  smem_commit_clock = shared_memory.get_smem_commit_clock(
      shared_memory.get_global_thread_id(device_id_as_int, thread_id_as_int)
  )
  barrier.arrive(
      clock,
      smem_commit_clock,
      logging_info=interpret_utils.GPULoggingInfo(
          device_id=device_id_as_int,
          grid_point_coords=grid_point_coords_as_tuple,
          thread_id=thread_id_as_int,
          source_info=source_info,
      ),
  )
  return token


def call_barrier_arrive(
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    allocation_key_as_array: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
):
  return callback.io_callback(
      functools.partial(_barrier_arrive, source_info=source_info),
      TOKEN_SHAPE_DTYPE,
      token,
      device_id,
      grid_point_coords,
      thread_id,
      allocation_key_as_array,
  )


def _assert_no_barriers_allocated(token):
  _get_shared_memory().assert_no_barriers_allocated()
  return token


def call_assert_no_barriers_allocated(token):
  return callback.io_callback(
      _assert_no_barriers_allocated, TOKEN_SHAPE_DTYPE, token
  )


class AsyncCopyTask:
  """
  An async task representing a TMA memory copy.

  Logically, this function is not running on any main thread but on a special
  ephemeral TMA thread, so the implementation and callbacks should not touch
  any main thread's VC.

  The implementation and callbacks are not re-entrant and not idempotent, so they
  must only be called once. There are no dynamic safety checks to verify this.
  """

  # The device on which the memory transfer is being executed.
  device_id: int

  grid_point_coords: jax.Array

  # The thread that initiated the memory transfer.
  thread_id: int

  # The pseudo-thread being used to execute the memory transfer.
  tma_thread_id: int

  # Allocation key and transforms for the source buffer.
  src_allocation_key: HostAllocationKey
  src_transforms: tuple[Any, ...]

  # Allocation key and transforms for the destination buffer.
  dst_allocation_key: HostAllocationKey
  dst_transforms: tuple[Any, ...]

  source_info: source_info_util.SourceInfo | None = None

  logging_info: interpret_utils.GPULoggingInfo | None = None

  data: np.ndarray | None = None

  def __init__(self,
      device_id: int,
      grid_point_coords: jax.Array,
      thread_id: int,
      src_allocation_key: HostAllocationKey,
      src_transforms: tuple[Any, ...],
      dst_allocation_key: HostAllocationKey,
      dst_transforms: tuple[Any, ...],
      source_info: source_info_util.SourceInfo | None = None,
  ):
    self.device_id = device_id
    self.grid_point_coords = grid_point_coords
    self.thread_id = thread_id
    self.src_allocation_key = src_allocation_key
    self.src_transforms = src_transforms
    self.dst_allocation_key = dst_allocation_key
    self.dst_transforms = dst_transforms
    self.source_info = source_info
    self.logging_info = interpret_utils.GPULoggingInfo(
        device_id=device_id,
        grid_point_coords=tuple(int(x) for x in grid_point_coords)
        if grid_point_coords is not None
        else (),
        thread_id=thread_id,
        source_info=source_info,
    )

  def __call__(self, tma_thread_id: int):
    shared_memory = _get_shared_memory()
    global_thread_id = shared_memory.get_global_thread_id(
        self.device_id, self.thread_id
    )

    self.pre_read(tma_thread_id, shared_memory)

    val, _, _ = shared_memory.get_buffer_content(
        self.src_allocation_key, interpret_utils.to_range(self.src_transforms),
        global_thread_id, logging_info=self.logging_info)
    assert val is not None

    self.post_read(tma_thread_id, shared_memory)

    shared_memory.store_buffer_content(
        self.dst_allocation_key, interpret_utils.to_range(self.dst_transforms),
        val,
        global_thread_id, logging_info=self.logging_info)

    self.post_write(tma_thread_id, shared_memory)

  def pre_read(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory):
    pass

  def post_read(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory):
    pass

  def post_write(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory):
    pass


class AsyncCopyGmemToSmemTask(AsyncCopyTask):
  """An async task representing a GMEM -> SMEM TMA memory copy."""

  barrier: memory.Barrier
  clock: vc.VectorClock | None = None
  smem_commit_clock: vc.VectorClock | None = None

  def __init__(
      self,
        device_id: int,
        grid_point_coords: jax.Array,
        thread_id: int,
        src_allocation_key: HostAllocationKey,
        src_transforms: tuple[Any, ...],
        dst_allocation_key: HostAllocationKey,
        dst_transforms: tuple[Any, ...],
        barrier_allocation_key: HostAllocationKey,
        source_info: source_info_util.SourceInfo | None,
        clock: vc.VectorClock | None = None,
        smem_commit_clock: vc.VectorClock | None = None):
    super().__init__(
        device_id=device_id,
        grid_point_coords=grid_point_coords,
        thread_id=thread_id,
        src_allocation_key=src_allocation_key,
        src_transforms=src_transforms,
        dst_allocation_key=dst_allocation_key,
        dst_transforms=dst_transforms,
        source_info=source_info)
    shared_memory = _get_shared_memory()
    self.barrier = shared_memory.get_barrier(barrier_allocation_key)
    self.clock = clock
    self.smem_commit_clock = smem_commit_clock

  def pre_read(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory):
    # TODO(paulbib): GMEM updates are only visible to the async proxy (TMA)
    # after a device-level proxy fence. However, no such functionality
    # is exposed in Pallas. When it is, we should use a `commit_gmem` clock here
    if shared_memory.detect_races:
      assert self.clock is not None
      assert self.smem_commit_clock is not None
      vc.inc_vector_clock(self.clock, tma_thread_id)
      get_races().check_read(
          self.device_id,
          self.thread_id,
          vc.copy_vector_clock(self.clock),
          self.src_allocation_key,
          interpret_utils.to_range(self.src_transforms),
          source_info=self.source_info,
      )

  def post_read(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory):
    if shared_memory.detect_races:
      assert self.clock is not None
      assert self.smem_commit_clock is not None
      vc.inc_vector_clock(self.clock, tma_thread_id)
      vc.inc_vector_clock(self.smem_commit_clock, tma_thread_id)

      get_races().check_write(
          self.device_id,
          self.thread_id,
          vc.copy_vector_clock(self.smem_commit_clock),
          self.dst_allocation_key,
          interpret_utils.to_range(self.dst_transforms),
          source_info=self.source_info,
      )

  def post_write(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory):
    self.barrier.arrive(
        clock=vc.copy_vector_clock(self.clock),
        smem_commit_clock=vc.copy_vector_clock(self.smem_commit_clock),
        logging_info=self.logging_info,
    )


class AsyncCopySmemToGmemTask(AsyncCopyTask):
  """An async task representing a SMEM -> GMEM TMA memory copy."""

  clock: vc.VectorClock | None = None
  smem_commit_clock: vc.VectorClock | None = None
  read_clock: vc.VectorClock | None = None
  write_clock: vc.VectorClock | None = None

  def __init__(
      self,
      device_id: int,
      grid_point_coords: jax.Array,
      thread_id: int,
      src_allocation_key: HostAllocationKey,
      src_transforms: tuple[Any, ...],
      dst_allocation_key: HostAllocationKey,
      dst_transforms: tuple[Any, ...],
      source_info: source_info_util.SourceInfo | None,
      clock: vc.VectorClock | None = None,
      smem_commit_clock: vc.VectorClock | None = None,
  ):
    super().__init__(
        device_id=device_id,
        grid_point_coords=grid_point_coords,
        thread_id=thread_id,
        src_allocation_key=src_allocation_key,
        src_transforms=src_transforms,
        dst_allocation_key=dst_allocation_key,
        dst_transforms=dst_transforms,
        source_info=source_info)
    self.clock = clock
    self.smem_commit_clock = smem_commit_clock

  def pre_read(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory):
    if shared_memory.detect_races:
      assert self.clock is not None
      assert self.smem_commit_clock is not None
      vc.inc_vector_clock(self.clock, tma_thread_id)
      vc.inc_vector_clock(self.smem_commit_clock, tma_thread_id)
      self.read_clock = vc.copy_vector_clock(self.clock)
      get_races().check_read(
          self.device_id,
          self.thread_id,
          self.smem_commit_clock,
          self.src_allocation_key,
          interpret_utils.to_range(self.src_transforms),
          source_info=self.source_info,
      )

  def post_read(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory):
    if shared_memory.detect_races:
      assert self.clock is not None
      assert self.smem_commit_clock is not None
      vc.inc_vector_clock(self.clock, tma_thread_id)
      self.write_clock = vc.copy_vector_clock(self.clock)
      get_races().check_write(
          self.device_id,
          self.thread_id,
          self.smem_commit_clock,
          self.dst_allocation_key,
          interpret_utils.to_range(self.dst_transforms),
          source_info=self.source_info,
      )

  def post_write(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory):
    if shared_memory.detect_races:
      assert self.read_clock is not None
      assert self.write_clock is not None
      shared_memory.add_copy_smem_to_gmem_clocks(
          shared_memory.get_global_thread_id(self.device_id, self.thread_id),
          self.read_clock,
          self.write_clock,
      )


NOOP_TRANSFORMS = (
    mosaic_gpu_core.UnswizzleRef,
    mosaic_gpu_core.UntilingTransform,
)


def _remove_noop_transforms(transforms: tuple[Any, ...]) -> tuple[Any, ...]:
  # TODO(jburnim): Instead of just filtering out these transforms, should we
  # check that every access of a buffer uses untiling and/or unswizzling
  # transforms that match how the buffer was allocated?
  return tuple(itertools.dropwhile(lambda t: isinstance(t, NOOP_TRANSFORMS),
                                   transforms))


def wgmma(
    *,
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    acc_allocation_key_as_array: jax.Array,
    acc_transforms: tuple[Any, ...],
    acc_dtype: jnp.dtype,
    a_allocation_key_as_array: jax.Array,
    a_transforms: tuple[Any, ...],
    b_allocation_key_as_array: jax.Array,
    b_transforms: tuple[Any, ...],
    source_info: source_info_util.SourceInfo | None = None,
):
  # TODO(jburnim): Vector clocks.
  # TODO(jburnim): Async wgmma.

  device_id: int = int(device_id)  # pyrefly: ignore[redefinition]
  grid_point_coords: tuple[int, ...] = tuple(int(x) for x in grid_point_coords)  # pyrefly: ignore[redefinition]
  thread_id: int = int(thread_id)  # pyrefly: ignore[redefinition]
  acc_allocation_key = HostAllocationKey.from_array(acc_allocation_key_as_array)
  a_allocation_key = HostAllocationKey.from_array(a_allocation_key_as_array)
  b_allocation_key = HostAllocationKey.from_array(b_allocation_key_as_array)
  a_transforms = jax.tree.map(int, _remove_noop_transforms(a_transforms))
  b_transforms = jax.tree.map(int, _remove_noop_transforms(b_transforms))
  acc_transforms = jax.tree.map(int, _remove_noop_transforms(acc_transforms))

  shared_memory = _get_shared_memory()
  global_thread_id = shared_memory.get_global_thread_id(device_id, thread_id)

  logging_info = interpret_utils.GPULoggingInfo(
      device_id=device_id,
      grid_point_coords=grid_point_coords,
      thread_id=thread_id,
      source_info=source_info,
  )

  a, _, _ = shared_memory.get_buffer_content(
      a_allocation_key, interpret_utils.to_range(a_transforms),
      global_thread_id, logging_info=logging_info)
  b, _, _ = shared_memory.get_buffer_content(
      b_allocation_key, interpret_utils.to_range(b_transforms),
      global_thread_id, logging_info=logging_info)
  assert a is not None
  assert b is not None
  acc_range = interpret_utils.to_range(acc_transforms)
  acc, _, _ = shared_memory.get_buffer_content(
      acc_allocation_key, acc_range, global_thread_id,
      logging_info=logging_info)

  res = acc + np.matmul(a, b, dtype=acc_dtype)

  shared_memory.store_buffer_content(
      acc_allocation_key, acc_range, res,
      global_thread_id, logging_info=logging_info)

  return token


def wgmma_accumulator_deref(
    *,
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    acc_allocation_key_as_array: jax.Array,
    wait_n: int | None,
    source_info: source_info_util.SourceInfo | None = None,
):
  # TODO(jburnim): wait_n for async wgmma.
  del wait_n

  device_id: int = int(device_id)  # pyrefly: ignore[redefinition]
  grid_point_coords: tuple[int, ...] = tuple(int(x) for x in grid_point_coords)  # pyrefly: ignore[redefinition]
  thread_id: int = int(thread_id)  # pyrefly: ignore[redefinition]
  acc_allocation_key = HostAllocationKey.from_array(acc_allocation_key_as_array)

  shared_memory = _get_shared_memory()
  global_thread_id = shared_memory.get_global_thread_id(device_id, thread_id)

  logging_info = interpret_utils.GPULoggingInfo(
      device_id=device_id,
      grid_point_coords=grid_point_coords,
      thread_id=0,
      source_info=source_info,
  )

  acc, _, _ = shared_memory.get_buffer_content(
      acc_allocation_key, (), global_thread_id, logging_info=logging_info)
  return token, acc


def copy_smem_to_gmem(
    *,
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    src_allocation_key_as_array: jax.Array,
    src_transforms: tuple[Any, ...],
    dst_allocation_key_as_array: jax.Array,
    dst_transforms: tuple[Any, ...],
    predicate: jax.Array | None,
    source_info: source_info_util.SourceInfo,
    commit_group: bool,
    reduction_op: mgpu.TMAReductionOp,
):
  # TODO(jburnim,paulbib): Implement commit_group.
  del commit_group
  device_id: int = int(device_id)  # pyrefly: ignore[redefinition]
  thread_id: int = int(thread_id)  # pyrefly: ignore[redefinition]
  src_allocation_key = HostAllocationKey.from_array(src_allocation_key_as_array)
  src_transforms = jax.tree.map(int, _remove_noop_transforms(src_transforms))
  dst_allocation_key = HostAllocationKey.from_array(dst_allocation_key_as_array)
  dst_transforms = jax.tree.map(int, _remove_noop_transforms(dst_transforms))

  if predicate is not None:
    raise NotImplementedError("predicate not supported")
  if reduction_op is not None:
    raise NotImplementedError("reduction_op not supported")

  clock = None
  smem_commit_clock = None

  shared_memory = _get_shared_memory()
  if shared_memory.detect_races:
    clock = shared_memory.incr_clock(
        shared_memory.get_global_thread_id(device_id, thread_id)
    )
    smem_commit_clock = shared_memory.get_smem_commit_clock(
        shared_memory.get_global_thread_id(device_id, thread_id)
    )

  task = AsyncCopySmemToGmemTask(
      device_id=device_id,
      grid_point_coords=grid_point_coords,
      thread_id=thread_id,
      src_allocation_key=src_allocation_key,
      src_transforms=src_transforms,
      dst_allocation_key=dst_allocation_key,
      dst_transforms=dst_transforms,
      source_info=source_info,
      clock=clock,
      smem_commit_clock=smem_commit_clock,
  )

  shared_memory = _get_shared_memory()
  shared_memory.execute_async_task(task, device_id, thread_id)

  return token


def wait_smem_to_gmem(
    *,
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    n: int,
    wait_read_only: bool,
    source_info: source_info_util.SourceInfo | None = None,
):
  del grid_point_coords, source_info
  device_id_as_int = int(device_id)
  thread_id_as_int = int(thread_id)
  shared_memory = _get_shared_memory()
  global_thread_id = shared_memory.get_global_thread_id(
      device_id_as_int, thread_id_as_int
  )
  shared_memory.wait_smem_to_gmem(
      global_thread_id, n, wait_read_only)
  return token


def copy_gmem_to_smem(
    *,
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    src_allocation_key_as_array: jax.Array,
    src_transforms: tuple[Any, ...],
    dst_allocation_key_as_array: jax.Array,
    dst_transforms: tuple[Any, ...],
    barrier_allocation_key_as_array: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
):
  device_id_as_int = int(device_id)
  thread_id_as_int = int(thread_id)
  src_transforms = jax.tree.map(int, _remove_noop_transforms(src_transforms))
  src_allocation_key = HostAllocationKey.from_array(src_allocation_key_as_array)
  dst_transforms = jax.tree.map(int, _remove_noop_transforms(dst_transforms))
  dst_allocation_key = HostAllocationKey.from_array(dst_allocation_key_as_array)
  barrier_allocation_key = HostAllocationKey.from_array(
      barrier_allocation_key_as_array
  )
  del device_id, thread_id

  clock = None
  smem_commit_clock = None

  shared_memory = _get_shared_memory()
  if shared_memory.detect_races:
    clock = shared_memory.incr_clock(
        shared_memory.get_global_thread_id(device_id_as_int, thread_id_as_int)
    )
    smem_commit_clock = shared_memory.get_smem_commit_clock(
        shared_memory.get_global_thread_id(device_id_as_int, thread_id_as_int)
    )

  transfer = AsyncCopyGmemToSmemTask(
      device_id=device_id_as_int,
      grid_point_coords=grid_point_coords,
      thread_id=thread_id_as_int,
      src_allocation_key=src_allocation_key,
      src_transforms=src_transforms,
      dst_allocation_key=dst_allocation_key,
      dst_transforms=dst_transforms,
      barrier_allocation_key=barrier_allocation_key,
      source_info=source_info,
      clock=clock,
      smem_commit_clock=smem_commit_clock,
  )

  shared_memory = _get_shared_memory()
  shared_memory.execute_async_task(transfer, device_id_as_int, thread_id_as_int)

  return token


def commit_smem(
    *,
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
):
  del grid_point_coords, source_info
  device_id_as_int = int(device_id)
  thread_id_as_int = int(thread_id)
  shared_memory = _get_shared_memory()
  global_thread_id = shared_memory.get_global_thread_id(
      device_id_as_int, thread_id_as_int
  )
  shared_memory.update_smem_commit_clock(global_thread_id)

  return token


def tcgen05_mma(
    *,
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    acc_allocation_key_as_array: jax.Array,
    acc_transforms: tuple[Any, ...],
    acc_dtype: jnp.dtype,
    a_allocation_key_as_array: jax.Array,
    a_transforms: tuple[Any, ...],
    b_allocation_key_as_array: jax.Array,
    b_transforms: tuple[Any, ...],
    accumulate: jax.Array,
    barrier_allocation_key_as_array: jax.Array | None = None,
    a_scale_allocation_key_as_array: jax.Array | None = None,
    a_scale_transforms: tuple[Any, ...] | None = None,
    b_scale_allocation_key_as_array: jax.Array | None = None,
    b_scale_transforms: tuple[Any, ...] | None = None,
    a_sparse_metadata_allocation_key_as_array: jax.Array | None = None,
    a_sparse_metadata_transforms: tuple[Any, ...] | None = None,
    source_info: source_info_util.SourceInfo | None = None,
):
  # TODO(jburnim): Support scales and sparse metadata.
  assert a_scale_allocation_key_as_array is None
  assert b_scale_allocation_key_as_array is None
  assert a_sparse_metadata_allocation_key_as_array is None
  del a_scale_transforms, b_scale_transforms, a_sparse_metadata_transforms

  device_id_as_int = int(device_id)
  grid_point_coords_as_tuple = tuple(int(x) for x in grid_point_coords)
  thread_id_as_int = int(thread_id)
  acc_allocation_key = HostAllocationKey.from_array(acc_allocation_key_as_array)
  a_allocation_key = HostAllocationKey.from_array(a_allocation_key_as_array)
  b_allocation_key = HostAllocationKey.from_array(b_allocation_key_as_array)
  acc_transforms = jax.tree.map(int, _remove_noop_transforms(acc_transforms))
  a_transforms = jax.tree.map(int, _remove_noop_transforms(a_transforms))
  b_transforms = jax.tree.map(int, _remove_noop_transforms(b_transforms))
  accumulate: bool = bool(accumulate)  # pyrefly: ignore[redefinition]

  shared_memory = _get_shared_memory()
  global_thread_id = shared_memory.get_global_thread_id(
      device_id_as_int, thread_id_as_int
  )

  logging_info = interpret_utils.GPULoggingInfo(
      device_id=device_id_as_int,
      grid_point_coords=grid_point_coords_as_tuple,
      thread_id=thread_id_as_int,
      source_info=source_info,
  )

  a, _, _ = shared_memory.get_buffer_content(
      a_allocation_key,
      interpret_utils.to_range(a_transforms),
      global_thread_id,
      logging_info=logging_info,
  )
  b, _, _ = shared_memory.get_buffer_content(
      b_allocation_key,
      interpret_utils.to_range(b_transforms),
      global_thread_id,
      logging_info=logging_info,
  )
  assert a is not None
  assert b is not None

  acc_range = interpret_utils.to_range(acc_transforms)

  if accumulate:
    acc, _, _ = shared_memory.get_buffer_content(
        acc_allocation_key,
        acc_range,
        global_thread_id,
        logging_info=logging_info,
    )
    assert acc is not None
    res = acc + np.matmul(a, b, dtype=acc_dtype)
  else:
    res = np.matmul(a, b, dtype=acc_dtype)

  shared_memory.store_buffer_content(
      acc_allocation_key,
      acc_range,
      res,
      global_thread_id,
      logging_info=logging_info,
  )

  if barrier_allocation_key_as_array is not None:
    barrier_key = HostAllocationKey.from_array(barrier_allocation_key_as_array)
    barrier, clock = shared_memory.get_barrier_and_increment_clock(
        barrier_key, device_id_as_int, thread_id_as_int)
    barrier.arrive(
        clock,
        logging_info=logging_info)

  return token


def async_load_tmem(
    *,
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
    src_allocation_key_as_array: jax.Array,
    src_transforms: tuple[Any, ...],
    source_info: source_info_util.SourceInfo | None = None,
):
  device_id_as_int = int(device_id)
  grid_point_coords_as_tuple = tuple(int(x) for x in grid_point_coords)
  thread_id_as_int = int(thread_id)
  src_allocation_key = HostAllocationKey.from_array(src_allocation_key_as_array)
  src_transforms = jax.tree.map(int, _remove_noop_transforms(src_transforms))

  shared_memory = _get_shared_memory()
  global_thread_id = shared_memory.get_global_thread_id(
      device_id_as_int, thread_id_as_int
  )

  logging_info = interpret_utils.GPULoggingInfo(
      device_id=device_id_as_int,
      grid_point_coords=grid_point_coords_as_tuple,
      thread_id=thread_id_as_int,
      source_info=source_info,
  )

  val, _, _ = shared_memory.get_buffer_content(
      src_allocation_key,
      interpret_utils.to_range(src_transforms),
      global_thread_id,
      logging_info=logging_info,
  )

  return token, val


def kernel_thread_finished(
    *,
    token: jax.Array,
    device_id: jax.Array,
    grid_point_coords: jax.Array,
    thread_id: jax.Array,
):
  del grid_point_coords
  device_id: int = int(device_id) # pyrefly: ignore[redefinition]
  thread_id: int = int(thread_id) # pyrefly: ignore[redefinition]
  shared_memory = _get_shared_memory()
  shared_memory.kernel_thread_finished(device_id, thread_id)
  return token
