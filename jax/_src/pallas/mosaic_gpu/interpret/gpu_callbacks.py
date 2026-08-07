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

from collections.abc import Sequence
import contextlib
import dataclasses
import functools
import itertools
import threading
from typing import Any

import jax
from jax import numpy as jnp
from jax._src import callback
from jax._src import source_info_util
from jax._src.pallas.mosaic.interpret import utils as interpret_utils
from jax._src.pallas.mosaic.interpret.race_detection_state import RaceDetectionState
from jax._src.pallas.mosaic_gpu import core as mosaic_gpu_core
from jax._src.pallas.mosaic_gpu.interpret import shared_memory as memory
from jax._src.pallas.mosaic_gpu.interpret.params import InterpretGPUParams
from jax._src.pallas.mosaic_gpu.interpret.shared_memory import HostAllocationKey
from jax._src.pallas.mosaic_gpu.interpret.shared_memory import HostAllocationRequest
from jax._src.state import indexing
from jax._src.state import types as state_types
from jax.experimental.mosaic import gpu as mgpu
import numpy as np


def is_gmem_memory_space(space: mosaic_gpu_core.MemorySpace | None) -> bool:
  return space == mosaic_gpu_core.MemorySpace.GMEM


_shared_memory: memory.GPUSharedMemory | None = None
_shared_memory_init_lock = threading.Lock()
_races: RaceDetectionState[memory.GPUSharedMemory.ThreadKey] | None = None


def _get_shared_memory() -> memory.GPUSharedMemory:
  assert _shared_memory is not None
  return _shared_memory


def _clear_shared_memory():
  global _shared_memory
  with _shared_memory_init_lock:
    _shared_memory = None


def get_races() -> RaceDetectionState[memory.GPUSharedMemory.ThreadKey]:
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


def _update_clocks_for_device_barrier(token, device: memory.Device):
  shared_memory = _get_shared_memory()
  shared_memory.update_clocks_for_device_barrier(device)
  return token


def call_update_clocks_for_device_barrier(token, device: memory.Device):
  return callback.io_callback(
      _update_clocks_for_device_barrier,
      TOKEN_SHAPE_DTYPE,
      token,
      device,
  )


def _make_allocation_request_array(
    *,
    token: jax.Array,
    compute_unit: memory.Thread | memory.Device,
    memory_space_id: int,
    initial_ref_count: int = 1,
) -> tuple[jax.Array, np.ndarray]:
  thread_id, block_id = (
      # TODO(paulbib): Support warp specialization
      (compute_unit.warpgroup_id, compute_unit.block_id)
      if isinstance(compute_unit, memory.Thread)
      else (0, 0)
  )
  return token, HostAllocationRequest(
      memory_space_id=memory_space_id,
      device_id=compute_unit.device_id,
      block_id=block_id,
      thread_id=thread_id,
      initial_ref_count=initial_ref_count,
  ).as_np_array


def call_make_allocation_request_array(
    *,
    token: jax.Array,
    compute_unit: memory.Thread | memory.Device,
    memory_space_id: int,
    initial_ref_count: int = 1,
) -> tuple[jax.Array, jax.Array]:
  return callback.io_callback(
      _make_allocation_request_array,
      (TOKEN_SHAPE_DTYPE, HostAllocationRequest.shape_and_dtype()),
      token=token,
      compute_unit=compute_unit,
      memory_space_id=memory_space_id,
      initial_ref_count=initial_ref_count,
      # The callback has no side-effect, so we allow this to be reordered
      # relative to other callbacks.
      ordered=False,
  )


def _allocate_buffer_for_all_threads(
    token: jax.Array,
    mesh_location: memory.MeshLocation | None,
    device: memory.Device,
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
      `HostAllocationRequest` with `thread_id`/`block_id` set to zero.
    value: Array of values to initialize the allocated buffer with.

  Returns:
    `AllocationKey` to refer to the allocated buffer.

  Raises:
    ValueError: If `thread_id`/`block_id` in `allocation_request` is not zero.
  """
  allocation_request = HostAllocationRequest.from_array(
      allocation_request_as_array
  )
  del allocation_request_as_array

  if allocation_request.thread_id != 0:
    raise ValueError(
        "`thread_id` must be zero when allocating a buffer for all threads"
    )
  if allocation_request.block_id != 0:
    raise ValueError(
        "`block_id` must be zero when allocating a buffer for all threads"
    )
  assert allocation_request.memory_space_id != memory.get_memory_space_idx(
      mosaic_gpu_core.MemorySpace.REGS
  )

  shared_memory = _get_shared_memory()

  key: HostAllocationKey | None = None
  buffer_id: int | None = None
  for thread in shared_memory.concurrent_threads(device):
    buffer_id_for_thread_id = shared_memory.get_next_buffer_id(thread)
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
        block_id=0,
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
        logging_info=memory.GPULoggingInfo(mesh_location, thread, source_info),
    )

  # We expect the `for`-loop above to have executed its body at least once.
  assert key is not None
  return token, key.as_np_array


def call_allocate_buffer_for_all_threads(
    token: jax.Array,
    mesh_location: memory.MeshLocation | None,
    device: memory.Device,
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
      mesh_location,
      device,
      allocation_request_as_array,
      value,
  )


def _allocate_buffer(
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
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
  allocation_request = HostAllocationRequest.from_array(
      allocation_request_as_array
  )
  del allocation_request_as_array

  shared_memory = _get_shared_memory()

  if allocation_request.memory_space_id == memory.get_memory_space_idx(
      mosaic_gpu_core.MemorySpace.REGS
  ):
    # For barrier and buffer identifiers to line up across threads, we rely on
    # each thread making the same sequence of allocations.  But threads are
    # permitted to make different REGS allocations, so we use a different
    # sequence of integer identifiers for REGS allocations.
    buffer_id = shared_memory.get_next_wgmma_accumulator_id(thread)
  else:
    buffer_id = shared_memory.get_next_buffer_id(thread)

  key = HostAllocationKey(
      memory_space_id=allocation_request.memory_space_id,
      device_id=allocation_request.device_id,
      block_id=allocation_request.block_id,
      thread_id=allocation_request.thread_id,
      initial_ref_count=allocation_request.initial_ref_count,
      buffer_id=buffer_id,
  )
  ref_count = allocation_request.initial_ref_count
  shared_memory.allocate_buffer(
      key,
      ref_count=ref_count,
      value=np.array(value),
      logging_info=memory.GPULoggingInfo(mesh_location, thread, source_info,),
  )
  return token, key.as_np_array


def call_allocate_buffer(
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    allocation_request_as_array: jax.Array,
    value: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
) -> tuple[jax.Array, jax.Array]:
  return callback.io_callback(
      functools.partial(_allocate_buffer, source_info=source_info),
      (TOKEN_SHAPE_DTYPE, HostAllocationKey.shape_and_dtype()),
      token,
      mesh_location,
      thread,
      allocation_request_as_array,
      value,
  )


def _deallocate_buffer(
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    allocation_key_as_array: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
):
  """Decreases the reference count of the buffer with `allocation_key` (Deallocates the buffer if its reference count becomes zero)."""
  allocation_key = HostAllocationKey.from_array(allocation_key_as_array)
  del allocation_key_as_array
  shared_memory = _get_shared_memory()

  shared_memory.deallocate_buffer(
      allocation_key,
      logging_info=memory.GPULoggingInfo(mesh_location, thread, source_info),
  )
  return token


def call_deallocate_buffer(
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    allocation_key_as_array: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
):
  return callback.io_callback(
      functools.partial(_deallocate_buffer, source_info=source_info),
      TOKEN_SHAPE_DTYPE,
      token,
      mesh_location,
      thread,
      allocation_key_as_array,
  )


def _window_shape(
    rnge: tuple[slice | int, ...], shape: Sequence[int]
) -> tuple[int, ...]:
  """Returns the shape of the window `rnge` selects from an array of `shape`.

  A range may be shorter than the array's rank, in which case the trailing
  dimensions are taken in full. Integer entries index a dimension away, so
  they do not contribute; slice entries contribute their length. The result
  describes the window that was *asked for*, which for a TMA copy may extend
  past the end of the array.
  """
  window = []
  for dim, idx in itertools.zip_longest(shape, rnge, fillvalue=None):
    if idx is None:  # Dimension not indexed, so taken in full.
      window.append(dim)
    elif isinstance(idx, (int, np.integer)):
      continue
    else:
      start = 0 if idx.start is None else int(idx.start)
      step = 1 if idx.step is None else int(idx.step)
      stop = dim if idx.stop is None else int(idx.stop)
      window.append((stop - start + step - 1) // step)
  return tuple(window)


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


# Transforms `to_range` can ignore, because they change neither which elements
# an access touches nor the order it yields them in:
# * layout transforms, since a buffer is stored in its logical shape
# * reshapes of a ref's leading dimensions, since indexes are written against logical shape
DROPPED_TRANSFORMS = (
    mosaic_gpu_core.UnswizzleRef,
    mosaic_gpu_core.SwizzleTransform,
    mosaic_gpu_core.UntilingTransform,
    mosaic_gpu_core.TilingTransform,
    mosaic_gpu_core.CollapseLeadingBatchDimensionsTransform,
    mosaic_gpu_core.ExpandLeadingBatchDimensionsTransform,
)


def _is_dynamic(indexer: indexing.NDIndexer) -> bool:
  return any(
      isinstance(idx, indexing.Slice)
      and (idx.is_dynamic_start or idx.is_dynamic_size)
      for idx in indexer.indices
  )


def _validate_transforms(transforms: tuple[Any, ...]) -> tuple[Any, ...]:
  """Validates that `transforms` is handleable by interpret mode and
  filters/modifies transforms to be later useable with _apply_transforms."""
  # TODO(jburnim): Instead of just filtering out these transforms, should we
  # check that every access of a buffer uses untiling and/or unswizzling
  # transforms that match how the buffer was allocated?
  out = []
  transpose_seen = None
  for t in transforms:
    if isinstance(t, indexing.NDIndexer):
      if _is_dynamic(t):
        raise ValueError("Dynamic indexing not supported in GPU interpret mode")
      if transpose_seen:
        # `to_range` computes the accessed range in the ref's own coordinates,
        # so an index applied after `transpose_ref` addresses different
        # elements than it appears to. Refuse rather than compute it.
        raise NotImplementedError(
            "GPU interpret mode does not support indexing a ref after"
            f" transposing it. Transforms: {transforms}"
        )
      out.append(t)
    elif isinstance(t, state_types.TransposeTransform):
      transpose_seen = True
      out.append(t)
    elif isinstance(t, DROPPED_TRANSFORMS):
      pass
    else:
      raise ValueError(f"Unsupported transform: {t}")
  return tuple(out)


def _apply_transforms(
    transforms: tuple[Any, ...],
    value: np.ndarray,
    *,
    invert: bool = False,
) -> np.ndarray:
  """Applies or undoes a sequence of transforms."""
  for t in reversed(transforms) if invert else transforms:
    if isinstance(t, indexing.NDIndexer):
      pass
    elif isinstance(t, state_types.TransposeTransform):
      perm = np.argsort(t.permutation) if invert else t.permutation
      value = np.transpose(value, perm)
    elif isinstance(t, DROPPED_TRANSFORMS):
      pass
    else:
      raise ValueError(f"Unsupported transform: {t}")
  return value


def _get(
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread | None,
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
  allocation_key = HostAllocationKey.from_array(allocation_key_as_array)
  del allocation_key_as_array

  _validate_transforms(transforms)
  transforms = _validate_transforms(transforms)
  #transforms = jax.tree.map(int, transforms)

  if input_name is not None:
    # NOTE: input_name, block_indices, and grid_loop_idx are set only if this
    # function is being called to read a block from a pallas_call input (at the
    # start of one iteration of the kernel body).
    assert block_indices is not None
    block_indices = tuple(int(x) for x in block_indices)
    assert grid_loop_idx is not None
    grid_loop_idx = tuple(int(x) for x in grid_loop_idx)

  shared_memory = _get_shared_memory()

  read_range = interpret_utils.to_range(transforms)
  ret, (shape, dtype), clock_ = shared_memory.get_buffer_content(
      allocation_key,
      read_range,
      thread,
      increment_clock=increment_clock,
      logging_info=memory.GPULoggingInfo(mesh_location, thread, source_info),
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

  ret = _apply_transforms(ret, transforms)

  if shared_memory.detect_races and thread is not None:
    assert clock is not None
    get_races().check_read(
        thread,
        clock.generic_clock,
        allocation_key,
        read_range,
        source_info=source_info,
    )
  return token, ret


def call_get(
    *,
    token: jax.Array,
    result_shape_and_dtype,
    mesh_location: memory.MeshLocation | None,
    thread: memory.Thread | None,
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
      mesh_location,
      thread,
      allocation_key_as_array,
      transforms,
      block_indices,
      grid_loop_idx,
      clock,
  )

def _swap(
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
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
  allocation_key = HostAllocationKey.from_array(allocation_key_as_array)
  del allocation_key_as_array

  _validate_transforms(transforms)
  transforms = _validate_transforms(transforms)
  # transforms = jax.tree.map(int, transforms)

  val_arr = _apply_transforms(np.array(val), transforms, invert=True)
  mask_arr = None

  if mask is not None:
    assert mask.shape == val.shape
    mask_arr = _apply_transforms(np.array(mask), transforms, invert=True)

  shared_memory = _get_shared_memory()

  read_write_range = interpret_utils.to_range(transforms)
  ret, (shape, _), clock_ = shared_memory.swap_buffer_content(
      allocation_key,
      read_write_range,
      val_arr,
      mask_arr,
      thread,
      increment_clock=increment_clock,
      logging_info=memory.GPULoggingInfo(mesh_location, thread, source_info),
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

  ret = _apply_transforms(ret, transforms)

  if shared_memory.detect_races:
    assert clock is not None
    get_races().check_write(
        thread,
        clock.generic_clock,
        allocation_key,
        read_write_range,
        source_info=source_info,
    )
  return token, ret


def call_swap(
    *,
    token: jax.Array,
    result_shape_and_dtype,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
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
      mesh_location,
      thread,
      allocation_key_as_array,
      transforms,
      val,
      mask,
      clock=clock,
  )


def _allocate_barriers(
    *,
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    num_arrivals: jax.Array,
    orders_tensor_core: jax.Array,
    flat_num_barriers: jax.Array,
    ref_count: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
) -> tuple[jax.Array, np.ndarray]:
  num_arrivals_as_int = int(num_arrivals)
  orders_tensor_core_as_bool = bool(orders_tensor_core)
  flat_num_barriers_as_int = int(flat_num_barriers)
  ref_count_as_int = int(ref_count)
  del num_arrivals, flat_num_barriers, ref_count

  shared_memory = _get_shared_memory()

  keys = []
  for _ in range(flat_num_barriers_as_int):
    # Advance `shared_memory`'s internal buffer id counter for all threads that
    # call into this function.
    barrier_id = shared_memory.get_next_buffer_id(thread)
    smem_space_id = memory.IDX_BY_GPU_MEMORY_SPACE[mosaic_gpu_core.SMEM]

    # Barriers are shared between threads. For each group of threads that share
    # a barrier, we compute the thread ID to be used for the allocation key.
    # Invariant: `thread_id` is the same for all threads in a group that
    # shares the barrier.
    key = HostAllocationKey(
        memory_space_id=smem_space_id,
        device_id=thread.device_id,
        block_id=thread.block_id,
        thread_id=0,
        initial_ref_count=ref_count_as_int,
        buffer_id=barrier_id,
    )

    shared_memory.allocate_barrier(
        key,
        ref_count=ref_count_as_int,
        num_arrivals=num_arrivals_as_int,
        orders_tensor_core=orders_tensor_core_as_bool,
        logging_info=memory.GPULoggingInfo(mesh_location, thread, source_info),
    )
    keys.append(key.as_np_array)

  assert len(keys) == flat_num_barriers_as_int
  return token, np.array(keys, dtype=np.int32)


def call_allocate_barriers(
    *,
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    num_arrivals: jax.Array,
    orders_tensor_core: bool,
    flat_num_barriers: int | jax.Array,
    ref_count: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
) -> tuple[jax.Array, jax.Array]:
  shape_and_dtype = HostAllocationKey.shape_and_dtype()
  result_shape = (flat_num_barriers, *shape_and_dtype.shape)
  result_shape_and_dtype = jax.ShapeDtypeStruct(
      result_shape, shape_and_dtype.dtype
  )
  return callback.io_callback(
      functools.partial(
          _allocate_barriers,
          source_info=source_info,
      ),
      (TOKEN_SHAPE_DTYPE, result_shape_and_dtype),
      token=token,
      mesh_location=mesh_location,
      thread=thread,
      num_arrivals=num_arrivals,
      orders_tensor_core=orders_tensor_core,
      flat_num_barriers=flat_num_barriers,
      ref_count=ref_count,
  )


def _deallocate_barrier(
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    allocation_key_as_array: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
    cluster_barrier: bool = False,
):
  # TODO(paulbib): Add race-check validation on deallocation, i.e.: make sure
  # there are no outstanding async copies from or to the deallocated buffer.

  flat_allocation_keys = np.reshape(
      allocation_key_as_array, (-1, *HostAllocationKey.shape_and_dtype().shape)
  )
  num_barriers = flat_allocation_keys.shape[0]

  keys_to_deallocate = []
  for i in range(num_barriers):
    keys_to_deallocate.append(flat_allocation_keys[i, :])

  shared_memory = _get_shared_memory()

  # TODO(nrink): If we had a dedicated memory space for cluster barriers
  # (and/or for 'normal' barriers too), we could select the deallocation
  # function based on the memory space of the allocation key (instead of needing
  # the `deallocate_cluster_barrier` argument).
  deallocate_fn = (
      shared_memory.deallocate_cluster_barrier
      if cluster_barrier
      else shared_memory.deallocate_barrier
  )

  for key in keys_to_deallocate:
    barrier_allocation_key = HostAllocationKey.from_array(key)
    deallocate_fn(
        barrier_allocation_key,
        logging_info=memory.GPULoggingInfo(mesh_location, thread, source_info),
    )
  return token


def call_deallocate_barrier(
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    allocation_key_as_array: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
    cluster_barrier: bool = False,
):
  return callback.io_callback(
      functools.partial(_deallocate_barrier, source_info=source_info),
      TOKEN_SHAPE_DTYPE,
      token,
      mesh_location,
      thread,
      allocation_key_as_array,
      cluster_barrier=cluster_barrier,
  )


def _barrier_wait(
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    allocation_key_as_array: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
):
  barrier_key = HostAllocationKey.from_array(allocation_key_as_array)
  del allocation_key_as_array

  shared_memory = _get_shared_memory()

  barrier, _ = shared_memory.get_barrier_and_increment_clock(
      barrier_key, thread,
  )
  barrier.wait(
      thread,
      logging_info=memory.GPULoggingInfo(mesh_location, thread, source_info),
  )
  return token


# Note that this callback is also used for waiting on cluster barriers.
def call_barrier_wait(
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    allocation_key_as_array: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
):
  return callback.io_callback(
      functools.partial(_barrier_wait, source_info=source_info),
      TOKEN_SHAPE_DTYPE,
      token,
      mesh_location,
      thread,
      allocation_key_as_array,
  )


def _barrier_arrive(
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    allocation_key_as_array: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
):
  barrier_key = HostAllocationKey.from_array(allocation_key_as_array)
  del allocation_key_as_array

  shared_memory = _get_shared_memory()

  barrier, clock = shared_memory.get_barrier_and_increment_clock(
      barrier_key, thread
  )
  if isinstance(barrier, memory.ClusterBarrier):
    barrier.arrive(
        mesh_location=mesh_location,
        thread=thread,
        clock=clock,
        logging_info=memory.GPULoggingInfo(mesh_location, thread, source_info),
    )
  elif isinstance(barrier, memory.Barrier):
    barrier.arrive(
        thread,
        clock,
        memory.GPULoggingInfo(mesh_location, thread, source_info),
    )
  else:
    raise ValueError(f"Unsupported barrier type: {type(barrier)}")
  return token


# Note that this callback is also used for arriving at cluster barriers.
def call_barrier_arrive(
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    allocation_key_as_array: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
):
  return callback.io_callback(
      functools.partial(_barrier_arrive, source_info=source_info),
      TOKEN_SHAPE_DTYPE,
      token,
      mesh_location,
      thread,
      allocation_key_as_array,
  )


def _assert_no_barriers_allocated(token):
  _get_shared_memory().assert_no_barriers_allocated()
  return token


def call_assert_no_barriers_allocated(token):
  return callback.io_callback(
      _assert_no_barriers_allocated, TOKEN_SHAPE_DTYPE, token
  )


def _allocate_cluster_barriers(
    *,
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    axes_dims: tuple[int, ...],
    is_axis_collective: tuple[bool, ...],
    num_arrivals: jax.Array,
    flat_num_barriers: jax.Array,
    ref_count: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
) -> tuple[jax.Array, np.ndarray]:
  num_arrivals_as_int = int(num_arrivals)
  flat_num_barriers_as_int = int(flat_num_barriers)
  ref_count_as_int = int(ref_count)
  del num_arrivals, flat_num_barriers, ref_count

  shared_memory = _get_shared_memory()

  keys = []
  for _ in range(flat_num_barriers_as_int):
    # Advance `shared_memory`'s internal buffer id counter for all threads that
    # call into this function.
    barrier_id = shared_memory.get_next_buffer_id(thread)

    # Use `SMEM` for the cluster barrier's allocation key below. Note that on
    # real GPUs, there is not a single cluster barrier object that resides in
    # SMEM (or any other memory space). We nonetheless use `SMEM` here to
    # indicate that the (per thread-block) `Barrier`s that the `ClusterBarrier`
    # is composed of are each allocated in `SMEM` (on a real GPU).
    smem_space_id = memory.IDX_BY_GPU_MEMORY_SPACE[mosaic_gpu_core.SMEM]

    # Cluster barriers are shared between all threads in a cluster. Hence use 0
    # for the thread/block ID in the allocation `key` below.
    key = HostAllocationKey(
        memory_space_id=smem_space_id,
        device_id=thread.device_id,
        block_id=0,
        thread_id=0,
        initial_ref_count=ref_count_as_int,
        buffer_id=barrier_id,
    )

    shared_memory.allocate_cluster_barrier(
        key,
        axes_dims=axes_dims,
        is_axis_collective=is_axis_collective,
        ref_count=ref_count_as_int,
        num_arrivals=num_arrivals_as_int,
        logging_info=memory.GPULoggingInfo(mesh_location, thread, source_info),
    )
    keys.append(key.as_np_array)

  assert len(keys) == flat_num_barriers_as_int
  return token, np.array(keys, dtype=np.int32)


def call_allocate_cluster_barriers(
    *,
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    axes_dims: tuple[int, ...],
    is_axis_collective: tuple[bool, ...],
    num_arrivals: jax.Array,
    flat_num_barriers: int | jax.Array,
    ref_count: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
) -> tuple[jax.Array, jax.Array]:
  shape_and_dtype = HostAllocationKey.shape_and_dtype()
  result_shape = (flat_num_barriers, *shape_and_dtype.shape)
  result_shape_and_dtype = jax.ShapeDtypeStruct(
      result_shape, shape_and_dtype.dtype
  )
  return callback.io_callback(
      functools.partial(
          _allocate_cluster_barriers,
          source_info=source_info,
          axes_dims=axes_dims,
          is_axis_collective=is_axis_collective,
      ),
      (TOKEN_SHAPE_DTYPE, result_shape_and_dtype),
      token=token,
      mesh_location=mesh_location,
      thread=thread,
      num_arrivals=num_arrivals,
      flat_num_barriers=flat_num_barriers,
      ref_count=ref_count,
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

  # The logical and physical location of the thread that initiated the task
  mesh_location: memory.MeshLocation
  thread: memory.Thread

  # The pseudo-thread being used to execute the memory transfer.
  tma_thread_id: int

  # Allocation key and transforms for the source buffer.
  src_allocation_key: HostAllocationKey
  src_transforms: tuple[Any, ...]

  # Allocation key and transforms for the destination buffer.
  dst_allocation_key: HostAllocationKey
  dst_transforms: tuple[Any, ...]

  source_info: source_info_util.SourceInfo | None = None

  logging_info: memory.GPULoggingInfo | None = None

  data: np.ndarray | None = None

  def __init__(self,
      mesh_location: memory.MeshLocation,
      thread: memory.Thread,
      src_allocation_key: HostAllocationKey,
      src_transforms: tuple[Any, ...],
      dst_allocation_key: HostAllocationKey,
      dst_transforms: tuple[Any, ...],
      source_info: source_info_util.SourceInfo | None = None,
  ):
    self.mesh_location = mesh_location
    self.thread = thread
    self.src_allocation_key = src_allocation_key
    self.src_transforms = src_transforms
    self.dst_allocation_key = dst_allocation_key
    self.dst_transforms = dst_transforms
    self.source_info = source_info
    self.logging_info = memory.GPULoggingInfo(
        mesh_location, thread, source_info
    )

  def __call__(self, tma_thread_id: int):
    shared_memory = _get_shared_memory()

    self.pre_read(tma_thread_id, shared_memory)

    src_range = interpret_utils.to_range(self.src_transforms)
    val, (src_shape, src_dtype), _ = shared_memory.get_buffer_content(
        self.src_allocation_key,
        src_range,
        self.thread,
        logging_info=self.logging_info,
    )
    val = self.fill_out_of_bounds(
        val, src_range, src_shape, src_dtype, shared_memory
    )
    assert val is not None

    self.post_read(tma_thread_id, shared_memory)

    shared_memory.store_buffer_content(
        self.dst_allocation_key,
        interpret_utils.to_range(self.dst_transforms),
        val,
        self.thread,
        logging_info=self.logging_info,
    )

    self.post_write(tma_thread_id, shared_memory)

  def fill_out_of_bounds(
      self,
      val: np.ndarray | None,
      src_range: tuple[slice | int, ...],
      src_shape: Sequence[int],
      src_dtype: np.dtype,
      shared_memory: memory.GPUSharedMemory,
  ) -> np.ndarray | None:
    """Pads a copy that reads past the end of its source."""
    del src_range, src_shape, src_dtype, shared_memory
    return val

  def pre_read(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory):
    pass

  def post_read(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory):
    pass

  def post_write(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory):
    pass


class AsyncCopyGmemToSmemTask(AsyncCopyTask):
  """An async task representing a GMEM -> SMEM TMA memory copy."""

  VectorClock = memory.GPUSharedMemory.VectorClock

  barrier: memory.Barrier
  clock: VectorClock | None = None
  oob_mode: mgpu.OOBFillMode | None = None

  def __init__(
      self,
      mesh_location: memory.MeshLocation,
      thread: memory.Thread,
      src_allocation_key: HostAllocationKey,
      src_transforms: tuple[Any, ...],
      dst_allocation_key: HostAllocationKey,
      dst_transforms: tuple[Any, ...],
      barrier_allocation_key: HostAllocationKey,
      source_info: source_info_util.SourceInfo | None,
      clock: VectorClock | None = None,
      oob_mode: mgpu.OOBFillMode | None = None,
  ):
    super().__init__(
        mesh_location=mesh_location,
        thread=thread,
        src_allocation_key=src_allocation_key,
        src_transforms=src_transforms,
        dst_allocation_key=dst_allocation_key,
        dst_transforms=dst_transforms,
        source_info=source_info)
    shared_memory = _get_shared_memory()
    self.barrier = shared_memory.get_barrier(barrier_allocation_key)
    self.clock = clock
    self.oob_mode = oob_mode

  def fill_out_of_bounds(
      self, val, src_range, src_shape, src_dtype, shared_memory
  ):
    """Fills the part of the window that lies past the end of the GMEM array."""
    # Note that this is allowed behavior for TMA copies and the user specifies
    # what they want the extra data padded with. This is separate from interpret
    # mode filling uninitialized memory with a fixed value to detect bugs.
    window_shape = _window_shape(src_range, src_shape)
    if val is not None and val.shape == window_shape:
      return val

    # match default behavior of plgpu.copy_gmem_to_smem
    oob_mode = (
        mgpu.OOBFillMode.ZEROS if self.oob_mode is None else self.oob_mode
    )
    if oob_mode == mgpu.OOBFillMode.PROMISE_IN_BOUNDS:
      raise IndexError(
          f"Out-of-bounds copy from {self.src_allocation_key}: reading"
          f" [{src_range}] of a buffer with shape {tuple(src_shape)}, but the"
          " copy promised to stay in bounds"
          " (`oob_mode=OOBFillMode.PROMISE_IN_BOUNDS`)."
      )
    if oob_mode == mgpu.OOBFillMode.ZEROS:
      fill = np.zeros(window_shape, dtype=src_dtype)
    else:
      # `OOBFillMode.UNDEFINED`: the hardware leaves these elements
      # unspecified, so use the preconfigured interpret mode uninit memory value
      fill = np.full(
          window_shape,
          interpret_utils.get_uninitialized_value(
              src_dtype, shared_memory.uninitialized_memory
          ),
          dtype=src_dtype,
      )
    if val is not None:
      fill[tuple(slice(s) for s in val.shape)] = val
    return fill

  def pre_read(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory):
    # TODO(paulbib): GMEM updates are only visible to the async proxy (TMA)
    # after a device-level proxy fence. However, no such functionality
    # is exposed in Pallas. When it is, we should use a `commit_gmem` clock here
    if shared_memory.detect_races:
      assert self.clock is not None
      self.clock.inc(tma_thread_id)
      get_races().check_read(
          self.thread,
          self.clock.generic_clock.copy(),
          self.src_allocation_key,
          interpret_utils.to_range(self.src_transforms),
          source_info=self.source_info,
      )

  def post_read(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory):
    if shared_memory.detect_races:
      assert self.clock is not None
      self.clock.inc(tma_thread_id)

      get_races().check_write(
          self.thread,
          self.clock.async_smem_clock.copy(),
          self.dst_allocation_key,
          interpret_utils.to_range(self.dst_transforms),
          source_info=self.source_info,
      )

  def post_write(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory):
    self.barrier.arrive(
        self.thread,
        self.clock.copy() if self.clock is not None else None,
        self.logging_info,
    )


class AsyncCopySmemToGmemTask(AsyncCopyTask):
  """An async task representing a SMEM -> GMEM TMA memory copy."""

  VectorClock = memory.GPUSharedMemory.VectorClock

  clock: VectorClock | None = None
  read_clock: VectorClock | None = None
  write_clock: VectorClock | None = None

  def __init__(
      self,
      mesh_location: memory.MeshLocation,
      thread: memory.Thread,
      src_allocation_key: HostAllocationKey,
      src_transforms: tuple[Any, ...],
      dst_allocation_key: HostAllocationKey,
      dst_transforms: tuple[Any, ...],
      source_info: source_info_util.SourceInfo | None,
      clock: VectorClock | None = None,
  ):
    super().__init__(
        mesh_location=mesh_location,
        thread=thread,
        src_allocation_key=src_allocation_key,
        src_transforms=src_transforms,
        dst_allocation_key=dst_allocation_key,
        dst_transforms=dst_transforms,
        source_info=source_info)
    self.clock = clock

  def pre_read(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory):
    if shared_memory.detect_races:
      assert self.clock is not None
      self.clock.inc(tma_thread_id)
      self.read_clock = self.clock.copy()
      get_races().check_read(
          self.thread,
          self.clock.async_smem_clock.copy(),
          self.src_allocation_key,
          interpret_utils.to_range(self.src_transforms),
          source_info=self.source_info,
      )

  def post_read(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory):
    if shared_memory.detect_races:
      assert self.clock is not None
      self.clock.inc(tma_thread_id)
      self.write_clock = self.clock.copy()
      get_races().check_write(
          self.thread,
          self.clock.async_smem_clock.copy(),
          self.dst_allocation_key,
          interpret_utils.to_range(self.dst_transforms),
          source_info=self.source_info,
      )

  def post_write(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory):
    if shared_memory.detect_races:
      assert self.read_clock is not None
      assert self.write_clock is not None
      shared_memory.add_copy_smem_to_gmem_clocks(
          self.thread,
          self.read_clock,
          self.write_clock,
      )


def wgmma(
    *,
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
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

  acc_allocation_key = HostAllocationKey.from_array(acc_allocation_key_as_array)
  a_allocation_key = HostAllocationKey.from_array(a_allocation_key_as_array)
  b_allocation_key = HostAllocationKey.from_array(b_allocation_key_as_array)
  a_transforms = _validate_transforms(a_transforms)
  b_transforms = _validate_transforms(b_transforms)
  acc_transforms = _validate_transforms(acc_transforms)

  shared_memory = _get_shared_memory()

  logging_info = memory.GPULoggingInfo(mesh_location, thread, source_info)
  a, _, _ = shared_memory.get_buffer_content(
      a_allocation_key,
      interpret_utils.to_range(a_transforms),
      thread,
      logging_info=logging_info,
  )
  b, _, _ = shared_memory.get_buffer_content(
      b_allocation_key,
      interpret_utils.to_range(b_transforms),
      thread,
      logging_info=logging_info,
  )
  assert a is not None
  assert b is not None
  acc_range = interpret_utils.to_range(acc_transforms)
  acc, _, _ = shared_memory.get_buffer_content(
      acc_allocation_key,
      acc_range,
      thread,
      logging_info=logging_info,
  )

  res = acc + np.matmul(a, b, dtype=acc_dtype)

  shared_memory.store_buffer_content(
      acc_allocation_key,
      acc_range,
      res,
      thread,
      logging_info=logging_info,
  )

  return token


def wgmma_accumulator_deref(
    *,
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    acc_allocation_key_as_array: jax.Array,
    wait_n: int | None,
    source_info: source_info_util.SourceInfo | None = None,
):
  # TODO(jburnim): wait_n for async wgmma.
  del wait_n

  acc_allocation_key = HostAllocationKey.from_array(acc_allocation_key_as_array)

  shared_memory = _get_shared_memory()

  logging_info = memory.GPULoggingInfo(mesh_location, thread, source_info)
  acc, _, _ = shared_memory.get_buffer_content(
      acc_allocation_key, (), thread, logging_info=logging_info
  )
  return token, acc


def copy_smem_to_gmem(
    *,
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Warpgroup,
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
  src_allocation_key = HostAllocationKey.from_array(src_allocation_key_as_array)
  src_transforms = _validate_transforms(src_transforms)
  dst_allocation_key = HostAllocationKey.from_array(dst_allocation_key_as_array)
  dst_transforms = _validate_transforms(dst_transforms)

  if predicate is not None:
    raise NotImplementedError("predicate not supported")
  if reduction_op is not None:
    raise NotImplementedError("reduction_op not supported")

  clock = None

  shared_memory = _get_shared_memory()
  if shared_memory.detect_races:
    clock = shared_memory.incr_clock(thread)

  task = AsyncCopySmemToGmemTask(
      mesh_location=mesh_location,
      thread=thread,
      src_allocation_key=src_allocation_key,
      src_transforms=src_transforms,
      dst_allocation_key=dst_allocation_key,
      dst_transforms=dst_transforms,
      source_info=source_info,
      clock=clock,
  )

  shared_memory = _get_shared_memory()
  shared_memory.execute_async_task(task)

  return token


def wait_smem_to_gmem(
    *,
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    n: int,
    wait_read_only: bool,
    source_info: source_info_util.SourceInfo | None = None,
):
  del source_info, mesh_location
  shared_memory = _get_shared_memory()
  shared_memory.wait_smem_to_gmem(thread, n, wait_read_only)
  return token


def copy_gmem_to_smem(
    *,
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Warpgroup,
    src_allocation_key_as_array: jax.Array,
    src_transforms: tuple[Any, ...],
    dst_allocation_key_as_array: jax.Array,
    dst_transforms: tuple[Any, ...],
    barrier_allocation_key_as_array: jax.Array,
    source_info: source_info_util.SourceInfo | None = None,
    oob_mode: mgpu.OOBFillMode | None = None,
):
  src_transforms = _validate_transforms(src_transforms)
  src_allocation_key = HostAllocationKey.from_array(src_allocation_key_as_array)
  dst_transforms = _validate_transforms(dst_transforms)
  dst_allocation_key = HostAllocationKey.from_array(dst_allocation_key_as_array)
  barrier_allocation_key = HostAllocationKey.from_array(
      barrier_allocation_key_as_array
  )

  clock = None

  shared_memory = _get_shared_memory()
  if shared_memory.detect_races:
    clock = shared_memory.incr_clock(thread)

  transfer = AsyncCopyGmemToSmemTask(
      mesh_location=mesh_location,
      thread=thread,
      src_allocation_key=src_allocation_key,
      src_transforms=src_transforms,
      dst_allocation_key=dst_allocation_key,
      dst_transforms=dst_transforms,
      barrier_allocation_key=barrier_allocation_key,
      source_info=source_info,
      clock=clock,
      oob_mode=oob_mode,
  )

  shared_memory = _get_shared_memory()
  shared_memory.execute_async_task(transfer)

  return token


def commit_smem(
    *,
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    source_info: source_info_util.SourceInfo | None = None,
):
  del mesh_location, source_info
  shared_memory = _get_shared_memory()
  shared_memory.commit_smem(thread)

  return token


@dataclasses.dataclass(frozen=True, kw_only=True)
class TcGen05Mma(memory.PipelineableAsyncTask):
  mesh_location: memory.MeshLocation
  thread: memory.Thread
  acc_key: HostAllocationKey
  acc_transforms: tuple[Any, ...]
  acc_dtype: jnp.dtype
  a_key: HostAllocationKey
  a_transforms: tuple[Any, ...]
  b_key: HostAllocationKey
  b_transforms: tuple[Any, ...]
  accumulate: bool
  barrier_key: HostAllocationKey | None = None
  a_scale_key: HostAllocationKey | None = None
  a_scale_transforms: tuple[Any, ...] | None = None
  b_scale_key: HostAllocationKey | None = None
  b_scale_transforms: tuple[Any, ...] | None = None
  a_sparse_metadata_key: HostAllocationKey | None = None
  a_sparse_metadata_transforms: tuple[Any, ...] | None = None
  collective_axis: str | None = None
  source_info: source_info_util.SourceInfo | None = None

  def forms_pipeline(self, parent: memory.PipelineableAsyncTask) -> bool:
    if isinstance(parent, TcGen05Mma):
      # In order to pipeline with another mma, the accumulator, collective block
      # count, and dtype must match
      return (
          parent.acc_key == self.acc_key
          and interpret_utils.to_range(parent.acc_transforms)
          == interpret_utils.to_range(self.acc_transforms)
          and parent.acc_dtype == self.acc_dtype
          and parent.collective_axis == self.collective_axis
      )
    if isinstance(parent, TcGen05Copy):
      return parent.collective_axis == self.collective_axis
    return False

  def __call__(
      self,
      pipeline_clock: memory.GpuClockBundle | None,
      tma_thread_id: int,
  ) -> memory.GpuClockBundle | None:
    # TODO(paulbib): Support scales and sparse metadata.
    assert self.a_scale_key is None
    assert self.b_scale_key is None
    assert self.a_sparse_metadata_key is None
    assert self.collective_axis is None, "collective_axis not supported yet"

    shared_memory = _get_shared_memory()

    logging_info = memory.GPULoggingInfo(
        self.mesh_location, self.thread, self.source_info
    )
    a, _, _ = shared_memory.get_buffer_content(
        self.a_key,
        interpret_utils.to_range(self.a_transforms),
        self.thread,
        logging_info=logging_info,
    )
    b, _, _ = shared_memory.get_buffer_content(
        self.b_key,
        interpret_utils.to_range(self.b_transforms),
        self.thread,
        logging_info=logging_info,
    )
    assert a is not None
    assert b is not None

    clock = None
    if shared_memory.detect_races:
      initiating_clock = shared_memory.get_clock(self.thread)
      assert initiating_clock is not None
      if pipeline_clock is not None:
        initiating_clock.update(pipeline_clock)
      initiating_clock.inc(tma_thread_id)
      clock = initiating_clock

      # a can be in either SMEM or TMEM, but b will always be in SMEM.
      a_clock = (
          clock.async_smem_clock
          if self.a_key.memory_space_id
          == memory.IDX_BY_GPU_MEMORY_SPACE[mosaic_gpu_core.MemorySpace.SMEM]
          else clock.generic_clock
      )

      get_races().check_read(
          self.thread,
          a_clock,
          self.a_key,
          interpret_utils.to_range(self.a_transforms),
          source_info=self.source_info,
      )
      get_races().check_read(
          self.thread,
          clock.async_smem_clock,
          self.b_key,
          interpret_utils.to_range(self.b_transforms),
          source_info=self.source_info,
      )

    acc_range = interpret_utils.to_range(self.acc_transforms)

    if self.accumulate:
      acc, _, _ = shared_memory.get_buffer_content(
          self.acc_key,
          acc_range,
          None,
          logging_info=logging_info,
      )
      assert acc is not None
      res = acc + np.matmul(a, b, dtype=self.acc_dtype)
    else:
      res = np.matmul(a, b, dtype=self.acc_dtype)

    shared_memory.store_buffer_content(
        self.acc_key,
        acc_range,
        res,
        self.thread,
        increment_clock=False,
        logging_info=logging_info,
    )

    if shared_memory.detect_races:
      assert clock is not None
      get_races().check_write(
          self.thread,
          clock.generic_clock,
          self.acc_key,
          acc_range,
          source_info=self.source_info,
      )

    if self.barrier_key:
      barrier = shared_memory.get_barrier(self.barrier_key)
      if not isinstance(barrier, memory.Barrier):
        raise ValueError("tcgen05_mma only allows arriving on a Barrier")
      if not barrier.orders_tensor_core:
        raise ValueError(
            "tcgen05_mma only allows arriving on a Barrier that orders tensor"
            " core"
        )
      barrier.arrive(
          thread=self.thread,
          clock=clock,
          logging_info=logging_info,
      )

    return clock.copy() if clock is not None else None


def tcgen05_mma(
    *,
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
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
    collective_axis: str | None = None,
    source_info: source_info_util.SourceInfo | None = None,
):

  acc_allocation_key = HostAllocationKey.from_array(acc_allocation_key_as_array)
  a_allocation_key = HostAllocationKey.from_array(a_allocation_key_as_array)
  b_allocation_key = HostAllocationKey.from_array(b_allocation_key_as_array)
  def _maybe_key(array: jax.Array | None):
    return HostAllocationKey.from_array(array) if array is not None else None

  a_scale_allocation_key = _maybe_key(a_scale_allocation_key_as_array)
  b_scale_allocation_key = _maybe_key(b_scale_allocation_key_as_array)
  a_sparse_metadata_allocation_key = _maybe_key(
      a_sparse_metadata_allocation_key_as_array
  )

  acc_transforms = _validate_transforms(acc_transforms)
  a_transforms = _validate_transforms(a_transforms)
  b_transforms = _validate_transforms(b_transforms)
  if a_scale_transforms is not None:
    a_scale_transforms = _validate_transforms(a_scale_transforms)
  if b_scale_transforms is not None:
    b_scale_transforms = _validate_transforms(b_scale_transforms)
  if a_sparse_metadata_transforms is not None:
    a_sparse_metadata_transforms = _validate_transforms(a_sparse_metadata_transforms)

  accumulate: bool = bool(accumulate)  # pyrefly: ignore[redefinition]

  barrier_key = _maybe_key(barrier_allocation_key_as_array)

  shared_memory = _get_shared_memory()
  if shared_memory.detect_races:
    shared_memory.incr_clock(thread)

  shared_memory.execute_pipelineable_async_task(
      TcGen05Mma(
          mesh_location=mesh_location,
          thread=thread,
          acc_key=acc_allocation_key,
          acc_transforms=acc_transforms,
          acc_dtype=acc_dtype,
          a_key=a_allocation_key,
          a_transforms=a_transforms,
          b_key=b_allocation_key,
          b_transforms=b_transforms,
          accumulate=accumulate,
          barrier_key=barrier_key,
          a_scale_key=a_scale_allocation_key,
          a_scale_transforms=a_scale_transforms,
          b_scale_key=b_scale_allocation_key,
          b_scale_transforms=b_scale_transforms,
          a_sparse_metadata_key=a_sparse_metadata_allocation_key,
          a_sparse_metadata_transforms=a_sparse_metadata_transforms,
          collective_axis=collective_axis,
          source_info=source_info,
      ),
      thread,
  )

  return token


@dataclasses.dataclass(frozen=True, kw_only=True)
class TcGen05Copy(memory.PipelineableAsyncTask):
  mesh_location: memory.MeshLocation
  thread: memory.Thread
  smem_key: HostAllocationKey
  smem_transforms: tuple[Any, ...]
  tmem_key: HostAllocationKey
  tmem_transforms: tuple[Any, ...]
  collective_axis: str | None = None
  source_info: source_info_util.SourceInfo | None = None

  def forms_pipeline(self, parent: memory.PipelineableAsyncTask) -> bool:
    return False

  def __call__(
      self,
      pipeline_clock: memory.GpuClockBundle | None,
      tma_thread_id: int,
  ) -> memory.GpuClockBundle | None:
    assert (
        self.collective_axis is None
    ), "Collective axis not supported for copy yet"

    shared_memory = _get_shared_memory()

    logging_info = memory.GPULoggingInfo(
        self.mesh_location, self.thread, self.source_info
    )
    smem, _, _ = shared_memory.get_buffer_content(
        self.smem_key,
        interpret_utils.to_range(self.smem_transforms),
        self.thread,
        logging_info=logging_info,
    )
    assert smem is not None

    clock = None
    if shared_memory.detect_races:
      initiating_clock = shared_memory.get_clock(self.thread)
      assert initiating_clock is not None
      if pipeline_clock is not None:
        initiating_clock.update(pipeline_clock)
      initiating_clock.inc(tma_thread_id)
      clock = initiating_clock

      get_races().check_read(
          self.thread,
          clock.async_smem_clock,
          self.smem_key,
          interpret_utils.to_range(self.smem_transforms),
          source_info=self.source_info,
      )

    tmem_range = interpret_utils.to_range(self.tmem_transforms)

    shared_memory.store_buffer_content(
        self.tmem_key,
        tmem_range,
        smem,
        self.thread,
        increment_clock=False,
        logging_info=logging_info,
    )

    if shared_memory.detect_races:
      assert clock is not None
      get_races().check_write(
          self.thread,
          clock.generic_clock,
          self.tmem_key,
          tmem_range,
          source_info=self.source_info,
      )

    return clock.copy() if clock is not None else None


def async_copy_smem_to_tmem(
    *,
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    smem_allocation_key_as_array: jax.Array,
    smem_transforms: tuple[Any, ...],
    tmem_allocation_key_as_array: jax.Array,
    tmem_transforms: tuple[Any, ...],
    collective_axis: str | None = None,
    source_info: source_info_util.SourceInfo | None = None,
):
  smem_allocation_key = HostAllocationKey.from_array(
      smem_allocation_key_as_array
  )
  tmem_allocation_key = HostAllocationKey.from_array(
      tmem_allocation_key_as_array
  )
  smem_transforms = _validate_transforms(smem_transforms)
  tmem_transforms = _validate_transforms(tmem_transforms)

  shared_memory = _get_shared_memory()
  if shared_memory.detect_races:
    shared_memory.incr_clock(thread)

  shared_memory.execute_pipelineable_async_task(
      TcGen05Copy(
          mesh_location=mesh_location,
          thread=thread,
          smem_key=smem_allocation_key,
          smem_transforms=smem_transforms,
          tmem_key=tmem_allocation_key,
          tmem_transforms=tmem_transforms,
          collective_axis=collective_axis,
          source_info=source_info,
      ),
      thread,
  )
  return token


def tcgen05_commit_arrive(
    *,
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    barrier_key_as_array: jax.Array,
    collective_axis: str | None = None,
    source_info: source_info_util.SourceInfo | None = None,
):
  # TODO(paulbib): Support collective_axis.
  del collective_axis
  barrier_key = HostAllocationKey.from_array(barrier_key_as_array)

  shared_memory = _get_shared_memory()
  if shared_memory.detect_races:
    shared_memory.incr_clock(thread)

  def f(tma_thread_id: int):
    shared_memory = _get_shared_memory()
    barrier = shared_memory.get_barrier(barrier_key)
    if not isinstance(barrier, memory.Barrier):
      raise ValueError(
          "tcgen05_commit_arrive only allows arriving on a Barrier"
      )
    if not barrier.orders_tensor_core:
      raise ValueError(
          "tcgen05_commit_arrive only allows arriving on a Barrier that orders"
          " tensor core"
      )

    clock = None
    if shared_memory.detect_races:
      clock = shared_memory.get_clock(thread)
      completions_clock = shared_memory.get_tcgen05_async_clock(thread)
      assert clock is not None
      if completions_clock is not None:
        clock.update(completions_clock)

    barrier.arrive(
        thread,
        clock,
        memory.GPULoggingInfo(mesh_location, thread, source_info),
    )

  _get_shared_memory().execute_async_task(f)
  return token


def async_store_tmem(
    *,
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    dst_allocation_key_as_array: jax.Array,
    dst_transforms: tuple[Any, ...],
    vals: np.ndarray,
    source_info: source_info_util.SourceInfo | None = None,
):
  dst_allocation_key = HostAllocationKey.from_array(dst_allocation_key_as_array)
  dst_transforms = _validate_transforms(dst_transforms)
  logging_info = memory.GPULoggingInfo(mesh_location, thread, source_info)

  def f(tma_thread_id: int):
    shared_memory = _get_shared_memory()
    shared_memory.store_buffer_content(
        dst_allocation_key,
        interpret_utils.to_range(dst_transforms),
        vals,
        thread,
        increment_clock=False,
        logging_info=logging_info,
    )
    if shared_memory.detect_races:
      clock = shared_memory.get_clock(thread)
      assert clock is not None
      clock.inc(tma_thread_id)
      get_races().check_write(
          thread,
          clock.generic_clock,
          dst_allocation_key,
          interpret_utils.to_range(dst_transforms),
          source_info=source_info,
      )
      shared_memory.add_store_tmem_clock(thread, clock)

  _get_shared_memory().execute_async_task(f)

  return token


def async_load_tmem(
    *,
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    src_allocation_key_as_array: jax.Array,
    src_transforms: tuple[Any, ...],
    source_info: source_info_util.SourceInfo | None = None,
):
  src_allocation_key = HostAllocationKey.from_array(src_allocation_key_as_array)
  src_transforms = _validate_transforms(src_transforms)
  logging_info = memory.GPULoggingInfo(mesh_location, thread, source_info)

  def f(tma_thread_id: int):
    shared_memory = _get_shared_memory()

    val, _, _ = shared_memory.get_buffer_content(
        src_allocation_key,
        interpret_utils.to_range(src_transforms),
        None,
        logging_info=logging_info,
    )

    if shared_memory.detect_races:
      clock = shared_memory.get_clock(thread)
      assert clock is not None
      clock.inc(tma_thread_id)
      get_races().check_read(
          thread,
          clock.generic_clock,
          src_allocation_key,
          interpret_utils.to_range(src_transforms),
          source_info=source_info,
      )
      shared_memory.add_load_tmem_clock(thread, clock)

    return val

  result = _get_shared_memory().execute_async_task(f)
  return token, result


def commit_tmem(
    *,
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    source_info: source_info_util.SourceInfo | None = None,
):
  del mesh_location, source_info
  shared_memory = _get_shared_memory()
  shared_memory.wait_tmem_stores(thread)
  return token


def wait_load_tmem(
    *,
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
    source_info: source_info_util.SourceInfo | None = None,
):
  del mesh_location, source_info
  shared_memory = _get_shared_memory()
  shared_memory.wait_tmem_loads(thread)
  return token


def sync_warps_with_warpgroup(
    *,
    token: jax.Array,
    warpgroup: memory.Warpgroup,
):
  """Updates the warpgroup's warps' clocks with the warpgroup's clock"""
  shared_memory = _get_shared_memory()
  if shared_memory.detect_races:
    for i in range(mosaic_gpu_core.WarpMesh._NUM_WARPS_PER_WARPGROUP):
      shared_memory.update_clock(warpgroup, warpgroup.warp(i))
  return token


def sync_warpgroup_with_warps(
    *,
    token: jax.Array,
    warpgroup: memory.Warpgroup,
):
  """Updates the warpgroup's clock with the warpgroup's warps' clocks."""
  shared_memory = _get_shared_memory()
  if shared_memory.detect_races:
    for i in range(mosaic_gpu_core.WarpMesh._NUM_WARPS_PER_WARPGROUP):
      shared_memory.update_clock(warpgroup.warp(i), warpgroup)
  return token


def kernel_thread_finished(
    *,
    token: jax.Array,
    mesh_location: memory.MeshLocation,
    thread: memory.Thread,
):
  del mesh_location
  shared_memory = _get_shared_memory()
  shared_memory.kernel_thread_finished(thread)
  return token


def cluster_finished(
    *,
    token: jax.Array,
):
  """Called when a cluster finishes execution of a kernel.

  Since clusters are executed sequentially in interpret mode, when this
  function is called no code is running and we can reset shared memory state
  if needed
  """
  shared_memory = _get_shared_memory()
  shared_memory.reset_per_cluster_state()
  return token
