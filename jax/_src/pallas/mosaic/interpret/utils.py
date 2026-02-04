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

from collections.abc import Sequence
import dataclasses
import enum
import math
import threading
from typing import Any, Literal

from jax import lax
from jax._src import core as jax_core
from jax._src.pallas import primitives
from jax._src.util import safe_map
import jax.numpy as jnp
import numpy as np


def get_uninitialized_value(
    dtype, uninitialized_memory: Literal["nan", "zero"]
):
  if uninitialized_memory == "nan":
    if jnp.issubdtype(dtype, jnp.floating):
      return np.nan
    elif jnp.issubdtype(dtype, jnp.integer):
      return jnp.iinfo(dtype).max
    elif jnp.issubdtype(dtype, jnp.bool):
      return True
  if uninitialized_memory == "zero":
    return 0
  raise NotImplementedError(uninitialized_memory + " + " + str(dtype))


@dataclasses.dataclass(frozen=True, kw_only=True)
class InterpretParams:
  """Parameters for kernel interpret mode.

  Interpret mode is a way to run Pallas kernels on CPU, while simulating TPU/GPU
  shared memory, communication, and synchronization operations.

  Attributes:
    detect_races: If True, a dynamic, happens-before race detector will be used
      to detect data races during kernel interpretation.  If any races are
      detected, a message will be printed and `races.races_found` will be set to
      True.
      Default: False.
    out_of_bounds_reads: If "raise", an exception will be raised on any
      out-of-bounds read of a buffer.  If "uninitialized_value", any parts of
      the read that are out-of-bounds will return the value used to fill
      uninitialized memory, which can be configured via the
      "uninitialized_memory".
      Default: "raise".
    skip_floating_point_ops: If True, operations that produce only floating
      point values will not be interpreted; instead, their results will be
      replaced with arrays all of `jnp.inf`. Additionally any floating point
      operands to any operation will be replaced with (arrays of) `jnp.inf`.
      Default: False.
    uninitialized_memory: If "nan", allocated buffers are initialized to contain
      all NaNs (or to their maximum possible value for integers). If "zero",
      allocated buffers are initialized to all zeros.
      Default: "nan".
    num_cores_or_threads: The number of cores per device (TPU) or threads per
      block (GPU). Note that for interpreting GPU kernels, we currently only
      support a single block in the grid. (So the number of threads per block on
      the GPU can be thought of as the number of threads that runs concurrently
      on the GPU.)
      Default: 1.
    vector_clock_size: The number of entries in the vector clocks. This should
      be an integer bigger then the total number of cores, i.e. bigger than
      `number of devices * num_cores_per_device`. If `None`, the vector clock
      size that is used in the interpreter will default to twice the total
      number of cores.
      Default: None.
  """

  detect_races: bool = False
  out_of_bounds_reads: Literal["raise", "uninitialized"] = "raise"
  skip_floating_point_ops: bool = False
  uninitialized_memory: Literal["nan", "zero"] = "nan"
  num_cores_or_threads: int = 1
  vector_clock_size: int | None = None

  def __post_init__(self):
    if self.num_cores_or_threads < 1:
      raise ValueError(
          "Number of cores or threads must be at least 1, but got"
          f" {self.num_cores_or_threads}."
      )

  def get_vector_clock_size(self, num_devices) -> int:
    """Returns the number of vector clocks to use.`"""
    num_cores_or_threads = num_devices * self.num_cores_or_threads
    if self.vector_clock_size is not None:
      if num_cores_or_threads >= self.vector_clock_size:
        raise ValueError(
            f"Vector clock size ({self.vector_clock_size}) must be greater than"
            f" the total number of cores/threads ({num_cores_or_threads})."
        )
      return self.vector_clock_size
    else:
      # Default to twice the total number of cores/threads.
      return 2 * num_cores_or_threads

  def get_uninitialized_array(self, shape, dtype):
    return jnp.full(
        shape,
        get_uninitialized_value(dtype, self.uninitialized_memory),
        dtype,
    )

  def pad_to_block_dimension(self, value, block_shape):
    """Pads values so the shape evenly divides into block dimensions.

    For example, if values has a shape of (33, 2, 5) with a block_shape of
    (32, 2, 4), this function will pad the value of shape to (64, 2, 8).

    Args:
      value: Array to be padded.
      block_shape: Block shapes to use for padding. If None, no padding will be
        performed.

    Returns:
      A padded array.
    """
    padded_shape = tuple(
        ((v - 1) // b + 1) * b for v, b in zip(value.shape, block_shape)
    )
    if padded_shape != value.shape:
      pad_width = tuple((0, a - b) for a, b in zip(padded_shape, value.shape))
      pad_value = self.get_uninitialized_array((), value.dtype)
      value = jnp.pad(value, pad_width, constant_values=pad_value)
    return value


class LoggingMode(enum.Flag):
  """Logging mode for GPU interpret mode.

  Attrs:
    BARRIER: Enable logging inside barrier object.
    SHARED_MEMORY: Enable logging in the shared memory object.
  """

  BARRIER = enum.auto()
  SHARED_MEMORY = enum.auto()


@dataclasses.dataclass(frozen=True, kw_only=True)
class InterpretGPUParams(InterpretParams):
  """Parameters for GPU interpret mode.

  GPU interpret mode is a way run Pallas GPU kernels on CPU, while simulating
  a GPU's shared memory spaces (GMEM, SMEM, etc.), threads and synchronization
  operations (e.g. barriers). This mode is intended for debugging and testing.

  To run a kernel under GPU interpret mode, pass an instance of
  ``InterpretParams`` as an argument for the ``interpret`` parameter of
  :func:`pallas_call`, :func:`core_map` or :func:`kernel`.

  NOTE: If an exception is raised while interpreting a kernel, you must call
  :func:`reset_gpu_interpret_mode_state` before using GPU interpret mode
  again in the same process.

  Attrs:
    logging_mode: Logging mode for GPU interpret mode.
  """

  logging_mode: LoggingMode | None = None


class Counter:
  """A simple counter that is thread-safe."""

  def __init__(self, initial_value: int):
    self.value = initial_value
    self.lock = threading.Lock()

  def get_next(self):
    with self.lock:
      result = self.value
      self.value += 1
    return result


# TODO(sharadmv): De-dup this w/ the impl in primitives.py.
def _device_id_dict_to_mesh(device_id_dict, axis_sizes, axis_indices):
  physical_axis_dict = {}
  axis_names = axis_sizes.keys()
  for axis, idx in device_id_dict.items():
    if isinstance(axis, tuple) and any(a in axis_names for a in axis):
      if not all(a in axis_names for a in axis):
        raise NotImplementedError(
            f"{axis} mixes JAX mesh and Pallas mesh grid axes"
        )
      axes_dimensions = [axis_sizes[name] for name in axis]
      for axis_index, axis_name in enumerate(axis):
        axis_size = axis_sizes[axis_name]
        inner_mesh_size = math.prod(axes_dimensions[axis_index + 1 :])
        minor_divisor = inner_mesh_size

        # Fast path for power of 2s
        if inner_mesh_size & (inner_mesh_size - 1) == 0:
          shift_len = (inner_mesh_size & -inner_mesh_size).bit_length() - 1
          partial_device_idx = idx >> shift_len
        else:
          partial_device_idx = idx // minor_divisor

        if axis_size & (axis_size - 1) == 0:
          device_idx = partial_device_idx & (axis_size - 1)
        else:
          device_idx = partial_device_idx % axis_size
        physical_axis_dict[axis_name] = device_idx
    else:
      physical_axis_dict[axis] = idx
  device_id = []
  for axis in axis_names:
    if axis in physical_axis_dict:
      device_id.append(physical_axis_dict[axis])
    else:
      device_id.append(axis_indices[axis])
  non_mesh_axes = {
      k: v for k, v in physical_axis_dict.items() if k not in axis_names
  }
  return tuple(device_id), non_mesh_axes


def device_coords_to_logical_id(device_coords, axis_sizes, axis_indices):
  if isinstance(device_coords, dict):
    device_coords, non_mesh_axes = _device_id_dict_to_mesh(
        device_coords, axis_sizes, axis_indices
    )
    if non_mesh_axes:
      raise NotImplementedError(non_mesh_axes)
  if not isinstance(device_coords, tuple):
    device_coords = (device_coords,)
  assert len(device_coords) == len(axis_sizes)
  sizes = list(axis_sizes.values())
  ret = 0
  for i in range(len(device_coords)):
    ret += device_coords[i] * math.prod(sizes[i + 1 :])
  return ret


def _device_id_to_logical(device_id, device_id_type, axis_sizes, axis_indices):
  if device_id is None:
    return None
  if device_id_type == primitives.DeviceIdType.MESH:
    return device_coords_to_logical_id(device_id, axis_sizes, axis_indices)
  elif device_id_type == primitives.DeviceIdType.LOGICAL:
    return device_id
  else:
    raise ValueError(f"Unsupported device ID type: {device_id_type}")


def is_int(dtype):
  return jnp.issubdtype(dtype, jnp.integer)


def is_float(dtype):
  return jnp.issubdtype(dtype, jnp.floating)


@dataclasses.dataclass(frozen=True)
class Placeholder:
  """Placeholder for use in `JaxprEnv` below instead of storing a concrete value."""

  shape: tuple[int, ...]
  dtype: jnp.dtype


class JaxprEnv:
  """An environment for interpreting jaxprs, mapping variables to values."""

  def __init__(
      self,
      *,
      vars: Sequence[jax_core.Var] | None = None,
      values: Sequence[Any] | None = None,
      sentinel_for_floating_point_values: Any = None,
  ):
    self._sentinel_for_floating_point_values = (
        sentinel_for_floating_point_values
    )
    self._env: dict[jax_core.Var, Any] = {}

    if vars is None and values is None:
      return

    vars = vars or []
    values = values or []
    self.write_many(vars, values)

  def read(self, var):
    if isinstance(var, jax_core.Literal):
      result = var.val
    else:
      result = self._env[var]
    if isinstance(result, Placeholder):
      result = lax.full(
          result.shape, self._sentinel_for_floating_point_values, result.dtype
      )
    return result

  def read_many(self, vars):
    return safe_map(self.read, vars)

  def write(self, var, value):
    if self._sentinel_for_floating_point_values and is_float(value.dtype):
      value = Placeholder(value.shape, value.dtype)
    self._env[var] = value

  def write_many(self, vars, values):
    safe_map(self.write, vars, values)


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


def to_range(transforms) -> tuple[slice | int, ...]:
  ret = ()
  for transform in transforms:
    # For now, assume only NDIndexer transforms.
    ret = _compose_slice_or_index(
        ret, tuple(_transform_slice_or_index(i) for i in transform.indices)
    )
  return ret


def get_next_indices(grid, indices):
  next_indices = []
  carry = True
  for dim_size, index in reversed(list(zip(grid, indices))):
    i = jnp.where(carry, index + 1, index)
    carry = dim_size == i
    next_indices.append(jnp.where(carry, 0, i))
  return tuple(reversed(next_indices))


def get_indices(grid, loop_index):
  indices = []
  for dim_size in reversed(grid):
    i = loop_index % dim_size
    loop_index = loop_index // dim_size
    indices.append(i)
  return tuple(reversed(indices))
