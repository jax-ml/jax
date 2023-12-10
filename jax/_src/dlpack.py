# Copyright 2020 The JAX Authors.
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

import enum
from typing import Any
import warnings

from jax import numpy as jnp
from jax._src import array
from jax._src import xla_bridge
from jax._src.lib import xla_client
from jax._src.typing import Array


SUPPORTED_DTYPES = frozenset({
    jnp.int8, jnp.int16, jnp.int32, jnp.int64, jnp.uint8, jnp.uint16,
    jnp.uint32, jnp.uint64, jnp.float16, jnp.bfloat16, jnp.float32,
    jnp.float64, jnp.complex64, jnp.complex128})


# Mirror of dlpack.h enum
class DLDeviceType(enum.IntEnum):
  kDLCPU = 1
  kDLCUDA = 2
  kDLROCM = 10


def to_dlpack(x: Array, take_ownership: bool = False,
              stream: int | Any | None = None):
  """Returns a DLPack tensor that encapsulates a :class:`~jax.Array` ``x``.

  Args:
    x: a :class:`~jax.Array`, on either CPU or GPU.
    take_ownership: Deprecated. It is a no-op to set take_ownership. Will be
      deleted in 01/2024.
    stream: optional platform-dependent stream to wait on until the buffer is
      ready. This corresponds to the `stream` argument to ``__dlpack__``
      documented in https://dmlc.github.io/dlpack/latest/python_spec.html.
  """
  if not isinstance(x, array.ArrayImpl):
    raise TypeError("Argument to to_dlpack must be a jax.Array, "
                    f"got {type(x)}")
  assert len(x.devices()) == 1
  if take_ownership:
    warnings.warn(
        "take_ownership in to_dlpack is deprecated and it is a no-op."
    )
  return xla_client._xla.buffer_to_dlpack_managed_tensor(
      x.addressable_data(0), stream=stream
  )  # type: ignore



def from_dlpack(external_array):
  """Returns a :class:`~jax.Array` representation of a DLPack tensor.

  The returned :class:`~jax.Array` shares memory with ``external_array``.

  Args:
    external_array: an array object that has __dlpack__ and __dlpack_device__
      methods, or a DLPack tensor on either CPU or GPU (legacy API).

  Returns:
    A jax.Array
  """
  if hasattr(external_array, "__dlpack__"):
    dl_device_type, device_id = external_array.__dlpack_device__()
    try:
      device_platform = {
          DLDeviceType.kDLCPU: "cpu",
          DLDeviceType.kDLCUDA: "cuda",
          DLDeviceType.kDLROCM: "rocm",
      }[dl_device_type]
    except TypeError:
      # https://dmlc.github.io/dlpack/latest/python_spec.html recommends using
      # TypeError.
      raise TypeError(
          "Array passed to from_dlpack is on unsupported device type "
          f"(DLDeviceType: {dl_device_type}, array: {external_array}")

    backend = xla_bridge.get_backend(device_platform)
    device = backend.device_from_local_hardware_id(device_id)
    try:
      stream = device.get_stream_for_external_ready_events()
    except xla_client.XlaRuntimeError as err:  # type: ignore
      if "UNIMPLEMENTED" in str(err):
        stream = None
      else:
        raise
    dlpack = external_array.__dlpack__(stream=stream)

    return jnp.asarray(xla_client._xla.dlpack_managed_tensor_to_buffer(
        dlpack, device, stream))
  else:
    # Legacy path
    dlpack = external_array
    cpu_backend = xla_bridge.get_backend("cpu")
    try:
      gpu_backend = xla_bridge.get_backend("cuda")
    except RuntimeError:
      gpu_backend = None

    # Try ROCm if CUDA backend not found
    if gpu_backend is None:
      try:
        gpu_backend = xla_bridge.get_backend("rocm")
      except RuntimeError:
        gpu_backend = None

    return jnp.asarray(xla_client._xla.dlpack_managed_tensor_to_buffer(
        dlpack, cpu_backend, gpu_backend))
