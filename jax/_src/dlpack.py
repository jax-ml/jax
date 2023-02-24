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

from jax import numpy as jnp
from jax._src import core
from jax._src import device_array
from jax._src import dispatch
from jax._src import array
from jax._src import xla_bridge
from jax._src.lib import xla_client

SUPPORTED_DTYPES = frozenset({
    jnp.int8, jnp.int16, jnp.int32, jnp.int64, jnp.uint8, jnp.uint16,
    jnp.uint32, jnp.uint64, jnp.float16, jnp.bfloat16, jnp.float32,
    jnp.float64, jnp.complex64, jnp.complex128})


def to_dlpack(x: device_array.DeviceArrayProtocol, take_ownership: bool = False):
  """Returns a DLPack tensor that encapsulates a ``DeviceArray`` `x`.

  Takes ownership of the contents of ``x``; leaves `x` in an invalid/deleted
  state.

  Args:
    x: a ``DeviceArray``, on either CPU or GPU.
    take_ownership: If ``True``, JAX hands ownership of the buffer to DLPack,
      and the consumer is free to mutate the buffer; the JAX buffer acts as if
      it were deleted. If ``False``, JAX retains ownership of the buffer; it is
      undefined behavior if the DLPack consumer writes to a buffer that JAX
      owns.
  """
  if not isinstance(x, (device_array.DeviceArray, array.ArrayImpl)):
    raise TypeError("Argument to to_dlpack must be a DeviceArray or Array, got {}"
                    .format(type(x)))
  if isinstance(x, array.ArrayImpl):
    assert len(x._arrays) == 1
    buf = x._arrays[0]
  else:
    buf = x.device_buffer
  return xla_client._xla.buffer_to_dlpack_managed_tensor(
      buf, take_ownership=take_ownership)

def from_dlpack(dlpack):
  """Returns a ``DeviceArray`` representation of a DLPack tensor.

  The returned ``DeviceArray`` shares memory with ``dlpack``.

  Args:
    dlpack: a DLPack tensor, on either CPU or GPU.
  """
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

  buf = xla_client._xla.dlpack_managed_tensor_to_buffer(
      dlpack, cpu_backend, gpu_backend)
  if isinstance(buf, array.ArrayImpl):
    aval = buf.aval
  else:
    xla_shape = buf.xla_shape()
    assert not xla_shape.is_tuple()
    aval = core.ShapedArray(xla_shape.dimensions(), xla_shape.numpy_dtype())
  return jnp.asarray(           # asarray ensures dtype canonicalization
      dispatch.maybe_create_array_from_da(buf, aval, buf.device()))
