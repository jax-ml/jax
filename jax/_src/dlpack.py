# Copyright 2020 Google LLC
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

from jax import core
from jax import lazy
from jax.interpreters import xla
import jax.lib
from jax.lib import xla_client
from jax.lib import xla_bridge

def to_dlpack(x: xla.DeviceArrayProtocol, take_ownership: bool = False):
  """Returns a DLPack tensor that encapsulates a DeviceArray `x`.

  Takes ownership of the contents of `x`; leaves `x` in an invalid/deleted
  state.

  Args:
    x: a `DeviceArray`, on either CPU or GPU.
    take_ownership: If ``True``, JAX hands ownership of the buffer to DLPack,
      and the consumer is free to mutate the buffer; the JAX buffer acts as if
      it were deleted. If ``False``, JAX retains ownership of the buffer; it is
      undefined behavior if the DLPack consumer writes to a buffer that JAX
      owns.
  """
  if not isinstance(x, xla.DeviceArray):
    raise TypeError("Argument to to_dlpack must be a DeviceArray, got {}"
                    .format(type(x)))
  buf = xla._force(x).device_buffer
  if jax.lib.version >= (0, 1, 57):
    return xla_client._xla.buffer_to_dlpack_managed_tensor(
        buf, take_ownership=take_ownership)
  else:
    # Jaxlibs before 0.1.57 always take ownership.
    if not take_ownership:
      raise ValueError(
          "to_dlpack with take_ownership=False requires jaxlib >= 0.1.57")
    return xla_client._xla.buffer_to_dlpack_managed_tensor(buf)

def from_dlpack(dlpack, backend=None):
  """Returns a `DeviceArray` representation of a DLPack tensor `dlpack`.

  The returned `DeviceArray` shares memory with `dlpack`.

  Args:
    dlpack: a DLPack tensor, on either CPU or GPU.
    backend: experimental, optional: the platform on which `dlpack` lives.
  """
  # TODO(phawkins): ideally the user wouldn't need to provide a backend and we
  # would be able to figure it out from the DLPack.
  backend = backend or xla_bridge.get_backend()
  client = getattr(backend, "client", backend)
  buf = xla_client._xla.dlpack_managed_tensor_to_buffer(dlpack, client)
  # TODO(jblespiau): We can simply use buf.xla_shape() when version 0.1.58 is
  # the default.
  xla_shape = getattr(buf, "xla_shape", buf.shape)()
  assert not xla_shape.is_tuple()
  aval = core.ShapedArray(xla_shape.dimensions(), xla_shape.numpy_dtype())
  return xla.make_device_array(aval, buf.device(), lazy.array(aval.shape), buf)  # pytype: disable=attribute-error
