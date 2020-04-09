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

from . import core
from . import lazy
from .interpreters import xla
from .lib import xla_client
from .lib import xla_bridge

def to_dlpack(x: xla.DeviceArray):
  """Returns a DLPack tensor that encapsulates a DeviceArray `x`.

  The DLPack shares memory with `x`.

  Args:
    x: a `DeviceArray`, on either CPU or GPU.
  """
  if not isinstance(x, xla.DeviceArray):
    raise TypeError("Argument to to_dlpack must be a DeviceArray, got {}"
                    .format(type(x)))
  buf = xla._force(x).device_buffer
  return xla_client._xla.BufferToDLPackManagedTensor(buf)

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
  buf = xla_client._xla.DLPackManagedTensorToBuffer(dlpack, backend.client)
  xla_shape = buf.shape()
  assert not xla_shape.is_tuple()
  aval = core.ShapedArray(xla_shape.dimensions(), xla_shape.numpy_dtype())
  return xla.DeviceArray(aval, buf.device(), lazy.array(aval.shape), buf)  # pytype: disable=attribute-error
