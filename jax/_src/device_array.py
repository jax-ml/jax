# Copyright 2018 Google LLC
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

# On-device arrays.

from functools import partial, partialmethod
import operator
from typing import (Any, List, Optional, Union)
import weakref

import numpy as np

import jax
from jax import core
from jax._src.config import config
from jax._src import abstract_arrays
from jax._src import dtypes
from jax._src import profiler
from jax._src.lib import xla_client as xc
import jax._src.util as util

### device-persistent data

xe = xc._xla

Device = xc.Device
Buffer = xe.Buffer


def _forward_method(attrname, self, fun, *args):
  return fun(getattr(self, attrname), *args)
_forward_to_value = partial(_forward_method, "_value")


# The following is used for the type xc.Buffer or _DeviceArray.
DeviceArrayProtocol = Any
DeviceArray = xc.DeviceArrayBase


def make_device_array(
    aval: core.ShapedArray,
    device: Optional[Device],
    device_buffer: Buffer,
) -> Union[Buffer, "_DeviceArray"]:
  """Returns a DeviceArray implementation based on arguments.

  This is to be used only within JAX. It will return either a PythonDeviceArray
  or a C++ equivalent implementation.
  """
  if isinstance(device_buffer, xc.Buffer):

    if device_buffer.aval == aval and device_buffer._device == device:
      return device_buffer
    device_buffer = device_buffer.clone()
    device_buffer._device = device
    device_buffer.aval = aval
    device_buffer.weak_type = aval.weak_type
    return device_buffer

  return _DeviceArray(aval, device, device_buffer)


def type_is_device_array(x):
  """Returns `True` if `x` is a non-sharded DeviceArray.

  Use this function instead of `type(x) is Devicearray`.
  """
  type_x = type(x)
  return type_x is _DeviceArray or type_x is xc.Buffer


def device_array_supports_weakrefs():
  try:
    weakref.ref(DeviceArray())
    return True
  except TypeError:
    return False


class _DeviceArray(DeviceArray):  # type: ignore
  """A DeviceArray is an ndarray backed by a single device memory buffer."""
  # We don't subclass ndarray because that would open up a host of issues,
  # but lax_numpy.py overrides isinstance behavior and attaches ndarray methods.
  __slots__ = [
      "aval", "device_buffer", "_npy_value", "_device", "__weakref__"
  ]
  __array_priority__ = 100

  # DeviceArray has methods that are dynamically populated in lax_numpy.py,
  # and this annotation is needed to make pytype happy.
  _HAS_DYNAMIC_ATTRIBUTES = True

  def __init__(self, aval: core.ShapedArray, device: Optional[Device],
               device_buffer: Buffer):
    """Initializer.

    Args:
      aval: The abstract value associated to this array (shape+dtype+weak_type).
      device:  The optional sticky device. See
        https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices
      device_buffer: The underlying buffer owning the on-device data.
    """
    DeviceArray.__init__(self)
    self.aval = aval
    self.device_buffer = device_buffer
    self._device = device

    self._npy_value = None
    if config.jax_enable_checks:
      assert type(aval) is core.ShapedArray
      npy_value = self._value
      assert npy_value.dtype == aval.dtype and npy_value.shape == aval.shape, (
          aval, npy_value.shape, npy_value.dtype)
      assert (device is None) or device is device_buffer.device()

  def _check_if_deleted(self):
    if self.device_buffer is deleted_buffer:
      raise RuntimeError("DeviceArray has been deleted.")

  @profiler.annotate_function
  def block_until_ready(self):
    """Blocks the caller until the buffer's value has been computed on device.

    This method is mostly useful for timing microbenchmarks that wish to
    time how long a computation takes, without transferring the result back
    to the host.

    Returns the buffer object (`self`).
    """
    self._check_if_deleted()
    self.device_buffer.block_until_ready()
    return self

  @property
  def _value(self):
    self._check_if_deleted()
    if self._npy_value is None:
      self._npy_value = self.device_buffer.to_py()  # pytype: disable=attribute-error  # bind-properties
      self._npy_value.flags.writeable = False
    return self._npy_value

  @property
  def shape(self):
    return self.aval.shape

  @property
  def dtype(self):
    return self.aval.dtype

  @property
  def size(self):
    return util.prod(self.aval.shape)

  @property
  def ndim(self):
    return len(self.aval.shape)

  def device(self):
    self._check_if_deleted()
    return self.device_buffer.device()  # pytype: disable=attribute-error

  def copy_to_host_async(self):
    """Requests a copy of the buffer to the host."""
    self._check_if_deleted()
    if self._npy_value is None:
      self.device_buffer.copy_to_host_async()  # pytype: disable=attribute-error

  def delete(self):
    """Deletes the device array and any cached copy on the host.

    It is an error to access the contents of a `DeviceArray` after it has
    been deleted.

    Use of this method is optional; device buffers will be reclaimed
    automatically by Python when a DeviceArray object is garbage collected.
    However, it is sometimes useful to have more explicit control over the
    time of deletion.
    """
    self.device_buffer.delete()  # pytype: disable=attribute-error
    self.device_buffer = deleted_buffer
    self._npy_value = None

  @property
  def __cuda_array_interface__(self):
    return self.device_buffer.__cuda_array_interface__  # pytype: disable=attribute-error  # bind-properties


# Adding methods dynamically to both _DeviceArray and xc.Buffer
# pylint: disable=protected-access
for device_array in [DeviceArray]:

  def __repr__(self):
    line_width = np.get_printoptions()["linewidth"]
    prefix = '{}('.format(self.__class__.__name__.lstrip('_'))
    s = np.array2string(self._value, prefix=prefix, suffix=',',
                        separator=', ', max_line_width=line_width)
    if self.aval is not None and self.aval.weak_type:
      dtype_str = f'dtype={self.dtype.name}, weak_type=True)'
    else:
      dtype_str = f'dtype={self.dtype.name})'
    last_line_len = len(s) - s.rfind('\n') + 1
    sep = ' '
    if last_line_len + len(dtype_str) + 1 > line_width:
      sep = ' ' * len(prefix)
    return f"{prefix}{s},{sep}{dtype_str}"

  setattr(device_array, "__repr__", __repr__)

  def item(self):
    if dtypes.issubdtype(self.dtype, np.complexfloating):
      return complex(self)
    elif dtypes.issubdtype(self.dtype, np.floating):
      return float(self)
    elif dtypes.issubdtype(self.dtype, np.integer):
      return int(self)
    elif dtypes.issubdtype(self.dtype, np.bool_):
      return bool(self)
    else:
      raise TypeError(self.dtype)

  setattr(device_array, "item", item)

  def __len__(self):
    try:
      return self.aval.shape[0]
    except IndexError as err:
      raise TypeError("len() of unsized object") from err  # same as numpy error

  setattr(device_array, "__len__", __len__)

  def __iter__(self):
    if self.ndim == 0:
      raise TypeError("iteration over a 0-d array")  # same as numpy error
    else:
      return (sl for chunk in self._chunk_iter(100) for sl in chunk._unstack())

  setattr(device_array, "__iter__", __iter__)

  def __reversed__(self):
    return iter(self[::-1])

  setattr(device_array, "__reversed__", __reversed__)

  def __format__(self, format_spec):
    # Simulates behavior of https://github.com/numpy/numpy/pull/9883
    if self.ndim == 0:
      return format(self._value[()], format_spec)
    else:
      return format(self._value, format_spec)

  setattr(device_array, "__format__", __format__)

  def __array__(self, dtype=None, context=None):
    return np.asarray(self._value, dtype=dtype)

  setattr(device_array, "__array__", __array__)

  def __dlpack__(self):
    from jax.dlpack import to_dlpack  # pylint: disable=g-import-not-at-top
    return to_dlpack(self)

  setattr(device_array, "__dlpack__", __dlpack__)

  def __reduce__(self):
    fun, args, arr_state = self._value.__reduce__()
    aval_state = {'weak_type': self.aval.weak_type,
                  'named_shape': self.aval.named_shape}
    return (reconstruct_device_array, (fun, args, arr_state, aval_state))

  setattr(device_array, "__reduce__", __reduce__)

  setattr(device_array, "__str__", partialmethod(_forward_to_value, str))
  setattr(device_array, "__bool__", partialmethod(_forward_to_value, bool))
  setattr(device_array, "__nonzero__", partialmethod(_forward_to_value, bool))
  setattr(device_array, "__float__", lambda self: self._value.__float__())
  setattr(device_array, "__int__", lambda self: self._value.__int__())
  setattr(device_array, "__complex__", lambda self: self._value.__complex__())
  setattr(device_array, "__hex__", partialmethod(_forward_to_value, hex))
  setattr(device_array, "__oct__", partialmethod(_forward_to_value, oct))
  setattr(device_array, "__index__", partialmethod(_forward_to_value,
                                                   operator.index))
  to_bytes = lambda self, order="C": self._value.tobytes(order)
  setattr(device_array, "tobytes", to_bytes)
  del to_bytes
  setattr(device_array, "tolist", lambda self: self._value.tolist())

  # explicitly set to be unhashable.
  setattr(device_array, "__hash__", None)

  # clobbered when jax.numpy is imported, but useful in tests
  setattr(device_array, "__eq__", lambda self, other: self._value == other)

  # The following methods are dynamically overridden in lax_numpy.py.
  def raise_not_implemented():
    raise NotImplementedError

  setattr(device_array, "__getitem__", lambda self, i: raise_not_implemented())
# pylint: enable=protected-access


def reconstruct_device_array(fun, args, arr_state, aval_state):
  """Method to reconstruct a device array from a serialized state."""
  np_value = fun(*args)
  np_value.__setstate__(arr_state)
  jnp_value = jax.device_put(np_value)
  jnp_value.aval = jnp_value.aval.update(**aval_state)
  return jnp_value


class DeletedBuffer(object): pass
deleted_buffer = DeletedBuffer()


device_array_types: List[type] = [xc.Buffer, _DeviceArray]
for _device_array in device_array_types:
  core.literalable_types.add(_device_array)
  core.pytype_aval_mappings[_device_array] = abstract_arrays.canonical_concrete_aval
