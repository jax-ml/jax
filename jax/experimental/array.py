# Copyright 2021 Google LLC
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

import numpy as np
from typing import Sequence, Tuple, Callable, Union, Optional, cast, List

from jax import core
from jax._src import dispatch
from jax._src.config import config
from jax._src.util import prod
from jax._src.lib import xla_client as xc
from jax._src.api import device_put
from jax.interpreters import pxla, xla
from jax.experimental.sharding import (Sharding, SingleDeviceSharding,
                                       XLACompatibleSharding)

Shape = Tuple[int, ...]
Device = xc.Device
DeviceArray = xc.Buffer
Index = Tuple[slice, ...]
ArrayLike = Union[np.ndarray, DeviceArray]


class Shard:
  """A single data shard of an Array.

  Attributes:
    device : Which device this shard resides on.
    index : The index into the global array of this shard.
    replica_id : Integer id indicating which replica of the global array this
      shard is part of. Always 0 for fully sharded data
      (i.e. when there’s only 1 replica).
    data : The data of this shard. None if ``device`` is non-local.
  """

  def __init__(self, device: Device, sharding: Sharding, global_shape: Shape,
               data: Optional[Array] = None):
    self.device = device
    self._sharding = sharding
    self._global_shape = global_shape
    self.data = data

  def __repr__(self):
    try:
      return (f'Shard(device={repr(self.device)}, index={self.index}, '
              f'replica_id={self.replica_id}, data={self.data})')
    except ValueError:
      return f'Shard(device={repr(self.device)}, data={self.data})'

  @property
  def index(self) -> Index:
    try:
      device_indices_fn = self._sharding.device_indices
    except AttributeError:
      raise ValueError('Cannot calculate indices from sharding: '
                       f'{self._sharding}. Please create a device to index '
                       'mapping for your sharding.') from None
    index = device_indices_fn(self.device, self._global_shape)
    assert index is not None
    return index

  @property
  def replica_id(self) -> int:
    try:
      device_replica_id_fn = self._sharding.device_replica_id_map  # pytype: disable=attribute-error
    except AttributeError:
      raise ValueError('Cannot calculate replica ids from sharding: '
                       f'{self._sharding}. Please create a device to replica id '
                       'mapping for your sharding.') from None
    return device_replica_id_fn(self._global_shape)[self.device]


class Array:
  # TODO(yashkatariya): Add __slots__ here.

  def __init__(self, shape: Shape, sharding: Sharding,
               arrays: Union[Sequence[DeviceArray], Sequence[Array]], committed: bool):
    self._shape = shape
    self._sharding = sharding
    # Extract DeviceArrays from arrays with `SingleDeviceSharding` to keep the
    # code handling `self._arrays` simpler.
    # TODO(yashkatariya): This will be slower as it will happen during
    # `__init__` on single controller environment. Make it lazy.
    self._arrays: List[DeviceArray] = [a if isinstance(a, DeviceArray) else a._arrays[0]
                                       for a in arrays]
    # See https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices
    # for what committed means.
    self._committed = committed
    self._npy_value = None

    dtype = self._arrays[0].dtype
    if config.jax_enable_checks:
      assert all(db.dtype == dtype for db in self._arrays), (
          "Input arrays to `Array` must have matching dtypes, "
          f"got: {[db.dtype for db in self._arrays]}")
    self.dtype = dtype

    # Rearrange arrays based on the device assignment.
    if isinstance(sharding, XLACompatibleSharding):
      device_to_buffer = {db.device().id: db for db in self._arrays}
      self._arrays = [device_to_buffer[device.id]
                      for device in self.sharding._addressable_device_assignment()]

  @property
  def shape(self) -> Shape:
    return self._shape

  # TODO(yashkatariya): Remove this and take aval as an input to account for
  # weak types.
  @property
  def aval(self) -> core.ShapedArray:
    return core.ShapedArray(self.shape, self.dtype)

  @property
  def ndim(self):
    return len(self.shape)

  @property
  def size(self):
    return prod(self.shape)

  @property
  def sharding(self):
    return self._sharding

  def __repr__(self):
    prefix = '{}('.format(self.__class__.__name__.lstrip('_'))
    # TODO(yashkatariya): Add weak_type to the repr and handle weak_type
    # generally too.
    dtype_str = f'dtype={self.dtype.name})'

    if self.is_fully_addressable():
      line_width = np.get_printoptions()["linewidth"]
      s = np.array2string(self._value, prefix=prefix, suffix=',',
                          separator=', ', max_line_width=line_width)
      last_line_len = len(s) - s.rfind('\n') + 1
      sep = ' '
      if last_line_len + len(dtype_str) + 1 > line_width:
        sep = ' ' * len(prefix)
      return f"{prefix}{s},{sep}{dtype_str}"
    else:
      return f"{prefix}{self.shape}{dtype_str}"

  def is_fully_addressable(self) -> bool:
    return self.sharding.is_fully_addressable()

  def __array__(self, dtype=None):
    return np.asarray(self._value, dtype=dtype)

  @pxla.maybe_cached_property
  def addressable_shards(self) -> Sequence[Shard]:
    self._check_if_deleted()
    out = []
    for db in self._arrays:
      db = pxla._set_aval(db)
      device = db.device()
      # Wrap the device arrays in `Array` until C++ returns an Array instead
      # of a DA.
      array = Array(db.shape, SingleDeviceSharding(device), [db], committed=True)
      out.append(Shard(device, self.sharding, self.shape, array))
    return out

  def delete(self):
    if self._arrays is None:
      return
    for buf in self._arrays:
      buf.delete()
    self._arrays = None
    self._npy_value = None

  def _check_if_deleted(self):
    if self._arrays is None:
      raise ValueError("Array has been deleted.")

  def block_until_ready(self):
    self._check_if_deleted()
    for db in self._arrays:
      db.block_until_ready()
    return self

  def copy_to_host_async(self):
    self._check_if_deleted()
    if self._npy_value is None:
      try:
        self.addressable_shards[0].replica_id
        replica_id_exists = True
      except ValueError:
        replica_id_exists = False

      for s in self.addressable_shards:
        if not replica_id_exists or s.replica_id == 0:
          s.data._arrays[0].copy_to_host_async()  # pytype: disable=attribute-error

  @property
  def _value(self) -> np.ndarray:
    self._check_if_deleted()
    if not self.is_fully_addressable():
      raise RuntimeError("Fetching value for `jax.Array` that spans "
                         "non-addressable devices is not possible. You can use "
                         "`jax.experimental.multihost_utils.process_allgather` "
                         "for this use case.")
    if self._npy_value is None:
      self.copy_to_host_async()
      npy_value = np.empty(self.shape, self.dtype)

      try:
        self.addressable_shards[0].replica_id
        replica_id_exists = True
      except ValueError:
        replica_id_exists = False

      for s in self.addressable_shards:
        if not replica_id_exists or s.replica_id == 0:
          npy_value[s.index] = s.data._arrays[0].to_py()  # type: ignore  # [union-attr]
      self._npy_value = npy_value  # type: ignore
    # https://docs.python.org/3/library/typing.html#typing.cast
    return cast(np.ndarray, self._npy_value)


def make_array_from_callback(shape: Shape, sharding: Sharding,
                             data_callback: Callable[[Optional[Index]], ArrayLike]) -> Array:
  arrays = [
      device_put(data_callback(sharding.device_indices(device, shape)), device)
      for device in sharding.addressable_devices
  ]
  return Array(shape, sharding, arrays, committed=True)


core.pytype_aval_mappings[Array] = lambda x: core.ShapedArray(x.shape, x.dtype)
xla.pytype_aval_mappings[Array] = lambda x: core.ShapedArray(x.shape, x.dtype)
xla.canonicalize_dtype_handlers[Array] = pxla.identity


def _device_put_array(x, device: Optional[Device]):
  # TODO(yashkatariya): Remove this restriction and the round trip via host
  # once lowering to XLA goes through `lower_mesh_computation`.
  assert x.is_fully_addressable()
  if isinstance(x.sharding, SingleDeviceSharding):
    x = dispatch._copy_device_array_to_device(pxla._set_aval(x._arrays[0]), device)
    return (x,)
  else:
    # Round trip via host if x is sharded. SDA also does a round trip via host.
    return dispatch._device_put_array(x._value, device)

dispatch.device_put_handlers[Array] = _device_put_array


def _array_shard_arg(x, devices, indices):
  return x._arrays
pxla.shard_arg_handlers[Array] = _array_shard_arg


def _array_result_handler(global_aval, out_sharding):
  return lambda bufs: Array(global_aval.shape, out_sharding, bufs, committed=True)
pxla.global_result_handlers[(core.ShapedArray, pxla.OutputType.Array)] = _array_result_handler
pxla.global_result_handlers[(core.ConcreteArray, pxla.OutputType.Array)] = _array_result_handler
