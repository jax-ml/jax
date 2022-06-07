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

import dataclasses
import numpy as np
from typing import Sequence, Tuple, Callable, Union, Optional

from jax._src.config import config
from jax._src.util import prod
from jax._src.lib import xla_client as xc
from jax._src.api import device_put
from jax.interpreters import pxla
from jax.experimental.sharding import Sharding, SingleDeviceSharding, XLACompatibleSharding

Shape = Tuple[int, ...]
Device = xc.Device
DeviceArray = xc.Buffer
Index = Tuple[slice, ...]
ArrayLike = Union[np.ndarray, DeviceArray]


@dataclasses.dataclass(frozen=True)
class Shard:
  """A single data shard of an Array.

  Args:
    device : Which device this shard resides on.
    index : The index into the global array of this shard.
    replica_id : Integer id indicating which replica of the global array this
      shard is part of. Always 0 for fully sharded data
      (i.e. when there’s only 1 replica).
    data : The data of this shard. None if ``device`` is non-local.
  """
  device: Device
  index: Index
  replica_id: int
  # None if this `Shard` lives on a non-addressable device.
  data: Optional[Array] = None


class Array:
  # TODO(yashkatariya): Add __slots__ here.

  def __init__(self, shape: Shape, sharding: Sharding,
               arrays: Sequence[DeviceArray], committed: bool):
    self._shape = shape
    self._sharding = sharding
    self._arrays = arrays
    # See https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices
    # for what committed means.
    self._committed = committed

    dtype = arrays[0].dtype
    if config.jax_enable_checks:
      assert all(db.dtype == dtype for db in arrays), (
          "Input arrays to `Array` must have matching dtypes, "
          f"got: {[db.dtype for db in arrays]}")
    self.dtype = dtype

    # Rearrange arrays based on the device assignment.
    if isinstance(sharding, XLACompatibleSharding):
      device_to_buffer = {db.device().id: db for db in self._arrays}
      self._arrays = [device_to_buffer[device.id]
                      for device in self.sharding._addressable_device_assignment()]

  @property
  def shape(self) -> Shape:
    return self._shape

  @property
  def ndim(self):
    return len(self.shape)

  @property
  def size(self):
    return prod(self.shape)

  @property
  def sharding(self):
    return self._sharding

  def is_fully_addressable(self) -> bool:
    return len(self.sharding.device_set) == len(self.sharding.addressable_devices)

  @pxla.maybe_cached_property
  def addressable_shards(self) -> Sequence[Shard]:
    device_to_index = self.sharding.devices_indices_map(self.shape)
    device_to_replica_id = self.sharding.device_replica_id_map(self.shape)

    out = []
    for db in self._arrays:
      db = pxla._set_aval(db)
      device = db.device()
      index = device_to_index[device]
      rid = device_to_replica_id[device]
      # Wrap the device arrays in `Array` until C++ returns an Array instead
      # of a DA.
      array = Array(db.aval, SingleDeviceSharding(device), [db], committed=True)
      out.append(Shard(device, index, rid, array))
    return out

  def copy_to_host_async(self):
    for s in self.addressable_shards:
      if s.replica_id == 0:
        s.data._arrays[0].copy_to_host_async()

  def _value(self) -> np.ndarray:
    # TODO(yashkatariya): Cache the numpy value if its already set.
    if not self.is_fully_addressable():
      raise RuntimeError("Fetching value for `jax.Array` that spans "
                         "non-addressable devices is not possible. You can use "
                         "`jax.experimental.multihost_utils.process_allgather` "
                         "for this use case.")
    self.copy_to_host_async()
    npy_value = np.empty(self.shape, self.dtype)
    for s in self.addressable_shards:
      if s.replica_id == 0:
        npy_value[s.index] = s.data._arrays[0].to_py()  # type: ignore  # [union-attr]
    return npy_value


def make_array_from_callback(shape: Shape, sharding: Sharding,
                             data_callback: Callable[[Optional[Index]], ArrayLike]) -> Array:
  dbs = [
      device_put(data_callback(sharding.device_indices(device, shape)), device)
      for device in sharding.addressable_devices
  ]
  return Array(shape, sharding, dbs, committed=True)
