# Copyright 2021 The JAX Authors.
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

import functools
from typing import (Mapping, Optional, Sequence, Set, Tuple)

from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.lib import xla_client as xc

Shape = Tuple[int, ...]
Device = xc.Device
Index = Tuple[slice, ...]
XLADeviceAssignment = Sequence[Device]


@functools.lru_cache(maxsize=4096)
def _addressable_devices_indices_map(
    sharding: Sharding, global_shape: Shape) -> Mapping[Device, Optional[Index]]:
  if sharding.is_fully_addressable:
    return sharding.devices_indices_map(global_shape)
  return {d: ind for d, ind in sharding.devices_indices_map(global_shape).items()
          if d.process_index == d.client.process_index()}


@util.use_cpp_class(xc.Sharding)
class Sharding:
  """Abstract ``Sharding`` interface which describes how a ``jax.Array`` is laid out
  across devices.
  """

  # Abstract methods below that subclasses should implement.
  @property
  def device_set(self) -> Set[Device]:
    """A ``set`` of global devices that this ``Sharding`` spans.

    In multi-controller JAX, the set of devices is global, i.e., includes
    non-addressable devices from other processes.
    """
    raise NotImplementedError('Subclasses should implement this method.')

  def devices_indices_map(
      self, global_shape: Shape) -> Mapping[Device, Optional[Index]]:
    """A global mapping from device to the slice of the global data it contains.

    The devices in this mapping are global devices i.e. includes
    non-addressable devices from other processes.
    """
    raise NotImplementedError('Subclasses should implement this method.')

  def shard_shape(self, global_shape: Shape) -> Shape:
    """Returns the shape of the data on each device.

    The shard shape returned by this function is calculated from the global
    shape (it takes as an input) and the properties of the sharding.
    """
    raise NotImplementedError('Subclasses should implement this method.')

  def is_equivalent_to(self, other: Sharding, ndim: int) -> bool:
    """Returns True if two shardings put the same logical array
    (sharded/unsharded) on the same device(s).

    For example, every XLACompatibleSharding lowers to GSPMDSharding which
    is a general representation. So `jax.sharding.NamedSharding` is equivalent
    to `jax.sharding.PositionalSharding` if both of them lower to the same
    GSPMDSharding.
    """
    raise NotImplementedError('Subclasses should implement this method.')

  @property
  def is_fully_replicated(self) -> bool:
    """Returns if a sharding is fully replicated on all the devices."""
    raise NotImplementedError('Subclasses should implement this method.')

  #############################################################################
  # Default implementations below that all subclasses will inherit.

  @functools.cached_property
  def addressable_devices(self) -> Set[Device]:
    """A set of devices that are addressable by the current process."""
    # Add a fast path for single controller runtimes.
    if xb.process_count() == 1:
      return self.device_set
    return {d for d in self.device_set
            if d.process_index == d.client.process_index()}

  @functools.cached_property
  def is_fully_addressable(self) -> bool:
    """True if the current process can address all of the devices in device_set.
    """
    # The pytype disable is because pytype can't recognize a cached property.
    return len(self.device_set) == len(self.addressable_devices)  # type: ignore

  def addressable_devices_indices_map(
      self, global_shape: Shape) -> Mapping[Device, Optional[Index]]:
    """A mapping from addressable device to the slice of global data it contains.

    ``addressable_devices_indices_map`` contains that part of
    ``device_indices_map`` that applies to the addressable devices.
    """
    return _addressable_devices_indices_map(self, global_shape)
