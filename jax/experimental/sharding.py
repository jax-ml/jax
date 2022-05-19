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

import abc
import numpy as np
from typing import Sequence, Tuple, Optional, Mapping

from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax.interpreters import pxla

Shape = Tuple[int, ...]
Device = xc.Device
Index = Tuple[slice, ...]


class Sharding(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def device_set(self) -> np.ndarray:
    """A unique set of devices that this sharding represents.

    Devices can be non-addressable too.
    """
    raise NotImplementedError('Subclasses should implement this method.')

  @pxla.maybe_cached_property
  def addressable_devices(self) -> Sequence[Device]:
    """A list of addressable devices by the current process"""
    process_index = xb.process_index()
    return [d for d in self.device_set().flat if d.process_index == process_index]

  @abc.abstractmethod
  def device_indices(self, device: Device,
                     global_shape: Shape) -> Optional[Index]:
    raise NotImplementedError('Subclasses should implement this method.')

  @abc.abstractmethod
  def devices_indices_map(
      self, global_shape: Shape) -> Mapping[Device, Optional[Index]]:
    raise NotImplementedError('Subclasses should implement this method.')


class XLACompatibleSharding(Sharding):

  @abc.abstractmethod
  def to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
    raise NotImplementedError('Subclasses should implement this method.')


class MeshPspecSharding(XLACompatibleSharding):

  def __init__(self, mesh: pxla.Mesh, pspec: pxla.PartitionSpec):
    self.mesh = mesh
    self.pspec = pspec

  def device_set(self):
    return self.mesh.devices

  def device_indices(self, device: Device, global_shape: Shape) -> Optional[Index]:
    return self.devices_indices_map()[device]

  def devices_indices_map(
      self, global_shape: Shape) -> Mapping[Device, Optional[Index]]:
    # TODO(yashkatariya): Remove this when utilities are moved to pxla.py.
    from jax.experimental import global_device_array

    return global_device_array.get_shard_indices(global_shape, self.mesh, self.pspec)

  def to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
    from jax.experimental.pjit import get_array_mapping, _prepare_axis_resources

    parsed_pspec, _, _, _ = _prepare_axis_resources(self.pspec, "pspec")
    array_mapping = get_array_mapping(parsed_pspec)
    # TODO(yashkatariya): Move away from sharding spec in MeshPspecSharding
    # since we don't really need sharding spec.
    sharding_spec = pxla.mesh_sharding_specs(
        self.mesh.shape, self.mesh.axis_names)(num_dimensions, array_mapping)
    return sharding_spec.sharding_proto()

