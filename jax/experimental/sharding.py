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
from typing import Sequence, Tuple, Optional, Mapping, Set

from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax.interpreters import pxla

Shape = Tuple[int, ...]
Device = xc.Device
Index = Tuple[slice, ...]
XLADeviceAssignment = Sequence[Device]


class Sharding(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def device_set(self) -> Set[Device]:
    """A unique set of devices that this sharding represents.

    Devices can be non-addressable too.
    """
    raise NotImplementedError('Subclasses should implement this method.')

  @pxla.maybe_cached_property
  def addressable_devices(self) -> Set[Device]:
    """A set of addressable devices by the current process"""
    process_index = xb.process_index()
    return {d for d in self.device_set() if d.process_index == process_index}

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
  def _to_xla_op_sharding_and_device_assignment(
      self, num_dimensions: int) -> Tuple[xc.OpSharding, XLADeviceAssignment]:
    raise NotImplementedError('Subclasses should implement this method.')


class MeshPspecSharding(XLACompatibleSharding):

  def __init__(self, mesh: pxla.Mesh, spec: pxla.PartitionSpec):
    self.mesh = mesh
    self.spec = spec

  def device_set(self) -> Set[Device]:
    return set(self.mesh.devices.flat)

  def device_indices(self, device: Device, global_shape: Shape) -> Optional[Index]:
    return self.devices_indices_map(global_shape)[device]

  def devices_indices_map(
      self, global_shape: Shape) -> Mapping[Device, Optional[Index]]:
    # TODO(yashkatariya): Remove this when utilities are moved to pxla.py.
    from jax.experimental import global_device_array

    # `get_shard_indices` is cached.
    return global_device_array.get_shard_indices(global_shape, self.mesh, self.spec)

  def _to_xla_op_sharding_and_device_assignment(
      self, num_dimensions: int) -> Tuple[xc.OpSharding, XLADeviceAssignment]:
    from jax.experimental.pjit import get_array_mapping, _prepare_axis_resources

    parsed_spec, _, _, _ = _prepare_axis_resources(self.spec, "spec")
    array_mapping = get_array_mapping(parsed_spec)
    # TODO(yashkatariya): Move away from sharding spec in MeshPspecSharding
    # since we don't really need sharding spec.
    sharding_spec = pxla.new_mesh_sharding_specs(
        self.mesh.shape, self.mesh.axis_names)(num_dimensions, array_mapping)
    return sharding_spec.sharding_proto(), list(self.mesh.devices.flat),
