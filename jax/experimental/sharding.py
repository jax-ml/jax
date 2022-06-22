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
from collections import Counter
from typing import Sequence, Tuple, Optional, Mapping, Dict, Set

from jax._src.util import cache, safe_zip
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax.interpreters import pxla

import numpy as np

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
    return {d for d in self.device_set if d.process_index == process_index}

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
  def _device_assignment(self) -> XLADeviceAssignment:
    raise NotImplementedError('Subclasses should implement this method.')

  @abc.abstractmethod
  def device_replica_id_map(self, global_shape: Shape) -> Mapping[Device, int]:
    raise NotImplementedError('Subclasses should implement this method.')

  @cache()
  def _addressable_device_assignment(self) -> XLADeviceAssignment:
    process_index = xb.process_index()
    return [d for d in self._device_assignment() if d.process_index == process_index]

  @abc.abstractmethod
  def _to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
    raise NotImplementedError('Subclasses should implement this method.')


class MeshPspecSharding(XLACompatibleSharding):

  def __init__(self, mesh: pxla.Mesh, spec: pxla.PartitionSpec):
    self.mesh = mesh
    self.spec = spec

  @pxla.maybe_cached_property
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

  def _hashed_index(self, x) -> int:
    return hash(tuple((v.start, v.stop) for v in x))

  @cache()
  def device_replica_id_map(self, global_shape: Shape) -> Mapping[Device, int]:
    index_to_replica: Dict[int, int] = Counter()
    out = {}
    for device, index in self.devices_indices_map(global_shape).items():
      h_index = self._hashed_index(index)
      replica_id = index_to_replica[h_index]
      index_to_replica[h_index] += 1
      out[device] = replica_id
    return out

  def _device_assignment(self) -> XLADeviceAssignment:
    return list(self.mesh.devices.flat)

  def _to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
    from jax.experimental.pjit import get_array_mapping, _prepare_axis_resources

    parsed_spec, _, _, _ = _prepare_axis_resources(self.spec, "spec")
    array_mapping = get_array_mapping(parsed_spec)
    # TODO(yashkatariya): Move away from sharding spec in MeshPspecSharding
    # since we don't really need sharding spec.
    sharding_spec = pxla.new_mesh_sharding_specs(
        self.mesh.shape, self.mesh.axis_names)(num_dimensions, array_mapping)
    return sharding_spec.sharding_proto()


class SingleDeviceSharding(XLACompatibleSharding):

  def __init__(self, device: Device):
    self._device = device

  @pxla.maybe_cached_property
  def device_set(self) -> Set[Device]:
    return {self._device}

  def device_indices(self, device: Device, global_shape: Shape) -> Optional[Index]:
    return self.devices_indices_map(global_shape)[device]

  @cache()
  def devices_indices_map(
      self, global_shape: Shape) -> Mapping[Device, Optional[Index]]:
    return {self._device: (slice(None),) * len(global_shape)}

  @cache()
  def device_replica_id_map(self, global_shape: Shape) -> Mapping[Device, int]:
    return {self._device: 0}

  def _device_assignment(self) -> XLADeviceAssignment:
    return [self._device]

  def _to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
    proto = xc.OpSharding()
    proto.type = xc.OpSharding.Type.REPLICATED
    return proto


class PmapSharding(XLACompatibleSharding):

  def __init__(self, devices: np.ndarray, sharding_spec: pxla.ShardingSpec):
    self.devices = devices
    # The sharding spec should be pmap's sharding spec.
    self.sharding_spec = sharding_spec

  @pxla.maybe_cached_property
  def device_set(self) -> Set[Device]:
    return set(self.devices.flat)

  def device_indices(self, device: Device, global_shape: Shape) -> Optional[Index]:
    return self.devices_indices_map(global_shape)[device]

  @pxla.maybe_cached_property
  def sharded_dim(self):
    for i, s in enumerate(self.sharding_spec.sharding):
      if isinstance(s, pxla.Unstacked):
        return i
    return None

  @cache()
  def devices_indices_map(
      self, global_shape: Shape) -> Mapping[Device, Optional[Index]]:
    indices = pxla.spec_to_indices(global_shape, self.sharding_spec)
    return {d: i for d, i in safe_zip(self.devices.flat, indices)}  # type: ignore

  def _hashed_index(self, x) -> int:
    return hash(
        tuple((v.start, v.stop) if isinstance(v, slice) else v for v in x))

  @cache()
  def device_replica_id_map(self, global_shape: Shape) -> Mapping[Device, int]:
    index_to_replica: Dict[int, int] = Counter()
    out = {}
    for device, index in self.devices_indices_map(global_shape).items():
      h_index = self._hashed_index(index)
      replica_id = index_to_replica[h_index]
      index_to_replica[h_index] += 1
      out[device] = replica_id
    return out

  def _device_assignment(self) -> XLADeviceAssignment:
    return list(self.devices.flat)

  def _to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
    raise NotImplementedError("pmap doesn't use OpSharding.")
