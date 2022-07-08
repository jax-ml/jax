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
from jax.interpreters import pxla, mlir

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
  def normalize(self):
    raise NotImplementedError('Subclasses should implement this method.')

  @abc.abstractmethod
  def device_replica_id_map(self, global_shape: Shape) -> Mapping[Device, int]:
    raise NotImplementedError('Subclasses should implement this method.')

  @cache()
  def _addressable_device_assignment(self) -> XLADeviceAssignment:
    process_index = xb.process_index()
    return [d for d in self._device_assignment() if d.process_index == process_index]

  @abc.abstractmethod
  def _to_xla_op_sharding(self, num_dimensions: int) -> Optional[xc.OpSharding]:
    raise NotImplementedError('Subclasses should implement this method.')


class MeshPspecSharding(XLACompatibleSharding):

  def __init__(
      self, mesh: pxla.Mesh, spec: pxla.PartitionSpec, _parsed_pspec = None):

    self.mesh = mesh
    self.spec = spec

    # This split exists because you can pass `_parsed_pspec` that has been
    # modified from the original. For example: Adding extra dimension to
    # axis_resources for vmap handlers. In such cases you need to preserve the
    # `sync` attribute of parsed pspecs.
    # PartitionSpec is inferred from the parsed pspec in this case.
    # TODO(yaskatariya): Remove this and replace this with a normalized
    # representation of Parsed Pspec
    if _parsed_pspec is None:
      from jax.experimental import pjit
      self._parsed_pspec, _, _, _ = pjit._prepare_axis_resources(
          self.spec, "MeshPspecSharding spec")
    else:
      self._parsed_pspec = _parsed_pspec

  def __repr__(self):
    return f'MeshPspecSharding(\n  mesh={self.mesh},\n  partition_spec={self.spec})'

  def __hash__(self):
    return hash((self.mesh, self.spec))

  def __eq__(self, other):
    return self.mesh == other.mesh and self.spec == other.spec

  def normalize(self):
    from jax.experimental import pjit
    cp = pjit.CanonicalizedParsedPartitionSpec(self._parsed_pspec)
    return MeshPspecSharding._from_parsed_pspec(self.mesh, cp)

  @classmethod
  def _from_parsed_pspec(cls, mesh, parsed_pspec):
    from jax.experimental import pjit
    return cls(mesh, pjit._get_single_pspec(parsed_pspec), parsed_pspec)

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

  @cache()
  def _device_assignment(self) -> XLADeviceAssignment:
    return list(self.mesh.devices.flat)

  @cache()
  def _to_xla_op_sharding(
      self, num_dimensions: int,
      axis_ctx: Optional[mlir.SPMDAxisContext] = None) -> Optional[xc.OpSharding]:
    from jax.experimental.pjit import get_array_mapping, _prepare_axis_resources

    parsed_spec, _, _, _ = _prepare_axis_resources(self.spec, "spec")
    array_mapping = get_array_mapping(parsed_spec)
    # TODO(yashkatariya): Move away from sharding spec in MeshPspecSharding
    # since we don't really need sharding spec.
    sharding_spec = pxla.new_mesh_sharding_specs(
        self.mesh.shape, self.mesh.axis_names)(num_dimensions, array_mapping)
    # Used in `with_sharding_constraint`.
    special_axes = {}
    if axis_ctx is not None:
      axis_names = self.mesh.axis_names
      for manual_axis in axis_ctx.manual_axes:
        special_axes[axis_names.index(manual_axis)] = xc.OpSharding.Type.MANUAL
    return sharding_spec.sharding_proto(special_axes=special_axes)


class SingleDeviceSharding(XLACompatibleSharding):

  def __init__(self, device: Device):
    self._device = device

  def __repr__(self):
    return f"SingleDeviceSharding(device={self._device})"

  def __hash__(self):
    return hash(self._device)

  def __eq__(self, other):
    if not isinstance(other, SingleDeviceSharding):
      return False
    return self._device == other._device

  def normalize(self):
    return SingleDeviceSharding(self._device)

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

  @cache()
  def _to_xla_op_sharding(self, num_dimensions: int) -> Optional[xc.OpSharding]:
    proto = xc.OpSharding()
    proto.type = xc.OpSharding.Type.REPLICATED
    return proto


class PmapSharding(XLACompatibleSharding):

  def __init__(self, devices: np.ndarray, sharding_spec: pxla.ShardingSpec):
    self.devices = devices
    # The sharding spec should be pmap's sharding spec.
    self.sharding_spec = sharding_spec

  def normalize(self):
    return PmapSharding(self.devices, self.sharding_spec)

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

  @cache()
  def _device_assignment(self) -> XLADeviceAssignment:
    return list(self.devices.flat)

  def _to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
    raise NotImplementedError("pmap doesn't use OpSharding.")
