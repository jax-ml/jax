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
"""Implementation of GlobalShardedDeviceArray."""

import dataclasses
import numpy as np
from typing import Callable, Sequence, Tuple, Union, Mapping
from .. import core
from jax._src.lib import xla_client as xc
from ..interpreters import pxla
from .._src.util import prod, safe_zip
from .._src.api import device_put
from ..interpreters.sharded_jit import PartitionSpec
from .pjit import get_array_mapping, _prepare_axis_resources

Shape = Tuple[int, ...]
MeshAxes = Sequence[Union[str, Tuple[str], None]]
DeviceArray = xc.Buffer
Device = xc.Device
ArrayLike = Union[np.ndarray, DeviceArray]
Index = Tuple[slice, ...]


@dataclasses.dataclass(frozen=True)
class _HashableSlice:
  val: Index

  def __hash__(self):
    return hash(tuple([(v.start, v.stop, v.step) for v in self.val]))

  def __eq__(self, other):
    return self.val == other.val


def shard_indices(global_shape: Shape, global_mesh: pxla.Mesh,
                 mesh_axes: MeshAxes) -> Mapping[Device, Index]:
  if not isinstance(mesh_axes, PartitionSpec):
    pspec = PartitionSpec(*mesh_axes)
  else:
    pspec = mesh_axes
  parsed_pspec, _, _ = _prepare_axis_resources(pspec, "mesh_axes")
  array_mapping = get_array_mapping(parsed_pspec)
  # The dtype doesn't matter for creating sharding specs.
  aval = core.ShapedArray(global_shape, np.float32)
  sharding_spec = pxla.mesh_sharding_specs(
      global_mesh.shape, global_mesh.axis_names)(aval, array_mapping)
  indices = pxla.spec_to_indices(global_shape, sharding_spec)
  for index in indices:
    assert isinstance(index, tuple)
    for idx in index:
      assert isinstance(idx, slice)
  # The type: ignore is to ignore the type returned by `spec_to_indices`.
  return dict((d, i) for d, i in safe_zip(global_mesh.devices.flat, indices))  # type: ignore


def shard_shape(global_shape, global_mesh, mesh_axes) -> Shape:
  chunk_size = []
  for mesh_axis, size in zip(mesh_axes, global_shape):
    if not mesh_axis:
      chunk_size.append(size)
    elif isinstance(mesh_axis, tuple):
      m = prod([global_mesh.shape[ma] for ma in mesh_axis])
      chunk_size.append(size // m)
    else:
      chunk_size.append(size // global_mesh.shape[mesh_axis])
  if len(chunk_size) != len(global_shape):
    chunk_size.extend(global_shape[len(chunk_size):])
  return tuple(chunk_size)


@dataclasses.dataclass(frozen=True)
class Shard:
  device: Device
  index: Index
  replica_id: int
  data: DeviceArray


class GlobalShardedDeviceArray:

  def __init__(self,
               global_shape: Shape,
               dtype,
               global_mesh: pxla.Mesh,
               mesh_axes: MeshAxes,
               device_buffers: Sequence[DeviceArray]):
    self._global_shape = global_shape
    self._dtype = dtype
    self._global_mesh = global_mesh
    self._mesh_axes = mesh_axes
    assert len(device_buffers) == len(self._global_mesh.local_devices)
    self._local_shards = self._create_local_shards(device_buffers)

    ss = shard_shape(self._global_shape, self._global_mesh, self._mesh_axes)
    assert all(db.shape == ss for db in device_buffers), (
        f"Expected shard shape {ss} doesn't match the device buffer "
        f"shape {device_buffers[0].shape}")

  @property
  def shape(self) -> Shape:
    return self._global_shape

  # TODO(yashkatariya): Make this `create_shards` and create global_shards
  # Then source the local_shards and add the data to it.
  def _create_local_shards(
      self, device_buffers: Sequence[DeviceArray]) -> Sequence[Shard]:
    indices = shard_indices(self._global_shape, self._global_mesh,
                           self._mesh_axes)

    device_to_replica = {}
    index_to_replica = {}
    for device, index in indices.items():
      h_index = _HashableSlice(index)
      if h_index not in index_to_replica:
        index_to_replica[h_index] = 0
      else:
        index_to_replica[h_index] += 1
      device_to_replica[device] = index_to_replica[h_index]

    shards = []
    # device_buffers are always local to the process.
    for db in device_buffers:
      d = db.device()
      shards.append(Shard(d, indices[d], device_to_replica[d], db))
    return shards

  @property
  def local_shards(self) -> Sequence[Shard]:
    return self._local_shards

  @classmethod
  def from_callback(cls, global_shape: Shape, dtype, global_mesh: pxla.Mesh,
                    mesh_axes: MeshAxes,
                    data_callback: Callable[[Index], ArrayLike]):
    indices = shard_indices(global_shape, global_mesh, mesh_axes)
    dbs = [
        device_put(data_callback(indices[device]), device)
        for device in global_mesh.local_devices
    ]
    return cls(global_shape, dtype, global_mesh, mesh_axes, dbs)

  @classmethod
  def from_batched_callback(
      cls, global_shape: Shape, dtype, global_mesh: pxla.Mesh,
      mesh_axes: MeshAxes, data_callback: Callable[[Sequence[Index]], Sequence[ArrayLike]]):
    raise NotImplementedError("Not implemented yet.")

  @classmethod
  def from_batched_callback_with_devices(
      cls, global_shape: Shape, dtype, global_mesh: pxla.Mesh,
      mesh_axes: MeshAxes,
      data_callback: Callable[[Sequence[Tuple[Index, Tuple[Device]]]], Sequence[DeviceArray]]):
    raise NotImplementedError("Not implemented yet.")
