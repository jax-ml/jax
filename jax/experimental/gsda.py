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

import asyncio
from collections import defaultdict, Counter
import dataclasses
import numpy as np
from typing import Callable, Sequence, Tuple, Union, Mapping, Optional, List, Dict

from . import maps
from .. import core
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
from ..interpreters import pxla, xla
from .._src.util import prod, safe_zip
from .._src.api import device_put
from ..interpreters.sharded_jit import PartitionSpec

Shape = Tuple[int, ...]
MeshAxes = Sequence[Union[str, Tuple[str], None]]
DeviceArray = xc.Buffer
Device = xc.Device
ArrayLike = Union[np.ndarray, DeviceArray]
Index = Tuple[slice, ...]


@dataclasses.dataclass(frozen=True)
class _HashableIndex:
  val: Index

  def __hash__(self):
    return hash(tuple([(v.start, v.stop, v.step) for v in self.val]))

  def __eq__(self, other):
    return self.val == other.val


def get_shard_indices(global_shape: Shape, global_mesh: pxla.Mesh,
                      mesh_axes: MeshAxes) -> Mapping[Device, Index]:
  # Import here to avoid cyclic import error when importing gsda in pjit.py.
  from .pjit import get_array_mapping, _prepare_axis_resources

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
  return dict(
      (d, i)
      for d, i in safe_zip(global_mesh.devices.flat, indices))  # type: ignore


def get_shard_shape(global_shape, global_mesh, mesh_axes) -> Shape:
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
  # None if this `Shard` lives on a non-local device.
  data: Optional[DeviceArray] = None


class GlobalShardedDeviceArray:

  def __init__(self, global_shape: Shape, global_mesh: pxla.Mesh,
               mesh_axes: MeshAxes, device_buffers: Sequence[DeviceArray]):
    self._global_shape = global_shape
    self._global_mesh = global_mesh
    self._mesh_axes = mesh_axes
    assert len(device_buffers) == len(self._global_mesh.local_devices)
    self._global_shards, self._local_shards = self._create_shards(
        device_buffers)

    ss = get_shard_shape(self._global_shape, self._global_mesh, self._mesh_axes)
    assert all(db.shape == ss for db in device_buffers), (
        f"Expected shard shape {ss} doesn't match the device buffer "
        f"shape {device_buffers[0].shape}")

    dtype = device_buffers[0].dtype
    assert all(db.dtype == dtype for db in device_buffers), (
        "Input arrays to GlobalShardedDeviceArray must have matching dtypes, "
        f"got: {[db.dtype for db in device_buffers]}")
    self.dtype = dtype

  @property
  def shape(self) -> Shape:
    return self._global_shape

  def _create_shards(
      self, device_buffers: Sequence[DeviceArray]
  ) -> Tuple[Sequence[Shard], Sequence[Shard]]:
    indices = get_shard_indices(self._global_shape, self._global_mesh,
                                self._mesh_axes)
    device_to_buffer = dict((db.device(), db) for db in device_buffers)
    gs, ls = [], []
    index_to_replica: Dict[_HashableIndex, int] = Counter()
    for device, index in indices.items():
      h_index = _HashableIndex(index)
      replica_id = index_to_replica[h_index]
      index_to_replica[h_index] += 1
      local_shard = device.process_index == xb.process_index()
      buf = device_to_buffer[device] if local_shard else None
      sh = Shard(device, index, replica_id, buf)
      gs.append(sh)
      if local_shard:
        if sh.data is None:
          raise ValueError("Local shard's data field should not be None.")
        ls.append(sh)
    return gs, ls

  @property
  def local_shards(self) -> Sequence[Shard]:
    return self._local_shards

  @property
  def global_shards(self) -> Sequence[Shard]:
    return self._global_shards

  @classmethod
  def from_callback(cls, global_shape: Shape, global_mesh: pxla.Mesh,
                    mesh_axes: MeshAxes, data_callback: Callable[[Index],
                                                                 ArrayLike]):
    indices = get_shard_indices(global_shape, global_mesh, mesh_axes)
    dbs = [
        device_put(data_callback(indices[device]), device)
        for device in global_mesh.local_devices
    ]
    return cls(global_shape, global_mesh, mesh_axes, dbs)

  @classmethod
  async def from_async_callback(cls, global_shape: Shape,
                                global_mesh: pxla.Mesh, mesh_axes: MeshAxes,
                                data_callback: Callable[[Index], asyncio.Future]):
    indices = get_shard_indices(global_shape, global_mesh, mesh_axes)
    future_arrays = [data_callback(indices[d])
                    for d in global_mesh.local_devices]
    # Pause here and come back to `from_async_callback()` when future_arrays are
    # ready.
    local_arrays = await asyncio.gather(*future_arrays)
    dbs = pxla.device_put(local_arrays, global_mesh.local_devices)
    return cls(global_shape, global_mesh, mesh_axes, dbs)

  @classmethod
  def from_batched_callback(cls, global_shape: Shape,
                            global_mesh: pxla.Mesh, mesh_axes: MeshAxes,
                            data_callback: Callable[[Sequence[Index]],
                                                    Sequence[ArrayLike]]):
    indices = get_shard_indices(global_shape, global_mesh, mesh_axes)
    local_indices = [indices[d] for d in global_mesh.local_devices]
    local_arrays = data_callback(local_indices)
    dbs = pxla.device_put(local_arrays, global_mesh.local_devices)
    return cls(global_shape, global_mesh, mesh_axes, dbs)

  @classmethod
  def from_batched_callback_with_devices(
      cls, global_shape: Shape, global_mesh: pxla.Mesh,
      mesh_axes: MeshAxes,
      data_callback: Callable[[Sequence[Tuple[Index, Tuple[Device, ...]]]],
                              Sequence[DeviceArray]]):
    indices = get_shard_indices(global_shape, global_mesh, mesh_axes)

    index_to_device: Mapping[_HashableIndex, List[Device]] = defaultdict(list)
    for device in global_mesh.local_devices:
      h_index = _HashableIndex(indices[device])
      index_to_device[h_index].append(device)

    cb_inp = [
        (index.val, tuple(device)) for index, device in index_to_device.items()
    ]
    dbs = data_callback(cb_inp)
    return cls(global_shape, global_mesh, mesh_axes, dbs)


core.pytype_aval_mappings[GlobalShardedDeviceArray] = lambda x: core.ShapedArray(x.shape, x.dtype)
xla.pytype_aval_mappings[GlobalShardedDeviceArray] = lambda x: core.ShapedArray(x.shape, x.dtype)
xla.canonicalize_dtype_handlers[GlobalShardedDeviceArray] = pxla.identity

def _gsda_shard_arg(x, devices, indices):
  pjit_mesh = maps.thread_resources.env.physical_mesh
  if x._global_mesh != pjit_mesh:
    raise ValueError("Pjit's mesh and GSDA's mesh should be equal. Got Pjit "
                     f"mesh: {pjit_mesh},\n GSDA mesh: {x._global_mesh}")
  return [s.data for s in x.local_shards]

pxla.shard_arg_handlers[GlobalShardedDeviceArray] = _gsda_shard_arg
