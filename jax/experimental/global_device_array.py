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
"""Implementation of GlobalDeviceArray."""

import dataclasses
import numpy as np
from typing import Callable, Sequence, Tuple, Union, Mapping, Optional, List, Dict, NamedTuple

from jax.experimental import maps
from jax import core
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax.interpreters import pxla, xla
from jax._src.util import prod, safe_zip
from jax._src.api import device_put
from jax.tree_util import tree_flatten
from jax.interpreters.sharded_jit import PartitionSpec

Shape = Tuple[int, ...]
MeshAxes = Sequence[Union[str, Tuple[str], None]]
DeviceArray = xc.Buffer
Device = xc.Device
ArrayLike = Union[np.ndarray, DeviceArray]
Index = Tuple[slice, ...]


def _canonicalize_mesh_axes(mesh_axes):
  if not isinstance(mesh_axes, PartitionSpec):
    pspec = PartitionSpec(*mesh_axes)
  else:
    pspec = mesh_axes
  return pspec

def _get_indices(global_shape: Shape, global_mesh: pxla.Mesh,
                 mesh_axes: MeshAxes) -> Tuple[pxla.Index, ...]:
  # Import here to avoid cyclic import error when importing gda in pjit.py.
  from jax.experimental.pjit import get_array_mapping, _prepare_axis_resources

  pspec = _canonicalize_mesh_axes(mesh_axes)
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
  return indices


def get_shard_indices(global_shape: Shape, global_mesh: pxla.Mesh,
                      mesh_axes: MeshAxes) -> Mapping[Device, Index]:
  indices = _get_indices(global_shape, global_mesh, mesh_axes)
  # The type: ignore is to ignore the type returned by `spec_to_indices`.
  return dict(
      (d, i)
      for d, i in safe_zip(global_mesh.devices.flat, indices))  # type: ignore


def _calc_replica_ids(global_mesh: pxla.Mesh, mesh_axes: MeshAxes):
  pspec = _canonicalize_mesh_axes(mesh_axes)
  mesh_values = list(global_mesh.shape.values())
  flattened_pspec, _ = tree_flatten(tuple(pspec))
  # Get the location (coordinates) of each device in the device mesh.
  device_location = np.array(np.unravel_index(
      [d.id for d in global_mesh.devices.flat], mesh_values))
  # Find all the axes that were replicated.
  # If mesh_axes = (('x', 'y'), None, 'z') and ('x', 'y', 'z') were the mesh's
  # axis, then replicated axes will be None since all axes are being used to
  # shard the input.
  replicated_axis = np.isin(list(global_mesh.shape.keys()), flattened_pspec,
                            invert=True)
  # If all elements in replicated_axis are False then the input is fully sharded
  # so replica ids should be all 0s.
  if not any(replicated_axis):
    return [0] * global_mesh.devices.size
  else:
    # Drop all the sharded axes and find the location of coordinates in a linear
    # array.
    return np.ravel_multi_index(device_location[replicated_axis],
                                np.array(mesh_values)[replicated_axis])


def get_shard_indices_replica_ids(
    global_shape: Shape, global_mesh: pxla.Mesh,
    mesh_axes: MeshAxes) -> Mapping[Device, Tuple[Index, int]]:
  indices = _get_indices(global_shape, global_mesh, mesh_axes)
  replica_ids = _calc_replica_ids(global_mesh, mesh_axes)
  return dict((d, (i, r))
              for d, i, r in safe_zip(global_mesh.devices.flat, indices, replica_ids))


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


_hashed_index = lambda x: hash(tuple((v.start, v.stop) for v in x))


@dataclasses.dataclass(frozen=True)
class Shard:
  """A single data shard of a GlobalDeviceArray.

  Attributes:
    device: Which device this shard resides on.
    index: The index into the global array of this shard.
    replica_id: Integer id indicating which replica of the global array this
      shard is part of. Always `0` for fully sharded data
      (i.e. when there’s only 1 replica).
    data: The data of this shard. None if `device` is non-local.
  """
  device: Device
  index: Index
  replica_id: int
  # None if this `Shard` lives on a non-local device.
  data: Optional[DeviceArray] = None


class _GdaFastPathArgs(NamedTuple):
  local_indices_replica_ids: Mapping[Device, Tuple[Index, int]]
  local_devices: Sequence[Device]


class GlobalDeviceArray:
  """A logical array with data sharded across multiple devices and processes.

  If you’re not already familiar with JAX’s multi-process programming model,
  please read https://jax.readthedocs.io/en/latest/multi_process.html.

  A GlobalDeviceArray (GDA) can be thought of as a view into a single logical
  array sharded across processes. The logical array is the “global” array, and
  each process has a GlobalDeviceArray object referring to the same global array
  (similarly to how each process runs a multi-process pmap or pjit). Each process
  can access the shape, dtype, etc. of the global array via the GDA, pass the
  GDA into multi-process pjits, and get GDAs as pjit outputs (coming soon: xmap
  and pmap). However, each process can only directly access the shards of the
  global array data stored on its local devices.

  GDAs can help manage the inputs and outputs of multi-process computations.
  A GDA keeps track of which shard of the global array belongs to which device,
  and provides callback-based APIs to materialize the correct shard of the data
  needed for each local device of each process.

  A GDA consists of data shards. Each shard is stored on a different device.
  There are local shards and global shards. Local shards are those on local
  devices, and the data is visible to the current process. Global shards are
  those across all devices (including local devices), and the data isn’t visible
  if the shard is on a non-local device with respect to the current process.
  Please see the `Shard` class to see what information is stored inside that
  data structure.

  Note: to make pjit output GlobalDeviceArrays, set the environment variable
  `JAX_PARALLEL_FUNCTIONS_OUTPUT_GDA=true` or add the following to your code:
  `jax.config.update('jax_parallel_functions_output_gda', True)`

  Attributes:
    shape: The global shape of the array.
    dtype: dtype of the global array.
    local_shards: List of `Shard`s on the local devices of the current process.
      Data is available for all local shards.
    global_shards: List of all `Shard`s of the global array. Data isn’t
      available if a shard is on a non-local device with respect to the current
      process.

  Example:

  ```python
  # Logical mesh is (hosts, devices)
  assert global_mesh.shape == {'x': 4, 'y': 8}

  global_input_shape = (64, 32)
  mesh_axes = P('x', 'y')

  # Dummy example data; in practice we wouldn't necessarily materialize global data
  # in a single process.
  global_input_data = np.arange(
      np.prod(global_input_shape)).reshape(global_input_shape)

  def get_local_data_slice(index):
    # index will be a tuple of slice objects, e.g. (slice(0, 16), slice(0, 4))
    # This method will be called per-local device from the GSDA constructor.
    return global_input_data[index]

  gda = GlobalDeviceArray.from_callback(
          global_input_shape, global_mesh, mesh_axes, get_local_data_slice)

  f = pjit(lambda x: x @ x.T, out_axis_resources = P('y', 'x'))

  with mesh(global_mesh.shape, global_mesh.axis_names):
    out = f(gda)

  print(type(out))  # GlobalDeviceArray
  print(out.shape)  # global shape == (64, 64)
  print(out.local_shards[0].data)  # Access the data on a single local device,
                                   # e.g. for checkpointing
  print(out.local_shards[0].data.shape)  # per-device shape == (8, 16)
  print(out.local_shards[0].index) # Numpy-style index into the global array that
                                   # this data shard corresponds to

  # `out` can be passed to another pjit call, out.local_shards can be used to
  # export the data to non-jax systems (e.g. for checkpointing or logging), etc.
  ```
  """

  def __init__(self, global_shape: Shape, global_mesh: pxla.Mesh,
               mesh_axes: MeshAxes, device_buffers: Sequence[DeviceArray],
               _gda_fast_path_args: Optional[_GdaFastPathArgs] = None):
    """Constructor of GlobalDeviceArray class.

    Args:
      global_shape: The global shape of the array
      global_mesh: The global mesh representing devices across multiple
        processes.
      mesh_axes: A sequence with length less than or equal to the rank of the
      global array (i.e. the length of the global shape). Each element can be:
        * An axis name of `global_mesh`, indicating that the corresponding
          global array axis is partitioned across the given device axis of
          `global_mesh`.
        * A tuple of axis names of `global_mesh`. This is like the above option
          except the global array axis is partitioned across the product of axes
          named in the tuple.
        * None indicating that the corresponding global array axis is not
          partitioned.
        For more information, please see:
        https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html#more-information-on-partitionspec
      device_buffers: DeviceArrays that are on the local devices of
        `global_mesh`.
    """
    self._global_shape = global_shape
    self._global_mesh = global_mesh
    self._mesh_axes = mesh_axes
    self._device_buffers = device_buffers
    # Optionally precomputed for performance.
    self._gda_fast_path_args = _gda_fast_path_args
    self._current_process = xb.process_index()

    if self._gda_fast_path_args is None:
      self._local_devices = self._global_mesh.local_devices
    else:
      self._local_devices = self._gda_fast_path_args.local_devices
    assert len(device_buffers) == len(self._local_devices)

    self._local_shards = self._create_local_shards()

    ss = get_shard_shape(self._global_shape, self._global_mesh, self._mesh_axes)
    assert all(db.shape == ss for db in device_buffers), (
        f"Expected shard shape {ss} doesn't match the device buffer "
        f"shape {device_buffers[0].shape}")

    dtype = device_buffers[0].dtype
    assert all(db.dtype == dtype for db in device_buffers), (
        "Input arrays to GlobalDeviceArray must have matching dtypes, "
        f"got: {[db.dtype for db in device_buffers]}")
    self.dtype = dtype

  def __str__(self):
    return f'GlobalDeviceArray(shape={self.shape}, dtype={self.dtype})'

  def __repr__(self):
    return (f'GlobalDeviceArray(shape={self.shape}, dtype={self.dtype}, '
            f'global_mesh_shape={dict(self._global_mesh.shape)}, '
            f'mesh_axes={self._mesh_axes})')

  @property
  def shape(self) -> Shape:
    return self._global_shape

  @property
  def is_fully_replicated(self) -> bool:
    return self.shape == self.local_data(0).shape

  def _create_local_shards(self) -> Sequence[Shard]:
    if self._gda_fast_path_args is not None:
      local_idx_rid = self._gda_fast_path_args.local_indices_replica_ids
    else:
      global_indices_rid = get_shard_indices_replica_ids(
        self._global_shape, self._global_mesh, self._mesh_axes)
      local_idx_rid = dict((d, global_indices_rid[d])
                           for d in self._local_devices)
    device_to_buffer = dict((db.device(), db) for db in self._device_buffers)
    return [
        Shard(d, index, rid, device_to_buffer[d])
        for d, (index, rid) in local_idx_rid.items()
    ]

  @pxla.maybe_cached_property
  def local_shards(self) -> Sequence[Shard]:
    """Returns local shards belonging to the current process.

    The data will is always materialized for local shards.
    """
    for s in self._local_shards:
      # Ignore the type because mypy thinks data is None but local_shards
      # cannot have data=None which is checked in `_create_local_shards`.
      if s.data.aval is None:  # type: ignore
        s.data.aval = core.ShapedArray(s.data.shape, s.data.dtype)  # type: ignore
    return self._local_shards

  @pxla.maybe_cached_property
  def global_shards(self) -> Sequence[Shard]:
    """Returns global shards across all processes.

    The data will be None for non-local devices with respect to the current
    process.
    """
    # Populating global_shards lazily (i.e. when requested) because populating
    # sthem eagerly leads to a performance regression when training on large
    # models.
    # Also as this a cached property, once calculated, it should be cached. So
    # multiple accesses should be cheap.
    global_indices_rid = get_shard_indices_replica_ids(
        self._global_shape, self._global_mesh, self._mesh_axes)
    device_to_buffer = dict((db.device(), db) for db in self._device_buffers)
    global_shards = []
    for device, (index, rid) in global_indices_rid.items():
      local_shard = device.process_index == self._current_process
      buf = device_to_buffer[device] if local_shard else None
      if buf is not None and buf.aval is None:
        buf.aval = core.ShapedArray(buf.shape, buf.dtype)
      sh = Shard(device, index, rid, buf)
      global_shards.append(sh)
    return global_shards

  def local_data(self, index) -> DeviceArray:
    return self.local_shards[index].data

  @classmethod
  def from_callback(cls, global_shape: Shape, global_mesh: pxla.Mesh,
                    mesh_axes: MeshAxes, data_callback: Callable[[Index],
                                                                 ArrayLike]):
    global_indices_rid = get_shard_indices_replica_ids(
        global_shape, global_mesh, mesh_axes)
    local_devices = global_mesh.local_devices
    dbs = [
        device_put(data_callback(global_indices_rid[device][0]), device)
        for device in local_devices
    ]
    local_idx_rid = dict((d, global_indices_rid[d]) for d in local_devices)
    return cls(global_shape, global_mesh, mesh_axes, dbs,
               _gda_fast_path_args=_GdaFastPathArgs(local_idx_rid, local_devices))

  @classmethod
  def from_batched_callback(cls, global_shape: Shape,
                            global_mesh: pxla.Mesh, mesh_axes: MeshAxes,
                            data_callback: Callable[[Sequence[Index]],
                                                    Sequence[ArrayLike]]):
    global_indices_rid = get_shard_indices_replica_ids(
        global_shape, global_mesh, mesh_axes)
    local_devices = global_mesh.local_devices
    local_indices = [global_indices_rid[d][0] for d in local_devices]
    local_arrays = data_callback(local_indices)
    dbs = pxla.device_put(local_arrays, local_devices)
    local_idx_rid = dict((d, global_indices_rid[d]) for d in local_devices)
    return cls(global_shape, global_mesh, mesh_axes, dbs,
               _gda_fast_path_args=_GdaFastPathArgs(local_idx_rid, local_devices))

  @classmethod
  def from_batched_callback_with_devices(
      cls, global_shape: Shape, global_mesh: pxla.Mesh,
      mesh_axes: MeshAxes,
      data_callback: Callable[[Sequence[Tuple[Index, Tuple[Device, ...]]]],
                              Sequence[DeviceArray]]):
    global_indices_rid = get_shard_indices_replica_ids(
        global_shape, global_mesh, mesh_axes)
    local_devices = global_mesh.local_devices

    index_to_device: Dict[int, Tuple[Index, List[Device]]] = {}
    for device in local_devices:
      index = global_indices_rid[device][0]
      h_index = _hashed_index(index)
      if h_index not in index_to_device:
        index_to_device[h_index] = (index, [device])
      else:
        index_to_device[h_index][1].append(device)

    cb_inp = [
        (index, tuple(devices)) for index, devices in index_to_device.values()
    ]
    dbs = data_callback(cb_inp)
    local_idx_rid = dict((d, global_indices_rid[d]) for d in local_devices)
    return cls(global_shape, global_mesh, mesh_axes, dbs,
               _gda_fast_path_args=_GdaFastPathArgs(local_idx_rid, local_devices))


core.pytype_aval_mappings[GlobalDeviceArray] = lambda x: core.ShapedArray(
    x.shape, x.dtype)
xla.pytype_aval_mappings[GlobalDeviceArray] = lambda x: core.ShapedArray(
    x.shape, x.dtype)
xla.canonicalize_dtype_handlers[GlobalDeviceArray] = pxla.identity

def _gda_shard_arg(x, devices, indices):
  pjit_mesh = maps.thread_resources.env.physical_mesh
  if x._global_mesh != pjit_mesh:
    raise ValueError("Pjit's mesh and GDA's mesh should be equal. Got Pjit "
                     f"mesh: {pjit_mesh},\n GDA mesh: {x._global_mesh}")
  return [s.data for s in x.local_shards]
pxla.shard_arg_handlers[GlobalDeviceArray] = _gda_shard_arg


def _gda_array_result_handler(global_aval, out_axis_resources, global_mesh):
  global_idx_rid = get_shard_indices_replica_ids(global_aval.shape, global_mesh,
                                                 out_axis_resources)
  local_devices = global_mesh.local_devices
  local_idx_rid = dict((d, global_idx_rid[d]) for d in local_devices)
  fast_path_args = _GdaFastPathArgs(local_idx_rid, local_devices)
  return lambda bufs: GlobalDeviceArray(
      global_aval.shape, global_mesh, out_axis_resources, bufs, fast_path_args)
pxla.global_result_handlers[core.ShapedArray] = _gda_array_result_handler
pxla.global_result_handlers[core.ConcreteArray] = _gda_array_result_handler
