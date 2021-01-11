# Copyright 2018 Google LLC
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
"""Implementation of pmap and related functionality."""

# A ShardingSpec describes at a high level how a logical array is sharded across
# devices (each ShardedDeviceArray has a ShardingSpec, and ShardingSpecs also
# describe how to shard inputs to a parallel computation). spec_to_indices()
# encodes exactly how a given ShardingSpec is translated to device buffers, i.e.
# how the sharded array is "laid out" across devices. Given a sequence of
# devices, we shard the data across the devices in row-major order, with
# replication treated as an extra inner dimension.
#
# For example, given the logical data array [1, 2, 3, 4], if we were to
# partition this array 4 ways with a replication factor of 2, for a total of 8
# devices, the data on each device would be: [1, 1], [2, 2], [3, 3], [4, 4].
#
# This encoding is assumed by various parts of the system, e.g. generating
# replica groups for collective operations.

import sys
from contextlib import contextmanager
from collections import defaultdict, OrderedDict
import itertools as it
import operator as op
import threading
from typing import (Any, Callable, Dict, List, Optional, Sequence, Set, Tuple,
                    Type, Union, Iterable, no_type_check, NamedTuple, TYPE_CHECKING)

from absl import logging
import numpy as np

from ..config import flags, config
from .. import core
from .. import linear_util as lu
from .. import lazy
from ..abstract_arrays import array_types
from ..core import ConcreteArray, ShapedArray, Var, Literal
from .._src.util import (partial, unzip2, unzip3, prod, safe_map, safe_zip,
                         extend_name_stack, wrap_name, assert_unreachable,
                         tuple_insert, tuple_delete, taggedtuple, curry)
from ..lib import xla_bridge as xb
from ..lib import xla_client as xc
from ..tree_util import tree_flatten, tree_map
from .batching import broadcast, not_mapped, moveaxis
from . import batching
from . import partial_eval as pe
from . import xla
from . import ad

if sys.version_info >= (3, 9):
  OrderedDictType = OrderedDict
else:
  OrderedDictType = Dict

xops = xc.ops

FLAGS = flags.FLAGS

unsafe_map, map = map, safe_map  # type: ignore

Index = Union[int, slice, Tuple[Union[int, slice], ...]]


# mypy is very unhappy about taggedtuple
if TYPE_CHECKING:
  class Unstacked(NamedTuple):
    size: int
else:
  Unstacked = taggedtuple('Unstacked', ('size',))

class Chunked:
  chunks: Tuple[int, ...]

  def __init__(self, chunks: Union[int, Tuple[int, ...]]):
    if not isinstance(chunks, tuple):
      chunks = (chunks,)
    object.__setattr__(self, 'chunks', chunks)

  def __setattr__(self, name, value):
    raise RuntimeError("Chunked is immutable")

  def __delattr__(self, name):
    raise RuntimeError("Chunked is immutable")

  def __hash__(self):
    return hash(self.chunks)

  def __eq__(self, other):
    return type(other) is Chunked and self.chunks == other.chunks

  def __repr__(self):
    return f'Chunked({self.chunks})'

"""
Represents all the ways we can shard a dimension.
- `None` means no sharding;
- `Chunked` means that the dimension is split into the specified number of chunks,
  but the split dimension itself is preserved inside the map;
- `Unstacked` means that the dimension is split into chunks of size 1, and doesn't
  appear inside the map.
"""
AvalDimSharding = Union[Unstacked, Chunked, None]

# mypy is very unhappy about taggedtuple
if TYPE_CHECKING:
  class ShardedAxis(NamedTuple):
    axis: int
  class Replicated(NamedTuple):
    replicas: int
else:
  ShardedAxis = taggedtuple('ShardedAxis', ('axis',))
  Replicated = taggedtuple('Replicated', ('replicas',))

"""
Assigns sharded axes to mesh dimensions.

When no axis is assigned, the data is replicated.
Note that `ShardedAxis(2)` refers to the second actually sharded axis (i.e.
counting as if the None dimensions of sharding were filtered out). For example,
given the sharding `[Unstacked(n), None, Chunked(m)]`, an entry of `ShardedAxis(1)`
refers to the `Chunked(m)` axis, not the `None`.
"""
MeshDimAssignment = Union[ShardedAxis, Replicated]

class ShardingSpec:
  """Describes the sharding of an ndarray.

  `sharding` specifies how the array is supposed to get partitioned into chunks.
  Its length should match the rank of the array. See the docstring of
  `AvalDimSharding` for the supported partitioning schemes.

  `mesh_mapping` describes an assignments of the array chunks created by `sharding`
  to a logical device mesh. The length of the tuple is equal to the rank of the mesh.
  Each mesh dimension can either get partitions of data varying along one of the
  sharded dimensions, or the data can be replicated. See the docstring of
  `MeshDimAssignment` for more information.
  """
  sharding: Tuple[AvalDimSharding, ...]
  mesh_mapping: Tuple[MeshDimAssignment, ...]

  def __init__(self,
               sharding: Iterable[AvalDimSharding],
               mesh_mapping: Iterable[MeshDimAssignment]):
    self.sharding = tuple(sharding)
    self.mesh_mapping = tuple(mesh_mapping)

  @property
  def mesh_shape(self):
    sharded_axis_sizes = []
    for sharding in self.sharding:
      if sharding is None:
        continue
      elif isinstance(sharding, Unstacked):
        sharded_axis_sizes.append(sharding.size)
      elif isinstance(sharding, Chunked):
        sharded_axis_sizes.extend(sharding.chunks)
      else:
        assert_unreachable(sharding)
    return tuple(sharded_axis_sizes[a.axis] if isinstance(a, ShardedAxis) else a.replicas
                 for a in self.mesh_mapping)

  def sharding_proto(self):
    """Converts a ShardingSpec to an OpSharding proto.

    See
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/xla_data.proto#L601
    for details on the OpSharding proto.
    Unfortunately the semantics are not very well described in the proto spec, but the code here might help:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/experimental/xla_sharding/xla_sharding.py
    """
    mesh_shape = self.mesh_shape
    mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)

    sharded_axes = {}  # maps sharded axis identifiers to mesh axis indices to which they're mapped
    replicated_maxes = []  # lists mesh axis identifiers to replicate over
    for maxis, assignment in enumerate(self.mesh_mapping):
      if isinstance(assignment, Replicated):
        replicated_maxes.append(maxis)
      elif isinstance(assignment, ShardedAxis):
        sharded_axes[assignment.axis] = maxis
      else:
        assert_unreachable(assignment)

    proto = xc.OpSharding()
    if len(replicated_maxes) == len(self.mesh_mapping):
      proto.type = xc.OpSharding.Type.REPLICATED
      return proto
    else:
      proto.type = xc.OpSharding.Type.OTHER

    mesh_permutation = []
    new_mesh_shape = []
    next_sharded_axis = 0
    for axis, sharding in enumerate(self.sharding):
      if sharding is None:
        new_mesh_shape.append(1)  # Add a dummy mesh axis we won't be sharding over
      elif isinstance(sharding, Chunked):
        for nchunks in sharding.chunks:
          maxis = sharded_axes[next_sharded_axis]
          assert mesh_shape[maxis] == nchunks
          mesh_permutation.append(maxis)
          next_sharded_axis += 1
        new_mesh_shape.append(int(np.prod(sharding.chunks)))
      elif isinstance(sharding, Unstacked):
        raise RuntimeError("Cannot convert unstacked sharding specs to XLA OpSharding")
      else:
        assert_unreachable(sharding)

    # Create the partial sharding proto if tensor is replicated over some mesh axes
    if replicated_maxes:
      new_mesh_shape.append(-1)
      mesh_permutation.extend(replicated_maxes)
      proto.replicate_on_last_tile_dim = True

    proto_mesh = mesh.transpose(mesh_permutation).reshape(new_mesh_shape)
    proto.tile_assignment_dimensions = list(proto_mesh.shape)
    proto.tile_assignment_devices = list(proto_mesh.flat)
    return proto

  def indices(self, shape: Tuple[int, ...]) -> np.ndarray:
    """Returns NumPy-style indices corresponding to a sharding spec.

    Args:
      shape: The shape of the logical array being sharded.

    Returns:
      An ndarray with the same shape as the logical mesh (as derived form
      `mesh_mapping`). Each entry is a NumPy-style index selecting the subset of
      the data array to be placed on a corresponding device. The indices can be
      ints, slice objects with step=1, or tuples of those.
    """
    assert len(shape) == len(self.sharding), (shape, self.sharding)

    axis_indices: List[Sequence[Index]] = []
    shard_indices_shape = []
    for dim, sharding in enumerate(self.sharding):
      axis_size = shape[dim]
      if sharding is None:
        axis_indices.append([slice(None)])
        # NOTE: We don't append unsharded dimensions to shard_indices_shape here,
        #       because they do not appear in the mesh mapping.
      elif isinstance(sharding, Unstacked):
        assert axis_size == sharding.size, f'{axis_size} != {sharding.size}'
        axis_indices.append(range(axis_size))
        shard_indices_shape.append(axis_size)
      elif isinstance(sharding, Chunked):
        total_chunks = int(np.prod(sharding.chunks))
        shard_size, ragged = divmod(axis_size, total_chunks)
        assert not ragged, (axis_size, total_chunks, dim)
        axis_indices.append([slice(i * shard_size, (i + 1) * shard_size)
                             for i in range(total_chunks)])
        shard_indices_shape.extend(sharding.chunks)
      else:
        assert_unreachable(sharding)

    # shard_indices is an ndarray representing the sharded axes of the logical array,
    # with each dimension having size equal to the number of shards across the corresponding
    # logical array dimension, and each element containing the multi-dimensional index that
    # is used to extract the corresponding shard of the logical array.
    shard_indices = np.empty([prod(shard_indices_shape)], dtype=np.object)
    for i, idxs in enumerate(it.product(*axis_indices)):
      shard_indices[i] = idxs
    shard_indices = shard_indices.reshape(shard_indices_shape)

    # Ensure that each sharded axis is used exactly once in the mesh mapping
    num_sharded_dim = len(shard_indices_shape)
    sharded_dim_perm = [a.axis for a in self.mesh_mapping if isinstance(a, ShardedAxis)]
    assert (set(sharded_dim_perm) == set(range(num_sharded_dim)) and
            len(sharded_dim_perm) == num_sharded_dim)
    # Replicate/reorder the indices according to the mesh mapping
    replica_sizes = tuple(a.replicas for a in self.mesh_mapping if isinstance(a, Replicated))
    replica_dim, sharded_dim = it.count(0), iter(sharded_dim_perm)
    perm = [next(replica_dim) if isinstance(a, Replicated) else
            len(replica_sizes) + next(sharded_dim)
            for a in self.mesh_mapping]
    return (np.broadcast_to(shard_indices, replica_sizes + shard_indices.shape)
              .transpose(perm))

  def __eq__(self, other):
    return (self.sharding, self.mesh_mapping) == (other.sharding, other.mesh_mapping)

  def __hash__(self):
    return hash((self.sharding, self.mesh_mapping))

  def __repr__(self):
    return f'ShardingSpec({self.sharding}, {self.mesh_mapping})'

def spec_to_indices(shape: Tuple[int, ...],
                    spec: ShardingSpec) -> Tuple[Index, ...]:
  """Returns numpy-style indices corresponding to a sharding spec.

  Each index describes a shard of the array. The order of the indices is the
  same as the device_buffers of a ShardedDeviceArray (i.e. the data is laid out
  row-major).

  Args:
    shape: The shape of the logical array being sharded.
    spec: Describes how the array is sharded and how the shards are assigned to
          the logical mesh.

  Returns:
    A tuple of length equal to the size of the mesh (inferred as the product of
    sharded dimension sizes and all replication factors).  Each element is an
    int, a slice object with step=1, or a tuple thereof, to be treated as an
    index into the full logical array.
  """
  return tuple(spec.indices(shape).flat)


### util

def identity(x): return x

# TODO(skye): expose PyLocalBuffers in xla_client
def shard_args(devices: Sequence[xb.xla_client.Device],
               indices: Sequence[Sequence[Index]],
               args) -> Sequence[Sequence[xb.xla_client._xla.PyLocalBuffer]]:
  """Shard each argument data array along its leading axis.

  Args:
    devices: sequence of Devices mapping replica index to a physical device.
    indices: sequence of the same length as `args` describing how each arg
      should be sharded/replicated across `devices`. Each element in `indices`
      is the same length as `devices`.
    args: a sequence of JaxTypes representing arguments to be sharded according
      to `indices` and placed on `devices`.

  Returns:
    A list of device buffers with the same length as `devices` indexed by
    replica number, so that the nth element is the argument to be passed to the
    nth replica.
  """
  nargs, nrep = len(args), len(devices)
  buffers = [[None] * nargs for _ in range(nrep)]
  for a, arg in enumerate(args):
    # The shard_arg_handlers allow an extensible set of types to be sharded, but
    # inline handling for ShardedDeviceArray as a special case for performance
    # NOTE: we compare indices instead of sharding_spec because
    # pmap_benchmark.pmap_shard_args_benchmark indicates this is faster.
    if type(arg) is ShardedDeviceArray and indices[a] == arg.indices:
      for r, buf in enumerate(arg.device_buffers):
        buffers[r][a] = (buf if buf.device() == devices[r]
                         else buf.copy_to_device(devices[r]))
    else:
      arg = xla.canonicalize_dtype(arg)
      bufs = shard_arg_handlers[type(arg)](arg, devices, indices[a])
      for r, buf in enumerate(bufs):
        buffers[r][a] = buf

  return buffers


shard_arg_handlers: Dict[Any, Callable[[Any, Any, Any], Sequence[Any]]] = {}
shard_arg_handlers[core.Unit] = \
    lambda x, devices, _: device_put(core.unit, devices, replicate=True)
def _shard_array(x, devices, indices):
  return device_put([x[i] for i in indices], devices)
for _t in array_types:
  shard_arg_handlers[_t] = _shard_array

def _shard_device_array(x, devices, indices):
  start_indices, limit_indices, removed_dims = map(tuple, unzip3(
      _as_slice_indices(x, idx) for idx in indices))
  shards = x._multi_slice(start_indices, limit_indices, removed_dims)
  return device_put(shards, devices)
shard_arg_handlers[xla._DeviceArray] = _shard_device_array
shard_arg_handlers[xla._CppDeviceArray] = _shard_device_array


# NOTE(skye): we could refactor to generate _multi_slice parameters directly
# from the input ShardingSpec, rather than the indices. However, this would
# require duplicating the ordering logic of spec_to_indices, which is more
# subtle and more likely to change than the index logic we have to support here.
def _as_slice_indices(arr: xla.DeviceArrayProtocol, idx: Index) -> Tuple[
    Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
  """Returns start_indices, limit_indices, removed_dims"""
  start_indices = [0] * arr.ndim
  limit_indices = list(arr.shape)
  removed_dims = []

  tuple_idx = idx if isinstance(idx, tuple) else (idx,)
  for dim, sub_idx in enumerate(tuple_idx):
    if isinstance(sub_idx, int):
      start_indices[dim] = sub_idx
      limit_indices[dim] = sub_idx + 1
      removed_dims.append(dim)
    elif sub_idx == slice(None):
      continue
    else:
      assert isinstance(sub_idx, slice), sub_idx
      assert isinstance(sub_idx.start, int), sub_idx
      assert isinstance(sub_idx.stop, int), sub_idx
      start_indices[dim] = sub_idx.start
      limit_indices[dim] = sub_idx.stop

  return tuple(start_indices), tuple(limit_indices), tuple(removed_dims) # type: ignore


def shard_aval(size, axis: int, aval):
  try:
    return shard_aval_handlers[type(aval)](size, axis, aval)
  except KeyError as err:
    raise TypeError(f"No shard_aval handler for type: {type(aval)}") from err
shard_aval_handlers: Dict[Type[core.AbstractValue], Callable[[int, int, Any], Any]] = {}
shard_aval_handlers[core.AbstractUnit] = lambda size, axis, x: x
def _shard_abstract_array(size, axis: int, x):
  try:
    if x.shape[axis] != size:
      raise ValueError(f"Axis size {size} does not match dimension {axis} of "
                       f"shape {x.shape}")
  except IndexError:
    raise ValueError("Cannot split a {x.dim}D value along axis {axis}") from None
  return ShapedArray(tuple_delete(x.shape, axis), x.dtype)
shard_aval_handlers[ShapedArray] = _shard_abstract_array

# TODO(skye): expose PyLocalBuffers in xla_client
def aval_to_result_handler(sharding_spec: Optional[ShardingSpec],
                           indices: Optional[Tuple[Index]],
                           aval: core.AbstractValue) -> Callable[
                               [List[xb.xla_client._xla.PyLocalBuffer]], Any]:
  """Returns a function for handling the raw buffers of a single output aval.

  Args:
    sharding_spec: indicates how the output is sharded across devices, or None
      for non-array avals.
    indices: the pre-computed result of spec_to_indices, or None for non-array
      avals.
    aval: the output AbstractValue.

  Returns:
    A function for handling the PyLocalBuffers that will eventually be produced
    for this output. The function will return an object suitable for returning
    to the user, e.g. a ShardedDeviceArray.
  """
  try:
    return pxla_result_handlers[type(aval)](sharding_spec, indices, aval)
  except KeyError as err:
    raise TypeError("No pxla_result_handler for type: {}".format(type(aval))
                    ) from err

PxlaResultHandler = Callable[..., Callable[[List[xb.xla_client._xla.PyLocalBuffer]], Any]]
pxla_result_handlers: Dict[Type[core.AbstractValue], PxlaResultHandler] = {}
pxla_result_handlers[core.AbstractUnit] = lambda *_: lambda _: core.unit
def array_result_handler(sharding_spec, indices, aval: ShapedArray):
  return lambda bufs: ShardedDeviceArray(aval, sharding_spec, bufs, indices)
pxla_result_handlers[ShapedArray] = array_result_handler
pxla_result_handlers[ConcreteArray] = array_result_handler


### lazy device-memory persistence and result handling

class ShardedDeviceArray(xla._DeviceArray):
  """A ShardedDeviceArray is an ndarray sharded across devices.

  The purpose of a ShardedDeviceArray is to reduce the number of transfers when
  executing replicated computations, by allowing results to persist on the
  devices that produced them. That way dispatching a similarly replicated
  computation that consumes the same sharded memory layout does not incur any
  transfers.

  A ShardedDeviceArray represents one logical ndarray value, and simulates the
  behavior of an ndarray so that it can be treated by user code as an ndarray;
  that is, it is only an optimization to reduce transfers.

  Attributes:
    aval: A ShapedArray indicating the shape and dtype of this array.
    sharding_spec: describes how this array is sharded across `device_buffers`.
    device_buffers: the buffers containing the data for this array. Each buffer
      is the same shape and on a different device. Buffers are in row-major
      order, with replication treated as an extra innermost dimension.
    indices: the result of spec_to_indices(sharding_spec). Can optionally be
      precomputed for efficiency. A list the same length as
      `device_buffers`. Each index indicates what portion of the full array is
      stored in the corresponding device buffer, i.e. `array[indices[i]] ==
      device_buffers[i].to_py()`.
  """
  __slots__ = ["device_buffers", "sharding_spec", "indices",
               "_one_replica_buffer_indices"]

  # TODO(skye): expose PyLocalBuffers in xla_client
  def __init__(self,
               aval: ShapedArray,
               sharding_spec, # TODO(skye): add type annotation back, see below
               device_buffers: List[xb.xla_client._xla.PyLocalBuffer] = None,
               indices: Optional[Tuple[Index, ...]] = None):
    xla.DeviceArray.__init__(self)

    # TODO(skye): this is temporary staging while we switch users over to
    # providing sharding_spec. It assumes that any pre-existing callers are
    # creating pmap-style ShardedDeviceArrays over the first dimension.
    if device_buffers is None:
      device_buffers = sharding_spec
      sharded_aval = ShapedArray(aval.shape[1:], aval.dtype)
      sharding_spec = _pmap_sharding_spec(aval.shape[0], aval.shape[0],
                                          1, None, sharded_aval, 0)

    # TODO(skye): assert invariants. Keep performance in mind though.
    if indices is None:
      indices = spec_to_indices(aval.shape, sharding_spec)
    self.aval = aval
    self.device_buffers = device_buffers
    self.sharding_spec = sharding_spec
    self.indices = indices
    self._npy_value = None
    self._one_replica_buffer_indices = None
    if not core.skip_checks:
      assert type(aval) is ShapedArray

  @property
  def one_replica_buffer_indices(self):
    """Indices of buffers containing one complete copy of the array data."""
    if self._one_replica_buffer_indices is None:
      one_replica_indices = []
      seen_index_hashes = set()
      for i, index in enumerate(self.indices):
        hashed_index = _hashable_index(index)
        if hashed_index not in seen_index_hashes:
          one_replica_indices.append(i)
          seen_index_hashes.add(hashed_index)
      self._one_replica_buffer_indices = one_replica_indices
    return self._one_replica_buffer_indices

  def copy_to_host_async(self):
    for buffer_index in self.one_replica_buffer_indices:
      self.device_buffers[buffer_index].copy_to_host_async()

  def delete(self):
    for buf in self.device_buffers:
      buf.delete()
    self.device_buffers = None
    self._npy_value = None

  def _check_if_deleted(self):
    if self.device_buffers is None:
      raise ValueError("ShardedDeviceArray has been deleted.")

  def block_until_ready(self):
    self._check_if_deleted()
    for buf in self.device_buffers:
      buf.block_host_until_ready()
    return self

  @property
  def _value(self):
    if self._npy_value is None:
      self.copy_to_host_async()
      npy_value = np.empty(self.aval.shape, self.aval.dtype)
      for i in self.one_replica_buffer_indices:
        npy_value[self.indices[i]] = self.device_buffers[i].to_py()
      self._npy_value = npy_value
    return self._npy_value

  def __getitem__(self, idx):
    if not isinstance(idx, tuple):
      cidx = (idx,) + (slice(None),) * (len(self.aval.shape) - 1)
    else:
      cidx = idx + (slice(None),) * (len(self.aval.shape) - len(idx))
    if self._npy_value is None:
      try:
        buf_idx = self.indices.index(cidx)
      except ValueError:
        buf_idx = None
      if buf_idx is not None:
        buf = self.device_buffers[buf_idx]
        # TODO(jblespiau): We can simply use buf.xla_shape() when version 0.1.58
        # is the default.
        aval = ShapedArray(
            getattr(buf, "xla_shape", buf.shape)().dimensions(),
            self.aval.dtype)
        return xla.make_device_array(aval, None, lazy.array(aval.shape), buf)
    return super(ShardedDeviceArray, self).__getitem__(idx)


def _hashable_index(idx):
  return tree_map(lambda x: (x.start, x.stop) if type(x) == slice else x,
                  idx)

# The fast path is handled directly in shard_args().
# TODO(skye): is there a simpler way to rewrite this using sharding_spec?
def _shard_sharded_device_array_slow_path(x, devices, indices):
  candidates = defaultdict(list)
  for buf, idx in safe_zip(x.device_buffers, x.indices):
    candidates[_hashable_index(idx)].append(buf)

  bufs = []
  for idx, device in safe_zip(indices, devices):
    # Look up all buffers that contain the correct slice of the logical array.
    candidates_list = candidates[_hashable_index(idx)]
    if not candidates_list:
      # This array isn't sharded correctly. Reshard it via host roundtrip.
      # TODO(skye): more efficient reshard?
      return shard_arg_handlers[type(x._value)](x._value, devices, indices)
    # Try to find a candidate buffer already on the correct device,
    # otherwise copy one of them.
    for buf in candidates_list:
      if buf.device() == device:
        bufs.append(buf)
        break
    else:
      bufs.append(buf.copy_to_device(device))
  return bufs
shard_arg_handlers[ShardedDeviceArray] = _shard_sharded_device_array_slow_path

def _sharded_device_array_constant_handler(c, val, canonicalize_types=True):
  return xb.constant(c, np.asarray(val), canonicalize_types=canonicalize_types)
xb.register_constant_handler(ShardedDeviceArray, _sharded_device_array_constant_handler)

core.pytype_aval_mappings[ShardedDeviceArray] = ConcreteArray
xla.device_put_handlers[ShardedDeviceArray] = xla._device_put_array
xla.pytype_aval_mappings[ShardedDeviceArray] = op.attrgetter('aval')
xla.canonicalize_dtype_handlers[ShardedDeviceArray] = identity


### the xla_pmap primitive and its rules are comparable to xla_call in xla.py

def xla_pmap_impl(fun: lu.WrappedFun, *args, backend, axis_name, axis_size,
                  global_axis_size, devices, name, in_axes, out_axes_thunk,
                  donated_invars, global_arg_shapes):
  abstract_args = unsafe_map(xla.abstractify, args)
  compiled_fun = parallel_callable(fun, backend, axis_name, axis_size,
                                   global_axis_size, devices, name,
                                   in_axes, out_axes_thunk,
                                   donated_invars, global_arg_shapes,
                                   *abstract_args)
  return compiled_fun(*args)

@lu.cache
def parallel_callable(fun: lu.WrappedFun,
                      backend_name: Optional[str],
                      axis_name,
                      axis_size: int,
                      global_axis_size: Optional[int],
                      devices: Optional[Sequence[Any]],
                      name: str,
                      in_axes: Iterable[Optional[int]],
                      out_axes_thunk: Callable[[], Sequence[Optional[int]]],
                      donated_invars: Iterable[bool],
                      global_arg_shapes,
                      *avals):
  if devices is not None and len(devices) == 0:
    raise ValueError("'devices' argument to pmap must be non-empty, or None.")

  # Determine global_axis_size for use in AxisEnv.
  # TODO(mattjj,skyewm): revive this check (inner_pmap always False now)
  # if xb.host_count() > 1 and global_axis_size is None and inner_pmap:
  #   raise ValueError("'axis_size' must be specified for nested multi-host pmaps")
  if (xb.host_count() == 1 and global_axis_size is not None and
      global_axis_size != axis_size):
    raise ValueError(
        f"Specified axis_size {global_axis_size} doesn't match received "
        f"axis_size {axis_size}.")

  must_run_on_all_devices = False
  no_nested_sharding = False
  if global_axis_size is None:
    if xb.host_count() == 1:
      global_axis_size = axis_size
    elif devices:
      # This allows each host in a multi-host pmap to run on a different number
      # of devices, but precludes nested sharding (i.e. inner pmaps or
      # sharded_jits).
      global_axis_size = len(devices)
      no_nested_sharding = True
    else:
      # This assumes all hosts run on the same number of devices. We make sure
      # this assumption is true by requiring that the pmap is run on all devices
      # (and making the further assumption that each host has the same number of
      # devices). Nested sharding is ok in this case.
      global_axis_size = axis_size * xb.host_count()
      assert all(len(xb.local_devices(host_id)) == xb.local_device_count()
                 for host_id in xb.host_ids())
      must_run_on_all_devices = True

  if devices:
    local_devices = [d for d in devices if d.host_id == xb.host_id()]
    assert len(local_devices) > 0
  else:
    local_devices = None  # type: ignore

  if config.omnistaging_enabled:
    sharded_avals = tuple(shard_aval(axis_size, axis, aval) if axis is not None else aval
                          for axis, aval in safe_zip(in_axes, avals))
    if any(s is not None for s in global_arg_shapes):
      # TODO(skye): we could take this branch unconditionally if we handled
      # grad of global_arg_shapes correctly.
      global_sharded_avals = [
          ShapedArray(shape, aval.dtype) if shape is not None else aval
          for shape, aval in safe_zip(global_arg_shapes, sharded_avals)]
    else:
      global_sharded_avals = sharded_avals  # type: ignore
    logging.vlog(2, "sharded_avals: %s", sharded_avals)
    logging.vlog(2, "global_sharded_avals: %s", global_sharded_avals)

    with core.extend_axis_env(axis_name, global_axis_size, None):  # type: ignore
      jaxpr, out_sharded_avals, consts = pe.trace_to_jaxpr_final(fun, global_sharded_avals)
    jaxpr = xla.apply_outfeed_rewriter(jaxpr)
  else:
    @lu.wrap_init
    def dynamic_fun(dummy, *args):
      with extend_dynamic_axis_env(axis_name, dummy._trace, global_axis_size):  # type: ignore
        return fun.call_wrapped(*args)

    sharded_avals = tuple(shard_aval(axis_size, axis, aval) if axis is not None else aval
                          for axis, aval in safe_zip(in_axes, avals))
    pvals = [pe.PartialVal.unknown(aval) for aval in sharded_avals]
    # We add a dummy first invar, to carry the trace  details to `dynamic_fun`
    pval = pe.PartialVal.unknown(core.abstract_unit)  # dummy value for axis env
    jaxpr, out_pvals, consts = pe.trace_to_jaxpr(  # type: ignore
      dynamic_fun, [pval] + pvals, instantiate=False, stage_out=True, bottom=True)  # type: ignore
    jaxpr.invars = jaxpr.invars[1:]  # ignore dummy
    jaxpr = xla.apply_outfeed_rewriter(jaxpr)

    out_pvs, out_consts = unzip2(out_pvals)
    global_sharded_avals = sharded_avals  # type: ignore

  out_axes = out_axes_thunk()
  if config.omnistaging_enabled:
    assert len(out_sharded_avals) == len(out_axes), (len(out_sharded_avals), len(out_axes))
  else:
    assert len(out_pvals) == len(out_axes), (len(out_pvals), len(out_axes))
    assert all(out_axis == 0 for out_axis in out_axes)

  # TODO(skye,mattjj): allow more collectives on multi-host as we test them, but
  # for now raise an error
  if devices is not None:
    is_multi_host_pmap = len(local_devices) != len(devices)
  else:
    is_multi_host_pmap = xb.host_count() > 1
  if is_multi_host_pmap:
    check_multihost_collective_allowlist(jaxpr)

  if not config.omnistaging_enabled:
    if all(pv is None for pv in out_pvs):
      # When the output doesn't depend on the input we don't need to compile an
      # XLA computation at all; we handle this as a special case so we can stage
      # out multi-replica XLA computations regardless of the hardware available.
      # The 'None' values here are just dummies we know will be ignored.
      handlers = [
        _pval_to_result_handler(  # type: ignore
          axis_size, None, None, None, pval, local_devices, backend_name)  # type: ignore
        for pval in out_pvals  # type: ignore
      ]
      results = [handler(None) for handler in handlers]
      return lambda *_: results


  # TODO(skyewm): replace this with a chain of pmaps and/or sharded_jits
  jaxpr_replicas = xla.jaxpr_replicas(jaxpr)
  num_local_replicas = axis_size * jaxpr_replicas
  num_global_replicas = global_axis_size * jaxpr_replicas

  (arg_parts, out_parts, num_partitions, local_arg_parts, local_out_parts,
   local_num_partitions) = _find_partitions(jaxpr)

  if local_num_partitions is None:
    local_num_partitions = num_partitions

  if local_arg_parts is None:
    local_arg_parts = arg_parts
  if local_out_parts is None:
    local_out_parts = out_parts

  logging.vlog(2, "num_replicas: %d  num_local_replicas: %d",
               num_global_replicas, num_local_replicas)
  logging.vlog(2, "num_partitions: %d  local_num_partitions: %d",
               num_partitions, local_num_partitions)
  logging.vlog(2, "arg_parts: %s", arg_parts)
  logging.vlog(2, "local_arg_parts: %s", local_arg_parts)
  logging.vlog(2, "out_parts: %s", out_parts)
  logging.vlog(2, "local_out_parts: %s", local_out_parts)
  logging.vlog(2, "devices: %s", devices)
  logging.vlog(2, "local_devices: %s", local_devices)

  num_local_shards = num_local_replicas * local_num_partitions
  num_global_shards = num_global_replicas * num_partitions

  if (xb.host_count() > 1 and must_run_on_all_devices and
      num_local_shards != xb.local_device_count()):
    if num_local_shards == axis_size:
      raise ValueError(
         f"On multi-host platforms, the input to pmapped functions must have "
         f"leading axis size equal to the number of local devices if no "
         f"`devices` argument is specified. Got axis_size={axis_size}, "
         f"num_local_devices={xb.local_device_count()}")
    else:
      raise ValueError(
        f"On multi-host platforms, pmapped functions must run across all "
        f"devices, i.e. num_replicas * num_partitions should equal the "
        f"number of local devices. Got num_replicas={num_local_replicas}, "
        f"num_partitions={num_partitions}, and "
        f"num_local_devices={xb.local_device_count()}")

  if no_nested_sharding and (jaxpr_replicas > 1 or num_partitions > 1):
    raise ValueError(
      f"On multi-host platforms, pmapped functions that both have `devices` "
      f"specified and contain an inner_pmap or sharded_jit must specify an "
      f"`axis_size` (or remove the `devices` argument). Got nested_replicas="
      f"{jaxpr_replicas} and nested_partitions={num_partitions}")

  log_priority = logging.WARNING if FLAGS.jax_log_compiles else logging.DEBUG
  logging.log(log_priority,
              f"Compiling {fun.__name__} for {num_global_shards} devices with "
              f"args {avals}. (num_replicas={num_global_replicas} "
              f"num_partitions={num_partitions})")

  axis_env = xla.AxisEnv(num_global_replicas, (axis_name,), (global_axis_size,))

  tuple_args = len(global_sharded_avals) > 100  # pass long arg lists as tuple for TPU

  c = xb.make_computation_builder("pmap_{}".format(fun.__name__))
  xla_consts = map(partial(xb.constant, c), consts)
  replicated_args = [axis is None for axis in in_axes]
  xla_args, donated_invars = xla._xla_callable_args(c, global_sharded_avals, tuple_args,
                                                    replicated=replicated_args,
                                                    partitions=arg_parts,
                                                    donated_invars=donated_invars)
  with maybe_extend_axis_env(axis_name, global_axis_size, None):  # type: ignore
    out_nodes = xla.jaxpr_subcomp(c, jaxpr, backend_name, axis_env, xla_consts,
                                  extend_name_stack(wrap_name(name, 'pmap')), *xla_args)
  build_out_tuple = partial(xops.Tuple, c, out_nodes)
  if out_parts is not None:
    out_tuple = xb.with_sharding(c, out_parts, build_out_tuple)
  else:
    out_tuple = build_out_tuple()
  backend = xb.get_backend(backend_name)
  if backend.platform in ("gpu", "tpu"):
    donated_invars = xla.set_up_aliases(c, xla_args, out_tuple, donated_invars, tuple_args)
  built = c.Build(out_tuple)

  if devices is None:
    if num_global_shards > xb.device_count(backend):
      msg = ("compiling computation that requires {} logical devices, but only {} XLA "
             "devices are available (num_replicas={}, num_partitions={})")
      raise ValueError(msg.format(num_global_shards, xb.device_count(backend),
                                  num_global_replicas, num_partitions))

    # On a single host, we use the platform's default device assignment to
    # potentially take advantage of device locality. On multiple hosts, the
    # default device assignment may interleave different hosts' replicas,
    # violating pmap's semantics where data is sharded across replicas in
    # row-major order. Instead, manually create a device assignment that ensures
    # each host is responsible for a continguous set of replicas.
    if num_global_shards > num_local_shards:
      # TODO(skye): use a locality-aware assignment that satisfies the above
      # constraint.
      devices = [d for host_id in xb.host_ids()
                 for d in xb.local_devices(host_id)]
    else:
      devices = xb.get_backend(backend).get_default_device_assignment(
          num_global_replicas, num_partitions)
  else:
    if num_local_shards != len(local_devices):
      local_devices_str = ", ".join(map(str, local_devices))
      raise ValueError(
          "Leading axis size of input to pmapped function must equal the "
          "number of local devices passed to pmap. Got axis_size=%d, "
          "num_local_devices=%d.\n(Local devices passed to pmap: %s)"
          % (axis_size, len(local_devices), local_devices_str))
    if num_global_shards != len(devices):
      raise ValueError("compiling computation that creates %s shards, "
                       "but %s devices were specified" %
                       (num_global_shards, len(devices)))

  # 'devices' may be 1D or 2D at this point (e.g.
  # get_default_device_assignment() returns 2D assignment, caller may have
  # provided 1D list of devices).
  device_assignment = tree_map(lambda d: d.id, devices)
  # Convert to 2D in case it's 1D and we have > 1 partitions.
  device_assignment = np.array(device_assignment).reshape(
      (num_global_replicas, num_partitions))
  # TODO(b/162356737): Enabling SPMD partitioning causes issues with some
  # non-partitioned workloads, so disable unless needed.
  use_spmd_partitioning = num_partitions > 1
  compile_options = xb.get_compile_options(
      num_replicas=num_global_replicas,
      num_partitions=num_partitions,
      device_assignment=device_assignment,
      use_spmd_partitioning=use_spmd_partitioning,
  )
  compile_options.parameter_is_tupled_arguments = tuple_args
  compiled = xla.backend_compile(backend, built, compile_options)

  local_arg_parts_ = local_arg_parts or [None] * len(avals)
  input_sharding_specs = [
      _pmap_sharding_spec(num_local_replicas, axis_size, local_num_partitions,
                          parts, aval, in_axis)
      if aval is not core.abstract_unit else None
      for aval, parts, in_axis in safe_zip(sharded_avals, local_arg_parts_, in_axes)]
  input_indices = [spec_to_indices(aval.shape, spec)
                   if spec is not None else None
                   for aval, spec in safe_zip(avals, input_sharding_specs)]
  handle_args = partial(shard_args, compiled.local_devices(), input_indices)
  if config.omnistaging_enabled:
    nouts = len(out_sharded_avals)
    if out_parts is None:
      out_parts = (None,) * nouts
    if local_out_parts is None:
      local_out_parts = (None,) * nouts

    local_out_avals = [get_local_aval(aval, parts, lparts)
                       for aval, parts, lparts
                       in safe_zip(out_sharded_avals, out_parts, local_out_parts)]
    local_unmapped_avals = [core.unmapped_aval(axis_size, out_axis, aval)
                            if out_axis is not None else aval
                            for aval, out_axis in safe_zip(local_out_avals, out_axes)]

    out_specs = [_pmap_sharding_spec(num_local_replicas, axis_size, local_num_partitions,
                                     parts, aval, out_axis)
                if aval is not core.abstract_unit else None
                for parts, aval, out_axis in safe_zip(local_out_parts, local_out_avals, out_axes)]
    handle_outs = avals_to_results_handler(
        num_local_replicas, local_num_partitions, out_specs, local_unmapped_avals)
  else:
    handle_outs = _pvals_to_results_handler(axis_size, num_local_replicas,  # type: ignore
                                            local_num_partitions,
                                            local_out_parts, out_pvals,
                                            compiled.local_devices(), backend)

  return partial(execute_replicated, compiled, backend, handle_args, handle_outs)

multi_host_supported_collectives: Set[core.Primitive] = set()


def check_multihost_collective_allowlist(jaxpr):
  used_collectives = set(xla.jaxpr_collectives(jaxpr))
  if not used_collectives.issubset(multi_host_supported_collectives):
    bad_collectives = used_collectives - multi_host_supported_collectives
    msg = "using collectives that aren't supported for multi-host: {}"
    raise TypeError(msg.format(", ".join(map(str, bad_collectives))))


PartitionsOrReplicated = Optional[Tuple[int, ...]]

def _find_partitions(jaxpr) -> Tuple[
    Optional[Tuple[PartitionsOrReplicated, ...]],
    Optional[Tuple[PartitionsOrReplicated, ...]],
    int,
    Optional[Tuple[PartitionsOrReplicated, ...]],
    Optional[Tuple[PartitionsOrReplicated, ...]],
    Optional[int]]:
  """Returns (in_partitions, out_partitions, num_partitions, local_in_parts,
              local_out_parts, local_num_partitions).
  """
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == "sharded_call":
      if len(jaxpr.eqns) > 1:
        raise NotImplementedError(
            "pmap of sharded_jit + non-sharded operations not yet implemented.")
      num_partitions = reconcile_num_partitions(eqn.params["call_jaxpr"],
                                                eqn.params["nparts"])
      return (eqn.params["in_parts"],
              eqn.params["out_parts_thunk"](),
              num_partitions,
              eqn.params["local_in_parts"],
              eqn.params["local_out_parts_thunk"](),
              eqn.params["local_nparts"])
  return None, None, 1, None, None, None

def reconcile_num_partitions(jaxpr, outer_num_parts: Optional[int]):
  """Returns the total number of partitions to use.

  Validates that any inner partitioning matches outer_num_parts if provided, and
  returns the number of partitions to use based on outer_num_parts and any inner
  partitioning.
  """
  inner_num_parts = _inner_partitions(jaxpr, outer_num_parts)
  if outer_num_parts is None and inner_num_parts is None:
    # No partitions specified anywhere, everything is replicated.
    return 1
  if outer_num_parts is None:
    return inner_num_parts
  return outer_num_parts


def _inner_partitions(jaxpr, expected_num_parts: Optional[int]):
  """Returns the total number of partitions from PartitionSpecs inside `jaxpr`.

  Also validates that this number matches `expected_num_parts` if provided.
  """
  for eqn in jaxpr.eqns:
    if eqn.primitive.name in ["sharding_constraint", "infeed"]:
      parts = eqn.params["partitions"]
      nparts = get_num_partitions(parts)
      if expected_num_parts is None:
        expected_num_parts = nparts
      elif nparts is not None and nparts != expected_num_parts:
        # TODO(skye): raise this error as we trace the jaxpr
        raise ValueError(
            f"with_sharding_constraint with partitions={parts} "
            f"(total partitions: {nparts}) doesn't match expected number of "
            f"partitions: {expected_num_parts}. If these partitions look "
            f"right, check outer sharded_jit and/or other "
            f"with_sharding_constraint calls.")
    else:
      for subjaxpr in core.jaxprs_in_params(eqn.params):
        expected_num_parts = _inner_partitions(subjaxpr, expected_num_parts)
  return expected_num_parts


def get_num_partitions(*partitions):
  partition_specs = tree_flatten(partitions)[0]
  if len(partition_specs) == 0:
    # Everything is specified as replicated (all Nones).
    return None
  num_partitions_set = {np.prod(spec) for spec in partition_specs}
  if len(num_partitions_set) > 1:
    raise ValueError(
        f"All partition specs must use the same number of total partitions, "
        f"got {partitions}, with distinct number of partitions "
        f"{num_partitions_set} (the total number of partitions is the product "
        f"of a partition spec)")
  assert len(num_partitions_set) == 1
  return num_partitions_set.pop()


def get_global_aval(local_aval, global_parts: PartitionsOrReplicated,
                    local_parts: PartitionsOrReplicated):
  if local_aval is core.abstract_unit:
    return local_aval
  if global_parts is None:
    return local_aval
  assert local_parts is not None
  global_shape = [dim * _safe_div(ngparts, nlparts)
                  for dim, ngparts, nlparts
                  in safe_zip(local_aval.shape, global_parts, local_parts)]
  return ShapedArray(global_shape, local_aval.dtype)


def get_local_aval(global_aval, global_parts: PartitionsOrReplicated,
                   local_parts: PartitionsOrReplicated):
  if global_aval is core.abstract_unit:
    return global_aval
  if global_parts is None:
    return global_aval
  assert local_parts is not None
  local_shape = [_safe_div(dim, _safe_div(ngparts, nlparts))
                 for dim, ngparts, nlparts
                 in safe_zip(global_aval.shape, global_parts, local_parts)]
  return ShapedArray(local_shape, global_aval.dtype)


def _safe_div(x, y):
  result, ragged = divmod(x, y)
  assert not ragged, f"{x} % {y} != 0"
  return result


class ResultToPopulate: pass
result_to_populate = ResultToPopulate()

def avals_to_results_handler(nrep, npart, out_specs, unmapped_local_out_avals):
  nouts = len(unmapped_local_out_avals)
  out_indices = [spec_to_indices(aval.shape, spec)
                 if aval is not core.abstract_unit else None
                 for aval, spec in safe_zip(unmapped_local_out_avals, out_specs)]  # pytype: disable=attribute-error
  handlers = [aval_to_result_handler(spec, idcs, aval)
              for spec, idcs, aval in safe_zip(out_specs, out_indices, unmapped_local_out_avals)]

  def handler(out_bufs):
    assert nrep * npart == len(out_bufs)
    buffers = [[result_to_populate] * nrep * npart for _ in range(nouts)]
    for r, tuple_buf in enumerate(out_bufs):
      for i, buf in enumerate(tuple_buf):
        buffers[i][r] = buf
    assert not any(buf is result_to_populate for bufs in buffers
                  for buf in bufs)
    return [h(bufs) for h, bufs in safe_zip(handlers, buffers)]
  return handler

def replicate(val, axis_size, nrep, devices=None, backend=None, in_axis=0):
  """Replicates ``val`` across multiple devices.

  Args:
    val: the value to be replicated.
    axis_size: the length of the output, i.e. the logical number of replicas to
    create. Usually equal to `nrep`, but in the case of nested pmaps, `nrep` may
    be a multiple of `axis_size`.
    nrep: the number of replicas to create. If ``devices`` is set, must be equal
      to ``len(devices)``.
    devices: the devices to replicate across. If None, ``nrep`` will be used to
      generate a default device assignment.
    backend: string specifying which backend to use.
    in_axis: axis along which the value is to be replciated.

  Returns:
    A ShardedDeviceArray of length `axis_size` where each shard is equal to
    ``val``.
  """
  device_count = (len(devices) if devices else xb.local_device_count(backend))
  if nrep > device_count:
    msg = ("Cannot replicate across %d replicas because only %d local devices "
           "are available." % (nrep, device_count))
    if devices:
      msg += (" (local devices = %s)"
              % ", ".join(map(str, devices)) if devices else str(None))
    raise ValueError(msg)

  if devices is None:
    assert nrep is not None
    # TODO(skye): use different device assignment on multihost
    devices = xb.get_backend(backend).get_default_device_assignment(nrep)
  assert nrep == len(devices)

  aval = xla.abstractify(val)  # type: ShapedArray
  replicated_aval = ShapedArray((axis_size,) + aval.shape, aval.dtype)
  # TODO(skye): figure out how partitioning should work here
  sharding_spec = _pmap_sharding_spec(nrep, axis_size, 1, None, aval, in_axis)
  device_buffers = device_put(val, devices, replicate=True)
  return ShardedDeviceArray(replicated_aval, sharding_spec, device_buffers)

def _pmap_sharding_spec(nrep, axis_size, npart, parts, sharded_aval, map_axis: Optional[int]):
  """Sharding spec for arguments or results of a pmap.
  Args:
    nrep: number of local XLA replicas (product of local axis sizes)
    axis_size: local axis size for outer pmap
    npart: total number of XLA partitions (required by sharded_jit calls)
    parts: the partitioning of the value or None
    sharded_aval: the aval of the value inside the outer pmap, an instance of
      a ShapedArray.
    map_axis: the axis along which the value is mapped in the outer pmap
  Returns:
    A ShardingSpec.
  """
  assert isinstance(sharded_aval, ShapedArray), sharded_aval
  replication_factor, ragged = divmod(nrep, axis_size)
  assert not ragged
  # get the sharding spec from inner sharded_jits as if we weren't in a pmap
  pspec = partitioned_sharding_spec(npart, parts, sharded_aval)
  maybe_replicate = () if replication_factor == 1 else (Replicated(replication_factor),)
  if map_axis is not None:
    sharded_in_axis = sum(s is not None for s in pspec.sharding[:map_axis])
    def shift_sharded_axis(a: MeshDimAssignment):
      if isinstance(a, ShardedAxis) and a.axis >= sharded_in_axis:
        return ShardedAxis(a.axis + 1)
      return a
    # replication_factor represents the product of inner pmaps, so it goes
    # after the outer pmapped axis at index 0
    return ShardingSpec(
      sharding=tuple_insert(pspec.sharding, map_axis, Unstacked(axis_size)),
      mesh_mapping=it.chain([ShardedAxis(sharded_in_axis)],
                            maybe_replicate,
                            map(shift_sharded_axis, pspec.mesh_mapping)))
  else:
    return ShardingSpec(
      sharding=pspec.sharding,
      mesh_mapping=(Replicated(axis_size),) + maybe_replicate + pspec.mesh_mapping)

def partitioned_sharding_spec(num_partitions: int,
                              partitions: Optional[Sequence[int]],
                              aval) -> ShardingSpec:
  if partitions is None:
    maybe_replicate = () if num_partitions == 1 else (Replicated(num_partitions),)
    return ShardingSpec(sharding=[None] * len(aval.shape),
                        mesh_mapping=maybe_replicate)
  else:
    assert len(partitions) == len(aval.shape)
    return ShardingSpec(sharding=map(Chunked, partitions),
                        mesh_mapping=map(ShardedAxis, range(len(partitions))))


def execute_replicated(compiled, backend, in_handler, out_handler, *args):
  input_bufs = in_handler(args)
  out_bufs = compiled.execute_on_local_devices(list(input_bufs))
  return out_handler(out_bufs)


xla_pmap_p = core.MapPrimitive('xla_pmap')
xla_pmap = xla_pmap_p.bind
xla_pmap_p.def_impl(xla_pmap_impl)

# Set param update handlers to update `donated_invars` just like xla_call_p
pe.call_param_updaters[xla_pmap_p] = pe.call_param_updaters[xla.xla_call_p]
ad.call_param_updaters[xla_pmap_p] = ad.call_param_updaters[xla.xla_call_p]
ad.call_transpose_param_updaters[xla_pmap_p] = \
    ad.call_transpose_param_updaters[xla.xla_call_p]

def _pmap_translation_rule(c, axis_env,
                           in_nodes, name_stack, axis_name, axis_size,
                           global_axis_size, devices, name,
                           call_jaxpr, *, backend=None, in_axes, out_axes,
                           donated_invars, global_arg_shapes):
  del donated_invars  # Unused.
  # We in-line here rather than generating a Call HLO as in the xla_call
  # translation rule just because the extra tuple stuff is a pain.
  if axis_env.names and devices is not None:
    raise ValueError("Nested pmap with explicit devices argument.")
  if global_axis_size is None:
    global_axis_size = axis_size
  new_env = xla.extend_axis_env(axis_env, axis_name, global_axis_size)
  # Shard the in_nodes that are mapped
  in_avals = [v.aval for v in call_jaxpr.invars]
  in_nodes_sharded = (
    _xla_shard(c, aval, new_env, in_node, in_axis) if in_axis is not None else in_node
    for aval, in_node, in_axis in safe_zip(in_avals, in_nodes, in_axes))

  with maybe_extend_axis_env(axis_name, global_axis_size, None):  # type: ignore
    sharded_outs = xla.jaxpr_subcomp(
        c, call_jaxpr, backend, new_env, (),
        extend_name_stack(name_stack, wrap_name(name, 'pmap')), *in_nodes_sharded)
  out_avals = [v.aval for v in call_jaxpr.outvars]
  outs = [_xla_unshard(c, aval, new_env, out_axis, shard, backend=backend)
          for aval, out_axis, shard in safe_zip(out_avals, out_axes, sharded_outs)]
  return xops.Tuple(c, outs)

xla.call_translations[xla_pmap_p] = _pmap_translation_rule
ad.primitive_transposes[xla_pmap_p] = partial(ad.map_transpose, xla_pmap_p)

def _xla_shard(c, aval, axis_env, x, in_axis):
  if aval is core.abstract_unit:
    return x
  elif aval is core.abstract_token:
    return x
  elif isinstance(aval, ShapedArray):
    dims = list(c.get_shape(x).dimensions())
    zero = xb.constant(c, np.zeros((), dtype=np.uint32))
    idxs = [zero] * (len(dims) - 1)
    idxs.insert(in_axis, _unravel_index(c, axis_env))
    dims_unsqueezed = dims.copy()
    dims_unsqueezed[in_axis] = 1
    dims_squeezed = dims.copy()
    dims_squeezed.pop(in_axis)
    return xops.Reshape(xops.DynamicSlice(x, idxs, dims_unsqueezed), dims_squeezed)
  else:
    raise TypeError((aval, c.get_shape(x)))

# TODO(b/110096942): more efficient gather
def _xla_unshard(c, aval, axis_env, out_axis, x, backend):
  if aval is core.abstract_unit:
    return x
  elif aval is core.abstract_token:
    return x
  elif isinstance(aval, ShapedArray):
    # TODO(mattjj): remove this logic when AllReduce PRED supported on CPU / GPU
    convert_bool = (np.issubdtype(aval.dtype, np.bool_)
                    and xb.get_backend(backend).platform in ('cpu', 'gpu'))
    if convert_bool:
      x = xops.ConvertElementType(x, xb.dtype_to_etype(np.float32))

    xla_shape = c.get_shape(x)
    dims = list(xla_shape.dimensions())
    padded = xops.Broadcast(xb.constant(c, np.array(0, xla_shape.numpy_dtype())),
                         [axis_env.sizes[-1]] + dims)
    zero = xb.constant(c, np.zeros((), dtype=np.uint32))
    idxs = [_unravel_index(c, axis_env)] + [zero] * len(dims)
    padded = xops.DynamicUpdateSlice(padded, xops.Reshape(x, [1] + dims), idxs)
    replica_groups_protos = xc.make_replica_groups(
      xla.axis_groups(axis_env, axis_env.names[-1]))
    out = xops.CrossReplicaSum(padded, replica_groups_protos)
    if out_axis != 0:
      # TODO(apaszke,mattjj): Change the indices to DynamicUpdateSlice instead
      perm = list(range(1, len(dims)))
      perm.insert(out_axis, 0)
      out = xops.Transpose(out, perm)

    # TODO(mattjj): remove this logic when AllReduce PRED supported on CPU / GPU
    if convert_bool:
      nonzero = xops.Ne(out, xb.constant(c, np.array(0, dtype=np.float32)))
      out = xops.ConvertElementType(nonzero, xb.dtype_to_etype(np.bool_))
    return out
  else:
    raise TypeError((aval, c.get_shape(x)))

def _unravel_index(c, axis_env):
  div = xb.constant(c, np.array(axis_env.nreps // prod(axis_env.sizes), np.uint32))
  mod = xb.constant(c, np.array(axis_env.sizes[-1], np.uint32))
  return xops.Rem(xops.Div(xops.ReplicaId(c), div), mod)

# ------------------- xmap -------------------

MeshAxisName = Any
"""
ArrayMapping specifies how an ndarray should map to mesh axes.

Note that the ordering is crucial for the cases when this mapping is non-injective
(i.e. when multiple mesh axes map to the same positional axis). Then, the
order of entries of the mapping determines a major-to-minor order on mesh axes,
according to which chunks of the value along the repeated dimension will be assigned.

For example, consider a mapping {'x': 1, 'y': 1} and a mesh with shape {'x': 2, 'y': 3}.
The second dimension of the value would get chunked into 6 pieces, and assigned to the
mesh in a way that treats 'y' as the fastest changing (minor) dimension. In this case,
that would mean that a flat list of chunks would get assigned to a flattened list of
mesh devices without any modifications. If the mapping was {'y': 1, 'x': 1}, then the
mesh devices ndarray would have to be transposed before flattening and assignment.
"""
ArrayMapping = OrderedDictType[MeshAxisName, int]

class Mesh:
  __slots__ = ('devices', 'axis_names')

  def __init__(self, devices: np.ndarray, axis_names: Sequence[MeshAxisName]):
    assert devices.ndim == len(axis_names)
    # TODO: Make sure that devices are unique? At least with the quick and
    #       dirty check that the array size is not larger than the number of
    #       available devices?
    self.devices = devices
    self.axis_names = tuple(axis_names)

  @property
  def shape(self):
    return OrderedDict((name, size) for name, size in safe_zip(self.axis_names, self.devices.shape))

  @property
  def size(self):
    return np.prod(list(self.shape.values()))

  # TODO: This is pretty expensive to compute. Cache this on the mesh object?
  @property
  def local_mesh(self):
    host_id = xb.host_id()
    is_local_device = np.vectorize(lambda d: d.host_id == host_id, otypes=[bool])(self.devices)
    subcube_indices = []
    # We take the smallest slice of each dimension that doesn't skip any local device.
    for axis in range(self.devices.ndim):
      other_axes = tuple_delete(tuple(range(self.devices.ndim)), axis)
      # NOTE: This re-reduces over many axes multiple times, so we could definitely
      #       optimize it, but I hope it won't be a bottleneck anytime soon.
      local_slices = is_local_device.any(other_axes, keepdims=False)
      nonzero_indices = np.flatnonzero(local_slices)
      start, end = int(np.min(nonzero_indices)), int(np.max(nonzero_indices))
      subcube_indices.append(slice(start, end + 1))
    subcube_indices = tuple(subcube_indices)
    # We only end up with all conditions being true if the local devices formed a
    # subcube of the full array. This is because we were biased towards taking a
    # "hull" spanned by the devices, and in case the local devices don't form a
    # subcube that hull will contain non-local devices.
    assert is_local_device[subcube_indices].all()
    return Mesh(self.devices[subcube_indices], self.axis_names)

  def __getitem__(self, new_axes):
    indices = [0] * len(self.axis_names)
    axis_pos = {name: i for i, name in enumerate(self.axis_names)}
    for axis in new_axes:
      indices[axis_pos[axis]] = slice(None)
    new_devices = self.devices[tuple(indices)]
    new_devices = new_devices.transpose(tuple(axis_pos[axis] for axis in new_axes))
    return Mesh(new_devices, new_axes)

  @property
  def device_ids(self):
    return np.vectorize(lambda d: d.id, otypes=[int])(self.devices)

def tile_aval_nd(axis_sizes, in_axes: ArrayMapping, aval):
  if aval is core.abstract_unit:
    return aval
  assert isinstance(aval, ShapedArray)
  shape = list(aval.shape)
  for name, axis in in_axes.items():
    assert shape[axis] % axis_sizes[name] == 0
    shape[axis] //= axis_sizes[name]
  return ShapedArray(tuple(shape), aval.dtype)

def untile_aval_nd(axis_sizes, out_axes: ArrayMapping, aval):
  if aval is core.abstract_unit:
    return aval
  assert isinstance(aval, ShapedArray)
  shape = list(aval.shape)
  for name, axis in out_axes.items():
    shape[axis] *= axis_sizes[name]
  return ShapedArray(tuple(shape), aval.dtype)

def mesh_tiled_callable(fun: lu.WrappedFun,
                        transformed_name: str,
                        backend_name: Optional[str],
                        mesh: Mesh,
                        in_axes: Sequence[ArrayMapping],
                        out_axes: Sequence[ArrayMapping],
                        spmd_lowering,
                        *local_in_untiled_avals):
  assert config.omnistaging_enabled
  local_mesh = mesh.local_mesh
  global_axis_sizes = mesh.shape
  local_axis_sizes = local_mesh.shape

  log_priority = logging.WARNING if FLAGS.jax_log_compiles else logging.DEBUG
  logging.log(log_priority,
              f"Compiling {fun.__name__} for {tuple(global_axis_sizes.items())} "
              f"mesh with args {local_in_untiled_avals}. Argument mapping: {in_axes}.")

  # 1. Trace to jaxpr and preprocess/verify it
  in_tiled_avals = [tile_aval_nd(local_axis_sizes, aval_in_axes, aval)
                    for aval, aval_in_axes in safe_zip(local_in_untiled_avals, in_axes)]
  if spmd_lowering:
    # TODO: Consider handling xmap's 'vectorize' in here. We can vmap once instead of vtile twice!
    for name, size in reversed(mesh.shape.items()):
      fun = vtile(fun,
                  tuple(a.get(name, None) for a in in_axes),
                  tuple(a.get(name, None) for a in out_axes),
                  tile_size=size, axis_name=name)
    global_in_untiled_avals = [untile_aval_nd(global_axis_sizes, aval_in_axes, aval)
                               for aval, aval_in_axes in safe_zip(in_tiled_avals, in_axes)]
    in_jaxpr_avals = global_in_untiled_avals
  else:
    in_jaxpr_avals = in_tiled_avals
  with core.extend_axis_env_nd(mesh.shape.items()):
    jaxpr, out_jaxpr_avals, consts = pe.trace_to_jaxpr_final(fun, in_jaxpr_avals)
  assert len(out_axes) == len(out_jaxpr_avals)
  if spmd_lowering:
    global_out_untiled_avals = out_jaxpr_avals
    out_tiled_avals = [tile_aval_nd(global_axis_sizes, aval_out_axes, aval)
                       for aval, aval_out_axes in safe_zip(global_out_untiled_avals, out_axes)]
  else:
    out_tiled_avals = out_jaxpr_avals
  local_out_untiled_avals = [untile_aval_nd(local_axis_sizes, aval_out_axes, aval)
                             for aval, aval_out_axes in safe_zip(out_tiled_avals, out_axes)]
  _sanitize_mesh_jaxpr(jaxpr)
  if local_mesh.shape != mesh.shape:
    check_multihost_collective_allowlist(jaxpr)
  jaxpr = xla.apply_outfeed_rewriter(jaxpr)

  # 3. Build up the HLO
  c = xb.make_computation_builder(f"xmap_{fun.__name__}")
  xla_consts = map(partial(xb.constant, c), consts)
  donated_invars = (False,) * len(in_jaxpr_avals)  # TODO(apaszke): support donation
  tuple_args = len(in_jaxpr_avals) > 100  # pass long arg lists as tuple for TPU
  in_partitions: Optional[List]
  if spmd_lowering:
    replicated_args = [False] * len(in_jaxpr_avals)
    global_sharding_spec = mesh_sharding_specs(global_axis_sizes, mesh.axis_names)
    in_partitions = [global_sharding_spec(aval, aval_in_axes).sharding_proto()
                     if aval is not core.abstract_unit else None
                     for aval, aval_in_axes in safe_zip(global_in_untiled_avals, in_axes)]
    out_partitions = [global_sharding_spec(aval, aval_out_axes).sharding_proto()
                      for aval, aval_out_axes in safe_zip(global_out_untiled_avals, out_axes)]
    partitions_proto = True
    axis_env = xla.AxisEnv(nreps=1, names=(), sizes=())  # All named axes have been vmapped
  else:
    replicated_args = [not axis for axis in in_axes]
    in_partitions = None
    partitions_proto = False
    axis_env = xla.AxisEnv(nreps=mesh.size,
                           names=tuple(global_axis_sizes.keys()),
                           sizes=tuple(global_axis_sizes.values()))
  xla_args, donated_invars = xla._xla_callable_args(
      c, in_jaxpr_avals, tuple_args,
      replicated=replicated_args,
      partitions=in_partitions,
      partitions_proto=partitions_proto,
      donated_invars=donated_invars)
  with core.extend_axis_env_nd(mesh.shape.items()):
    out_nodes = xla.jaxpr_subcomp(
        c, jaxpr, backend_name, axis_env, xla_consts,
        extend_name_stack(wrap_name(transformed_name, 'xmap')), *xla_args)
  backend = xb.get_backend(backend_name)
  if spmd_lowering:
    out_partitions_t = xb.tuple_sharding_proto(out_partitions)
    out_tuple = xb.with_sharding_proto(c, out_partitions_t, xops.Tuple, c, out_nodes)
  else:
    out_tuple = xops.Tuple(c, out_nodes)
  # TODO(apaszke): Does that work with SPMD sharding?
  if backend.platform in ("gpu", "tpu"):
    donated_invars = xla.set_up_aliases(c, xla_args, out_tuple, donated_invars, tuple_args)
  built = c.Build(out_tuple)

  # 4. Compile the HLO
  if spmd_lowering:
    num_replicas, num_partitions = 1, mesh.size
    num_local_replicas, num_local_partitions = 1, local_mesh.size
  else:
    num_replicas, num_partitions = mesh.size, 1
    num_local_replicas, num_local_partitions = local_mesh.size, 1
  device_assignment = mesh.device_ids.reshape((num_replicas, num_partitions))
  compile_options = xb.get_compile_options(
      num_replicas=num_replicas,
      num_partitions=num_partitions,
      device_assignment=device_assignment,
      use_spmd_partitioning=spmd_lowering,
  )
  compile_options.parameter_is_tupled_arguments = tuple_args
  compiled = xla.backend_compile(backend, built, compile_options)

  # 5. Argument sharding / output wrapping
  local_sharding_spec = mesh_sharding_specs(local_axis_sizes, mesh.axis_names)
  local_input_specs = [local_sharding_spec(aval, aval_in_axes)
                       if aval is not core.abstract_unit else None
                       for aval, aval_in_axes in safe_zip(local_in_untiled_avals, in_axes)]
  input_indices = [spec_to_indices(aval.shape, spec)
                   if spec is not None else None
                   for aval, spec in safe_zip(local_in_untiled_avals, local_input_specs)]
  handle_args = partial(shard_args, compiled.local_devices(), input_indices)

  local_output_specs = [local_sharding_spec(aval, aval_out_axes)
                        for aval, aval_out_axes in safe_zip(local_out_untiled_avals, out_axes)]
  handle_outs = avals_to_results_handler(num_local_replicas, num_local_partitions,
                                         local_output_specs, local_out_untiled_avals)

  return partial(execute_replicated, compiled, backend, handle_args, handle_outs)

# NOTE: This divides the in_axes by the tile_size and multiplies the out_axes by it.
def vtile(f_flat,
          in_axes_flat: Tuple[Optional[int], ...],
          out_axes_flat: Tuple[Optional[int], ...],
          tile_size: Optional[int], axis_name):
  if tile_size == 1:
    return f_flat

  @curry
  def tile_axis(arg, axis: Optional[int], tile_size):
    if axis is None:
      return arg
    shape = list(arg.shape)
    shape[axis:axis+1] = [tile_size, shape[axis] // tile_size]
    return arg.reshape(shape)

  def untile_axis(out, axis: Optional[int]):
    if axis is None:
      return out
    shape = list(out.shape)
    shape[axis:axis+2] = [shape[axis] * shape[axis+1]]
    return out.reshape(shape)

  @lu.transformation
  def _map_to_tile(*args_flat):
    sizes = (x.shape[i] for x, i in safe_zip(args_flat, in_axes_flat) if i is not None)
    tile_size_ = tile_size or next(sizes, None)
    assert tile_size_ is not None, "No mapped arguments?"
    outputs_flat = yield map(tile_axis(tile_size=tile_size_), args_flat, in_axes_flat), {}
    yield map(untile_axis, outputs_flat, out_axes_flat)

  return _map_to_tile(
    batching.batch_fun(f_flat,
                       in_axes_flat,
                       out_axes_flat,
                       axis_name=axis_name))

_forbidden_primitives = {
  'xla_pmap': 'pmap',
  'soft_pmap': 'soft_pmap',
  'sharded_call': 'sharded_jit',
}
def _sanitize_mesh_jaxpr(jaxpr):
  for eqn in jaxpr.eqns:
    if eqn.primitive.name in _forbidden_primitives:
      raise RuntimeError(f"Nesting {_forbidden_primitives[eqn.primitive.name]} "
                         f"inside xmaps not supported!")
    core.traverse_jaxpr_params(_sanitize_mesh_jaxpr, eqn.params)


def mesh_sharding_specs(axis_sizes, axis_names):
  mesh_axis_pos = {name: i for i, name in enumerate(axis_names)}
  # NOTE: This takes in the non-sharded avals!
  def mk_sharding_spec(aval, aval_axes):
    sharding = [None] * len(aval.shape)
    mesh_mapping = [Replicated(axis_size) for axis_size in axis_sizes.values()]
    next_sharded_axis = 0
    aval_shape = list(aval.shape)
    # NOTE: sorted is stable, which is important when multiple resources
    #       map to the same axis.
    for name, axis in sorted(aval_axes.items(), key=lambda x: x[1]):
      assert aval_shape[axis] % axis_sizes[name] == 0, (axis_sizes[name], aval.shape[axis])
      aval_shape[axis] //= axis_sizes[name]
      if sharding[axis] is None:
        sharding[axis] = Chunked(())
      sharding[axis] = Chunked(sharding[axis].chunks + (axis_sizes[name],))
      assert isinstance(mesh_mapping[mesh_axis_pos[name]], Replicated), \
          "Value mapped to the same mesh axis twice"
      mesh_mapping[mesh_axis_pos[name]] = ShardedAxis(next_sharded_axis)
      next_sharded_axis += 1
    return ShardingSpec(sharding, mesh_mapping)
  return mk_sharding_spec


# ------------------- soft_pmap -------------------

def soft_pmap_impl(fun: lu.WrappedFun, *args, axis_name, axis_size, in_axes, out_axes_thunk):
  abstract_args = unsafe_map(xla.abstractify, args)
  compiled_fun = _soft_pmap_callable(fun, axis_name, axis_size, in_axes, out_axes_thunk,
                                     *abstract_args)
  return compiled_fun(*args)

@lu.cache
def _soft_pmap_callable(fun, axis_name, axis_size, in_axes, out_axes_thunk, *avals):
  mapped_avals = [core.mapped_aval(axis_size, in_axis, aval) if in_axis is not None else aval
                  for in_axis, aval in safe_zip(in_axes, avals)]
  with core.extend_axis_env(axis_name, axis_size, None):  # type: ignore
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, mapped_avals)
  out_axes = out_axes_thunk()
  assert all(out_axis == 0 for out_axis in out_axes)
  jaxpr = xla.apply_outfeed_rewriter(jaxpr)

  num_devices = xb.local_device_count()
  chunk_size, ragged = divmod(axis_size, num_devices)
  if ragged:
    msg = f"number of devices {num_devices} must divide axis size {axis_size}"
    raise NotImplementedError(msg)

  jaxpr, _, consts = _soft_pmap_jaxpr(jaxpr, consts, in_axes,
                                      axis_name, axis_size, chunk_size)
  jaxpr_replicas = xla.jaxpr_replicas(jaxpr)
  if jaxpr_replicas != 1: raise NotImplementedError

  tuple_args = len(avals) > 100  # pass long arg lists as tuple for TPU

  c = xb.make_computation_builder("soft_pmap_{}".format(fun.__name__))
  xla_consts = map(partial(xb.constant, c), consts)
  chunked_avals = [core.unmapped_aval(chunk_size, in_axis, aval) if in_axis is not None else aval
                   for in_axis, aval in safe_zip(in_axes, mapped_avals)]
  xla_args, _ = xla._xla_callable_args(c, chunked_avals, tuple_args)
  axis_env = xla.AxisEnv(num_devices, (axis_name,), (num_devices,))
  out_nodes = xla.jaxpr_subcomp(c, jaxpr, None, axis_env, xla_consts,
                                'soft_pmap', *xla_args)
  built = c.Build(xops.Tuple(c, out_nodes))

  compile_options = xb.get_compile_options(
          num_replicas=num_devices, num_partitions=1, device_assignment=None)
  compile_options.tuple_arguments = tuple_args
  backend = xb.get_backend(None)
  compiled = xla.backend_compile(backend, built, compile_options)

  input_specs = [
      ShardingSpec(sharding=tuple_insert((None,) * (aval.ndim - 1),
                                         in_axis,
                                         Chunked(num_devices)),
                   mesh_mapping=[ShardedAxis(0)])
      if in_axis is not None else
      ShardingSpec(sharding=[None] * aval.ndim,
                   mesh_mapping=[Replicated(num_devices)])
      for aval, in_axis in safe_zip(avals, in_axes)]
  input_indices = [spec and spec_to_indices(aval.shape, spec)
                   for aval, spec in safe_zip(avals, input_specs)]
  handle_args = partial(shard_args, compiled.local_devices(), input_indices)
  handle_outs = soft_pmap_avals_to_results_handler(num_devices, chunk_size, out_avals)

  return partial(execute_replicated, compiled, backend, handle_args, handle_outs)

def _soft_pmap_jaxpr(jaxpr, consts, in_axes, axis_name, axis_size, chunk_size):
  assert all(in_axis is None or in_axis == 0 for in_axis in in_axes), in_axes
  mapped_invars = [in_axis is not None for in_axis in in_axes]
  fun = partial(_soft_pmap_interp, chunk_size, jaxpr, consts, mapped_invars)
  in_avals = [core.unmapped_aval(chunk_size, in_axis, v.aval) if in_axis is not None else v.aval
              for v, in_axis in safe_zip(jaxpr.invars, in_axes)]
  with core.extend_axis_env(axis_name, axis_size, None):
    return pe.trace_to_jaxpr_dynamic(lu.wrap_init(fun), in_avals)

def _soft_pmap_interp(chunk_size, jaxpr, consts, mapped_invars, *args):

  env: Dict[Var, Tuple[Any, bool]] = {}

  def read(atom: Union[Var, Literal]) -> Tuple[Any, bool]:
    if isinstance(atom, Literal):
      return (atom.val, False)
    else:
      return env[atom]

  def write(v: Var, val: Any, mapped: bool) -> None:
    env[v] = (val, mapped)

  write(core.unitvar, core.unit, False)
  map(write, jaxpr.constvars, consts, (False,) * len(consts))
  map(write, jaxpr.invars, args, mapped_invars)
  for eqn in jaxpr.eqns:
    in_vals, in_mapped = unzip2(map(read, eqn.invars))
    if eqn.primitive in xla.parallel_translations:
      rule = soft_pmap_rules[eqn.primitive]
      out_vals, out_mapped = rule(in_vals, in_mapped, chunk_size, **eqn.params)
      if not eqn.primitive.multiple_results:
        out_vals, out_mapped = [out_vals], [out_mapped]
    elif isinstance(eqn.primitive, core.CallPrimitive):
      # we just inline here for convenience
      call_jaxpr, params = core.extract_call_jaxpr(eqn.primitive, eqn.params)
      out_vals = _soft_pmap_interp(chunk_size, call_jaxpr, (), in_mapped, *in_vals)
      out_mapped = [True] * len(out_vals)
    elif isinstance(eqn.primitive, core.MapPrimitive):
      raise NotImplementedError  # TODO
    else:
      if any(in_mapped):
        rule = batching.get_primitive_batcher(eqn.primitive, None)
        in_axes = [0 if m else batching.not_mapped for m in in_mapped]
        out_vals, out_axes = rule(in_vals, in_axes, **eqn.params)
        if not eqn.primitive.multiple_results:
          out_vals, out_axes = [out_vals], [out_axes]
        out_vals = [moveaxis(x, d, 0) if d is not not_mapped and d != 0 else x
                    for x, d in safe_zip(out_vals, out_axes)]
        out_mapped = [d is not not_mapped for d in out_axes]
      else:
        out_vals = eqn.primitive.bind(*in_vals, **eqn.params)
        if not eqn.primitive.multiple_results:
          out_vals = [out_vals]
        out_mapped = [False for _ in out_vals]
    map(write, eqn.outvars, out_vals, out_mapped)

  out_vals, out_mapped = unzip2(map(read, jaxpr.outvars))
  out_vals = [out if mapped else broadcast(out, chunk_size, 0)
              for out, mapped in safe_zip(out_vals, out_mapped)]
  return out_vals

# TODO(mattjj): dedup w/ with other aval_to_result_handler via ShardingSpec
def soft_pmap_avals_to_results_handler(num_devices, chunk_size, out_avals):
  nouts = len(out_avals)
  handlers = [soft_pmap_aval_to_result_handler(chunk_size, num_devices, aval)
              for aval in out_avals]
  def handler(out_bufs):
    buffers = [[result_to_populate] * num_devices for _ in range(nouts)]
    for r, tuple_buf in enumerate(out_bufs):
      for i, buf in enumerate(tuple_buf):
        buffers[i][r] = buf
    assert not any(buf is result_to_populate for bufs in buffers
                   for buf in bufs)
    return [h(bufs) for h, bufs in safe_zip(handlers, buffers)]
  return handler

def soft_pmap_aval_to_result_handler(chunk_size, num_devices, aval):
  axis_size = chunk_size * num_devices
  if aval is core.abstract_unit:
    return lambda _: core.unit
  elif isinstance(aval, core.ShapedArray):
    new_aval = ShapedArray((axis_size,) + aval.shape, aval.dtype)
    spec = ShardingSpec(sharding=(Chunked(num_devices),) + (None,) * aval.ndim,
                        mesh_mapping=(ShardedAxis(0),))
    return lambda bufs: ShardedDeviceArray(new_aval, spec, bufs)
  else:
    raise TypeError(aval)

soft_pmap_p = core.MapPrimitive('soft_pmap')
soft_pmap = soft_pmap_p.bind
soft_pmap_p.def_impl(soft_pmap_impl)

soft_pmap_rules: Dict[core.Primitive, Callable] = {}

@contextmanager
def maybe_extend_axis_env(*args, **kwargs):
  with core.extend_axis_env(*args, **kwargs):
    yield

@config.register_omnistaging_disabler
@no_type_check
def omnistaging_disabler() -> None:
  global DynamicAxisEnvFrame, DynamicAxisEnv, _ThreadLocalState, \
      _thread_local_state, extend_dynamic_axis_env, unmapped_device_count, \
      apply_parallel_primitive, parallel_pure_rules, \
      _pvals_to_results_handler, _pval_to_result_handler, replicate, \
      axis_index, maybe_extend_axis_env

  @contextmanager
  def maybe_extend_axis_env(*args, **kwargs):
    yield

  def _pvals_to_results_handler(
      size, nrep, npart,
      out_parts: Optional[Tuple[PartitionsOrReplicated, ...]],
      out_pvals, devices, backend):
    nouts = len(out_pvals)
    if out_parts is None:
      out_parts = (None,) * len(out_pvals)
    handlers = [
        _pval_to_result_handler(size, nrep, npart, parts, pval, devices, backend)
        for pval, parts in safe_zip(out_pvals, out_parts)  # type: ignore
    ]

    def handler(out_bufs):
      assert nrep * npart == len(out_bufs)
      buffers = [[result_to_populate] * nrep * npart for _ in range(nouts)]
      for r, tuple_buf in enumerate(out_bufs):
        for i, buf in enumerate(tuple_buf):
          buffers[i][r] = buf
      assert not any(buf is result_to_populate for bufs in buffers
                    for buf in bufs)
      return [h(bufs) for h, bufs in safe_zip(handlers, buffers)]
    return handler

  def _pval_to_result_handler(axis_size, nrep, npart, parts, pval, devices, backend):
    if devices:
      assert all(d.host_id == xb.host_id(backend) for d in devices)
    pv, const = pval
    if pv is None:
      if nrep is None:
        nrep = axis_size
        # If 'const' is a ShardedDeviceArray, it must have come from a pmap nested
        # inside the one we're currently evaluating, and we should replicate
        # 'const' across the total number of devices needed. We don't necessarily
        # know the nested pmap's axis_size (e.g. the jaxpr for
        # pmap(pmap(lambda x: 3)) is trivial, with no pmaps), but we can use the
        # axis size of the output 'const'.
        # TODO: we might be doing unnecessary device transfers in the inner pmap.
        if isinstance(const, ShardedDeviceArray):
          nrep *= len(const)

      bcast_const = (core.unit if const is core.unit
                    else replicate(const, axis_size, nrep, devices, backend))  # type: ignore
      return lambda _: bcast_const  # type: ignore
    else:
      if pv is not core.abstract_unit:
        unsharded_aval = ShapedArray((axis_size,) + pv.shape, pv.dtype)
        sharding_spec = _pmap_sharding_spec(nrep, axis_size, npart, parts, pv, 0)
        indices = spec_to_indices(unsharded_aval.shape, sharding_spec)
      else:
        sharding_spec = indices = None
        unsharded_aval = pv
      return aval_to_result_handler(sharding_spec, indices, unsharded_aval)

  @contextmanager
  def extend_dynamic_axis_env(axis_name, pmap_trace, hard_size):
    dynamic_axis_env = _thread_local_state.dynamic_axis_env
    dynamic_axis_env.append(DynamicAxisEnvFrame(axis_name, pmap_trace, hard_size))
    try:
      yield
    finally:
      dynamic_axis_env.pop()

  def unmapped_device_count(backend=None):
    dynamic_axis_env = _thread_local_state.dynamic_axis_env
    mapped = prod(frame.hard_size for frame in dynamic_axis_env)
    unmapped, ragged = divmod(xb.device_count(backend), mapped)
    assert not ragged and unmapped > 0
    return unmapped

  def apply_parallel_primitive(prim, *args, **params):
    # This is the op-by-op version of applying a collective primitive, like a psum
    # that doesn't have a data dependence on the argument of a pmap function. In
    # particular, this code gets hit when we write `axis_size = psum(1, 'i')`. We
    # look up information in the dynamic axis env.
    dynamic_axis_env = _thread_local_state.dynamic_axis_env
    axis_name = params.pop('axis_name')
    axis_index_groups = params.pop('axis_index_groups')
    if axis_index_groups is not None:
      shape = (len(axis_index_groups[0]),)
    else:
      logical_size = lambda frame: frame.hard_size
      if isinstance(axis_name, (list, tuple)):
        shape = tuple(logical_size(dynamic_axis_env[name]) for name in axis_name)
      else:
        shape = (logical_size(dynamic_axis_env[axis_name]),)
    return parallel_pure_rules[prim](*args, shape=shape, **params)

  pe.staged_out_calls.add(xla_pmap_p)  # type: ignore

parallel_pure_rules = {}  # type: ignore

class DynamicAxisEnvFrame(object):
  __slots__ = ["name", "pmap_trace", "hard_size"]
  def __init__(self, name, pmap_trace, hard_size):
    self.name = name
    self.pmap_trace = pmap_trace
    self.hard_size = hard_size

class DynamicAxisEnv(list):
  def __contains__(self, axis_name):
    return axis_name in (frame.name for frame in self)

  def __getitem__(self, axis_name):
    if axis_name not in self:
      raise NameError("unbound axis name: {}".format(axis_name))
    for frame in reversed(self):
      if frame.name == axis_name:
        return frame

    raise AssertionError

  @property
  def sizes(self):
    return tuple(frame.hard_size for frame in self)

  @property
  def nreps(self):
    return prod(frame.hard_size for frame in self)

class _ThreadLocalState(threading.local):
  def __init__(self):
    self.dynamic_axis_env = DynamicAxisEnv()

_thread_local_state = _ThreadLocalState()

def device_put(x, devices: Sequence[xb.xla_client.Device], replicate: bool=False) -> List[xb.xla_client._xla.PyLocalBuffer]:
  """Call device_put on a sequence of devices and return a flat sequence of buffers."""
  if replicate:
    return list(it.chain.from_iterable(xla.device_put(x, device) for device in devices))
  else:
    return list(it.chain.from_iterable(xla.device_put(val, device) for val, device in safe_zip(x, devices)))
