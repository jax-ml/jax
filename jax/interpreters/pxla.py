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

from collections import defaultdict
from itertools import product
import operator as op
from typing import (Any, Callable, Dict, List, Optional, Sequence, Set, Tuple,
                    Type, Union)

from absl import logging
import numpy as onp

from ..config import flags
from .. import core
from .. import linear_util as lu
from .. import lazy
from ..abstract_arrays import ConcreteArray, ShapedArray, array_types
from ..core import Var, Literal
from ..util import (partial, unzip2, unzip3, prod, safe_map, safe_zip,
                    extend_name_stack, wrap_name)
from ..lib import xla_bridge as xb
from ..lib import xla_client as xc
from ..tree_util import tree_flatten, tree_map
from .batching import broadcast, not_mapped, moveaxis
from . import batching
from . import partial_eval as pe
from . import xla
from . import ad


xops = xc.ops

FLAGS = flags.FLAGS

unsafe_map, map = map, safe_map

Index = Union[int, slice, Tuple[Union[int, slice], ...]]


# TODO(skye): make this a namedtuple. This may allow us to use ShardingSpecs in
# performance-sensitive code, e.g. shard_args.
class ShardingSpec:
  """Describes how a logical array is sharded across devices.

  Note this does not specify the physical devices to be sharded across, nor a
  logical ordering of data shards. Use `spec_to_indices` to resolve a
  ShardingSpec to the specific logical ordering expected throughout the system.

  Sharding includes "replication", where the same data is present on multiple
  devices. Replication is always applied to the entire logical array, i.e. the
  whole array is copied N times, although each copy may still be sharded
  according to the rest of the ShardingSpec. This means that unlike other kinds
  of sharding, replication isn't associated with a particular logical array
  axis. However, it does have a position relative to the logical array axes,
  which is necessary to specify how replication is mapped to devices in
  `spec_to_indices`. One to think about this is if you added an extra length-N
  logical axis containing the N copies of the original array, where would that
  new axis go? This would affect the final buffer order computed in
  `spec_to_indices`.

  Attributes:
    shards_per_axis: a tuple the same length as the array shape. Indicates how
      many shards each axis is divided into. Each axis must be divided into
      equal-sized shards (i.e. array_shape[i] % shards_per_axis[i] == 0).
    is_axis_materialized: a tuple the same length as the array shape. Indicates
      whether each axis of the array is represented in the on-device shape
      (i.e. sum(is_axis_materialized) == len(device_buffer.shape())). Any
      unmaterialized axes must be sharded into size-1 chunks
      (i.e. array_shape[i] == shards_per_axis[i]).
    replication_factors: list of tuples of (factor, index) describing how many
      times the array is replicated and before which logical axis index each
      virtual replication axis is inserted.
  """

  def __init__(self,
               shards_per_axis: Tuple[int, ...],
               is_axis_materialized: Tuple[bool, ...],
               replication_factors: List[Tuple[int, int]]):
    assert len(shards_per_axis) == len(is_axis_materialized)
    self.shards_per_axis = shards_per_axis
    self.is_axis_materialized = is_axis_materialized
    self.replication_factors = replication_factors

  def __eq__(self, other):
    return (self.shards_per_axis == other.shards_per_axis and
            self.is_axis_materialized == other.is_axis_materialized and
            self.replication_factors == other.replication_factors)

  def __repr__(self):
    return ("ShardingSpec(shards_per_axis=%s, is_axis_materialized=%s, "
            "replication_factors=%s)" %
            (self.shards_per_axis, self.is_axis_materialized,
             self.replication_factors))


def spec_to_indices(shape: Tuple[int, ...],
                    sharding_spec: ShardingSpec) -> Tuple[Index, ...]:
  """Returns numpy-style indices corresponding to sharding_spec.

  Each index describes a shard of the array. The order of the indices is the
  same as the device_buffers of a ShardedDeviceArray (i.e. the data is laid out
  row-major, with replication treated as an extra innermost dimension).

  Args:
    shape: The shape of the logical array being sharded.
    sharding_spec: Describes how the array is sharded.

  Returns:
    A tuple of length `prod(sharding_spec.shards_per_axis) *
    prod(factor for factor, index in sharding_spec.replication_factors)`. Each
    element is an int, a slice object with step=1, or a tuple thereof, to be
    treated as an index into the full logical array.
  """
  assert len(shape) == len(sharding_spec.shards_per_axis)
  if not shape:
    # special case: scalars can only be indexed by `()`
    total_replication_factor = int(prod(factor for factor, index in
                                        sharding_spec.replication_factors))
    return ((),) * total_replication_factor

  replication_factors = sorted(sharding_spec.replication_factors,
                               key=op.itemgetter(1))
  logical_index = 0
  indices_per_mesh_axis = []
  for mesh_index in range(len(shape) + len(sharding_spec.replication_factors)):
    if replication_factors and replication_factors[0][1] == logical_index:
      # Insert a placeholder `None` to represent a replication factor. These
      # will all be removed later, since they don't correspond to logical axes.
      factor, _ = replication_factors.pop(0)
      indices_per_mesh_axis.append([None] * factor)
    else:
      indices = _axis_indices(
          shape[logical_index],
          sharding_spec.shards_per_axis[logical_index],
          sharding_spec.is_axis_materialized[logical_index])
      indices_per_mesh_axis.append(indices)
      logical_index += 1
  assert logical_index == len(shape) and not replication_factors

  indices = list(product(*indices_per_mesh_axis))

  # remove placeholder `None`s and trailing colons, then unwrap
  # single-element tuples
  def canonicalize(index):
    index = [i for i in index if i is not None]
    while len(index) > 1 and index[-1] == slice(None):
      index.pop(-1)
    assert index
    if len(index) == 1:
      return index[0]
    return tuple(index)
  return tuple(canonicalize(index) for index in indices)


def _axis_indices(axis_size, num_shards, is_materialized):
  if not is_materialized:
    assert axis_size == num_shards, f'{axis_size} != {num_shards}'
    return list(range(axis_size))
  if num_shards == 1:
    return [slice(None)]
  shard_size, ragged = divmod(axis_size, num_shards)
  assert not ragged
  return [slice(i * shard_size, (i + 1) * shard_size) for i in range(num_shards)]


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
    lambda x, devices, _: [xla.device_put(core.unit, d) for d in devices]
def _shard_array(x, devices, indices):
  return [xla.device_put(x[i], d) for (i, d) in zip(indices, devices)]
for _t in array_types:
  shard_arg_handlers[_t] = _shard_array

def _shard_device_array(x, devices, indices):
  start_indices, limit_indices, removed_dims = map(tuple, unzip3(
      _as_slice_indices(x, idx) for idx in indices))
  shards = x._multi_slice(start_indices, limit_indices, removed_dims)
  return [xla.device_put(s, d) for s, d in zip(shards, devices)]
shard_arg_handlers[xla.DeviceArray] = _shard_device_array

# NOTE(skye): we could refactor to generate _multi_slice parameters directly
# from the input ShardingSpec, rather than the indices. However, this would
# require duplicating the ordering logic of spec_to_indices, which is more
# subtle and more likely to change than the index logic we have to support here.
def _as_slice_indices(arr: xla.DeviceArray, idx: Index) -> Tuple[
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
      assert isinstance(sub_idx, slice)
      assert isinstance(sub_idx.start, int)
      assert isinstance(sub_idx.stop, int)
      start_indices[dim] = sub_idx.start
      limit_indices[dim] = sub_idx.stop

  return tuple(start_indices), tuple(limit_indices), tuple(removed_dims) # type: ignore


def shard_aval(size, aval):
  try:
    return shard_aval_handlers[type(aval)](size, aval)
  except KeyError as err:
    raise TypeError("No shard_aval handler for type: {}".format(type(aval))
                    ) from err
shard_aval_handlers: Dict[Type[core.AbstractValue], Callable[[int, Any], Any]] = {}
shard_aval_handlers[core.AbstractUnit] = lambda size, x: x
def _shard_abstract_array(size, x):
  if not x.shape:
    raise ValueError("Scalar cannot be split across {} shards.".format(size))
  if x.shape[0] != size:
    raise ValueError("Axis size {} does not match leading dimension of "
                     "shape {}".format(size, x.shape))
  return ShapedArray(x.shape[1:], x.dtype)
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

class ShardedDeviceArray(xla.DeviceArray):
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
    # TODO(skye): this is temporary staging while we switch users over to
    # providing sharding_spec. It assumes that any pre-existing callers are
    # creating pmap-style ShardedDeviceArrays.
    if device_buffers is None:
      device_buffers = sharding_spec
      sharded_aval = ShapedArray(aval.shape[1:], aval.dtype)
      sharding_spec = _pmap_sharding_spec(aval.shape[0], aval.shape[0],
                                          1, None, sharded_aval, True)

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
      npy_value = onp.empty(self.aval.shape, self.aval.dtype)
      for i in self.one_replica_buffer_indices:
        npy_value[self.indices[i]] = self.device_buffers[i].to_py()
      self._npy_value = npy_value
    return self._npy_value

  def __getitem__(self, idx):
    if self._npy_value is None and idx in self.indices:
      buf = self.device_buffers[self.indices.index(idx)]
      aval = ShapedArray(buf.shape().dimensions(), self.aval.dtype)
      return xla.DeviceArray(aval, None, lazy.array(aval.shape), buf)
    else:
      return super(ShardedDeviceArray, self).__getitem__(idx)


def _hashable_index(idx):
  return tree_map(lambda x: (x.start, x.stop) if type(x) == slice else x,
                  idx)

# The fast path is handled directly in shard_args().
# TODO(skye): is there a simpler way to rewrite this using sharding_spec?
def _shard_sharded_device_array_slow_path(x, devices, indices):
  candidates = defaultdict(list)
  for buf, idx in zip(x.device_buffers, x.indices):
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
  return xb.constant(c, onp.asarray(val), canonicalize_types=canonicalize_types)
xb.register_constant_handler(ShardedDeviceArray, _sharded_device_array_constant_handler)

core.pytype_aval_mappings[ShardedDeviceArray] = ConcreteArray
xla.device_put_handlers[ShardedDeviceArray] = xla._device_put_array
xla.pytype_aval_mappings[ShardedDeviceArray] = op.attrgetter('aval')
xla.canonicalize_dtype_handlers[ShardedDeviceArray] = identity


### the xla_pmap primitive and its rules are comparable to xla_call in xla.py

def xla_pmap_impl(fun: lu.WrappedFun, *args, backend, axis_name, axis_size,
                  global_axis_size, devices, name, mapped_invars, donated_invars):
  abstract_args = unsafe_map(xla.abstractify, args)
  compiled_fun = parallel_callable(fun, backend, axis_name, axis_size,
                                   global_axis_size, devices, name, mapped_invars,
                                   donated_invars, *abstract_args)
  return compiled_fun(*args)

@lu.cache
def parallel_callable(fun, backend, axis_name, axis_size, global_axis_size,
                      devices, name, mapped_invars, donated_invars, *avals):
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
    local_devices = None

  sharded_avals = tuple(shard_aval(axis_size, aval) if m else aval
                        for m, aval in zip(mapped_invars, avals))
  with core.extend_axis_env(axis_name, axis_size):
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, sharded_avals)
  jaxpr, uses_outfeed = xla.apply_outfeed_rewriter(jaxpr)

  # TODO(skye,mattjj): allow more collectives on multi-host as we test them, but
  # for now raise an error
  if devices is not None:
    is_multi_host_pmap = any(d.host_id != xb.host_id() for d in devices)
  else:
    is_multi_host_pmap = xb.host_count() > 1
  if is_multi_host_pmap:
    used_collectives = set(xla.jaxpr_collectives(jaxpr))
    if not used_collectives.issubset(multi_host_supported_collectives):
      msg = "using collectives that aren't supported for multi-host: {}"
      raise TypeError(msg.format(", ".join(map(str, used_collectives))))


  # TODO(skyewm): replace this with a chain of pmaps and/or sharded_jits
  jaxpr_replicas = xla.jaxpr_replicas(jaxpr)
  num_local_replicas = axis_size * jaxpr_replicas
  num_global_replicas = global_axis_size * jaxpr_replicas
  arg_parts, out_parts, num_partitions = _find_partitions(jaxpr)

  num_local_shards = num_local_replicas * num_partitions
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
              f"num_partitions={num_partitions}")

  axis_env = xla.AxisEnv(num_global_replicas, (axis_name,), (global_axis_size,), devices)

  tuple_args = len(sharded_avals) > 100  # pass long arg lists as tuple for TPU

  c = xb.make_computation_builder("pmap_{}".format(fun.__name__))
  xla_consts = map(partial(xb.constant, c), consts)
  xla_args = xla._xla_callable_args(c, sharded_avals, tuple_args,
                                    map(op.not_, mapped_invars), arg_parts)
  out_nodes = xla.jaxpr_subcomp(c, jaxpr, backend, axis_env, xla_consts,
                                extend_name_stack(wrap_name(name, 'pmap')), *xla_args)
  build_out_tuple = partial(xops.Tuple, c, out_nodes)
  if out_parts is not None:
    out_tuple = xb.with_sharding(c, out_parts, build_out_tuple)
  else:
    out_tuple = build_out_tuple()
  backend = xb.get_backend(backend)
  if backend.platform == "tpu":
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
    if num_global_replicas > num_local_replicas:
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
  device_assignment = onp.array(device_assignment).reshape(
      (num_global_replicas, num_partitions))
  compile_options = xb.get_compile_options(
      num_replicas=num_global_replicas,
      num_partitions=num_partitions,
      device_assignment=device_assignment)
  compile_options.parameter_is_tupled_arguments = tuple_args
  compiled = backend.compile(built, compile_options=compile_options)

  arg_parts_ = arg_parts or [None] * len(avals)
  input_sharding_specs = [
      _pmap_sharding_spec(num_local_replicas, axis_size, num_partitions, parts,
                          aval, mapped)
      if aval is not core.abstract_unit else None
      for aval, parts, mapped in zip(sharded_avals, arg_parts_, mapped_invars)]
  input_indices = [spec_to_indices(aval.shape, spec)
                   if spec is not None else None
                   for aval, spec in zip(avals, input_sharding_specs)]
  handle_args = partial(shard_args, compiled.local_devices(), input_indices)
  handle_outs = avals_to_results_handler(
      axis_size, num_local_replicas, num_partitions, out_parts, out_avals)

  return partial(execute_replicated, compiled, uses_outfeed, backend, handle_args,
                 handle_outs)

multi_host_supported_collectives: Set[core.Primitive] = set()


PartitionsOrReplicated = Optional[Tuple[int, ...]]

def _find_partitions(jaxpr) -> Tuple[
    Optional[Tuple[PartitionsOrReplicated, ...]],
    Optional[Tuple[PartitionsOrReplicated, ...]],
    int]:
  """Returns (in_partitions, out_partitions, num_partitions)."""
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == "sharded_call":
      if len(jaxpr.eqns) > 1:
        raise NotImplementedError(
            "pmap of sharded_jit + non-sharded operations not yet implemented.")
      num_partitions = reconcile_num_partitions(eqn.params["call_jaxpr"],
                                                eqn.params["num_partitions"])
      return (eqn.params["in_parts"], eqn.params["out_parts_thunk"](),
              num_partitions)
  return None, None, 1


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
  num_partitions_set = set(onp.prod(spec) for spec in partition_specs)
  if len(num_partitions_set) > 1:
    raise ValueError(
        f"All partition specs must use the same number of total partitions, "
        f"got {partitions}, with distinct number of partitions "
        f"{num_partitions_set} (the total number of partitions is the product "
        f"of a partition spec)")
  assert len(num_partitions_set) == 1
  return num_partitions_set.pop()


class ResultToPopulate: pass
result_to_populate = ResultToPopulate()

def avals_to_results_handler(size, nrep, npart, out_parts, out_avals):
  nouts = len(out_avals)
  if out_parts is None:
    out_parts = (None,) * len(out_avals)

  # TODO(mattjj,skyewm): can probably clean up this logic
  out_specs = [_pmap_sharding_spec(nrep, size, npart, parts, aval, True)
               if aval is not core.abstract_unit else None
               for parts, aval in zip(out_parts, out_avals)]
  out_indices = [spec_to_indices(core.unmapped_aval(size, aval).shape, spec)
                 if aval is not core.abstract_unit else None
                 for aval, spec in zip(out_avals, out_specs)]  # pytype: disable=attribute-error
  handlers = [aval_to_result_handler(spec, idcs, core.unmapped_aval(size, aval))
              for spec, idcs, aval in zip(out_specs, out_indices, out_avals)]

  def handler(out_bufs):
    assert nrep * npart == len(out_bufs)
    buffers = [[result_to_populate] * nrep * npart for _ in range(nouts)]
    for r, tuple_buf in enumerate(out_bufs):
      for i, buf in enumerate(tuple_buf):
        buffers[i][r] = buf
    assert not any(buf is result_to_populate for bufs in buffers
                   for buf in bufs)
    return [h(bufs) for h, bufs in zip(handlers, buffers)]
  return handler

def _pmap_sharding_spec(nrep, axis_size, npart, parts, sharded_aval, mapped):
  """Sharding spec for arguments or results of a pmap.
  Args:
    nrep: number of local XLA replicas (product of local axis sizes)
    axis_size: local axis size for outer pmap
    npart: total number of XLA partitions (required by sharded_jit calls)
    parts: the partitioning of the value or None
    sharded_aval: the aval of the value inside the outer pmap, an instance of
      a ShapedArray.
    mapped: whether the value is mapped in the outer pmap
  Returns:
    A ShardingSpec.
  """
  assert isinstance(sharded_aval, ShapedArray), sharded_aval
  replication_factor, ragged = divmod(nrep, axis_size)
  assert not ragged
  # get the sharding spec from inner sharded_jits as if we weren't in a pmap
  shard_spec = partitioned_sharding_spec(npart, parts, sharded_aval)
  assert shard_spec is not None  # hint for pytype
  if mapped:
    # replication_factor represents the product of inner pmaps, so it goes
    # after the outer pmapped axis at index 0
    replication_factors = [] if replication_factor == 1 else [(replication_factor, 1)]
    replication_factors.extend((factor, index + 1) for factor, index
                                in shard_spec.replication_factors)
    return ShardingSpec(
        shards_per_axis=(axis_size,) + shard_spec.shards_per_axis,
        is_axis_materialized=(False,) + shard_spec.is_axis_materialized,
        replication_factors=replication_factors)
  else:
    return ShardingSpec(
        shards_per_axis=shard_spec.shards_per_axis,
        is_axis_materialized=shard_spec.is_axis_materialized,
        replication_factors=[(replication_factor * axis_size, 0)] +
            shard_spec.replication_factors)


def partitioned_sharding_spec(num_partitions: int,
                              partitions: Optional[Sequence[int]], aval):
  if partitions is None:
    # hit by both replicated sharded_jit and no sharded_jit
    # we drop the extra singleton replication factor in the latter case
    # where we put the replication doesn't matter because all the shards_per_axis
    # are 1
    return ShardingSpec(
        shards_per_axis=(1,) * len(aval.shape),
        is_axis_materialized=(True,) * len(aval.shape),
        replication_factors=[] if num_partitions == 1 else [(num_partitions, 0)])
  else:
    assert len(partitions) == len(aval.shape)
    return ShardingSpec(
        shards_per_axis=tuple(partitions),
        is_axis_materialized=(True,) * len(aval.shape),
        replication_factors=[])


def execute_replicated(compiled,
                       uses_outfeed, backend, in_handler, out_handler, *args):
  xla.check_before_outfeed_execution(uses_outfeed)
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

# Set param update handlers to update `donated_invars` just like xla_call_p
pe.call_param_updaters[xla_pmap_p] = pe.call_param_updaters[xla.xla_call_p]
ad.call_param_updaters[xla_pmap_p] = ad.call_param_updaters[xla.xla_call_p]
ad.call_transpose_param_updaters[xla_pmap_p] = \
    ad.call_transpose_param_updaters[xla.xla_call_p]

def _pmap_translation_rule(c, axis_env,
                           in_nodes, name_stack, axis_name, axis_size,
                           global_axis_size, devices, name,
                           call_jaxpr, *, backend=None, mapped_invars,
                           donated_invars):
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
    _xla_shard(c, aval, new_env, in_node) if in_node_mapped else in_node
    for aval, in_node, in_node_mapped in zip(in_avals, in_nodes, mapped_invars))

  sharded_outs = xla.jaxpr_subcomp(
      c, call_jaxpr, backend, new_env, (),
      extend_name_stack(name_stack, wrap_name(name, 'pmap')), *in_nodes_sharded)
  out_avals = [v.aval for v in call_jaxpr.outvars]
  outs = [_xla_unshard(c, aval, new_env, shard, backend=backend)
          for aval, shard in zip(out_avals, sharded_outs)]
  return xops.Tuple(c, outs)

xla.call_translations[xla_pmap_p] = _pmap_translation_rule
ad.primitive_transposes[xla_pmap_p] = partial(ad.map_transpose, xla_pmap_p)

def _xla_shard(c, aval, axis_env, x):
  if aval is core.abstract_unit:
    return x
  elif isinstance(aval, ShapedArray):
    dims = list(c.get_shape(x).dimensions())
    zero = xb.constant(c, onp.zeros((), dtype=onp.uint32))
    idxs = [_unravel_index(c, axis_env)] + [zero] * (len(dims) - 1)
    return xops.Reshape(xops.DynamicSlice(x, idxs, [1] + dims[1:]), dims[1:])
  else:
    raise TypeError((aval, c.get_shape(x)))

# TODO(b/110096942): more efficient gather
def _xla_unshard(c, aval, axis_env, x, backend):
  if aval is core.abstract_unit:
    return x
  elif isinstance(aval, ShapedArray):
    # TODO(mattjj): remove this logic when AllReduce PRED supported on CPU / GPU
    convert_bool = (onp.issubdtype(aval.dtype, onp.bool_)
                    and xb.get_backend(backend).platform in ('cpu', 'gpu'))
    if convert_bool:
      x = xops.ConvertElementType(x, xb.dtype_to_etype(onp.float32))

    xla_shape = c.get_shape(x)
    dims = list(xla_shape.dimensions())
    padded = xops.Broadcast(xb.constant(c, onp.array(0, xla_shape.numpy_dtype())),
                         [axis_env.sizes[-1]] + dims)
    zero = xb.constant(c, onp.zeros((), dtype=onp.uint32))
    idxs = [_unravel_index(c, axis_env)] + [zero] * len(dims)
    padded = xops.DynamicUpdateSlice(padded, xops.Reshape(x, [1] + dims), idxs)
    replica_groups_protos = xc.make_replica_groups(
      xla.axis_groups(axis_env, axis_env.names[-1]))
    out = xops.CrossReplicaSum(padded, replica_groups_protos)

    # TODO(mattjj): remove this logic when AllReduce PRED supported on CPU / GPU
    if convert_bool:
      nonzero = xops.Ne(out, xb.constant(c, onp.array(0, dtype=onp.float32)))
      out = xops.ConvertElementType(nonzero, xb.dtype_to_etype(onp.bool_))
    return out
  else:
    raise TypeError((aval, c.get_shape(x)))

def _unravel_index(c, axis_env):
  div = xb.constant(c, onp.array(axis_env.nreps // prod(axis_env.sizes), onp.uint32))
  mod = xb.constant(c, onp.array(axis_env.sizes[-1], onp.uint32))
  return xops.Rem(xops.Div(xops.ReplicaId(c), div), mod)


def soft_pmap_impl(fun: lu.WrappedFun, *args, axis_name, axis_size, mapped_invars):
  abstract_args = unsafe_map(xla.abstractify, args)
  compiled_fun = _soft_pmap_callable(fun, axis_name, axis_size, mapped_invars,
                                     *abstract_args)
  return compiled_fun(*args)

@lu.cache
def _soft_pmap_callable(fun, axis_name, axis_size, mapped_invars, *avals):
  mapped_avals = [core.mapped_aval(axis_size, aval) if m else aval
                  for m, aval in zip(mapped_invars, avals)]
  with core.extend_axis_env(axis_name, axis_size):
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, mapped_avals)
  jaxpr, uses_outfeed = xla.apply_outfeed_rewriter(jaxpr)

  num_devices = xb.local_device_count()
  chunk_size, ragged = divmod(axis_size, num_devices)
  if ragged:
    msg = f"number of devices {num_devices} must divide axis size {axis_size}"
    raise NotImplementedError(msg)

  jaxpr, _, consts = _soft_pmap_jaxpr(jaxpr, consts, mapped_invars,
                                      axis_name, chunk_size)
  jaxpr_replicas = xla.jaxpr_replicas(jaxpr)
  if jaxpr_replicas != 1: raise NotImplementedError

  tuple_args = len(avals) > 100  # pass long arg lists as tuple for TPU

  c = xb.make_computation_builder("soft_pmap_{}".format(fun.__name__))
  xla_consts = map(partial(xb.constant, c), consts)
  chunked_avals = [core.unmapped_aval(chunk_size, aval) if m else aval
                   for m, aval in zip(mapped_invars, mapped_avals)]
  xla_args = xla._xla_callable_args(c, chunked_avals, tuple_args)
  axis_env = xla.AxisEnv(num_devices, (axis_name,), (num_devices,), None)
  out_nodes = xla.jaxpr_subcomp(c, jaxpr, None, axis_env, xla_consts,
                                'soft_pmap', *xla_args)
  built = c.Build(xops.Tuple(c, out_nodes))

  compile_options = xb.get_compile_options(
          num_replicas=num_devices, num_partitions=1, device_assignment=None)
  compile_options.tuple_arguments = tuple_args
  backend = xb.get_backend(None)
  compiled = backend.compile(built, compile_options=compile_options)

  input_specs = [
      ShardingSpec(shards_per_axis=(num_devices,) + (1,) * (aval.ndim - 1),
                   is_axis_materialized=(True,) * aval.ndim,
                   replication_factors=[])
      if mapped else
      ShardingSpec(shards_per_axis=(1,) * aval.ndim,
                   is_axis_materialized=(False,) + (True,) * (aval.ndim - 1),
                   replication_factors=[(num_devices, 0)])
      for aval, mapped in zip(avals, mapped_invars)]
  input_indices = [spec and spec_to_indices(aval.shape, spec)
                   for aval, spec in zip(avals, input_specs)]
  handle_args = partial(shard_args, compiled.local_devices(), input_indices)
  handle_outs = soft_pmap_avals_to_results_handler(num_devices, chunk_size, out_avals)

  return partial(execute_replicated, compiled, uses_outfeed, backend,
                 handle_args, handle_outs)

def _soft_pmap_jaxpr(jaxpr, consts, mapped_invars, axis_name, chunk_size):
  fun = partial(_soft_pmap_interp, chunk_size, jaxpr, consts, mapped_invars)
  in_avals = [core.unmapped_aval(chunk_size, v.aval) if m else v.aval
              for v, m in zip(jaxpr.invars, mapped_invars)]
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
      rule = batching.get_primitive_batcher(eqn.primitive)
      in_axes = [0 if m else batching.not_mapped for m in in_mapped]
      out_vals, out_axes = rule(in_vals, in_axes, **eqn.params)
      if not eqn.primitive.multiple_results:
        out_vals, out_axes = [out_vals], [out_axes]
      out_vals = [moveaxis(x, d, 0) if d is not not_mapped and d != 0 else x
                  for x, d in zip(out_vals, out_axes)]
      out_mapped = [d is not not_mapped for d in out_axes]
    map(write, eqn.outvars, out_vals, out_mapped)

  out_vals, out_mapped = unzip2(map(read, jaxpr.outvars))
  out_vals = [out if mapped else broadcast(out, chunk_size, 0)
              for out, mapped in zip(out_vals, out_mapped)]
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
    return [h(bufs) for h, bufs in zip(handlers, buffers)]
  return handler

def soft_pmap_aval_to_result_handler(chunk_size, num_devices, aval):
  axis_size = chunk_size * num_devices
  if aval is core.abstract_unit:
    return lambda _: core.unit
  elif isinstance(aval, core.ShapedArray):
    new_aval = ShapedArray((axis_size,) + aval.shape, aval.dtype)
    spec = ShardingSpec(shards_per_axis=(num_devices,) + (1,) * aval.ndim,
                        is_axis_materialized=(True,) * new_aval.ndim,
                        replication_factors=[])
    return lambda bufs: ShardedDeviceArray(new_aval, spec, bufs)
  else:
    raise TypeError(aval)

soft_pmap_p = core.MapPrimitive('soft_pmap')
soft_pmap = soft_pmap_p.bind
soft_pmap_p.def_impl(soft_pmap_impl)

soft_pmap_rules: Dict[core.Primitive, Callable] = {}


def _axis_index_soft_pmap_rule(vals, mapped, chunk_size, *, axis_name):
  assert not vals and not mapped
  idx = core.axis_index(axis_name)
  return idx * chunk_size + onp.arange(chunk_size), True
soft_pmap_rules[core.axis_index_p] = _axis_index_soft_pmap_rule
