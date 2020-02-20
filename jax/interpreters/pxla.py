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


from collections import namedtuple, defaultdict
from contextlib import contextmanager
import itertools as it
import operator as op
import threading

from absl import logging
import numpy as onp

from ..config import flags
from .. import core
from .. import dtypes
from .. import linear_util as lu
from .. import lazy
from ..abstract_arrays import (ConcreteArray, ShapedArray, array_types,
                               raise_to_shaped)
from ..util import (partial, unzip2, concatenate, prod, safe_map, safe_zip,
                    extend_name_stack, wrap_name)
from ..lib import xla_bridge as xb
from ..map_util import not_mapped, shard_aval
from ..tree_util import tree_map
from .xla import aval_to_xla_shape, xla_destructure
from .batching import broadcast
from . import batching
from . import partial_eval as pe
from . import xla
from . import ad

FLAGS = flags.FLAGS

_map = safe_map


### util

def identity(x): return x

def shard_args(backend, devices, indices, axis_size, tuple_args, args):
  """Shard or replicate each argument data array.

  Args:
    backend: the platform to be used
    devices: list of Devices mapping replica index to a physical device.
    indices: list of n_args lists of logical indices, each with the same
      length as `devices`. Each logical index can be an int, a slice object,
      or a tuple thereof. This describes how the input should be sharded
      across devices; see the ShardedDeviceArray constructor for details.
    axis_size: int, size of the leading axis to be sharded.
    tuple_args: whether to wrap the returned args in an XLA tuple.
    args: a sequence of JaxTypes representing arguments to be sharded along
      their leading axes and placed on `devices`.

  Returns:
    A list of lists (or XLA tuples) of device buffers, each with the same
    length as `devices` and indexed by replica number, so that the nth element
    of each is the argument to be passed to the nth replica.
  """
  nargs, nrep = len(args), len(devices)
  buffers = [[None] * nargs for _ in range(nrep)]
  for a, (arg, logical_indices) in enumerate(zip(args, indices)):
    # The shard_arg_handlers allow an extensible set of types to be sharded, but
    # inline handling for ShardedDeviceArray as a special case for performance
    if type(arg) is ShardedDeviceArray and logical_indices == arg.logical_indices:
      for r, buf in enumerate(arg.device_buffers):
        buffers[r][a] = (buf if buf.device() == devices[r]
                         else buf.copy_to_device(devices[r]))
    else:
      bufs = shard_arg_handlers[type(arg)](arg, devices, logical_indices)
      for r, buf in enumerate(bufs):
        buffers[r][a] = buf

  if tuple_args:
    buffers = [[xla.make_tuple(bufs, devices[r], backend)]
               for r, bufs in enumerate(buffers)]

  return buffers


shard_arg_handlers = {}
shard_arg_handlers[core.Unit] = \
    lambda x, devices, _: [xla.device_put(core.unit, d) for d in devices]
def _shard_array(x, devices, indices):
  return [xla.device_put(x[idx], device)
          for idx, device in zip(indices, devices)]
for _t in array_types:
  shard_arg_handlers[_t] = _shard_array
shard_arg_handlers[xla.DeviceArray] = _shard_array

def _shard_scalar(x, devices, indices):
  return [xla.device_put(dtypes.coerce_to_array(x[idx] if idx else x), device)
          for idx, device in zip(indices, devices)]
for _t in dtypes.python_scalar_dtypes.keys():
  shard_arg_handlers[_t] = _shard_scalar

def _shard_device_array(x, devices, indices):
  nrep = len(devices)
  return (xla.device_put(x[indices[r]], devices[r])
          for r in range(nrep))
shard_arg_handlers[xla.DeviceArray] = _shard_device_array

def aval_to_result_handler(axis_size, nrep, aval):
  try:
    return pxla_result_handlers[type(aval)](axis_size, nrep, aval)
  except KeyError:
    raise TypeError("No pxla_result_handler for type: {}".format(type(aval)))
pxla_result_handlers = {}
pxla_result_handlers[core.AbstractUnit] = lambda *_: lambda _: (core.unit, not_mapped)
def array_result_handler(axis_size, nrep, aval):
  full_aval = ShapedArray((axis_size,) + aval.shape, aval.dtype)
  indices = assign_shards_to_replicas(nrep, axis_size)
  return lambda buffers: (ShardedDeviceArray(full_aval, indices, buffers), 0)
pxla_result_handlers[ShapedArray] = array_result_handler
pxla_result_handlers[ConcreteArray] = array_result_handler


def assign_shards_to_replicas(nrep, size):
  """Produce a mapping from replica id to shard index.

  Args:
    nrep: int, number of replicas (a computation-dependent value).
    size: int, size of the data array axis being sharded.

  Returns:
    A tuple of integers of length nrep in which the elements take on values from
    0 to size-1. Replica n is assigned shard data_array[indices[n]].
  """
  groupsize, ragged = divmod(nrep, size)
  assert not ragged
  indices = onp.tile(onp.arange(size)[:, None], (1, groupsize))
  return list(indices.ravel())


### applying parallel primitives in op-by-op Python dispatch

# There are at least two cases where we might want to evaluate a parallel
# primitive dispatched from Python, rather than being staged out:
#   1. axis_size = psum(1, 'axis_name'),
#   2. to enable an implicit outermost pmap-like context for multi-host
#      multi-controller SPMD programs.
# In each case, we can't rely on any data dependence on a pmap trace; instead we
# need some dynamic context, basically modeling the axis name environment stack.
# To handle the former case, we don't need to communicate at all; we instead
# have a table of parallel_pure_rules. To handle the latter case, we'll have a
# globally-scoped root environment frame and compile and execute a single-op
# XLA collective.

class DynamicAxisEnvFrame(object):
  __slots__ = ["name", "pmap_trace", "hard_size", "soft_trace", "soft_size"]
  def __init__(self, name, pmap_trace, hard_size):
    self.name = name
    self.pmap_trace = pmap_trace
    self.hard_size = hard_size
    self.soft_trace = None
    self.soft_size = None

class DynamicAxisEnv(list):
  def __contains__(self, axis_name):
    return axis_name in (frame.name for frame in self)

  def __getitem__(self, axis_name):
    if axis_name not in self:
      raise NameError("unbound axis name: {}".format(axis_name))
    for frame in reversed(self):
      if frame.name == axis_name:
        return frame
    else:
      assert False

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
  logical_size = lambda frame: frame.hard_size * (frame.soft_size or 1)
  if isinstance(axis_name, (list, tuple)):
    shape = tuple(logical_size(dynamic_axis_env[name]) for name in axis_name)
  else:
    shape = (logical_size(dynamic_axis_env[axis_name]),)
  return parallel_pure_rules[prim](*args, shape=shape, **params)

parallel_pure_rules = {}


def axis_index(axis_name):
  dynamic_axis_env = _thread_local_state.dynamic_axis_env
  frame = dynamic_axis_env[axis_name]
  sizes = dynamic_axis_env.sizes[:dynamic_axis_env.index(frame)+1]
  nreps = dynamic_axis_env.nreps
  dummy_arg = frame.pmap_trace.pure(core.unit)
  if frame.soft_trace:
    dummy_arg = frame.soft_trace.pure(dummy_arg)

  return axis_index_p.bind(dummy_arg, nreps=nreps, sizes=sizes,
                           soft_size=frame.soft_size, axis_name=axis_name)

def _axis_index_partial_eval(trace, _, **params):
  # This partial_eval rule adds the axis_index primitive into the jaxpr formed
  # during pmap lowering. It is like the standard JaxprTrace.process_primitive
  # rule except that we don't attempt to lower out of the trace.
  out_aval = ShapedArray((), onp.int32)
  out_tracer = pe.JaxprTracer(trace, pe.PartialVal((out_aval, core.unit)), None)
  eqn = pe.new_eqn_recipe([], [out_tracer], axis_index_p, params)
  out_tracer.recipe = eqn
  return out_tracer

def _axis_index_translation_rule(c, nreps, sizes, soft_size, axis_name):
  div = c.Constant(onp.array(nreps // prod(sizes), dtype=onp.uint32))
  mod = c.Constant(onp.array(sizes[-1], dtype=onp.uint32))
  unsigned_index = c.Rem(c.Div(c.ReplicaId(), div), mod)
  return c.ConvertElementType(unsigned_index, xb.dtype_to_etype(onp.int32))

axis_index_p = core.Primitive('axis_index')
xla.translations[axis_index_p] = _axis_index_translation_rule
pe.custom_partial_eval_rules[axis_index_p] = _axis_index_partial_eval


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

  Each device buffer resides on a distinct device, and has a corresponding index
  indicating what portion of the overall logical array it represents using
  standard NumPy indexing. It's possible for multiple buffers to represent the
  same portion of a ShardedDeviceArray; in that case, the buffers contain the
  same values but on different devices.

  Attributes:
    aval: A ShapedArray indicating the shape and dtype of this array.
    device_buffers: A list of PyLocalBuffers. Each buffer is on a different
      device and contains a portion of the full array. Multiple buffers may
      contain the same logical shard of the array.
    logical_indices: A list the same length as device_buffers. Each element is
      an int, a slice object with step=1, or a tuple thereof, to be treated as
      an index into the full array, and indicates what portion of the full array
      is stored in the corresponding device buffer. In other words, for each
      index/buffer pair, ``array[logical_indices[i]] ==
      device_buffers[i].to_py()`. Together, these indices cover the entire
      array, i.e. every array element is included in at least one index.
  """
  __slots__ = ["device_buffers", "logical_indices"]

  def __init__(self, aval, logical_indices, device_buffers):
    assert len(logical_indices) == len(device_buffers)
    # TODO(skye): assert that logical_indices fully cover aval and only contain
    # valid indices (e.g. no strides). Keep performance in mind though.
    self.aval = aval
    self.device_buffers = device_buffers
    self.logical_indices = list(logical_indices)
    self._npy_value = None
    if not core.skip_checks:
      assert type(aval) is ShapedArray

  def copy_to_host_async(self):
    if self._npy_value is None:
      for buf in self.device_buffers:
        buf.copy_to_host_async()

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
      # TODO(skye): remove this to avoid transferring replicated buffers?
      self.copy_to_host_async()
      self._npy_value = onp.empty(self.aval.shape, self.aval.dtype)
      # TODO(skye): benchmark and possibly switch to a set (maybe with with
      # cached hashable indices?)
      already_copied_indices = []
      for buf, idx in zip(self.device_buffers, self.logical_indices):
        if idx in already_copied_indices:
          continue
        self._npy_value[idx] = buf.to_py()
        already_copied_indices.append(idx)
    return self._npy_value

  def __getitem__(self, idx):
    if self._npy_value is None and idx in self.logical_indices:
      buf = self.device_buffers[self.logical_indices.index(idx)]
      aval = ShapedArray(buf.shape().dimensions(), self.aval.dtype)
      return xla.DeviceArray(aval, None, lazy.array(aval.shape), buf)
    else:
      return super(ShardedDeviceArray, self).__getitem__(idx)

def _match_sharded_axis(size, src, dst, arr):
  if arr is core.unit or src == dst:
    return arr
  assert isinstance(arr, ShardedDeviceArray)
  if dst == not_mapped:
    # this requires replicated buffers, but that requirement is unchecked
    # TODO(jekbradbury): keep track of replication in order to check it
    aval = ShapedArray(arr.shape[:src] + arr.shape[src + 1:], arr.dtype)
    scalar_types = (*dtypes.python_scalar_dtypes.keys(), onp.generic)
    indices = [() if isinstance(ind, scalar_types)
               else ind[:src] + ind[src + 1:]
               for ind in arr.logical_indices]
    return ShardedDeviceArray(aval, indices, arr.device_buffers)
  elif src == not_mapped:
    nrep = len(arr.device_buffers)
    aval = ShapedArray(arr.shape[:dst] + (size,) + arr.shape[dst:], arr.dtype)
    replica_indices = assign_shards_to_replicas(nrep, size)
    indices = [ind[:dst] + (i,) + ind[dst:]
               for i, ind in zip(replica_indices, arr.logical_indices)]
    return ShardedDeviceArray(aval, indices, arr.device_buffers)
  else:
    shape = list(arr.shape)
    aval_size = shape.pop(src)
    assert aval_size == size
    shape[dst:dst] = size
    aval = ShapedArray(shape, arr.dtype)
    indices = [ind[:src] + ind[src + 1:] + ind[src:src] for ind in arr.logical_indices]
    indices = [ind[:dst] + ind[-1:] + ind[dst:-1] for ind in indices]
    return ShardedDeviceArray(aval, indices, arr.device_buffers)

def _hashable_index(idx):
  return tree_map(lambda x: (x.start, x.stop) if type(x) == slice else x,
                  idx)

# The fast path is handled directly in shard_args().
def _shard_sharded_device_array_slow_path(x, devices, indices):
  candidates = defaultdict(list)
  for buf, buf_idx in zip(x.device_buffers, x.logical_indices):
    candidates[_hashable_index(buf_idx)].append(buf)

  bufs = []
  for idx, device in safe_zip(indices, devices):
    # Look up all buffers that contain the correct slice of the logical array.
    # TODO(skye): dedup equivalent indices?
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
  return c.Constant(onp.asarray(val), canonicalize_types=canonicalize_types)
xb.register_constant_handler(ShardedDeviceArray, _sharded_device_array_constant_handler)

core.pytype_aval_mappings[ShardedDeviceArray] = ConcreteArray
xla.device_put_handlers[ShardedDeviceArray] = xla._device_put_array
xla.pytype_aval_mappings[ShardedDeviceArray] = op.attrgetter('aval')
xla.canonicalize_dtype_handlers[ShardedDeviceArray] = identity


### the xla_pmap primitive and its rules are comparable to xla_call in xla.py

def xla_pmap_impl(fun, *args, backend, axis_name, axis_size, global_axis_size,
                  in_dims, out_dim_dests, devices, name):
  abstract_args = map(xla.abstractify, args)
  compiled_fun = parallel_callable(
      fun, backend, axis_name, axis_size, global_axis_size, in_dims, devices,
      name, *abstract_args)
  out_vals, out_dims = compiled_fun(*args)
  if not isinstance(out_dim_dests, tuple):
    out_dim_dests = out_dim_dests()
  assert len(args) == len(in_dims)
  assert len(out_vals) == len(out_dims) == len(out_dim_dests)
  return _map(partial(_match_sharded_axis, axis_size), out_dims, out_dim_dests, out_vals)

@lu.cache
def parallel_callable(fun, backend, axis_name, axis_size, global_axis_size,
                      in_dims, devices, name, *avals):
  if devices is not None and len(devices) == 0:
    raise ValueError("'devices' argument to pmap must be non-empty, or None.")

  # Determine global_axis_size for use in AxisEnv.
  if devices:
    assert global_axis_size is None  # Checked in api.py
    global_axis_size = len(devices)
  elif xb.host_count() > 1:
    if global_axis_size is None:
      # TODO(skye): relax this constraint or provide functionality for
      # automatically passing appropriate `devices`.
      # TODO(trevorcai): This check forces us to provide global_axis_size for
      # all pmaps on pmap-on-pod. Can we do it after tracing?
      if axis_size != xb.local_device_count():
        raise ValueError(
            "On multi-host platforms, the input to pmapped functions must have "
            "leading axis size equal to the number of local devices if no "
            "`devices` argument is specified. Got axis_size=%d, "
            "num_local_devices=%d" % (axis_size, xb.local_device_count()))
      global_axis_size = xb.device_count()
  else:
    if global_axis_size is not None:
      if global_axis_size != axis_size:
        raise ValueError(
            "Specified axis_size {} doesn't match received axis_size {}.".format(
                global_axis_size, axis_size))
    else:
      global_axis_size = axis_size

  log_priority = logging.WARNING if FLAGS.jax_log_compiles else logging.DEBUG
  logging.log(log_priority,
              "Compiling {} for {} devices with args {}.".format(
                  fun.__name__, global_axis_size, avals))

  if devices:
    local_devices = [d for d in devices if d.host_id == xb.host_id()]
    assert len(local_devices) > 0
  else:
    local_devices = None

  @lu.wrap_init
  def dynamic_fun(dummy, *args):
    with extend_dynamic_axis_env(axis_name, dummy._trace, global_axis_size):
      return fun.call_wrapped(*args)

  sharded_avals = _map(shard_aval, avals, in_dims)
  replicated = [d is not_mapped for d in in_dims]
  pvals = [pe.PartialVal((aval, core.unit)) for aval in sharded_avals]
  # We add a dummy first invar, to carry the trace's axis env to `dynamic_fun`
  pval = pe.PartialVal([core.abstract_unit, core.unit])
  with core.new_master(pe.StagingJaxprTrace, True) as master:
    jaxpr, (out_pvals, consts, env) = pe.trace_to_subjaxpr(
        dynamic_fun, master, False).call_wrapped([pval] + pvals)
    jaxpr.invars = jaxpr.invars[1:]  # ignore dummy
    assert not env
    del master
  out_pvs, out_consts = unzip2(out_pvals)

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

  if all(pv is None for pv in out_pvs):
    # When the output doesn't depend on the input we don't need to compile an
    # XLA computation at all; we handle this as a special case so we can stage
    # out multi-replica XLA computations regardless of the hardware available.
    # The 'None' values here are just dummies we know will be ignored.
    handlers = [_pval_to_result_handler(axis_size, None, pval, local_devices,
                                        backend)
                for pval in out_pvals]
    results = [handler(None) for handler in handlers]
    return lambda *_: unzip2(results)

  jaxpr_replicas = xla.jaxpr_replicas(jaxpr)
  num_local_replicas = axis_size * jaxpr_replicas
  num_global_replicas = global_axis_size * jaxpr_replicas
  axis_env = xla.AxisEnv(num_global_replicas, (axis_name,), (global_axis_size,), devices)

  tuple_args = len(sharded_avals) > 100  # pass long arg lists as tuple for TPU

  c = xb.make_computation_builder("pmap_{}".format(fun.__name__))
  xla_consts = _map(c.Constant, consts)
  xla_args = xla._xla_callable_args(c, sharded_avals, tuple_args, replicated)
  out_nodes = xla.jaxpr_subcomp(c, jaxpr, backend, axis_env, xla_consts,
                                extend_name_stack(wrap_name(name, 'pmap')), *xla_args)
  built = c.Build(c.Tuple(*out_nodes))

  if devices is None:
    if num_global_replicas > xb.device_count(backend):
      msg = ("compiling computation that requires {} replicas, but only {} XLA "
             "devices are available")
      raise ValueError(msg.format(num_global_replicas, xb.device_count(backend)))

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
          num_global_replicas)
  else:
    if num_local_replicas != len(local_devices):
      local_devices_str = ", ".join(map(str, local_devices))
      raise ValueError(
          "Leading axis size of input to pmapped function must equal the "
          "number of local devices passed to pmap. Got axis_size=%d, "
          "num_local_devices=%d.\n(Local devices passed to pmap: %s)"
          % (axis_size, len(local_devices), local_devices_str))
    if num_global_replicas != len(devices):
      raise ValueError("compiling computation that requires %s replicas, "
                       "but %s devices were specified"
                       % (num_global_replicas, len(devices)))

  device_assignment = tuple(d.id for d in devices)
  compiled = built.Compile(
      compile_options=xb.get_compile_options(
          num_replicas=num_global_replicas,
          num_partitions=1,
          device_assignment=device_assignment),
      backend=xb.get_backend(backend))

  # these are the indices into the sharded axis of any sharded input arrays
  # for each XLA logical device
  replica_indices = assign_shards_to_replicas(num_local_replicas, axis_size)
  # TODO(jekbradbury): check that this isn't a performance bottleneck, or fix
  arg_indices = [[() if dim is not_mapped else (slice(None),) * (dim - 1) + (idx,)
                  for idx in replica_indices] for dim in in_dims]
  handle_args = partial(shard_args, backend, compiled.local_devices(),
                        arg_indices, axis_size, tuple_args)
  handle_outs = _pvals_to_results_handler(axis_size, num_local_replicas,
                                          out_pvals, compiled.local_devices(),
                                          backend)
  return partial(execute_replicated, compiled, backend, handle_args, handle_outs)

multi_host_supported_collectives = set()

class ResultToPopulate(object): pass
result_to_populate = ResultToPopulate()

def _pvals_to_results_handler(size, nrep, out_pvals, devices, backend):
  nouts = len(out_pvals)
  handlers = [_pval_to_result_handler(size, nrep, pval, devices, backend)
              for pval in out_pvals]
  def handler(out_bufs):
    buffers = [[result_to_populate] * nrep for _ in range(nouts)]
    for r, tuple_buf in enumerate(out_bufs):
      for i, buf in enumerate(tuple_buf.destructure()):
        buffers[i][r] = buf
    assert not any(buf is result_to_populate for bufs in buffers
                   for buf in bufs)
    return [h(bufs) for h, bufs in zip(handlers, buffers)]
  return handler

def replicate(val, nrep, devices=None, backend=None):
  """Replicates ``val`` across multiple devices.

  Args:
    val: the value to be replicated.
    nrep: the number of replicas to create. If ``devices`` is set, must be equal
      to ``len(devices)``.
    devices: the devices to replicate across. If None, ``nrep`` will be used to
      generate a default device assignment.
    backend: string specifying which backend to use.

  Returns:
    A ShardedDeviceArray on ``nrep`` devices where each shard is equal to ``val``.
  """
  device_count = (len(devices) if devices else xb.local_device_count())
  if nrep > device_count:
    msg = ("Cannot replicate across %d replicas because only %d local devices "
           "are available." % (nrep, device_count))
    if devices:
      msg += (" (local devices = %s)"
              % ", ".join(map(str, devices)) if devices else str(None))
    raise ValueError(msg)

  if devices is None:
    assert nrep is not None
    devices = xb.get_backend(backend).get_default_device_assignment(nrep)
  assert nrep == len(devices)

  aval = xla.abstractify(val)
  indices = [()] * nrep
  device_buffers = [xla.device_put(val, d) for d in devices]
  return ShardedDeviceArray(aval, indices, device_buffers)

def _pval_to_result_handler(axis_size, nrep, pval, devices, backend):
  if devices:
    assert all(d.host_id == xb.host_id(backend) for d in devices)
  aval, const = pval
  if aval is None:
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
                   else replicate(const, nrep, devices, backend))
    return lambda _: (bcast_const, not_mapped)
  else:
    return aval_to_result_handler(axis_size, nrep, aval)

def execute_replicated(compiled, backend, in_handler, out_handler, *args):
  input_bufs = in_handler(args)
  out_bufs = compiled.ExecutePerReplica(list(input_bufs))
  return unzip2(out_handler(out_bufs))


xla_pmap_p = core.Primitive('xla_pmap')
xla_pmap_p.call_primitive = True
xla_pmap_p.multiple_results = True
xla_pmap = partial(core.call_bind, xla_pmap_p)
xla_pmap_p.def_custom_bind(xla_pmap)
xla_pmap_p.def_impl(xla_pmap_impl)

def _pmap_translation_rule(c, axis_env, in_nodes, name_stack, call_jaxpr,
                           axis_name, axis_size, global_axis_size, in_dims,
                           out_dim_dests, devices, name, backend=None):
  # We in-line here rather than generating a Call HLO as in the xla_call
  # translation rule just because the extra tuple stuff is a pain.
  if axis_env.devices is not None or (axis_env.names and devices is not None):
    raise ValueError("Nested pmaps with explicit devices argument.")
  if global_axis_size is None:
    global_axis_size = axis_size
  new_env = xla.extend_axis_env(axis_env, axis_name, global_axis_size)
  # Shard the in_nodes that are mapped
  in_nodes_sharded = [_xla_shard(c, new_env, in_node, dim)
                      for in_node, dim in zip(in_nodes, in_dims)]

  sharded_outs = xla.jaxpr_subcomp(
      c, call_jaxpr, backend, new_env, (),
      extend_name_stack(name_stack, wrap_name(name, 'pmap')), *in_nodes_sharded)
  outs = [_xla_unshard(c, new_env, shard, dim)
          for shard, dim in zip(sharded_outs, out_dim_dests)]
  return c.Tuple(*outs)

xla.call_translations[xla_pmap_p] = _pmap_translation_rule
ad.primitive_transposes[xla_pmap_p] = partial(ad.map_transpose, xla_pmap_p)
pe.map_primitives.add(xla_pmap_p)

def _xla_shard(c, axis_env, x, dim):
  if dim is not_mapped:
    return x
  xla_shape = c.GetShape(x)
  if xla_shape.is_tuple():
    assert not xla_shape.tuple_shapes()
    return x
  else:
    dims = list(xla_shape.dimensions())
    zero = c.Constant(onp.zeros((), dtype=onp.uint32))
    idxs = [zero] * dim + [_unravel_index(c, axis_env)] + [zero] * (len(dims) - dim - 1)
    return c.Reshape(c.DynamicSlice(x, idxs, dims[:dim] + [1] + dims[dim + 1:]),
                     None, dims[:dim] + dims[dim + 1:])

# TODO(b/110096942): more efficient gather
def _xla_unshard(c, axis_env, x, dim):
  if dim is not_mapped:
    return x
  xla_shape = c.GetShape(x)
  if xla_shape.is_tuple():
    assert not xla_shape.tuple_shapes()
    return x
  else:
    dims = list(xla_shape.dimensions())
    padded = c.Broadcast(c.Constant(onp.array(0, xla_shape.numpy_dtype())),
                         dims[:dim] + [axis_env.sizes[-1]] + dims[dim:])
    zero = c.Constant(onp.zeros((), dtype=onp.uint32))
    idxs = [zero] * dim + [_unravel_index(c, axis_env)] + [zero] * (len(dims) - dim)
    padded = c.DynamicUpdateSlice(
        padded, c.Reshape(x, None, dims[:dim] + [1] + dims[dim:]), idxs)
    return c.CrossReplicaSum(padded, xla.axis_groups(axis_env, axis_env.names[-1]))

def _unravel_index(c, axis_env):
  div = c.Constant(onp.array(axis_env.nreps // prod(axis_env.sizes), onp.uint32))
  mod = c.Constant(onp.array(axis_env.sizes[-1], onp.uint32))
  return c.Rem(c.Div(c.ReplicaId(), div), mod)


### soft_pmap axis split transformation

# To allow pmap to map over logical axes larger than the number of XLA devices
# available, we use a transformation that effectively simulates having more
# devices in software. The strategy is to split the mapped axis into two axes,
# one to be hardware-mapped and the other to be software-mapped. Thus the
# transformation rewrites the function to be mapped so that it accepts a new
# leading axis (the software-mapped axis), and so that collectives in the
# original function correspond to both device-local operations and collective
# communication operations across hardware devices that implement the original
# logical semantics.

@lu.transformation
def split_axis(axis_name, chunk_size, *args):
  with core.new_master(SplitAxisTrace) as master:
    trace = SplitAxisTrace(master, core.cur_sublevel())
    in_tracers = list(map(partial(SplitAxisTracer, trace, axis_name), args))
    with add_chunk_to_axis_env(axis_name, trace, chunk_size):
      outs = yield in_tracers, {}
    out_tracers = list(map(trace.full_raise, outs))
    out_vals, out_names = unzip2((t.val, t.axis_name) for t in out_tracers)
    del master, out_tracers
  out_vals = [broadcast(x, chunk_size, 0) if d is not_mapped else x
              for x, d in zip(out_vals, out_names)]
  yield out_vals

@lu.transformation_with_aux
def split_axis_subtrace(master, names, *vals):
  trace = SplitAxisTrace(master, core.cur_sublevel())
  outs = yield list(map(partial(SplitAxisTracer, trace), names, vals)), {}
  out_tracers = list(map(trace.full_raise, outs))
  out_vals, out_names = unzip2((t.val, t.axis_name) for t in out_tracers)
  yield out_vals, out_names

@contextmanager
def add_chunk_to_axis_env(axis_name, soft_trace, soft_size):
  dynamic_axis_env = _thread_local_state.dynamic_axis_env
  dynamic_axis_env[axis_name].soft_trace = soft_trace
  dynamic_axis_env[axis_name].soft_size = soft_size
  yield
  dynamic_axis_env[axis_name].soft_trace = None
  dynamic_axis_env[axis_name].soft_size = None

class SplitAxisTracer(core.Tracer):
  def __init__(self, trace, axis_name, val):
    self._trace = trace
    self.axis_name = axis_name
    self.val = val

  @property
  def aval(self):
    aval = raise_to_shaped(core.get_aval(self.val))
    if self.axis_name is not_mapped:
      return aval
    else:
      assert isinstance(aval, ShapedArray)
      return ShapedArray(aval.shape[1:], aval.dtype)

  def full_lower(self):
    if self.axis_name is not_mapped:
      return core.full_lower(self.val)
    else:
      return self

class SplitAxisTrace(core.Trace):
  def pure(self, val):
    return SplitAxisTracer(self, not_mapped, val)

  def lift(self, val):
    return SplitAxisTracer(self, not_mapped, val)

  def sublift(self, val):
    return SplitAxisTracer(self, val.axis_name, val.val)

  def process_primitive(self, primitive, tracers, params):
    vals_in, names_in = unzip2((t.val, t.axis_name) for t in tracers)
    if primitive is axis_index_p:
      dummy, = vals_in
      hard_idx = primitive.bind(dummy, **params)
      val_out = hard_idx * params['soft_size'] + onp.arange(params['soft_size'])
      return SplitAxisTracer(self, params['axis_name'], val_out)
    elif all(axis_name is not_mapped for axis_name in names_in):
      return primitive.bind(*vals_in, **params)
    else:
      name, = set(n for n in names_in if n is not not_mapped)
      if primitive in xla.parallel_translations:
        # if it's a pmap collective primitive, do something special
        if name == params['axis_name']:
          # if the name matches this tracer's name, apply the split_axis rule
          try:
            rule = split_axis_rules[primitive]
          except KeyError:
            msg = "split_axis for {} not implemented. Open a feature request!"
            raise NotImplementedError(msg.format(primitive))
          which_mapped = [n is not not_mapped for n in names_in]
          val_out, is_mapped = rule(vals_in, which_mapped, **params)
          name_out = name if is_mapped else not_mapped
          return SplitAxisTracer(self, name_out, val_out)
        else:
          # if not, bind the primitive without any processing
          val_out = primitive.bind(*vals_in, **params)
          return SplitAxisTracer(self, name, val_out)
      else:
        # if it's not a pmap collective primitive, act just like batching
        rule = batching.get_primitive_batcher(primitive)
        axes_in = [n if n is not_mapped else 0 for n in names_in]
        val_out, axis_out = rule(vals_in, axes_in, **params)
        def new_tracer(x, a):
          if a is not_mapped:
            return SplitAxisTracer(self, not_mapped, x)
          else:
            return SplitAxisTracer(self, name, batching.moveaxis(x, a, 0))
        if primitive.multiple_results:
          return [new_tracer(x, a) for x, a in zip(val_out, axis_out)]
        else:
          return new_tracer(val_out, axis_out)

  def process_call(self, call_primitive, f, tracers, params):
    assert call_primitive.multiple_results
    if call_primitive in pe.map_primitives:
      return self.process_map(call_primitive, f, tracers, params)
    else:
      vals, names = unzip2((t.val, t.axis_name) for t in tracers)
      if all(name is not_mapped for name in names):
        return call_primitive.bind(f, *vals, **params)
      else:
        f, names_out = split_axis_subtrace(f, self.master, names)
        vals_out = call_primitive.bind(f, *vals, **params)
        return [SplitAxisTracer(self, a, x) for a, x in zip(names_out(), vals_out)]

  def process_map(self, map_primitive, f, tracers, params):
    vals, names = unzip2((t.val, t.axis_name) for t in tracers)
    if all(name is not_mapped for name in names):
      return map_primitive.bind(f, *vals, **params)
    else:
      # because the map primitive maps over leading axes, we need to transpose
      # the software-mapped axis on any mapped arguments to be the second axis;
      # then we call the map primitive and resume the trace under the call
      vals_trans = [batching.moveaxis(x, 0, 1) if d is not not_mapped else x
                    for x, d in zip(vals, names)]
      f, names_out = split_axis_subtrace(f, self.master, names)
      vals_out_trans = map_primitive.bind(f, *vals_trans, **params)
      vals_out = [batching.moveaxis(x, 1, 0) if d is not not_mapped else x
                  for x, d in zip(vals_out_trans, names_out())]
      return [SplitAxisTracer(self, a, x) for a, x in zip(names_out(), vals_out)]

  def post_process_call(self, call_primitive, out_tracer, params):
    val, name = out_tracer.val, out_tracer.axis_name
    master = self.master
    def todo(x):
      trace = SplitAxisTrace(master, core.cur_sublevel())
      return  SplitAxisTracer(trace, name, x)
    return  val, todo


split_axis_rules = {}
