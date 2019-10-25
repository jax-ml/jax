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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from contextlib import contextmanager
import itertools as it
import operator as op
import threading

from absl import logging
import numpy as onp
import six
from six.moves import reduce

from .. import core
from .. import linear_util as lu
from ..abstract_arrays import (ConcreteArray, ShapedArray, array_types,
                               raise_to_shaped)
from ..util import partial, unzip2, concatenate, prod
from ..lib import xla_bridge as xb
from .xla import aval_to_xla_shape, xla_destructure
from .partial_eval import trace_to_subjaxpr, merge_pvals, JaxprTrace, PartialVal
from .batching import broadcast, not_mapped
from . import batching
from . import partial_eval as pe
from . import xla
from . import ad


### util

def identity(x): return x

def shard_args(backend, devices, assignments, axis_size, tuple_args, args):
  """Shard an argument data array arg along its leading axis.

  Args:
    devices: list of Devices of length num_replicas mapping a logical replica
      index to a physical device.
    assignments: replica to shard assignment.
    axis_size: int, size of the axis to be sharded.
    args: a sequence of JaxTypes representing arguments to be sharded along
      their leading axes (or the leading axess of their leaves in the tuple
      case) and placed on `devices`.

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
    if type(arg) is ShardedDeviceArray:
      if nrep == len(arg.device_buffers):
        for r, buf in enumerate(arg.device_buffers):
          buffers[r][a] = (buf if buf.device() == devices[r]
                           else buf.copy_to_device(devices[r]))
      else:
        for r, buf in enumerate(arg.device_buffers):
          buffers[r][a] = xla.device_put(x[assignments[r]], devices[r], backend=backend)
    else:
      bufs = shard_arg_handlers[type(arg)](arg, devices, assignments, backend=backend)
      for r, buf in enumerate(bufs):
        buffers[r][a] = buf

  if tuple_args:
    buffers = [[xla.make_tuple(bufs, devices[r], backend)]
               for r, bufs in enumerate(buffers)]

  return buffers

shard_arg_handlers = {}
shard_arg_handlers[core.Unit] = \
    lambda x, devices, _, backend=None: [
        xla.device_put(core.unit, d, backend=backend) for d in devices]
def _shard_array(x, devices, assignments, backend=None):
  nrep = len(devices)
  return (xla.device_put(x[assignments[r]], devices[r], backend=backend) for r in range(nrep))
for _t in array_types:
  shard_arg_handlers[_t] = _shard_array

def _shard_device_array(x, devices, assignments, backend=None):
  nrep = len(devices)
  xs = x._unstack()
  return (xla.device_put(xs[assignments[r]], devices[r], backend=backend)
          for r in range(nrep))
shard_arg_handlers[xla.DeviceArray] = _shard_device_array

def shard_aval(size, aval):
  try:
    return shard_aval_handlers[type(aval)](size, aval)
  except KeyError:
    raise TypeError("No shard_aval handler for type: {}".format(type(aval)))
shard_aval_handlers = {}
shard_aval_handlers[core.AbstractUnit] = lambda size, x: x
def _shard_abstract_array(size, x):
  if x.shape[0] != size:
    raise ValueError("Axis size {} does not match leading dimension of "
                     "shape {}".format(size, x.shape))
  return ShapedArray(x.shape[1:], x.dtype)
shard_aval_handlers[ShapedArray] = _shard_abstract_array

def aval_to_result_handler(size, nrep, aval):
  try:
    return pxla_result_handlers[type(aval)](size, nrep, aval)
  except KeyError:
    raise TypeError("No pxla_result_handler for type: {}".format(type(aval)))
pxla_result_handlers = {}
pxla_result_handlers[core.AbstractUnit] = lambda *_: lambda _: core.unit
def array_result_handler(size, nrep, aval):
  full_aval = ShapedArray((size,) + aval.shape, aval.dtype)
  return partial(ShardedDeviceArray, full_aval)
pxla_result_handlers[ShapedArray] = array_result_handler
pxla_result_handlers[ConcreteArray] = array_result_handler


def assign_shards_to_replicas(nrep, size):
  """Produce a mapping from replica id to shard index.

  Args:
    nrep: int, number of replicas (a computation-dependent value).
    size: int, size of the data array axis being sharded.

  Returns:
    A tuple of integers of length nrep in which the elements take on values from
    0 to size-1. Replica n is assgined shard data_array[assignments[n]].
  """
  groupsize, ragged = divmod(nrep, size)
  assert not ragged
  indices = onp.tile(onp.arange(size)[:, None], (1, groupsize))
  return tuple(indices.ravel())

def replica_groups(nrep, mesh_spec, mesh_axes):
  """Compute XLA replica groups from a replica count and device mesh data.

  Args:
    nrep: int, number of replicas (a computation-dependent value).
    mesh_spec: tuple of integers, a specification of the logical device mesh,
      which depends on the lexical context of nested xla_pmaps. In particular,
      each xla_pmap effectively appends its mapped axis size to this tuple.
    mesh_axes: tuple of ints, logical device mesh axis indices indicating the
      axes along which collective operations are to be executed.

  Returns:
    replica_groups, a list of lists of ints encoding a partition of the set
      {0, 1, ..., nrep} into equally-sized replica groups (within which
      collectives are executed). XLA consumes this replica group specification.
  """
  trailing_size, ragged = divmod(nrep, prod(mesh_spec))
  assert not ragged
  full_spec = mesh_spec + [trailing_size]
  iota = onp.arange(prod(full_spec)).reshape(full_spec)
  groups = onp.reshape(
      onp.moveaxis(iota, mesh_axes, onp.arange(len(mesh_axes))),
      (prod(onp.take(full_spec, mesh_axes)), -1))
  return tuple(map(tuple, groups.T))


### the main pmap machinery lowers SPMD jaxprs to multi-replica XLA computations

def compile_replicated(jaxpr, backend, axis_name, axis_size, global_axis_size,
                       devices, consts, tuple_args, *abstract_args):
  jaxpr_replicas = xla.jaxpr_replicas(jaxpr)
  num_local_replicas = axis_size * jaxpr_replicas
  num_replicas = global_axis_size * jaxpr_replicas
  logging.vlog(
      1, "compile_replicated: axis_size=%d global_axis_size=%d jaxpr_replicas=%d"
      % (axis_size, global_axis_size, jaxpr_replicas))

  if devices is None:
    if num_replicas > xb.device_count(backend):
      msg = ("compiling computation that requires {} replicas, but only {} XLA "
             "devices are available")
      raise ValueError(msg.format(num_replicas, xb.device_count(backend)))
    device_assignment = None
  else:
    assert any(d.host_id == xb.host_id() for d in devices)
    local_devices = [d for d in devices if d.host_id == xb.host_id()]
    assert len(local_devices) > 0
    if num_local_replicas != len(local_devices):
      local_devices_str = ", ".join(map(str, local_devices))
      raise ValueError(
          "Leading axis size of input to pmapped function must equal the "
          "number of local devices passed to pmap. Got axis_size=%d, "
          "num_local_devices=%d.\n(Local devices passed to pmap: %s)"
          % (axis_size, len(local_devices), local_devices_str))
    if num_replicas != len(devices):
      raise ValueError("compiling computation that requires %s replicas, "
                       "but %s devices were specified"
                       % (num_replicas, len(devices)))
    device_assignment = tuple(d.id for d in devices)

  axis_env = xla.AxisEnv(num_replicas, [axis_name], [global_axis_size], devices)
  arg_shapes = list(map(aval_to_xla_shape, abstract_args))
  built_c = xla.jaxpr_computation(jaxpr, backend, axis_env, consts, (), arg_shapes,
                                  tuple_args=tuple_args, inner=False)
  compiled = built_c.Compile(
      compile_options=xb.get_compile_options(num_replicas, device_assignment),
      backend=xb.get_backend(backend))
  return compiled, num_local_replicas


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
  dummy_arg = frame.pmap_trace.pure(core.unit)
  if frame.soft_trace:
    dummy_arg = frame.soft_trace.pure(dummy_arg)
  return axis_index_p.bind(dummy_arg, hard_size=frame.hard_size,
                           soft_size=frame.soft_size, axis_name=axis_name)

def _axis_index_partial_eval(trace, _, **params):
  # This partial_eval rule adds the axis_index primitive into the jaxpr formed
  # during pmap lowering. It is like the standard JaxprTrace.process_primitive
  # rule except that we don't attempt to lower out of the trace.
  out_aval = ShapedArray((), onp.int32)
  out_tracer = pe.JaxprTracer(trace, pe.PartialVal((out_aval, core.unit)), None)
  eqn = core.new_jaxpr_eqn([], [out_tracer], axis_index_p, (), params)
  out_tracer.recipe = eqn
  return out_tracer

def _axis_index_translation_rule(c, hard_size, soft_size, axis_name):
  unsigned_index = c.Rem(c.ReplicaId(),
                         c.Constant(onp.array(hard_size, onp.uint32)))
  return c.ConvertElementType(unsigned_index, xb.dtype_to_etype(onp.int32))

axis_index_p = core.Primitive('axis_index')
xla.translations[axis_index_p] = _axis_index_translation_rule
pe.custom_partial_eval_rules[axis_index_p] = _axis_index_partial_eval


### lazy device-memory persistence and result handling

class ShardedDeviceValue(xla.DeviceValue):
  def _check_if_deleted(self):
    if self.device_buffers is None:
      raise ValueError("ShardedDeviceValue has been deleted.")

  def block_until_ready(self):
    self._check_if_deleted()
    for buf in self.device_buffers:
      buf.block_host_until_ready()
    return self


class ShardedDeviceArray(ShardedDeviceValue, xla.DeviceArray):
  """A ShardedDeviceArray is an ndarray sharded across devices.

  The purpose of a ShardedDeviceArray is to reduce the number of transfers when
  executing replicated computations, by allowing results to persist on the
  devices that produced them. That way dispatching a similarly replicated
  computation that consumes the same sharded memory layout does not incur any
  transfers.

  A ShardedDeviceArray represents one logical ndarray value, and simulates the
  behavior of an ndarray so that it can be treated by user code as an ndarray;
  that is, it is only an optimization to reduce transfers.

  The number of device buffers underlying a ShardedDeviceArray instance is equal
  to the number of replicas of the computation that produced it. Each buffer
  represents a shard of the original array, meaning a slice along its leading
  axis. These component buffers reside on distinct devices, but need not
  represent distinct logical shards. The correspondence can be computed with
  the assign_shards_to_replicas function.
  """
  __slots__ = ["device_buffers", "axis_size"]
  _collect = staticmethod(onp.stack)

  def __init__(self, aval, device_buffers):
    self.aval = aval
    self.device_buffers = device_buffers
    self.axis_size = aval.shape[0]
    self._npy_value = None
    if not core.skip_checks:
      assert type(aval) is ShapedArray

  def _ids(self):
    num_bufs = len(self.device_buffers)
    assignments = assign_shards_to_replicas(num_bufs, self.axis_size)
    _, ids = onp.unique(assignments, return_index=True)
    return ids

  def copy_to_host_async(self):
    if self._npy_value is None:
      for buf in self.device_buffers:
        buf.copy_to_host_async()

  def delete(self):
    for buf in self.device_buffers:
      buf.delete()
    self.device_buffers = None
    self._npy_value = None

  @property
  def _value(self):
    if self._npy_value is None:
      ids = self._ids()
      self.copy_to_host_async()
      self._npy_value = self._collect([self.device_buffers[i].to_py() for i in ids])
    return self._npy_value

  def __getitem__(self, idx):
    if self._npy_value is None and type(idx) is int:
      ids = self._ids()
      device_buffer = self.device_buffers[ids[idx]]
      aval = ShapedArray(self.aval.shape[1:], self.aval.dtype)
      handler = xla.aval_to_result_handler(aval)
      return handler(device_buffer)
    else:
      return super(ShardedDeviceArray, self).__getitem__(idx)

# This handler code is effectively dead because we in-lined it in shard_args for
# performance reasons.
def _shard_sharded_device_array(x, devices, assignments):
  n = len(devices)
  if n == len(x.device_buffers):
    return (b if b.device() == devices[r] else b.copy_to_device(devices[r])
            for r, b in enumerate(x.device_buffers))
  else:
    return (xla.device_put(x[assignments[r]], devices[r]) for r in range(n))
shard_arg_handlers[ShardedDeviceArray] = _shard_sharded_device_array

core.pytype_aval_mappings[ShardedDeviceArray] = ConcreteArray
xla.device_put_handlers[ShardedDeviceArray] = xla._device_put_array
xla.pytype_aval_mappings[ShardedDeviceArray] = lambda x: x.aval
xla.canonicalize_dtype_handlers[ShardedDeviceArray] = identity
xb.register_constant_handler(ShardedDeviceArray, xla._device_array_constant_handler)


class ChunkedDeviceArray(ShardedDeviceArray):
  __slots__ = []
  _collect = staticmethod(onp.concatenate)

  def __init__(self, axis_size, aval, device_buffers):
    super(ChunkedDeviceArray, self).__init__(aval, device_buffers)
    self.axis_size = axis_size

  def __getitem__(self, idx):
    return xla.DeviceArray.__getitem__(self, idx)

shard_arg_handlers[ChunkedDeviceArray] = _shard_array

core.pytype_aval_mappings[ChunkedDeviceArray] = ConcreteArray
xla.device_put_handlers[ChunkedDeviceArray] = xla._device_put_array
xla.pytype_aval_mappings[ChunkedDeviceArray] = lambda x: x.aval
xla.canonicalize_dtype_handlers[ChunkedDeviceArray] = identity
xb.register_constant_handler(ChunkedDeviceArray,
                             xla._device_array_constant_handler)


### the xla_pmap primitive and its rules are comparable to xla_call in xla.py

def xla_pmap_impl(fun, *args, **params):
  axis_name = params.pop('axis_name')
  axis_size = params.pop('axis_size')
  devices = params.pop('devices')
  backend = params.pop('backend', None)
  assert not params

  abstract_args = map(xla.abstractify, args)
  compiled_fun = parallel_callable(fun, backend, axis_name, axis_size, devices,
                                   *abstract_args)
  return compiled_fun(*args)

@lu.cache
def parallel_callable(fun, backend, axis_name, axis_size, devices, *avals):
  if devices is not None and len(devices) == 0:
    raise ValueError("'devices' argument to pmap must be non-empty, or None.")

  avals = tuple(map(partial(shard_aval, axis_size), avals))
  pvals = [PartialVal((aval, core.unit)) for aval in avals]
  pval = PartialVal([core.abstract_unit, core.unit])  # dummy value

  if devices:
    global_axis_size = len(devices)
  elif xb.host_count() > 1:
    # TODO(skye): relax this constraint or provide functionality for
    # automatically passing appropriate `devices`.
    if axis_size != xb.local_device_count():
      raise ValueError(
          "On multi-host platforms, the input to pmapped functions must have "
          "leading axis size equal to the number of local devices if no "
          "`devices` argument is specified. Got axis_size=%d, "
          "num_local_devices=%d" % (axis_size, xb.local_device_count()))
    global_axis_size = xb.device_count()
  else:
    global_axis_size = axis_size

  @lu.wrap_init
  def dynamic_fun(dummy, *args):
    with extend_dynamic_axis_env(axis_name, dummy.trace, global_axis_size):
      return fun.call_wrapped(*args)

  with core.new_master(JaxprTrace, True) as master:
    jaxpr, (out_pvals, consts, env) = \
        trace_to_subjaxpr(dynamic_fun, master, False).call_wrapped([pval] + pvals)
    jaxpr.invars = jaxpr.invars[1:]  # ignore dummy
    assert not env
    del master
  out_pvs, out_consts = unzip2(out_pvals)
  if all(pv is None for pv in out_pvs):
    # When the output doesn't depend on the input we don't need to compile an
    # XLA computation at all; we handle this as a special case so we can stage
    # out multi-replica XLA computations regardless of the hardware available.
    # The 'None' values here are just dummies we know will be ignored.
    handlers = [_pval_to_result_handler(axis_size, None, pval) for pval in out_pvals]
    results = [handler(None) for handler in handlers]
    return lambda *_: results
  else:
    # Condense many arguments into single tuple argument to avoid a TPU issue.
    tuple_args = len(avals) > 100
    compiled, nrep = compile_replicated(jaxpr, backend, axis_name, axis_size,
                                        global_axis_size, devices, consts,
                                        tuple_args, *avals)
    local_devices = compiled.local_devices()
    assignments = assign_shards_to_replicas(nrep, axis_size)
    handle_args = partial(shard_args, backend, local_devices, assignments,
                          axis_size, tuple_args)
    handle_outs = _pvals_to_results_handler(axis_size, nrep, out_pvals)
    return partial(execute_replicated, compiled, backend, nrep, handle_args, handle_outs)

def _pvals_to_results_handler(size, nrep, out_pvals):
  nouts = len(out_pvals)
  handlers = [_pval_to_result_handler(size, nrep, pval) for pval in out_pvals]
  def handler(out_bufs):
    buffers = [[None] * nrep for _ in range(nouts)]
    for r, tuple_buf in enumerate(out_bufs):
      for i, buf in enumerate(tuple_buf.destructure()):
        buffers[i][r] = buf
    return [h(bufs) for h, bufs in zip(handlers, buffers)]
  return handler

def _pval_to_result_handler(size, nrep, pval):
  pv, const = pval
  if pv is None:
    # TODO make a ShardedDeviceArray here?
    bcast_const = core.unit if const is core.unit else broadcast(const, size, 0)
    return lambda _: bcast_const
  else:
    return aval_to_result_handler(size, nrep, pv)

def execute_replicated(compiled, backend, nrep, in_handler, out_handler, *args):
  if nrep > xb.device_count(backend):
    msg = ("executing pmap computation that requires {} replicas, but only {} "
           "XLA devices are available")
    raise ValueError(msg.format(nrep, xb.device_count(backend)))
  input_bufs = in_handler(args)
  out_bufs = compiled.ExecutePerReplica(list(input_bufs))
  return out_handler(out_bufs)


xla_pmap_p = core.Primitive('xla_pmap')
xla_pmap_p.multiple_results = True
xla_pmap = partial(core.call_bind, xla_pmap_p)
xla_pmap_p.def_custom_bind(xla_pmap)
xla_pmap_p.def_impl(xla_pmap_impl)

def _pmap_translation_rule(c, jaxpr, axis_env, const_nodes, freevar_nodes,
                           in_nodes, axis_name, axis_size, devices, backend=None):
  # We in-line here rather than generating a Call HLO as in the xla_call
  # translation rule just because the extra tuple stuff is a pain.
  if axis_env.devices is not None or (axis_env.names and devices is not None):
    raise ValueError("Nested pmaps with explicit devices argument.")
  new_env = xla.extend_axis_env(axis_env, axis_name, axis_size)
  in_nodes_sharded = list(map(partial(_xla_shard, c), in_nodes))
  sharded_outs = xla.jaxpr_subcomp(c, jaxpr, backend, new_env, const_nodes,
                                   freevar_nodes, *in_nodes_sharded)
  outs = [_xla_unshard(c, xla.axis_groups(new_env, axis_name), r)
          for r in sharded_outs]
  return c.Tuple(*outs)

xla.call_translations[xla_pmap_p] = _pmap_translation_rule
ad.primitive_transposes[xla_pmap_p] = partial(ad.map_transpose, xla_pmap_p)
pe.map_primitives.add(xla_pmap_p)

def _xla_shard(c, x):
  xla_shape = c.GetShape(x)
  if xla_shape.is_tuple():
    assert not xla_shape.tuple_shapes()
    return x
  else:
    dims = list(xla_shape.dimensions())
    start_indices = _xla_shard_start_indices(c, dims[0], len(dims))
    return c.Reshape(c.DynamicSlice(x, start_indices, [1] + dims[1:]),
                     None, dims[1:])

# TODO(b/110096942): more efficient gather
def _xla_unshard(c, replica_groups, x):
  xla_shape = c.GetShape(x)
  if xla_shape.is_tuple():
    assert not xla_shape.tuple_shapes()
    return x
  else:
    axis_size = len(replica_groups[0])
    dims = list(xla_shape.dimensions())
    start_indices = _xla_shard_start_indices(c, axis_size, len(dims) + 1)
    padded = c.Broadcast(c.Constant(onp.array(0, xla_shape.numpy_dtype())),
                        [axis_size] + dims)
    padded = c.DynamicUpdateSlice(padded, c.Reshape(x, None, [1] + dims),
                                  start_indices)
    return c.CrossReplicaSum(padded, replica_groups)

# TODO(mattjj): use more ergonimic form of DynamicUpdateSlice instead!
def _xla_shard_start_indices(c, axis_size, ndim):
  idx = c.Rem(c.ReplicaId(), c.Constant(onp.array(axis_size, onp.uint32)))
  zero = onp.zeros(ndim - 1, onp.uint32)
  return c.Concatenate([c.Reshape(idx, None, [1]), c.Constant(zero)], 0)


def _device_put_rep_impl(x, devices, backend=None):
  sharded_aval = xla.abstractify(x)
  if sharded_aval is core.abstract_unit:
    aval = core.abstract_unit
  else:
    aval = ShapedArray((len(devices),) + sharded_aval.shape, sharded_aval.dtype)
  bufs = [xla.device_put(x, device, backend=backend) for device in devices]
  return ShardedDeviceArray(aval, bufs)

def _device_put_rep_abstract_eval(x, devices):
  if x is core.abstract_unit:
    return x
  if type(x) is ShapedArray:
    return ShapedArray((len(devices),) + x.shape, x.dtype)
  else:
    raise NotImplementedError

device_put_rep_p = core.Primitive('device_put_replicated')
device_put_rep_p.def_impl(_device_put_rep_impl)
device_put_rep_p.def_abstract_eval(_device_put_rep_abstract_eval)
ad.deflinear(device_put_rep_p, lambda t, **_: [t.sum(axis=0)])
xla.translations[device_put_rep_p] = \
    lambda c, x, devices: c.Broadcast(x, len(devices))
# TODO batching rule


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
    self.trace = trace
    self.axis_name = axis_name
    self.val = val

  @property
  def aval(self):
    aval = raise_to_shaped(core.get_aval(self.val))
    if self.axis_name is not_mapped:
      return aval
    else:
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
