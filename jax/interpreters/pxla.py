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
import itertools as it
import operator as op

import numpy as onp
import six
from six.moves import reduce

from .. import core
from .. import ad_util
from .. import tree_util
from .. import linear_util as lu
from ..abstract_arrays import ConcreteArray, ShapedArray
from ..util import partial, unzip2, concatenate, prod, memoize_unary
from ..lib import xla_bridge as xb
from .xla import (xla_shape, xla_destructure, translation_rule,
                  xla_shape_to_result_shape, jaxpr_computation)
from .partial_eval import trace_to_subjaxpr, merge_pvals, JaxprTrace, PartialVal
from .batching import dimsize, broadcast
from . import partial_eval as pe
from . import parallel
from . import xla
from . import ad


### util


# TODO(mattjj, phawkins): improve re-distribution not to copy to host
def shard_arg(device_ordinals, axis_size, arg):
  """Shard an argument data array arg along its leading axis.

  Args:
    device_ordinals: list of integers of length num_replicas mapping a logical
      replica index to a physical device number.
    axis_size: int, size of the axis to be sharded.
    arg: a JaxType representing an argument to be sharded along its leading axis
      (or the leading axis of its leaves in the tuple case) and placed on the
      devices indicated by `device_ordinals`.

  Returns:
    A list of device buffers with the same length as `device_ordinals` indexed
    by replica number, so that the nth element is the argument to be passed to
    the nth replica.
  """
  nrep = len(device_ordinals)
  assignments = assign_shards_to_replicas(nrep, axis_size)
  if (type(arg) in (ShardedDeviceArray, ShardedDeviceTuple)
      and nrep == len(arg.device_buffers)):
    _, ids = onp.unique(assignments, return_index=True)
    get_shard = memoize_unary(lambda i: arg.device_buffers[i].to_py())
    return [buf if buf.device() == device_ordinals[r]
            else xla.device_put(get_shard(i), device_ordinals[r])
            for r, (i, buf) in enumerate(zip(assignments, arg.device_buffers))]
  else:
    shards = [(_slice(arg, assignments[i]), device_ordinals[i])
              for i in range(len(assignments))]
    return xla.device_put_many(shards)

def _slice(x, i):
  """Return the ith slice of a JaxType (tuple or array)."""
  if isinstance(x, core.JaxTuple):
    return core.pack(_slice(elt, i) for elt in x)
  else:
    return x[i]


def sharded_result_handler(axis_size, aval):
  full_aval = add_axis_to_aval(axis_size, aval)
  t = type(aval)
  if t is ShapedArray:
    return partial(sharded_array_result_handler, full_aval)
  elif t is core.AbstractTuple:
    return partial(sharded_tuple_result_handler, axis_size, full_aval)
  else:
    raise TypeError(t)

def sharded_array_result_handler(aval, replica_results):
  t, = set(map(type, replica_results))
  if t is xla.DeviceArray:
    bufs = [r.device_buffer for r in replica_results]
    return ShardedDeviceArray(aval, bufs)
  elif issubclass(t, (onp.ndarray, xla.DeviceConstant, ShardedDeviceArray)):
    assignments = assign_shards_to_replicas(len(replica_results), aval.shape[0])
    _, ids = onp.unique(assignments, return_index=True)
    return onp.stack([replica_results[i] for i in ids])
  else:
    raise TypeError(t)

def sharded_tuple_result_handler(axis_size, aval, replica_results):
  t, = set(map(type, replica_results))
  if t is xla.DeviceTuple:
    bufs = [r.device_buffer for r in replica_results]
    return ShardedDeviceTuple(axis_size, aval, bufs)
  elif t is core.JaxTuple:
    # e.g. pmap(lambda x: core.pack((3, x)))(...)
    reduced_aval = remove_axis_from_aval(aval)
    all_results = zip(*replica_results)
    return core.pack([sharded_result_handler(axis_size, elt_aval)(results)
                      for elt_aval, results in zip(reduced_aval, all_results)])
  else:
    raise TypeError(t)


def add_axis_to_aval(n, aval):
  t = type(aval)
  if t is core.AbstractTuple:
    return core.AbstractTuple(map(partial(add_axis_to_aval, n), aval))
  elif t is ShapedArray:
    return ShapedArray((n,) + aval.shape, aval.dtype)
  else:
    raise TypeError(t)

def remove_axis_from_aval(aval):
  t = type(aval)
  if t is core.AbstractTuple:
    return core.AbstractTuple(map(remove_axis_from_aval, aval))
  elif t is ShapedArray:
    return ShapedArray(aval.shape[1:], aval.dtype)
  else:
    raise TypeError(t)


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

def xla_shard(c, sizes, x):
  """Analog of shard_arg that performs sharding within an XLA computation."""
  def _xla_shard(shape, x):
    if shape.is_tuple():
      elts = map(_xla_shard, shape.tuple_shapes(), xla_destructure(c, x))
      return c.Tuple(*elts)
    else:
      return shard_array(shape, x)

  def shard_array(shape, x):
    dims = list(shape.dimensions())
    assert dims[0] == sizes[-1]
    start_indices = _xla_shard_start_indices(c, dims[0], len(dims))
    return c.Reshape(c.DynamicSlice(x, start_indices, [1] + dims[1:]),
                     None, dims[1:])

  return _xla_shard(c.GetShape(x), x)

def _xla_shard_start_indices(c, axis_size, ndim):
  idx = c.Rem(c.ReplicaId(), c.Constant(onp.array(axis_size, onp.uint32)))
  zero = onp.zeros(ndim - 1, onp.uint32)
  return c.Concatenate([c.Reshape(idx, None, [1]), c.Constant(zero)], 0)

# TODO(b/110096942): more efficient gather
def xla_unshard(c, device_groups, x):
  """Analog of unshard_output that un-shards within an XLA computation."""
  def _xla_unshard(shape, x):
    if shape.is_tuple():
      elts = map(_xla_unshard, shape.tuple_shapes(), xla_destructure(c, x))
      return c.Tuple(*elts)
    else:
      return unshard_array(shape, x)

  def unshard_array(shape, x):
    group_size = len(device_groups[0])
    broadcasted = c.Broadcast(x, (group_size,))
    return c.AllToAll(broadcasted, 0, 0, device_groups)

  return _xla_unshard(c.GetShape(x), x)


### xla_pmap


AxisEnv = namedtuple("AxisEnv", ["nreps", "names", "sizes"])

def axis_read(axis_env, axis_name):
  return max(i for i, name in enumerate(axis_env.names) if name == axis_name)

def axis_groups(axis_env, name):
  if isinstance(name, (list, tuple)):
    mesh_axes = tuple(map(partial(axis_read, axis_env), name))
  else:
    mesh_axes = (axis_read(axis_env, name),)
  return replica_groups(axis_env.nreps, axis_env.sizes, mesh_axes)

def compile_replicated(jaxpr, axis_name, axis_size, consts, *abstract_args):
  num_replicas = axis_size * jaxpr_replicas(jaxpr)
  axis_env = AxisEnv(num_replicas, [axis_name], [axis_size])
  arg_shapes = list(map(xla_shape, abstract_args))
  built_c = replicated_comp(jaxpr, axis_env, consts, (), *arg_shapes)
  result_shape = xla_shape_to_result_shape(built_c.GetReturnValueShape())
  compiled = built_c.Compile(arg_shapes, xb.get_compile_options(num_replicas),
                             backend=xb.get_backend())
  return compiled, num_replicas, result_shape

def jaxpr_replicas(jaxpr):
  return _max(eqn.params['axis_size'] * jaxpr_replicas(eqn.bound_subjaxprs[0][0])
              for eqn in jaxpr.eqns if eqn.primitive is xla_pmap_p)
def _max(itr): return max(list(itr) or [1])

def replicated_comp(jaxpr, ax_env, const_vals, freevar_shapes, *arg_shapes):
  assert not any(type(invar) in (tuple, list) for invar in jaxpr.invars)
  c = xb.make_computation_builder("replicated_computation")

  def read(v):
    if type(v) is core.Literal:
      return c.Constant(v.val)
    else:
      return env[v]

  def write(v, node):
    assert node is not None
    env[v] = node

  def axis_env_extend(name, size):
    return AxisEnv(ax_env.nreps, ax_env.names + [name], ax_env.sizes + [size])

  env = {}
  write(core.unitvar, c.Tuple())
  if const_vals:
    _map(write, jaxpr.constvars, map(c.Constant, const_vals))
    _map(write, jaxpr.freevars, map(c.ParameterWithShape, freevar_shapes))
  else:
    all_freevars = it.chain(jaxpr.constvars, jaxpr.freevars)
    _map(write, all_freevars, map(c.ParameterWithShape, freevar_shapes))
  _map(write, jaxpr.invars, map(c.ParameterWithShape, arg_shapes))
  for eqn in jaxpr.eqns:
    if not eqn.restructure:
      in_nodes = list(map(read, eqn.invars))
    else:
      in_nodes = [xla.xla_pack(c, map(read, invars)) if type(invars) is tuple
                  else read(invars) for invars in eqn.invars]
    if eqn.primitive in parallel_translation_rules:
      name = eqn.params['axis_name']
      params = {k: eqn.params[k] for k in eqn.params if k != 'axis_name'}
      rule = parallel_translation_rules[eqn.primitive]
      ans = rule(c, *in_nodes, device_groups=axis_groups(ax_env, name), **params)
    elif eqn.bound_subjaxprs:
      if eqn.primitive is xla_pmap_p:
        name, size = eqn.params['axis_name'], eqn.params['axis_size']
        new_env = axis_env_extend(name, size)
        in_shards = tuple(map(partial(xla_shard, c, new_env.sizes), in_nodes))
        (subjaxpr, const_bindings, freevar_bindings), = eqn.bound_subjaxprs
        subc = replicated_comp(
            subjaxpr, new_env, (),
            tuple(map(c.GetShape, map(read, const_bindings + freevar_bindings))),
            *map(c.GetShape, in_shards))
        subfun = (subc, tuple(map(read, const_bindings + freevar_bindings)))
        sharded_result = xla.xla_call_translation_rule(c, subfun, *in_shards)
        ans = xla_unshard(c, axis_groups(new_env, name), sharded_result)
      else:
        subcs = [
            jaxpr_computation(
                subjaxpr, (),
                tuple(map(c.GetShape, map(read, const_bindings + freevar_bindings))),
                *map(c.GetShape, in_nodes))
            for subjaxpr, const_bindings, freevar_bindings in eqn.bound_subjaxprs]
        subfuns = [(subc, tuple(map(read, const_bindings + freevar_bindings)))
                    for subc, (_, const_bindings, freevar_bindings)
                    in zip(subcs, eqn.bound_subjaxprs)]
        ans = translation_rule(eqn.primitive)(c, *(subfuns + in_nodes), **eqn.params)
    else:
      ans = translation_rule(eqn.primitive)(c, *in_nodes, **eqn.params)

    out_nodes = xla_destructure(c, ans) if eqn.destructure else [ans]
    _map(write, eqn.outvars, out_nodes)
  return c.Build(read(jaxpr.outvar))

def _map(f, *xs):
  return tuple(map(f, *xs))


class ShardedDeviceTuple(xla.DeviceTuple):
  """A ShardedDeviceTuple is a JaxTuple sharded across devices.

  The purpose of a ShardedDeviceTuple is to reduce the number of transfers when
  executing replicated computations, by allowing results to persist on the
  devices that produced them. That way dispatching a similarly replicated
  computation that consumes the same sharded memory layout does not incur any
  transfers.

  A ShardedDeviceTuple represents one logical JaxTuple value, and simulates the
  behavior of a JaxTuple so that it can be treated by user code as a JaxTuple;
  that is, it is only an optimization to reduce transfers.

  The number of device buffers underlying a ShardedDeviceTuple instance is equal
  to the number of replicas of the computation that produced it. Each buffer
  represents a shard of the logical tuple value represented by the
  ShardedDeviceTuple, where a shard of an array is a slice along its leading
  axis, and a shard of a tuple is a tuple of corresponding shards of its
  elements. These component buffers reside on distinct devices, but need not
  represent distinct logical shards.
  """
  __slots__ = ["device_buffers", "axis_size", "aval"]

  def __init__(self, axis_size, aval, device_buffers):
    assert device_buffers
    self.device_buffers = device_buffers
    self.axis_size = axis_size
    self.aval = aval

  # To destructure, we destructure the constituent buffers on each device, then
  # logically concatenate those shards across devices producing one logically
  # concatenated result per element. The logical concatenation is performed with
  # the result handler logic applied to the elements.
  def __iter__(self):
    all_bufs = zip(*[buf.destructure() for buf in self.device_buffers])
    handlers = map(partial(tuple_element_handler, self.axis_size), self.aval)
    elts = [handler(bufs) for handler, bufs in zip(handlers, all_bufs)]
    return iter(elts)

  def __len__(self):
    return len(self.aval)

  def __repr__(self):
    return 'ShardedDeviceTuple(len={length})'.format(length=len(self))

def tuple_element_handler(axis_size, aval):
  t = type(aval)
  if t is core.AbstractTuple:
    return partial(ShardedDeviceTuple, axis_size, aval)
  elif t is ShapedArray:
    return partial(ShardedDeviceArray, aval)
  else:
    raise TypeError(t)


core.pytype_aval_mappings[ShardedDeviceTuple] = core.pytype_aval_mappings[core.JaxTuple]
xla.pytype_aval_mappings[ShardedDeviceTuple] = op.attrgetter('aval')
xla.canonicalize_dtype_handlers[ShardedDeviceTuple] = \
    xla.canonicalize_dtype_handlers[xla.DeviceTuple]


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

  The number of device buffers underlying a ShardedDeviceArray instance is equal
  to the number of replicas of the computation that produced it. Each buffer
  represents a shard of the original array, meaning a slice along its leading
  axis. These component buffers reside on distinct devices, but need not
  represent distinct logical shards. The correspondence can be computed with
  the assign_shards_to_replicas function.
  """
  __slots__ = ["device_buffers"]

  def __init__(self, aval, device_buffers):
    self.device_buffers = device_buffers
    self.shape, self.dtype = aval.shape, aval.dtype
    self.ndim, self.size = len(aval.shape), prod(aval.shape)
    self._npy_value = None

  @property
  def _value(self):
    if self._npy_value is None:
      npy_shards = [buf.to_py() for buf in self.device_buffers]
      assignments = assign_shards_to_replicas(len(npy_shards), self.shape[0])
      _, ids = onp.unique(assignments, return_index=True)
      self._npy_value = onp.stack([npy_shards[i] for i in ids])
    return self._npy_value

core.pytype_aval_mappings[ShardedDeviceArray] = ConcreteArray
xla.pytype_aval_mappings[ShardedDeviceArray] = \
    xla.pytype_aval_mappings[xla.DeviceArray]
xla.canonicalize_dtype_handlers[ShardedDeviceArray] = \
    xla.canonicalize_dtype_handlers[xla.DeviceArray]

xb.register_constant_handler(ShardedDeviceArray,
                             xla._device_array_constant_handler)


def xla_pmap_impl(fun, *args, **params):
  axis_name = params.pop('axis_name')
  axis_size = params.pop('axis_size')
  assert not params
  abstract_args = map(partial(abstractify, axis_size), args)
  compiled_fun = parallel_callable(fun, axis_name, axis_size, *abstract_args)
  return compiled_fun(*args)

def abstractify(axis_size, x):
  return _shard_aval(axis_size, xla.abstractify(x))

def _shard_aval(axis_size, aval):
  if type(aval) is core.AbstractTuple:
    return core.AbstractTuple(map(partial(_shard_aval, axis_size), aval))
  elif type(aval) is ShapedArray:
    assert aval.shape[0] == axis_size
    return ShapedArray(aval.shape[1:], aval.dtype)
  else:
    raise TypeError(type(aval))

@lu.memoize
def parallel_callable(fun, axis_name, axis_size, *avals):
  pvals = [PartialVal((aval, core.unit)) for aval in avals]
  with core.new_master(JaxprTrace, True) as master:
    jaxpr, (pval, consts, env) = trace_to_subjaxpr(fun, master, False).call_wrapped(pvals)
    assert not env
    out = compile_replicated(jaxpr, axis_name, axis_size, consts, *avals)
    compiled, nrep, shard_result_shape = out
    del master, consts, jaxpr, env
  handle_arg = partial(shard_arg, compiled.DeviceOrdinals(), axis_size)
  handle_replica_result = xla.result_handler(shard_result_shape)
  handle_full_result = sharded_result_handler(axis_size, merged_aval(pval))
  return partial(execute_replicated, compiled, pval, nrep,
                 handle_arg, handle_replica_result, handle_full_result)

def merged_aval(pval):
  pv, const = pval
  if isinstance(pv, core.AbstractValue):
    return pv
  elif isinstance(pv, pe.JaxprTracerTuple):
    return core.AbstractTuple(map(merged_aval, zip(pv, const)))
  elif pv is None:
    return xla.abstractify(const)
  else:
    raise TypeError(type(pv))

def execute_replicated(compiled, pval, nrep, handle_in,
                       handle_replica_result, handle_full_result, *args):
  input_bufs = zip(*map(handle_in, args)) if args else [[]] * nrep
  out_bufs = compiled.ExecutePerReplica(list(input_bufs))
  results = [merge_pvals(handle_replica_result(buf), pval) for buf in out_bufs]
  return handle_full_result(results)


xla_pmap_p = core.Primitive('xla_pmap')
xla_pmap = partial(core.call_bind, xla_pmap_p)
xla_pmap_p.def_custom_bind(xla_pmap)
xla_pmap_p.def_impl(xla_pmap_impl)
ad.primitive_transposes[xla_pmap_p] = partial(ad.map_transpose, xla_pmap_p)
pe.map_primitives.add(xla_pmap_p)
# TODO(mattjj): enable pjit inside jit, maybe by merging xla_pmap and xla_call
# xla.translations[xla_pmap_p] = xla.xla_call_translation_rule


parallel_translation_rules = {}
