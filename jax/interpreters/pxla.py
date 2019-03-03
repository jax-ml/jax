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

import numpy as onp
import six
from six.moves import reduce

from .. import core
from .. import ad_util
from .. import tree_util
from .. import linear_util as lu
from ..abstract_arrays import ConcreteArray, ShapedArray
from ..util import partial, unzip2, concatenate, safe_map, prod
from ..lib import xla_bridge as xb
from .xla import (xla_shape, xla_destructure, translation_rule,
                  xla_shape_to_result_shape, jaxpr_computation)
from .partial_eval import trace_to_subjaxpr, merge_pvals, JaxprTrace, PartialVal
from .batching import dimsize, broadcast
from . import partial_eval as pe
from . import parallel
from . import xla
from . import ad

map = safe_map


### util


def shard_arg(device_ordinals, arg):
  """Shard an argument data array arg along its leading axis.

  Args:
    device_ordinals: list of integers of length num_replicas mapping a logical
      replica index to a physical device number.
    arg: an array type representing an argument value to be sharded along its
      leading axis and placed on the devices/replicas.

  Returns:
    A list of length num_replicas of device buffers indexed by replica number,
    where the nth element is the argument to be passed to the nth replica.
  """
  nrep = len(device_ordinals)
  assignments = assign_shards_to_replicas(nrep, arg.shape[0])
  if type(arg) is ShardedDeviceArray and nrep == len(arg.device_buffers):
    # TODO(mattjj, phawkins): improve re-distribution not to copy to host
    _, ids = onp.unique(assignments, return_index=True)
    shards = [arg.device_buffers[i].to_py() for i in ids]  # TODO(mattjj): lazy
    return [buf if buf.device() == device_ordinals[r]
            else xla.device_put(shards[i], device_ordinals[r])
            for r, (i, buf) in enumerate(zip(assignments, arg.device_buffers))]
  else:
    shards = [arg[i] for i in range(arg.shape[0])]
    return [xla.device_put(shards[i], device_ordinals[r])
            for r, i in enumerate(assignments)]

def unshard_output(axis_size, replica_results):
  """Collect together replica results into a result value.

  Args:
    axis_size: size of the sharded output data axis.
    replica_results: list of either ndarrays or DeviceArrays indexed by replica
      number.

  Returns:
    Either an ndarray or a ShardedDeviceArray representing the result of the
    computation, stacking together the results from the replicas.
  """
  nrep = len(replica_results)
  if all(type(res) is xla.DeviceArray for res in replica_results):
    return ShardedDeviceArray(axis_size, replica_results)
  else:
    assignments = assign_shards_to_replicas(nrep, axis_size)
    _, ids = onp.unique(assignments, return_index=True)
    return onp.stack([replica_results[i] for i in ids])

def assign_shards_to_replicas(nrep, size):
  """Produce a mapping from replica id to shard index.

  Args:
    nrep: int, number of relpicas (a computation-dependent value).
    size: int, size of the data array axis being sharded.

  Returns:
    A tuple of integers of length nrep in which the elements take on values from
    0 to size-1. Replica n is assgined shard data_array[assignments[n]].
  """
  groupsize, ragged = divmod(nrep, size)
  assert not ragged
  indices = onp.tile(onp.arange(size)[:, None], (1, groupsize))
  return tuple(indices.ravel())

def replica_groups(nrep, mesh_spec, mesh_axis):
  trailing_size, ragged = divmod(nrep, prod(mesh_spec))
  assert not ragged
  full_spec = mesh_spec + [trailing_size]
  groups = onp.split(onp.arange(prod(full_spec)).reshape(full_spec),
                     full_spec[mesh_axis], axis=mesh_axis)
  groups = map(onp.ravel, groups)
  return tuple(tuple(group) for group in zip(*groups))

def xla_shard(c, sizes, x):
  def _xla_shard(shape, x):
    if shape.is_tuple():
      elts = map(_xla_shard, shape.tuple_shapes(), xla_destructure(c, x))
      return c.Tuple(*elts)
    else:
      return shard_array(shape, x)

  def shard_array(shape, x):
    dims = list(shape.dimensions())
    assert dims[0] == sizes[-1]
    start_indices = xla_shard_start_indices(c, dims[0], len(dims))
    return c.Reshape(c.DynamicSlice(x, start_indices, [1] + dims[1:]),
                     None, dims[1:])

  return _xla_shard(c.GetShape(x), x)

# TODO(b/110096942): more efficient gather
def xla_unshard(c, device_groups, x):
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

def xla_shard_start_indices(c, axis_size, ndim):
  idx = c.Rem(c.ReplicaId(), c.Constant(onp.array(axis_size, onp.uint32)))
  zero = onp.zeros(ndim - 1, onp.uint32)
  return c.Concatenate([c.Reshape(idx, None, [1]), c.Constant(zero)], 0)


### xla_pcall


AxisEnv = namedtuple("AxisEnv", ["nreps", "names", "sizes"])

def axis_read(axis_env, axis_name):
  return max(i for i, name in enumerate(axis_env.names) if name == axis_name)

def axis_groups(axis_env, name):
  mesh_axis = axis_read(axis_env, name)
  return replica_groups(axis_env.nreps, axis_env.sizes, mesh_axis)

def compile_replicated(jaxpr, axis_name, axis_size, consts, *abstract_args):
  num_replicas = axis_size * jaxpr_replicas(jaxpr)
  axis_env = AxisEnv(num_replicas, [axis_name], [axis_size])
  arg_shapes = list(map(xla_shape, abstract_args))
  built_c = replicated_comp(jaxpr, axis_env, consts, (), *arg_shapes)
  result_shape = xla_shape_to_result_shape(built_c.GetReturnValueShape())
  compiled = built_c.Compile(arg_shapes, xb.get_compile_options(num_replicas))
  return compiled, num_replicas, result_shape

def jaxpr_replicas(jaxpr):
  return _max(eqn.params['axis_size'] * jaxpr_replicas(eqn.bound_subjaxprs[0][0])
              for eqn in jaxpr.eqns if eqn.primitive is xla_pcall_p)
def _max(itr): return max(list(itr) or [1])

def replicated_comp(jaxpr, ax_env, const_vals, freevar_shapes, *arg_shapes):
  c = xb.make_computation_builder("replicated_computation")

  def read(v):
    return env[v]

  def write(v, node):
    assert node is not None
    env[v] = node

  def axis_env_extend(name, size):
    return AxisEnv(ax_env.nreps, ax_env.names + [name], ax_env.sizes + [size])

  env = {}
  write(core.unitvar, c.Tuple())
  if const_vals:
    map(write, jaxpr.constvars, map(c.Constant, const_vals))
    map(write, jaxpr.freevars, map(c.ParameterWithShape, freevar_shapes))
  else:
    all_freevars = it.chain(jaxpr.constvars, jaxpr.freevars)
    map(write, all_freevars, map(c.ParameterWithShape, freevar_shapes))
  map(write, jaxpr.invars, map(c.ParameterWithShape, arg_shapes))
  for eqn in jaxpr.eqns:
    in_nodes = map(read, eqn.invars)
    if eqn.primitive in parallel_translation_rules:
      name = eqn.params['axis_name']
      params = {k: eqn.params[k] for k in eqn.params if k != 'axis_name'}
      rule = parallel_translation_rules[eqn.primitive]
      ans = rule(c, *in_nodes, device_groups=axis_groups(ax_env, name), **params)
    elif eqn.bound_subjaxprs:
      if eqn.primitive is xla_pcall_p:
        name, size = eqn.params['axis_name'], eqn.params['axis_size']
        new_env = axis_env_extend(name, size)
        in_shards = map(partial(xla_shard, c, new_env.sizes), in_nodes)
        in_shapes = map(c.GetShape, in_shards)
        (subjaxpr, const_bindings, freevar_bindings), = eqn.bound_subjaxprs
        subc = replicated_comp(
            subjaxpr, new_env, (),
            map(c.GetShape, map(read, const_bindings + freevar_bindings)),
            *in_shapes)
        subfun = (subc, tuple(map(read, const_bindings + freevar_bindings)))
        sharded_result = xla.xla_call_translation_rule(c, subfun, *in_shards)
        ans = xla_unshard(c, axis_groups(new_env, name), sharded_result)
      else:
        in_shapes = map(c.GetShape, in_nodes)
        subcs = [
            jaxpr_computation(
                subjaxpr, (),
                map(c.GetShape, map(read, const_bindings + freevar_bindings)),
                *in_shapes)
            for subjaxpr, const_bindings, freevar_bindings in eqn.bound_subjaxprs]
        subfuns = [(subc, tuple(map(read, const_bindings + freevar_bindings)))
                    for subc, (_, const_bindings, freevar_bindings)
                    in zip(subcs, eqn.bound_subjaxprs)]
        ans = translation_rule(eqn.primitive)(c, *(subfuns + in_nodes), **eqn.params)
    else:
      ans = translation_rule(eqn.primitive)(c, *in_nodes, **eqn.params)

    out_nodes = xla_destructure(c, ans) if eqn.destructure else [ans]
    map(write, eqn.outvars, out_nodes)
  return c.Build(read(jaxpr.outvar))


class ShardedDeviceArray(xla.DeviceArray):
  __slots__ = ["device_buffers"]

  def __init__(self, axis_size, replica_results):
    self.device_buffers = [res.device_buffer for res in replica_results]
    r = replica_results[0]
    self.shape = (axis_size,) + r.shape
    self.dtype = r.dtype
    self.ndim = 1 + r.ndim
    self.size = axis_size * r.size
    self._npy_value = None

  @property
  def _value(self):
    if self._npy_value is None:
      npy_shards = [buf.to_py() for buf in self.device_buffers]
      self._npy_value = unshard_output(self.shape[0], npy_shards)
    return self._npy_value

core.pytype_aval_mappings[ShardedDeviceArray] = ConcreteArray
xla.pytype_aval_mappings[ShardedDeviceArray] = \
    xla.pytype_aval_mappings[xla.DeviceArray]
xla.canonicalize_dtype_handlers[ShardedDeviceArray] = \
    xla.canonicalize_dtype_handlers[xla.DeviceArray]

xb.register_constant_handler(ShardedDeviceArray,
                             xla._device_array_constant_handler)


def xla_pcall_impl(fun, *args, **params):
  axis_name = params.pop('axis_name')
  axis_size = params.pop('axis_size')
  assert not params

  flat_args, in_trees = unzip2(map(xla.tree_flatten, args))
  flat_args = concatenate(flat_args)
  fun, out_tree = xla.flatten_fun(fun, in_trees)

  abstract_args = map(partial(abstractify, axis_size), flat_args)
  compiled_fun = parallel_callable(fun, axis_name, axis_size, *abstract_args)
  flat_ans = compiled_fun(out_tree(), *flat_args)

  if out_tree() is xla.leaf:
    return flat_ans
  else:
    return xla.build_tree(iter(flat_ans), out_tree())

def abstractify(axis_size, x):
  return _shard_aval(axis_size, xla.abstractify(x))

def _shard_aval(axis_size, aval):
  if type(aval) is core.AbstractTuple:
    return AbstractTuple(map(partial(_shard_aval, axis_size), aval))
  elif type(aval) is ShapedArray:
    assert aval.shape[0] == axis_size
    return ShapedArray(aval.shape[1:], aval.dtype)
  else:
    raise TypeError(type(aval))

@lu.memoize
def parallel_callable(fun, axis_name, axis_size, *avals):
  pvals = [PartialVal((aval, core.unit)) for aval in avals]
  with core.new_master(JaxprTrace, True) as master:
    jaxpr, (pval, consts, env) = trace_to_subjaxpr(fun, master).call_wrapped(pvals)
    assert not env
    out = compile_replicated(jaxpr, axis_name, axis_size, consts, *avals)
    compiled, nrep, result_shape = out
    del master, consts, jaxpr, env
  handle_arg = partial(shard_arg, compiled._device_ordinals)
  handle_result = xla.result_handler(result_shape)
  return partial(execute_replicated, compiled, pval, axis_size, nrep,
                 handle_arg, handle_result)

def execute_replicated(compiled, pval, axis_size, nrep, handle_in, handle_out,
                       out_tree, *args):
  input_bufs = zip(*map(handle_in, args)) if args else [[]] * nrep
  out_bufs = compiled.ExecutePerReplica(input_bufs)
  replica_results = [merge_pvals(handle_out(buf), pval) for buf in out_bufs]
  if out_tree is xla.leaf:
    return unshard_output(axis_size, replica_results)
  else:
    return map(partial(unshard_output, axis_size), zip(*replica_results))


xla_pcall_p = core.Primitive('xla_pcall')
xla_pcall = partial(core.call_bind, xla_pcall_p)
xla_pcall_p.def_custom_bind(xla_pcall)
xla_pcall_p.def_impl(xla_pcall_impl)
ad.primitive_transposes[xla_pcall_p] = partial(ad.map_transpose, xla_pcall_p)
pe.map_primitives.add(xla_pcall_p)
# TODO(mattjj): enable pjit inside jit, maybe by merging xla_pcall and xla_call
# xla.translations[xla_pcall_p] = xla.xla_call_translation_rule


parallel_translation_rules = {}
