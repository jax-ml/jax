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
from ..abstract_arrays import ShapedArray
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


def shard_arg(nrep, arg):
  sz = arg.shape[0]
  shards = [arg[i] for i in range(sz)]
  assignments = assign_shards(nrep, sz)
  return [xb.device_put(shards[i], n) for n, i in enumerate(assignments)]

def unshard_output(nrep, axis_size, out_shards):
  _, ids = onp.unique(assign_shards(nrep, axis_size), return_index=True)
  return onp.stack([out_shards[i] for i in ids])

def assign_shards(nrep, size):
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

def xla_shard(c, axis_sizes, x):
  def _xla_shard(shape, x):
    if shape.is_tuple():
      elts = map(_xla_shard, shape.tuple_shapes(), xla_destructure(c, x))
      return c.Tuple(*elts)
    else:
      return shard_array(shape, x)

  def shard_array(shape, x):
    if xb.get_replica_count() == 1:
      # TODO(mattjj): remove this special case, used for debugging on CPU
      # because CPU doesn't have some collectives implemented
      dims = c.GetShape(x).dimensions()
      return c.Reshape(x, None, dims[1:])
    else:
      size = onp.array(prod(axis_sizes), onp.uint32)
      idx = c.Rem(c.ReplicaId(), c.Constant(size))
      dims = list(shape.dimensions())
      zero = onp.zeros(len(dims) - 1, onp.uint32)
      start_indices = c.Concatenate([c.Reshape(idx, None, [1]), c.Constant(zero)], 0)
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
      return unshard_array(x)

  def unshard_array(x):
      # TODO(mattjj): remove this special case, used for debugging on CPU
      # because CPU doesn't have some collectives implemented
    if xb.get_replica_count() == 1:
      dims = c.GetShape(x).dimensions()
      return c.Reshape(x, None, (1,) + tuple(dims))
    else:
      group_size = len(device_groups[0])
      broadcasted = c.Broadcast(x, (group_size,))
      return c.AllToAll(broadcasted, 0, 0, device_groups)

  return _xla_unshard(c.GetShape(x), x)


### xla_pcall


AxisEnv = namedtuple("AxisEnv", ["nreps", "names", "sizes"])

def axis_read(axis_env, axis_name):
  return max(i for i, name in enumerate(axis_env.names) if name == axis_name)

def axis_groups(axis_env, name):
  mesh_axis = axis_read(axis_env, name)
  return replica_groups(axis_env.nreps, axis_env.sizes, mesh_axis)

def compile_replicated(jaxpr, axis_name, axis_size, consts, *abstract_args):
  num_replicas = axis_size * jaxpr_replicas(jaxpr)
  assert num_replicas <= xb.get_replica_count()
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
        name = eqn.params['axis_name']
        new_env = axis_env_extend(name, eqn.params['axis_size'])
        in_nodes = map(partial(xla_shard, c, new_env.sizes), in_nodes)
        in_shapes = map(c.GetShape, in_nodes)
        (subjaxpr, const_bindings, freevar_bindings), = eqn.bound_subjaxprs
        subc = replicated_comp(
            subjaxpr, new_env, (),
            map(c.GetShape, map(read, const_bindings + freevar_bindings)),
            *in_shapes)
        subfun = (subc, tuple(map(read, const_bindings + freevar_bindings)))
        sharded_result = xla.xla_call_translation_rule(c, subfun, *in_nodes)
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
  assert onp.shape(x)[0] == axis_size
  aval = xla.abstractify(x)
  return ShapedArray(aval.shape[1:], aval.dtype)

@lu.memoize
def parallel_callable(fun, axis_name, axis_size, *avals):
  pvals = [PartialVal((aval, core.unit)) for aval in avals]
  with core.new_master(JaxprTrace, True) as master:
    jaxpr, (pval, consts, env) = trace_to_subjaxpr(fun, master).call_wrapped(pvals)
    assert not env
    compiled, nrep, _ = compile_replicated(jaxpr, axis_name, axis_size, consts, *avals)
    del master, consts, jaxpr, env
  return partial(execute_replicated, compiled, pval, axis_size, nrep)

def execute_replicated(compiled, pval, axis_size, nrep, out_tree, *args):
  input_bufs = zip(*map(partial(shard_arg, nrep), args)) if args else [[]] * nrep
  out_bufs = compiled.ExecutePerReplica(input_bufs)
  out_shards = [merge_pvals(buf.to_py(), pval) for buf in out_bufs]
  if out_tree is xla.leaf:
    return unshard_output(nrep, axis_size, out_shards)
  else:
    return map(partial(unshard_output, nrep, axis_size), zip(*out_shards))


xla_pcall_p = core.Primitive('xla_pcall')
xla_pcall = partial(core.call_bind, xla_pcall_p)
xla_pcall_p.def_custom_bind(xla_pcall)
xla_pcall_p.def_impl(xla_pcall_impl)
ad.primitive_transposes[xla_pcall_p] = partial(ad.map_transpose, xla_pcall_p)
pe.map_primitives.add(xla_pcall_p)
# TODO(mattjj): enable pjit inside jit, maybe by merging xla_pcall and xla_call
# xla.translations[xla_pcall_p] = xla.xla_call_translation_rule


parallel_translation_rules = {}
