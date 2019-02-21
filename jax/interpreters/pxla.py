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


def shard_arg(arg):
  sz = arg.shape[0]
  shards = [arg[i] for i in range(sz)]
  return [xb.device_put(shards[i], n) for n, i in enumerate(assign_shards(sz))]

def unshard_output(axis_size, out_shards):
  _, ids = onp.unique(assign_shards(axis_size), return_index=True)
  return onp.stack([out_shards[i] for i in ids])

def assign_shards(size):
  groupsize, ragged = divmod(xb.get_replica_count(), size)
  assert not ragged
  indices = onp.tile(onp.arange(size)[:, None], (1, groupsize))
  return tuple(indices.ravel())

def replica_groups(mesh_spec, mesh_axis):
  mesh_spec = mesh_spec + [xb.get_replica_count() // prod(mesh_spec)]
  groups = onp.split(onp.arange(prod(mesh_spec)).reshape(mesh_spec),
                     mesh_spec[mesh_axis], axis=mesh_axis)
  groups = map(onp.ravel, groups)
  return tuple(tuple(group) for group in zip(*groups))


### xla_pcall


AxisEnv = namedtuple("AxisEnv", ["names", "sizes"])

def extend_env(axis_env, name, size):
  return AxisEnv(axis_env.names + [name], axis_env.sizes + [size])

def compile_replicated(jaxpr, axis_name, axis_size, consts, *abstract_args):
  axis_env = AxisEnv([axis_name], [axis_size])
  arg_shapes = list(map(xla_shape, abstract_args))
  built_c = replicated_comp(jaxpr, axis_env, consts, (), *arg_shapes)
  result_shape = xla_shape_to_result_shape(built_c.GetReturnValueShape())
  return built_c.Compile(arg_shapes, xb.get_compile_options()), result_shape

def replicated_comp(jaxpr, axis_env, const_vals, freevar_shapes, *arg_shapes):
  c = xb.make_computation_builder("replicated_computation")

  def read(v):
    return env[v]

  def write(v, node):
    assert node is not None
    env[v] = node

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
      # if we see an spmd primitive (one with a parallel translation rule), then
      # call that parallel translation rule using axis_env for device_groups
      name = eqn.params['axis_name']
      device_groups = replica_groups(axis_env.sizes, axis_env.names.index(name))
      params = {k: eqn.params[k] for k in eqn.params if k != 'axis_name'}
      rule = parallel_translation_rules[eqn.primitive]
      ans = rule(c, *in_nodes, device_groups=device_groups, **params)
    elif eqn.bound_subjaxprs:
      # if there are bound subjaxprs, we either recursively call
      # replicated_computation or call into xla.jaxpr_computation
      in_shapes = map(c.GetShape, in_nodes)
      if eqn.primitive is xla_pcall_p:
        # if we're processing an xla_pcall, extend the axis environment and
        # recursively call replicated_computation
        (subjaxpr, const_bindings, freevar_bindings), = eqn.bound_subjaxprs
        subc = replicated_comp(
            subjaxpr,
            extend_env(axis_env, eqn.params['axis_name'], eqn.params['axis_size']),
            (), map(c.GetShape, map(read, const_bindings + freevar_bindings)),
            *in_shapes)
        subfun = (subc, tuple(map(read, const_bindings + freevar_bindings)))
        # select the correct subarray for this replica, call subfun
        in_nodes = map(partial(xla_split, c), in_nodes)
        ans = translation_rule(eqn.primitive)(c, subfun, *in_nodes)
      else:
        # otherwise, act like xla.jaxpr_computation
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
      # if this is a standard translation rule (not an spmd primitive, not a
      # call with bound subjaxprs) then we lower like xla.jaxpr_computation does
      ans = translation_rule(eqn.primitive)(c, *in_nodes, **eqn.params)

    out_nodes = xla_destructure(c, ans) if eqn.destructure else [ans]
    map(write, eqn.outvars, out_nodes)
  return c.Build(read(jaxpr.outvar))

def xla_split(c, x):
  shape = list(c.GetShape(x).dimensions())
  start_indices = [c.ReplicaId()] + [c.Constant(0)] * len(shape)
  return c.Reshape(c.DynamicSlice(x, start_indices, [1] + shape[1:]), shape[1:])


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
    compiled, _ = compile_replicated(jaxpr, axis_name, axis_size, consts, *avals)
    del master, consts, jaxpr, env
  return partial(execute_replicated, compiled, pval, axis_size)

def execute_replicated(compiled, pval, axis_size, out_tree, *args):
  input_bufs = zip(*map(shard_arg, args)) if args else [[]] * xb.get_replica_count()
  out_bufs = compiled.ExecutePerReplica(input_bufs)
  out_shards = [merge_pvals(buf.to_py(), pval) for buf in out_bufs]
  if out_tree is xla.leaf:
    return unshard_output(axis_size, out_shards)
  else:
    return map(partial(unshard_output, axis_size), zip(*out_shards))


xla_pcall_p = core.Primitive('xla_pcall')
xla_pcall = partial(core.call_bind, xla_pcall_p)
xla_pcall_p.def_custom_bind(xla_pcall)
xla_pcall_p.def_impl(xla_pcall_impl)
ad.primitive_transposes[xla_pcall_p] = partial(ad.call_transpose, xla_pcall_p)
xla.translations[xla_pcall_p] = xla.xla_call_translation_rule
pe.map_primitives.add(xla_pcall_p)


parallel_translation_rules = {}
