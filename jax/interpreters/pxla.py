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

from collections import namedtuple, defaultdict
from distutils.util import strtobool
import itertools as it
import operator as op
import os

import numpy as onp
import six
from six.moves import xrange

from .. import core
from .. import ad_util
from .. import tree_util
from .. import linear_util as lu
from ..abstract_arrays import ShapedArray
from ..util import partial, unzip2, concatenate, safe_map, prod
from ..lib import xla_bridge as xb
from .xla import (flatten_fun, tree_flatten, build_tree, leaf, xla_shape,
                  xla_destructure, translation_rule, abstractify,
                  xla_shape_to_result_shape)
from .partial_eval import trace_to_subjaxpr, merge_pvals, JaxprTrace, PartialVal

map = safe_map


### util


def canonicalize_axis_spec(in_trees, spec):
  spec = (spec,) * len(in_trees) if type(spec) is int else spec
  spec = map(build_axis_spec_tree, spec, in_trees)
  spec = tuple(tree_util.tree_flatten(spec)[0])
  return spec

def build_axis_spec_tree(spec, in_tree):
  if in_tree is leaf:
    assert type(spec) is int
    return spec
  elif type(in_tree) is JTupleTreeDef:
    spec_type = type(spec)
    if spec_type is int:
      return tuple(map(partial(build_axis_spec_tree, spec), in_tree.child_specs))
    elif spec_type is tuple:
      return tuple(map(build_axis_spec_tree, spec, in_tree.child_specs))
    else:
      raise TypeError(spec_type)
  else:
    raise TypeError(type(in_tree))

def remove_mapped_dims(aval, *axes):
  assert type(aval) is ShapedArray
  axes = [d for d in axes if d is not None]
  if len(set(axes)) != len(axes):
    raise ValueError("multiple names mapped to the same axis")
  shape = tuple(onp.delete(aval.shape, axes))
  return ShapedArray(shape, aval.dtype)

def meshgroups(mesh_spec, mesh_axis):
  groups = onp.split(onp.arange(prod(mesh_spec)).reshape(mesh_spec),
                     mesh_spec[mesh_axis], axis=mesh_axis)
  return tuple(tuple(group.ravel()) for group in groups)

def shard_array(mesh_spec, mesh_map, axis_map, x):
  # axis_map , e.g. {'i': 0, 'j': None}  (axis indices)
  # mesh_map , e.g. {'i': 0, 'k': 2}     (mesh indices)
  # mesh_spec, e.g. (2, 4, 4)
  # return flat list of device buffers - one per replica
  mesh_ndim = len(mesh_spec)
  mesh_size = onp.prod(mesh_spec)
  mesh_map_inverted = {v : k for k, v in mesh_map.items()}
  ordered_idx_names = map(mesh_map_inverted.get, range(mesh_ndim))  # [i,None,k]
  axes = map(axis_map.get, ordered_idx_names)
  xs = list_ravel(unstack_axes(mesh_spec, axes, x))
  return map(xb.device_put, xs, range(mesh_size))

def unstack_axes(mesh_spec, axes, x):
  # axes: list of ints. logical axes of x to split, ordered as in mesh_spec
  #  e.g. zeros(4, 10, 2, 6)
  # axes  (2, 0, None[broadcast_size=10])
  # results in (8,) nested lists each with (10,6) arrays
  if axes:
    ax0 = axes[0]
    recur = partial(unstack_axes, mesh_spec[1:], axes[1:])
    if ax0 is None:
      return [recur(x)]  * mesh_spec[0]
    else:
      assert x.shape[ax0] == mesh_spec[0]
      xs = split_array(x, axes[0])
      return list(map(recur, xs))
  else:
    return x

def unshard_array(mesh_spec, mesh_map, axis_map, xs):
  mesh_ndim = len(mesh_spec)
  mesh_size = onp.prod(mesh_spec)
  mesh_map_inverted = {v : k for k, v in mesh_map.items()}
  ordered_idx_names = map(mesh_map_inverted.get, range(mesh_ndim))  # [i,None,k]
  axes = map(axis_map.get, ordered_idx_names)

  example_shard = xs[0]
  num_splits = len([i for i in axes if i is not None])
  shape = iter(example_shard.shape)
  newshape = [1 if i in axes else next(shape)
              for i in range(num_splits + example_shard.ndim)]

  xs = [x.reshape(newshape) for x in xs]
  xs = list_reshape(mesh_spec, iter(xs))
  x = stack_axes(mesh_spec, axes, xs)
  return x

def list_reshape(mesh_spec, flat_xs):
  if mesh_spec:
    return [list_reshape(mesh_spec[1:], flat_xs) for _ in range(mesh_spec[0])]
  else:
    return next(flat_xs)

def stack_axes(mesh_spec, axes, xs):
  if mesh_spec:
    if mesh_spec[0] is None:
      return stack_axes(mesh_spec[1:], axes[1:], xs[0])
    else:
      components = map(partial(stack_axes, mesh_spec[1:], axes[1:]), xs)
      return onp.concatenate(components, axis=axes[0])
  else:
    return xs

def list_ravel(xs):
  if isinstance(xs, list):
    return concatenate(map(list_ravel, xs))
  else:
    return [xs]

def split_array(x, axis):
  def get_nth_subarray(n):
    idx = [slice(None)] * x.ndim
    idx[axis] = n
    return x[tuple(idx)]
  return map(get_nth_subarray, range(x.shape[axis]))


### xla_pcall


def compile_replicated(jaxpr, devicegrps, consts, *abstract_args):
  arg_shapes = list(map(xla_shape, abstract_args))
  built_c = replicated_jaxpr_computation(jaxpr, devicegrps, consts, (),
                                         *arg_shapes)
  result_shape = xla_shape_to_result_shape(built_c.GetReturnValueShape())
  return built_c.Compile(arg_shapes, xb.get_compile_options()), result_shape

def replicated_jaxpr_computation(jaxpr, devicegrps,
                                 const_vals, freevar_shapes, *arg_shapes):
  c = xb.make_computation_builder("replicated_jaxpr_computation")

  def read(v):
    return env[v]

  def write(v, node):
    assert node is not None
    env[v] = node

  env = {}
  consts_env = dict(zip(jaxpr.constvars, const_vals))
  write(core.unitvar, c.Tuple())
  map(write, jaxpr.constvars, map(c.Constant, const_vals))
  map(write, jaxpr.freevars, map(c.ParameterWithShape, freevar_shapes))
  map(write, jaxpr.invars, map(c.ParameterWithShape, arg_shapes))
  for eqn in jaxpr.eqns:
    in_nodes = map(read, eqn.invars)
    if eqn.primitive in parallel_translation_rules:
      rule = parallel_translation_rules[eqn.primitive]
      axis_name = eqn.params['axis_name']
      params = {k: eqn.params[k] for k in eqn.params if k != 'axis_name'}
      ans = rule(c, *in_nodes, device_groups=devicegrps[axis_name], **params)
    else:
      if eqn.bound_subjaxprs: raise NotImplementedError  # TODO check primitive
      ans = translation_rule(eqn.primitive)(c, *in_nodes, **eqn.params)

    out_nodes = xla_destructure(c, ans) if eqn.destructure else [ans]
    map(write, eqn.outvars, out_nodes)
  return c.Build(read(jaxpr.outvar))


def xla_pcall_impl(fun, *args, **params):
  axis_map = params.pop('axis_map')    # e.g. {'i': 0, 'j': (None, 1)}
  mesh_map = params.pop('mesh_map')    # e.g. {'i': 0, 'j': 2}
  mesh_spec = params.pop('mesh_spec')  # e.g. (2, 2, 2)
  out_axis_map = params.pop('out_axis_map')    # e.g. {'i': 0, 'j': (None, 1)}
  assert not params

  flat_args, in_trees = unzip2(map(tree_flatten, args))
  flat_args = concatenate(flat_args)
  fun, out_tree = flatten_fun(fun, in_trees)

  mesh_map = tuple(sorted(mesh_map.items()))
  axis_map = tuple((axis_name, canonicalize_axis_spec(in_trees, spec))
                   for axis_name, spec in sorted(axis_map.items()))
  compiled_fun = xla_parallel_callable(fun, axis_map, mesh_map, mesh_spec,
                                       *map(abstractify, flat_args))

  out_axis_map = {axis_name : canonicalize_axis_spec([out_tree()], spec)[0]
                  for axis_name, spec in out_axis_map.items()}
  flat_ans = compiled_fun(out_tree(), out_axis_map, *args)

  if out_tree() is leaf:
    return flat_ans
  else:
    return build_tree(iter(flat_ans), out_tree())

@lu.memoize
def xla_parallel_callable(fun, axis_map, mesh_map, mesh_spec, *abstract_args):
  axis_map, mesh_map = dict(axis_map), dict(mesh_map)

  # check that all mapped axes have the right size
  for axis_name in axis_map:
    if not all(axis is None or arg.shape[axis] == mesh_spec[mesh_map[axis_name]]
               for arg, axis in zip(abstract_args, axis_map[axis_name])):
      msg = "axis size does not match mesh size for axis name {}"
      raise ValueError(msg.format(axis_name))

  # construct abstract args with sharded dimensions removed
  abstract_args = map(remove_mapped_dims, abstract_args, *axis_map.values())

  # process mesh_spec and mesh_map into a mapping to device groups
  devicegrps = {axis_name : meshgroups(mesh_spec, mesh_map[axis_name])
                for axis_name in mesh_map}

  pvals = [PartialVal((aval, core.unit)) for aval in abstract_args]
  with core.new_master(JaxprTrace, True) as master:
    jaxpr, (pval, consts, env) = trace_to_subjaxpr(fun, master).call_wrapped(pvals)
    assert not env
    compiled, result_shape = compile_replicated(jaxpr, devicegrps,
                                                consts, *abstract_args)
    del master, consts, jaxpr, env

  return partial(execute_replicated, axis_map, mesh_map, mesh_spec, compiled, pval)

def execute_replicated(axis_map, mesh_map, mesh_spec, compiled, pval,
                       out_tree, out_axis_map, *args):
  axis_maps = [{axis_name : axes[i] for axis_name, axes in axis_map.items()}
               for i in range(len(args))]
  input_bufs = map(partial(shard_array, mesh_spec, mesh_map), axis_maps, args)
  out_bufs = compiled.ExecutePerReplica(zip(*input_bufs))
  if out_tree is leaf:
    # TODO sharded device persistence
    out_shards = [merge_pvals(out_buf.to_py(), pval) for out_buf in out_bufs]
    return unshard_array(mesh_spec, mesh_map, out_axis_map, out_shards)
  else:
    raise NotImplementedError  # TODO


xla_pcall_p = core.Primitive('xla_pcall')
xla_pcall = partial(core.call_bind, xla_pcall_p)
xla_pcall_p.def_custom_bind(xla_pcall)
xla_pcall_p.def_impl(xla_pcall_impl)


parallel_translation_rules = {}
