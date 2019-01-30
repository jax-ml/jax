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
from contextlib import contextmanager
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
from ..tree_util import leaf
from ..lib import xla_bridge as xb
from .xla import (flatten_fun, tree_flatten, build_tree, xla_shape,
                  xla_destructure, translation_rule, abstractify,
                  xla_shape_to_result_shape, jaxpr_computation)
from .partial_eval import trace_to_subjaxpr, merge_pvals, JaxprTrace, PartialVal
from .parallel import parallel_translation_rules
from .batching import moveaxis
from . import parallel
from . import xla

map = safe_map


mesh_spec = None


### util


def chunk_transform(fun, chunksize, name, in_axes, out_axes_dst):
  """Rewrite SPMD operations to act first on local chunks then cross-replica."""
  temp_name = object()  # TODO gensym
  fun = parallel.axisvar_split(fun, name, (temp_name, name))
  fun, out_axes_src = parallel.pmap_transform(fun, temp_name, in_axes)
  fun = move_output_axis_transform(fun, chunksize, out_axes_src, out_axes_dst)
  return fun

@lu.transformation
def move_output_axis_transform(chunksize, src, dst, *args):
  """Function transformation that moves output axes from src to dst."""
  ans = yield args
  yield moveaxis(chunksize, dst(), src(), ans)

def chunk_aval(chunksize, aval, axis):
  """Transform an abstract value's shape to have chunksize extent along axis."""
  if axis is None:
    return aval
  else:
    shape = list(aval.shape)
    shape[axis] = chunksize
    return ShapedArray(tuple(shape), aval.dtype)

def canonicalize_in_axis_spec(in_tree, spec_tree_prefix):
  """Given argument list in_tree, canonicalize and flatten an in_axes spec."""
  spec_tree = build_axis_spec_tree(spec_tree_prefix, in_tree)
  return flatten_axis_spec_tree(spec_tree)

def canonicalize_out_axis_spec(out_tree, spec_tree_prefix):
  """Given output out_tree, canonicalize and flatten an out_axes spec."""
  if out_tree is leaf:
    return spec_tree_prefix
  else:
    spec_tree = build_axis_spec_tree(spec_tree_prefix, out_tree)
    return flatten_axis_spec_tree(spec_tree)

def build_axis_spec_tree(spec, treedef):
  """Given a tree_util treedef, canonicalize an axis spec for that treedef."""
  spec_type = type(spec)
  if treedef is leaf:
    if spec_type is int:
      return spec
    elif spec_type is type(None):
      return no_mapped_axis
    else:
      raise TypeError(spec_type)
  else:
    if spec_type is int:
      return tuple(map(partial(build_axis_spec_tree, spec), treedef.children))
    elif spec_type is tuple:
      return tuple(map(build_axis_spec_tree, spec, treedef.children))
    else:
      raise TypeError(spec_type)

def flatten_axis_spec_tree(spec_tree):
  """Flatten an axis spec tree and replace no_mapped_axis with None."""
  spec_flat, _ = tree_util.tree_flatten(spec_tree)
  return tuple(None if i is no_mapped_axis else i for i in spec_flat)

# We use a special symbol for 'no mapped axis' instead of using None because
# tree_util.py treats None as a tree node.
class NoMappedAxis(object): pass
no_mapped_axis = NoMappedAxis()

def shard_arg(mesh_spec, mesh_axis, axis, arg):
  """Shard and device_put an input array argument along a logical axis."""
  num_replicas = xb.get_replica_count()
  if prod(mesh_spec) != num_replicas:
    msg = "mesh spec {} total size of {} doesn't match number of replicas {}."
    raise ValueError(msg.format(mesh_spec, prod(mesh_spec), num_replicas))
  shards = split_array(arg, mesh_spec[mesh_axis], axis)
  replica_shards = [shards[i] for i in shard_assignments(mesh_spec, mesh_axis)]
  return map(xb.device_put, replica_shards, range(num_replicas))

def unshard_output(mesh_spec, mesh_axis, out_axis, out_shards):
  """Collect and concatenate sharded device results."""
  _, ids = onp.unique(shard_assignments(mesh_spec, mesh_axis), return_index=True)
  shards = [out_shards[i] for i in ids]
  return onp.concatenate(shards, out_axis)  # TODO device persistence

def shard_assignments(mesh_spec, mesh_axis):
  """Given a mesh axis long which to shard data, compute replica assignments."""
  indices_shape = [1] * len(mesh_spec)
  indices_shape[mesh_axis] = mesh_spec[mesh_axis]
  indices = onp.arange(mesh_spec[mesh_axis]).reshape(indices_shape)
  return tuple(onp.broadcast_to(indices, mesh_spec).ravel())

def replica_groups(mesh_spec, mesh_axis):
  """Given a mesh axis along which to operate, compute XLA replica_groups."""
  groups = onp.split(onp.arange(prod(mesh_spec)).reshape(mesh_spec),
                     mesh_spec[mesh_axis], axis=mesh_axis)
  groups = map(onp.ravel, groups)
  return tuple(tuple(group) for group in zip(*groups))

def split_array(x, num_splits, axis):
  """A special-case of numpy.split implemented in terms of indexing."""
  if axis is None:
    return [x] * num_splits
  else:
    assert x.shape[axis] % num_splits == 0
    split_size = x.shape[axis] // num_splits
    def get_nth_subarray(n):
      idx = [slice(None)] * x.ndim
      idx[axis] = slice(n * split_size, (n+1) * split_size)
      return x[tuple(idx)]
    return map(get_nth_subarray, range(num_splits))

def chunk_size(mesh_axis, in_axes, abstract_args):
  """Compute the chunk size for mapped axes, checking for errors."""
  global mesh_spec
  axis_sizes = {arg.shape[axis] for arg, axis in zip(abstract_args, in_axes)
                if axis is not None}
  if len(axis_sizes) == 0:
    msg = "axis name '{}' not bound to any input axes."
    raise ValueError(msg.format(axis_name))
  elif len(axis_sizes) > 1:
    msg = "axis name '{}' bound to multiple axes with different sizes: {}."
    raise ValueError(msg.format(axis_name, axis_sizes))
  else:
    axis_size = axis_sizes.pop()
    if axis_size % mesh_spec()[mesh_axis]:
      msg = ("axis name '{}' bound to input axis of size {} mapped to mesh "
             "axis index {} with size {}, which does not evenly divide {}.")
      raise ValueError(msg.format(axis_name, axis_size, mesh_axis,
                                  mesh_spec()[mesh_axis], axis_size))

  return axis_size // mesh_spec()[mesh_axis]


def mesh_spec():
  global _mesh_spec
  return _mesh_spec or (xb.get_replica_count(),)
_mesh_spec = None

@contextmanager
def device_mesh(spec):
  global _mesh_spec
  _mesh_spec, prev_spec = spec, _mesh_spec
  yield
  _mesh_spec = prev_spec


### xla_pcall


def compile_replicated(jaxpr, axis_env, consts, *abstract_args):
  arg_shapes = list(map(xla_shape, abstract_args))
  built_c = replicated_computation(jaxpr, axis_env, consts, (), *arg_shapes)
  result_shape = xla_shape_to_result_shape(built_c.GetReturnValueShape())
  return built_c.Compile(arg_shapes, xb.get_compile_options()), result_shape

def replicated_computation(jaxpr, axis_env, const_vals, freevar_shapes,
                                 *arg_shapes):
  c = xb.make_computation_builder("replicated_computation")

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
      device_groups = axis_env[eqn.params['axis_name']]
      params = {k: eqn.params[k] for k in eqn.params if k != 'axis_name'}
      ans = rule(c, *in_nodes, device_groups=device_groups, **params)
    else:
      if eqn.bound_subjaxprs:
        in_shapes = map(c.GetShape, in_nodes)
        if eqn.primitive is xla_pcall_p:
          device_groups = replica_groups(mesh_spec(), eqn.params['mesh_axis'])
          new_axis_binding = {eqn.params['axis_name'] : device_groups}
          (subjaxpr, const_bindings, freevar_bindings), = eqn.bound_subjaxprs
          subc = replicated_computation(
              subjaxpr, extend_env(new_axis_binding, axis_env),
              [consts_env[b] for b in const_bindings],
              map(c.GetShape, map(read, freevar_bindings)),
              *in_shapes)
          subfun = (subc, tuple(map(read, freevar_bindings)))
          ans = translation_rule(eqn.primitive)(c, subfun, *in_nodes)
        else:
          subcs = [jaxpr_computation(subjaxpr,
                                     [consts_env[b] for b in const_bindings],
                                     map(c.GetShape, map(read, freevar_bindings)),
                                     *in_shapes)
                   for subjaxpr, const_bindings, freevar_bindings
                   in eqn.bound_subjaxprs]
          subfuns = [(subc, tuple(map(read, freevar_bindings)))
                      for subc, (_, _, freevar_bindings)
                      in zip(subcs, eqn.bound_subjaxprs)]
          ans = translation_rule(eqn.primitive)(c, *(subfuns + in_nodes), **eqn.params)
      else:
        ans = translation_rule(eqn.primitive)(c, *in_nodes, **eqn.params)

    out_nodes = xla_destructure(c, ans) if eqn.destructure else [ans]
    map(write, eqn.outvars, out_nodes)
  return c.Build(read(jaxpr.outvar))


def xla_pcall_impl(fun, *args, **params):
  axis_name = params.pop('axis_name')  # e.g. 'i'
  in_axes = params.pop('in_axes')      # e.g. 0 or (0, None)
  out_axes = params.pop('out_axes')    # e.g. 0 or (None, 1)
  mesh_axis = params.pop('mesh_axis')  # e.g. 0 or 1
  assert not params

  if all(arg is core.unit for arg in args):
    return fun.call_wrapped(*args)
  else:
    compiled_fun = xla_parallel_callable(fun, axis_name, in_axes, mesh_axis,
                                         mesh_spec(), *map(abstractify, args))
    return compiled_fun(out_axes(), *args)

@lu.memoize
def xla_parallel_callable(fun, axis_name, in_axes, mesh_axis, mesh_spec,
                          *abstract_args):
  chunksize = chunk_size(mesh_axis, in_axes, abstract_args)
  abstract_args = map(partial(chunk_aval, chunksize), abstract_args, in_axes)
  axis_env = new_env({axis_name: replica_groups(mesh_spec, mesh_axis)})
  pvals = [PartialVal((aval, core.unit)) for aval in abstract_args]
  with core.new_master(JaxprTrace, True) as master:
    jaxpr, (pval, consts, env) = trace_to_subjaxpr(fun, master).call_wrapped(pvals)
    assert not env
    compiled, _ = compile_replicated(jaxpr, axis_env, consts, *abstract_args)
    del master, consts, jaxpr, env
  return partial(execute_replicated, in_axes, mesh_axis, mesh_spec, compiled, pval)

def execute_replicated(in_axes, mesh_axis, mesh_spec, compiled, pval,
                       out_axes, *args):
  input_bufs = map(partial(shard_arg, mesh_spec, mesh_axis), in_axes, args)
  out_bufs = compiled.ExecutePerReplica(zip(*input_bufs))
  out_shards = [merge_pvals(buf.to_py(), pval) for buf in out_bufs]  # TODO
  if type(out_axes) is int:
    return unshard_output(mesh_spec, mesh_axis, out_axes, out_shards)
  else:
    return map(partial(unshard_output, mesh_spec, mesh_axis), out_axes,
               zip(*out_shards))


xla_pcall_p = core.Primitive('xla_pcall')
xla_pcall = partial(core.call_bind, xla_pcall_p)
xla_pcall_p.def_custom_bind(xla_pcall)
xla_pcall_p.def_impl(xla_pcall_impl)
xla.translations[xla_pcall_p] = xla.xla_call_translation_rule



def new_env(d):
  return d

def extend_env(d1, d2):
  return dict(d1, **d2)


# def new_env(d):
#   return AxisEnv(d, {})

# def extend_env(d1, d2):
#   return AxisEnv(d1, d2)

# class AxisEnv(dict):
#   def __init__(self, d1, d2):
#     self.update(d1)
#     self.parent = d2

#   def __missing__(self, key):
#     return self.parent[key]

#   def __repr__(self):
#     flat = dict(self, **self.parent)
#     return repr(flat)
