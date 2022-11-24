# Copyright 2022 The JAX Authors.
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

"""Shard map implementation.

Shard map allows you to drop in to a per-device view within pjit.

You could do this with xmap, but there are sharp edges currently
and this is a nicer API!

This is _highly_ experimental!
"""
from functools import partial

from typing import Callable, Optional, Tuple

from jax import core
from jax import linear_util as lu
from jax import util
from jax._src import source_info_util
from jax.api_util import flatten_fun_nokwargs
from jax.experimental import maps
from jax.experimental import pjit
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe
from jax.interpreters import pxla
from jax.tree_util import tree_flatten
from jax.tree_util import tree_map
from jax.tree_util import tree_unflatten
import numpy as np
import math

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


# Notes:
#  1. it'd be nice if we could avoid annotating the input shardings, and instead
#     just use argument sharding. that could work eagerly, but when staged out
#     we have a phase ordering problem: xla spmd partitioner doesn't decide
#     shardings until we've generated all the HLO for it, but we'd need to
#     change what (static-shape) HLO we generate based on input shardings
#     (specifically the axis sizes). So for now we just have explicit
#     annotations. When we lower, we can generate with_sharding_constraint. That
#     is, we'd like
#       shard_map(f, x)
#     but today we can only do
#       shard_map(f, x, sharding)
#  2. May be nice to do shard_map(f, x, sharding) or shard_map(f)(x, sharding)
#     rather than shard_map(f, sharding)(x) just to underscore that we should
#     think of 'sharding' as bound at the same time as 'x' (rather than 'x'
#     being bound later), per #1 above. On the other hand, it looks unusual...
#  3. We need names. Shardings don't have names. Maybe we want to pass in one
#     Mesh, and a PSpec for each argument? At least for staged version.
#  4. That API looks similar to old pjit, except: (a) mesh is explicit, (b) we
#     don't need output sharding annotations (trivially inferrable b/c map).
#  5. Need either annotations for outputs, or compute it ourselves by some
#     policy (per primitive) and return to user.  # TODO justify why
#  6. TODO Handle partial manual control. How to express "don't care, spmd
#     partitioner can handle this axis" vs "do not handle this axis, but I want
#     it replicated"?
#  7. TODO actually we do need transfer rules for each primitive, for how names
#     transfer through, so that we can support eager. Analogy to vmap.
#  8. Maybe we can have: staged out jaxpr form has explicit output annotations
#     (eg for transposition), but trace-time form can infer output (needed for
#     eager).


class ShardMapPrimitive(core.Primitive):
  multiple_results = True

  def bind(self, fun, *args, mesh, in_pspecs, out_pspecs_thunk):
    top_trace = core.find_top_trace(args)
    fun, env_trace_todo = process_env_traces(
        fun, top_trace and top_trace.level, mesh)
    tracers = map(top_trace.full_raise, args)
    outs = top_trace.process_shard_map(  # pytype: disable=attribute-error
        fun, tracers, mesh=mesh, in_pspecs=in_pspecs,
        out_pspecs_thunk=out_pspecs_thunk)
    return map(core.full_lower, core.apply_todos(env_trace_todo(), outs))

  def get_bind_params(self, params):
    raise NotImplementedError


shard_map_p = ShardMapPrimitive('shard_map')

def process_env_traces(fun, top_trace, mesh):
  return fun, lambda: []  # TODO

def _shard_map_impl(trace, fun, args, *, mesh, in_pspecs, out_pspecs_thunk):
  # TODO check pspecs are consistent with args shardings, by comparing
  # OpShardings
  raise NotImplementedError
core.EvalTrace.process_shard_map = _shard_map_impl

def pspec_to_in_shape(mesh: maps.Mesh, pspec, dim):
  if pspec is None:
    return dim
  elif isinstance(pspec, str):
    return dim // mesh.shape[pspec]
  else:
    return dim // math.prod([mesh.shape[p] for p in pspec])

def pspec_to_out_shape(mesh, pspec, dim):
  if pspec is None:
    return dim
  elif isinstance(pspec, str):
    return dim * mesh.shape[pspec]
  else:
    return dim * math.prod([mesh.shape[p] for p in pspec])

def _shard_map_staging(trace, fun, in_tracers, *, mesh, in_pspecs, out_pspecs_thunk):
  in_avals = [x.aval.update(shape=tuple(pspec_to_in_shape(mesh, n, d)
                                        for d, n in zip(x.shape, p)))
              for x, p in zip(in_tracers, in_pspecs)]
  with core.new_sublevel(), core.extend_axis_env_nd(mesh.shape.items()):
    jaxpr, out_avals_, consts = pe.trace_to_subjaxpr_dynamic(
        fun, trace.main, in_avals)
  out_pspecs = [canonicalize_pspec(x.aval.shape, p)
                for x, p in zip(jaxpr.outvars, out_pspecs_thunk())]
  out_avals = [x.aval.update(shape=tuple(pspec_to_out_shape(mesh, n, d)
                                         for d, n in zip(x.aval.shape, p)))
               for x, p in zip(jaxpr.outvars, out_pspecs)]
  source_info = source_info_util.current()
  out_tracers = [pe.DynamicJaxprTracer(trace, a, source_info)
                 for a in out_avals]
  invars = map(trace.getvar, in_tracers)
  constvars = map(trace.getvar, map(trace.instantiate_const, consts))
  outvars = map(trace.makevar, out_tracers)
  in_pspecs = [(None,)] * len(jaxpr.constvars) + in_pspecs
  params = dict(mesh=mesh, in_pspecs=in_pspecs, out_pspecs=out_pspecs,
                jaxpr=pe.convert_constvars_jaxpr(jaxpr))
  eqn = pe.new_jaxpr_eqn([*constvars, *invars], outvars, shard_map_p,
                         params, jaxpr.effects, source_info)
  trace.frame.add_eqn(eqn)
  return out_tracers
pe.DynamicJaxprTrace.process_shard_map = _shard_map_staging

def _shard_map_lowering(ctx, *in_nodes, jaxpr, mesh, in_pspecs, out_pspecs):
  in_avals_ = [v.aval for v in jaxpr.invars]
  in_nodes_ = map(partial(_shard, mesh), in_pspecs, ctx.avals_in, in_avals_,
                  in_nodes)
  new_axis_context = mlir.SPMDAxisContext(mesh, frozenset(mesh.axis_names))
  sub_ctx = ctx.module_context.replace(
      # TODO(sharadmv): name stack
      axis_context=new_axis_context)
  with core.extend_axis_env_nd(tuple(mesh.shape.items())):
    out_nodes_, _ = mlir.jaxpr_subcomp(sub_ctx, jaxpr, mlir.TokenSet(),
                                       (), *in_nodes_,
                                       dim_var_values=ctx.dim_var_values)
  out_avals_ = [v.aval for v in jaxpr.outvars]
  out_nodes = map(partial(_unshard, mesh), out_pspecs, out_avals_,
                  ctx.avals_out, out_nodes_)
  return out_nodes
mlir.register_lowering(shard_map_p, _shard_map_lowering)

def _shard(mesh, pspec, aval_in, aval_out, x):
  manual_proto = pxla._manual_proto(aval_in, frozenset(mesh.axis_names), mesh)
  result_type, = mlir.aval_to_ir_types(aval_out)
  array_mapping = pxla._get_array_mapping(pjit.PartitionSpec(*pspec))
  sharding_proto = pxla.mesh_sharding_specs(mesh.shape, mesh.axis_names)(
      aval_in, array_mapping).sharding_proto()
  sx = mlir.wrap_with_sharding_op(x, sharding_proto, unspecified_dims=set())
  return [mlir.wrap_with_full_to_shard_op(result_type, sx, manual_proto, set())]

def _unshard(mesh, pspec, aval_in, aval_out, xs):
  x, = xs
  manual_proto = pxla._manual_proto(aval_in, frozenset(mesh.axis_names), mesh)
  result_type, = mlir.aval_to_ir_types(aval_out)
  sx = mlir.wrap_with_sharding_op(x, manual_proto, unspecified_dims=set())
  array_mapping = pxla._get_array_mapping(pjit.PartitionSpec(*pspec))
  sharding_proto = pxla.mesh_sharding_specs(mesh.shape, mesh.axis_names)(
      aval_out, array_mapping).sharding_proto()
  return mlir.wrap_with_shard_to_full_op(result_type, sx, sharding_proto, set())

class ShardMapTrace(core.Trace):
  def __init__(self, *args, mesh):
    super().__init__(*args)
    self.mesh = mesh

  def pure(self, val):
    return ShardMapTracer(self, val, (None,) * np.ndim(val))

  def sublift(self, tracer):
    return ShardMapTracer(self, tracer.val, tracer.pspec)

  def process_primitive(self, primitive, tracers, params):
    # TODO execute_sharded_on_local_devices(...)
    raise NotImplementedError("Eager shard map not supported.")

class ShardMapTracer(core.Tracer):
  def __init__(self, trace, val, pspec):
    self._trace = trace
    self.val = val
    self.pspec = pspec

  @property
  def aval(self):
    aval = core.raise_to_shaped(core.get_aval(self.val))
    mesh_shape = self._trace.mesh.shape
    new_shape = [d if axis_name is None else d // mesh_shape[axis_name]
                 for d, axis_name in zip(aval.shape, self.pspec)]
    return aval.update(shape=tuple(new_shape))

  def full_lower(self):
    return self

def canonicalize_pspec(shape: Tuple[int, ...], p: pjit.PartitionSpec
                       ) -> Tuple[Optional[core.AxisName], ...]:
  assert len(p) == len(shape)
  return tuple(p)

def shard_map(f: Callable, mesh: maps.Mesh, in_pspecs, out_pspecs):
  def wrapped(*args):
    fun = lu.wrap_init(f)
    with mesh:
      args = tree_map(pjit.with_sharding_constraint, args, in_pspecs)
    args_flat, in_tree = tree_flatten(args)
    in_pspecs_flat, in_tree_ = tree_flatten(in_pspecs)
    in_pspecs_flat = [canonicalize_pspec(x.shape, p)
                      for x, p in zip(args_flat, in_pspecs_flat)]
    assert in_tree == in_tree_
    flat_fun, out_tree = flatten_fun_nokwargs(fun, in_tree)
    def out_pspecs_thunk():
      out_pspecs_flat, out_tree_ = tree_flatten(out_pspecs)
      assert out_tree() == out_tree_
      return out_pspecs_flat
    out_flat = shard_map_p.bind(
        flat_fun, *args_flat, mesh=mesh, in_pspecs=in_pspecs_flat,
        out_pspecs_thunk=out_pspecs_thunk)
    return tree_unflatten(out_tree(), out_flat)
  return wrapped
