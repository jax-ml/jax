# Copyright 2020 Google LLC
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

import enum
import threading
import contextlib
from collections import namedtuple
from typing import Callable, Iterable, List, Tuple, Optional, Dict, Any
from warnings import warn
from functools import wraps, partial

import jax
from .. import numpy as jnp
from .. import core
from .. import linear_util as lu
from ..api import _mapped_axis_size, _check_callable, _check_arg
from ..tree_util import tree_flatten, tree_unflatten
from ..api_util import flatten_fun
from ..interpreters import partial_eval as pe
from ..interpreters import batching
from ..util import safe_map, safe_zip, curry

map = safe_map
zip = safe_zip

# Multi-dimensional generalized map

# TODO: Use a more lax type annotation (we need == and hash)
AxisName = str
ResourceAxisName = str

# TODO: At least support sequential mapping
class ResourceEnv(threading.local):
  def __init__(self):
    self.axes : Dict[ResourceAxisName, int] = {}
thread_resource_env = ResourceEnv()

@contextlib.contextmanager
def resources(**axes):
  old_axes = thread_resource_env.axes
  thread_resource_env.axes = axes
  try:
    yield
  finally:
    thread_resource_env.axes = old_axes

# This is really a Dict[AxisName, int], but we don't define a
# pytree instance for it, so that it is treated as a leaf.
class AxisNamePos(dict):
  pass

A = AxisNamePos

# TODO: Some syntactic sugar to make the API more usable in a single-axis case?
# TODO: Are the resource axes scoped lexically or dynamically? Dynamically for now!
def xmap(fun: Callable,
         in_axes,  # PyTree[AxisNamePos]
         out_axes,  # PyTree[AxisNamePos],
         schedule: Iterable[Tuple[AxisName, ResourceAxisName]]):
  warn("xmap is an experimental feature and probably has bugs!")
  _check_callable(fun)

  def fun_mapped(*args, **kwargs):
    args_flat, in_tree = tree_flatten((args, kwargs))
    for arg in args_flat: _check_arg(arg)
    # TODO: Check that:
    #         - every scheduled axis name appears in at least one input
    #         - every used resource axis name appears in the resource env
    #         - every axis name is scheduled to a single resource axis only once
    #         - every out axis has a distinct index
    #         - two axes mapped to the same resource never coincide (even inside f)
    in_axes_flat, in_axes_tree = tree_flatten(in_axes)
    # TODO: Verify that in_axes are equal, or better expand their prefix
    # assert in_axes_tree == in_tree
    out_axes_flat, out_axes_tree = tree_flatten(out_axes)
    # TODO: Verify that out_axes are equal, or better expand their prefix
    # assert out_axes_tree == in_tree

    resource_to_axis: Dict[ResourceAxisName, List[AxisName]] = dict()
    for (axis, resource) in schedule:
      if resource not in resource_to_axis:
        resource_to_axis[resource] = []
      resource_to_axis[resource].append(axis)

    # TODO: The order of maps should be derived from the schedule, not from the
    #       resource env. This doesn't really matter for as long as we only support
    #       vmap, but will be important (e.g. for tiling).
    #       We should be able to do that by building a graph of dependencies between
    #       resources based on the order in which they appear within each axis.
    #       If it has cycles then we cannot realize it. Otherwise, if the DAG doesn't
    #       uniquely identify a linear order, we should use the order of entries in
    #       the schedule to break ties.
    resource_map = {resource: (pri, size)
                    for pri, (resource, size) in enumerate(thread_resource_env.axes.items())}
    resource_map['vectorize'] = (len(resource_map), None)
    map_sequence = sorted(resource_to_axis.items(),
                          key=lambda item: resource_map[item[0]][0])
    axis_subst = {}
    for axis, resource in schedule:
      if axis not in axis_subst:
        axis_subst[axis] = []
      if resource == 'vectorize':
        resource = f'v_{axis}'
      else:
        resource = f'r_{resource}'
      axis_subst[axis].append(resource)
    axis_subst = {axis: tuple(resources) for axis, resources in axis_subst.items()}

    axis_sizes = _get_axis_sizes(args_flat, in_axes_flat)
    jaxpr, out_tree = _trace_mapped_jaxpr(fun, args_flat, in_axes_flat, axis_sizes, in_tree)
    jaxpr = jaxpr.map_jaxpr(partial(subst_axis_names, axis_subst=axis_subst))
    f = lu.wrap_init(core.jaxpr_as_fun(jaxpr))
    f = hide_mapped_axes(f, in_axes_flat, out_axes_flat)
    for resource, resource_axes in map_sequence[::-1]:
      # TODO: Support sequential
      # XXX: Even though multiple axes might be mapped to the 'vectorized'
      #      resource, we cannot vectorize them jointly, because they
      #      might require different axis sizes.
      if resource == 'vectorize':
        maps = [(f'v_{name}', [name]) for i, name in enumerate(resource_axes)]
      else:
        maps = [(f'r_{resource}', resource_axes)]
      for raxis_name, axes in maps:
        map_in_axes = map(lambda spec: lookup_exactly_one_of(spec, axes), in_axes_flat)
        map_out_axes = map(lambda spec: lookup_exactly_one_of(spec, axes), out_axes_flat)
        map_size = resource_map[resource][1]
        f = vtile(f, map_in_axes, map_out_axes, tile_size=map_size, axis_name=raxis_name)
    flat_out = f.call_wrapped(*args_flat)
    return tree_unflatten(out_tree, flat_out)

  return fun_mapped

def _delete_aval_axes(aval, axes: AxisNamePos):
  assert isinstance(aval, core.ShapedArray)
  shape = list(aval.shape)
  for i in sorted(axes.values(), reverse=True):
    del shape[i]
  return core.ShapedArray(tuple(shape), aval.dtype)

def _with_axes(axes: Iterable[Tuple[AxisName, int]], f):
  for name, size in axes:
    f = core.extend_axis_env(name, size, None)(f)
  return f()

def _trace_mapped_jaxpr(fun,
                        args_flat,
                        in_axes_flat: List[AxisNamePos],
                        axis_sizes: Dict[AxisName, int],
                        in_tree):
  fun_flat, out_tree = flatten_fun(lu.wrap_init(fun), in_tree)
  avals_flat = [core.raise_to_shaped(core.get_aval(arg)) for arg in args_flat]
  mapped_pvals = [pe.PartialVal.unknown(_delete_aval_axes(aval, in_axes))
                  for aval, in_axes in zip(avals_flat, in_axes_flat)]
  jaxpr, _, consts = _with_axes(axis_sizes.items(),
                                lambda: pe.trace_to_jaxpr(fun_flat, mapped_pvals))
  return core.ClosedJaxpr(jaxpr, consts), out_tree()

def _get_axis_sizes(args_flat: Iterable[Any], in_axes_flat: Iterable[AxisNamePos]):
  axis_sizes: Dict[AxisName, int] = {}
  for arg, in_axes in zip(args_flat, in_axes_flat):
    for name, dim in in_axes.items():
      if name in axis_sizes:
        assert axis_sizes[name] == arg.shape[dim]
      else:
        axis_sizes[name] = arg.shape[dim]
  return axis_sizes


def lookup_exactly_one_of(d: AxisNamePos, names: List[AxisName]) -> Optional[int]:
  res = None
  for name in names:
    if name in d:
      if res is not None:
        raise ValueError("An input was mapped to the same resource twice")
      res = d[name]
  return res

def _squeeze_mapped_axes(arg, axes: AxisNamePos):
  for dim in sorted(axes.values(), reverse=True):
    arg = arg.squeeze(dim)
  return arg

def _unsqueeze_mapped_axes(out, axes: AxisNamePos):
  for dim in sorted(axes.values()):
    out = jnp.expand_dims(out, dim)
  return out

@lu.transformation
def hide_mapped_axes(flat_in_axes, flat_out_axes, *flat_args):
  squeezed_args = map(_squeeze_mapped_axes, flat_args, flat_in_axes)
  flat_outputs = yield squeezed_args, {}
  yield map(_unsqueeze_mapped_axes, flat_outputs, flat_out_axes)

@curry
def tile_axis(arg, axis: Optional[int], tile_size):
  if axis is None:
    return arg
  shape = list(arg.shape)
  shape[axis:axis+1] = [tile_size, shape[axis] // tile_size]
  return arg.reshape(shape)

def untile_axis(out, axis: Optional[int]):
  if axis is None:
    return out
  shape = list(out.shape)
  shape[axis:axis+2] = [shape[axis] * shape[axis+1]]
  return out.reshape(shape)

# NOTE: This divides the in_axes by the tile_size and multiplies the out_axes by it.
def vtile(f_flat, in_axes_flat, out_axes_flat, tile_size: Optional[int], axis_name):
  @lu.transformation
  def _map_to_tile(*args_flat):
    real_tile_size = tile_size
    for arg, in_axis in zip(args_flat, in_axes_flat):
      if real_tile_size is not None:
        break
      if in_axis is None:
        continue
      real_tile_size = arg.shape[in_axis]
    assert real_tile_size is not None, "No mapped arguments?"
    outputs_flat = yield map(tile_axis(tile_size=real_tile_size), args_flat, in_axes_flat), {}
    yield map(untile_axis, outputs_flat, out_axes_flat)

  return _map_to_tile(
    batching.batch_fun(f_flat,
                       in_axes_flat,
                       out_axes_flat,
                       axis_name=axis_name))

# Single-dimensional generalized map

def gmap(fun: Callable, schedule, axis_name = None) -> Callable:
  warn("gmap is an experimental feature and probably has bugs!")
  _check_callable(fun)
  binds_axis_name = axis_name is not None
  axis_name = core._TempAxisName(fun) if axis_name is None else axis_name

  @wraps(fun)
  def f_gmapped(*args, **kwargs):
    f = lu.wrap_init(fun)
    args_flat, in_tree = tree_flatten((args, kwargs))
    in_axes = (0,) * len(args_flat)
    axis_size = _mapped_axis_size(in_tree, args_flat, (0,) * len(args_flat), "gmap")
    parsed_schedule = _normalize_schedule(schedule, axis_size, binds_axis_name)
    for arg in args_flat: _check_arg(arg)
    flat_fun, out_tree = flatten_fun(f, in_tree)
    outs = gmap_p.bind(
        flat_fun, *args_flat,
        axis_name=axis_name,
        axis_size=axis_size,
        in_axes=in_axes,
        schedule=parsed_schedule,
        binds_axis_name=binds_axis_name)
    return tree_unflatten(out_tree(), outs)
  return f_gmapped


class LoopType(enum.Enum):
    vectorized = enum.auto()
    parallel = enum.auto()
    sequential = enum.auto()

Loop = namedtuple('Loop', ['type', 'size'])


def _normalize_schedule(schedule, axis_size, binds_axis_name):
  if not schedule:
    raise ValueError("gmap expects a non-empty schedule")

  scheduled = 1
  seen_none = False
  for loop in schedule:
    if loop[1] is not None:
      scheduled *= loop[1]
    elif seen_none:
      raise ValueError("gmap schedule can only contain at most a single None size specification")
    else:
      seen_none = True
  unscheduled = axis_size // scheduled

  new_schedule = []
  for i, loop in enumerate(schedule):
    loop_type = _parse_name(loop[0])
    if loop_type is LoopType.vectorized and i < len(schedule) - 1:
      raise ValueError("vectorized loops can only appear as the last component of the schedule")
    if loop_type is LoopType.sequential and binds_axis_name:
      raise ValueError("gmaps that bind a new axis name cannot have sequential components in the schedule")
    new_schedule.append(Loop(loop_type, loop[1] or unscheduled))
  return tuple(new_schedule)

def _parse_name(name):
  if isinstance(name, LoopType):
    return name
  try:
    return LoopType[name]
  except KeyError as err:
    raise ValueError(f"Unrecognized loop type: {name}") from err


def gmap_impl(fun: lu.WrappedFun, *args, axis_size, axis_name, binds_axis_name, in_axes, schedule):
  avals = [core.raise_to_shaped(core.get_aval(arg)) for arg in args]
  scheduled_fun = _apply_schedule(fun, axis_size, axis_name, binds_axis_name,
                                  in_axes, schedule, *avals)
  return scheduled_fun(*args)

class _GMapSubaxis:
  def __init__(self, axis_name, index):
    self.axis_name = axis_name
    self.index = index
  def __repr__(self):
    return f'<subaxis {self.index} of {self.axis_name}>'
  def __hash__(self):
    return hash((self.axis_name, self.index))
  def __eq__(self, other):
    return (isinstance(other, _GMapSubaxis) and
            self.axis_name == other.axis_name and
            self.index == other.index)

@lu.cache
def _apply_schedule(fun: lu.WrappedFun,
                    axis_size, full_axis_name, binds_axis_name,
                    in_axes,
                    schedule,
                    *avals):
  mapped_avals = [core.mapped_aval(axis_size, in_axis, aval)
                  if in_axis is not None else aval
                  for aval, in_axis in zip(avals, in_axes)]
  with core.extend_axis_env(full_axis_name, axis_size, None):
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, mapped_avals)

  axis_names = tuple(_GMapSubaxis(full_axis_name, i) for i in range(len(schedule)))
  if binds_axis_name:
    jaxpr = subst_axis_names(jaxpr, {full_axis_name: (axis_names,)})  # type: ignore

  sched_fun = lambda *args: core.eval_jaxpr(jaxpr, consts, *args)
  for (ltype, size), axis_name in list(zip(schedule, axis_names))[::-1]:
    if ltype is LoopType.vectorized:
      sched_fun = jax.vmap(sched_fun, axis_name=axis_name, in_axes=in_axes)
    elif ltype is LoopType.parallel:
      sched_fun = jax.pmap(sched_fun, axis_name=axis_name, in_axes=in_axes)
    elif ltype is LoopType.sequential:
      if binds_axis_name:
        raise NotImplementedError("gmaps with sequential components of the schedule don't support "
                                  "collectives yet. Please open a feature request!")
      assert not binds_axis_name
      sched_fun = lambda *args, sched_fun=sched_fun: _sequential_map(sched_fun, in_axes, args)

  dim_sizes = tuple(loop.size for loop in schedule)
  def sched_fun_wrapper(*args):
    split_args = [arg.reshape(arg.shape[:in_axis] + dim_sizes + arg.shape[in_axis + 1:])
                  for arg, in_axis in zip(args, in_axes)]
    results = sched_fun(*split_args)
    return [res.reshape((axis_size,) + res.shape[len(dim_sizes):]) for res in results]
  return sched_fun_wrapper

gmap_p = core.MapPrimitive('gmap')
gmap_p.def_impl(gmap_impl)

def _sequential_map(f, flat_in_axes, args):
  flat_args, treedef = tree_flatten(args)
  flat_args_leading = [jnp.moveaxis(arg, in_axis, 0)
                       for arg, in_axis in zip(flat_args, flat_in_axes)]
  args_leading = tree_unflatten(treedef, flat_args_leading)
  return jax.lax.map(lambda xs: f(*xs), args_leading)

def subst_axis_names(jaxpr, axis_subst: Dict[AxisName, Tuple[AxisName]]):
  eqns = [subst_eqn_axis_names(eqn, axis_subst) for eqn in jaxpr.eqns]
  return core.Jaxpr(jaxpr.constvars, jaxpr.invars, jaxpr.outvars, eqns)

def subst_eqn_axis_names(eqn, axis_subst: Dict[AxisName, Tuple[AxisName]]):
  # TODO: Support custom_vjp, custom_jvp
  if isinstance(eqn.primitive, (core.CallPrimitive, core.MapPrimitive)):
    bound_name = eqn.params.get('axis_name', None)
    if bound_name in axis_subst:  # Check for shadowing
      sub_subst = dict(axis_subst)
      del sub_subst[bound_name]
    else:
      sub_subst = axis_subst
    new_call_jaxpr = subst_axis_names(eqn.params['call_jaxpr'], sub_subst)
    return eqn._replace(params=dict(eqn.params, call_jaxpr=new_call_jaxpr))
  if 'axis_name' not in eqn.params:
    return eqn
  axis_names = eqn.params['axis_name']
  if not isinstance(axis_names, (tuple, list)):
    axis_names = (axis_names,)
  new_axis_names = sum((axis_subst.get(name, (name,)) for name in axis_names), ())
  if len(new_axis_names) == 1:
    new_axis_names = new_axis_names[0]  # type: ignore
  return eqn._replace(params=dict(eqn.params, axis_name=new_axis_names))
