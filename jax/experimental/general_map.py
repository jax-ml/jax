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
import numpy as np
from collections import namedtuple, OrderedDict
from typing import Callable, Iterable, List, Tuple, Optional, Dict, Any, Set
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
from ..interpreters import pxla
from ..util import safe_map, safe_zip, curry

map = safe_map
zip = safe_zip

# Multi-dimensional generalized map

# TODO: Use a more concrete type annotation (we need __eq__ and __hash__)
AxisName = Any
ResourceAxisName = Any

Mesh = pxla.Mesh

# TODO: Support sequential mapping
class ResourceEnv(threading.local):
  def __init__(self):
    self.physical_mesh : Mesh = Mesh(np.empty((), dtype=object), ())
    self.fake_resources : Dict[ResourceAxisName, int] = {}

  @property
  def physical_resource_axes(self) -> Set[ResourceAxisName]:
    return set(self.physical_mesh.axis_names)

  @property
  def fake_resource_axes(self) -> Set[ResourceAxisName]:
    return set(self.fake_resources.keys())

  @property
  def resource_axes(self) -> Set[ResourceAxisName]:
    return self.physical_resource_axes | self.fake_resource_axes

  @property
  def shape(self):
    shape = OrderedDict(self.physical_mesh.shape)
    shape.update((name, size) for name, size in self.fake_resources.items())
    return shape

# TODO: Make this thread local
thread_resource_env = ResourceEnv()

@contextlib.contextmanager
def fake_resources(**axes):
  old_axes = thread_resource_env.fake_resources
  thread_resource_env.fake_resources = axes
  try:
    yield
  finally:
    thread_resource_env.axes = old_axes

@contextlib.contextmanager
def mesh(*args, **kwargs):
  old = thread_resource_env.physical_mesh
  thread_resource_env.physical_mesh = Mesh(*args, **kwargs)
  try:
    yield
  finally:
      thread_resource_env.physical_mesh = old

_next_resource_id = 0
class UniqueResourceName:
  def __init__(self, uid): self.uid = uid
  def __eq__(self, other): return type(other) is UniqueResourceName and self.uid == other.uid
  def __hash__(self): return hash(self.uid)
def fresh_resource_name():
  global _next_resource_id
  try:
    return UniqueResourceName(_next_resource_id)
  finally:
    _next_resource_id += 1

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
    # Putting this outside of fun_mapped would make resources lexically scoped
    resource_env = thread_resource_env

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
    # assert out_axes_tree == out_tree

    axis_sizes = _get_axis_sizes(args_flat, in_axes_flat)
    jaxpr, out_tree = _trace_mapped_jaxpr(fun, args_flat, in_axes_flat, axis_sizes, in_tree)

    # TODO: The order of maps should be derived from the schedule, not from the
    #       resource env. This doesn't really matter for as long as we only support
    #       vectorization and parallelization, but will be important for sequential.
    #       We should be able to do that by building a graph of dependencies between
    #       resources based on the order in which they appear within each axis.
    #       If it has cycles then we cannot realize it. Otherwise, if the DAG doesn't
    #       uniquely identify a linear order, we should use the order of entries in
    #       the schedule to break ties.
    # Technically the order doesn't matter right now, but we use the ordered dict
    # to at least limit the amount of non-determinism in this code.
    fake_resource_map: Dict[ResourceAxisName, Set[AxisName]] = OrderedDict()
    physical_resource_map: Dict[ResourceAxisName, Set[AxisName]] = OrderedDict()
    vectorized: Dict[AxisName, ResourceAxisName] = OrderedDict()

    axis_subst: Dict[AxisName, List[ResourceAxisName]] = {}
    for axis, resource in schedule:
      if resource == 'vectorize':
        assert axis not in vectorized
        resource = fresh_resource_name()
        vectorized[axis] = resource
      elif resource in resource_env.physical_resource_axes:
        # TODO: Make sure that axis was not in the set?
        physical_resource_map.setdefault(resource, set()).add(axis)
      elif resource in resource_env.fake_resource_axes:
        # TODO: Make sure that axis was not in the set?
        fake_resource_map.setdefault(resource, set()).add(axis)
      else:
        raise ValueError(f"Mapping axis {axis} to an undefined resource axis {resource}. "
                         f"The resource axes currently in scope are: {resource_env.resource_axes}")
      axis_subst.setdefault(axis, []).append(resource)

    axis_subst_t = {axis: tuple(resources) for axis, resources in axis_subst.items()}
    jaxpr = jaxpr.map_jaxpr(partial(subst_axis_names, axis_subst=axis_subst_t))

    f = lu.wrap_init(core.jaxpr_as_fun(jaxpr))
    f = hide_mapped_axes(f, in_axes_flat, out_axes_flat)

    for naxis, raxis in vectorized.items():
      map_in_axes = map(lambda spec: spec.get(naxis, None), in_axes_flat)
      map_out_axes = map(lambda spec: spec.get(naxis, None), out_axes_flat)
      f = vtile(f, map_in_axes, map_out_axes, tile_size=None, axis_name=raxis)

    resource_env_shape = resource_env.shape
    for raxis, naxes in fake_resource_map.items():
      map_in_axes = map(lambda spec: lookup_exactly_one_of(spec, naxes), in_axes_flat)
      map_out_axes = map(lambda spec: lookup_exactly_one_of(spec, naxes), out_axes_flat)
      map_size = resource_env_shape[raxis]
      f = vtile(f, map_in_axes, map_out_axes, tile_size=map_size, axis_name=raxis)

    if physical_resource_map:
      submesh = resource_env.physical_mesh[tuple(physical_resource_map.keys())]
      in_avals = [core.raise_to_shaped(core.get_aval(arg)) for arg in args_flat]
      def to_mesh_axes(axes):
        mesh_axes = {}
        for paxis, naxes in physical_resource_map.items():
          axis = lookup_exactly_one_of(axes, naxes)
          if axis is None:
              continue
          mesh_axes[paxis] = axis
        return mesh_axes
      mesh_in_axes = map(to_mesh_axes, in_axes)
      mesh_out_axes = map(to_mesh_axes, out_axes)
      f = pxla.mesh_tiled_callable(*in_avals, fun=f,
                                   transformed_name=f.__name__,
                                   backend_name=None,
                                   mesh=submesh,
                                   in_axes=mesh_in_axes,
                                   out_axes_thunk=lambda: mesh_out_axes)
      flat_out = f(*args_flat)
    else:
      flat_out = f.call_wrapped(*args_flat)

    return tree_unflatten(out_tree, flat_out)

  return fun_mapped

def _delete_aval_axes(aval, axes: AxisNamePos):
  assert isinstance(aval, core.ShapedArray)
  shape = list(aval.shape)
  for i in sorted(axes.values(), reverse=True):
    del shape[i]
  return core.ShapedArray(tuple(shape), aval.dtype)

def _trace_mapped_jaxpr(fun,
                        args_flat,
                        in_axes_flat: List[AxisNamePos],
                        axis_sizes: Dict[AxisName, int],
                        in_tree):
  fun_flat, out_tree = flatten_fun(lu.wrap_init(fun), in_tree)
  avals_flat = [core.raise_to_shaped(core.get_aval(arg)) for arg in args_flat]
  mapped_avals = [_delete_aval_axes(aval, in_axes)
                  for aval, in_axes in zip(avals_flat, in_axes_flat)]
  with core.extend_axis_env_nd(axis_sizes.items()):
    jaxpr, _, consts = pe.trace_to_jaxpr_final(fun_flat, mapped_avals)
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

def lookup_exactly_one_of(d: AxisNamePos, names: Set[AxisName]) -> Optional[int]:
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
        out_axes_thunk=lambda: (0,) * out_tree().num_leaves,
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


def gmap_impl(fun: lu.WrappedFun, *args, axis_size, axis_name,
              binds_axis_name, in_axes, out_axes_thunk, schedule):
  avals = [core.raise_to_shaped(core.get_aval(arg)) for arg in args]
  scheduled_fun = _apply_schedule(fun, axis_size, axis_name, binds_axis_name,
                                  in_axes, out_axes_thunk, schedule, *avals)
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
                    out_axes_thunk,
                    schedule,
                    *avals):
  mapped_avals = [core.mapped_aval(axis_size, in_axis, aval)
                  if in_axis is not None else aval
                  for aval, in_axis in zip(avals, in_axes)]
  with core.extend_axis_env(full_axis_name, axis_size, None):
    jaxpr, _, consts = pe.trace_to_jaxpr_final(fun, mapped_avals)
  out_axes = out_axes_thunk()
  assert all(out_axis == 0 for out_axis in out_axes)

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
