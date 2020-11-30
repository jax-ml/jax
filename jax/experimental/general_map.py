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
import itertools as it
from collections import namedtuple, OrderedDict
from typing import (Callable, Iterable, List, Tuple, Optional, Dict, Any, Set,
                    NamedTuple)
from warnings import warn
from functools import wraps, partial

import jax
from .. import numpy as jnp
from .. import core
from .. import linear_util as lu
from ..api import _mapped_axis_size, _check_callable, _check_arg
from ..tree_util import tree_flatten, tree_unflatten, tree_leaves
from ..api_util import flatten_fun, flatten_fun_nokwargs, flatten_axes
from ..interpreters import partial_eval as pe
from ..interpreters import batching
from ..interpreters import pxla
from ..interpreters import xla
from ..lib import xla_bridge as xb
from ..lib import xla_client as xc
from ..util import safe_map, safe_zip, curry, HashableFunction
from .._src.lax.parallel import _axis_index_translation_rule

map, unsafe_map = safe_map, map
zip = safe_zip

xops = xc.ops

class FrozenDict:  # dataclasses might remove some boilerplate here
  def __init__(self, *args, **kwargs):
    self.contents = dict(*args, **kwargs)

  allowed_methods = {'items', 'values', 'keys', 'get'}
  def __getattr__(self, name):
    if name in self.allowed_methods:
      return getattr(self.contents, name)
    raise AttributeError(name)

  def __iter__(self):
    return self.contents.__iter__()

  def __len__(self):
    return self.contents.__len__()

  def __getitem__(self, name):
    return self.contents.__getitem__(name)

  def __eq__(self, other):
    return isinstance(other, FrozenDict) and self.contents == other.contents

  def __hash__(self):
    return hash(tuple(self.contents.items()))

# Multi-dimensional generalized map

AxisName = core.AxisName
ResourceAxisName = AxisName  # Different name just for documentation purposes
Mesh = pxla.Mesh

# TODO: Support sequential mapping
class ResourceEnv:
  __slots__ = ('physical_mesh', 'fake_resources')
  physical_mesh: Mesh
  fake_resources: FrozenDict

  def __init__(self, physical_mesh: Mesh, fake_resources: FrozenDict):
    super().__setattr__('physical_mesh', physical_mesh)
    super().__setattr__('fake_resources', fake_resources)

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

  def __setattr__(self, name, value):
    raise RuntimeError("ResourceEnv is immutable!")

  def __delattr__(self):
    raise RuntimeError("ResourceEnv is immutable!")

  def __eq__(self, other):
    return (type(other) is ResourceEnv and
            self.physical_mesh == other.physical_mesh and
            self.fake_resources == other.fake_resources)

  def __hash__(self):
    return hash((self.physical_mesh, self.fake_resources))

thread_resources = threading.local()
thread_resources.env = ResourceEnv(Mesh(np.empty((), dtype=object), ()), FrozenDict())

@contextlib.contextmanager
def fake_resources(**axes):
  old_env = thread_resources.env
  thread_resources.env = ResourceEnv(old_env.physical_mesh, FrozenDict(axes))
  try:
    yield
  finally:
    thread_resources.env = old_env

@contextlib.contextmanager
def mesh(*args, **kwargs):
  old_env = thread_resources.env
  thread_resources.env = ResourceEnv(Mesh(*args, **kwargs), old_env.fake_resources)
  try:
    yield
  finally:
    thread_resources.env = old_env

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
class AxisNamePos(FrozenDict):
  pass

A = AxisNamePos


# TODO: Some syntactic sugar to make the API more usable in a single-axis case?
# TODO: Are the resource axes scoped lexically or dynamically? Dynamically for now!
def xmap(fun: Callable,
         in_axes,  # PyTree[AxisNamePos]
         out_axes,  # PyTree[AxisNamePos],
         schedule: Iterable[Tuple[AxisName, ResourceAxisName]],
         backend: Optional[str] = None):
  warn("xmap is an experimental feature and probably has bugs!")
  _check_callable(fun)

  frozen_schedule = tuple(tuple(x) for x in schedule)

  # To be a tree prefix of the positional args tuple, in_axes can never be a
  # list: if in_axes is not a leaf, it must be a tuple of trees. However,
  # in cases like these users expect tuples and lists to be treated
  # essentially interchangeably, so we canonicalize lists to tuples here
  # rather than raising an error. https://github.com/google/jax/issues/2367
  if isinstance(in_axes, list):
    in_axes = tuple(in_axes)
  if isinstance(out_axes, list):
    out_axes = tuple(out_axes)

  in_axes_entries = tree_leaves(in_axes)
  out_axes_entries = tree_leaves(out_axes)
  # Check that {in|out}_axes have the right types, and don't use the same positional axis twice
  if not all(isinstance(x, A) for x in in_axes_entries):
    raise TypeError(f"xmap in_axes must be AxisNamePos (A) instances or (nested) "
                    f"containers with those types as leaves, but got {in_axes}")
  if not all(isinstance(x, A) for x in out_axes_entries):
    raise TypeError(f"xmap out_axes must be AxisNamePos (A) instances or (nested) "
                    f"containers with those types as leaves, but got {in_axes}")
  for x in in_axes_entries:
    if len(set(x.values())) != len(x):
      raise ValueError(f"Positional dimension indices should be unique within each "
                       f"in_axes dictionary, but one of the entries is: {x}")
  for x in out_axes_entries:
    if len(set(x.values())) != len(x):
      raise ValueError(f"Positional dimension indices should be unique within each "
                       f"in_axes dictionary, but one of the entries is: {x}")

  in_axes_names = set(it.chain(*(spec.keys() for spec in in_axes_entries)))
  scheduled_axes = set(x[0] for x in frozen_schedule)
  if scheduled_axes != in_axes_names:
    raise ValueError("The set of axes names appearing in in_axes has to equal the "
                     "set of scheduled axes, but {in_axes_names} != {scheduled_axes}")

  necessary_resources = set(x[1] for x in frozen_schedule if x[1] != 'vectorize')
  if len(set(frozen_schedule)) != len(frozen_schedule):
    raise ValueError(f"xmap schedule contains duplicate entries: {frozen_schedule}")

  def fun_mapped(*args):
    # Putting this outside of fun_mapped would make resources lexically scoped
    resource_env = thread_resources.env
    available_resources = set(resource_env.shape.keys())

    if necessary_resources > available_resources:
      raise ValueError(f"In-scope resources are insufficient to execute the "
                       f"xmapped function. The missing resources are: "
                       f"{necessary_resources - available_resources}")

    args_flat, in_tree = tree_flatten(args)
    for arg in args_flat: _check_arg(arg)
    fun_flat, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
    # TODO: Check that:
    #         - two axes mapped to the same resource never coincide (even inside f)
    in_axes_flat = flatten_axes("xmap in_axes", in_tree, in_axes)
    out_axes_thunk = HashableFunction(
      lambda: tuple(flatten_axes("xmap out_axes", out_tree(), out_axes)),
      key=out_axes)
    axis_sizes = _get_axis_sizes(args_flat, in_axes_flat)
    out_flat = xmap_p.bind(
      fun_flat, *args_flat,
      name=fun.__name__,
      in_axes=tuple(in_axes_flat),
      out_axes_thunk=out_axes_thunk,
      axis_sizes=FrozenDict(axis_sizes),
      schedule=frozen_schedule,
      resource_env=resource_env,
      backend=backend)
    return tree_unflatten(out_tree(), out_flat)

  return fun_mapped

def xmap_impl(fun: lu.WrappedFun, *args, name, in_axes, out_axes_thunk, axis_sizes, schedule, resource_env, backend):
  in_avals = [core.raise_to_shaped(core.get_aval(arg)) for arg in args]
  return make_xmap_callable(fun, name, in_axes, out_axes_thunk, axis_sizes, schedule,
                            resource_env, backend, *in_avals)(*args)

@lu.cache
def make_xmap_callable(fun: lu.WrappedFun,
                       name,
                       in_axes, out_axes_thunk, axis_sizes,
                       schedule, resource_env, backend,
                       *in_avals):
  plan = EvaluationPlan.from_schedule(schedule, resource_env)

  # TODO: Making axis substitution final style would allow us to avoid
  #       tracing to jaxpr here
  mapped_in_avals = [_delete_aval_axes(aval, in_axes)
                     for aval, in_axes in zip(in_avals, in_axes)]
  with core.extend_axis_env_nd(axis_sizes.items()):
    jaxpr, _, consts = pe.trace_to_jaxpr_final(fun, mapped_in_avals)
  out_axes = out_axes_thunk()
  jaxpr = subst_axis_names(jaxpr, plan.axis_subst)

  f = lu.wrap_init(core.jaxpr_as_fun(core.ClosedJaxpr(jaxpr, consts)))
  f = hide_mapped_axes(f, tuple(in_axes), tuple(out_axes))
  f = plan.vectorize(f, in_axes, out_axes, resource_env)

  used_resources = _jaxpr_resources(jaxpr, resource_env) | _schedule_resources(schedule)
  used_mesh_axes = used_resources & resource_env.physical_resource_axes
  if used_mesh_axes:
    submesh = resource_env.physical_mesh[sorted(used_mesh_axes, key=str)]
    mesh_in_axes, mesh_out_axes = plan.to_mesh_axes(in_axes, out_axes)
    return pxla.mesh_tiled_callable(f,
                                    name,
                                    backend,
                                    submesh,
                                    mesh_in_axes,
                                    mesh_out_axes,
                                    *in_avals)
  else:
    return f.call_wrapped

class EvaluationPlan(NamedTuple):
  """Encapsulates preprocessing common to top-level xmap invocations and its translation rule."""
  fake_resource_map: Dict[ResourceAxisName, Set[AxisName]]
  physical_resource_map: Dict[ResourceAxisName, Set[AxisName]]
  vectorized: Dict[AxisName, ResourceAxisName]
  axis_subst: Dict[AxisName, Tuple[ResourceAxisName, ...]]

  @classmethod
  def from_schedule(cls, schedule: Tuple[Tuple[AxisName, ResourceAxisName], ...], resource_env):
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
        physical_resource_map.setdefault(resource, set()).add(axis)
      elif resource in resource_env.fake_resource_axes:
        fake_resource_map.setdefault(resource, set()).add(axis)
      else:
        raise ValueError(f"Mapping axis {axis} to an undefined resource axis {resource}. "
                            f"The resource axes currently in scope are: {resource_env.resource_axes}")
      axis_subst.setdefault(axis, []).append(resource)
    axis_subst_t = {name: tuple(axes) for name, axes in axis_subst.items()}
    return cls(fake_resource_map, physical_resource_map, vectorized, axis_subst_t)

  def vectorize(self, f: lu.WrappedFun, in_axes, out_axes, resource_env):
    for naxis, raxis in self.vectorized.items():
      map_in_axes = tuple(unsafe_map(lambda spec: spec.get(naxis, None), in_axes))
      map_out_axes = tuple(unsafe_map(lambda spec: spec.get(naxis, None), out_axes))
      f = vtile(f, map_in_axes, map_out_axes, tile_size=None, axis_name=raxis)

    resource_env_shape = resource_env.shape
    for raxis, naxes in self.fake_resource_map.items():
      map_in_axes = tuple(unsafe_map(lambda spec: lookup_exactly_one_of(spec, naxes), in_axes))
      map_out_axes = tuple(unsafe_map(lambda spec: lookup_exactly_one_of(spec, naxes), out_axes))
      map_size = resource_env_shape[raxis]
      f = vtile(f, map_in_axes, map_out_axes, tile_size=map_size, axis_name=raxis)

    return f

  def to_mesh_axes(self, in_axes, out_axes):
    """
    Convert in/out_axes parameters ranging over logical dimensions to
    in/out_axes that range over the mesh dimensions.
    """
    def to_mesh(axes):
      mesh_axes = {}
      for paxis, naxes in self.physical_resource_map.items():
        axis = lookup_exactly_one_of(axes, naxes)
        if axis is None:
            continue
        mesh_axes[paxis] = axis
      return A(mesh_axes)
    return (tuple(unsafe_map(to_mesh, in_axes)),
            tuple(unsafe_map(to_mesh, out_axes)))

# -------- xmap primitive and its transforms --------

# xmap has a different set of parameters than pmap, so we make it its own primitive type
class XMapPrimitive(core.Primitive):
  multiple_results = True
  map_primitive = True  # Not really, but it gives us a few good behaviors

  def __init__(self):
    super().__init__('xmap')
    self.def_impl(xmap_impl)
    self.def_custom_bind(self.bind)

  def bind(self, fun, *args, **params):
    assert len(params['in_axes']) == len(args)
    return core.call_bind(self, fun, *args, **params)  # type: ignore

  def process(self, trace, fun, tracers, params):
    return trace.process_xmap(self, fun, tracers, params)

  def post_process(self, trace, out_tracers, params):
    raise NotImplementedError

xmap_p = XMapPrimitive()
core.EvalTrace.process_xmap = core.EvalTrace.process_call  # type: ignore
def _process_xmap_default(self, call_primitive, f, tracers, params):
  raise NotImplementedError(f"{type(self)} must override process_xmap to handle xmap")
core.Trace.process_xmap = _process_xmap_default  # type: ignore


# This is DynamicJaxprTrace.process_map with some very minor modifications
def _dynamic_jaxpr_process_xmap(self, primitive, f, tracers, params):
  from jax.interpreters.partial_eval import (
    trace_to_subjaxpr_dynamic, DynamicJaxprTracer, source_info_util,
    convert_constvars_jaxpr, call_param_updaters, new_jaxpr_eqn)
  assert primitive is xmap_p
  in_avals = [t.aval for t in tracers]
  axis_sizes = params['axis_sizes']
  mapped_in_avals = [_delete_aval_axes(a, a_in_axes)
                     for a, a_in_axes in zip(in_avals, params['in_axes'])]
  with core.extend_axis_env_nd(params['axis_sizes'].items()):
    jaxpr, mapped_out_avals, consts = trace_to_subjaxpr_dynamic(
        f, self.main, mapped_in_avals)
  out_axes = params['out_axes_thunk']()
  out_avals = [_insert_aval_axes(a, a_out_axes, axis_sizes)
               for a, a_out_axes in zip(mapped_out_avals, out_axes)]
  source_info = source_info_util.current()
  out_tracers = [DynamicJaxprTracer(self, a, source_info) for a in out_avals]
  invars = map(self.getvar, tracers)
  constvars = map(self.getvar, map(self.instantiate_const, consts))
  outvars = map(self.makevar, out_tracers)
  new_in_axes = (None,) * len(consts) + params['in_axes']
  new_params = dict(params, in_axes=new_in_axes, out_axes=out_axes,
                    call_jaxpr=convert_constvars_jaxpr(jaxpr))
  del new_params['out_axes_thunk']
  update_params = call_param_updaters.get(primitive)
  if update_params:
    new_params = update_params(new_params, [True] * len(tracers))
  eqn = new_jaxpr_eqn([*constvars, *invars], outvars, primitive,
                      new_params, source_info)
  self.frame.eqns.append(eqn)
  return out_tracers
pe.DynamicJaxprTrace.process_xmap = _dynamic_jaxpr_process_xmap  # type: ignore


# -------- nested xmap handling --------

def _xmap_translation_rule(c, axis_env,
                           in_nodes, name_stack, *,
                           call_jaxpr, name,
                           in_axes, out_axes, axis_sizes,
                           schedule, resource_env, backend):
  plan = EvaluationPlan.from_schedule(schedule, resource_env)

  # TODO: Make sure that the resource env matches the outer xmap
  local_mesh = resource_env.physical_mesh.local_mesh
  local_mesh_shape = local_mesh.shape
  mesh_in_axes, mesh_out_axes = plan.to_mesh_axes(in_axes, out_axes)

  assert type(call_jaxpr) is core.Jaxpr
  local_avals = [pxla.tile_aval_nd(
                    local_mesh_shape, aval_mesh_in_axes,
                    _insert_aval_axes(v.aval, aval_in_axes, axis_sizes))
                 for v, aval_in_axes, aval_mesh_in_axes
                 in zip(call_jaxpr.invars, in_axes, mesh_in_axes)]
  f = lu.wrap_init(core.jaxpr_as_fun(core.ClosedJaxpr(call_jaxpr, ())))
  f = hide_mapped_axes(f, tuple(in_axes), tuple(out_axes))
  f = plan.vectorize(f, in_axes, out_axes, resource_env)
  vectorized_jaxpr, _, consts = pe.trace_to_jaxpr_final(f, local_avals)
  assert not consts

  tiled_ins = (
    _xla_tile(c, axis_env, in_node, arg_in_axes, local_mesh_shape)
    if v.aval is not core.abstract_unit else in_node
    for v, in_node, arg_in_axes in zip(call_jaxpr.invars, in_nodes, mesh_in_axes))

  # We in-line here rather than generating a Call HLO as in the xla_call
  # translation rule just because the extra tuple stuff is a pain.
  with core.extend_axis_env_nd(axis_sizes.items()):
    tiled_outs = xla.jaxpr_subcomp(
        c, vectorized_jaxpr, backend, axis_env, (),
        xla.extend_name_stack(name_stack, xla.wrap_name(name, 'xmap')), *tiled_ins)

  outs = [_xla_untile(c, axis_env, tiled_out, ans_out_axes, local_mesh_shape, backend)
          if v.aval is not core.abstract_unit else tiled_out
          for v, tiled_out, ans_out_axes
          in zip(call_jaxpr.outvars, tiled_outs, mesh_out_axes)]

  return xops.Tuple(c, outs)
xla.call_translations[xmap_p] = _xmap_translation_rule

def _xla_tile(c, axis_env, x, in_axes, axis_sizes):
  if not in_axes:
    return x
  shape = list(c.get_shape(x).dimensions())
  zero = xb.constant(c, np.zeros((), dtype=np.int32))
  start_idxs = [zero] * len(shape)
  tiled_shape = list(shape)
  for name, axis in in_axes.items():
    axis_size = axis_sizes[name]

    assert tiled_shape[axis] % axis_size == 0
    tiled_shape[axis] //= axis_size

    axis_size_c = xb.constant(c, np.array(axis_size, np.int32))
    assert start_idxs[axis] is zero  # TODO(apaszke): tiling over multiple mesh axes
    axis_index = _axis_index_translation_rule(
        c, axis_name=name, axis_env=axis_env, platform=None)
    start_idxs[axis] = xops.Mul(axis_index, axis_size_c)
  return xops.DynamicSlice(x, start_idxs, tiled_shape)

# TODO(b/110096942): more efficient gather
def _xla_untile(c, axis_env, x, out_axes, axis_sizes, backend):
  xla_shape = c.get_shape(x)
  x_dtype = xla_shape.numpy_dtype()
  # TODO(mattjj): remove this logic when AllReduce PRED supported on CPU / GPU
  convert_bool = (np.issubdtype(x_dtype, np.bool_)
                  and xb.get_backend(backend).platform in ('cpu', 'gpu'))
  if convert_bool:
    x = xops.ConvertElementType(x, xb.dtype_to_etype(np.float32))

  untiled_shape = list(xla_shape.dimensions())
  zero_idx = xb.constant(c, np.zeros((), dtype=np.int32))
  start_idxs = [zero_idx] * len(untiled_shape)
  for name, axis in out_axes.items():
    axis_size = axis_sizes[name]

    untiled_shape[axis] *= axis_size

    axis_size_c = xb.constant(c, np.array(axis_size, np.int32))
    assert start_idxs[axis] is zero_idx  # TODO(apaszke): tiling over multiple mesh axes
    axis_index = _axis_index_translation_rule(
        c, axis_name=name, axis_env=axis_env, platform=None)
    start_idxs[axis] = xops.Mul(axis_index, axis_size_c)

  zero = xb.constant(c, np.array(0, x_dtype))
  padded = xops.Broadcast(zero, untiled_shape)
  padded = xops.DynamicUpdateSlice(padded, x, start_idxs)
  replica_groups_protos = xc.make_replica_groups(
    xla.axis_groups(axis_env, tuple(out_axes.keys())))
  out = xops.CrossReplicaSum(padded, replica_groups_protos)

  # TODO(mattjj): remove this logic when AllReduce PRED supported on CPU / GPU
  if convert_bool:
    nonzero = xops.Ne(out, xb.constant(c, np.array(0, dtype=np.float32)))
    out = xops.ConvertElementType(nonzero, xb.dtype_to_etype(np.bool_))
  return out


# -------- helper functions --------

def _delete_aval_axes(aval, axes: AxisNamePos):
  assert isinstance(aval, core.ShapedArray)
  shape = list(aval.shape)
  for i in sorted(axes.values(), reverse=True):
    del shape[i]
  return core.ShapedArray(tuple(shape), aval.dtype)


def _insert_aval_axes(aval, axes: AxisNamePos, axis_sizes):
  assert isinstance(aval, core.ShapedArray)
  shape = list(aval.shape)
  for name, axis in sorted(axes.items()):
    shape.insert(axis, axis_sizes[name])
  return core.ShapedArray(tuple(shape), aval.dtype)


# TODO: pmap has some very fancy error messages for this function!
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


@lu.transformation
def hide_mapped_axes(flat_in_axes, flat_out_axes, *flat_args):
  def _squeeze_mapped_axes(arg, axes: AxisNamePos):
    for dim in sorted(axes.values(), reverse=True):
      arg = arg.squeeze(dim)
    return arg

  def _unsqueeze_mapped_axes(out, axes: AxisNamePos):
    for dim in sorted(axes.values()):
      out = jnp.expand_dims(out, dim)
    return out

  squeezed_args = map(_squeeze_mapped_axes, flat_args, flat_in_axes)
  flat_outputs = yield squeezed_args, {}
  yield map(_unsqueeze_mapped_axes, flat_outputs, flat_out_axes)


# NOTE: This divides the in_axes by the tile_size and multiplies the out_axes by it.
def vtile(f_flat,
          in_axes_flat: Tuple[AxisNamePos, ...],
          out_axes_flat: Tuple[AxisNamePos, ...],
          tile_size: Optional[int], axis_name):
  if tile_size == 1:
    return f_flat

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

  @lu.transformation
  def _map_to_tile(*args_flat):
    sizes = (x.shape[i] for x, i in zip(args_flat, in_axes_flat) if i is not None)
    tile_size_ = tile_size or next(sizes, None)
    assert tile_size_ is not None, "No mapped arguments?"
    outputs_flat = yield map(tile_axis(tile_size=tile_size_), args_flat, in_axes_flat), {}
    yield map(untile_axis, outputs_flat, out_axes_flat)

  return _map_to_tile(
    batching.batch_fun(f_flat,
                       in_axes_flat,
                       out_axes_flat,
                       axis_name=axis_name))


def _schedule_resources(schedule) -> Set[ResourceAxisName]:
  return {resource for _, resource in schedule}


def _jaxpr_resources(jaxpr, resource_env) -> Set[ResourceAxisName]:
  used_resources = set()
  for eqn in jaxpr.eqns:
    if eqn.primitive is xmap_p:
      if eqn.params['resource_env'] != resource_env:
        raise RuntimeError("Changing the resource environment (e.g. hardware mesh "
                           "spec) is not allowed inside xmap.")
      used_resources |= _schedule_resources(eqn.params['schedule'])
    updates = core.traverse_jaxpr_params(
        partial(_jaxpr_resources, resource_env=resource_env), eqn.params)
    for update in updates:
      used_resources |= update
  return used_resources


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
    jaxpr = subst_axis_names(jaxpr, {full_axis_name: axis_names})  # type: ignore

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
  return eqn._replace(params=dict(eqn.params, axis_name=new_axis_names))
