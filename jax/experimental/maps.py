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

import threading
import contextlib
import numpy as np
import itertools as it
from collections import OrderedDict
from typing import (Callable, Iterable, Tuple, Optional, Dict, Any, Set,
                    NamedTuple, Union)
from warnings import warn
from functools import wraps, partial

from .. import numpy as jnp
from .. import core
from .. import linear_util as lu
from ..api import _check_callable, _check_arg
from ..tree_util import tree_flatten, tree_unflatten, all_leaves
from ..api_util import flatten_fun_nokwargs, flatten_axes
from ..interpreters import partial_eval as pe
from ..interpreters import pxla
from ..interpreters import xla
from ..lib import xla_bridge as xb
from ..lib import xla_client as xc
from .._src.util import safe_map, safe_zip, HashableFunction
from .._src.lax.parallel import _axis_index_translation_rule

map, unsafe_map = safe_map, map
zip = safe_zip

xops = xc.ops

EXPERIMENTAL_SPMD_LOWERING = False

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

  def __repr__(self):
    return f"FrozenDict({self.contents})"

# Multi-dimensional generalized map

AxisName = core.AxisName
ResourceAxisName = AxisName  # Different name just for documentation purposes
Mesh = pxla.Mesh

# TODO: Support sequential mapping
class ResourceEnv:
  __slots__ = ('physical_mesh',)
  physical_mesh: Mesh

  def __init__(self, physical_mesh: Mesh):
    super().__setattr__('physical_mesh', physical_mesh)

  @property
  def physical_resource_axes(self) -> Set[ResourceAxisName]:
    return set(self.physical_mesh.axis_names)

  @property
  def resource_axes(self) -> Set[ResourceAxisName]:
    return self.physical_resource_axes

  @property
  def shape(self):
    return OrderedDict(self.physical_mesh.shape)

  def __setattr__(self, name, value):
    raise RuntimeError("ResourceEnv is immutable!")

  def __delattr__(self):
    raise RuntimeError("ResourceEnv is immutable!")

  def __eq__(self, other):
    return (type(other) is ResourceEnv and
            self.physical_mesh == other.physical_mesh)

  def __hash__(self):
    return hash(self.physical_mesh)

thread_resources = threading.local()
thread_resources.env = ResourceEnv(Mesh(np.empty((), dtype=object), ()))

@contextlib.contextmanager
def mesh(*args, **kwargs):
  old_env = thread_resources.env
  thread_resources.env = ResourceEnv(Mesh(*args, **kwargs))
  try:
    yield
  finally:
    thread_resources.env = old_env

_next_resource_id = 0
class _UniqueResourceName:
  def __init__(self, uid, tag=None):
    self.uid = uid
    self.tag = tag
  def __eq__(self, other):
    return type(other) is _UniqueResourceName and self.uid == other.uid
  def __hash__(self):
    return hash(self.uid)
  def __repr__(self):
    return f"<UniqueResource {self.tag} {self.uid}>"

def fresh_resource_name(tag=None):
  global _next_resource_id
  try:
    return _UniqueResourceName(_next_resource_id, tag)
  finally:
    _next_resource_id += 1

# This is really a Dict[AxisName, int], but we don't define a
# pytree instance for it, so that it is treated as a leaf.
class AxisNamePos(FrozenDict):
  pass

def _parse_entry(arg_name, entry):
  # Dictionaries mapping axis names to positional axes
  if isinstance(entry, dict) and all(isinstance(v, int) for v in entry.keys()):
    result = AxisNamePos((name, axis) for axis, name in entry.items())
    num_mapped_dims = len(entry)
  # Non-empty lists or tuples that terminate with an ellipsis
  elif isinstance(entry, (tuple, list)) and entry and entry[-1] == ...:
    result = AxisNamePos((name, axis) for axis, name in enumerate(entry[:-1])
                         if name is not None)
    num_mapped_dims = sum(name is not None for name in entry[:-1])
  else:
    raise TypeError(f"""\
Value mapping specification in xmap {arg_name} pytree can be either:
- lists of axis names, ending with ellipsis (...)
- dictionaries that map axis names to positional axes (integers)
but got: {entry}""")
  if len(result) != num_mapped_dims:
    raise ValueError(f"Named axes should be unique within each {arg_name} argument "
                     f"specification, but one them is: {entry}")
  return result

def _is_axes_leaf(entry):
  if isinstance(entry, dict) and all_leaves(entry.values()):
    return True
  # NOTE: `None`s are not considered leaves by `all_leaves`
  if isinstance(entry, (tuple, list)) and all_leaves(v for v in entry if v is not None):
    return True
  return False

def _prepare_axes(axes, arg_name):
  entries, treedef = tree_flatten(axes, is_leaf=_is_axes_leaf)
  entries = map(partial(_parse_entry, arg_name), entries)
  return tree_unflatten(treedef, entries), entries

# TODO: Some syntactic sugar to make the API more usable in a single-axis case?
# TODO: Are the resource axes scoped lexically or dynamically? Dynamically for now!
def xmap(fun: Callable,
         in_axes,
         out_axes,
         axis_resources: Dict[AxisName, Union[ResourceAxisName, Tuple[ResourceAxisName, ...]]] = {},
         backend: Optional[str] = None):
  warn("xmap is an experimental feature and probably has bugs!")
  _check_callable(fun)


  # To be a tree prefix of the positional args tuple, in_axes can never be a
  # list: if in_axes is not a leaf, it must be a tuple of trees. However,
  # in cases like these users expect tuples and lists to be treated
  # essentially interchangeably, so we canonicalize lists to tuples here
  # rather than raising an error. https://github.com/google/jax/issues/2367
  if isinstance(in_axes, list):
    in_axes = tuple(in_axes)
  if isinstance(out_axes, list):
    out_axes = tuple(out_axes)

  in_axes, in_axes_entries = _prepare_axes(in_axes, "in_axes")
  out_axes, out_axes_entries = _prepare_axes(out_axes, "out_axes")

  in_axes_names = set(it.chain(*(spec.keys() for spec in in_axes_entries)))
  out_axes_names = set(it.chain(*(spec.keys() for spec in out_axes_entries)))
  normalized_axis_resources: Dict[AxisName, Tuple[ResourceAxisName, ...]] = \
      {axis: (resources if isinstance(resources, tuple) else (resources,))
       for axis, resources in axis_resources.items()}
  for axis in in_axes_names:
    normalized_axis_resources.setdefault(axis, ())
  frozen_axis_resources = FrozenDict(normalized_axis_resources)
  necessary_resources = set(it.chain(*frozen_axis_resources.values()))

  axes_with_resources = set(frozen_axis_resources.keys())
  if axes_with_resources > in_axes_names:
    raise ValueError(f"All axes that were assigned resources have to appear in "
                     f"in_axes, but the following are missing: "
                     f"{axes_with_resources - in_axes_names}")
  if out_axes_names > in_axes_names:
    raise ValueError(f"All axis names appearing in out_axes must also appear in "
                     f"in_axes, but the following are missing: "
                     f"{out_axes_names - in_axes_names}")

  for axis, resources in frozen_axis_resources.items():
    if len(set(resources)) != len(resources):
      raise ValueError(f"Resource assignment of a single axis must be a tuple of "
                       f"distinct resources, but specified {resources} for axis {axis}")

  @wraps(fun)
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
      closure=out_axes)
    axis_sizes = _get_axis_sizes(args_flat, in_axes_flat)
    out_flat = xmap_p.bind(
      fun_flat, *args_flat,
      name=fun.__name__,
      in_axes=tuple(in_axes_flat),
      out_axes_thunk=out_axes_thunk,
      axis_sizes=FrozenDict(axis_sizes),
      axis_resources=frozen_axis_resources,
      resource_env=resource_env,
      backend=backend)
    return tree_unflatten(out_tree(), out_flat)

  return fun_mapped

def xmap_impl(fun: lu.WrappedFun, *args, name, in_axes, out_axes_thunk, axis_sizes,
              axis_resources, resource_env, backend):
  in_avals = [core.raise_to_shaped(core.get_aval(arg)) for arg in args]
  return make_xmap_callable(fun, name, in_axes, out_axes_thunk, axis_sizes,
                            axis_resources, resource_env, backend, *in_avals)(*args)

@lu.cache
def make_xmap_callable(fun: lu.WrappedFun,
                       name,
                       in_axes, out_axes_thunk, axis_sizes,
                       axis_resources, resource_env, backend,
                       *in_avals):
  plan = EvaluationPlan.from_axis_resources(axis_resources, resource_env)

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
  f = plan.vectorize(f, in_axes, out_axes)

  used_resources = _jaxpr_resources(jaxpr, resource_env) | set(it.chain(*axis_resources.values()))
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
                                    EXPERIMENTAL_SPMD_LOWERING,
                                    *in_avals)
  else:
    return f.call_wrapped

class EvaluationPlan(NamedTuple):
  """Encapsulates preprocessing common to top-level xmap invocations and its translation rule."""
  physical_axis_resources: Dict[AxisName, Tuple[ResourceAxisName, ...]]
  axis_subst: Dict[AxisName, Tuple[ResourceAxisName, ...]]

  @classmethod
  def from_axis_resources(cls, axis_resources: Dict[AxisName, Tuple[ResourceAxisName, ...]], resource_env):
    # TODO: Support sequential resources
    physical_axis_resources = axis_resources  # NB: We only support physical resources at the moment
    axis_subst = {name: axes + (fresh_resource_name(name),) for name, axes in axis_resources.items()}
    return cls(physical_axis_resources, axis_subst)

  def vectorize(self, f: lu.WrappedFun, in_axes, out_axes):
    for naxis, raxes in self.axis_subst.items():
      vaxis = raxes[-1]
      map_in_axes = tuple(unsafe_map(lambda spec: spec.get(naxis, None), in_axes))
      map_out_axes = tuple(unsafe_map(lambda spec: spec.get(naxis, None), out_axes))
      f = pxla.vtile(f, map_in_axes, map_out_axes, tile_size=None, axis_name=vaxis)
    return f

  def to_mesh_axes(self, in_axes, out_axes):
    """
    Convert in/out_axes parameters ranging over logical dimensions to
    in/out_axes that range over the mesh dimensions.
    """
    def to_mesh(axes):
      return OrderedDict((physical_axis, pos_axis)
                         for logical_axis, pos_axis in axes.items()
                         for physical_axis in self.physical_axis_resources[logical_axis])
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

def _xmap_translation_rule(*args, **kwargs):
  if EXPERIMENTAL_SPMD_LOWERING:
    return _xmap_translation_rule_spmd(*args, **kwargs)
  else:
    return _xmap_translation_rule_replica(*args, **kwargs)
xla.call_translations[xmap_p] = _xmap_translation_rule

def _xmap_translation_rule_replica(c, axis_env,
                                   in_nodes, name_stack, *,
                                   call_jaxpr, name,
                                   in_axes, out_axes, axis_sizes,
                                   axis_resources, resource_env, backend):
  plan = EvaluationPlan.from_axis_resources(axis_resources, resource_env)

  local_mesh = resource_env.physical_mesh.local_mesh
  local_mesh_shape = local_mesh.shape
  mesh_in_axes, mesh_out_axes = plan.to_mesh_axes(in_axes, out_axes)
  raise NotImplementedError("TODO: Substitute axis names!")

  assert type(call_jaxpr) is core.Jaxpr
  local_avals = [pxla.tile_aval_nd(
                    local_mesh_shape, aval_mesh_in_axes,
                    _insert_aval_axes(v.aval, aval_in_axes, axis_sizes))
                 for v, aval_in_axes, aval_mesh_in_axes
                 in zip(call_jaxpr.invars, in_axes, mesh_in_axes)]
  f = lu.wrap_init(core.jaxpr_as_fun(core.ClosedJaxpr(call_jaxpr, ())))
  f = hide_mapped_axes(f, tuple(in_axes), tuple(out_axes))
  f = plan.vectorize(f, in_axes, out_axes)
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

def _xmap_translation_rule_spmd(c, axis_env,
                                in_nodes, name_stack, *,
                                call_jaxpr, name,
                                in_axes, out_axes, axis_sizes,
                                axis_resources, resource_env, backend):
  # TODO(apaszke): This is quite difficult to implement given the current lowering
  #                in mesh_tiled_callable. There, we vmap the mapped axes, but we
  #                have no idea which positional axes they end up being in this
  #                translation rule!
  raise NotImplementedError


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


def _jaxpr_resources(jaxpr, resource_env) -> Set[ResourceAxisName]:
  used_resources = set()
  for eqn in jaxpr.eqns:
    if eqn.primitive is xmap_p:
      if eqn.params['resource_env'] != resource_env:
        raise RuntimeError("Changing the resource environment (e.g. hardware mesh "
                           "spec) is not allowed inside xmap.")
      used_resources |= set(it.chain(*eqn.params['axis_resources'].values()))
    updates = core.traverse_jaxpr_params(
        partial(_jaxpr_resources, resource_env=resource_env), eqn.params)
    for update in updates:
      used_resources |= update
  return used_resources

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
