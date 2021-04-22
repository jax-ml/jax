# Copyright 2021 Google LLC
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

import numpy as np
from collections import OrderedDict
from typing import (Callable, Optional, Sequence, Tuple, Union)
from warnings import warn

from . import maps
from .. import core
from .. import linear_util as lu
from .._src.api import _check_callable, _check_arg
from ..api_util import (argnums_partial_except, flatten_axes,
                        flatten_fun_nokwargs, _ensure_index_tuple,
                        donation_vector, rebase_donate_argnums)
from ..interpreters import ad
from ..interpreters import pxla
from ..interpreters import xla
from ..interpreters import partial_eval as pe
from ..lib import xla_bridge as xb
from ..lib import xla_client as xc
from ..tree_util import tree_flatten, tree_unflatten
from .._src.util import (extend_name_stack, HashableFunction, safe_zip,
                         wrap_name, wraps, distributed_debug_log,
                         as_hashable_function)
xops = xc._xla.ops

def pjit(fun: Callable,
         in_axis_resources,
         out_axis_resources,
         static_argnums: Union[int, Sequence[int]] = (),
         donate_argnums: Union[int, Sequence[int]] = ()):
  warn("pjit is an experimental feature and probably has bugs!")
  _check_callable(fun)

  # To be a tree prefix of the positional args tuple, in_axes can never be a
  # list: if in_axes is not a leaf, it must be a tuple of trees. However,
  # in cases like these users expect tuples and lists to be treated
  # essentially interchangeably, so we canonicalize lists to tuples here
  # rather than raising an error. https://github.com/google/jax/issues/2367
  if isinstance(in_axis_resources, list):
    in_axis_resources = tuple(in_axis_resources)
  if isinstance(out_axis_resources, list):
    out_axis_resources = tuple(out_axis_resources)

  static_argnums = _ensure_index_tuple(static_argnums)
  donate_argnums = _ensure_index_tuple(donate_argnums)
  donate_argnums = rebase_donate_argnums(donate_argnums, static_argnums)

  @wraps(fun)
  def wrapped(*args, **kwargs):
    if kwargs:
      raise NotImplementedError("pjit over kwargs not yet supported")
    if max(static_argnums + donate_argnums, default=-1) >= len(args):
      raise ValueError(f"jitted function has static_argnums={static_argnums}, "
                       f"donate_argnums={donate_argnums} but "
                       f"was called with only {len(args)} positional arguments.")

    # Putting this outside of wrapped would make resources lexically scoped
    resource_env = maps.thread_resources.env

    f = lu.wrap_init(fun)
    if static_argnums:
      f, dyn_args = argnums_partial_except(
          f, static_argnums, args, allow_invalid=False)
    else:
      dyn_args = args

    args_flat, in_tree = tree_flatten(args)
    for arg in args_flat: _check_arg(arg)
    flat_fun, out_tree = flatten_fun_nokwargs(f, in_tree)
    if donate_argnums:
      donated_invars = donation_vector(donate_argnums, dyn_args, ())
    else:
      donated_invars = (False,) * len(args_flat)

    in_axis_resources_flat = tuple(flatten_axes("pjit in_axis_resources",
                                                in_tree, in_axis_resources))
    out_axis_resources_thunk = HashableFunction(
        lambda: tuple(flatten_axes("pjit out_axis_resources", out_tree(),
                                   out_axis_resources)),
        closure=out_axis_resources)
    _check_shapes_against_resources("pjit arguments", resource_env,
                                    args_flat, in_axis_resources_flat)
    flat_fun = _check_output_shapes(flat_fun, resource_env,
                                    out_axis_resources_thunk)

    out = pjit_call_p.bind(
        flat_fun,
        *args_flat,
        in_axis_resources=in_axis_resources_flat,
        out_axis_resources_thunk=out_axis_resources_thunk,
        resource_env=resource_env,
        donated_invars=donated_invars,
        name=flat_fun.__name__)
    return tree_unflatten(out_tree(), out)

  return wrapped

@lu.transformation
def _check_output_shapes(resource_env, out_axis_resources_thunk, *args, **kwargs):
  outputs = yield (args, kwargs)
  _check_shapes_against_resources("pjit outputs", resource_env, outputs, out_axis_resources_thunk())
  yield outputs

def _check_shapes_against_resources(what: str, resource_env, flat_avals, flat_axis_resources):
  resource_sizes = resource_env.local_shape
  for aval, aval_axis_resources in zip(flat_avals, flat_axis_resources):
    if aval_axis_resources is None:
      continue
    shape = aval.shape
    if len(shape) < len(aval_axis_resources):
      raise ValueError(f"One of {what} was given the resource assignment "
                       f"of {aval_axis_resources}, which implies that it has "
                       f"a rank of at least {len(aval_axis_resources)}, but "
                       f"it is {len(shape)}")
    for i, axis_resources in enumerate(aval_axis_resources):
      if axis_resources is None:
        continue
      try:
        if isinstance(axis_resources, (list, tuple)):
          size = int(np.prod([resource_sizes[resource] for resource in axis_resources], dtype=np.int64))
        else:
          size = resource_sizes[axis_resources]
      except KeyError as e:
        raise ValueError(f"One of {what} was given the resource assignment "
                         f"of {aval_axis_resources}, but resource axis {e.args[0]} "
                         f"is undefined. Did you forget to declare the mesh?")
      if shape[i] % size != 0:
        raise ValueError(f"One of {what} was given the resource assignment "
                         f"of {aval_axis_resources}, which implies that the size of "
                         f"its dimension {i} should be divisible by {size}, but it "
                         f"is equal to {shape[i]}")

def _pjit_call_impl(fun: lu.WrappedFun, *args, in_axis_resources,
                    out_axis_resources_thunk, resource_env, donated_invars,
                    name):
  in_avals = [core.raise_to_shaped(core.get_aval(arg)) for arg in args]
  pjit_callable = _pjit_callable(
      fun, in_axis_resources, out_axis_resources_thunk, resource_env,
      donated_invars, name, *in_avals)
  distributed_debug_log(("Running pjit'd function", name),
                        ("python function", fun.f),
                        ("mesh", resource_env.physical_mesh),
                        ("abstract args", in_avals))
  return pjit_callable(*args)

pjit_call_p = core.CallPrimitive("pjit_call")
pjit_call_p.def_impl(_pjit_call_impl)

# None indicates unpartitioned dimension
ArrayAxisPartitioning = Optional[Union[pxla.MeshAxisName, Tuple[pxla.MeshAxisName, ...]]]
# None indicates fully replicated array value
ArrayPartitioning = Optional[Tuple[ArrayAxisPartitioning, ...]]

@lu.cache
def _pjit_callable(
    fun: lu.WrappedFun,
    in_axis_resources: Tuple[ArrayPartitioning, ...],
    out_axis_resources_thunk: Callable[[], Tuple[ArrayPartitioning, ...]],
    resource_env,
    donated_invars,
    name: str,
    *in_avals):

  in_axes = [get_array_mapping(axes) for axes in in_axis_resources]
  out_axes = lambda: [get_array_mapping(axes) for axes in out_axis_resources_thunk()]
  # TODO(skye): allow for using a submesh of physical_mesh
  return pxla.mesh_callable(fun, name, None, resource_env.physical_mesh,
                            in_axes, out_axes, donated_invars,
                            True, *in_avals, tile_by_mesh_axes=False)

# -------------------- pjit rules --------------------

def _pjit_translation_rule(c, axis_env, in_nodes, name_stack, backend, name,
                           call_jaxpr, in_axis_resources, out_axis_resources,
                           resource_env, donated_invars):
  mesh = resource_env.physical_mesh
  subc = xc.XlaBuilder(f"pjit_{name}")

  args = []
  for i, (n, axis_resources) in enumerate(safe_zip(in_nodes, in_axis_resources)):
    # N.B. inlined calls shouldn't have shardings set directly on the inputs or
    # outputs (set_sharding_proto adds an identity operation).
    arg = xb.parameter(subc, i, c.GetShape(n))
    args.append(xb.set_sharding_proto(subc, arg,
                                      get_sharding_proto(c, n, axis_resources, mesh)))

  out_nodes = xla.jaxpr_subcomp(
      subc, call_jaxpr, backend, axis_env, (),
      extend_name_stack(name_stack, wrap_name(name, "pjit")), *args)
  out_nodes = [xb.set_sharding_proto(subc, out,
                                     get_sharding_proto(c, n, axis_resources, mesh))
               for out, axis_resources in safe_zip(out_nodes, out_axis_resources)]

  subc = subc.build(xops.Tuple(subc, out_nodes))
  return xops.Call(c, subc, list(in_nodes))
xla.call_translations[pjit_call_p] = _pjit_translation_rule

def _pjit_partial_eval_update_params(params, in_unknowns):
  call_jaxpr = params['call_jaxpr']
  donated_invars = params['donated_invars']
  in_axis_resources = params['in_axis_resources']
  out_axis_resources_thunk = params['out_axis_resources_thunk']
  if not in_unknowns and donated_invars:
    # JaxprTrace.post_process_call creates a call with no input tracers
    new_donated_invars = (False,) * len(call_jaxpr.invars)
    new_in_axis_resources = (None,) * len(call_jaxpr.invars)
  else:
    # JaxprTrace.process_call drops known input tracers and prepends constants
    num_consts = len(call_jaxpr.invars) - len(donated_invars)
    def filter_unknown(l):
      return tuple(x for x, uk in zip(l, in_unknowns) if uk)
    new_donated_invars = (False,) * num_consts + filter_unknown(donated_invars)
    new_in_axis_resources = (None,) * num_consts + filter_unknown(in_axis_resources)
  new_out_axis_resources = out_axis_resources_thunk()
  new_params = dict(params,
                    donated_invars=new_donated_invars,
                    in_axis_resources=new_in_axis_resources,
                    out_axis_resources=new_out_axis_resources)
  del new_params['out_axis_resources_thunk']
  return new_params
pe.call_param_updaters[pjit_call_p] = _pjit_partial_eval_update_params

def _pjit_jvp_update_params(params, nz_tangents, nz_tangents_out_thunk):
  donated_invars = params['donated_invars']
  in_axis_resources = params['in_axis_resources']
  out_axis_resources_thunk = params['out_axis_resources_thunk']
  def filter_nonzero_ins(l):
    return tuple(x for x, nz in zip(l, nz_tangents) if nz)
  @as_hashable_function(closure=(tuple(nz_tangents), out_axis_resources_thunk))
  def new_out_axis_resources_thunk():
    out_axis_resources = out_axis_resources_thunk()
    return (*out_axis_resources,
            *(ax for ax, nz in zip(out_axis_resources, nz_tangents_out_thunk()) if nz))
  return dict(params,
              donated_invars=(donated_invars + filter_nonzero_ins(donated_invars)),
              in_axis_resources=(in_axis_resources + filter_nonzero_ins(in_axis_resources)),
              out_axis_resources_thunk=new_out_axis_resources_thunk)
ad.call_param_updaters[pjit_call_p] = _pjit_jvp_update_params


# -------------------- with_sharding_constraint --------------------

def with_sharding_constraint(x, axis_resources):
  x_flat, tree = tree_flatten(x)
  axis_resources_flat = tuple(
      flatten_axes("with_sharding_constraint axis_resources",
                   tree, axis_resources))
  resource_env = maps.thread_resources.env
  _check_shapes_against_resources(
      "with_sharding_constraint arguments",
      resource_env, x_flat, axis_resources_flat)
  outs = [sharding_constraint_p.bind(y, axis_resources=r, resource_env=resource_env)
          for y, r in safe_zip(x_flat, axis_resources_flat)]
  return tree_unflatten(tree, outs)

def _sharding_constraint_impl(x, axis_resources, resource_env):
  # TODO(skye): can we also prevent this from being called in other
  # non-pjit contexts? (e.g. pmap, control flow)
  raise NotImplementedError(
      "with_sharding_constraint() should only be called inside pjit()")

def _sharding_constraint_translation_rule(c, x_node, axis_resources, resource_env):
  mesh = resource_env.physical_mesh
  return xb.set_sharding_proto(c, x_node,
                               get_sharding_proto(c, x_node, axis_resources, mesh))

sharding_constraint_p = core.Primitive("sharding_constraint")
sharding_constraint_p.def_impl(_sharding_constraint_impl)
sharding_constraint_p.def_abstract_eval(lambda x, **unused: x)
ad.deflinear2(sharding_constraint_p,
              lambda ct, _, axis_resources, resource_env: (
                  with_sharding_constraint(ct, axis_resources),))
xla.translations[sharding_constraint_p] = _sharding_constraint_translation_rule

# -------------------- helpers --------------------

def get_array_mapping(axis_resources: ArrayPartitioning) -> pxla.ArrayMapping:
  if axis_resources is None:
    return OrderedDict()
  return OrderedDict(entry
                     for i, axis_or_axes in enumerate(axis_resources)
                     for entry in _array_mapping_entries(axis_or_axes, i))

def _array_mapping_entries(partitioning: ArrayAxisPartitioning, i: int):
  if partitioning is None:
    return
  if not isinstance(partitioning, (list, tuple)):
    yield (partitioning, i)
  else:
    for axis in partitioning:
      yield (axis, i)

def get_sharding_proto(c, xla_op, axis_resources, mesh):
  xla_shape = c.GetShape(xla_op)
  aval = core.ShapedArray(xla_shape.dimensions(), xla_shape.element_type())
  array_mapping = get_array_mapping(axis_resources)
  sharding_spec = pxla.mesh_sharding_specs(mesh.shape, mesh.axis_names)(
      aval, array_mapping)
  return sharding_spec.sharding_proto()
