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

from collections import OrderedDict
from typing import (Callable, Optional, Sequence, Tuple, Union)
from warnings import warn

from . import maps
from .. import core
from .. import linear_util as lu
from ..api import _check_callable, _check_arg
from ..api_util import (argnums_partial_except, flatten_axes,
                        flatten_fun_nokwargs, _ensure_index_tuple,
                        donation_vector, rebase_donate_argnums)
from ..interpreters import ad
from ..interpreters import pxla
from ..interpreters import xla
from ..lib import xla_bridge as xb
from ..lib import xla_client as xc
from ..tree_util import tree_flatten, tree_unflatten
from .._src.util import (extend_name_stack, HashableFunction, safe_zip,
                         wrap_name, wraps)
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
      f, dyn_args = argnums_partial_except(f, static_argnums, args)
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

def _pjit_call_impl(fun: lu.WrappedFun, *args, in_axis_resources,
                    out_axis_resources_thunk, resource_env, donated_invars,
                    name):
  in_avals = [core.raise_to_shaped(core.get_aval(arg)) for arg in args]
  return _pjit_callable(
      fun, in_axis_resources, out_axis_resources_thunk, resource_env,
      donated_invars, name, *in_avals)(*args)

def _pjit_translation_rule(c, axis_env, in_nodes, name_stack, backend, name,
                           call_jaxpr, in_axis_resources, out_axis_resources_thunk,
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
  out_axis_resources = out_axis_resources_thunk()
  out_nodes = [xb.set_sharding_proto(subc, out,
                                     get_sharding_proto(c, n, axis_resources, mesh))
               for out, axis_resources in safe_zip(out_nodes, out_axis_resources)]

  subc = subc.build(xops.Tuple(subc, out_nodes))
  return xops.Call(c, subc, list(in_nodes))

pjit_call_p = core.CallPrimitive("pjit_call")
pjit_call_p.def_impl(_pjit_call_impl)
xla.call_translations[pjit_call_p] = _pjit_translation_rule

# None indicates unpartitioned dimension
ArrayAxisPartitioning = Union[pxla.MeshAxisName, Tuple[pxla.MeshAxisName, ...], None]
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
  out_axes_thunk = lambda: [get_array_mapping(axes)
                            for axes in out_axis_resources_thunk()]
  # TODO(skye): allow for using a submesh of physical_mesh
  return pxla.mesh_callable(fun, name, None, resource_env.physical_mesh,
                            in_axes, out_axes_thunk, donated_invars,
                            True, *in_avals, tile_by_mesh_axes=False)


def with_sharding_constraint(x, axis_resources):
  resource_env = maps.thread_resources.env
  return sharding_constraint_p.bind(x, axis_resources=axis_resources,
                                    resource_env=resource_env)

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
      assert isinstance(axis, str)
      yield (axis, i)

def get_sharding_proto(c, xla_op, axis_resources, mesh):
  xla_shape = c.GetShape(xla_op)
  aval = core.ShapedArray(xla_shape.dimensions(), xla_shape.element_type())
  array_mapping = get_array_mapping(axis_resources)
  sharding_spec = pxla.mesh_sharding_specs(mesh.shape, mesh.axis_names)(
      aval, array_mapping)
  return sharding_spec.sharding_proto()
