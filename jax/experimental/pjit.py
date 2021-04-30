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
from collections import OrderedDict, Counter
from typing import Callable, Sequence, Tuple, Union
from warnings import warn
import itertools as it
from functools import partial

from . import maps
from . import PartitionSpec
from .. import core
from .. import linear_util as lu
from .._src.api import _check_callable, _check_arg
from .._src import source_info_util
from ..api_util import (argnums_partial_except, flatten_axes,
                        flatten_fun_nokwargs, _ensure_index_tuple,
                        donation_vector, rebase_donate_argnums)
from ..errors import JAXTypeError
from ..interpreters import ad
from ..interpreters import pxla
from ..interpreters import xla
from ..interpreters import partial_eval as pe
from ..lib import xla_bridge as xb
from ..lib import xla_client as xc
from ..tree_util import tree_flatten, tree_unflatten
from .._src.util import (extend_name_stack, HashableFunction, safe_zip,
                         wrap_name, wraps, distributed_debug_log,
                         split_list, cache)
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

  in_axis_resources, in_axis_resources_entries, _ = \
      _prepare_axis_resources(in_axis_resources, "in_axis_resources")
  out_axis_resources, out_axis_resources_entries, out_axis_treedef = \
      _prepare_axis_resources(out_axis_resources, "out_axis_resources")
  out_axis_resources_entries = tuple(out_axis_resources_entries)

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
    mesh = resource_env.physical_mesh

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

    local_in_avals = tuple(core.raise_to_shaped(core.get_aval(a)) for a in args_flat)
    jaxpr, in_axis_resources_flat, out_axis_resources_flat = \
        _pjit_jaxpr(flat_fun, mesh, local_in_avals,
                    in_tree, in_axis_resources,
                    HashableFunction(out_tree, closure=()), out_axis_resources)

    out = pjit_p.bind(
        *args_flat,
        jaxpr=jaxpr,
        in_axis_resources=in_axis_resources_flat,
        out_axis_resources=out_axis_resources_flat,
        resource_env=resource_env,
        donated_invars=donated_invars,
        name=flat_fun.__name__)
    return tree_unflatten(out_tree(), out)

  return wrapped

class _ListWithW(list):
  __slots__ = ('__weakref__',)

@lu.cache
def _pjit_jaxpr(fun, mesh, local_in_avals,
                in_tree, in_axis_resources,
                out_tree, out_axis_resources):
  in_axis_resources_flat = tuple(flatten_axes("pjit in_axis_resources",
                                              in_tree, in_axis_resources))
  _check_shapes_against_resources("pjit arguments", False, mesh.local_mesh.shape,
                                  local_in_avals, in_axis_resources_flat)
  global_in_avals = local_to_global(mesh, local_in_avals, in_axis_resources_flat)

  jaxpr, global_out_avals, consts = pe.trace_to_jaxpr_dynamic(fun, global_in_avals)
  jaxpr = core.ClosedJaxpr(jaxpr, consts)

  out_axis_resources_flat = tuple(flatten_axes("pjit out_axis_resources",
                                               out_tree(), out_axis_resources))
  _check_shapes_against_resources("pjit outputs", mesh.is_multi_process, mesh.shape,
                                  global_out_avals, out_axis_resources_flat)
  # lu.cache needs to be able to create weakrefs to outputs, so we can't return a plain tuple
  return _ListWithW([jaxpr, in_axis_resources_flat, out_axis_resources_flat])


class ParsedPartitionSpec:
  def __init__(self, user_spec, partitions):
    self.partitions = tuple(partitions)
    self.user_spec = user_spec

  @classmethod
  def from_user_input(cls, entry, arg_name):
    if entry is None:
      return cls(entry, ())
    if not isinstance(entry, PartitionSpec):
      raise TypeError(f"{arg_name} are expected to be "
                      f"PartitionSpec instances or None, but got {entry}")
    axis_specs = []
    for axis_spec in entry:
      if axis_spec is None:
        axis_spec = ()
      elif isinstance(axis_spec, (list, tuple)):
        axis_spec = tuple(axis_spec)
      else:
        axis_spec = (axis_spec,)
      axis_specs.append(axis_spec)
    return cls(entry, axis_specs)

  def __hash__(self):
    return hash(self.partitions)

  def __eq__(self, other):
    return (self.partitions, self.user_spec) == (other.partitions, other.user_spec)

  def __str__(self):
    return str(self.user_spec)

  def __len__(self):
    return len(self.partitions)

  def __getitem__(self, i):
    return self.partitions[i]

  def __iter__(self):
    return iter(self.partitions)

REPLICATED = ParsedPartitionSpec(None, ())


def _prepare_axis_resources(axis_resources, arg_name):
  # PyTrees don't treat None values as leaves, so we explicitly need
  # to explicitly declare them as such
  entries, treedef = tree_flatten(axis_resources, is_leaf=lambda x: x is None)
  what = f"{arg_name} leaf specifications"
  entries = [ParsedPartitionSpec.from_user_input(entry, what) for entry in entries]
  _check_unique_resources(entries, arg_name)
  return tree_unflatten(treedef, entries), entries, treedef

def _check_unique_resources(axis_resources, arg_name):
  for arg_axis_resources in axis_resources:
    if not arg_axis_resources: continue
    resource_counts = Counter(it.chain.from_iterable(arg_axis_resources))
    if resource_counts.most_common(1)[0][1] > 1:
      multiple_uses = [r for r, c in resource_counts.items() if c > 1]
      if multiple_uses:
        raise ValueError(f"A single {arg_name} specification can map every mesh axis "
                         f"to at most one positional dimension, but {arg_axis_resources} "
                         f"has duplicate entries for {maps.show_axes(multiple_uses)}")

def _check_shapes_against_resources(what: str, is_global_shape: bool, mesh_shape, flat_avals, flat_axis_resources):
  global_str = " global" if is_global_shape else ""
  for aval, aval_axis_resources in zip(flat_avals, flat_axis_resources):
    shape = aval.shape
    if len(shape) < len(aval_axis_resources):
      raise ValueError(f"One of {what} was given the resource assignment "
                       f"of {aval_axis_resources!s}, which implies that "
                       f"it has a rank of at least {len(aval_axis_resources)}, "
                       f"but it is {len(shape)}")
    for i, axis_resources in enumerate(aval_axis_resources):
      try:
        size = int(np.prod([mesh_shape[resource] for resource in axis_resources], dtype=np.int64))
      except KeyError as e:
        raise ValueError(f"One of {what} was given the resource assignment "
                         f"of {aval_axis_resources!s}, but resource axis "
                         f"{e.args[0]} is undefined. Did you forget to declare the mesh?") from None
      if shape[i] % size != 0:
        raise ValueError(f"One of {what} was given the resource assignment "
                         f"of {aval_axis_resources!s}, which implies that "
                         f"the{global_str} size of its dimension {i} should be "
                         f"divisible by {size}, but it is equal to {shape[i]}")

# -------------------- pjit rules --------------------

pjit_p = core.Primitive("pjit")
pjit_p.multiple_results = True


def _pjit_call_impl(*args, jaxpr,
                    in_axis_resources, out_axis_resources,
                    resource_env, donated_invars, name):
  compiled = pjit_callable(
      jaxpr, in_axis_resources, out_axis_resources,
      resource_env, donated_invars, name)
  distributed_debug_log(("Running pjit'd function", name),
                        ("mesh", resource_env.physical_mesh))
  return compiled(*args)
pjit_p.def_impl(_pjit_call_impl)

@cache()
def pjit_callable(
    jaxpr: core.ClosedJaxpr,
    in_axis_resources: Tuple[ParsedPartitionSpec, ...],
    out_axis_resources: Tuple[ParsedPartitionSpec, ...],
    resource_env,
    donated_invars,
    name: str):

  in_axes = [get_array_mapping(axes) for axes in in_axis_resources]
  out_axes = [get_array_mapping(axes) for axes in out_axis_resources]
  fun = lu.wrap_init(core.jaxpr_as_fun(jaxpr))
  local_in_avals = global_to_local(resource_env.physical_mesh,
                                   jaxpr.in_avals, in_axis_resources)
  # TODO(skye): allow for using a submesh of physical_mesh
  return pxla.mesh_callable(fun, name, None, resource_env.physical_mesh,
                            in_axes, out_axes, donated_invars,
                            True, *local_in_avals, tile_by_mesh_axes=False,
                            do_resource_typecheck="pjit")



def _pjit_abstract_eval(*args, jaxpr, out_axis_resources, resource_env, **_):
  return global_to_local(resource_env.physical_mesh,
                         jaxpr.out_avals, out_axis_resources)
pjit_p.def_abstract_eval(_pjit_abstract_eval)


def _pjit_translation_rule(c, axis_env, in_nodes, name_stack, backend, name,
                           jaxpr, in_axis_resources, out_axis_resources,
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

  # TODO: Think about how to avoid duplicating constants with the outer jaxpr
  out_nodes = xla.jaxpr_subcomp(
      subc, jaxpr.jaxpr, backend, axis_env, xla._xla_consts(subc, jaxpr.consts),
      extend_name_stack(name_stack, wrap_name(name, "pjit")), *args)
  out_nodes = [
      xb.set_sharding_proto(subc, out,
                            get_sharding_proto(subc, out, axis_resources, mesh))
      for out, axis_resources in safe_zip(out_nodes, out_axis_resources)
  ]

  subc = subc.build(xops.Tuple(subc, out_nodes))
  return xops.Call(c, subc, list(in_nodes))
xla.call_translations[pjit_p] = _pjit_translation_rule


def _pjit_jvp(primals_in, tangents_in,
              jaxpr, in_axis_resources, out_axis_resources,
              resource_env, donated_invars, name):
  is_nz_tangents_in = [type(t) is not ad.Zero for t in tangents_in]
  jaxpr_jvp, is_nz_tangents_out = ad.jvp_jaxpr(
      jaxpr, is_nz_tangents_in, instantiate=False)

  def _filter_zeros(is_nz_l, l):
    return (x for nz, x in zip(is_nz_l, l) if nz)
  _filter_zeros_in = partial(_filter_zeros, is_nz_tangents_in)
  _filter_zeros_out = partial(_filter_zeros, is_nz_tangents_out)
  outputs = pjit_p.bind(
      *primals_in, *_filter_zeros_in(tangents_in),
      jaxpr=jaxpr_jvp,
      in_axis_resources=(*in_axis_resources, *_filter_zeros_in(in_axis_resources)),
      out_axis_resources=(*out_axis_resources, *_filter_zeros_out(out_axis_resources)),
      resource_env=resource_env,
      donated_invars=(*donated_invars, *_filter_zeros_in(donated_invars)),
      name=wrap_name(name, 'jvp'))

  primals_out, tangents_out = split_list(outputs, [-len(is_nz_tangents_out)])
  assert len(primals_out) == len(jaxpr.jaxpr.outvars)
  tangents_out_it = iter(tangents_out)
  return primals_out, [next(tangents_out_it) if nz else ad.Zero(aval)
                       for nz, aval in zip(is_nz_tangents_out, jaxpr.out_avals)]
ad.primitive_jvps[pjit_p] = _pjit_jvp


def _check_resources_against_named_axes(what, aval, pos_axis_resources, named_axis_resources):
  pjit_resources = set(it.chain.from_iterable(pos_axis_resources))
  aval_resources = set(it.chain.from_iterable(
    named_axis_resources[a] for a in aval.named_shape))
  overlap = pjit_resources & aval_resources
  if overlap:
    raise JAXTypeError(
        f"{what} has an axis resources specification of {pos_axis_resources} "
        f"that uses one or more mesh axes already used by xmap to partition "
        f"a named axis appearing in its named_shape (both use mesh axes "
        f"{maps.show_axes(overlap)})")

def _resource_typing_pjit(avals, params, source_info, named_axis_resources):
  jaxpr = params["jaxpr"]
  what = "pjit input"
  for aval, pos_axis_resources in zip(jaxpr.in_avals, params['in_axis_resources']):
    _check_resources_against_named_axes(what, aval, pos_axis_resources, named_axis_resources)
  pxla.resource_typecheck(
      jaxpr.jaxpr, named_axis_resources,
      lambda: (f"a pjit'ed function {params['name']} "
               f"(pjit called at {source_info_util.summarize(source_info)})"))
  what = "pjit output"
  for aval, pos_axis_resources in zip(jaxpr.out_avals, params['out_axis_resources']):
    _check_resources_against_named_axes(what, aval, pos_axis_resources, named_axis_resources)
pxla.custom_resource_typing_rules[pjit_p] = _resource_typing_pjit


# -------------------- with_sharding_constraint --------------------

def with_sharding_constraint(x, axis_resources):
  x_flat, tree = tree_flatten(x)
  parsed_axis_resources, entries, _ = _prepare_axis_resources(axis_resources, "axis_resources")
  axis_resources_flat = tuple(
      flatten_axes("with_sharding_constraint axis_resources",
                   tree, parsed_axis_resources))
  resource_env = maps.thread_resources.env
  mesh = resource_env.physical_mesh
  _check_shapes_against_resources(
      "with_sharding_constraint arguments",
      mesh.is_multi_process, mesh.shape,
      x_flat, axis_resources_flat)
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
                  sharding_constraint_p.bind(
                      ct, axis_resources=axis_resources, resource_env=resource_env),))
xla.translations[sharding_constraint_p] = _sharding_constraint_translation_rule

def _resource_typing_sharding_constraint(avals, params, source_info, named_axis_resources):
  aval, = avals
  _check_resources_against_named_axes(
    "with_sharding_constraint input", aval,
    params['axis_resources'], named_axis_resources)
pxla.custom_resource_typing_rules[sharding_constraint_p] = \
    _resource_typing_sharding_constraint

# -------------------- helpers --------------------

def get_array_mapping(axis_resources: ParsedPartitionSpec) -> pxla.ArrayMapping:
  return OrderedDict((axis, i)
                     for i, axes in enumerate(axis_resources)
                     for axis in axes)

def get_sharding_proto(c, xla_op, axis_resources, mesh):
  xla_shape = c.GetShape(xla_op)
  if xla_shape.is_token():
    aval = core.abstract_token
    assert axis_resources is REPLICATED
  else:
    aval = core.ShapedArray(xla_shape.dimensions(), xla_shape.element_type())
  array_mapping = get_array_mapping(axis_resources)
  sharding_spec = pxla.mesh_sharding_specs(mesh.shape, mesh.axis_names)(
      aval, array_mapping)
  return sharding_spec.sharding_proto()

def global_to_local(mesh, avals, axes):
  return [mesh.global_to_local(get_array_mapping(aval_axes), aval)
          for aval, aval_axes in zip(avals, axes)]

def local_to_global(mesh, avals, axes):
  return [mesh.local_to_global(get_array_mapping(aval_axes), aval)
          for aval, aval_axes in zip(avals, axes)]
