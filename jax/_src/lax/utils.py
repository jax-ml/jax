# Copyright 2018 The JAX Authors.
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

# This module contains utility functions split out of jax._src.lax.lax to
# avoid cyclic dependencies. Definitions that are used at import time by
# multiple modules can go here.

from functools import partial

from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import mesh as mesh_lib
from jax._src.util import safe_zip
from jax._src.partition_spec import PartitionSpec as P
from jax._src.named_sharding import NamedSharding

zip, unsafe_zip = safe_zip, zip

import numpy as np

def _input_dtype(x, *_, **__):
  return dtypes.canonicalize_dtype(x.dtype, allow_extended_dtype=True)

def _argnum_weak_type(*argnums):
  return lambda *args, **_: all(args[i].weak_type for i in argnums)

def standard_primitive(shape_rule, dtype_rule, name,
                       weak_type_rule=None, sharding_rule=None):
  weak_type_rule = weak_type_rule or _standard_weak_type_rule
  prim = core.Primitive(name)
  prim.def_impl(partial(dispatch.apply_primitive, prim))
  prim.def_abstract_eval(
      partial(standard_abstract_eval, prim, shape_rule, dtype_rule,
              weak_type_rule, sharding_rule))
  return prim

def _get_array_abstraction_level(a): return a.array_abstraction_level

def _get_abstract_mesh_from_avals(in_avals) -> mesh_lib.AbstractMesh:
  m = None
  for a in in_avals:
    if a is core.abstract_token:
      continue
    if a.sharding.mesh.empty:
      continue
    if m is not None and m != a.sharding.mesh:
      if m._are_all_axes_auto and a.sharding.mesh._are_all_axes_auto:
        return mesh_lib.empty_abstract_mesh
      raise ValueError(
          f'Mesh for all inputs should be equal. Got one mesh: {m} and'
          f' another mesh: {a.sharding.mesh}')
    m = a.sharding.mesh
  return mesh_lib.empty_abstract_mesh if m is None else m


def call_sharding_rule(prim, rule, num_out, *avals, **kwargs):
  cur_mesh = mesh_lib.get_abstract_mesh()
  aval_mesh = _get_abstract_mesh_from_avals(avals)
  if ((cur_mesh.empty or cur_mesh._are_all_axes_auto_or_manual) and
      (aval_mesh.empty or aval_mesh._are_all_axes_auto_or_manual)):
    aval_mesh = cur_mesh if aval_mesh.empty else aval_mesh
    s = NamedSharding(aval_mesh, P())
    return s if num_out is None else [s] * num_out
  if rule is None:
    raise ValueError(
        f'sharding rule for {prim.name} is not implemented. Please file a'
        ' bug at https://github.com/jax-ml/jax/issues. You can work around'
        ' this error by dropping that operation into full auto sharding'
        ' mode via: `jax.experimental.shard.auto_axes(fun, out_shardings=...)`')
  return rule(*avals, **kwargs)

def standard_abstract_eval(prim, shape_rule, dtype_rule, weak_type_rule,
                           sharding_rule, *avals, **kwargs):
  assert all(isinstance(aval, core.UnshapedArray) for aval in avals), avals
  assert not prim.multiple_results
  weak_type = weak_type_rule(*avals, **kwargs)
  least_specialized = type(max(avals, key=_get_array_abstraction_level))
  if least_specialized is core.ShapedArray:
    core.check_avals_context_mesh(avals, prim.name)
    out_aval = core.ShapedArray(
        shape_rule(*avals, **kwargs), dtype_rule(*avals, **kwargs),
        weak_type=weak_type,
        sharding=call_sharding_rule(prim, sharding_rule, None, *avals, **kwargs))
    core.check_avals_context_mesh([out_aval], prim.name)
    return out_aval
  elif least_specialized is core.DShapedArray:
    shape = shape_rule(*avals, **kwargs)
    ty = (core.ShapedArray if all(type(d) is int for d in shape)
          else core.DShapedArray)
    return ty(shape, dtype_rule(*avals, **kwargs), weak_type)
  elif least_specialized is core.UnshapedArray:
    return core.UnshapedArray(dtype_rule(*avals, **kwargs), weak_type=weak_type)
  else:
    raise TypeError(avals, least_specialized)

def standard_multi_result_abstract_eval(
    prim, shape_rule, dtype_rule, weak_type_rule, sharding_rule,
    *avals, **kwargs):
  assert prim.multiple_results
  assert all(isinstance(aval, core.UnshapedArray) for aval in avals), avals
  least_specialized = max(map(type, avals), key=_get_array_abstraction_level)
  weak_types = weak_type_rule(*avals, **kwargs)
  if least_specialized is core.ShapedArray:
    out_shapes = shape_rule(*avals, **kwargs)
    out_dtypes = dtype_rule(*avals, **kwargs)
    core.check_avals_context_mesh(avals, prim.name)
    out_shardings = call_sharding_rule(
        prim, sharding_rule, len(out_shapes), *avals, **kwargs)
    if isinstance(weak_types, bool):
      weak_types = (weak_types,) * len(out_shapes)
    out_avals = [core.ShapedArray(s, d, weak_type=weak_type, sharding=sh)
                 for s, d, weak_type, sh in zip(out_shapes, out_dtypes,
                                                weak_types, out_shardings)]
    core.check_avals_context_mesh(out_avals, prim.name)
    return out_avals
  elif least_specialized is core.UnshapedArray:
    out_dtypes = dtype_rule(*avals, **kwargs)
    if isinstance(weak_types, bool):
      weak_types = (weak_types,) * len(out_dtypes)
    return [core.UnshapedArray(dtype, weak_type=weak_type)
            for dtype, weak_type in zip(out_dtypes, weak_types)]
  else:
    raise TypeError(avals, least_specialized)


def _standard_weak_type_rule(*avals, **kwargs):
  return all(aval.weak_type for aval in avals)

def dtype_to_string(dtype):
  try:
    return str(np.dtype(dtype).name)
  except TypeError:
    pass
  try:
    return dtype.name
  except AttributeError:
    pass
  return str(dtype)
