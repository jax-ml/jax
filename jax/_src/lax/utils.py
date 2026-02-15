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

import numpy as np

from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import mesh as mesh_lib
from jax._src import state
from jax._src.named_sharding import DuplicateSpecError, NamedSharding
from jax._src.partition_spec import PartitionSpec as P
from jax._src.util import safe_zip
from jax._src.typing import DimSize, DType, Shape

zip, unsafe_zip = safe_zip, zip


def input_dtype(x, *_, **__):
  return x.dtype

def _argnum_weak_type(*argnums):
  return lambda *args, **_: all(args[i].weak_type for i in argnums)

def standard_primitive(shape_rule, dtype_rule, name,
                       weak_type_rule=None, sharding_rule=None, vma_rule=None,
                       unreduced_rule=None, reduced_rule=None,
                       memory_space_rule=None):
  weak_type_rule = weak_type_rule or _standard_weak_type_rule
  prim = core.Primitive(name)
  prim.def_impl(partial(dispatch.apply_primitive, prim))
  prim.def_abstract_eval(
      partial(standard_abstract_eval, prim, shape_rule, dtype_rule,
              weak_type_rule, sharding_rule, vma_rule, unreduced_rule,
              reduced_rule, memory_space_rule))
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
      if m.are_all_axes_auto and a.sharding.mesh.are_all_axes_auto:
        return mesh_lib.empty_abstract_mesh
      raise ValueError(
          f'Mesh for all inputs should be equal. Got one mesh: {m} and'
          f' another mesh: {a.sharding.mesh}')
    m = a.sharding.mesh
  return mesh_lib.empty_abstract_mesh if m is None else m

def call_reduced_rule(prim, reduced_rule, out_s, num_out, *avals, **kwargs):
  if reduced_rule is not None:
    return reduced_rule(out_s, *avals, **kwargs)
  if any(a.sharding.spec.reduced for a in avals):
    raise NotImplementedError(
        f'reduced rule for {prim.name} is not implemented. Please file an'
        ' issue at https://github.com/jax-ml/jax/issues')
  if any(s.spec.reduced for s in ([out_s] if num_out is None else out_s)
         if s is not None):
    raise NotImplementedError(
        f'reduced rule for {prim.name} is not implemented. Please file an'
        ' issue at https://github.com/jax-ml/jax/issues')
  return out_s

def call_unreduced_rule(prim, unreduced_rule, out_s, num_out, *avals, **kwargs):
  if unreduced_rule is not None:
    return unreduced_rule(out_s, *avals, **kwargs)

  if any(a.sharding.spec.unreduced for a in avals):
    raise NotImplementedError(
        f'unreduced rule for {prim.name} is not implemented. Please file an'
        ' issue at https://github.com/jax-ml/jax/issues')
  if any(s.spec.unreduced for s in ([out_s] if num_out is None else out_s)
         if s is not None):
    raise NotImplementedError(
        f'unreduced rule for {prim.name} is not implemented. Please file an'
        ' issue at https://github.com/jax-ml/jax/issues')
  return out_s

def call_sharding_rule(prim, sh_rule, unreduced_rule, reduced_rule, num_out,
                       *avals, **kwargs):
  cur_mesh = mesh_lib.get_abstract_mesh()
  aval_mesh = _get_abstract_mesh_from_avals(avals)
  if ((cur_mesh.empty or cur_mesh._are_all_axes_auto_or_manual) and
      (aval_mesh.empty or aval_mesh._are_all_axes_auto_or_manual)):
    aval_mesh = cur_mesh if aval_mesh.empty else aval_mesh
    out_s = NamedSharding(aval_mesh, P())
    out_s = out_s if num_out is None else [out_s] * num_out
    out_s = call_reduced_rule(
        prim, reduced_rule, out_s, num_out, *avals, **kwargs)
    out_s = call_unreduced_rule(
        prim, unreduced_rule, out_s, num_out, *avals, **kwargs)
    return out_s
  if sh_rule is None:
    raise core.ShardingTypeError(
        f'sharding rule for {prim.name} is not implemented. Please file an'
        ' issue at https://github.com/jax-ml/jax/issues. You can work around'
        ' this error by dropping that operation into full auto sharding'
        ' mode via: `jax.sharding.auto_axes(fun, out_shardings=...)`')
  out_sharding = sh_rule(*avals, **kwargs)
  out_sharding = call_reduced_rule(
      prim, reduced_rule, out_sharding, num_out, *avals, **kwargs)
  out_sharding = call_unreduced_rule(
      prim, unreduced_rule, out_sharding, num_out, *avals, **kwargs)
  return out_sharding

def call_shape_dtype_sharding_rule(
    prim, shape_rule, dtype_rule, sharding_rule, unreduced_rule, reduced_rule,
    multi_out, *avals, **kwargs):
  out_shapes = shape_rule(*avals, **kwargs)
  out_dtypes = dtype_rule(*avals, **kwargs)
  num_out = len(out_shapes) if multi_out else None
  try:
    out_shardings = call_sharding_rule(
        prim, sharding_rule, unreduced_rule, reduced_rule, num_out,
        *avals, **kwargs)
  except DuplicateSpecError as e:
    if multi_out:
      raise
    avals_str = ', '.join(i.str_short(short_dtypes=True) for i in avals)
    mesh = mesh_lib.empty_abstract_mesh if e.mesh is None else e.mesh
    out_aval_str = core.str_short_aval(
        out_shapes, out_dtypes, mesh, e.pspec, frozenset(),
        core.MemorySpace.Device, short_dtypes=True)
    raise core.ShardingTypeError(
        f'{prim} operation with inputs: {avals_str} produces an illegally'
        f' sharded result: {out_aval_str}') from e
  return out_shapes, out_dtypes, out_shardings

def _default_memory_space_rule(prim, *avals, **kwargs):
  if all(a.memory_space == core.MemorySpace.Any for a in avals):
    return core.MemorySpace.Any
  prev_aval = None
  for a in avals:
    if not a.ndim:
      continue
    if prev_aval is not None and prev_aval.memory_space != a.memory_space:
      raise ValueError(
          f'memory_space of all inputs passed to `{prim.name}` must be the'
          f' same. Got one operand with type: {prev_aval.str_short()} and'
          f' another operand with type: {a.str_short()}')
    prev_aval = a
  if prev_aval is None:
    return core.MemorySpace.Device
  return prev_aval.memory_space

def multi_mem_space_rule(prim, num_out, *avals, **kwargs):
  out_mem_space = _default_memory_space_rule(prim, *avals, **kwargs)
  return [out_mem_space] * num_out


def standard_abstract_eval(
    prim, shape_rule, dtype_rule, weak_type_rule, sharding_rule, vma_rule,
    unreduced_rule, reduced_rule, memory_space_rule, *avals, **kwargs):
  assert not prim.multiple_results
  for a in avals:
    if isinstance(a, state.AbstractRef):
      raise ValueError(f'Attempting to pass a Ref {a} to a primitive: '
                       f'{prim} -- did you forget to unpack ([...]) the ref?')
    if not isinstance(a, core.ShapedArray):
      raise ValueError(f'Attempting to pass an unexpected type {a} to a '
                       f'primitive: {prim}')
  weak_type = weak_type_rule(*avals, **kwargs)
  least_specialized = type(max(avals, key=_get_array_abstraction_level))
  if least_specialized is core.ShapedArray:
    core.check_avals_context_mesh(avals, prim.name)
    out_shape, out_dtype, out_sharding = call_shape_dtype_sharding_rule(
        prim, shape_rule, dtype_rule, sharding_rule, unreduced_rule,
        reduced_rule, False, *avals, **kwargs)
    out_vma = vma_rule(*avals, **kwargs)
    out_mem_space = (_default_memory_space_rule(prim, *avals, **kwargs)
                     if memory_space_rule is None else
                     memory_space_rule(*avals, **kwargs))
    out_aval = core.ShapedArray(
        out_shape, out_dtype, weak_type=weak_type, sharding=out_sharding,
        vma=out_vma, memory_space=out_mem_space)
    core.check_avals_context_mesh([out_aval], prim.name)
    return out_aval
  else:
    raise TypeError(avals, least_specialized)

def standard_multi_result_abstract_eval(
    prim, shape_rule, dtype_rule, weak_type_rule, sharding_rule, vma_rule,
    unreduced_rule, reduced_rule, *avals, **kwargs):
  assert prim.multiple_results
  assert all(isinstance(aval, core.ShapedArray) for aval in avals), avals
  least_specialized = max(map(type, avals), key=_get_array_abstraction_level)
  weak_types = weak_type_rule(*avals, **kwargs)
  if least_specialized is core.ShapedArray:
    core.check_avals_context_mesh(avals, prim.name)
    out_shapes, out_dtypes, out_shardings = call_shape_dtype_sharding_rule(
        prim, shape_rule, dtype_rule, sharding_rule, unreduced_rule,
        reduced_rule, True, *avals, **kwargs)
    out_vmas = vma_rule(*avals, **kwargs)
    out_mem_spaces = multi_mem_space_rule(prim, len(out_shapes), *avals, **kwargs)
    if isinstance(weak_types, bool):
      weak_types = (weak_types,) * len(out_shapes)
    out_avals = [core.ShapedArray(s, d, weak_type=weak_type, sharding=sh,
                                  vma=vma, memory_space=ms)
                 for s, d, weak_type, sh, vma, ms in zip(
                     out_shapes, out_dtypes, weak_types, out_shardings,
                     out_vmas, out_mem_spaces)]
    core.check_avals_context_mesh(out_avals, prim.name)
    return out_avals
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

_int32_max = np.iinfo(np.int32).max  # pyrefly: ignore[no-matching-overload]  # pyrefly#2398
_uint32_max = np.iinfo(np.uint32).max  # pyrefly: ignore[no-matching-overload]  # pyrefly#2398

def int_dtype_for_dim(d: DimSize, *, signed: bool) -> DType:
  """Returns a integer dtype large enough to contain indices in dimension d."""
  if signed:
    if not core.is_constant_dim(d):
      return dtypes.default_int_dtype()
    return np.dtype(np.int64) if d > _int32_max else np.dtype(np.int32)
  else:
    if not core.is_constant_dim(d):
      return dtypes.default_uint_dtype()
    return np.dtype(np.uint64) if d > _uint32_max else np.dtype(np.uint32)

def int_dtype_for_shape(shape: Shape, *, signed: bool) -> DType:
  """Returns a integer dtype large enough to contain indices in `shape`."""
  if signed:
    for d in shape:
      if core.is_constant_dim(d):
        if d > _int32_max:
          return np.dtype(np.int64)
      else:
        return dtypes.default_int_dtype()
    return np.dtype(np.int32)
  else:
    for d in shape:
      if core.is_constant_dim(d):
        if d > _uint32_max:
          return np.dtype(np.uint64)
      else:
        return dtypes.default_uint_dtype()
    return np.dtype(np.uint32)
