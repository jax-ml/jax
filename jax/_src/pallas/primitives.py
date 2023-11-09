# Copyright 2023 The JAX Authors.
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

"""Module for pallas-specific JAX primitives and functions."""
from __future__ import annotations
import enum
import functools

from typing import Any

import jax
from jax import lax
from jax import tree_util
from jax._src import ad_util
from jax._src import core as jax_core
from jax._src import pretty_printer as pp
from jax._src import state
from jax._src.util import (safe_map, safe_zip)
from jax._src.state import primitives as state_primitives
from jax._src.state import discharge as state_discharge
from jax.interpreters import ad
from jax.interpreters import mlir
from jax.interpreters import xla
import jax.numpy as jnp

from jax._src.pallas import core as pallas_core
from jax._src.pallas import indexing

# TODO(sharadmv): enable type checking
# mypy: ignore-errors

partial = functools.partial
Slice = indexing.Slice
NDIndexer = indexing.NDIndexer

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

program_id_p = jax_core.Primitive("program_id")

def program_id(axis):
  return program_id_p.bind(axis=axis)

def program_id_bind(*, axis: int):
  grid_env = pallas_core.current_grid_env()
  if grid_env:
    return grid_env[axis].axis_index
  return jax_core.Primitive.bind(program_id_p, axis=axis)
program_id_p.def_custom_bind(program_id_bind)

def _program_id_impl(*, axis: int):
  grid_env = pallas_core.current_grid_env()
  return grid_env[axis].axis_index
program_id_p.def_impl(_program_id_impl)

mlir.register_lowering(program_id_p, functools.partial(xla.apply_primitive,
                                                       program_id_p))

def _program_id_abstract_eval(**_):
  return jax_core.ShapedArray((), jnp.int32)
program_id_p.def_abstract_eval(_program_id_abstract_eval)

class AtomicOpType(enum.Enum):
  XCHG = "xchg"
  ADD = "add"
  MAX = "max"
  MIN = "min"
  AND = "and"
  OR = "or"
  XOR = "xor"

atomic_rmw_p = jax_core.Primitive("atomic_rmw")


def _atomic_rmw_discharge_rule(
    in_avals, out_avals, *args_flat, args_tree, atomic_type: AtomicOpType
):
  del out_avals  # Unused.
  ref, idx, val, mask = args_tree.unflatten(args_flat)

  if mask is not None:
    raise NotImplementedError

  if atomic_type == AtomicOpType.ADD:
    monoid = lambda x, y: x + y
  elif atomic_type == AtomicOpType.MAX:
    monoid = jnp.maximum
  elif atomic_type == AtomicOpType.MIN:
    monoid = jnp.minimum
  else:
    raise NotImplementedError(atomic_type)

  if all((isinstance(s, Slice) or not s.shape) for s in idx.indices):
    indices = idx.indices
    scalar_dims = [not isinstance(s, Slice) and s.shape == () for s in indices]
    slice_starts = [s.start if isinstance(s, Slice) else s for s in indices]
    slice_sizes = tuple(s.size if isinstance(s, Slice) else 1 for s in indices)
    out_ones = lax.dynamic_slice(ref, slice_starts, slice_sizes=slice_sizes)
    val_indexer = tuple(None if scalar else slice(None) for scalar in scalar_dims)
    val = val[val_indexer]
    val = monoid(val, out_ones)
    x_new = lax.dynamic_update_slice(ref, val, start_indices=slice_starts)
    out_indexer = tuple(0 if scalar else slice(None) for scalar in scalar_dims)
    out = out_ones[out_indexer]
  elif all(not isinstance(s, Slice) for s in idx.indices):
    out = ref[idx.indices]
    x_new = ref.at[idx.indices].set(monoid(out, val))
  else:
    raise NotImplementedError
  return (x_new,) + (None,) * (len(in_avals) - 1), out


state_discharge.register_discharge_rule(atomic_rmw_p)(_atomic_rmw_discharge_rule)


def _atomic_abstract_eval(*avals_flat, args_tree, atomic_type: AtomicOpType):
  ref, _, _, _ = args_tree.unflatten(avals_flat)
  if ref.dtype == jnp.dtype("float16") and atomic_type != AtomicOpType.ADD:
    raise ValueError(f"`atomic_{atomic_type.value}` does not support f16.")
  if ref.dtype in {
      jnp.dtype("bool"),
      jnp.dtype("int8"),
      jnp.dtype("int16"),
      jnp.bfloat16,
  }:
    raise ValueError(
        f"`atomic_{atomic_type.value}` does not support {ref.dtype}."
    )
  return _swap_abstract_eval(*avals_flat, args_tree=args_tree)


atomic_rmw_p.def_effectful_abstract_eval(_atomic_abstract_eval)

def atomic_rmw(x_ref, idx, val, *, mask: Any | None = None,
               atomic_type: AtomicOpType):
  idx = NDIndexer.from_indices_shape(idx, x_ref.shape)
  args_flat, args_tree = tree_util.tree_flatten((x_ref, idx, val, mask))
  return atomic_rmw_p.bind(
      *args_flat, args_tree=args_tree, atomic_type=atomic_type
  )

atomic_xchg = functools.partial(atomic_rmw, atomic_type=AtomicOpType.XCHG)
atomic_add = functools.partial(atomic_rmw, atomic_type=AtomicOpType.ADD)
atomic_max = functools.partial(atomic_rmw, atomic_type=AtomicOpType.MAX)
atomic_min = functools.partial(atomic_rmw, atomic_type=AtomicOpType.MIN)
atomic_and = functools.partial(atomic_rmw, atomic_type=AtomicOpType.AND)
atomic_or = functools.partial(atomic_rmw, atomic_type=AtomicOpType.OR)
atomic_xor = functools.partial(atomic_rmw, atomic_type=AtomicOpType.XOR)

atomic_cas_p = jax_core.Primitive("atomic_cas")

def _atomic_cas_abstract_eval(ref_aval, cmp_aval, val_aval):
  if cmp_aval.dtype != val_aval.dtype:
    raise ValueError("Dtypes in cmp/val need to match")
  if ref_aval.shape:
    raise ValueError("Ref must be scalar.")
  if cmp_aval.shape:
    raise ValueError("Cmp must be scalar.")
  if val_aval.shape:
    raise ValueError("Val must be scalar.")
  if cmp_aval.shape != val_aval.shape:
    raise ValueError("Dtypes in cmp/val need to match")
  return jax_core.ShapedArray(val_aval.shape, val_aval.dtype), {state.WriteEffect(0)}
atomic_cas_p.def_effectful_abstract_eval(_atomic_cas_abstract_eval)

def atomic_cas(ref, cmp, val):
  return atomic_cas_p.bind(ref, cmp, val)

@state_discharge.register_discharge_rule(atomic_cas_p)
def _atomic_cas_discharge_rule(in_avals, out_avals, ref, cmp, val):
  del in_avals, out_avals
  new_val = jnp.where(ref == cmp, val, ref)
  return (new_val, None, None), ref

max_contiguous_p = jax_core.Primitive("max_contiguous")

max_contiguous_p.def_impl(lambda x, **_: x)
mlir.register_lowering(max_contiguous_p, lambda _, x, **__: [x])

def max_contiguous(x, values):
  if not isinstance(values, list):
    values = [values]
  return max_contiguous_p.bind(x, values=values)

def _max_contiguous_abstract_eval(aval, **_):
  return aval
max_contiguous_p.def_abstract_eval(_max_contiguous_abstract_eval)

multiple_of_p = jax_core.Primitive("multiple_of")

multiple_of_p.def_impl(lambda x, **_: x)
mlir.register_lowering(multiple_of_p, lambda _, x, **__: [x])

def multiple_of(x, values):
  if not isinstance(values, list):
    values = [values]
  return multiple_of_p.bind(x, values=values)

def _multiple_of_abstract_eval(aval, **_):
  return aval
multiple_of_p.def_abstract_eval(_multiple_of_abstract_eval)

load_p = jax_core.Primitive('masked_load')


def _load_abstract_eval(*avals_flat, args_tree, **_):
  ref, idx, _, _ = args_tree.unflatten(avals_flat)
  return (
      jax_core.ShapedArray(idx.get_indexer_shape(), ref.dtype),
      {state.ReadEffect(0)},
  )


load_p.def_effectful_abstract_eval(_load_abstract_eval)

def _pp_dslice(dim: int, slice: Slice, context):
  size = pp.text(str(slice.size))
  if isinstance(slice.start, int):
    if slice.start == 0:
      start = pp.text("")
    else:
      start = pp.text(str(slice.start))
    if slice.size == dim:
      end = pp.text("")
    else:
      end = pp.text(str(slice.start + slice.size))
  else:
    start = pp.text(jax_core.pp_var(slice.start, context))
    end = pp.concat([start, pp.text("+"), size])
  return pp.concat([start, pp.text(":"), end])

def _pp_idx(ref_aval, idx: NDIndexer, context):
  docs = [
      _pp_dslice(d, s, context) if isinstance(s, Slice)
      else pp.text(jax_core.pp_var(s, context))
      for s, d in zip(idx.indices, ref_aval.shape)]
  if not docs:
    return pp.text("")
  doc = [docs[0]]
  for d in docs[1:]:
    doc.append(pp.text(","))
    doc.append(d)
  return pp.concat(doc)

def _load_pp_rule(eqn, context, settings):
  # Pretty prints `a = load x i` as `x[i] <- a`
  y, = eqn.outvars
  x, idx, _, _ = eqn.params["args_tree"].unflatten(eqn.invars)
  idx = _pp_idx(eqn.invars[0].aval, idx, context)
  lhs = jax_core.pp_vars([y], context, print_shapes=settings.print_shapes)
  return pp.concat([lhs, pp.text(' <- '), state_primitives.pp_ref(pp.concat([
    pp.text(jax_core.pp_var(x, context)), pp.text('['), idx, pp.text(']')
    ]))])
jax_core.pp_eqn_rules[load_p] = _load_pp_rule


def _load_jvp(primals, tangents, args_tree, **params):
  ref_primal, idx, mask, other_primal = args_tree.unflatten(primals)
  ref_tangent, _, _, other_tangent = args_tree.unflatten(tangents)
  if other_tangent is not None:
    other_tangent = ad_util.instantiate(other_tangent)
  return (
      load_p.bind(
          *tree_util.flatten(ref_primal, idx, mask, other_primal),
          args_tree=args_tree,
          **params,
      ),
      load_p.bind(
          *tree_util.flatten(ref_tangent, idx, mask, other_tangent),
          args_tree=args_tree,
          **params,
      ),
  )


ad.primitive_jvps[load_p] = _load_jvp


def _load_discharge_rule(in_avals, out_avals, *args_flat, args_tree, **_):
  del out_avals  # Unused.
  ref, idx, mask, other = args_tree.unflatten(args_flat)
  if all((isinstance(s, Slice) or not s.shape) for s in idx.indices):
    indices = idx.indices
    scalar_dims = [not isinstance(s, Slice) and s.shape == () for s in indices]
    slice_starts = [s.start if isinstance(s, Slice) else s for s in indices]
    slice_sizes = tuple(s.size if isinstance(s, Slice) else 1 for s in indices)
    out_ones = lax.dynamic_slice(ref, slice_starts, slice_sizes=slice_sizes)
    out_indexer = tuple(0 if scalar else slice(None) for scalar in scalar_dims)
    out = out_ones[out_indexer]
  elif all(not isinstance(s, Slice) for s in idx.indices):
    out = ref[idx.indices]
  else:
    raise NotImplementedError
  if mask is not None and other is not None:
    out = jnp.where(mask, out, other)
  return (None,) * len(in_avals), out


state_discharge.register_discharge_rule(load_p)(_load_discharge_rule)

swap_p = jax_core.Primitive('masked_swap')


def _swap_abstract_eval(*avals_flat, args_tree, **_):
  ref, idx, val, _ = args_tree.unflatten(avals_flat)
  expected_output_shape = idx.get_indexer_shape()
  if expected_output_shape != val.shape:
    raise ValueError(
        f"Invalid shape for `swap`. Ref shape: {ref.shape}. "
        f"Value shape: {val.shape}. Indices: {idx}. "
    )
  if ref.dtype != val.dtype:
    raise ValueError(
        f"Invalid dtype for `swap`. Ref dtype: {ref.dtype}. "
        f"Value dtype: {val.dtype}. "
    )
  return (
      jax_core.ShapedArray(expected_output_shape, ref.dtype),
      {state.WriteEffect(0)},
  )


swap_p.def_effectful_abstract_eval(_swap_abstract_eval)

def _swap_pp_rule(eqn, context, settings):
  # Pretty prints `a = swap x v i` as `a, x[i] <- x[i], v`
  # or:
  # Pretty prints `_ = swap x v i` as `x[i] <- v`
  y, = eqn.outvars
  x, idx, val, _ = eqn.params["args_tree"].unflatten(eqn.invars)
  idx = _pp_idx(eqn.invars[0].aval, idx, context)
  x_i = pp.concat([pp.text(jax_core.pp_var(x, context)),
                   pp.text('['), idx, pp.text(']')])
  if isinstance(y, jax_core.DropVar):
    return pp.concat([state_primitives.pp_ref(
      x_i), pp.text(" <- "), pp.text(jax_core.pp_var(val, context))])
  y = jax_core.pp_vars([y], context, print_shapes=settings.print_shapes)
  return pp.concat([y, pp.text(', '), state_primitives.pp_ref(x_i),
                    pp.text(' <- '), state_primitives.pp_ref(x_i),
                    pp.text(', '), pp.text(jax_core.pp_var(val, context))])
jax_core.pp_eqn_rules[swap_p] = _swap_pp_rule


def _swap_jvp(primals, tangents, *, args_tree, **params):
  ref_primal, idx, val_primal, mask = args_tree.unflatten(primals)
  ref_tangent, _, val_tangent, _ = args_tree.unflatten(tangents)
  val_tangent = ad_util.instantiate(val_tangent)
  return (
      swap_p.bind(
          *tree_util.flatten(ref_primal, idx, val_primal, mask),
          args_tree=args_tree,
          **params,
      ),
      swap_p.bind(
          *tree_util.flatten(ref_tangent, idx, val_tangent, mask),
          args_tree=args_tree,
          **params,
      ),
  )


ad.primitive_jvps[swap_p] = _swap_jvp


def _swap_discharge_rule(in_avals, out_avals, *args_flat, args_tree, **_):
  del out_avals  # Unused.
  ref, idx, val, mask = args_tree.unflatten(args_flat)
  if all((isinstance(s, Slice) or not s.shape) for s in idx.indices):
    indices = idx.indices
    scalar_dims = [
        i
        for i, s in enumerate(indices)
        if not isinstance(s, Slice) and not s.shape
    ]
    slice_starts = [s.start if isinstance(s, Slice) else s for s in indices]
    slice_sizes = tuple(s.size if isinstance(s, Slice) else 1 for s in indices)
    out = lax.dynamic_slice(ref, slice_starts, slice_sizes=slice_sizes)
    out = jnp.squeeze(out, scalar_dims)
    if mask is not None:
      out_ = out
      out = jnp.where(mask, out, val)
      val = jnp.where(mask, val, out_)
    val = jnp.expand_dims(val, scalar_dims)
    x_new = lax.dynamic_update_slice(ref, val, start_indices=slice_starts)
  elif all(not isinstance(s, Slice) for s in idx.indices):
    out = ref[idx.indices]
    if mask is not None:
      out_ = out
      out = jnp.where(mask, out, val)
      val = jnp.where(mask, val, out_)
    x_new = ref.at[idx.indices].set(val)
  else:
    raise NotImplementedError
  return (x_new,) + (None,) * (len(in_avals) - 1), out


state_discharge.register_discharge_rule(swap_p)(_swap_discharge_rule)


def load(x_ref, idx, *, mask=None, other=None, cache_modifier="",
         eviction_policy="", volatile=False):
  idx = NDIndexer.from_indices_shape(idx, x_ref.shape)
  args_flat, args_tree = tree_util.tree_flatten((x_ref, idx, mask, other))
  return load_p.bind(
      *args_flat,
      args_tree=args_tree,
      cache_modifier=cache_modifier,
      eviction_policy=eviction_policy,
      is_volatile=volatile,
  )

def swap(x_ref, idx, val, *, mask=None, eviction_policy="") -> Any:
  idx = NDIndexer.from_indices_shape(idx, x_ref.shape)
  args_flat, args_tree = tree_util.tree_flatten((x_ref, idx, val, mask))
  return swap_p.bind(
      *args_flat, args_tree=args_tree, eviction_policy=eviction_policy
  )

def store(x_ref, idx, val, *, mask=None, eviction_policy="") -> None:
  _ = swap(x_ref, idx, val, mask=mask, eviction_policy=eviction_policy)

def dot(a, b, trans_a: bool = False, trans_b: bool = False,
        allow_tf32: bool | None = None, precision=None):
  if (a.ndim != 2) or (b.ndim != 2):
    raise ValueError("`a` and `b` must be 2D arrays.")
  lhs_contract_dim = 0 if trans_a else 1
  rhs_contract_dim = 0 if not trans_b else 1
  if allow_tf32 is not None:
    if precision is not None:
      raise ValueError("Only one of allow_tf32 and precision can be specified")
    precision = lax.Precision.HIGH if allow_tf32 else lax.Precision.HIGHEST
  return jax.lax.dot_general(
      a,
      b,
      dimension_numbers=(((lhs_contract_dim,), (rhs_contract_dim,)), ((), ())),
      precision=precision,
      preferred_element_type=jnp.float32,
  )
