# Copyright 2022 The JAX Authors.
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
"""Module for state primitives."""
from __future__ import annotations

from functools import partial
from typing import Any, Union

import numpy as np


from jax._src import ad_util
from jax._src import core
from jax._src import pretty_printer as pp
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import lax
from jax._src.typing import Array
from jax._src.state.types import (AbstractRef, ReadEffect, WriteEffect,
                                  AccumEffect)
from jax._src.util import safe_map, safe_zip, tuple_insert


## General utilities

## JAX utilities

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

## get/swap/addupdate implementations

# `get` reads a value from a `Ref` type, a.k.a.:
# a = get_p.bind(x)
# or we can read using indices:
# a = get_p.bind(x, 0, 1)
# Staging out `a = get_p.bind(x)` where the aval of `x` is
# `Ref((3,), np.dtype('float32'))` leads to a jaxpr eqn printed like
#   a:f32[3] <- x[]
get_p = core.Primitive("get")

def _get_impl(ref: AbstractRef, *idx: int, **_):
  del ref, idx
  raise ValueError("Cannot run stateful primitive.")
get_p.def_impl(_get_impl)

Indexer = tuple[Union[int, slice, Array], ...]
# or Ellipsis, but that can't be annotated until Python 3.10? (types.EllipsisType)

def _is_trivial_indexer(idx: Indexer) -> bool:
  if idx is ...:
    return True
  if type(idx) is tuple:
    if len(idx) == 0:
      return True
    return len(idx) == 1 and idx[0] is ...
  return False

def _unpack_idx(idx: Indexer, ndim: int
               ) -> tuple[tuple[Array, ...], tuple[bool, ...]]:
  if _is_trivial_indexer(idx):
    idx = tuple(slice(None) for _ in range(ndim))
  indexed_dims_ = []
  non_slice_idx = []
  for i in idx:
    if isinstance(i, slice):
      if i.start is not None or i.stop is not None or i.step is not None:
        raise NotImplementedError("Reference indexing only supports trivial slices")
      indexed_dims_.append(False)
    else:
      non_slice_idx.append(i)
      indexed_dims_.append(True)
  indexed_dims = indexed_dims_ + [False] * (ndim - len(indexed_dims_))
  import jax.numpy as jnp
  return (tuple(map(jnp.int32, non_slice_idx)), tuple(indexed_dims))

def _get_slice_output_shape(in_shape: tuple[int, ...],
                            idx_shapes: tuple[tuple[int, ...], ...],
                            indexed_dims: tuple[bool, ...]) -> tuple[int, ...]:
  shape_suffix = [d for i, d in zip(indexed_dims, in_shape) if not i]
  shape_prefix, = set(idx_shapes) or [()]  # tie fighter
  # Move shape prefix dimensions to the front
  shape = (*shape_prefix, *shape_suffix)
  return shape

def _get_indexer(ref: AbstractRef, idx: Indexer
                ) -> tuple[Indexer, tuple[bool, ...]]:
  if isinstance(ref.inner_aval, core.ShapedArray):
    non_slice_idx, indexed_dims = _unpack_idx(idx, ref.ndim)
  else:
    if not _is_trivial_indexer(idx):
      raise ValueError(
          f"Cannot use nontrivial slice on non-shaped `Ref`: {idx}.")
    non_slice_idx, indexed_dims = (), ()
  return non_slice_idx, indexed_dims

def ref_get(ref: Any, idx: Indexer) -> Array:
  """Reads a value from a `Ref`, a.k.a. value <- ref[idx]."""
  ref_aval = core.get_aval(ref)
  if not isinstance(ref_aval, AbstractRef):
    raise ValueError(f"Can only call `get` on a `Ref`: {ref}")
  non_slice_idx, indexed_dims = _get_indexer(ref, idx)
  return get_p.bind(ref, *non_slice_idx, indexed_dims=indexed_dims)

# `swap` mutates a `Ref`, setting its value and returns its previous value.
# b = swap_p.bind(x, a)
# It generalizes the setting operation for a `Ref` as we can ignore the return
# value:
# _ = swap_p.bind(x, a)
# `swap_p` also takes in index arguments following the value, i.e.:
# _ = swap_p.bind(x, a, 0, 1)
# Staging out `b = swap_p.bind(x, a)` where the aval of `x` is
# `Ref((3,), np.dtype('float32'))` and the aval of `a` is
# `ShapedArray((3,), np.dtype('float32'))` leads to a jaxpr eqn printed like
#   b:f32[3], x:Ref{f32[3]} <- x, a
# Staging out `_ = swap_p.bind(x, a, i, j)` where the aval of `x` is
# `Ref((3,), np.dtype('float32'))` , the aval of `a` is
# `ShapedArray((3,), np.dtype('float32'))`, and the avals of both `i` and `j`
# are `ShapedArray((), np.dtype('int32'))` leads to a jaxpr eqn printed like
#   x:Ref{f32[3]}[i, j] <- a
swap_p = core.Primitive("swap")

def _swap_impl(ref: AbstractRef, value: Array, *idx: int, **_):
  del ref, value, idx
  raise ValueError("Cannot run stateful primitive.")
swap_p.def_impl(_swap_impl)

def ref_swap(ref: AbstractRef, idx: Indexer, value: Array) -> Array:
  """Sets a `Ref`'s value and returns the original value."""
  ref_aval = core.get_aval(ref)
  if not isinstance(ref_aval, AbstractRef):
    raise ValueError(f"Can only call `swap` on a `Ref`: {ref}")
  non_slice_idx, indexed_dims = _get_indexer(ref, idx)
  return swap_p.bind(ref, value, *non_slice_idx, indexed_dims=indexed_dims)

def ref_set(ref: AbstractRef, idx: Indexer, value: Array) -> None:
  """Sets a `Ref`'s value, a.k.a. ref[idx] <- value."""
  ref_swap(ref, idx, value)

# `addupdate_p` mutates a `Ref`, adding a value to its existing value.
# Semantically,
# ```
# addupdate ref a *idx
# ```
# is equivalent to
# ```
# b = get ref *idx
# c = add b x
# _ = swap ref c *idx
# ```
addupdate_p = core.Primitive('addupdate')
addupdate_p.multiple_results = True

def _addupdate_impl(ref: AbstractRef, value: Array, *idx: int):
  del ref, idx, value
  raise ValueError("Can't evaluate `addupdate` outside a stateful context.")
addupdate_p.def_impl(_addupdate_impl)

def ref_addupdate(ref: AbstractRef, idx: Indexer, x: Array) -> None:
  """Mutates a ref with an additive update i.e. `ref[idx] += x`."""
  ref_aval = core.get_aval(ref)
  if not isinstance(ref_aval, AbstractRef):
    raise ValueError(f"Can only call `addupdate` on a `Ref`: {ref}")
  non_slice_idx, indexed_dims = _get_indexer(ref, idx)
  return addupdate_p.bind(ref, x, *non_slice_idx, indexed_dims=indexed_dims)

## get/set/addupdate abstract evaluation rules

def _get_abstract_eval(ref_aval: AbstractRef, *idx,
                       indexed_dims):
  if not isinstance(ref_aval, AbstractRef):
    raise ValueError(f"`get` must be called on `Ref` types: {ref_aval}.")
  if isinstance(ref_aval.inner_aval, core.ShapedArray):
    if not isinstance(ref_aval.inner_aval, core.ShapedArray):
      raise ValueError("`get` with nontrivial indexing must be called "
                       f"on `ShapedArray` `Ref`: {ref_aval}.")
    if len(indexed_dims) != len(ref_aval.shape):
      raise ValueError("`indexed_dims` must be the same length as `Ref` shape.")
    if sum(indexed_dims) != len(idx):
      raise ValueError(f"Invalid `idx` and `indexed_dims`: {idx}, {indexed_dims}")
    idx_shapes = tuple(i.shape for i in idx)
    shape = _get_slice_output_shape(ref_aval.shape, idx_shapes, indexed_dims)
    out_aval = ref_aval.inner_aval.update(shape=shape)
  else:
    if idx:
      raise ValueError("Cannot index non-shaped array with nontrivial indices.")
    out_aval = ref_aval.inner_aval
  return (out_aval, {ReadEffect(0)})
get_p.def_effectful_abstract_eval(_get_abstract_eval)

def _swap_abstract_eval(ref_aval: AbstractRef,
                        val_aval: core.AbstractValue,
                        *idx: core.ShapedArray, indexed_dims: tuple[bool]):
  out_aval: core.AbstractValue
  if not isinstance(ref_aval, AbstractRef):
    raise ValueError(f"`swap` must be called on `Ref` types: {ref_aval}.")
  if isinstance(ref_aval.inner_aval, core.ShapedArray):
    if len(indexed_dims) != len(ref_aval.shape):
      raise ValueError("`indexed_dims` must be the same length as `Ref` shape.")
    if sum(indexed_dims) != len(idx):
      raise ValueError(f"Invalid `idx` and `indexed_dims`: {idx}, {indexed_dims}")
    val_aval = core.raise_to_shaped(val_aval)
    assert isinstance(val_aval, core.ShapedArray)
    idx_shapes = tuple(i.shape for i in idx)
    expected_output_shape = _get_slice_output_shape(
        ref_aval.shape, idx_shapes, indexed_dims)
    if expected_output_shape != val_aval.shape:
      raise ValueError("Invalid shape for `swap`. "
                       f"Ref shape: {ref_aval.shape}. "
                       f"Value shape: {val_aval.shape}. "
                       f"Indices: {idx}. ")
    if ref_aval.dtype != val_aval.dtype:
      raise ValueError("Invalid dtype for `swap`. "
                       f"Ref dtype: {ref_aval.dtype}. "
                       f"Value shape: {val_aval.dtype}. ")
    out_aval = core.ShapedArray(expected_output_shape, ref_aval.dtype)
  else:
    if idx:
      raise ValueError("`swap` with nontrivial indexing must be called "
                       f"on `ShapedArray` `Ref`: {ref_aval}.")
    out_aval = ref_aval.inner_aval
  return (out_aval, {WriteEffect(0)})
swap_p.def_effectful_abstract_eval(_swap_abstract_eval)


def _addupdate_abstract_eval(ref_aval: AbstractRef,
                             val_aval: core.AbstractValue,
                             *idx: core.ShapedArray, indexed_dims: tuple[bool]):
  if not isinstance(ref_aval, AbstractRef):
    raise ValueError(f"`addupdate` must be called on `Ref` types: {ref_aval}.")
  if idx and not isinstance(ref_aval.inner_aval, core.ShapedArray):
    raise ValueError("`addupdate` with nontrivial indexing must be called "
                     f"on `ShapedArray` `Ref`: {ref_aval}.")
  if isinstance(ref_aval.inner_aval, core.ShapedArray):
    if len(indexed_dims) != len(ref_aval.shape):
      raise ValueError("`indexed_dims` must be the same length as `Ref` shape.")
    if sum(indexed_dims) != len(idx):
      raise ValueError(f"Invalid `idx` and `indexed_dims`: {idx}, {indexed_dims}")
    val_aval = core.raise_to_shaped(val_aval)
    assert isinstance(val_aval, core.ShapedArray)
    idx_shapes = tuple(i.shape for i in idx)
    slice_shape = _get_slice_output_shape(
        ref_aval.shape, idx_shapes, indexed_dims)
    if slice_shape != val_aval.shape:
      raise ValueError("Invalid shape for `addupdate`. "
                       f"Ref shape: {ref_aval.shape}. "
                       f"Value shape: {val_aval.shape}. "
                       f"Indices: {idx}. ")
    if ref_aval.dtype != val_aval.dtype:
      raise ValueError("Invalid dtype for `addupdate`. "
                       f"Ref dtype: {ref_aval.dtype}. "
                       f"Value shape: {val_aval.dtype}. ")
  elif idx:
    raise ValueError("`addupdate` with nontrivial indexing must be called "
                     f"on `ShapedArray` `Ref`: {ref_aval}.")
  return [], {AccumEffect(0)}
addupdate_p.def_effectful_abstract_eval(_addupdate_abstract_eval)

## Pretty printing for `get` and `swap` in jaxprs

pp_ref = partial(pp.color, intensity=pp.Intensity.NORMAL,
                 foreground=pp.Color.GREEN)

def _pp_idx(context, non_slice_idx, indexed_dims):
  idx_iter = iter(non_slice_idx)
  idx = ','.join(core.pp_var(next(idx_iter), context) if indexed else ':'
                 for indexed in indexed_dims)
  assert next(idx_iter, None) is None
  return pp.text(idx)

def _get_pp_rule(eqn, context, settings) -> pp.Doc:
  # Pretty prints `a = get x i` as `x[i] <- a`
  y, = eqn.outvars
  x, *idx = eqn.invars
  idx = _pp_idx(context, idx, eqn.params["indexed_dims"])
  lhs = core.pp_vars([y], context, print_shapes=settings.print_shapes)
  # TODO more general get
  return pp.concat([lhs, pp.text(' <- '), pp_ref(pp.concat([
      pp.text(core.pp_var(x, context)), pp.text('['), idx, pp.text(']')]))])
core.pp_eqn_rules[get_p] = _get_pp_rule

def _swap_pp_rule(eqn, context, settings) -> pp.Doc:
  y, = eqn.outvars
  x, v, *idx = eqn.invars
  idx = _pp_idx(context, idx, eqn.params["indexed_dims"])
  if type(y) is core.DropVar:
    # In the case of a set (ignored return value),
    # pretty print `_ = swap x v i` as `x[i] <- v`
    del y
    return pp.concat([
        pp_ref(pp.concat([
            pp.text(core.pp_var(x, context)),
            pp.text('['), idx, pp.text(']')
        ])), pp.text(' <- '), pp.text(core.pp_var(v, context))])
  else:
    # pretty-print `y:T = swap x v i` as `y:T, x[i] <- x[i], v`
    x_i = pp.concat([pp.text(core.pp_var(x, context)),
                     pp.text('['), idx, pp.text(']')])
    y = core.pp_vars([y], context, print_shapes=settings.print_shapes)
    return pp.concat([y, pp.text(', '), pp_ref(x_i), pp.text(' <- '),
                      pp_ref(x_i), pp.text(', '),
                      pp.text(core.pp_var(v, context))])
core.pp_eqn_rules[swap_p] = _swap_pp_rule

def _addupdate_pp_rule(eqn, context, settings) -> pp.Doc:
  # pretty-print ` = addupdate x i v` as `x[i] += v`
  () = eqn.outvars
  x, v, *idx = eqn.invars
  idx = _pp_idx(context, idx, eqn.params["indexed_dims"])
  return pp.concat([
    pp_ref(pp.concat([
        pp.text(core.pp_var(x, context)),
        pp.text('['), idx, pp.text(']')
    ])), pp.text(' += '), pp.text(core.pp_var(v, context))])
core.pp_eqn_rules[addupdate_p] = _addupdate_pp_rule

## get/swap/addupdate JVP rules

def _get_jvp(primals: list[Any], tangents: list[Any], **params: Any):
  ref_primal, *idx = primals
  assert isinstance(ref_primal.aval, AbstractRef)
  ref_tangent, *_ = tangents
  assert isinstance(ref_tangent.aval, AbstractRef)
  return (get_p.bind(ref_primal, *idx, **params),
          get_p.bind(ref_tangent, *idx, **params))  # type: ignore[arg-type]
ad.primitive_jvps[get_p] = _get_jvp

def _swap_jvp(primals: list[Any], tangents: list[Any], **params: Any):
  ref_primal, x_primal, *idx = primals
  assert isinstance(ref_primal.aval, AbstractRef)
  ref_tangent, x_tangent, *_ = tangents
  assert isinstance(ref_tangent.aval, AbstractRef)
  x_tangent = ad_util.instantiate(x_tangent)
  return (swap_p.bind(ref_primal, x_primal, *idx, **params),  # type: ignore[arg-type]
          swap_p.bind(ref_tangent, x_tangent, *idx, **params))  # type: ignore[arg-type]
ad.primitive_jvps[swap_p] = _swap_jvp

def addupdate_jvp_rule(primals: list[Any], tangents: list[Any], **params: Any):
  ref_primal, x_primal, *idx = primals
  ref_tangent, x_tangent, *_ = tangents
  x_tangent = ad_util.instantiate(x_tangent)
  addupdate_p.bind(ref_primal, x_primal, *idx, **params)
  addupdate_p.bind(ref_tangent, x_tangent, *idx, **params)
  return [], []
ad.primitive_jvps[addupdate_p] = addupdate_jvp_rule

##  get/swap/addupdate transpose rules

def _get_transpose(g, ref, *idx, **params):
  # get transpose is addupdate
  if type(g) is not ad_util.Zero:
    addupdate_p.bind(ref, g, *idx, **params)
  return [None] + [None] * len(idx)
ad.primitive_transposes[get_p] = _get_transpose

def _swap_transpose(g, ref, x, *idx, **params):
  # swap transpose is swap
  x_bar = swap_p.bind(ref, ad_util.instantiate(g), *idx, **params)
  return [None, x_bar] + [None] * len(idx)
ad.primitive_transposes[swap_p] = _swap_transpose

def addupdate_transpose(cts_in, ref, x, *idx, **params):
  # addupdate transpose is get
  del cts_in, x
  g = get_p.bind(ref, *idx, **params)
  return [None, g] + [None] * len(idx)
ad.primitive_transposes[addupdate_p] = addupdate_transpose

## get/swap/addupdate partial_eval_custom rules

def _state_partial_eval_custom(prim, saveable, unks_in, inst_in, eqn):
  if any(unks_in):
    res = [v for v, inst in zip(eqn.invars, inst_in) if not inst]
    return None, eqn, [True] * len(eqn.outvars), [True] * len(eqn.outvars), res
  elif saveable(prim, *[var.aval for var in eqn.invars], **eqn.params):
    return eqn, None, [False] * len(eqn.outvars), [False] * len(eqn.outvars), []
  res = [v for v, inst in zip(eqn.invars, inst_in) if not inst]
  return eqn, eqn, [False] * len(eqn.outvars), [True] * len(eqn.outvars), res

pe.partial_eval_jaxpr_custom_rules[get_p] = partial(_state_partial_eval_custom,
                                                    get_p)
pe.partial_eval_jaxpr_custom_rules[swap_p] = partial(_state_partial_eval_custom,
                                                     swap_p)
pe.partial_eval_jaxpr_custom_rules[addupdate_p] = partial(
    _state_partial_eval_custom, addupdate_p)

##  get/swap/addupdate batching rules

def _output_bdim(indexed_dims: tuple[bool, ...], ref_dim: int,
                 idxs_shape: tuple[int, ...]):
  num_idxs_to_left = sum(indexed_dims[:ref_dim])
  return ref_dim - num_idxs_to_left + len(idxs_shape)

def _get_vmap(batched_args, batched_dims, *, indexed_dims):
  axis_size, = {x.shape[d] for x, d in zip(batched_args, batched_dims)
                if d is not batching.not_mapped}
  ref, *idxs = batched_args
  ref_dim, *idx_dims = batched_dims

  ref_is_batched = ref_dim is not batching.not_mapped
  idx_is_batched = any(i_dim is not batching.not_mapped for i_dim in idx_dims)
  bdim_out = 0

  if idx_is_batched:
    # If at least one of the idx is batched, we broadcast them all and move the
    # batch dim to the front.
    idxs = tuple(batching.bdim_at_front(i, d, axis_size) for i, d
                 in zip(idxs, idx_dims))
  idxs_shape, = {i.shape for i in idxs} or [()]
  if ref_is_batched:
    # If ref is batched, we are doing a `get` with an additional axis. If `idxs`
    # are also batched, then we are indexing into the batch axis with an `iota`.
    indexed_dims = tuple_insert(indexed_dims, ref_dim, idx_is_batched)
    if idx_is_batched:
      # If we have batched idx, we need to insert the new iota index. The place
      # where we add in the new `iota` index is `ref_dim` so we need to compute
      # what `ref_dim` *would be* if we inserted it into `idxs` instead, because
      # `idxs` doesn't include the non indexed dims.
      idx_place = [i for i, i_dim in enumerate(indexed_dims)
                   if i_dim].index(ref_dim)
      iota = lax.broadcasted_iota(np.dtype('int32'), idxs_shape, 0)
      idxs = tuple_insert(idxs, idx_place, iota)
    else:
      bdim_out = _output_bdim(indexed_dims, ref_dim, idxs_shape)
  return get_p.bind(ref, *idxs, indexed_dims=indexed_dims), bdim_out
batching.primitive_batchers[get_p] = _get_vmap

def _swap_vmap(batched_args, batched_dims, *, indexed_dims):
  axis_size, = {x.shape[d] for x, d in zip(batched_args, batched_dims)
                if d is not batching.not_mapped}
  ref, val, *idxs = batched_args
  ref_dim, val_dim, *idx_dims = batched_dims
  ref_is_batched = ref_dim is not batching.not_mapped
  val_is_batched = val_dim is not batching.not_mapped
  idx_is_batched = any(i_dim is not batching.not_mapped for i_dim in idx_dims)
  if idx_is_batched:
    # If at least one of the idx is batched, we broadcast them all and move the
    # batch dim to the front.
    idxs = tuple(batching.bdim_at_front(i, d, axis_size) for i, d
                 in zip(idxs, idx_dims))
  idxs_shape, = {i.shape for i in idxs} or [()]
  if ref_is_batched and not idx_is_batched:
    indexed_dims = tuple_insert(indexed_dims, ref_dim, False)
    bdim_out = _output_bdim(indexed_dims, ref_dim, idxs_shape)
    if not val_is_batched:
      val = batching.broadcast(val, axis_size, 0)
      val_dim = 0
    val = batching.moveaxis(val, val_dim, bdim_out)
  elif idx_is_batched:
    assert ref_is_batched and val_is_batched
    indexed_dims = tuple_insert(indexed_dims, ref_dim, True)
    idx_place = [i for i, i_dim in enumerate(indexed_dims)
                 if i_dim].index(ref_dim)
    iota = lax.broadcasted_iota(np.dtype('int32'), idxs_shape, 0)
    idxs = tuple_insert(idxs, idx_place, iota)
    val = batching.moveaxis(val, val_dim, 0)
    bdim_out = 0
  return swap_p.bind(ref, val, *idxs, indexed_dims=indexed_dims), bdim_out
batching.primitive_batchers[swap_p] = _swap_vmap

def _addupdate_vmap(batched_args, batched_dims, *, indexed_dims):
  axis_size, = {x.shape[d] for x, d in zip(batched_args, batched_dims)
                if d is not batching.not_mapped}
  ref, val, *idxs = batched_args
  ref_dim, val_dim, *idx_dims = batched_dims
  ref_is_batched = ref_dim is not batching.not_mapped
  val_is_batched = val_dim is not batching.not_mapped
  idx_is_batched = any(i_dim is not batching.not_mapped for i_dim in idx_dims)
  if idx_is_batched:
    # If at least one of the idx is batched, we ensure all have bdims at front.
    idxs = tuple(batching.bdim_at_front(i, d, axis_size)
                 for i, d in zip(idxs, idx_dims))
  idxs_shape, = {i.shape for i in idxs} or [()]
  if ref_is_batched and not idx_is_batched:
    indexed_dims = tuple_insert(indexed_dims, ref_dim, False)
    bdim_out = _output_bdim(indexed_dims, ref_dim, idxs_shape)
    if not val_is_batched:
      val = batching.broadcast(val, axis_size, 0)
      val_dim = 0
    val = batching.moveaxis(val, val_dim, bdim_out)
  elif idx_is_batched:
    assert ref_is_batched and val_is_batched
    indexed_dims = tuple_insert(indexed_dims, ref_dim, True)
    idx_place = [i for i, i_dim in enumerate(indexed_dims)
                 if i_dim].index(ref_dim)
    idxs_shape, = {i.shape for i in idxs} or [()]
    iota = lax.broadcasted_iota(np.dtype('int32'), idxs_shape, 0)
    idxs = tuple_insert(idxs, idx_place, iota)
    val = batching.moveaxis(val, val_dim, 0)
  return addupdate_p.bind(ref, val, *idxs, indexed_dims=indexed_dims), []
batching.primitive_batchers[addupdate_p] = _addupdate_vmap
