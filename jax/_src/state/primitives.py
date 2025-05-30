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
import types
from typing import Any, Union

import numpy as np

from jax._src import ad_util
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import pretty_printer as pp
from jax._src import traceback_util
from jax._src import tree_util
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import lax
from jax._src.state import indexing
from jax._src.state.types import (
    AbstractRef,
    AccumEffect,
    ReadEffect,
    Transform,
    TransformedRef,
    WriteEffect,
)
from jax._src.typing import Array
from jax._src.util import safe_map, safe_zip


## General utilities

## JAX utilities

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip
traceback_util.register_exclusion(__file__)

## get/swap/addupdate implementations

# `get` reads a value from a `Ref` type, a.k.a.:
# a = get_p.bind(x)
# or we can read using indices:
# a = get_p.bind(x, 0, 1)
# Staging out `a = get_p.bind(x)` where the aval of `x` is
# `Ref((3,), np.dtype('float32'))` leads to a jaxpr eqn printed like
#   a:f32[3] <- x[]
get_p = core.Primitive("get")
get_p.def_impl(partial(dispatch.apply_primitive, get_p))
batching.ragged_prop_rules[get_p] = batching.ragged_mask_transfer_identity

Indexer = Union[int, slice, Array, types.EllipsisType]


def get_ref_and_transforms(
    ref_or_view: Any,
    idx: Indexer | tuple[Indexer, ...] | None,
    function_name: str,
    force_trailing_indexer: bool = True,  # TODO(apaszke): Clean this up.
) -> tuple[Any, tuple[Transform, ...]]:
  if isinstance(ref_or_view, TransformedRef):
    ref, transforms = ref_or_view.ref, ref_or_view.transforms
  else:
    ref, transforms = ref_or_view, ()
  ref_aval = core.get_aval(ref)
  if not isinstance(ref_aval, AbstractRef):
    raise ValueError(f"Can only call `{function_name}` on a `Ref`: {ref}.")
  if not isinstance(ref_aval.inner_aval, core.ShapedArray):
    return ref, ()

  if idx is None or idx is Ellipsis:
    idx = ()
  elif not isinstance(idx, tuple):
    idx = (idx,)

  if not idx and not force_trailing_indexer:
    return ref, transforms
  if not idx and transforms and isinstance(transforms[-1], indexing.NDIndexer):
    return ref, transforms
  nd_indexer = indexing.NDIndexer.from_indices_shape(idx, ref_or_view.shape)
  return ref, (*transforms, nd_indexer)


def ref_get(
    ref_or_view: Any, idx: Indexer | tuple[Indexer, ...] | None = None
) -> Array:
  """Reads a value from a `Ref`, a.k.a. value <- ref[idx]."""
  ref, transforms = get_ref_and_transforms(ref_or_view, idx, "ref_get")
  flat_transforms, tree = tree_util.tree_flatten(transforms)
  return get_p.bind(ref, *flat_transforms, tree=tree)


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
swap_p.def_impl(partial(dispatch.apply_primitive, swap_p))


def swap_ragged_prop_rule(eqn_params, invar_raggedness, outvars):
  assert len(invar_raggedness) == 2
  invar_raggedness_lhs = invar_raggedness[0]
  invar_raggedness_rhs = invar_raggedness[1]

  return [invar_raggedness_rhs, invar_raggedness_lhs], [None]


batching.ragged_prop_rules[swap_p] = swap_ragged_prop_rule

def ref_swap(
    ref_or_view: AbstractRef | TransformedRef,
    idx: Indexer | tuple[Indexer, ...] | None,
    value: Array,
    _function_name: str = "ref_swap",
) -> Array:
  """Sets a `Ref`'s value and returns the original value."""
  if hasattr(ref_or_view, 'dtype'):
    value = _maybe_implicit_cast(ref_or_view.dtype, value)
  ref, transforms = get_ref_and_transforms(ref_or_view, idx, _function_name)
  flat_transforms, tree = tree_util.tree_flatten(transforms)
  return swap_p.bind(ref, value, *flat_transforms, tree=tree)

# TODO(slebedev,mattjj): replace with special handling of Python numeric types:
# if (isinstance(value, (int, float, complex)) and
#     value == np.array(value, dtype).item()): return cast
def _maybe_implicit_cast(dtype, value):
  aval = core.typeof(value)
  if (aval.weak_type and
      (dtypes.issubdtype(dtype, np.floating) and
       dtypes.issubdtype(aval.dtype, np.floating)) or
      (dtypes.issubdtype(dtype, np.integer) and
       dtypes.issubdtype(aval.dtype, np.integer))):
    return lax.convert_element_type(value, dtype)
  return value


def ref_set(
    ref_or_view: AbstractRef | TransformedRef,
    idx: Indexer | tuple[Indexer, ...] | None,
    value: Array,
) -> None:
  """Sets a `Ref`'s value, a.k.a. ref[idx] <- value."""
  ref_swap(ref_or_view, idx, value, _function_name="ref_set")


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
addupdate_p.def_impl(partial(dispatch.apply_primitive, addupdate_p))


def ref_addupdate(
    ref_or_view: AbstractRef,
    idx: Indexer | tuple[Indexer, ...] | None,
    x: Array,
) -> None:
  """Mutates a ref with an additive update i.e. `ref[idx] += x`."""
  ref, transforms = get_ref_and_transforms(ref_or_view, idx, "ref_addupdate")
  flat_transforms, tree = tree_util.tree_flatten(transforms)
  return addupdate_p.bind(ref, x, *flat_transforms, tree=tree)


## get/set/addupdate abstract evaluation rules


def _shape_after_transforming(
    shape: tuple[int | Array, ...], transforms: tuple[Transform, ...]
) -> tuple[int | Array, ...]:
  for transform in transforms:
    shape = transform.transform_shape(shape)  # type: ignore
  assert shape is not None
  return shape


def _dtype_after_transforming(
    dtype: Any, transforms: tuple[Transform, ...]
) -> Any:
  for transform in transforms:
    dtype = transform.transform_dtype(dtype)
  assert dtype is not None
  return dtype


def _sharding_after_transforming(sharding, transforms):
  for transform in transforms:
    sharding = transform.transform_sharding(sharding)
  assert sharding is not None
  return sharding


def _get_abstract_eval(ref_aval: AbstractRef, *args,
                       tree):
  transforms = tree_util.tree_unflatten(tree, args)
  if not isinstance(ref_aval, AbstractRef):
    raise ValueError(f"`get` must be called on `Ref` types: {ref_aval}.")
  if isinstance(ref_aval.inner_aval, core.ShapedArray):
    out_shape = _shape_after_transforming(ref_aval.shape, transforms)
    out_dtype = _dtype_after_transforming(ref_aval.dtype, transforms)
    out_sharding = _sharding_after_transforming(ref_aval.sharding, transforms)
    out_aval = ref_aval.inner_aval.update(
        shape=out_shape, dtype=out_dtype, sharding=out_sharding)
  else:
    if transforms:
      raise ValueError("Cannot index non-shaped array with nontrivial indices.")
    out_aval = ref_aval.inner_aval
  return (out_aval, {ReadEffect(0)})
get_p.def_effectful_abstract_eval(_get_abstract_eval)

def _swap_abstract_eval(ref_aval: AbstractRef,
                        val_aval: core.AbstractValue,
                        *args: Any, tree):
  transforms = tree_util.tree_unflatten(tree, args)
  out_aval: core.AbstractValue
  if not isinstance(ref_aval, AbstractRef):
    raise ValueError(f"`swap` must be called on `Ref` types: {ref_aval}.")
  if isinstance(ref_aval.inner_aval, core.ShapedArray):
    assert isinstance(val_aval, core.ShapedArray)
    expected_out_shape = _shape_after_transforming(ref_aval.shape, transforms)
    expected_out_dtype = _dtype_after_transforming(ref_aval.dtype, transforms)
    if expected_out_shape != val_aval.shape:
      raise ValueError("Invalid shape for `swap`. "
                       f"Ref shape: {ref_aval.shape}. "
                       f"Expected shape: {expected_out_shape}. "
                       f"Value shape: {val_aval.shape}. "
                       f"Transforms: {transforms}. ")
    if expected_out_dtype != val_aval.dtype:
      raise ValueError(
          "Invalid dtype for `swap`. "
          f"Ref dtype: {expected_out_dtype}. "
          f"Value dtype: {val_aval.dtype}. "
      )
    out_aval = core.ShapedArray(expected_out_shape, expected_out_dtype)
  else:
    if transforms:
      raise ValueError("Cannot index non-shaped array with nontrivial indices.")
    out_aval = ref_aval.inner_aval
  return (out_aval, {WriteEffect(0)})
swap_p.def_effectful_abstract_eval(_swap_abstract_eval)


def _addupdate_abstract_eval(ref_aval: AbstractRef,
                             val_aval: core.AbstractValue,
                             *args: Any, tree):
  transforms = tree_util.tree_unflatten(tree, args)
  if not isinstance(ref_aval, AbstractRef):
    raise ValueError(f"`addupdate` must be called on `Ref` types: {ref_aval}.")
  if isinstance(ref_aval.inner_aval, core.ShapedArray):
    out_shape = _shape_after_transforming(ref_aval.shape, transforms)
    out_dtype = _dtype_after_transforming(ref_aval.dtype, transforms)
    assert isinstance(val_aval, core.ShapedArray)
    if out_shape != val_aval.shape:
      raise ValueError(
          "Invalid shape for `addupdate`. "
          f"Ref shape: {ref_aval.shape}. "
          f"Expected shape: {out_shape}. "
          f"Value shape: {val_aval.shape}. "
          f"Transforms: {transforms}. "
      )
    if out_dtype != val_aval.dtype:
      raise ValueError("Invalid dtype for `addupdate`. "
                       f"Ref dtype: {ref_aval.dtype}. "
                       f"Value shape: {val_aval.dtype}. ")
  else:
    # Check that the transforms are valid
    if transforms:
      raise ValueError("Cannot index non-shaped array with nontrivial indices.")
  return [], {AccumEffect(0)}
addupdate_p.def_effectful_abstract_eval(_addupdate_abstract_eval)

## Pretty printing for `get` and `swap` in jaxprs

pp_ref_var = partial(pp.color, intensity=pp.Intensity.NORMAL,
                 foreground=pp.Color.GREEN)


def _pp_transforms(
    context: core.JaxprPpContext,
    transforms: tuple[Transform, ...],
):
  if not transforms:
    return pp.text("[...]")
  return pp.concat(
      [transform.pretty_print(context) for transform in transforms]
  )


def pp_ref_transforms(context: core.JaxprPpContext, ref, transforms):
  return pp_ref_var(
      pp.concat([
          pp.text(core.pp_var(ref, context)),
          _pp_transforms(context, transforms),
      ])
  )


def _get_pp_rule(eqn, context, settings) -> pp.Doc:
  # Pretty prints `a = get x i` as `x[i] <- a`
  y, = eqn.outvars
  x, *flat_idx = eqn.invars
  transforms = tree_util.tree_unflatten(eqn.params["tree"], flat_idx)
  lhs = core.pp_vars([y], context, print_shapes=settings.print_shapes)
  return pp.concat(
      [lhs, pp.text(" <- "), pp_ref_transforms(context, x, transforms)]
  )
core.pp_eqn_rules[get_p] = _get_pp_rule

def _swap_pp_rule(eqn, context, settings) -> pp.Doc:
  y, = eqn.outvars
  x, v, *flat_idx = eqn.invars
  transforms = tree_util.tree_unflatten(eqn.params["tree"], flat_idx)
  if type(y) is core.DropVar:
    # In the case of a set (ignored return value),
    # pretty print `_ = swap x v i` as `x[i] <- v`
    del y
    return pp.concat([
        pp_ref_transforms(context, x, transforms),
        pp.text(" <- "),
        pp.text(core.pp_var(v, context)),
    ])
  else:
    # pretty-print `y:T = swap x v i` as `y:T, x[i] <- x[i], v`
    x_i = pp_ref_transforms(context, x, transforms)
    y = core.pp_vars([y], context, print_shapes=settings.print_shapes)
    return pp.concat([y, pp.text(', '), x_i, pp.text(' <- '),
                      x_i, pp.text(', '),
                      pp.text(core.pp_var(v, context))])
core.pp_eqn_rules[swap_p] = _swap_pp_rule

def _addupdate_pp_rule(eqn, context, settings) -> pp.Doc:
  del settings
  # pretty-print ` = addupdate x i v` as `x[i] += v`
  () = eqn.outvars
  x, v, *flat_idx = eqn.invars
  transforms = tree_util.tree_unflatten(eqn.params["tree"], flat_idx)
  return pp.concat([
      pp_ref_transforms(context, x, transforms),
      pp.text(" += "),
      pp.text(core.pp_var(v, context)),
  ])
core.pp_eqn_rules[addupdate_p] = _addupdate_pp_rule

## get/swap/addupdate JVP rules

def _get_jvp(primals: list[Any], tangents: list[Any], **params: Any):
  ref_primal, *idx = primals
  assert isinstance(ref_primal.aval, AbstractRef)
  ref_tangent, *_ = tangents
  assert isinstance(ref_tangent.aval, AbstractRef)
  return (get_p.bind(ref_primal, *idx, **params),
          get_p.bind(ref_tangent, *idx, **params))
ad.primitive_jvps[get_p] = _get_jvp

def _swap_jvp(primals: list[Any], tangents: list[Any], **params: Any):
  ref_primal, x_primal, *idx = primals
  assert isinstance(ref_primal.aval, AbstractRef)
  ref_tangent, x_tangent, *_ = tangents
  # if type(ref_tangent) is ad_util.Zero:
  #   raise Exception("you're an idiot")
  assert isinstance(ref_tangent.aval, AbstractRef)
  x_tangent = ad_util.instantiate(x_tangent)
  return (swap_p.bind(ref_primal, x_primal, *idx, **params),
          swap_p.bind(ref_tangent, x_tangent, *idx, **params))
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
  del x  # old value doesn't matter anymore
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

def _batch_indexer(
    indexer: indexing.NDIndexer,
    dims,
    axis_size: int,
    ref_shape: tuple[int, ...],
    ref_dim: int | batching.NotMapped,
    idx_is_batched: bool,
) -> indexing.NDIndexer:
  """Converts a batched indexer into an unbatched one.

  This function handles the complexity of `vmap`-style batching where either the
  `ref` being indexed, the indexer, or both may have batched dimensions. The
  goal is to produce a new indexer that acts as if applied in a batched context,
  but without actual batching, enabling downstream code to process it as usual.

  If any index in `indexer` is batched, all array indexers are normalized. If
  the array indexer contains a batched dimension, the dimension is moved to the
  front (axis 0). If the array indexer not batched, it is broadcasted to include
  a batch dimension at the front. This is to guarantee that all array indexers
  are still of the same shape.

  Slices are passed through unchanged unless they contain dynamic elements and
  are themselves batched, which is currently unsupported.

  If `ref` is batched (`ref_dim` is not `NotMapped`), we simulate per-example
  indexing by inserting a new iota array at the position corresponding to
  `ref_dim` in the indexer.

  It is worth noting that if the array indexers in the original indexer are
  contiguous, but become non-contiguous in the new indexer due to the insertion
  of the iota, the dimensions corresponding to the array indexers will be moved
  to the front in the indexing result. The batched dimension will be at axis 0,
  while the dimensions corresponding to the array indexers in the original
  indexer will start from axis 1. This behavior would cause a mismatch between
  the original indexer and the new indexer. Callers must take this behavior into
  account and properly transpose the arrays involved to avoid this mismatch.

  Args:
    indexer: An `NDIndexer` that indexes into `ref`.
    dims: A pytree with the same structure as `indexer`, indicating which
      dimension (if any) is batched for each array indexer.
    axis_size: Size of the batch dimension.
    ref_shape: Shape of `ref`.
    ref_dim: The dimension of `ref` that is batched (if any).
    idx_is_batched: Whether any index in the `indexer` is batched.
  """
  indices = indexer.indices
  indices_dims = dims.indices
  new_indices: list[Array | indexing.Slice | int] = []
  new_integer_indexer_shape = (axis_size, *indexer.int_indexer_shape)
  for idx, dim in zip(indices, indices_dims):
    if idx_is_batched:
      # If at least one of the idx is batched, we broadcast them all and move the
      # batch dim to the front.
      if isinstance(idx, indexing.Slice):
        # size is static, but start can be dynamic
        # Check if start is static (which it can be)
        is_static_slice = len(tree_util.tree_leaves(idx)) == 0
        if is_static_slice:
          new_indices.append(idx)
          continue
        dim = dim.start
        if dim is batching.not_mapped:
          # Broadcasting the slice is free (the start index stays the same)
          new_indices.append(idx)
        else:
          raise NotImplementedError(
              f"No support for vmapping over nontrivial slices just yet: {idx}")
      else:
        # Check if we are indexing with a scalar or not. If we are indexing
        # with a scalar and we are not batched, we can avoid broadcasting it.
        assert hasattr(idx, "shape")
        if not idx.shape:
          if dim is not batching.not_mapped:
            assert idx.shape == (axis_size,)
            idx = lax.broadcast_in_dim(idx, new_integer_indexer_shape, (0,))
          new_indices.append(idx)
        else:
          if dim is batching.not_mapped:
            bcast_dims = tuple(range(1, np.ndim(idx) + 1))
            idx = lax.broadcast_in_dim(idx, new_integer_indexer_shape,
                                       bcast_dims)
          else:
            idx = batching.moveaxis(idx, dim, 0)  # type: ignore[arg-type]
          new_indices.append(idx)
    else:
      if ref_dim is not batching.not_mapped:
        if not isinstance(idx, indexing.Slice):
          assert hasattr(idx, "shape")
          if idx.shape:
            bcast_dims = tuple(range(1, np.ndim(idx) + 1))
            idx = lax.broadcast_in_dim(idx, new_integer_indexer_shape,
                                      bcast_dims)
      new_indices.append(idx)
  if ref_dim is not batching.not_mapped:
    iota = lax.broadcasted_iota(np.dtype('int32'), new_integer_indexer_shape, 0)
    new_indices.insert(ref_dim, iota)
  return indexing.NDIndexer(
      tuple(new_indices), ref_shape, new_integer_indexer_shape, validate=True
  )

def _get_vmap(batched_args, batched_dims, *, tree):
  axis_size, = {x.shape[d] for x, d in zip(batched_args, batched_dims)
                if d is not batching.not_mapped}
  ref, *flat_idxs = batched_args
  ref_dim, *flat_idx_dims = batched_dims
  indexers = tree_util.tree_unflatten(tree, flat_idxs)
  indexers_dims = tree_util.tree_unflatten(tree, flat_idx_dims)

  idx_is_batched = any(i_dim is not batching.not_mapped
                       for i_dim in flat_idx_dims)
  if len(indexers) > 1:
    raise NotImplementedError("Batching with multiple indexers not supported.")
  # TODO(sharadmv): handle vmap of multiple indexers
  new_indexers = tuple(_batch_indexer(indexer, dims, axis_size,
                                  ref.shape, ref_dim, idx_is_batched)
                     for indexer, dims in zip(indexers, indexers_dims))
  flat_indexers, tree = tree_util.tree_flatten(new_indexers)

  is_int_indexing, _, _ = indexing.unpack_ndindexer(indexers[0])
  int_indexers_contiguous = bool(
      np.all(np.diff(np.where(is_int_indexing)[0]) == 1)
  )
  is_new_int_indexing, _, _ = indexing.unpack_ndindexer(new_indexers[0])
  new_int_indexers_contiguous = bool(
      np.all(np.diff(np.where(is_new_int_indexing)[0]) == 1)
  )

  out = get_p.bind(ref, *flat_indexers, tree=tree)
  if not int_indexers_contiguous:  # will always be moved to the front
    out_bdim = 0
  else:  # originally not going to be moved to the front
    if new_int_indexers_contiguous:  # now not going to be moved to the front
      out_bdim = is_new_int_indexing.index(True)
    else:  # now going to be moved to the front
      original_pos = is_int_indexing.index(True)
      array_indexer_shape = new_indexers[0].int_indexer_shape
      array_indexer_len = len(array_indexer_shape)

      transpose_order = list(range(len(out.shape)))
      transpose_order = (
          transpose_order[0],
          *transpose_order[array_indexer_len:array_indexer_len+original_pos],
          *transpose_order[1:array_indexer_len],
          *transpose_order[array_indexer_len+original_pos:],
      )

      out = lax.transpose(out, transpose_order)
      out_bdim = 0
  return out, out_bdim
batching.primitive_batchers[get_p] = _get_vmap

def _swap_vmap(batched_args, batched_dims, *, tree):
  axis_size, = {x.shape[d] for x, d in zip(batched_args, batched_dims)
                if d is not batching.not_mapped}
  ref, val, *flat_idxs = batched_args
  ref_dim, val_dim, *flat_idx_dims = batched_dims
  indexers = tree_util.tree_unflatten(tree, flat_idxs)
  indexers_dims = tree_util.tree_unflatten(tree, flat_idx_dims)

  ref_is_batched = ref_dim is not batching.not_mapped
  val_is_batched = val_dim is not batching.not_mapped
  idx_is_batched = any(i_dim is not batching.not_mapped
                       for i_dim in flat_idx_dims)

  if not ref_is_batched:
    raise Exception("performing a set/swap operation with vmapped value on "
                    "an unbatched mutable array reference "
                    f"of type {core.typeof(ref)}. Move the mutable array to be "
                    "an argument to the vmapped function?")

  if len(indexers) > 1:
    raise NotImplementedError("Batching with multiple indexers not supported.")
  # TODO(sharadmv): handle vmap of multiple indexers
  new_indexers = tuple(_batch_indexer(indexer, dims, axis_size,
                                  ref.shape, ref_dim, idx_is_batched)
                     for indexer, dims in zip(indexers, indexers_dims))
  flat_indexers, tree = tree_util.tree_flatten(new_indexers)

  is_int_indexing, _, _ = indexing.unpack_ndindexer(indexers[0])
  int_indexers_contiguous = bool(
      np.all(np.diff(np.where(is_int_indexing)[0]) == 1)
  )
  is_new_int_indexing, _, _ = indexing.unpack_ndindexer(new_indexers[0])
  new_int_indexers_contiguous = bool(
      np.all(np.diff(np.where(is_new_int_indexing)[0]) == 1)
  )

  if not new_int_indexers_contiguous:  # will be moved to the front
    batched_dim_in_result = 0
  else:
    batched_dim_in_result = is_new_int_indexing.index(True) + 0

  if not val_is_batched:
    if ref_is_batched or idx_is_batched:
      val = batching.broadcast(val, axis_size, batched_dim_in_result)
  else:
    val = batching.moveaxis(val, val_dim, batched_dim_in_result)

  transpose_order_inversed = None

  # Originally not going to be moved to the front, but now going to be moved to
  # the front.
  if int_indexers_contiguous and not new_int_indexers_contiguous:
    original_pos = is_int_indexing.index(True)
    array_indexer_shape = new_indexers[0].int_indexer_shape
    array_indexer_len = len(array_indexer_shape)

    transpose_order = list(range(len(val.shape)))
    transpose_order = (
        transpose_order[0],
        *transpose_order[1+original_pos:(1+original_pos)+(array_indexer_len-1)],
        *transpose_order[1:1+original_pos],
        *transpose_order[(1+original_pos)+(array_indexer_len-1):],
    )
    val = val.transpose(transpose_order)
    transpose_order_inversed = np.argsort(transpose_order)

  out = swap_p.bind(ref, val, *flat_indexers, tree=tree)

  # `val` should not be transposed, but we needed to transpose it to match
  # `swap_p`. As a result, the output of `swap_p` is also transposed. Now we
  # need to transpose it back.
  if transpose_order_inversed is not None:
    out = out.transpose(transpose_order_inversed)

  return out, batched_dim_in_result
batching.primitive_batchers[swap_p] = _swap_vmap

def _addupdate_vmap(batched_args, batched_dims, *, tree):
  axis_size, = {x.shape[d] for x, d in zip(batched_args, batched_dims)
                if d is not batching.not_mapped}
  ref, val, *flat_idxs = batched_args
  ref_dim, val_dim, *flat_idx_dims = batched_dims
  indexers = tree_util.tree_unflatten(tree, flat_idxs)
  indexers_dims = tree_util.tree_unflatten(tree, flat_idx_dims)

  ref_is_batched = ref_dim is not batching.not_mapped
  val_is_batched = val_dim is not batching.not_mapped
  idx_is_batched = any(i_dim is not batching.not_mapped
                       for i_dim in flat_idx_dims)
  if len(indexers) > 1:
    raise NotImplementedError("Batching with multiple indexers not supported.")
  # TODO(sharadmv): handle vmap of multiple indexers
  new_indexers = tuple(_batch_indexer(indexer, dims, axis_size,
                                  ref.shape, ref_dim, idx_is_batched)
                     for indexer, dims in zip(indexers, indexers_dims))
  flat_indexers, tree = tree_util.tree_flatten(new_indexers)

  is_int_indexing, _, _ = indexing.unpack_ndindexer(indexers[0])
  int_indexers_contiguous = bool(
      np.all(np.diff(np.where(is_int_indexing)[0]) == 1)
  )
  is_new_int_indexing, _, _ = indexing.unpack_ndindexer(new_indexers[0])
  new_int_indexers_contiguous = bool(
      np.all(np.diff(np.where(is_new_int_indexing)[0]) == 1)
  )

  if not new_int_indexers_contiguous:  # will be moved to the front
    batched_dim_in_result = 0
  else:
    batched_dim_in_result = is_new_int_indexing.index(True)

  if not val_is_batched:
    if ref_is_batched or idx_is_batched:
      val = batching.broadcast(val, axis_size, batched_dim_in_result)
  else:
    val = batching.moveaxis(val, val_dim, batched_dim_in_result)

  # Originally not going to be moved to the front, but now going to be moved to
  # the front.
  if int_indexers_contiguous and not new_int_indexers_contiguous:
    original_pos = is_int_indexing.index(True)
    array_indexer_shape = new_indexers[0].int_indexer_shape
    array_indexer_len = len(array_indexer_shape)

    transpose_order = list(range(len(val.shape)))
    transpose_order = (
        transpose_order[0],
        *transpose_order[1+original_pos:(1+original_pos)+(array_indexer_len-1)],
        *transpose_order[1:1+original_pos],
        *transpose_order[(1+original_pos)+(array_indexer_len-1):],
    )
    val = val.transpose(transpose_order)

  return addupdate_p.bind(ref, val, *flat_indexers, tree=tree), []
batching.primitive_batchers[addupdate_p] = _addupdate_vmap

# Currently, JAX doesn't have a primitive that does an equal-rank broadcast.
# We could use `jnp.broadcast_to` but that lowers to squeezing,
# then broadcast_in_dim. Triton has an equal-rank broadcast (`tl.broadcast_to`)
# so in the lowering, we have to expand out those squeezed dimensions again.
# Having a simple `broadcast_to` primitive allows us to lower directly
# to `tl.broadcast_to`.
broadcast_to_p = core.Primitive('broadcast_to')

def broadcast_to(a: Array, shape: tuple[int, ...]) -> Array:
  import jax.numpy as jnp
  a = jnp.asarray(a)
  if a.shape == shape:
    return a
  return broadcast_to_p.bind(a, shape=shape)

@broadcast_to_p.def_impl
def _broadcast_to_impl(a, *, shape):
  import jax.numpy as jnp
  return jnp.broadcast_to(a, shape)

@broadcast_to_p.def_abstract_eval
def _broadcast_to_abstract_eval(aval, *, shape):
  return core.ShapedArray(shape, aval.dtype)

mlir.register_lowering(
    broadcast_to_p, mlir.lower_fun(_broadcast_to_impl, False)
)

# === AD rules for mutable arrays ===

def _mut_jvp(primals, tangents):
  (init_val,), (init_val_dot,) = primals, tangents
  primal_out = core.mutable_array_p.bind(init_val)
  if type(init_val_dot) is ad_util.Zero:
    tangent_out = core.mutable_array_p.bind(ad_util.zeros_like_aval(init_val_dot.aval))
  else:
    tangent_out = core.mutable_array_p.bind(init_val_dot)
  return primal_out, tangent_out

ad.primitive_jvps[core.mutable_array_p] = _mut_jvp
ad.defjvp(core.freeze_p, lambda g, _: core.freeze(g))
