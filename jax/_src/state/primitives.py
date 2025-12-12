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
    AbstractLinVal,
    AccumEffect,
    ReadEffect,
    Transform,
    TransformedRef,
    WriteEffect,
)
from jax._src.typing import Array, ArrayLike
from jax._src.util import safe_map, safe_zip


# Stand-in for hi-jax inputs to Ref.
HijaxType = Any


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
get_p.is_effectful = lambda params: True  # type: ignore
get_p.def_impl(partial(dispatch.apply_primitive, get_p))

get_p.is_high = lambda ref_aval, *_, tree: ref_aval.is_high  # type: ignore
def _get_to_lojax(ref, *idx, tree):
  val_ty = core.typeof(ref._refs)
  transforms = tree_util.tree_unflatten(tree, idx)
  if transforms:
    ref = TransformedRef(ref, transforms[:-1])
    idx = transforms[-1]
    return val_ty.ref_get_to_lojax(ref, idx)
  return val_ty.raise_val(*map(ref_get, val_ty.lower_val(ref._refs)))
get_p.to_lojax = _get_to_lojax  # type: ignore

Indexer = Union[int, slice, Array, types.EllipsisType]


def get_ref_and_transforms(
    ref_or_view: Any,
    idx: Indexer | tuple[Indexer, ...] | None,
    function_name: str,
) -> tuple[Any, tuple[Transform, ...]]:
  if isinstance(ref_or_view, TransformedRef):
    ref, transforms = ref_or_view.ref, ref_or_view.transforms
  else:
    ref, transforms = ref_or_view, ()
  ref_aval = core.get_aval(ref)
  if not isinstance(ref_aval, AbstractRef):
    raise ValueError(f"Can only call `{function_name}` on a `Ref`: {ref}.")
  if (not isinstance(ref_aval.inner_aval, core.ShapedArray)
      and not ref_aval.inner_aval.is_high):
    return ref, ()

  if idx is None or idx is Ellipsis:
    idx = ()
  elif not isinstance(idx, tuple):
    idx = (idx,)

  if not idx:
    return ref, transforms
  if not idx and transforms and isinstance(transforms[-1], indexing.NDIndexer):
    return ref, transforms
  nd_indexer = indexing.NDIndexer.from_indices_shape(idx, ref_or_view.shape)
  return ref, (*transforms, nd_indexer)

@partial(traceback_util.api_boundary, repro_api_name="jax.ref.get")
def ref_get(
    ref: core.Ref | TransformedRef,
    idx: Indexer | tuple[Indexer, ...] | None = None
) -> Array | HijaxType:
  """Read a value from an Ref.

  This is equivalent to ``ref[idx]`` for a NumPy-style indexer ``idx``.
  For more on mutable array refs, refer to the `Ref guide`_.

  Args:
    ref: a :class:`jax.ref.Ref` object.
    idx: a NumPy-style indexer

  Returns:
    A :class:`jax.Array` object (note, not a :class:`jax.ref.Ref`) containing
    the indexed elements of the mutable reference.

  Examples:
    >>> import jax
    >>> ref = jax.new_ref(jax.numpy.arange(5))
    >>> jax.ref.get(ref, slice(1, 3))
    Array([1, 2], dtype=int32)

    Equivalent operation via indexing syntax:

    >>> ref[1:3]
    Array([1, 2], dtype=int32)

    Use ``...`` to extract the full buffer:

    >>> ref[...]
    Array([0, 1, 2, 3, 4], dtype=int32)

  .. _Ref guide: https://docs.jax.dev/en/latest/array_refs.html
  """
  ref, transforms = get_ref_and_transforms(ref, idx, "ref_get")
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
swap_p.is_effectful = lambda params: True  # type: ignore
swap_p.def_impl(partial(dispatch.apply_primitive, swap_p))

swap_p.is_high = lambda ref_aval, *_, tree: ref_aval.is_high  # type: ignore
def _swap_to_lojax(ref, val, *idx, tree):
  ref_val_ty = core.typeof(ref._refs)
  val_ty = core.typeof(val)
  transforms = tree_util.tree_unflatten(tree, idx)
  if transforms:
    ref = TransformedRef(ref, transforms[:-1])
    idx = transforms[-1]
    return ref_val_ty.ref_swap_to_lojax(ref, val, idx)
  lo_refs = ref_val_ty.lower_val(ref._refs)
  lo_vals = val_ty.lower_val(val)
  outs = [ref_swap(lo_ref, idx, lo_val) for lo_ref, lo_val
          in zip(lo_refs, lo_vals)]
  return val_ty.raise_val(*outs)
swap_p.to_lojax = _swap_to_lojax  # type: ignore


@partial(traceback_util.api_boundary, repro_api_name="jax.ref.swap")
def ref_swap(
    ref: core.Ref | TransformedRef,
    idx: Indexer | tuple[Indexer, ...] | None,
    value: ArrayLike | HijaxType,
    _function_name: str = "ref_swap",
) -> Array | HijaxType:
  """Update an array value inplace while returning the previous value.

  This is equivalent to ``ref[idx], prev = value, ref[idx]`` while returning
  ``prev``, for a NumPy-style indexer ``idx``.
  For more on mutable array refs, refer to the `Ref guide`_.

  Args:
    ref: a :class:`jax.ref.Ref` object. On return, the buffer will be
      mutated by this operation.
    idx: a NumPy-style indexer
    value: a :class:`jax.Array` object (note, not a :class:`jax.ref.Ref`)
      containing the values to set in the array.

  Returns:
    A :class:`jax.Array` containing the previous value at `idx`.

  Examples:
    >>> import jax
    >>> ref = jax.new_ref(jax.numpy.arange(5))
    >>> jax.ref.swap(ref, 3, 10)
    Array(3, dtype=int32)
    >>> ref
    Ref([ 0,  1,  2, 10,  4], dtype=int32)

    Equivalent operation via indexing syntax:

    >>> ref = jax.new_ref(jax.numpy.arange(5))
    >>> ref[3], prev = 10, ref[3]
    >>> prev
    Array(3, dtype=int32)
    >>> ref
    Ref([ 0,  1,  2, 10,  4], dtype=int32)

    Use ``...`` to swap the value of a scalar ref:

    >>> ref = jax.new_ref(jax.numpy.int32(5))
    >>> jax.ref.swap(ref, ..., 10)
    Array(5, dtype=int32)
    >>> ref
    Ref(10, dtype=int32)

  .. _Ref guide: https://docs.jax.dev/en/latest/array_refs.html
  """
  if hasattr(ref, 'dtype'):
    value = _maybe_implicit_cast(ref.dtype, value)
  ref, transforms = get_ref_and_transforms(ref, idx, _function_name)
  flat_transforms, tree = tree_util.tree_flatten(transforms)
  return swap_p.bind(ref, value, *flat_transforms, tree=tree)

# TODO(slebedev,mattjj): replace with special handling of Python numeric types:
# if (isinstance(value, (int, float, complex)) and
#     value == np.array(value, dtype).item()): return cast
def _maybe_implicit_cast(dtype, value):
  aval = core.typeof(value)
  if not isinstance(aval, core.ShapedArray):
    return value
  if (aval.weak_type and
      (dtypes.issubdtype(dtype, np.floating) and
       dtypes.issubdtype(aval.dtype, np.floating)) or
      (dtypes.issubdtype(dtype, np.integer) and
       dtypes.issubdtype(aval.dtype, np.integer))):
    return lax.convert_element_type(value, dtype)
  return value


@partial(traceback_util.api_boundary, repro_api_name="jax.ref.set")
def ref_set(
    ref: core.Ref | TransformedRef,
    idx: Indexer | tuple[Indexer, ...] | None,
    value: ArrayLike | HijaxType,
) -> None:
  """Set a value in an Ref in-place.

  This is equivalent to ``ref[idx] = value`` for a NumPy-style indexer
  ``idx``. For more on mutable array refs, refer to the `Ref guide`_.

  Args:
    ref: a :class:`jax.ref.Ref` object. On return, the buffer will be
      mutated by this operation.
    idx: a NumPy-style indexer
    value: a :class:`jax.Array` object (note, not a :class:`jax.ref.Ref`)
      containing the values to set in the array.

  Returns:
    None

  Examples:
    >>> import jax
    >>> ref = jax.new_ref(jax.numpy.zeros(5))
    >>> jax.ref.set(ref, 1, 10.0)
    >>> ref
    Ref([ 0., 10.,  0.,  0.,  0.], dtype=float32)

    Equivalent operation via indexing syntax:

    >>> ref = jax.new_ref(jax.numpy.zeros(5))
    >>> ref[1] = 10.0
    >>> ref
    Ref([ 0., 10.,  0.,  0.,  0.], dtype=float32)

    Use ``...`` to set the value of a scalar ref:

    >>> ref = jax.new_ref(jax.numpy.int32(0))
    >>> ref[...] = 4
    >>> ref
    Ref(4, dtype=int32)

  .. _Ref guide: https://docs.jax.dev/en/latest/array_refs.html
  """
  ref_swap(ref, idx, value, _function_name="ref_set")


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
addupdate_p.is_effectful = lambda params: True  # type: ignore
addupdate_p.multiple_results = True
addupdate_p.def_impl(partial(dispatch.apply_primitive, addupdate_p))


def ref_addupdate(
    ref: core.Ref | TransformedRef,
    idx: Indexer | tuple[Indexer, ...] | None,
    x: ArrayLike | HijaxType,
) -> None:
  """Add to an element in an Ref in-place.

  This is analogous to ``ref[idx] += value`` for a NumPy array ``ref`` and
  NumPy-style indexer ``idx``. However, for an Ref ``ref``, executing
  ``ref[idx] += value`` actually performs a ``ref_get``, add, and ``ref_set``,
  so using this function can be more efficient under autodiff. For more on
  mutable array refs, refer to the `Ref guide`_.

  Args:
    ref: a :class:`jax.ref.Ref` object. On return, the buffer will be
      mutated by this operation.
    idx: a NumPy-style indexer
    x: a :class:`jax.Array` object (note, not a :class:`jax.ref.Ref`)
      containing the values to add at the specified indices.

  Returns:
    None

  Examples:
    >>> import jax
    >>> ref = jax.new_ref(jax.numpy.arange(5))
    >>> jax.ref.addupdate(ref, 2, 10)
    >>> ref
    Ref([ 0,  1, 12,  3,  4], dtype=int32)

    Equivalent operation via indexing syntax:

    >>> ref = jax.new_ref(jax.numpy.arange(5))
    >>> ref[2] += 10
    >>> ref
    Ref([ 0,  1, 12,  3,  4], dtype=int32)

    Use ``...`` to add to a scalar ref:

    >>> ref = jax.new_ref(jax.numpy.int32(2))
    >>> ref[...] += 10
    >>> ref
    Ref(12, dtype=int32)

  .. _Ref guide: https://docs.jax.dev/en/latest/array_refs.html
  """
  ref, transforms = get_ref_and_transforms(ref, idx, "ref_addupdate")
  flat_transforms, tree = tree_util.tree_flatten(transforms)
  addupdate_p.bind(ref, x, *flat_transforms, tree=tree)


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
  if transforms and ref_aval.inner_aval.is_high:
    return ref_aval.inner_aval.ref_get_abstract_eval(ref_aval, *args, tree=tree)
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
  if transforms and ref_aval.inner_aval.is_high:
    return ref_aval.inner_aval.ref_swap_abstract_eval(
        ref_aval, val_aval, *args, tree=tree)
  out_aval: core.AbstractValue
  if not isinstance(ref_aval, AbstractRef):
    raise ValueError(f"`swap` must be called on `Ref` types: {ref_aval}.")
  if isinstance(val_aval, AbstractRef):
    raise ValueError("Cannot store a Ref into another Ref. "
                     "Did you forget to load from it using `[...]`?")
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
    out_sharding = _sharding_after_transforming(ref_aval.sharding, transforms)
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
    if ((out_sharding.mesh._any_axis_explicit or
         val_aval.sharding.mesh._any_axis_explicit) and
        out_sharding != val_aval.sharding):
      raise ValueError("Invalid sharding for `addupdate`. "
                       f"Ref sharding: {ref_aval.sharding}. "
                       f"Value sharding: {val_aval.sharding}. ")
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
  ref_tangent, *_ = tangents
  out_primal = get_p.bind(ref_primal, *idx, **params)
  if isinstance(ref_tangent, ad_util.Zero):
    out_tangent = ad_util.Zero(core.typeof(out_primal).to_tangent_aval())
  else:
    out_tangent = get_p.bind(ref_tangent, *idx, **params)
  return out_primal, out_tangent
ad.primitive_jvps[get_p] = _get_jvp

def _swap_jvp(primals: list[Any], tangents: list[Any], **params: Any):
  ref_primal, x_primal, *idx = primals
  ref_tangent, x_tangent, *_ = tangents
  out_primal = swap_p.bind(ref_primal, x_primal, *idx, **params)
  if isinstance(ref_tangent, ad_util.Zero) and isinstance(x_tangent, ad_util.Zero):
    out_tangent = ad_util.Zero(core.typeof(out_primal).to_tangent_aval())
  elif ref_tangent.aval.kind == "anselm_ref":
    out_tangent = ad_util.Zero(core.typeof(out_primal).to_tangent_aval())
  else:
    if isinstance(ref_tangent, ad_util.Zero):
      raise Exception("performing a set/swap operation with a differentiated "
                      "value on a non-differentiated array reference of type "
                      f"{core.typeof(ref_primal)}. Move the array reference "
                      "to be an argument of the differentiated function?")
    x_tangent = ad_util.instantiate(x_tangent)
    out_tangent = swap_p.bind(ref_tangent, x_tangent, *idx, **params)
  return out_primal, out_tangent
ad.primitive_jvps[swap_p] = _swap_jvp

def addupdate_jvp_rule(primals: list[Any], tangents: list[Any], **params: Any):
  ref_primal, x_primal, *idx = primals
  ref_tangent, x_tangent, *_ = tangents
  x_tangent = ad_util.instantiate(x_tangent)
  if ref_tangent.aval.kind != "anselm_ref":
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


def _get_transpose_fancy(g, ref_, *idx, **params):
  if idx and type(g) is not ad_util.Zero:
    addupdate_p.bind(ref_.inst().ref, g, *idx, **params)
  else:
    ref_.accum(g)
ad.fancy_transposes[get_p] = _get_transpose_fancy

def _swap_transpose_fancy(g, ref_, x, *idx, **params):
  if ref_.ref is None and type(g) is ad_util.Zero:
    return
  elif ref_.ref is None:
    swap_p.bind(ref_.inst().ref, ad_util.instantiate(g), *idx, **params)
  else:
    x_bar = swap_p.bind(ref_.inst().ref, ad_util.instantiate(g), *idx, **params)
    x.accum(x_bar)
ad.fancy_transposes[swap_p] = _swap_transpose_fancy

def addupdate_transpose_fancy(cts_in, ref_, x, *idx, **params):
  if ref_.ref is not None and isinstance(x, ad.GradAccum):
    x_bar = get_p.bind(ref_.ref, *idx, **params)
    x.accum(x_bar)
ad.fancy_transposes[addupdate_p] = addupdate_transpose_fancy

## get/swap/addupdate partial_eval_custom rules

def _array_ref_partial_eval_custom(saveable, unks_in, inst_in, eqn):
  del saveable  # ignored, always full remat array_ref on known input
  unk, = unks_in
  inst, = inst_in
  invar, = eqn.invars
  res = [invar] if not inst else []
  if unk:
    return None, eqn, [True], [True], res  # tangent operation
  else:
    return eqn, eqn, [False], [True], res  # full remat
pe.partial_eval_jaxpr_custom_rules[core.ref_p] = _array_ref_partial_eval_custom

def _array_ref_batched(axis_data, vals_in, dims_in, memory_space, kind):
  val, = vals_in
  dim, = dims_in
  if dim is None:
    # We defensively batch the ref, b/c it could later be hit with a batched val
    val2 = batching.broadcast(val, axis_data.size, 0,
                              axis_data.explicit_mesh_axis)
    return core.ref_p.bind(val2, memory_space=memory_space, kind=kind), 0
  else:
    return core.ref_p.bind(val, memory_space=memory_space, kind=kind), dim
batching.fancy_primitive_batchers[core.ref_p] = _array_ref_batched

def _freeze_batched(axis_data, vals_in, dims_in):
  ref, = vals_in
  dim, = dims_in
  return core.freeze_p.bind(ref), dim
batching.fancy_primitive_batchers[core.freeze_p] = _freeze_batched

def _state_partial_eval_custom(saveable, unks_in, inst_in, eqn):
  del saveable  # ignored, always full remat state ops on known inputs
                # (except for anselm_ref)
  ref_unk, *_ = unks_in
  ref_inst, *inst_in = inst_in
  _, *val_vars = eqn.invars
  assert ref_inst
  res = [v for v, inst in zip(val_vars, inst_in) if not inst]
  if ref_unk:
    return None, eqn, [True], [True], res  # tangent operation
  elif eqn.invars[0].aval.kind == "anselm_ref":
    return eqn, None, [False], [False], res
  else:
    return eqn, eqn, [False], [True], res  # full remat
pe.partial_eval_jaxpr_custom_rules[get_p] = _state_partial_eval_custom
pe.partial_eval_jaxpr_custom_rules[swap_p] = _state_partial_eval_custom

def _addupdate_partial_eval_custom(saveable, unks_in, inst_in, eqn):
  del saveable  # ignored, always full remat state ops on known inputs
  ref_unk, *_ = unks_in
  ref_inst, *inst_in = inst_in
  _, *val_vars = eqn.invars
  assert ref_inst
  res = [v for v, inst in zip(val_vars, inst_in) if not inst]
  if ref_unk:
    return None, eqn, [], [], res  # tangent operation
  else:
    return eqn, eqn, [], [], res  # full remat
pe.partial_eval_jaxpr_custom_rules[addupdate_p] = _addupdate_partial_eval_custom

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
    if indexer.int_indexer_shape:
      batch_idx = lax.broadcasted_iota(
          np.dtype('int32'), new_integer_indexer_shape, 0)
    else:
      batch_idx = indexing.Slice(0, axis_size)  # type: ignore
      new_integer_indexer_shape = ()
    new_indices.insert(ref_dim, batch_idx)
  return indexing.NDIndexer(
      tuple(new_indices), ref_shape, new_integer_indexer_shape, validate=True
  )

def _get_vmap(batched_args, batched_dims, *, tree):
  axis_size, = {x.shape[d] for x, d in zip(batched_args, batched_dims)
                if d is not batching.not_mapped}
  ref, *flat_idxs = batched_args
  ref_dim, *flat_idx_dims = batched_dims
  indexers = tree_util.tree_unflatten(tree, flat_idxs)
  if not indexers:
    return get_p.bind(ref, *flat_idxs, tree=tree), ref_dim
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
  # Note: _batch_indexer will add a slice for the batch dim if the int_indexer
  # shape is empty, else it will use advanced/int indexing.
  will_add_int_batcher = bool(indexers[0].int_indexer_shape)

  is_new_int_indexing, _, _ = indexing.unpack_ndindexer(new_indexers[0])
  new_int_indexers_contiguous = bool(
      np.all(np.diff(np.where(is_new_int_indexing)[0]) == 1)
  )

  out = get_p.bind(ref, *flat_indexers, tree=tree)
  should_transpose = (int_indexers_contiguous and
                      not new_int_indexers_contiguous)
  if will_add_int_batcher and should_transpose:
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
  else:
    if ref_dim is not batching.not_mapped:
      if will_add_int_batcher:
        if not int_indexers_contiguous:
          # In this case the indexer is always moved to the front.
          out_bdim = 0
        else:
          # In this case the indexer is not moved to the front.
          out_bdim = is_new_int_indexing.index(True)
      else:
        # We only trigger this case when the int_indexer shape is empty,
        # so we don't need to account for int_indexer_shape.
        int_indexers_before_ref_dim = int(np.sum(is_new_int_indexing[:ref_dim]))
        out_bdim = ref_dim - int_indexers_before_ref_dim
    else:
      out_bdim = 0
      if any(is_int_indexing):
        # The batch dim is the indexer's batch dim.
        original_pos = is_int_indexing.index(True)
        out_bdim = original_pos
  return out, out_bdim
batching.primitive_batchers[get_p] = _get_vmap

def _swap_vmap(axis_data, batched_args, batched_dims, *, tree):
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
                    f"an unbatched array reference of type {core.typeof(ref)}. "
                    "Move the array reference to be an argument to the vmapped "
                    "function?")
  if not indexers:
    if ref_is_batched and not val_is_batched:
      val = batching.broadcast(val, axis_data.size, ref_dim,
                               axis_data.explicit_mesh_axis)
    return swap_p.bind(ref, val, *flat_idxs, tree=tree), ref_dim
  if len(indexers) > 1:
    raise NotImplementedError("Batching with multiple indexers not supported.")
  # TODO(sharadmv): handle vmap of multiple indexers
  new_indexers = tuple(_batch_indexer(indexer, dims, axis_data.size,
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
    try:
      batched_dim_in_result = is_new_int_indexing.index(True) + 0
    except ValueError:
      batched_dim_in_result = ref_dim

  if not val_is_batched:
    if ref_is_batched or idx_is_batched:
      val = batching.broadcast(val, axis_data.size, batched_dim_in_result,
                               axis_data.explicit_mesh_axis)
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
batching.fancy_primitive_batchers[swap_p] = _swap_vmap

def _addupdate_vmap(axis_data, batched_args, batched_dims, *, tree):
  ref, val, *flat_idxs = batched_args
  ref_dim, val_dim, *flat_idx_dims = batched_dims
  indexers = tree_util.tree_unflatten(tree, flat_idxs)
  indexers_dims = tree_util.tree_unflatten(tree, flat_idx_dims)

  ref_is_batched = ref_dim is not batching.not_mapped
  val_is_batched = val_dim is not batching.not_mapped
  idx_is_batched = any(i_dim is not batching.not_mapped
                       for i_dim in flat_idx_dims)

  if not ref_is_batched:
    raise Exception("performing an addupdate operation with vmapped value on "
                    f"an unbatched array reference of type {core.typeof(ref)}. "
                    "Move the array reference to be an argument to the vmapped "
                    "function?")
  if not indexers:
    if val_dim != ref_dim:
      val = batching.matchaxis2(axis_data, val_dim, ref_dim, val)
    return addupdate_p.bind(ref, val, *flat_idxs, tree=tree), []
  if len(indexers) > 1:
    raise NotImplementedError("Batching with multiple indexers not supported.")

  # TODO(sharadmv): handle vmap of multiple indexers
  new_indexers = tuple(_batch_indexer(indexer, dims, axis_data.size,
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
    try:
      batched_dim_in_result = is_new_int_indexing.index(True)
    except ValueError:
      batched_dim_in_result = ref_dim

  if not val_is_batched:
    if ref_is_batched or idx_is_batched:
      val = batching.broadcast(val, axis_data.size, batched_dim_in_result,
                               axis_data.explicit_mesh_axis)
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
batching.fancy_primitive_batchers[addupdate_p] = _addupdate_vmap

# Currently, JAX doesn't have a primitive that does an equal-rank broadcast.
# We could use `jnp.broadcast_to` but that lowers to squeezing,
# then broadcast_in_dim. Triton has an equal-rank broadcast (`tl.broadcast_to`)
# so in the lowering, we have to expand out those squeezed dimensions again.
# Having a simple `broadcast_to` primitive allows us to lower directly
# to `tl.broadcast_to`.
broadcast_to_p = core.Primitive('broadcast_to')

def broadcast_to(a: Array, shape: tuple[int, ...]) -> Array:
  """Broadcasts an array to a new shape.

  Args:
    a: The array to broadcast.
    shape: The desired shape to broadcast to.

  Returns:
    An array of shape ``shape``.

  See Also:
    :func:`jax.numpy.broadcast_to`
  """
  import jax.numpy as jnp  # pytype: disable=import-error
  a = jnp.asarray(a)
  if a.shape == shape:
    return a
  return broadcast_to_p.bind(a, shape=shape)

@broadcast_to_p.def_impl
def _broadcast_to_impl(a, *, shape):
  import jax.numpy as jnp  # pytype: disable=import-error
  return jnp.broadcast_to(a, shape)

@broadcast_to_p.def_abstract_eval
def _broadcast_to_abstract_eval(aval, *, shape):
  return core.ShapedArray(shape, aval.dtype)

mlir.register_lowering(
    broadcast_to_p, mlir.lower_fun(_broadcast_to_impl, False)
)

# === AD rules for mutable arrays ===

def _ref_jvp(primals, tangents, *, memory_space, kind):
  (init_val,), (init_dot,) = primals, tangents
  primal_out = core.ref_p.bind(init_val, memory_space=memory_space, kind=kind)
  if type(init_dot) is ad_util.Zero:
    zero = ad_util.zeros_like_aval(init_dot.aval)
    tangent_out = core.ref_p.bind(zero, memory_space=memory_space, kind=kind)
  else:
    tangent_out = core.ref_p.bind(init_dot, memory_space=memory_space, kind=kind)
  return primal_out, tangent_out

def _ref_lin(nzs, x, *, memory_space, kind):
  nz, = nzs
  x_ref = core.ref_p.bind(x, memory_space=memory_space, kind=kind)
  def mut_lin(_, x_dot):
    zero = ad_util.instantiate(x_dot)
    return core.ref_p.bind(zero, memory_space=memory_space, kind=kind)
  return x_ref, True, None, mut_lin

ad.primitive_jvps[core.ref_p] = _ref_jvp
ad.primitive_linearizations[core.ref_p] = _ref_lin
# TODO(mattjj): lin rule for freeze and accum_grad_in_ref?
ad.defjvp(core.freeze_p, lambda g, _: core.freeze(g))
ad.defjvp(core.accum_grad_in_ref_p, lambda g, _: core.accum_grad_in_ref_p.bind(g))

# === pinned, chained LinearVals ===

def create_linear(ty, memory_space=None):
  return create_linear_p.bind(ty=ty, memory_space=memory_space)
create_linear_p = core.Primitive('create_linear')

@create_linear_p.def_abstract_eval
def _create_linear_abstract_eval(*, ty, memory_space):
  if not isinstance(ty, core.ShapedArray): raise NotImplementedError(ty)
  return AbstractLinVal(ty, memory_space)

def _lower_create_linear(ctx):
  out_aval, = ctx.avals_out
  return mlir.custom_call(
      "CreateBuffer",
      operands=[],
      result_types=[mlir.aval_to_ir_type(out_aval)],
  ).results
mlir.register_lowering(create_linear_p, _lower_create_linear)


def pin(x):
  return pin_p.bind(x)
pin_p = core.Primitive('pin')

@pin_p.def_abstract_eval
def _pin_abstract_eval(aval):
  if not isinstance(aval, core.ShapedArray): raise NotImplementedError(aval)
  return AbstractLinVal(aval)

def _lower_pin(ctx, x_op):
  out_aval, = ctx.avals_out
  return mlir.custom_call(
      "Pin",
      operands=mlir.flatten_ir_values([x_op]),
      result_types=[mlir.aval_to_ir_type(out_aval)],
  ).results
mlir.register_lowering(pin_p, _lower_pin)


def unpin(x):
  return unpin_p.bind(x)
unpin_p = core.Primitive('unpin')

@unpin_p.def_abstract_eval
def _unpin_abstract_eval(aval):
  if not isinstance(aval, AbstractLinVal): raise TypeError(aval)
  return aval.inner_aval

def _lower_unpin(ctx, x_op):
  out_aval, = ctx.avals_out
  return mlir.custom_call(
      "Unpin",
      operands=mlir.flatten_ir_values([x_op]),
      result_types=[mlir.aval_to_ir_type(out_aval)],
  ).results
mlir.register_lowering(unpin_p, _lower_unpin)


def _linval_to_mlir_type(a):
  return mlir.ir.MemRefType.get(a.shape, mlir.dtype_to_ir_type(a.dtype),
                                memory_space=a.memory_space)
mlir.ir_type_handlers[AbstractLinVal] = _linval_to_mlir_type
