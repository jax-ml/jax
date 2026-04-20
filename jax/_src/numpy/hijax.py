# Copyright 2026 The JAX Authors.
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

"""NumPy function implementations as hijax primitives."""
import functools
import operator
from typing import Any
from collections.abc import Callable

import numpy as np

from jax._src import ad_util
from jax._src import api
from jax._src import core
from jax._src import dtypes
from jax._src import numpy as jnp
from jax._src import tree_util
from jax._src.hijax import VJPHiPrimitive
from jax._src.lax import control_flow
from jax._src.lax import lax
from jax._src.numpy import util
from jax._src.typing import Array, ArrayLike, DTypeLike


class SearchSorted(VJPHiPrimitive):
  """HiJAX primitive for binary search."""
  valid_methods = ("compare_all", "scan", "scan_unrolled", "sort")

  side: str
  dimension: int
  batch_dims: int
  method: str
  out_dtype: np.dtype

  def __init__(
      self,
      sorted_arr_aval: core.ShapedArray,
      query_aval: core.ShapedArray,
      *,
      side: str,
      dimension: int,
      batch_dims: int,
      method: str,
      out_dtype: np.dtype):
    batch_dims = operator.index(batch_dims)
    if batch_dims < 0 or batch_dims >= sorted_arr_aval.ndim:
      raise ValueError(
          f"batch_dims={batch_dims} must be in range [0, {sorted_arr_aval.ndim})"
      )
    dimension = operator.index(dimension)
    if not batch_dims <= dimension < sorted_arr_aval.ndim:
      raise ValueError(
          f"dimension={dimension} must be in range [{batch_dims},"
          f" {sorted_arr_aval.ndim})"
      )
    if sorted_arr_aval.dtype != query_aval.dtype:
      raise ValueError(
          "dtypes of sorted_arr and query must match; got "
          f"{sorted_arr_aval.dtype} and {query_aval.dtype}"
      )
    if side not in ["left", "right"]:
      raise ValueError(
          f"invalid argument side={side!r}, expected 'left' or 'right'"
      )
    if method not in self.valid_methods:
      raise ValueError(
          f"invalid argument {method=}, expected one of {list(self.valid_methods)}"
      )
    if sorted_arr_aval.shape[:batch_dims] != query_aval.shape[:batch_dims]:
      raise ValueError(
          "batch dimension sizes must match; got"
          f" {sorted_arr_aval.shape[:batch_dims]} != {query_aval.shape[:batch_dims]}"
      )
    if not dtypes.issubdtype(out_dtype, np.integer):
      raise ValueError(f"out_dtype should be an integer type; got {out_dtype}")
    # Attempt this here to catch overflow errors early.
    out_dtype.type(sorted_arr_aval.shape[dimension])
    self.in_avals = (sorted_arr_aval, query_aval)
    self.out_aval = core.typeof(api.eval_shape(
      functools.partial(_searchsorted_impl,
        dimension=dimension, batch_dims=batch_dims, side=side,
        dtype=out_dtype, method=method),
        sorted_arr_aval, query_aval))
    self.params = dict(
      side=side,
      dimension=dimension,
      batch_dims=batch_dims,
      method=method,
    )
    super().__init__()

  def expand(self, sorted_arr: ArrayLike, query: ArrayLike) -> Array:  # pyrefly: ignore[bad-override]
    return _searchsorted_impl(
      sorted_arr, query,
      dimension=self.dimension,
      batch_dims=self.batch_dims,
      side=self.side,
      dtype=self.out_aval.dtype,
      method=self.method)

  def batch(
      self,
      axis_data: Any,
      args: tuple[Array, Array],
      dims: tuple[int | None, int | None]
  ) -> tuple[Array, int | None]:
    del axis_data  # unused
    sorted_arr, query = args
    batch_dims = self.batch_dims
    dimension = self.dimension

    if dims[0] is None and dims[1] is None:
      # Neither array is batched
      out_bdim = None
    elif dims[1] is None:
      # Only sorted_arr is batched
      assert dims[0] is not None  # for type checker
      sorted_arr = jnp.moveaxis(sorted_arr, dims[0], batch_dims)
      dimension += 1
      out_bdim = batch_dims
    elif dims[0] is None:
      # Only query is batched
      assert dims[1] is not None  # for type checker
      query = jnp.moveaxis(query, dims[1], batch_dims)
      out_bdim = sorted_arr.ndim - 1
    else:
      # Both are batched
      sorted_arr = jnp.moveaxis(sorted_arr, dims[0], 0)
      query = jnp.moveaxis(query, dims[1], 0)
      batch_dims += 1
      dimension += 1
      out_bdim = 0

    batched_prim = SearchSorted(
      core.typeof(sorted_arr),
      core.typeof(query),
      side=self.side,
      dimension=dimension,
      batch_dims=batch_dims,
      method=self.method,
      out_dtype=self.out_aval.dtype,
    )
    return batched_prim(sorted_arr, query), out_bdim

  def jvp(self, primals: tuple[Array, Array], tangents: Any) -> tuple[Array | np.ndarray, Array | np.ndarray]:
    del tangents  # unused
    primal_out = self(*primals)
    return primal_out, np.empty(primal_out.shape, dtype=dtypes.float0)

  def vjp_fwd(self, nzs_in: Any, *args: Array) -> tuple[Array, None]:
    return (self(*args), None)

  def vjp_bwd_retval(self, res: Any, g: Any):
    return (ad_util.zeros_like_aval(self.in_avals[0]),
            ad_util.zeros_like_aval(self.in_avals[1]))


class Nonzero(VJPHiPrimitive):
  """HiJAX primitive for nonzero."""

  size: int
  axes: tuple[int, ...]
  out_dtype: np.dtype

  def __init__(
      self,
      a_aval: core.ShapedArray,
      *,
      size: int,
      axes: tuple[int, ...],
      out_dtype: np.dtype):
    size = operator.index(size)
    if size < 0:
      raise ValueError(f"size must be a positive integer; got {size=}")
    if not dtypes.issubdtype(out_dtype, np.integer):
      raise ValueError(f"out_dtype must be integer typed; got {out_dtype=}")
    if not all(0 <= ax < a_aval.ndim for ax in axes):
      raise ValueError(f"axes out of range for array with {a_aval.ndim} dimensions:  {axes=}")
    if len(axes) != len(set(axes)):
      raise ValueError(f"duplicate axes are not allowed: {axes=}")
    self.in_avals = (a_aval,)

    # Evaluate shape to set out_aval
    self.out_aval = tree_util.tree_map(core.typeof, api.eval_shape(
        functools.partial(_nonzero_impl, size=size, axes=axes, out_dtype=out_dtype), a_aval))

    self.params = dict(
        size=size,
        axes=axes,
        out_dtype=out_dtype,
    )
    super().__init__()

  def expand(self, a: ArrayLike) -> tuple[Array, ...]:  # pyrefly: ignore[bad-override]
    return _nonzero_impl(a, size=self.size, axes=self.axes, out_dtype=self.out_dtype)

  def batch(
      self,
      axis_data: Any,
      args: tuple[Array],
      dims: tuple[int | None]
  ) -> tuple[tuple[Array, ...], int | tuple[int, ...] | None]:
    del axis_data  # unused
    a, = args
    d, = dims

    if d is None:
      return self(a), None

    # Adjust axes for the inserted batch dimension d at the front/elsewhere
    new_axes = tuple(ax + 1 if ax >= d else ax for ax in self.axes)

    batched_prim = Nonzero(
        core.typeof(a),
        size=self.size,
        axes=new_axes,
        out_dtype=self.out_dtype,
    )

    batch_axes = [ax for ax in range(a.ndim) if ax not in new_axes]
    out_bdim = batch_axes.index(d)

    out_dims = (out_bdim,) * len(new_axes)
    return batched_prim(a), out_dims

  def jvp(self, primals: tuple[Array], tangents: Any) -> tuple[tuple[Array, ...], tuple[Array | np.ndarray, ...]]:
    del tangents  # unused
    primal_out = self(*primals)
    tangents_out = tuple(np.empty(p.shape, dtype=dtypes.float0) for p in primal_out)
    return primal_out, tangents_out

  def vjp_fwd(self, nzs_in: Any, *args: Array) -> tuple[tuple[Array, ...], None]:
    return (self(*args), None)

  def vjp_bwd_retval(self, res: Any, g: Any):
    return (ad_util.zeros_like_aval(self.in_avals[0]),)


def _nonzero_impl(a: ArrayLike, *, size: int, axes: tuple[int, ...], out_dtype: np.dtype) -> tuple[Array, ...]:
  """Main implementation of nonzero primitive."""
  a = jnp.asarray(a)
  out_dtype = dtypes._maybe_canonicalize_explicit_dtype(out_dtype, "nonzero")
  axes = tuple(sorted(axes))

  if not axes:
    return ()

  batch_axes = [ax for ax in range(a.ndim) if ax not in axes]
  if a.size == 0 or size == 0:
    return tuple(jnp.empty((*batch_axes, size), dtype=out_dtype)
                 for _ in axes)

  transposed_a = jnp.transpose(a, (*batch_axes, *axes))

  batch_shape = transposed_a.shape[:len(batch_axes)]
  sub_shape = transposed_a.shape[len(batch_axes):]
  strides = np.cumprod(sub_shape[::-1])[::-1] // sub_shape
  strides = tuple(strides.tolist())

  flattened_a = transposed_a.reshape(*batch_shape, -1)
  mask = flattened_a if flattened_a.dtype == bool else (flattened_a != 0)
  cs_mask = jnp.cumsum(mask, axis=-1)

  bincount = jnp.zeros((*batch_shape, size), dtype=cs_mask.dtype)
  mesh_dims = jnp.ogrid[tuple(slice(None, sz) for sz in batch_shape)]
  mesh_dims = [lax.expand_dims(m, [m.ndim]) for m in mesh_dims]
  bincount = bincount.at[(*mesh_dims, cs_mask)].add(1, mode='drop')
  flat_indices = jnp.cumsum(bincount, axis=-1)

  out = [(flat_indices // stride) % sz for stride, sz in zip(strides, sub_shape)]
  counts = mask.sum(axis=-1, keepdims=True)
  fill_mask = lax.expand_dims(jnp.arange(size), range(counts.ndim - 1)) >= counts
  return tuple(lax.convert_element_type(jnp.where(fill_mask, 0, entry), out_dtype) for entry in out)


def _searchsorted_impl(sorted_arr: ArrayLike, query: ArrayLike, *, dimension: int,
                       batch_dims: int, side: str, dtype: np.dtype, method: str):
  """Main implementation of searchsorted primitive."""
  sorted_arr = jnp.moveaxis(sorted_arr, dimension, -1)
  dtype = dtypes._maybe_canonicalize_explicit_dtype(dtype, "searchsorted")

  if method == "scan":
    impl: Callable[..., Array] = functools.partial(_searchsorted_scan_impl, unrolled=False)
  elif method == "scan_unrolled":
    impl = functools.partial(_searchsorted_scan_impl, unrolled=True)
  elif method == "compare_all":
    impl = _searchsorted_compare_all_impl
  elif method == "sort":
    impl = _searchsorted_sort_impl
  else:
    raise ValueError(f"Unsupported method: {method}")

  fun = functools.partial(impl, side=side, dtype=dtype)
  for _ in range(sorted_arr.ndim - batch_dims - 1):
    fun = api.vmap(fun, in_axes=(0, None))
  for _ in range(batch_dims):
    fun = api.vmap(fun, in_axes=0)
  return fun(sorted_arr, query)


@functools.partial(api.jit, static_argnames=["side", "dtype", "unrolled"])
def _searchsorted_scan_impl(
    sorted_arr: Array, query: Array, side: str, dtype: np.dtype, unrolled: bool
) -> Array:
  """Scan-based implementation of searchsorted."""
  assert sorted_arr.ndim == 1
  assert side in ["left", "right"]
  (n,) = sorted_arr.shape
  if sorted_arr.size == 0:
    return lax.full(query.shape, fill_value=0, dtype=dtype)
  if query.ndim > 0:
    return api.vmap(
        functools.partial(_searchsorted_scan_impl, side=side, dtype=dtype, unrolled=unrolled),
        in_axes=(None, 0),
    )(sorted_arr, query)

  op = lax._sort_le_comparator if side == "left" else lax._sort_lt_comparator
  unsigned_dtype = np.uint64 if dtypes.iinfo(dtype).bits == 64 else np.uint32
  def body_fun(state, _):
    low, high = state
    mid = low + (high - low) // 2  # use this form to avoid overflow
    go_left = op(query, sorted_arr[mid])
    return (lax.select(go_left, low, mid), lax.select(go_left, mid, high)), ()
  n_levels = int(np.ceil(np.log2(n + 1)))
  vma = tuple(core.typeof(sorted_arr).mat.varying)
  init = (core.pvary(unsigned_dtype(0), vma), core.pvary(unsigned_dtype(n), vma))
  carry, _ = control_flow.scan(body_fun, init, (), length=n_levels,
                               unroll=n_levels if unrolled else 1)
  return carry[1].astype(dtype)


@functools.partial(api.jit, static_argnames=["side", "dtype"])
def _searchsorted_sort_impl(
    sorted_arr: Array, query: Array, side: str, dtype: np.dtype
) -> Array:
  """Cosorting-based implementation of searchsorted."""
  assert sorted_arr.ndim == 1
  assert side in ["left", "right"]
  working_dtype = (
      np.int32
      if sorted_arr.size + query.size <= np.iinfo(np.int32).max
      else np.int64
  )

  def _rank(x):
    idx = lax.iota(working_dtype, len(x))
    return lax.full_like(idx, 0).at[lax.sort_key_val(x, idx)[1]].set(idx)

  if side == "left":
    index = _rank(lax.concatenate([query.ravel(), sorted_arr], 0))[: query.size]
  else:
    index = _rank(lax.concatenate([sorted_arr, query.ravel()], 0))[
        sorted_arr.size :
    ]
  return lax.reshape(
      lax.sub(index, _rank(query.ravel())), np.shape(query)
  ).astype(dtype)


@functools.partial(api.jit, static_argnames=["side", "dtype"])
def _searchsorted_compare_all_impl(
    sorted_arr: Array, query: Array, side: str, dtype: np.dtype
) -> Array:
  assert sorted_arr.ndim == 1
  assert side in ["left", "right"]
  op = lax._sort_lt_comparator if side == "left" else lax._sort_le_comparator
  comparisons = api.vmap(op, in_axes=(0, None))(sorted_arr, query)
  return comparisons.sum(dtype=dtype, axis=0)


def searchsorted(
    sorted_arr: ArrayLike,
    query: ArrayLike,
    /,
    *,
    side: str = "left",
    dimension: int = 0,
    batch_dims: int = 0,
    method: str = "scan",
    dtype: DTypeLike = "int32",
):
  """Find indices of query values within a sorted array.

  This is a batch-aware implementation of :func:`numpy.searchsorted` built on a
  HiJAX primitive. It adds the `batch_dims` and `dimension` argument, which make
  the API closed under batching.

  Args:
    sorted_arr: N-dimensional array, which is assumed to be sorted in increasing
      order along ``dimension``.
    query: N-dimensional array of query values.
    side: 'left' (default) or 'right'. If 'left', find the index of the first
      suitable location. If 'right', find the index of the last.
    dimension: positive integer specifying the dimension of ``sorted_arr`` along
      which to insert query values. Defaults to the first dimension.
    batch_dims: integer specifying the number of leading dimensions of
      ``sorted_arr`` and ``query`` to treat as shared batch dimensions.
      Defaults to zero.
    method: string specifying the search method: one of 'scan' (default),
      'compare_all', or 'sort'. 'scan' uses a scan-based binary search implementation,
      'compare_all' directly compares all elements in `sorted_arr` to `query`, and
      'sort' uses a cosorting-based implementation.

  Returns:
    An array specifying the insertion locations of `query` into `sorted_arr`.
  """
  sorted_arr, query = core.standard_insert_pvary(sorted_arr, query)
  out_dtype = dtypes._maybe_canonicalize_explicit_dtype(np.dtype(dtype), "searchsorted")
  prim = SearchSorted(
    core.typeof(sorted_arr),
    core.typeof(query),
    side=side,
    dimension=dimension,
    batch_dims=batch_dims,
    method=method,
    out_dtype=out_dtype,
  )
  return prim(sorted_arr, query)


# TODO(jakevdp): delete this function when hijax is finalized.
def searchsorted_via_expand(
    sorted_arr: ArrayLike,
    query: ArrayLike,
    /,
    *,
    side: str = "left",
    dimension: int = 0,
    batch_dims: int = 0,
    method: str = "scan",
    dtype: DTypeLike = "int32",
):
  """Compute searchsorted() without binding the hijax primitive."""
  sorted_arr, query = core.standard_insert_pvary(sorted_arr, query)
  out_dtype = dtypes._maybe_canonicalize_explicit_dtype(np.dtype(dtype), "searchsorted")
  prim = SearchSorted(
    core.typeof(sorted_arr),
    core.typeof(query),
    side=side,
    dimension=dimension,
    batch_dims=batch_dims,
    method=method,
    out_dtype=out_dtype,
  )
  return prim.expand(sorted_arr, query)


def nonzero(
    a: ArrayLike,
    /,
    *,
    size: int,
    axes: tuple[int, ...] | None = None,
    dtype: DTypeLike = 'int32',
) -> tuple[Array, ...]:
  """Return indices of nonzero elements.

  This is a batch-aware implementation of :func:`numpy.nonzero` built on a
  HiJAX primitive.

  Args:
    a: N-dimensional array.
    size: static integer specifying the number of nonzero entries to return.
    axes: optional tuple of integers specifying the axes to compute the result over.
      Defaults to None (all axes).
    dtype: optional datatype for the returned indices. Defaults to int32.

  Returns:
    Tuple of length ``len(axes)`` containing the indices of each nonzero value.
  """
  a, = core.standard_insert_pvary(a)
  out_dtype = dtypes._maybe_canonicalize_explicit_dtype(np.dtype(dtype), "nonzero")
  axes = util.canonicalize_axis_tuple(axes, np.ndim(a))
  prim = Nonzero(
    core.typeof(a),
    size=size,
    axes=axes,
    out_dtype=out_dtype,
  )
  return prim(a)
