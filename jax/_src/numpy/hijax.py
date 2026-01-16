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
from typing import Any, Callable

import numpy as np

from jax._src import ad_util
from jax._src import api
from jax._src import core
from jax._src import dtypes
from jax._src import numpy as jnp
from jax._src.hijax import VJPHiPrimitive
from jax._src.lax import control_flow
from jax._src.lax import lax
from jax._src.typing import Array, ArrayLike, DTypeLike


class SearchSorted(VJPHiPrimitive):
  valid_methods = ("compare_all", "scan", "scan_unrolled", "sort")

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
    if dimension < 0 or dimension >= sorted_arr_aval.ndim:
      raise ValueError(
          f"dimension={dimension} must be in range [0, {sorted_arr_aval.ndim})"
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
    if not batch_dims <= dimension < sorted_arr_aval.ndim:
      raise ValueError(
          f"dimension={dimension} must be in range [{batch_dims},"
          f" {sorted_arr_aval.ndim})"
      )
    if sorted_arr_aval.shape[:batch_dims] != query_aval.shape[:batch_dims]:
      raise ValueError(
          "batch dimension sizes must match; got"
          f" {sorted_arr_aval.shape[:batch_dims]} != {query_aval.shape[:batch_dims]}"
      )
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

  def expand(self, sorted_arr: ArrayLike, query: ArrayLike) -> Array:
    return _searchsorted_impl(
      sorted_arr, query,
      dimension=self.dimension,
      batch_dims=self.batch_dims,
      side=self.side,
      dtype=self.out_aval.dtype,
      method=self.method)

  def batch(
      self,
      _axis_data: Any,
      args: tuple[Array, Array],
      bdims: tuple[int | None, int | None]
  ) -> tuple[Array, int | None]:
    del _axis_data  # unused
    sorted_arr, query = args
    batch_dims = self.batch_dims
    dimension = self.dimension

    if bdims[0] is None and bdims[1] is None:
      # Neither array is batched
      out_bdim = None
    elif bdims[1] is None:
      # Only sorted_arr is batched
      assert bdims[0] is not None  # for type checker
      sorted_arr = jnp.moveaxis(sorted_arr, bdims[0], batch_dims)
      dimension += 1
      out_bdim = batch_dims
    elif bdims[0] is None:
      # Only query is batched
      assert bdims[1] is not None  # for type checker
      query = jnp.moveaxis(query, bdims[1], batch_dims)
      out_bdim = sorted_arr.ndim - 1
    else:
      # Both are batched
      sorted_arr = jnp.moveaxis(sorted_arr, bdims[0], 0)
      query = jnp.moveaxis(query, bdims[1], 0)
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

  def jvp(self, primals: tuple[Array, Array], _: Any) -> tuple[Array | ad_util.Zero, Array | ad_util.Zero]:
    primal_out = self(*primals)
    return primal_out, ad_util.Zero.from_primal_value(primal_out)

  def vjp_fwd(self, nzs_in: Any, *args: Array) -> tuple[Array, None]:
    return (self(*args), None)

  def vjp_bwd_retval(self, res: Any, g: Any):
    return (ad_util.zeros_like_aval(self.in_avals[0]),
            ad_util.zeros_like_aval(self.in_avals[1]))


def _searchsorted_impl(sorted_arr: ArrayLike, query: ArrayLike, *, dimension: int,
                       batch_dims: int, side: str, dtype: np.dtype, method: str):
  """Main implementation of searchsorted primitive."""
  sorted_arr = jnp.moveaxis(sorted_arr, dimension, -1)

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
  if sorted_arr.shape[0] == 0:
    return lax.full(query.shape, fill_value=0, dtype=dtype)
  if query.ndim > 0:
    return api.vmap(
        functools.partial(_searchsorted_scan_impl, side=side, dtype=dtype, unrolled=unrolled),
        in_axes=(None, 0),
    )(sorted_arr, query)

  op = lax._sort_le_comparator if side == "left" else lax._sort_lt_comparator  # pylint: disable=protected-access

  def body_fun(_, state):
    low, high = state
    mid = (low + high) // 2
    go_left = op(query, sorted_arr[mid])
    return (lax.select(go_left, low, mid), lax.select(go_left, mid, high))

  (n,) = sorted_arr.shape
  n_levels = int(np.ceil(np.log2(n + 1)))
  vma = tuple(core.typeof(sorted_arr).vma)
  init = (dtype.type(0), dtype.type(n))
  init = tuple(core.pvary(i, vma) for i in init)
  return control_flow.fori_loop(
      0, n_levels, body_fun, init, unroll=n_levels if unrolled else 1
  )[1]


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
  op = lax._sort_lt_comparator if side == "left" else lax._sort_le_comparator  # pylint: disable=protected-access
  comparisons = api.vmap(op, in_axes=(0, None))(sorted_arr, query)
  return comparisons.sum(dtype=dtype, axis=0)


def searchsorted(
    sorted_arr: ArrayLike,
    query: ArrayLike,
    *,
    side: str = "left",
    dimension: int = 0,
    batch_dims: int = 0,
    method: str = "scan",
    dtype: DTypeLike = "int32",
):
  """Batch-aware searchsorted primitive."""
  sorted_arr, query = core.standard_insert_pvary(sorted_arr, query)
  prim = SearchSorted(
    core.typeof(sorted_arr),
    core.typeof(query),
    side=side,
    dimension=dimension,
    batch_dims=batch_dims,
    method=method,
    out_dtype=dtypes.dtype(dtype),
  )
  return prim(sorted_arr, query)
