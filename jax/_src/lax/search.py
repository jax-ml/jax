# Copyright 2022 Google LLC
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

from functools import partial

import numpy as np

import jax
from jax._src import ad_util
from jax._src import api
from jax._src import util
from jax._src.api_util import _ensure_index_tuple
from jax._src.lax import lax
from jax._src.lax.control_flow import fori_loop
from jax import core
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import xla


def searchsorted(sorted_arr, query, *, side='left', dimension=0, batch_dims=None, method="default"):
  """Find indices of query values within a sorted array.

  Args:
    sorted_arr : N-dimensional array, which is assumed to be sorted in increasing
      order along ``dimension``.
    query : N-dimensional array of query values.
    side : {'left', 'right'}. If 'left', find the index of the first suitable
      location. If 'right', find the index of the last.
    dimension : integer specifying the dimension along which to insert query values.
    batch_dims : length-2 tuple of sequences specifying corresponding batch indices
      for `sorted_arr` and `query`.
    method : {'default', 'scan', 'sort'} The method used to compute the result.
      Outside JIT, this defaults to "scan". Within JIT, this defaults to "scan"
      on CPU, and "sort" otherwise. Assume ``M = query.size`` and
      ``N = sorted_arr.shape[dimension]``. ``method='scan'`` uses an *O[M log N]*
      binary search algorithm, while ``method='sort'`` uses an *O[(M+N) log(M+N)]*
      co-sorting approach. Although ``method='scan'`` may appear to have better
      scaling overall, in practice it cannot take advantage of parallelism inherent
      to accelerators, and so in most cases is slower than ``method='sort'`` on GPU
      and TPU backends.

  Returns:
    indices : an array specifying the insertion locations of `query` into `sorted_arr`.
  """
  dimension = util.canonicalize_axis(dimension, len(sorted_arr.shape))
  batch_dims = ((), ()) if batch_dims is None else tuple(_ensure_index_tuple(d) for d in batch_dims)
  return searchsorted_p.bind(sorted_arr, query, side=side, dimension=dimension, batch_dims=batch_dims, method=method)

def _searchsorted_abstract_eval(sorted_arr, query, *, side, dimension, batch_dims, method):
  lhs_batch, rhs_batch = batch_dims
  if sorted_arr.dtype != query.dtype:
    raise ValueError("dtypes of sorted_arr and query must match; got "
                     f"{sorted_arr.dtype} and {query.dtype}")
  if method not in ["default", "scan", "sort"]:
    raise ValueError(f"invalid argument method={method!r}; expected 'default', 'scan', or 'sort'.")
  if side not in ["left", "right"]:
    raise ValueError(f"invalid argument side={side!r}, expected 'left' or 'right'")
  if not 0 <= dimension < sorted_arr.ndim:
    raise ValueError(f"dimension={dimension}")
  if any(dim == dimension for dim in lhs_batch):
    raise ValueError("dimension cannot appear among batch_dims")
  if any(len(set(dims)) != len(dims) for dims in batch_dims):
    raise ValueError(f"batch dimensions cannot have repeated entries; got {batch_dims}")
  if not all((0 <= d1 < sorted_arr.ndim) and (0 <= d2 < query.ndim) for d1, d2 in zip(*batch_dims)):
    raise ValueError(f"Out of range batch dimensions {batch_dims} for arrays of shape "
                     f"{sorted_arr.shape} and {query.shape}")
  if any(sorted_arr.shape[d1] != query.shape[d2] for d1, d2 in zip(*batch_dims)):
    raise ValueError(f"Incompatible batch dimensions {batch_dims} for arrays of shape "
                     f"{sorted_arr.shape} and {query.shape}")
  shape = (*(sorted_arr.shape[d] for d in lhs_batch),
           *(s for d, s in enumerate(sorted_arr.shape) if d not in (dimension, *lhs_batch)),
           *(s for d, s in enumerate(query.shape) if d not in rhs_batch))
  dtype = np.int32 if sorted_arr.shape[dimension] < np.iinfo(np.int32).max else np.int64
  return core.ShapedArray(shape, dtype)

def _searchsorted_impl(sorted_arr, query, *, side, dimension, batch_dims, method):
  if method == "scan":
    _searchsorted = _searchsorted_scan_unbatched
  elif method == "sort":
    _searchsorted = _searchsorted_sort_unbatched
  else:
    raise ValueError(f"invalid argument method={method!r}; expected 'sort' or 'scan'")
  out_aval = _searchsorted_abstract_eval(sorted_arr, query, side=side, dimension=dimension,
                                         batch_dims=batch_dims, method=method)
  lhs_batch, rhs_batch = batch_dims
  lhs_extra = lax.remaining(range(len(sorted_arr.shape)), lhs_batch, [dimension])
  rhs_extra = lax.remaining(range(len(query.shape)), rhs_batch)
  sorted_arr = lax.transpose(sorted_arr, (*lhs_batch, *lhs_extra, dimension))
  query = lax.transpose(query, (*rhs_batch, *rhs_extra))
  fun = partial(_searchsorted, side=side, dtype=out_aval.dtype)

  for _ in lhs_extra:
    fun = api.vmap(fun, in_axes=(0, None))
  for _ in lhs_batch:
    fun = api.vmap(fun, in_axes=0)
  return fun(sorted_arr, query)

@partial(jax.jit, static_argnames=['side', 'dtype'])
def _searchsorted_scan_unbatched(sorted_arr, query, *, side, dtype):
  assert sorted_arr.ndim == 1
  assert side in ['left', 'right']
  if len(sorted_arr) == 0:
    return lax._zeros(query, dtype=dtype)
  if query.ndim > 0:
    return api.vmap(partial(_searchsorted_scan_unbatched, side=side, dtype=dtype), in_axes=(None, 0))(sorted_arr, query)

  op = lax._sort_le_comparator if side == 'left' else lax._sort_lt_comparator

  def body_fun(i, state):
    low, high = state
    mid = (low + high) // 2
    go_left = op(query, sorted_arr[mid])
    return (lax.select(go_left, low, mid), lax.select(go_left, mid, high))

  N, = sorted_arr.shape
  n_levels = int(np.ceil(np.log2(N + 1)))
  return fori_loop(0, n_levels, body_fun, (dtype.type(0), dtype.type(N)))[1]

@partial(jax.jit, static_argnames=['side', 'dtype'])
def _searchsorted_sort_unbatched(sorted_arr, query, *, side, dtype):
  assert sorted_arr.ndim == 1
  assert side in ['left', 'right']
  def _rank(x):
    idx = lax.iota(dtype, len(x))
    return lax._zeros(idx).at[lax.sort_key_val(x, idx)[1]].set(idx)
  if side == 'left':
    index = _rank(lax.concatenate([query.ravel(), sorted_arr], 0))[:query.size]
  else:
    index = _rank(lax.concatenate([sorted_arr, query.ravel()], 0))[sorted_arr.size:]
  return lax.reshape(lax.sub(index, _rank(index)), np.shape(query))

def _searchsorted_batch_rule(batched_args, bdims, *, side, dimension, batch_dims, method):
  sorted_arr, _ = batched_args
  lhs_bdim, rhs_bdim = bdims

  lhs_batch, rhs_batch = batch_dims

  if lhs_bdim is not None:
    lhs_batch = tuple(d if d < lhs_bdim else d + 1 for d in lhs_batch)
    if dimension >= lhs_bdim:
      dimension += 1
  if rhs_bdim is not None:
    rhs_batch = tuple(d if d < rhs_bdim else d + 1 for d in rhs_batch)

  if lhs_bdim is None:
    out_bdim = sorted_arr.ndim - 1 + rhs_bdim - sum(d < rhs_bdim for d in rhs_batch)
  elif rhs_bdim is None:
    out_bdim = len(lhs_batch) + lhs_bdim - sum(d < lhs_bdim for d in (dimension, *lhs_batch))
  else:
    lhs_batch = (lhs_bdim, *lhs_batch)
    rhs_batch = (rhs_bdim, *rhs_batch)
    out_bdim = 0

  return searchsorted_p.bind(*batched_args, side=side, dimension=dimension, batch_dims=[lhs_batch, rhs_batch], method=method), out_bdim

def _searchsorted_impl_scan_default(sorted_arr, query, *, side, dimension, batch_dims, method):
  return _searchsorted_impl(sorted_arr, query, side=side, dimension=dimension, batch_dims=batch_dims,
                            method="scan" if method == "default" else method)

def _searchsorted_impl_sort_default(sorted_arr, query, *, side, dimension, batch_dims, method):
  return _searchsorted_impl(sorted_arr, query, side=side, dimension=dimension, batch_dims=batch_dims,
                            method="sort" if method == "default" else method)

def _searchsorted_jvp(primals, tangents, *, side, dimension, batch_dims, method):
  primal_out = searchsorted_p.bind(*primals, side=side, dimension=dimension, batch_dims=batch_dims, method=method)
  return primal_out, ad_util.Zero.from_value(primal_out)

searchsorted_p = core.Primitive("searchsorted")
searchsorted_p.def_abstract_eval(_searchsorted_abstract_eval)
searchsorted_p.def_impl(_searchsorted_impl_scan_default)
ad.primitive_jvps[searchsorted_p] = _searchsorted_jvp
batching.primitive_batchers[searchsorted_p] = _searchsorted_batch_rule

# Default to method=scan on CPU; method=sort on accelerators.
xla.register_translation(searchsorted_p, xla.lower_fun(
  _searchsorted_impl_sort_default, multiple_results=False, new_style=True))
xla.register_translation(searchsorted_p, xla.lower_fun(
  _searchsorted_impl_scan_default, multiple_results=False, new_style=True),
  platform="cpu")
