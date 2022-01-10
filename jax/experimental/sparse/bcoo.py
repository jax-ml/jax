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

"""BCOO (Bached coordinate format) matrix object and associated primitives."""
import functools
import operator
from typing import Any, NamedTuple, Sequence, Tuple
import warnings

import numpy as np

from jax import core
from jax import lax
from jax import tree_util
from jax import vmap
from jax.experimental.sparse._base import JAXSparse
from jax.experimental.sparse.util import _safe_asarray
from jax.interpreters import batching
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
import jax.numpy as jnp
from jax.interpreters import ad
from jax.util import safe_zip, unzip2, split_list
from jax._src import api_util
from jax._src.api_util import flatten_axes
from jax._src.lax.lax import (
  ranges_like, remaining, _dot_general_batch_dim_nums, _dot_general_shape_rule,
  DotDimensionNumbers)
from jax._src.numpy.lax_numpy import _unique

Dtype = Any
Shape = Tuple[int, ...]

#----------------------------------------------------------------------
# General utilities...
def broadcasting_vmap(fun, in_axes=0, out_axes=0):
  @functools.wraps(fun)
  def batched_fun(*args):
    args_flat, in_tree  = tree_util.tree_flatten(args)
    in_axes_flat = flatten_axes("vmap in_axes", in_tree, in_axes, kws=False)
    size = max(arg.shape[i] for arg, i in safe_zip(args_flat, in_axes_flat) if i is not None)
    if size > 1:
      if any(i is not None and arg.shape[i] not in (1, size)
             for arg, i in safe_zip(args_flat, in_axes_flat)):
        raise ValueError("broadcasting_vmap: mismatched input shapes")
      args_flat, in_axes_flat = zip(*(
          (arg, None) if i is None else (lax.squeeze(arg, (i,)), None) if arg.shape[i] == 1 else (arg, i)
          for arg, i in zip(args_flat, in_axes_flat)
      ))
    new_args = tree_util.tree_unflatten(in_tree, args_flat)
    new_in_axes = tree_util.tree_unflatten(in_tree, in_axes_flat)
    return vmap(fun, in_axes=new_in_axes, out_axes=out_axes)(*new_args)
  return batched_fun

#----------------------------------------------------------------------
# BCOO primitives: batched extension of COO.

def _bcoo_nse(mat, n_batch=0, n_dense=0):
  mat = jnp.asarray(mat)
  mask = (mat != 0)
  if n_dense > 0:
    mask = mask.any([-(i + 1) for i in range(n_dense)])
  mask = mask.sum(list(range(n_batch, mask.ndim)))
  return mask.max()

def _bcoo_sum_duplicates(data, indices, shape, nse=None, remove_zeros=True):
  if nse is None and isinstance(jnp.array(0), core.Tracer):
    raise ValueError("When used with JIT, vmap, or another transform, sum_duplicates() "
                     "requires passing a non-None value for the nse argument.")
  props = _validate_bcoo(data, indices, shape)
  f = functools.partial(_bcoo_sum_duplicates_unbatched, shape=shape[props.n_batch:],
                        nse=nse, remove_zeros=remove_zeros)
  for _ in range(props.n_batch):
    f = broadcasting_vmap(f)
  data_unique, indices_unique, nse_out = f(data, indices)
  if nse is None:
    nse = jnp.max(nse_out)
    data_unique = lax.slice_in_dim(data_unique, 0, nse, axis=props.n_batch)
    indices_unique = lax.slice_in_dim(indices_unique, 0, nse, axis=props.n_batch)
  return data_unique, indices_unique

def _bcoo_sum_duplicates_unbatched(data, indices, *, shape, nse, remove_zeros):
  props = _validate_bcoo(data, indices, shape)
  assert props.n_batch == 0
  if not props.n_sparse:
    nse = 1 if nse is None else nse
    data_unique = jnp.zeros_like(data, shape=(nse, *data.shape[1:])).at[0].set(data.sum(0))
    indices_unique = jnp.zeros_like(indices, shape=(nse, 0))
    return data_unique, indices_unique, nse
  fill_value = jnp.array(shape[:props.n_sparse], dtype=indices.dtype)
  out_of_bounds = (indices >= fill_value).any(-1, keepdims=True)
  if remove_zeros:
    data_all_zero = (data == 0).all(range(props.n_batch + 1, data.ndim))[:, None]
    out_of_bounds = out_of_bounds | data_all_zero
  indices = jnp.where(out_of_bounds, fill_value, indices)
  if nse is None:
    indices_unique, inv_idx, nse = _unique(
      indices, axis=0, return_inverse=True, return_true_size=True,
      size=props.nse, fill_value=fill_value)
    nse = nse - (indices == fill_value).any()
  else:
    indices_unique, inv_idx = jnp.unique(
      indices, axis=0, return_inverse=True,
      size=nse, fill_value=fill_value)
  data_shape = [indices_unique.shape[0], *data.shape[1:]]
  data_unique = jnp.zeros(data_shape, data.dtype).at[inv_idx].add(data)
  oob_mask = jnp.all(indices_unique == jnp.array(shape[:props.n_sparse]), 1)
  data_unique = jnp.where(oob_mask[(...,) + props.n_dense * (None,)], 0, data_unique)
  return data_unique, indices_unique, nse

def _unbatch_bcoo(data, indices, shape):
  n_batch = _validate_bcoo(data, indices, shape).n_batch
  if n_batch == 0:
    return data, indices
  data = jnp.broadcast_to(data, shape[:n_batch] + data.shape[n_batch:])
  indices = jnp.broadcast_to(indices, shape[:n_batch] + indices.shape[n_batch:])
  batch_indices = jnp.mgrid[tuple(slice(None, d) for d in indices.shape[:n_batch + 1])][:-1]
  batch_indices = batch_indices.reshape(n_batch, -1).T
  data = data.reshape(np.prod(data.shape[:n_batch + 1]), *data.shape[n_batch + 1:])
  indices = indices.reshape(np.prod(indices.shape[:n_batch + 1]), *indices.shape[n_batch + 1:])
  return data, jnp.hstack([batch_indices, indices])


class BCOOProperties(NamedTuple):
  n_batch: int
  n_sparse: int
  n_dense: int
  nse: int

class BCOOInfo(NamedTuple):
  shape: Shape


def _validate_bcoo(data: jnp.ndarray, indices: jnp.ndarray, shape: Sequence[int]) -> BCOOProperties:
  props = _validate_bcoo_indices(indices, shape)
  n_batch, n_sparse, n_dense, nse = props
  shape = tuple(shape)
  if any(s1 not in (1, s2) for s1, s2 in safe_zip(data.shape[:n_batch], shape[:n_batch])):
    raise ValueError("data batch dimensions not compatible for "
                     f"data.shape={data.shape}, shape={shape}")
  if data.shape[n_batch:] != (nse,) + shape[n_batch + n_sparse:]:
    raise ValueError(f"Invalid data.shape={data.shape} for "
                    f"nse={nse}, n_batch={n_batch}, n_dense={n_dense}")
  return props


def _validate_bcoo_indices(indices: jnp.ndarray, shape: Sequence[int]) -> BCOOProperties:
  assert jnp.issubdtype(indices.dtype, jnp.integer)
  shape = tuple(shape)
  nse, n_sparse = indices.shape[-2:]
  n_batch = indices.ndim - 2
  n_dense = len(shape) - n_batch - n_sparse
  assert n_dense >= 0
  if any(s1 not in (1, s2) for s1, s2 in safe_zip(indices.shape[:n_batch], shape[:n_batch])):
    raise ValueError("indices batch dimensions not compatible for "
                     f"indices.shape={indices.shape}, shape={shape}")
  if indices.shape[n_batch:] != (nse, n_sparse):
    raise ValueError(f"Invalid indices.shape={indices.shape} for "
                     f"nse={nse}, n_batch={n_batch}, n_dense={n_dense}")
  return BCOOProperties(n_batch=n_batch, n_sparse=n_sparse, n_dense=n_dense, nse=nse)


#----------------------------------------------------------------------
# bcoo_todense

bcoo_todense_p = core.Primitive('bcoo_todense')

def bcoo_todense(data, indices, *, spinfo):
  """Convert batched sparse matrix to a dense matrix.

  Args:
    data : array of shape ``batch_dims + (nse,) + block_dims``.
    indices : array of shape ``batch_dims + (n_sparse, nse)``
    spinfo : BCOOInfo. In particular, this includes the shape
      of the matrix, which is equal to ``batch_dims + sparse_dims + block_dims``
      where ``len(sparse_dims) == n_sparse``

  Returns:
    mat : array with specified shape and dtype matching ``data``
  """
  return bcoo_todense_p.bind(jnp.asarray(data), jnp.asarray(indices), spinfo=spinfo)

@bcoo_todense_p.def_impl
def _bcoo_todense_impl(data, indices, *, spinfo):
  shape = spinfo.shape
  n_batch, n_sparse, _, _ = _validate_bcoo(data, indices, shape)

  ind_slices = tuple(np.zeros(s, int) if i_s == 1 else np.arange(s)
                     for s, i_s in zip(shape[:n_batch], indices.shape[:n_batch]))
  grid = tuple(np.meshgrid(*ind_slices, indexing='ij', sparse=True))
  sparse_ind = tuple(indices[grid + (slice(None), i)] for i in range(n_sparse))

  batch_slices = tuple(np.arange(s) for s in shape[:n_batch])
  grid = np.meshgrid(*batch_slices, np.arange(1), indexing='ij', sparse=True)
  batch_ind = tuple(grid)[:-1]

  if not sparse_ind:
    data = data.sum(n_batch, keepdims=bool(batch_ind), dtype=data.dtype)
  return jnp.zeros(shape, data.dtype).at[batch_ind + sparse_ind].add(data)

@bcoo_todense_p.def_abstract_eval
def _bcoo_todense_abstract_eval(data, indices, *, spinfo):
  shape = spinfo.shape
  _validate_bcoo(data, indices, shape)
  return core.ShapedArray(shape, data.dtype)

def _bcoo_todense_jvp(data_dot, data, indices, *, spinfo):
  return bcoo_todense(data_dot, indices, spinfo=spinfo)

def _bcoo_todense_transpose(ct, data, indices, *, spinfo):
  shape = spinfo.shape
  assert ad.is_undefined_primal(data)
  if ad.is_undefined_primal(indices):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ct.shape == shape
  assert ct.dtype == data.aval.dtype
  return bcoo_extract(indices, ct), indices

def _bcoo_todense_batching_rule(batched_args, batch_dims, *, spinfo):
  data, indices = batched_args
  if any(b not in [0, None] for b in batch_dims):
    raise NotImplementedError(f"batch_dims={batch_dims}. Only 0 and None are supported.")
  if batch_dims[0] is None:
    data = data[None, ...]
  if batch_dims[1] is None:
    indices = indices[None, ...]
  new_spinfo = BCOOInfo(
    shape=(max(data.shape[0], indices.shape[0]), *spinfo.shape))
  return bcoo_todense(data, indices, spinfo=new_spinfo), 0

ad.defjvp(bcoo_todense_p, _bcoo_todense_jvp, None)
ad.primitive_transposes[bcoo_todense_p] = _bcoo_todense_transpose
batching.primitive_batchers[bcoo_todense_p] = _bcoo_todense_batching_rule
xla.register_translation(bcoo_todense_p, xla.lower_fun(
    _bcoo_todense_impl, multiple_results=False, new_style=True))

#--------------------------------------------------------------------
# bcoo_fromdense

bcoo_fromdense_p = core.Primitive('bcoo_fromdense')
bcoo_fromdense_p.multiple_results = True

_TRACED_NSE_ERROR = """
The error arose for the nse argument of bcoo_fromdense. In order for BCOO.fromdense()
to be used in traced/compiled code, you must pass a concrete value to the nse
(number of specified elements) argument.
"""

def bcoo_fromdense(mat, *, nse=None, n_batch=0, n_dense=0, index_dtype=jnp.int32):
  """Create COO-format sparse matrix from a dense matrix.

  Args:
    mat : array to be converted to COO, with ``ndim = n_batch + n_sparse + n_dense``.
    nse : number of specified elements in each batch
    n_batch : number of batch dimensions (default: 0)
    n_dense : number of block_dimensions (default: 0)
    index_dtype : dtype of sparse indices (default: int32)

  Returns:
    data : array of shape ``mat.shape[:n_batch] + (nse,) + mat.shape[mat.ndim - n_dense:]``
      and dtype ``mat.dtype``
    indices : array of shape ``mat.shape[:n_batch] + (n_sparse, nse)``
  """
  mat = jnp.asarray(mat)
  if nse is None:
    nse = _bcoo_nse(mat, n_batch, n_dense)
  nse = core.concrete_or_error(operator.index, nse, _TRACED_NSE_ERROR)
  return bcoo_fromdense_p.bind(mat, nse=nse, n_batch=n_batch, n_dense=n_dense,
                               index_dtype=index_dtype)

@bcoo_fromdense_p.def_impl
def _bcoo_fromdense_impl(mat, *, nse, n_batch, n_dense, index_dtype):
  mat = jnp.asarray(mat)
  n_sparse = mat.ndim - n_dense - n_batch
  mask = (mat != 0)
  if n_dense > 0:
    mask = mask.any([-(i + 1) for i in range(n_dense)])
  def _nonzero(a):
    if a.ndim:
      return jnp.nonzero(a, size=nse, fill_value=a.shape[:n_sparse])
    return ()
  for _ in range(n_batch):
    _nonzero = vmap(_nonzero, 0)
  indices = _nonzero(mask)
  if not indices:
    indices = jnp.zeros(mask.shape[:n_batch] + (nse, 0), index_dtype)
  else:
    indices = jnp.moveaxis(jnp.array(indices, index_dtype), 0, n_batch + 1)
  data = bcoo_extract(indices, mat)

  true_nonzeros = jnp.arange(nse) < mask.sum(list(range(n_batch, mask.ndim)))[..., None]
  true_nonzeros = true_nonzeros[(n_batch + 1) * (slice(None),) + n_dense * (None,)]
  data = jnp.where(true_nonzeros, data, 0)

  return data, indices

@bcoo_fromdense_p.def_abstract_eval
def _bcoo_fromdense_abstract_eval(mat, *, nse, n_batch, n_dense, index_dtype):
  n_sparse = mat.ndim - n_batch - n_dense
  data_shape = mat.shape[:n_batch] + (nse,) + mat.shape[n_batch + n_sparse:]
  index_shape = mat.shape[:n_batch] + (nse, n_sparse)
  return core.ShapedArray(data_shape, mat.dtype), core.ShapedArray(index_shape, index_dtype)

def _bcoo_fromdense_jvp(primals, tangents, *, nse, n_batch, n_dense, index_dtype):
  M, = primals
  Mdot, = tangents

  primals_out = bcoo_fromdense(M, nse=nse, n_batch=n_batch, n_dense=n_dense, index_dtype=index_dtype)
  data, indices = primals_out

  if type(Mdot) is ad.Zero:
    data_dot = ad.Zero.from_value(data)
  else:
    data_dot = bcoo_extract(indices, Mdot)

  tangents_out = (data_dot, ad.Zero.from_value(indices))

  return primals_out, tangents_out

def _bcoo_fromdense_transpose(ct, M, *, nse, n_batch, n_dense, index_dtype):
  data, indices = ct
  n_sparse = M.ndim = n_batch - n_dense
  assert data.shape == M.shape[:n_batch] + (nse,) + M.shape[n_batch + n_sparse:]
  assert indices.shape == M.shape[:n_batch] + (n_sparse, nse)
  assert indices.dtype == index_dtype
  if isinstance(indices, ad.Zero):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ad.is_undefined_primal(M)
  return bcoo_todense(data, indices, spinfo=BCOOInfo(M.aval.shape))

def _bcoo_fromdense_batching_rule(batched_args, batch_dims, *, nse, n_batch, n_dense, index_dtype):
  M, = batched_args
  if batch_dims != (0,):
    raise NotImplementedError(f"batch_dims={batch_dims}")
  return bcoo_fromdense(M, nse=nse, n_batch=n_batch + 1, n_dense=n_dense, index_dtype=index_dtype), (0, 0)

ad.primitive_jvps[bcoo_fromdense_p] = _bcoo_fromdense_jvp
ad.primitive_transposes[bcoo_fromdense_p] = _bcoo_fromdense_transpose
batching.primitive_batchers[bcoo_fromdense_p] = _bcoo_fromdense_batching_rule
xla.register_translation(bcoo_fromdense_p, xla.lower_fun(
    _bcoo_fromdense_impl, multiple_results=True, new_style=True))

#----------------------------------------------------------------------
# bcoo_extract

bcoo_extract_p = core.Primitive('bcoo_extract')

def bcoo_extract(indices, mat):
  """Extract BCOO values from dense matrix `mat` at given BCOO indices."""
  return bcoo_extract_p.bind(indices, mat)

@bcoo_extract_p.def_impl
def _bcoo_extract_impl(indices, mat):
  mat = jnp.asarray(mat)
  n_batch, n_sparse, _, _ = _validate_bcoo_indices(indices, mat.shape)

  ind_slices = tuple(np.zeros(s, int) if i_s == 1 else np.arange(s)
                     for s, i_s in zip(mat.shape[:n_batch], indices.shape[:n_batch]))
  grid = tuple(np.meshgrid(*ind_slices, indexing='ij', sparse=True))
  sparse_ind = tuple(indices[grid + (slice(None), i)] for i in range(n_sparse))

  batch_slices = tuple(np.arange(s) for s in mat.shape[:n_batch])
  grid = np.meshgrid(*batch_slices, np.arange(1), indexing='ij', sparse=True)
  batch_ind = tuple(grid)[:-1]

  if not sparse_ind + batch_ind:
    return mat[None]
  return mat.at[batch_ind + sparse_ind].get(mode='fill', fill_value=0)

@bcoo_extract_p.def_abstract_eval
def _bcoo_extract_abstract_eval(indices, mat):
  n_batch, _, n_dense, nse = _validate_bcoo_indices(indices, mat.shape)
  out_shape = mat.shape[:n_batch] + (nse,) + mat.shape[mat.ndim - n_dense:]
  return core.ShapedArray(out_shape, mat.dtype)

def _bcoo_extract_jvp(mat_dot, indices, mat):
  assert mat_dot.shape == mat.shape
  return bcoo_extract(indices, mat_dot)

def _bcoo_extract_transpose(ct, indices, mat):
  assert ad.is_undefined_primal(mat)
  if ad.is_undefined_primal(indices):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ct.dtype == mat.aval.dtype
  return indices, bcoo_todense(ct, indices, spinfo=BCOOInfo(mat.aval.shape))

def _bcoo_extract_batching_rule(batched_args, batch_dims):
  indices, mat = batched_args
  assert any(b is not None for b in batch_dims)
  if batch_dims[0] is None:
    bdim = batch_dims[1]
    indices = lax.expand_dims(indices, (bdim,))
  elif batch_dims[1] is None:
    bdim = batch_dims[0]
    mat = lax.expand_dims(mat, (bdim,))
  else:
    assert batch_dims[0] == batch_dims[1]
    bdim = batch_dims[0]
  n_batch = indices.ndim - 2
  if bdim >= n_batch:
    raise ValueError(f"batch_dims={batch_dims} out of range for indices with n_batch={n_batch}")
  return bcoo_extract(indices, mat), bdim

ad.defjvp(bcoo_extract_p, None, _bcoo_extract_jvp)
ad.primitive_transposes[bcoo_extract_p] = _bcoo_extract_transpose
batching.primitive_batchers[bcoo_extract_p] = _bcoo_extract_batching_rule
xla.register_translation(bcoo_extract_p, xla.lower_fun(
    _bcoo_extract_impl, multiple_results=False, new_style=True))

#----------------------------------------------------------------------
# bcoo_transpose
# transpose of a BCOO array

bcoo_transpose_p = core.Primitive('bcoo_transpose')
bcoo_transpose_p.multiple_results = True

def bcoo_transpose(data, indices, *, permutation: Sequence[int], spinfo: BCOOInfo):
  permutation = tuple(permutation)
  if permutation == tuple(range(len(spinfo.shape))):
    return data, indices
  else:
    return bcoo_transpose_p.bind(data, indices, permutation=permutation,
                                 spinfo=spinfo)

def _validate_permutation(data, indices, permutation, shape):
  if not isinstance(permutation, (tuple, list, np.ndarray)):
    raise TypeError(f"transpose permutation must be a tuple/list/ndarray, got {type(permutation)}.")
  if tuple(sorted(permutation)) != tuple(range(len(shape))):
    raise TypeError("transpose permutation isn't a permutation of operand dimensions, "
                    f"got permutation {permutation} for shape {shape}.")
  n_batch, n_sparse, n_dense, _ = _validate_bcoo(data, indices, shape)
  batch_perm = permutation[:n_batch]
  sparse_perm = [p - n_batch for p in permutation[n_batch: n_batch + n_sparse]]
  dense_perm = [p - n_sparse - n_batch for p in permutation[n_batch + n_sparse:]]
  if n_batch and tuple(sorted(batch_perm)) != tuple(range(n_batch)):
    raise NotImplementedError("transpose permutation cannot permute batch axes with non-batch axes; "
                              f"got permutation {permutation}, with n_batch={n_batch}.")
  if n_dense and tuple(sorted(dense_perm)) != tuple(range(n_dense)):
    raise NotImplementedError("transpose permutation cannot permute dense axes with non-dense axes; "
                              f"got permutation {permutation}, with n_dense={n_dense}.")
  return batch_perm, sparse_perm, dense_perm

@bcoo_transpose_p.def_impl
def _bcoo_transpose_impl(data, indices, *, permutation: Sequence[int], spinfo: BCOOInfo):
  batch_perm, sparse_perm, dense_perm = _validate_permutation(data, indices, permutation, spinfo.shape)
  n_batch = len(batch_perm)
  indices = indices[..., sparse_perm].transpose(*batch_perm, n_batch, n_batch + 1)
  data = data.transpose(*batch_perm, n_batch, *(d + n_batch + 1 for d in dense_perm))
  return data, indices

@bcoo_transpose_p.def_abstract_eval
def _bcoo_transpose_abstract_eval(data, indices, *, permutation: Sequence[int], spinfo: BCOOInfo):
  batch_perm, _, dense_perm = _validate_permutation(data, indices, permutation, spinfo.shape)
  n_batch = len(batch_perm)
  indices_shape = np.array(indices.shape)[[*batch_perm, n_batch, n_batch + 1]]
  data_shape = np.array(data.shape)[[*batch_perm, n_batch, *(d + n_batch + 1 for d in dense_perm)]]
  return core.ShapedArray(data_shape, data.dtype), core.ShapedArray(indices_shape, indices.dtype)

def _bcoo_transpose_jvp(primals, tangents, *, permutation: Sequence[int], spinfo: BCOOInfo):
  data, indices = primals
  data_dot, _ = tangents
  primals_out = bcoo_transpose(data, indices, permutation=permutation, spinfo=spinfo)
  data_dot_out, _ = bcoo_transpose(data_dot, indices, permutation=permutation, spinfo=spinfo)
  return primals_out, (data_dot_out, ad.Zero.from_value(indices))

def _bcoo_transpose_transpose(ct, data, indices, *, permutation: Sequence[int], spinfo: BCOOInfo):
  data_ct, indices_ct = ct
  assert isinstance(indices_ct, ad.Zero)
  if ad.is_undefined_primal(indices):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert data_ct.dtype == data.aval.dtype
  ct_spinfo = BCOOInfo(tuple(spinfo.shape[p] for p in permutation))
  rev_permutation = list(np.argsort(permutation))
  # TODO(jakevdp) avoid dummy indices?
  dummy_indices = jnp.zeros([1 for i in range(indices.ndim - 2)] + list(indices.shape[-2:]), dtype=int)
  data_trans, _ = bcoo_transpose(data_ct, dummy_indices, permutation=rev_permutation, spinfo=ct_spinfo)
  return data_trans, indices_ct

def _bcoo_transpose_batch_rule(batched_args, batch_dims, *, permutation: Sequence[int], spinfo: BCOOInfo):
  data, indices = batched_args
  batch_dims = list(batch_dims)
  batch_size = max(0 if dim is None else arg.shape[dim]
                   for arg, dim in zip(batched_args, batch_dims))
  if batch_dims[0] is None:
    data = data[None]
  else:
    assert batch_dims[0] == 0
  if batch_dims[1] is None:
    indices = indices[None]
  else:
    assert batch_dims[1] == 0
  batched_spinfo = BCOOInfo((batch_size, *spinfo.shape))
  batched_permutation = (0, *(p + 1 for p in permutation))
  data, indices = bcoo_transpose(data, indices, permutation=batched_permutation, spinfo=batched_spinfo)
  if batch_dims[0] is None:
    data = data[0]
  if batch_dims[1] is None:
    indices = indices[0]
  return (data, indices), batch_dims

ad.primitive_jvps[bcoo_transpose_p] = _bcoo_transpose_jvp
ad.primitive_transposes[bcoo_transpose_p] = _bcoo_transpose_transpose
batching.primitive_batchers[bcoo_transpose_p] = _bcoo_transpose_batch_rule
xla.register_translation(bcoo_transpose_p, xla.lower_fun(
    _bcoo_transpose_impl, multiple_results=True, new_style=True))

#----------------------------------------------------------------------
# bcoo_dot_general
# (batched) general dot product of a BCOO sparse ND array and a dense ND array,
# returning a dense ND array.

bcoo_dot_general_p = core.Primitive('bcoo_dot_general')

def _dot_general_validated_shape(lhs_shape: Shape, rhs_shape: Shape, dimension_numbers: DotDimensionNumbers) -> Shape:
  """Validate the inputs and return the output shape."""
  lhs = core.ShapedArray(lhs_shape, np.float32)
  rhs = core.ShapedArray(rhs_shape, np.float32)
  return _dot_general_shape_rule(
    lhs, rhs, dimension_numbers=dimension_numbers,
    precision=None, preferred_element_type=None)

def bcoo_dot_general(lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_spinfo: BCOOInfo):
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  cdims = (api_util._ensure_index_tuple(lhs_contract),
           api_util._ensure_index_tuple(rhs_contract))
  bdims = (api_util._ensure_index_tuple(lhs_batch),
           api_util._ensure_index_tuple(rhs_batch))
  return bcoo_dot_general_p.bind(jnp.asarray(lhs_data), jnp.asarray(lhs_indices), jnp.asarray(rhs),
                                 dimension_numbers=(cdims, bdims),
                                 lhs_spinfo=lhs_spinfo)

def bcoo_rdot_general(lhs, rhs_data, rhs_indices, *, dimension_numbers: DotDimensionNumbers, rhs_spinfo: BCOOInfo):
  # TODO(jakevdp): perhaps this should be part of the bcoo_dot_general primitive?
  result = bcoo_dot_general(rhs_data, rhs_indices, lhs, lhs_spinfo=rhs_spinfo,
                            dimension_numbers=[d[::-1] for d in dimension_numbers])
  n_contract, n_batch = (len(d[0]) for d in dimension_numbers)
  n_swap = len(rhs_spinfo.shape) - n_contract
  permutation = tuple([*range(n_batch), *range(n_swap, result.ndim), *range(n_batch, n_swap)])
  return lax.transpose(result, permutation)

@bcoo_dot_general_p.def_impl
def _bcoo_dot_general_impl(lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_spinfo: BCOOInfo):
  lhs_data = jnp.asarray(lhs_data)
  lhs_indices = jnp.asarray(lhs_indices)
  rhs = jnp.asarray(rhs)
  # Validate all inputs via abstract_eval
  out_aval = _bcoo_dot_general_abstract_eval(lhs_data.aval, lhs_indices.aval, rhs.aval,
                                             dimension_numbers=dimension_numbers,
                                             lhs_spinfo=lhs_spinfo)
  n_sparse = lhs_indices.shape[-1]
  n_batch = lhs_indices.ndim - 2

  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_contracting_b, rhs_contracting_b = unzip2([
    (l, r) for l, r in safe_zip(lhs_contracting, rhs_contracting) if l < n_batch])
  lhs_contracting_s, rhs_contracting_s = unzip2([
    (l, r) for l, r in safe_zip(lhs_contracting, rhs_contracting) if l >= n_batch])

  # Reorder lhs batch dimensions
  if lhs_batch or lhs_contracting_b:
    batch_perm = [*lhs_batch, *remaining(range(n_batch), lhs_batch, lhs_contracting_b), *lhs_contracting_b]
    lhs_data = lhs_data.transpose([*batch_perm, *range(n_batch, lhs_data.ndim)])
    lhs_indices = lhs_indices.transpose([*batch_perm, *range(n_batch, lhs_indices.ndim)])

  # Reorder lhs sparse dimensions
  if lhs_contracting_s:
    lhs_contracting_s = [d - n_batch for d in lhs_contracting_s]
    sparse_perm = jnp.array([*lhs_contracting_s, *remaining(range(n_sparse), lhs_contracting_s)])
    lhs_indices = lhs_indices[..., sparse_perm]

  # Reorder rhs dimensions
  rhs_perm = [*rhs_batch, *rhs_contracting_b, *rhs_contracting_s,
              *remaining(range(rhs.ndim), rhs_batch, rhs_contracting)]
  rhs = rhs.transpose(rhs_perm)

  def result(out_array, lhs_data, lhs_indices, rhs):
    idx = tuple(lhs_indices[..., i] for i in range(n_sparse))
    idx_right = idx[:len(lhs_contracting_s)]
    idx_out = idx[len(lhs_contracting_s):]
    if idx_right and lhs_indices.ndim > 2:
      idx_batch = jnp.meshgrid(
          *(jnp.arange(n) for n in lhs_indices.shape[:-1]),
          indexing='ij')[:lhs_indices.ndim - 2]
      idx_right = (*idx_batch, *idx_right)
    batch_dims = list(range(len(lhs_contracting_b) + bool(lhs_contracting_s)))
    prod = lax.dot_general(lhs_data, rhs.at[idx_right].get(mode='fill', fill_value=0),
                           (([], []), (batch_dims, batch_dims)))
    if idx_out:
      return out_array.at[idx_out].add(prod)
    else:
      return prod.sum(tuple(range(prod.ndim - out_array.ndim)), dtype=out_array.dtype)
  for _ in range(n_batch - len(lhs_contracting_b)):
    result = broadcasting_vmap(result)
  rhs = lax.expand_dims(rhs, range(len(rhs_batch), n_batch - len(lhs_contracting_b)))
  out_array = jnp.zeros(out_aval.shape, out_aval.dtype)
  return result(out_array, lhs_data, lhs_indices, rhs)

@bcoo_dot_general_p.def_abstract_eval
def _bcoo_dot_general_abstract_eval(lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_spinfo: BCOOInfo):
  if lhs_data.dtype != rhs.dtype:
    raise ValueError("bcoo_dot_general requires arguments to have matching dtypes; "
                     f"got lhs.dtype={lhs_data.dtype}, rhs.dtype={rhs.dtype}")

  (lhs_contracting, _), (lhs_batch, _) = dimension_numbers
  n_batch, n_sparse, _, _ = _validate_bcoo(lhs_data, lhs_indices, lhs_spinfo.shape)
  out_shape = _dot_general_validated_shape(lhs_spinfo.shape, rhs.shape, dimension_numbers)

  if lhs_batch and max(lhs_batch) >= n_batch:
    raise NotImplementedError(
      "bcoo_dot_general batch dimensions must be among the batch dimensions in the sparse representtaion.\n"
      f"got lhs_batch={lhs_batch}, n_batch={n_batch}")

  # TODO: support contraction of dense dimensions?
  if any(d >= n_batch + n_sparse for d in lhs_contracting):
    raise NotImplementedError("bcoo_dot_general: contracting over dense dimensions.")

  return core.ShapedArray(out_shape, lhs_data.dtype)

def _bcoo_dot_general_jvp_lhs(lhs_data_dot, lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_spinfo: BCOOInfo):
  return bcoo_dot_general(lhs_data_dot, lhs_indices, rhs, dimension_numbers=dimension_numbers, lhs_spinfo=lhs_spinfo)

def _bcoo_dot_general_jvp_rhs(rhs_dot, lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_spinfo: BCOOInfo):
  return bcoo_dot_general(lhs_data, lhs_indices, rhs_dot, dimension_numbers=dimension_numbers, lhs_spinfo=lhs_spinfo)

def _bcoo_dot_general_transpose(ct, lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_spinfo: BCOOInfo):
  assert not ad.is_undefined_primal(lhs_indices)
  if type(ct) is ad.Zero:
    return ad.Zero
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_ndim = len(lhs_spinfo.shape)
  rhs_ndim = rhs.aval.ndim if ad.is_undefined_primal(rhs) else rhs.ndim
  lhs_kept = remaining(range(lhs_ndim), lhs_contract, lhs_batch)
  rhs_kept = remaining(range(rhs_ndim), rhs_contract, rhs_batch)
  ans_batch, ans_lhs, ans_rhs = map(list, ranges_like(lhs_batch, lhs_kept, rhs_kept))
  if ad.is_undefined_primal(lhs_data):
    dims = ((ans_rhs, rhs_kept), (ans_batch, rhs_batch))
    lhs_contract_sorted_by_rhs = list(np.take(lhs_contract, np.argsort(rhs_contract)))
    permutation = list(lhs_batch) + lhs_kept + lhs_contract_sorted_by_rhs
    out_axes = list(np.argsort(permutation))

    # What follows is essentially this, but computed in terms of dot_general_sampled:
    # out_dense_T = lax.dot_general(ct, rhs, dimension_numbers=dims)
    # out_dense = lax.transpose(out_dense_T, out_axes)
    # result = bcoo_extract(lhs_indices, out_dense)

    # Instead we (1) un-transpose indices, (2) compute SDDMM, (3) re-transpose result
    dummy_data = jnp.ones([1 for i in range(lhs_indices.ndim - 2)] + [lhs_indices.shape[-2]])
    dummy_spinfo = BCOOInfo(tuple(lhs_indices.shape[:-2]) + tuple(1 for i in range(lhs_indices.shape[-1])))
    _, lhs_indices_T = bcoo_transpose(dummy_data, lhs_indices, permutation=permutation, spinfo=dummy_spinfo)
    result_T = bcoo_dot_general_sampled(ct, rhs, lhs_indices_T, dimension_numbers=dims)
    result, _ = bcoo_transpose(result_T, lhs_indices_T, permutation=out_axes, spinfo=dummy_spinfo)

    return result, lhs_indices, rhs
  else:
    dims = ((lhs_kept, ans_lhs), (lhs_batch, ans_batch))
    rhs_contract_sorted_by_lhs = list(np.take(rhs_contract, np.argsort(lhs_contract)))
    out_axes = list(np.argsort(list(rhs_batch) + rhs_contract_sorted_by_lhs + rhs_kept))
    result = bcoo_dot_general(lhs_data, lhs_indices, ct, lhs_spinfo=lhs_spinfo, dimension_numbers=dims)
    return lhs_data, lhs_indices, lax.transpose(result, out_axes)

def _bcoo_dot_general_batch_rule(batched_args, batch_dims, *, dimension_numbers, lhs_spinfo: BCOOInfo):
  lhs_data, lhs_indices, rhs = batched_args
  batch_dims = list(batch_dims)
  batch_size = max(0 if dim is None else arg.shape[dim]
                   for arg, dim in zip(batched_args, batch_dims))
  if batch_dims[0] is None:
    lhs_data = lhs_data[None]
    batch_dims[0] = 0
  if batch_dims[1] is None:
    lhs_indices = lhs_indices[None]
    batch_dims[1] = 0
  # TODO: handle different batchings between lhs_data and lhs_indices?
  assert batch_dims[0] == batch_dims[1] == 0
  new_dimension_numbers, result_batch_dim = _dot_general_batch_dim_nums(
      (len(lhs_spinfo.shape), rhs.ndim), (batch_dims[0], batch_dims[2]), dimension_numbers)
  new_shape = (batch_size, *lhs_spinfo.shape)
  batched_out = bcoo_dot_general(lhs_data, lhs_indices, rhs, lhs_spinfo=BCOOInfo(new_shape),
                                 dimension_numbers=new_dimension_numbers)
  return batched_out, result_batch_dim

ad.defjvp(bcoo_dot_general_p, _bcoo_dot_general_jvp_lhs, None, _bcoo_dot_general_jvp_rhs)
ad.primitive_transposes[bcoo_dot_general_p] = _bcoo_dot_general_transpose
batching.primitive_batchers[bcoo_dot_general_p] = _bcoo_dot_general_batch_rule
xla.register_translation(bcoo_dot_general_p, xla.lower_fun(
    _bcoo_dot_general_impl, multiple_results=False, new_style=True))

#----------------------------------------------------------------------
# bcoo_dot_general_sampled
# (batched) general sampled dot product of two dense ND arrays, with
# output computed only at a given set of sparse indices.

bcoo_dot_general_sampled_p = core.Primitive("bcoo_dot_general_sampled")

def bcoo_dot_general_sampled(A, B, indices, *, dimension_numbers):
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  cdims = (api_util._ensure_index_tuple(lhs_contract),
           api_util._ensure_index_tuple(rhs_contract))
  bdims = (api_util._ensure_index_tuple(lhs_batch),
           api_util._ensure_index_tuple(rhs_batch))
  return bcoo_dot_general_sampled_p.bind(A, B, indices,
                                         dimension_numbers=(cdims, bdims))

@bcoo_dot_general_sampled_p.def_impl
def _bcoo_dot_general_sampled_impl(A, B, indices, *, dimension_numbers):
  # TODO(jakevdp): use a more efficient implementation that avoids the full dot product.
  dense_result = lax.dot_general(A, B, dimension_numbers=dimension_numbers)
  return bcoo_extract(indices, dense_result)

@bcoo_dot_general_sampled_p.def_abstract_eval
def _bcoo_dot_general_sampled_abstract_eval(A, B, indices, *, dimension_numbers):
  dense_result, = pe.abstract_eval_fun(lambda *args: [lax.dot_general(*args, dimension_numbers=dimension_numbers)], A, B)
  sparse_result, = pe.abstract_eval_fun(lambda *args: [bcoo_extract(*args)], indices, dense_result)
  return sparse_result

def _bcoo_dot_general_sampled_transpose(ct, A, B, indices, *, dimension_numbers):
  A_shape = A.aval.shape if hasattr(A, 'aval') else A.shape
  B_shape = B.aval.shape if hasattr(B, 'aval') else B.shape
  mat_shape = _dot_general_validated_shape(A_shape, B_shape, dimension_numbers)
  mat = ad.UndefinedPrimal(core.ShapedArray(mat_shape, ct.dtype))
  indices, ct = _bcoo_extract_transpose(ct, indices, mat)
  kwds = {'dimension_numbers': dimension_numbers,
          'precision': None,
          'preferred_element_type': None}
  A, B = ad.get_primitive_transpose(lax.dot_general_p)(ct, A, B, **kwds)
  return A, B, indices

def _bcoo_dot_general_sampled_jvp_A(A_dot, A, B, indices, *, dimension_numbers):
  return bcoo_dot_general_sampled(A_dot, B, indices, dimension_numbers=dimension_numbers)

def _bcoo_dot_general_sampled_jvp_B(B_dot, A, B, indices, *, dimension_numbers):
  return bcoo_dot_general_sampled(A, B_dot, indices, dimension_numbers=dimension_numbers)

def _bcoo_dot_general_sampled_batch_rule(batched_args, batch_dims, *, dimension_numbers):
  def impl(A, B, indices):
    return _bcoo_dot_general_sampled_impl(A, B, indices, dimension_numbers=dimension_numbers)
  return vmap(impl, in_axes=batch_dims, out_axes=0)(*batched_args), 0

ad.defjvp(bcoo_dot_general_sampled_p, _bcoo_dot_general_sampled_jvp_A,
          _bcoo_dot_general_sampled_jvp_B, None)
ad.primitive_transposes[bcoo_dot_general_sampled_p] = _bcoo_dot_general_sampled_transpose
batching.primitive_batchers[bcoo_dot_general_sampled_p] = _bcoo_dot_general_sampled_batch_rule
xla.register_translation(bcoo_dot_general_sampled_p, xla.lower_fun(
    _bcoo_dot_general_sampled_impl, multiple_results=False, new_style=True))

#----------------------------------------------------------------------
# bcoo_spdot_general
# (batched) general dot product of two BCOO sparse arrays returning a
# Dense ND array.

bcoo_spdot_general_p = core.Primitive('bcoo_spdot_general')
bcoo_spdot_general_p.multiple_results = True

def bcoo_spdot_general(lhs_data, lhs_indices, rhs_data, rhs_indices, *, lhs_spinfo: BCOOInfo, rhs_spinfo: BCOOInfo, dimension_numbers: DotDimensionNumbers):
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  cdims = (api_util._ensure_index_tuple(lhs_contract),
           api_util._ensure_index_tuple(rhs_contract))
  bdims = (api_util._ensure_index_tuple(lhs_batch),
           api_util._ensure_index_tuple(rhs_batch))
  return bcoo_spdot_general_p.bind(lhs_data, lhs_indices, rhs_data, rhs_indices,
                                   lhs_spinfo=lhs_spinfo, rhs_spinfo=rhs_spinfo,
                                   dimension_numbers=(cdims, bdims))

def _bcoo_spdot_general_unbatched(lhs_data, lhs_indices, rhs_data, rhs_indices, *, lhs_spinfo, rhs_spinfo, lhs_contracting, rhs_contracting):
  lhs_shape = lhs_spinfo.shape
  rhs_shape = rhs_spinfo.shape

  lhs = _validate_bcoo(lhs_data, lhs_indices, lhs_shape)
  rhs = _validate_bcoo(rhs_data, rhs_indices, rhs_shape)

  assert lhs.n_batch == rhs.n_batch == 0
  assert lhs.n_dense == rhs.n_dense == 0
  assert [lhs_shape[d] for d in lhs_contracting] == [rhs_shape[d] for d in rhs_contracting]
  assert max(lhs_contracting, default=-1) < lhs.n_sparse
  assert max(rhs_contracting, default=-1) < rhs.n_sparse

  out_shape = (
    [s for i, s in enumerate(lhs_shape) if i not in lhs_contracting] +
    [s for i, s in enumerate(rhs_shape) if i not in rhs_contracting])

  lhs_i = lhs_indices[:, jnp.array(lhs_contracting, dtype=int)]
  rhs_i = rhs_indices[:, jnp.array(rhs_contracting, dtype=int)]
  lhs_j = lhs_indices[:, jnp.array(remaining(range(lhs.n_sparse), lhs_contracting), dtype=int)]
  rhs_j = rhs_indices[:, jnp.array(remaining(range(rhs.n_sparse), rhs_contracting), dtype=int)]

  # TODO(jakevdp): can we do this more efficiently than using an outer product? Note that
  #   jnp.isin() currently doesn't help much, because it also does all() over an outer
  #   comparison.
  overlap = (lhs_i[:, None] == rhs_i[None, :]).all(-1)
  lhs_valid = (lhs_i < jnp.array([lhs_shape[d] for d in lhs_contracting])).all(-1)
  rhs_valid = (rhs_i < jnp.array([rhs_shape[d] for d in rhs_contracting])).all(-1)
  out_data = jnp.where(overlap & lhs_valid[:, None] & rhs_valid,
                       lhs_data[:, None] * rhs_data[None, :], 0).ravel()

  out_indices = jnp.empty([lhs.nse, rhs.nse, lhs_j.shape[-1] + rhs_j.shape[-1]],
                          dtype=jnp.result_type(lhs_indices, rhs_indices))
  out_indices = out_indices.at[:, :, :lhs_j.shape[-1]].set(lhs_j[:, None])
  out_indices = out_indices.at[:, :, lhs_j.shape[-1]:].set(rhs_j[None, :])
  out_indices = out_indices.reshape(len(out_data), out_indices.shape[-1])
  out_nse = (lhs.nse if lhs_j.shape[1] else 1) * (rhs.nse if rhs_j.shape[1] else 1)
  return _bcoo_sum_duplicates(out_data, out_indices, out_shape, nse=out_nse)

@bcoo_spdot_general_p.def_impl
def _bcoo_spdot_general_impl(lhs_data, lhs_indices, rhs_data, rhs_indices, *, lhs_spinfo: BCOOInfo, rhs_spinfo: BCOOInfo, dimension_numbers):
  lhs_shape = lhs_spinfo.shape
  rhs_shape = rhs_spinfo.shape

  lhs = _validate_bcoo(lhs_data, lhs_indices, lhs_shape)
  rhs = _validate_bcoo(rhs_data, rhs_indices, rhs_shape)
  assert lhs.n_dense == rhs.n_dense == 0
  data_aval, indices_aval = _bcoo_spdot_general_abstract_eval(
    lhs_data.aval, lhs_indices.aval, rhs_data.aval, rhs_indices.aval,
    lhs_spinfo=lhs_spinfo, rhs_spinfo=rhs_spinfo, dimension_numbers=dimension_numbers)
  out_shape = _dot_general_validated_shape(lhs_shape, rhs_shape, dimension_numbers)
  _validate_bcoo(data_aval, indices_aval, out_shape)

  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers

  # Move batch dimensions to front of each array.
  lhs_batch_perm = [*lhs_batch, *remaining(range(lhs.n_batch), lhs_batch)]
  rhs_batch_perm = [*rhs_batch, *remaining(range(rhs.n_batch), rhs_batch)]
  lhs_data = lhs_data.transpose([*lhs_batch_perm, *range(lhs.n_batch, lhs_data.ndim)])
  rhs_data = rhs_data.transpose([*rhs_batch_perm, *range(rhs.n_batch, rhs_data.ndim)])
  lhs_indices = lhs_indices.transpose([*lhs_batch_perm, *range(lhs.n_batch, lhs_indices.ndim)])
  rhs_indices = rhs_indices.transpose([*rhs_batch_perm, *range(rhs.n_batch, rhs_indices.ndim)])

  # Implement batched dot product via vmap
  func = functools.partial(_bcoo_spdot_general_unbatched,
      lhs_spinfo=BCOOInfo(lhs_shape[lhs.n_batch:]),
      rhs_spinfo=BCOOInfo(rhs_shape[rhs.n_batch:]),
      lhs_contracting=[d - lhs.n_batch for d in lhs_contracting],
      rhs_contracting=[d - rhs.n_batch for d in rhs_contracting])

  for _ in reversed(range(len(rhs_batch), rhs.n_batch)):
    func = broadcasting_vmap(func, in_axes=(None, None, 0, 0))
  for _ in reversed(range(len(lhs_batch), lhs.n_batch)):
    func = broadcasting_vmap(func, in_axes=(0, 0, None, None))
  for _ in range(len(lhs_batch)):
    func = broadcasting_vmap(func, in_axes=0)
  return func(lhs_data, lhs_indices, rhs_data, rhs_indices)

@bcoo_spdot_general_p.def_abstract_eval
def _bcoo_spdot_general_abstract_eval(lhs_data, lhs_indices, rhs_data, rhs_indices, *, lhs_spinfo: BCOOInfo, rhs_spinfo: BCOOInfo, dimension_numbers):
  lhs_shape = lhs_spinfo.shape
  rhs_shape = rhs_spinfo.shape

  if lhs_data.dtype != rhs_data.dtype:
    raise ValueError("bcoo_spdot_general requires inputs to have matching dtypes; "
                     f"got lhs.dtype={lhs_data.dtype}, rhs.dtype={rhs_data.dtype}")
  lhs = _validate_bcoo(lhs_data, lhs_indices, lhs_shape)
  rhs = _validate_bcoo(rhs_data, rhs_indices, rhs_shape)
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  _ = _dot_general_validated_shape(lhs_shape, rhs_shape, dimension_numbers)

  if lhs.n_dense or rhs.n_dense:
    # TODO(jakevdp): handle dense dimensions
    raise NotImplementedError("bcoo_spdot_general with dense dimensions.")

  if (lhs_batch and max(lhs_batch) >= lhs.n_batch) or (rhs_batch and max(rhs_batch) >= rhs.n_batch):
    raise NotImplementedError("bcoo_spdot_general: batch_dims must correspond to batch dimensions of the sparse representation.")

  if lhs_contracting and (min(lhs_contracting) < lhs.n_batch or max(lhs_contracting) >= lhs.n_batch + lhs.n_sparse):
    raise NotImplementedError("bcoo_spdot_general only supports contraction of sparse indices.")

  if rhs_contracting and (min(rhs_contracting) < rhs.n_batch or max(rhs_contracting) >= rhs.n_batch + rhs.n_sparse):
    raise NotImplementedError("bcoo_spdot_general only supports contraction of sparse indices.")

  if rhs.n_batch > len(rhs_batch) and lhs.n_sparse > len(lhs_contracting):
    raise ValueError("bcoo_spdot_general: cannot have unused batch dims on rhs with unused sparse dims on lhs.")

  out_nse = (
    (lhs.nse if lhs.n_sparse > len(lhs_contracting) else 1) *
    (rhs.nse if rhs.n_sparse > len(rhs_contracting) else 1)
  )

  data_shape = (
    *(lhs_shape[dim] for dim in lhs_batch),
    *(lhs_data.shape[dim] for dim in range(lhs.n_batch) if dim not in lhs_batch),
    *(rhs_data.shape[dim] for dim in range(rhs.n_batch) if dim not in rhs_batch),
    out_nse)
  indices_shape = (
    *(lhs_shape[dim] for dim in lhs_batch),
    *(lhs_indices.shape[dim] for dim in range(lhs.n_batch) if dim not in lhs_batch),
    *(rhs_indices.shape[dim] for dim in range(rhs.n_batch) if dim not in rhs_batch),
    out_nse, lhs.n_sparse + rhs.n_sparse - 2 * len(lhs_contracting))
  return core.ShapedArray(data_shape, lhs_data.dtype), core.ShapedArray(indices_shape, lhs_indices.dtype)

def _bcoo_spdot_general_batch_rule(batched_args, batch_dims, *, lhs_spinfo: BCOOInfo, rhs_spinfo: BCOOInfo, dimension_numbers):
  lhs_shape = lhs_spinfo.shape
  rhs_shape = rhs_spinfo.shape

  lhs_data, lhs_indices, rhs_data, rhs_indices = batched_args
  batch_dims = list(batch_dims)
  batch_size = max(0 if dim is None else arg.shape[dim]
                   for arg, dim in zip(batched_args, batch_dims))
  if batch_dims[0] is None:
    lhs_data = lhs_data[None]
    batch_dims[0] = 0
  if batch_dims[1] is None:
    lhs_indices = lhs_indices[None]
    batch_dims[1] = 0
  assert batch_dims[0] == batch_dims[1] == 0
  if batch_dims[2] is None:
    rhs_data = rhs_data[None]
    batch_dims[2] = 0
  if batch_dims[3] is None:
    rhs_indices = rhs_indices[None]
    batch_dims[3] = 0
  if any(dim != 0 for dim in batch_dims):
    raise NotImplementedError("batching along non-leading dimension.")
  assert all(dim == 0 for dim in batch_dims)
  new_dimension_numbers, result_batch_dim = _dot_general_batch_dim_nums(
      (len(lhs_shape), len(rhs_shape)), (batch_dims[0], batch_dims[2]), dimension_numbers)
  new_lhs_shape = (batch_size, *lhs_shape)
  new_rhs_shape = (batch_size, *rhs_shape)
  batched_out = bcoo_spdot_general(lhs_data, lhs_indices, rhs_data, rhs_indices,
                                   dimension_numbers=new_dimension_numbers,
                                   lhs_spinfo=BCOOInfo(new_lhs_shape),
                                   rhs_spinfo=BCOOInfo(new_rhs_shape))
  return batched_out, (result_batch_dim, result_batch_dim)


def _bcoo_spdot_general_jvp(primals, tangents, **kwds):
  lhs_data, lhs_indices, rhs_data, rhs_indices = primals
  lhs_data_dot, lhs_indices_dot, rhs_data_dot, rhs_indices_dot = tangents
  primals_out = bcoo_spdot_general(*primals, **kwds)
  assert type(lhs_indices_dot) is ad.Zero
  assert type(rhs_indices_dot) is ad.Zero
  data_dot_out = 0
  if type(lhs_data_dot) is not ad.Zero:
    data_dot_out += bcoo_spdot_general(lhs_data_dot, lhs_indices, rhs_data, rhs_indices, **kwds)[0]
  if type(rhs_data_dot) is not ad.Zero:
    data_dot_out += bcoo_spdot_general(lhs_data, lhs_indices, rhs_data_dot, rhs_indices, **kwds)[0]
  return primals_out, [data_dot_out, ad.Zero.from_value(primals_out[1])]

# TODO(JVP): transpose rule
batching.primitive_batchers[bcoo_spdot_general_p] = _bcoo_spdot_general_batch_rule
ad.primitive_jvps[bcoo_spdot_general_p] = _bcoo_spdot_general_jvp
xla.register_translation(bcoo_spdot_general_p, xla.lower_fun(
    _bcoo_spdot_general_impl, multiple_results=True, new_style=True))

#----------------------------------------------------------------------
# BCOO functions that maybe should be primitives?

def bcoo_broadcast_in_dim(data, indices, *, spinfo, shape, broadcast_dimensions):
  """BCOO equivalent of lax.broadcast_in_dim"""
  if len(spinfo.shape) != len(broadcast_dimensions):
    raise ValueError(f"spinfo.shape={spinfo.shape} and broadcast_dimensions={broadcast_dimensions} must have the same length")
  props = _validate_bcoo(data, indices, spinfo.shape)
  batch_dims, sparse_dims, dense_dims = split_list(broadcast_dimensions, [props.n_batch, props.n_sparse])

  if max(batch_dims, default=0) > min(sparse_dims, default=len(shape)):
    raise ValueError("Cannot mix batch and sparse dimensions during broadcast_in_dim")
  if max(sparse_dims, default=0) > min(dense_dims, default=len(shape)):
    raise ValueError("Cannot mix sparse and dense dimensions during broadcast_in_dim")

  new_n_batch = props.n_batch and 1 + max(broadcast_dimensions[:props.n_batch])
  new_n_dense = props.n_dense and len(shape) - min(broadcast_dimensions[-props.n_dense:])
  new_n_sparse = len(shape) - new_n_batch - new_n_dense

  if np.prod(spinfo.shape[props.n_batch: props.n_batch + props.n_sparse]) != np.prod(shape[new_n_batch:new_n_batch + new_n_sparse]):
    raise NotImplementedError("Adding sparse dimensions with lengths != 1")
  nse = props.nse

  # batch & dense dimensions
  new_data = lax.broadcast_in_dim(data,
      shape=(*shape[:new_n_batch], nse, *shape[new_n_batch + new_n_sparse:]),
      broadcast_dimensions=(*batch_dims, new_n_batch, *(b + 1 - new_n_sparse for b in dense_dims)))
  new_indices = lax.broadcast_in_dim(indices,
      shape=(*shape[:new_n_batch], nse, props.n_sparse),
      broadcast_dimensions=(*batch_dims, new_n_batch, new_n_batch + 1))

  # sparse dimensions
  new_indices = (jnp.zeros_like(new_indices, shape=(*shape[:new_n_batch], nse, new_n_sparse))
                   .at[..., jnp.array(sparse_dims, int) - new_n_batch].set(new_indices))

  return new_data, new_indices

def _tuple_replace(tup, ind, val):
  return tuple(val if i == ind else t for i, t in enumerate(tup))

def bcoo_reduce_sum(data, indices, *, spinfo, axes):
  shape = spinfo.shape
  assert all(0 <= a < len(shape) for a in axes)
  n_batch, n_sparse, _, nse = _validate_bcoo(data, indices, shape)
  axes = sorted(set(axes))

  # Sum over dense dimensions -> sum over data
  dense_axes = tuple(ax - n_sparse + 1 for ax in axes if ax >= n_batch + n_sparse)
  data = data.sum(dense_axes)
  if n_sparse:
    # zero-out data corresponding to invalid indices.
    sparse_shape = jnp.array(shape[n_batch: n_batch + n_sparse])
    mask = jnp.all(indices < sparse_shape, -1)
    if data.ndim > mask.ndim:
      mask = lax.expand_dims(mask, tuple(range(mask.ndim, data.ndim)))
    data = jnp.where(mask, data, 0)

  # Sum over sparse dimensions -> drop index; sum is implicit
  sparse_idx = [i for i in range(n_sparse) if i + n_batch not in axes]
  if not sparse_idx:
    indices = jnp.zeros(_tuple_replace(indices.shape, n_batch + 1, 0), indices.dtype)
  else:
    indices = indices[..., np.array(sparse_idx)]

  # Sum over batch dimensions -> reshape into nse
  batch_axes = {ax for ax in axes if ax < n_batch}

  # First handle broadcasted batch dimensions
  for ax in batch_axes:
    if data.shape[ax] == 1:
      if indices.shape[ax] == 1:
        data = data * shape[ax]
      else:
        data = lax.broadcast_in_dim(data, _tuple_replace(data.shape, ax, shape[ax]), tuple(range(data.ndim)))
    else:
      if indices.shape[ax] == 1:
        data = data.sum(ax)
    assert data.shape[ax] == indices.shape[ax]

  new_batch_dims = tuple(sorted(set(range(n_batch)) - batch_axes))
  new_batch_shape = tuple(data.shape[i] for i in new_batch_dims)
  new_nse = int(nse * np.prod([data.shape[i] for i in batch_axes]))

  data = lax.reshape(data,
                     (*new_batch_shape, new_nse, *data.shape[n_batch + 1:]),
                     (*new_batch_dims, *batch_axes, *range(n_batch, data.ndim)))
  indices = lax.reshape(indices,
                        (*new_batch_shape, new_nse, *indices.shape[n_batch + 1:]),
                        (*new_batch_dims, *batch_axes, *range(n_batch, indices.ndim)))

  out_shape = tuple(shape[i] for i in range(len(shape)) if i not in axes)
  return data, indices, out_shape

def bcoo_multiply_sparse(lhs_data, lhs_indices, rhs_data, rhs_indices, *, lhs_spinfo, rhs_spinfo):
  lhs_shape = lhs_spinfo.shape
  rhs_shape = rhs_spinfo.shape

  lhs = _validate_bcoo(lhs_data, lhs_indices, lhs_shape)
  rhs = _validate_bcoo(rhs_data, rhs_indices, rhs_shape)
  if len(lhs_shape) != len(rhs_shape):
    # Similar requirement as lax.mul:
    raise TypeError("bcoo_multiply_sparse: arrays must have same number of dimensions, "
                    f"got {lhs_shape}, {rhs_shape}")
  if (lhs.n_batch, lhs.n_sparse, lhs.n_dense) != (rhs.n_batch, rhs.n_sparse, rhs.n_dense):
    raise NotImplementedError("bcoo_multiply_sparse: arrays with differing numbers of "
                              f"batch & dense dimensions: {lhs}, {rhs}")
  _mul = functools.partial(_bcoo_multiply_sparse_unbatched,
                           lhs_shape=lhs_shape[lhs.n_batch:],
                           rhs_shape=rhs_shape[rhs.n_batch:])
  for _ in range(lhs.n_batch):
    _mul = broadcasting_vmap(_mul)
  data, indices = _mul(lhs_data, lhs_indices, rhs_data, rhs_indices)
  return data, indices, jnp.broadcast_shapes(lhs_shape, rhs_shape)

def _bcoo_multiply_sparse_unbatched(lhs_data, lhs_indices, rhs_data, rhs_indices, *, lhs_shape, rhs_shape):
  lhs = _validate_bcoo(lhs_data, lhs_indices, lhs_shape)
  rhs = _validate_bcoo(rhs_data, rhs_indices, rhs_shape)
  assert lhs.n_batch == rhs.n_batch == 0
  dims = jnp.array([i for i, (s1, s2) in enumerate(safe_zip(lhs_shape[:lhs.n_sparse], rhs_shape[:rhs.n_sparse]))
                    if s1 != 1 and s2 != 1], dtype=int)

  # TODO(jakevdp): this nse can be tightened to min(lhs.nse, rhs.nse) if there
  # is no broadcasting and indices are unique.
  nse = lhs.nse * rhs.nse

  # TODO(jakevdp): this is pretty inefficient. Can we do this membership check
  # without constructing the full (lhs.nse, rhs.nse) masking matrix?
  mask = jnp.all(lhs_indices[:, None, dims] == rhs_indices[:, dims], -1)
  i_lhs, i_rhs = jnp.nonzero(mask, size=nse, fill_value=(lhs.nse, rhs.nse))
  data = (lhs_data.at[i_lhs].get(mode='fill', fill_value=0) *
          rhs_data.at[i_rhs].get(mode='fill', fill_value=0))
  indices = jnp.maximum(
      lhs_indices.at[i_lhs].get(mode='fill', fill_value=max(lhs_shape, default=0)),
      rhs_indices.at[i_rhs].get(mode='fill', fill_value=max(rhs_shape, default=0)))
  return data, indices

def bcoo_multiply_dense(data, indices, v, *, spinfo):
  """Broadcasted elementwise multiplication between a BCOO array and a dense array."""
  # TODO(jakevdp): the logic here is similar to bcoo_extract... can we reuse that?
  shape = spinfo.shape
  if v.ndim == 0:
    return lax.mul(data, v)
  if shape == v.shape:
    # Note: due to distributive property, no deduplication necessary!
    return lax.mul(data, bcoo_extract(indices, v))

  if lax.broadcast_shapes(v.shape, shape) != shape:
    raise NotImplementedError(
      "multiplication between sparse and dense is only implemented for cases "
      "where the output shape matches the sparse matrix shape. Got "
      f"shape={shape}, v.shape={v.shape}")
  v = lax.expand_dims(v, range(len(shape) - v.ndim))

  props = _validate_bcoo(data, indices, shape)

  def _mul(data, indices, v):
    assert indices.shape[1] == v.ndim - props.n_dense
    ind = tuple(indices[:, i] for i in range(indices.shape[1]))
    ind = tuple(i if s != 1 else 0 for i, s in zip(ind, v.shape))
    return data * v[ind]
  for _ in range(props.n_batch):
    _mul = broadcasting_vmap(_mul)
  return _mul(data, indices, v)

@tree_util.register_pytree_node_class
class BCOO(JAXSparse):
  """Experimental batched COO matrix implemented in JAX

  Args:
    (data, indices) : data and indices in batched COO format.
    shape : shape of sparse array.

  Attributes:
    data : ndarray of shape ``[*batch_dims, nse, *dense_dims]`` containing the
      explicitly stored data within the sparse matrix.
    indices : ndarray of shape ``[*batch_dims, nse, n_sparse]`` containing the
      indices of the explicitly stored data. Duplicate entries will be summed.

  Examples:
    Create a sparse array from a dense array:

    >>> M = jnp.array([[0., 2., 0.], [1., 0., 4.]])
    >>> M_sp = BCOO.fromdense(M)
    >>> M_sp
    BCOO(float32[2, 3], nse=3)

    Examine the internal representation:

    >>> M_sp.data
    DeviceArray([2., 1., 4.], dtype=float32)
    >>> M_sp.indices
    DeviceArray([[0, 1],
                 [1, 0],
                 [1, 2]], dtype=int32)

    Create a dense array from a sparse array:

    >>> M_sp.todense()
    DeviceArray([[0., 2., 0.],
                 [1., 0., 4.]], dtype=float32)

    Create a sparse array from COO data & indices:

    >>> data = jnp.array([1., 3., 5.])
    >>> indices = jnp.array([[0, 0],
    ...                      [1, 1],
    ...                      [2, 2]])
    >>> mat = BCOO((data, indices), shape=(3, 3))
    >>> mat
    BCOO(float32[3, 3], nse=3)
    >>> mat.todense()
    DeviceArray([[1., 0., 0.],
                 [0., 3., 0.],
                 [0., 0., 5.]], dtype=float32)
  """
  # Note: additional BCOO methods are defined in transform.py

  data: jnp.ndarray
  indices: jnp.ndarray
  shape: Shape
  nse = property(lambda self: self.indices.shape[-2])
  dtype = property(lambda self: self.data.dtype)
  n_batch = property(lambda self: self.indices.ndim - 2)
  n_sparse = property(lambda self: self.indices.shape[-1])
  n_dense = property(lambda self: self.data.ndim - 1 - self.n_batch)
  _info = property(lambda self: BCOOInfo(self.shape))

  def __init__(self, args, *, shape):
    # JAX transforms will sometimes instantiate pytrees with null values, so we
    # must catch that in the initialization of inputs.
    self.data, self.indices = _safe_asarray(args)
    super().__init__(args, shape=shape)

  @classmethod
  def fromdense(cls, mat, *, nse=None, index_dtype=np.int32, n_dense=0, n_batch=0):
    """Create a BCOO array from a (dense) :class:`DeviceArray`."""
    return cls(bcoo_fromdense(mat, nse=nse, index_dtype=index_dtype, n_dense=n_dense, n_batch=n_batch), shape=mat.shape)

  @classmethod
  def from_scipy_sparse(cls, mat, *, index_dtype=None, n_dense=0, n_batch=0):
    """Create a BCOO array from a :mod:`scipy.sparse` array."""
    if n_dense != 0 or n_batch != 0:
      raise NotImplementedError("BCOO.fromscipy with nonzero n_dense/n_batch")
    mat = mat.tocoo()
    data = jnp.asarray(mat.data)
    indices = jnp.column_stack((mat.row, mat.col)).astype(index_dtype or jnp.int32)
    return cls((data, indices), shape=mat.shape)

  @classmethod
  def _empty(cls, shape, *, dtype=None, index_dtype='int32', n_dense=0, n_batch=0):
    """Create an empty BCOO instance. Public method is sparse.empty()."""
    shape = tuple(shape)
    n_sparse = len(shape) - n_dense - n_batch
    if n_sparse < 0 or n_dense < 0 or n_batch < 0:
      raise ValueError("Invalid inputs: shape={shape}, n_dense={n_dense}, n_batch={n_batch}")
    batch_shape, sparse_shape, dense_shape = split_list(shape, [n_batch, n_sparse])
    data = jnp.empty((*batch_shape, 0, *dense_shape), dtype)
    indices = jnp.empty((*batch_shape, 0, n_sparse), index_dtype)
    return cls((data, indices), shape=shape)

  def _unbatch(self):
    """Return an unbatched representation of the BCOO matrix."""
    return BCOO(_unbatch_bcoo(self.data, self.indices, self.shape), shape=self.shape)

  def _dedupe(self):
    warnings.warn("_dedupe() is deprecated. Use sum_duplicates() instead.", FutureWarning)
    return self.sum_duplicates(nse=self.nse)

  def sum_duplicates(self, nse=None, remove_zeros=True):
    """Return a copy of the array with duplicate indices summed.

    Additionally, this operation will result in explicit zero entries removed, and
    indices being sorted in lexicographic order.

    Because the size of the resulting representation depends on the values in the
    arrays, this operation is not compatible with JIT or other transforms. To use
    ``sum_duplicates`` in such cases, you may pass a value to `nse` to specify the
    desired size of the output representation.

    Args:
      nse : integer (optional), if specified, gives the number of specified elements in
        the output sparse representation; if it is larger than the number required, data
        will be padded with zeros and indices will be padded with out-of-bounds values.
        If it is smaller than the number required, data will be silently discarded.
      remove_zeros : bool (default=True). If True, remove explicit zeros from the data
        as part of summing duplicates. If False, then explicit zeros at unique indices
        will remain among the specified elements.
    """
    data, indices = _bcoo_sum_duplicates(self.data, self.indices, self.shape,
                                         nse=nse, remove_zeros=remove_zeros)
    return BCOO((data, indices), shape=self.shape)

  def todense(self):
    """Create a dense version of the array."""
    return bcoo_todense(self.data, self.indices, spinfo=self._info)

  def transpose(self, axes=None):
    """Create a new array containing the transpose."""
    axes = np.arange(self.ndim)[::-1] if axes is None else axes
    data_T, indices_T = bcoo_transpose(self.data, self.indices, permutation=axes, spinfo=self._info)
    shape_T = tuple(self.shape[i] for i in axes)
    return BCOO((data_T, indices_T), shape=shape_T)

  def tree_flatten(self):
    return (self.data, self.indices), self._info._asdict()


# vmappable handlers
def _bcoo_to_elt(cont, _, val, axis):
  if axis is None:
    return val
  if axis >= val.n_batch:
    raise ValueError(f"Cannot map in_axis={axis} for BCOO array with n_batch={val.n_batch}. "
                     "in_axes for batched BCOO operations must correspond to a batch dimension.")
  return BCOO((cont(val.data, axis), cont(val.indices, axis)),
              shape= val.shape[:axis] + val.shape[axis + 1:])

def _bcoo_from_elt(cont, axis_size, elt, axis):
  if axis > elt.n_batch:
    raise ValueError(f"BCOO: cannot add out_axis={axis} for BCOO array with n_batch={elt.n_batch}. "
                     "BCOO batch axes must be a contiguous block of leading dimensions.")
  return BCOO((cont(axis_size, elt.data, axis), cont(axis_size, elt.indices, axis)),
              shape=elt.shape[:axis] + (axis_size,) + elt.shape[axis:])

batching.register_vmappable(BCOO, int, int, _bcoo_to_elt, _bcoo_from_elt, None)
