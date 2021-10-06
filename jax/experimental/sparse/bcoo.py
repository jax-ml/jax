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

import numpy as np

import jax
from jax import core
from jax import dtypes
from jax import lax
from jax import tree_util
from jax import vmap
from jax.interpreters import batching
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
import jax.numpy as jnp
from jax.interpreters import ad
from jax.util import safe_zip
from jax._src.lax.lax import (
  ranges_like, remaining, _dot_general_batch_dim_nums, _dot_general_shape_rule,
  DotDimensionNumbers)
from . import ops

Dtype = Any
Shape = Tuple[int, ...]

#----------------------------------------------------------------------
# BCOO primitives: batched extension of COO.

def _bcoo_nse(mat, n_batch=0, n_dense=0):
  mat = jnp.asarray(mat)
  mask = (mat != 0)
  if n_dense > 0:
    mask = mask.any([-(i + 1) for i in range(n_dense)])
  mask = mask.sum(list(range(n_batch, mask.ndim)))
  return mask.max()

def _dedupe_bcoo(data, indices, shape):
  props = _validate_bcoo(data, indices, shape)
  if indices.shape[:props.n_batch] != data.shape[:props.n_batch]:
    # TODO: handle broadcasted dimensions.
    raise NotImplementedError("dedupe_bcoo for broadcasted dimensions.")
  f = _dedupe_bcoo_one
  for _ in range(props.n_batch):
    f = vmap(f)
  return f(data, indices)

def _dedupe_bcoo_one(data, indices):
  assert indices.ndim == 2
  assert data.shape[:1] == indices.shape[:1]

  if indices.shape[1] == 0:
    return data, indices

  # This is a fixed-size version of jnp.unique() with return_indices=True
  # unique values are zero-filled at the end.
  perm = jnp.lexsort(indices.T[::-1])
  aux = indices[perm]
  mask = jnp.ones(indices.shape[0], dtype=bool)
  mask = mask.at[1:].set(jnp.any(aux[1:] != aux[:-1], 1))
  imask = jnp.cumsum(mask) - 1
  indices_unique = jnp.where(mask[:, None], aux, 0)[jnp.argsort(~mask)]
  inv_idx = jnp.zeros_like(imask).at[perm].set(imask)

  # With the above, de-duping is easy.
  data_unique = jnp.zeros_like(data).at[inv_idx].add(data)
  return data_unique, indices_unique


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


def _validate_bcoo(data: jnp.ndarray, indices: jnp.ndarray, shape: Sequence[int]) -> BCOOProperties:
  assert jnp.issubdtype(indices.dtype, jnp.integer)
  shape = tuple(shape)

  nse, n_sparse = indices.shape[-2:]
  n_batch = indices.ndim - 2
  n_dense = len(shape) - n_batch - n_sparse
  assert n_dense >= 0

  def _compatible(shape1, shape2):
    return all(s1 in (1, s2) for s1, s2 in safe_zip(shape1, shape2))

  if data is not None:
    if not _compatible(data.shape[:n_batch], shape[:n_batch]):
      raise ValueError("data batch dimensions not compatible for "
                      f"data.shape={data.shape}, shape={shape}")
    if data.shape[-(n_dense + 1):] != (nse,) + shape[n_batch + n_sparse:]:
      raise ValueError(f"Invalid data.shape={data.shape} for "
                      f"nse={nse}, n_batch={n_batch}, n_dense={n_dense}")
  if not _compatible(indices.shape[:n_batch], shape[:n_batch]):
    raise ValueError("indices batch dimensions not compatible for "
                     f"indices.shape={indices.shape}, shape={shape}")
  if indices.shape[n_batch:] != (nse, n_sparse):
    raise ValueError(f"Invalid indices.shape={indices.shape} for "
                     f"nse={nse}, n_batch={n_batch}, n_dense={n_dense}")

  return BCOOProperties(n_batch=n_batch, n_sparse=n_sparse, n_dense=n_dense, nse=nse)


#----------------------------------------------------------------------
# bcoo_todense

bcoo_todense_p = core.Primitive('bcoo_todense_p')

def bcoo_todense(data, indices, *, shape):
  """Convert batched sparse matrix to a dense matrix.

  Args:
    data : array of shape ``batch_dims + (nse,) + block_dims``.
    indices : array of shape ``batch_dims + (n_sparse, nse)``
    shape : tuple; the shape of the (batched) matrix. Equal to
      ``batch_dims + sparse_dims + block_dims``
      where ``len(sparse_dims) == n_sparse``

  Returns:
    mat : array with specified shape and dtype matching ``data``
  """
  return bcoo_todense_p.bind(jnp.asarray(data), jnp.asarray(indices), shape=tuple(shape))

@bcoo_todense_p.def_impl
def _bcoo_todense_impl(data, indices, *, shape):
  n_batch, n_sparse, _, _ = _validate_bcoo(data, indices, shape)

  ind_slices = tuple(np.zeros(s, int) if i_s == 1 else np.arange(s)
                     for s, i_s in zip(shape[:n_batch], indices.shape[:n_batch]))
  grid = tuple(np.meshgrid(*ind_slices, indexing='ij', sparse=True))
  sparse_ind = tuple(indices[grid + (slice(None), i)] for i in range(n_sparse))

  batch_slices = tuple(np.arange(s) for s in shape[:n_batch])
  grid = np.meshgrid(*batch_slices, np.arange(1), indexing='ij', sparse=True)
  batch_ind = tuple(grid)[:-1]

  if not sparse_ind:
    data = data.sum(n_batch, keepdims=bool(batch_ind))
  return jnp.zeros(shape, data.dtype).at[batch_ind + sparse_ind].add(data)

@bcoo_todense_p.def_abstract_eval
def _bcoo_todense_abstract_eval(data, indices, *, shape):
  _validate_bcoo(data, indices, shape)
  return core.ShapedArray(shape, data.dtype)

def _bcoo_todense_jvp(data_dot, data, indices, *, shape):
  return bcoo_todense(data_dot, indices, shape=shape)

def _bcoo_todense_transpose(ct, data, indices, *, shape):
  assert ad.is_undefined_primal(data)
  if ad.is_undefined_primal(indices):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ct.shape == shape
  assert ct.dtype == data.aval.dtype
  return bcoo_extract(indices, ct), indices

def _bcoo_todense_batching_rule(batched_args, batch_dims, *, shape):
  data, indices = batched_args
  if any(b not in [0, None] for b in batch_dims):
    raise NotImplementedError(f"batch_dims={batch_dims}. Only 0 and None are supported.")
  if batch_dims[0] is None:
    data = data[None, ...]
  if batch_dims[1] is None:
    indices = indices[None, ...]
  return bcoo_todense(data, indices, shape=(max(data.shape[0], indices.shape[0]), *shape)), 0

ad.defjvp(bcoo_todense_p, _bcoo_todense_jvp, None)
ad.primitive_transposes[bcoo_todense_p] = _bcoo_todense_transpose
batching.primitive_batchers[bcoo_todense_p] = _bcoo_todense_batching_rule
xla.translations[bcoo_todense_p] = xla.lower_fun(
    _bcoo_todense_impl, multiple_results=False)

#--------------------------------------------------------------------
# bcoo_fromdense

bcoo_fromdense_p = core.Primitive('bcoo_fromdense')
bcoo_fromdense_p.multiple_results = True

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
  nse = core.concrete_or_error(operator.index, nse, "nse argument of bcoo_fromdense")
  return bcoo_fromdense_p.bind(mat, nse=nse, n_batch=n_batch, n_dense=n_dense,
                               index_dtype=index_dtype)

@bcoo_fromdense_p.def_impl
def _bcoo_fromdense_impl(mat, *, nse, n_batch, n_dense, index_dtype):
  mat = jnp.asarray(mat)
  mask = (mat != 0)
  if n_dense > 0:
    mask = mask.any([-(i + 1) for i in range(n_dense)])
  nonzero = lambda a: jnp.nonzero(a, size=nse) if a.ndim else ()
  for _ in range(n_batch):
    nonzero = vmap(nonzero, 0)
  indices = nonzero(mask)
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
  return bcoo_todense(data, indices, shape=M.aval.shape)

def _bcoo_fromdense_batching_rule(batched_args, batch_dims, *, nse, n_batch, n_dense, index_dtype):
  M, = batched_args
  if batch_dims != (0,):
    raise NotImplementedError(f"batch_dims={batch_dims}")
  return bcoo_fromdense(M, nse=nse, n_batch=n_batch + 1, n_dense=n_dense, index_dtype=index_dtype), (0, 0)

ad.primitive_jvps[bcoo_fromdense_p] = _bcoo_fromdense_jvp
ad.primitive_transposes[bcoo_fromdense_p] = _bcoo_fromdense_transpose
batching.primitive_batchers[bcoo_fromdense_p] = _bcoo_fromdense_batching_rule
xla.translations[bcoo_fromdense_p] = xla.lower_fun(
    _bcoo_fromdense_impl, multiple_results=True)

#----------------------------------------------------------------------
# bcoo_extract

bcoo_extract_p = core.Primitive('bcoo_extract')

def bcoo_extract(indices, mat):
  """Extract BCOO values from dense matrix `mat` at given BCOO indices."""
  return bcoo_extract_p.bind(indices, mat)

@bcoo_extract_p.def_impl
def _bcoo_extract_impl(indices, mat):
  n_batch, n_sparse, _, _ = _validate_bcoo(None, indices, mat.shape)

  ind_slices = tuple(np.zeros(s, int) if i_s == 1 else np.arange(s)
                     for s, i_s in zip(mat.shape[:n_batch], indices.shape[:n_batch]))
  grid = tuple(np.meshgrid(*ind_slices, indexing='ij', sparse=True))
  sparse_ind = tuple(indices[grid + (slice(None), i)] for i in range(n_sparse))

  batch_slices = tuple(np.arange(s) for s in mat.shape[:n_batch])
  grid = np.meshgrid(*batch_slices, np.arange(1), indexing='ij', sparse=True)
  batch_ind = tuple(grid)[:-1]

  if not sparse_ind + batch_ind:
    return mat[None]
  return mat[batch_ind + sparse_ind]

@bcoo_extract_p.def_abstract_eval
def _bcoo_extract_abstract_eval(indices, mat):
  n_batch, _, n_dense, nse = _validate_bcoo(None, indices, mat.shape)
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
  return indices, bcoo_todense(ct, indices, shape=mat.aval.shape)

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
xla.translations[bcoo_extract_p] = xla.lower_fun(
    _bcoo_extract_impl, multiple_results=False)

#----------------------------------------------------------------------
# bcoo_transpose
# transpose of a BCOO array

bcoo_transpose_p = core.Primitive('bcoo_transpose')
bcoo_transpose_p.multiple_results = True

def bcoo_transpose(data, indices, *, permutation, shape):
  if tuple(permutation) == tuple(range(len(shape))):
    return data, indices
  else:
    return bcoo_transpose_p.bind(data, indices, permutation=permutation, shape=shape)

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
def _bcoo_transpose_impl(data, indices, *, permutation: Sequence[int], shape: Tuple[int]):
  batch_perm, sparse_perm, dense_perm = _validate_permutation(data, indices, permutation, shape)
  n_batch = len(batch_perm)
  indices = indices[..., sparse_perm].transpose(*batch_perm, n_batch, n_batch + 1)
  data = data.transpose(*batch_perm, n_batch, *(d + n_batch + 1 for d in dense_perm))
  return data, indices

@bcoo_transpose_p.def_abstract_eval
def _bcoo_transpose_abstract_eval(data, indices, *, permutation: Sequence[int], shape: Tuple[int]):
  batch_perm, _, dense_perm = _validate_permutation(data, indices, permutation, shape)
  n_batch = len(batch_perm)
  indices_shape = np.array(indices.shape)[[*batch_perm, n_batch, n_batch + 1]]
  data_shape = np.array(data.shape)[[*batch_perm, n_batch, *(d + n_batch + 1 for d in dense_perm)]]
  return core.ShapedArray(data_shape, data.dtype), core.ShapedArray(indices_shape, indices.dtype)

def _bcoo_transpose_jvp(primals, tangents, *, permutation, shape):
  data, indices = primals
  data_dot, _ = tangents
  primals_out = bcoo_transpose(data, indices, permutation=permutation, shape=shape)
  data_dot_out, _ = bcoo_transpose(data_dot, indices, permutation=permutation, shape=shape)
  return primals_out, (data_dot_out, ad.Zero.from_value(indices))

def _bcoo_transpose_transpose(ct, data, indices, *, permutation, shape):
  data_ct, indices_ct = ct
  assert isinstance(indices_ct, ad.Zero)
  if ad.is_undefined_primal(indices):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert data_ct.dtype == data.aval.dtype
  ct_shape = tuple(shape[p] for p in permutation)
  rev_permutation = np.argsort(permutation)
  # TODO(jakevdp) avoid dummy indices?
  dummy_indices = jnp.zeros([1 for i in range(indices.ndim - 2)] + list(indices.shape[-2:]), dtype=int)
  data_trans, _ = bcoo_transpose(data_ct, dummy_indices, permutation=rev_permutation, shape=ct_shape)
  return data_trans, indices_ct

def _bcoo_transpose_batch_rule(batched_args, batch_dims, *, permutation, shape):
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
  batched_shape = (batch_size, *shape)
  batched_permutation = (0, *(p + 1 for p in permutation))
  data, indices = bcoo_transpose(data, indices, permutation=batched_permutation, shape=batched_shape)
  if batch_dims[0] is None:
    data = data[0]
  if batch_dims[1] is None:
    indices = indices[0]
  return (data, indices), batch_dims

ad.primitive_jvps[bcoo_transpose_p] = _bcoo_transpose_jvp
ad.primitive_transposes[bcoo_transpose_p] = _bcoo_transpose_transpose
batching.primitive_batchers[bcoo_transpose_p] = _bcoo_transpose_batch_rule
xla.translations[bcoo_transpose_p] = xla.lower_fun(
    _bcoo_transpose_impl, multiple_results=True)

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

def bcoo_dot_general(lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_shape):
  return bcoo_dot_general_p.bind(jnp.asarray(lhs_data), jnp.asarray(lhs_indices), jnp.asarray(rhs),
                                 dimension_numbers=dimension_numbers, lhs_shape=tuple(lhs_shape))

def bcoo_rdot_general(lhs, rhs_data, rhs_indices, *, dimension_numbers, rhs_shape):
  # TODO(jakevdp): perhaps this should be part of the bcoo_dot_general primitive?
  result = bcoo_dot_general(rhs_data, rhs_indices, lhs, lhs_shape=rhs_shape,
                            dimension_numbers=[d[::-1] for d in dimension_numbers])
  n_contract, n_batch = (len(d[0]) for d in dimension_numbers)
  n_swap = len(rhs_shape) - n_contract
  permutation = tuple([*range(n_batch), *range(n_swap, result.ndim), *range(n_batch, n_swap)])
  return lax.transpose(result, permutation)

@bcoo_dot_general_p.def_impl
def _bcoo_dot_general_impl(lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_shape):
  lhs_data = jnp.asarray(lhs_data)
  lhs_indices = jnp.asarray(lhs_indices)
  rhs = jnp.asarray(rhs)
  # Validate all inputs via abstract_eval
  out_aval = _bcoo_dot_general_abstract_eval(lhs_data.aval, lhs_indices.aval, rhs.aval,
                                             dimension_numbers=dimension_numbers,
                                             lhs_shape=lhs_shape)

  (lhs_contracting, rhs_contracting) , (lhs_batch, rhs_batch) = dimension_numbers
  n_sparse = lhs_indices.shape[-1]
  n_batch = lhs_indices.ndim - 2

  # Move lhs batch dimensions to the front
  if lhs_batch:
    perm = list(lhs_batch) + remaining(range(n_batch), lhs_batch)
    lhs_data = lhs_data.transpose(perm + list(range(n_batch, lhs_data.ndim)))
    lhs_indices = lhs_indices.transpose(perm + list(range(n_batch, lhs_indices.ndim)))

  # Move lhs contracting dimensions to the front of sparse dims, in order
  n_contracting = len(lhs_contracting)
  lhs_contracting = [d - n_batch for d in lhs_contracting]
  perm = list(lhs_contracting) + remaining(range(n_sparse), lhs_contracting)
  lhs_indices = lhs_indices[..., jnp.array(perm)]

  # Move rhs batch dimensions then contracting dimensions to the front, in order
  perm = (list(rhs_batch) + list(rhs_contracting) +
          remaining(range(rhs.ndim), rhs_batch, rhs_contracting))
  rhs = rhs.transpose(perm)

  out_array = jnp.zeros(out_aval.shape, out_aval.dtype)
  def result(out_array, lhs_data, lhs_indices, rhs):
    idx = tuple(lhs_indices.T)
    idx_right, idx_out = idx[:n_contracting], idx[n_contracting:]
    ctc = [0] if n_contracting else []
    prod = lax.dot_general(lhs_data, rhs[idx_right], (([], []), (ctc, ctc)))
    return out_array.at[idx_out].add(prod) if idx_out else prod.sum(0, dtype=out_array.dtype)
  for i in range(n_batch)[::-1]:
    axes_in = [0, 0, 0, 0]
    if lhs_data.shape[i] == 1:
      lhs_data = lax.squeeze(lhs_data, (i,))
      axes_in[1] = None
    if lhs_indices.shape[i] == 1:
      lhs_indices = lax.squeeze(lhs_indices, (i,))
      axes_in[2] = None
    if i >= len(lhs_batch):
      axes_in[3] = None
    result = vmap(result, tuple(axes_in))
  return result(out_array, lhs_data, lhs_indices, rhs)

@bcoo_dot_general_p.def_abstract_eval
def _bcoo_dot_general_abstract_eval(lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_shape):
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  n_batch, n_sparse, _, _ = _validate_bcoo(lhs_data, lhs_indices, lhs_shape)
  out_shape = _dot_general_validated_shape(lhs_shape, rhs.shape, dimension_numbers)

  if lhs_batch and max(lhs_batch) >= n_batch:
    raise NotImplementedError(
      "bcoo_dot_general batch dimensions must be among the batch dimensions in the sparse representtaion.\n"
      f"got lhs_batch={lhs_batch}, n_batch={n_batch}")

  # TODO: support constraction of batch dimensions?
  if any(d < n_batch for d in lhs_contracting):
    raise NotImplementedError("bcoo_dot_general: contracting over batch dimensions.")

  # TODO: support contraction of dense dimensions?
  if any(d >= n_batch + n_sparse for d in lhs_contracting):
    raise NotImplementedError("bcoo_dot_general: contracting over dense dimensions.")

  out_dtype = jnp.promote_types(lhs_data.dtype, rhs.dtype)
  return core.ShapedArray(out_shape, out_dtype)

def _bcoo_dot_general_jvp_lhs(lhs_data_dot, lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_shape):
  return bcoo_dot_general(lhs_data_dot, lhs_indices, rhs, dimension_numbers=dimension_numbers, lhs_shape=lhs_shape)

def _bcoo_dot_general_jvp_rhs(rhs_dot, lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_shape):
  return bcoo_dot_general(lhs_data, lhs_indices, rhs_dot, dimension_numbers=dimension_numbers, lhs_shape=lhs_shape)

def _bcoo_dot_general_transpose(ct, lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_shape):
  assert not ad.is_undefined_primal(lhs_indices)
  if type(ct) is ad.Zero:
    return ad.Zero
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_ndim = len(lhs_shape)
  rhs_ndim = rhs.aval.ndim if ad.is_undefined_primal(rhs) else rhs.ndim
  lhs_kept = remaining(range(lhs_ndim), lhs_contract, lhs_batch)
  rhs_kept = remaining(range(rhs_ndim), rhs_contract, rhs_batch)
  ans_batch, ans_lhs, ans_rhs = map(list, ranges_like(lhs_batch, lhs_kept, rhs_kept))
  if ad.is_undefined_primal(lhs_data):
    dims = ((ans_rhs, rhs_kept), (ans_batch, rhs_batch))
    lhs_contract_sorted_by_rhs = list(np.take(lhs_contract, np.argsort(rhs_contract)))
    permutation = list(lhs_batch) + lhs_kept + lhs_contract_sorted_by_rhs
    out_axes = np.argsort(permutation)

    # What follows is essentially this, but computed in terms of dot_general_sampled:
    # out_dense_T = lax.dot_general(ct, rhs, dimension_numbers=dims)
    # out_dense = lax.transpose(out_dense_T, out_axes)
    # result = bcoo_extract(lhs_indices, out_dense)

    # Instead we (1) un-transpose indices, (2) compute SDDMM, (3) re-transpose result
    dummy_data = jnp.ones([1 for i in range(lhs_indices.ndim - 2)] + [lhs_indices.shape[-2]])
    dummy_shape = tuple(lhs_indices.shape[:-2]) + tuple(1 for i in range(lhs_indices.shape[-1]))
    _, lhs_indices_T = bcoo_transpose(dummy_data, lhs_indices, permutation=permutation, shape=dummy_shape)
    result_T = bcoo_dot_general_sampled(ct, rhs, lhs_indices_T, dimension_numbers=dims)
    result, _ = bcoo_transpose(result_T, lhs_indices_T, permutation=out_axes, shape=dummy_shape)

    return result, lhs_indices, rhs
  else:
    dims = ((lhs_kept, ans_lhs), (lhs_batch, ans_batch))
    rhs_contract_sorted_by_lhs = list(np.take(rhs_contract, np.argsort(lhs_contract)))
    out_axes = np.argsort(list(rhs_batch) + rhs_contract_sorted_by_lhs + rhs_kept)
    result = bcoo_dot_general(lhs_data, lhs_indices, ct, lhs_shape=lhs_shape, dimension_numbers=dims)
    return lhs_data, lhs_indices, lax.transpose(result, out_axes)

def _bcoo_dot_general_batch_rule(batched_args, batch_dims, *, dimension_numbers, lhs_shape):
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
      (len(lhs_shape), rhs.ndim), (batch_dims[0], batch_dims[2]), dimension_numbers)
  new_shape = (batch_size, *lhs_shape)
  batched_out = bcoo_dot_general(lhs_data, lhs_indices, rhs, lhs_shape=new_shape,
                                 dimension_numbers=new_dimension_numbers)
  return batched_out, result_batch_dim

ad.defjvp(bcoo_dot_general_p, _bcoo_dot_general_jvp_lhs, None, _bcoo_dot_general_jvp_rhs)
ad.primitive_transposes[bcoo_dot_general_p] = _bcoo_dot_general_transpose
batching.primitive_batchers[bcoo_dot_general_p] = _bcoo_dot_general_batch_rule
xla.translations[bcoo_dot_general_p] = xla.lower_fun(
    _bcoo_dot_general_impl, multiple_results=False)

#----------------------------------------------------------------------
# bcoo_dot_general_sampled
# (batched) general sampled dot product of two dense ND arrays, with
# output computed only at a given set of sparse indices.

bcoo_dot_general_sampled_p = core.Primitive("bcoo_dot_general_sampled")

def bcoo_dot_general_sampled(A, B, indices, *, dimension_numbers):
  return bcoo_dot_general_sampled_p.bind(A, B, indices, dimension_numbers=dimension_numbers)

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
xla.translations[bcoo_dot_general_sampled_p] = xla.lower_fun(
    _bcoo_dot_general_sampled_impl, multiple_results=False)

#----------------------------------------------------------------------
# bcoo_spdot_general
# (batched) general dot product of two BCOO sparse arrays returning a
# Dense ND array.

bcoo_spdot_general_p = core.Primitive('bcoo_spdot_general')
bcoo_spdot_general_p.multiple_results = True

def bcoo_spdot_general(lhs_data, lhs_indices, rhs_data, rhs_indices, *, lhs_shape, rhs_shape, dimension_numbers):
  return bcoo_spdot_general_p.bind(lhs_data, lhs_indices, rhs_data, rhs_indices,
                                   lhs_shape=lhs_shape, rhs_shape=rhs_shape, dimension_numbers=dimension_numbers)

def _bcoo_Mv(lhs_data, lhs_indices, rhs_data, rhs_indices, *, lhs_shape, rhs_shape, dtype, lhs_contract):
  """Helper function to compute the dot product of a sparse array and a sparse vector."""
  lhs = _validate_bcoo(lhs_data, lhs_indices, lhs_shape)
  rhs = _validate_bcoo(rhs_data, rhs_indices, rhs_shape)
  # Inputs should be unbatched; batching is handled by vmapping at the call site.
  assert lhs.n_batch == rhs.n_batch == 0
  assert lhs.n_dense == rhs.n_dense == 0
  assert lhs.n_sparse >= 1
  assert rhs.n_sparse == 1
  assert (lhs_shape[lhs_contract],) == rhs_shape
  rhs_data, rhs_indices = _dedupe_bcoo(rhs_data, rhs_indices, rhs_shape)
  lhs_i = lhs_indices[:, lhs_contract]
  rhs_i = rhs_indices[:, 0]
  mask = jnp.isin(lhs_i, rhs_i, assume_unique=True)
  lhs_i_inv = (lhs_i[None, :] == rhs_i[:, None]).argmax(0)
  out_data = lhs_data.at[jnp.arange(lhs.nse)].mul(jnp.where(mask, rhs_data[lhs_i_inv], 0))
  out_indices = jnp.concatenate([lhs_indices[:, :lhs_contract], lhs_indices[:, lhs_contract + 1:]], axis=1)
  return out_data, out_indices

@bcoo_spdot_general_p.def_impl
def _bcoo_spdot_general_impl(lhs_data, lhs_indices, rhs_data, rhs_indices, *, lhs_shape, rhs_shape, dimension_numbers):
  out_shape = _dot_general_validated_shape(lhs_shape, rhs_shape, dimension_numbers)
  data_aval, indices_aval = _bcoo_spdot_general_abstract_eval(
    lhs_data.aval, lhs_indices.aval, rhs_data.aval, rhs_indices.aval,
    lhs_shape=lhs_shape, rhs_shape=rhs_shape, dimension_numbers=dimension_numbers)
  _validate_bcoo(data_aval, indices_aval, out_shape)
  lhs = _validate_bcoo(lhs_data, lhs_indices, lhs_shape)
  rhs = _validate_bcoo(rhs_data, rhs_indices, rhs_shape)

  # Move batch dimension to front
  (lhs_contracting, _), (lhs_batch, rhs_batch) = dimension_numbers

  lhs_perm = tuple(lhs_batch) + tuple(i for i in range(lhs.n_batch) if i not in lhs_batch)
  rhs_perm = tuple(rhs_batch) + tuple(i for i in range(rhs.n_batch) if i not in rhs_batch)
  lhs_indices = lhs_indices.transpose(lhs_perm + (lhs.n_batch, lhs.n_batch + 1))
  rhs_indices = rhs_indices.transpose(rhs_perm + (rhs.n_batch, rhs.n_batch + 1))
  lhs_data = lhs_data.transpose(lhs_perm + (lhs.n_batch,))
  rhs_data = rhs_data.transpose(rhs_perm + (rhs.n_batch,))

  # Implement batched dot product via vmap
  func = functools.partial(_bcoo_Mv,
      lhs_shape=lhs_shape[lhs.n_batch:], rhs_shape=rhs_shape[rhs.n_batch:],
      dtype=data_aval.dtype, lhs_contract=lhs_contracting[0] - lhs.n_batch)

  if rhs_data.shape[:rhs.n_batch] != rhs_indices.shape[:rhs.n_batch]:
    raise NotImplementedError("unequal batches in rhs")
  if lhs_data.shape[:lhs.n_batch] != lhs_indices.shape[:lhs.n_batch]:
    raise NotImplementedError("unequal batches in lhs")

  for dim in reversed(range(len(rhs_batch), rhs.n_batch)):
    func = vmap(func, in_axes=(None, None, 0, 0))
  for dim in reversed(range(len(lhs_batch), lhs.n_batch)):
    func = vmap(func, in_axes=(0, 0, None, None))
  for dim in range(len(lhs_batch)):
    if lhs_data.shape[dim] != rhs_data.shape[dim]:
      raise NotImplementedError("unequal batches in batched dims")
    func = vmap(func, in_axes=0)
  return func(lhs_data, lhs_indices, rhs_data, rhs_indices)

@bcoo_spdot_general_p.def_abstract_eval
def _bcoo_spdot_general_abstract_eval(lhs_data, lhs_indices, rhs_data, rhs_indices, *, lhs_shape, rhs_shape, dimension_numbers):
  lhs = _validate_bcoo(lhs_data, lhs_indices, lhs_shape)
  rhs = _validate_bcoo(rhs_data, rhs_indices, rhs_shape)
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  _ = _dot_general_validated_shape(lhs_shape, rhs_shape, dimension_numbers)

  if not (lhs.n_dense == rhs.n_dense == 0):
    # TODO(jakevdp): handle dense dimensions
    raise NotImplementedError("bcoo_spdot_general with dense dimensions.")

  if not (rhs.n_sparse == 1):
    raise NotImplementedError("bcoo_spdot_general with n_sparse != 1 on the rhs.")

  if max(lhs_batch, default=-1) >= lhs.n_batch or max(rhs_batch, default=-1) >= rhs.n_batch:
    raise NotImplementedError("bcoo_spdot_general: batch_dims must correspond to batch dimensions of the sparse representation.")

  if tuple(rhs_contracting) != (rhs.n_batch,) or lhs_contracting[0] not in range(lhs.n_batch, lhs.n_batch + lhs.n_sparse):
    raise NotImplementedError("bcoo_spdot_general only supports contraction of sparse indices.")

  if rhs.n_batch > len(rhs_batch) and lhs.n_sparse > len(lhs_contracting):
    raise ValueError("Cannot have unused batch dims on rhs with unused sparse dims on lhs.")

  data_shape = (
    *(lhs_shape[dim] for dim in lhs_batch),
    *(lhs_data.shape[dim] for dim in range(lhs.n_batch) if dim not in lhs_batch),
    *(rhs_data.shape[dim] for dim in range(rhs.n_batch) if dim not in rhs_batch),
    lhs.nse)
  indices_shape = (
    *(lhs_shape[dim] for dim in lhs_batch),
    *(lhs_indices.shape[dim] for dim in range(lhs.n_batch) if dim not in lhs_batch),
    *(rhs_indices.shape[dim] for dim in range(rhs.n_batch) if dim not in rhs_batch),
    lhs.nse, lhs.n_sparse - len(lhs_contracting))
  out_dtype = jnp.promote_types(lhs_data.dtype, rhs_data.dtype)
  return core.ShapedArray(data_shape, out_dtype), core.ShapedArray(indices_shape, lhs_indices.dtype)

def _bcoo_spdot_general_batch_rule(batched_args, batch_dims, *, dimension_numbers, lhs_shape, rhs_shape):
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
                                 lhs_shape=new_lhs_shape, rhs_shape=new_rhs_shape)
  return batched_out, (result_batch_dim, result_batch_dim)

# TODO(JVP): jvp, transpose
batching.primitive_batchers[bcoo_spdot_general_p] = _bcoo_spdot_general_batch_rule
xla.translations[bcoo_spdot_general_p] = xla.lower_fun(
    _bcoo_spdot_general_impl, multiple_results=True)

#----------------------------------------------------------------------
# BCOO functions that maybe should be primitives?

def _tuple_replace(tup, ind, val):
  return tuple(val if i == ind else t for i, t in enumerate(tup))

def bcoo_reduce_sum(data, indices, *, shape, axes):
  assert all(0 <= a < len(shape) for a in axes)
  n_batch, n_sparse, _, nse = _validate_bcoo(data, indices, shape)
  axes = sorted(set(axes))

  # Sum over dense dimensions -> sum over data
  dense_axes = tuple(ax - n_sparse + 1 for ax in axes if ax >= n_batch + n_sparse)
  data = data.sum(dense_axes)

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
def _is_placeholder(*args):
  return all(type(arg) is object for arg in args) or all(arg is None for arg in args)

def _asarray_or_float0(arg):
  if isinstance(arg, np.ndarray) and arg.dtype == dtypes.float0:
    return arg
  return jnp.asarray(arg)

@tree_util.register_pytree_node_class
class BCOO(ops.JAXSparse):
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
  data: jnp.ndarray
  indices: jnp.ndarray
  shape: Shape
  nse = property(lambda self: self.indices.shape[-2])
  dtype = property(lambda self: self.data.dtype)
  n_batch = property(lambda self: self.indices.ndim - 2)
  n_sparse = property(lambda self: self.indices.shape[-1])
  n_dense = property(lambda self: self.data.ndim - 1 - self.n_batch)

  @property
  def _sparse_shape(self):
    return tuple(self.shape[self.n_batch:self.n_batch + self.n_sparse])

  def __init__(self, args, *, shape):
    # JAX transforms will sometimes instantiate pytrees with null values, so we
    # must catch that in the initialization of inputs.
    self.data, self.indices = args if _is_placeholder(*args) else map(_asarray_or_float0, args)
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
    indices = jnp.column_stack((mat.row, mat.col)).astype(index_dtype)
    return cls((data, indices), shape=mat.shape)

  def _unbatch(self):
    """Return an unbatched representation of the BCOO matrix."""
    return BCOO(_unbatch_bcoo(self.data, self.indices, self.shape), shape=self.shape)

  def _dedupe(self):
    """Return a de-duplicated representation of the BCOO matrix."""
    return BCOO(_dedupe_bcoo(self.data, self.indices, self.shape), shape=self.shape)

  @jax.jit
  def todense(self):
    """Create a dense version of the array."""
    return bcoo_todense(self.data, self.indices, shape=self.shape)

  def __matmul__(self, other):
    if isinstance(other, BCOO):
      dtype = jnp.promote_types(self.dtype, other.dtype)
      dimension_numbers = (([self.ndim - 1], [0]), ([], []))
      data, indices = bcoo_spdot_general(self.data.astype(dtype), self.indices,
                                         other.data.astype(dtype), other.indices,
                                         lhs_shape=self.shape, rhs_shape=other.shape,
                                         dimension_numbers=dimension_numbers)
      shape = _dot_general_validated_shape(self.shape, other.shape, dimension_numbers)
      return BCOO((data, indices), shape=shape)
    elif isinstance(other, ops.JAXSparse):
      raise NotImplementedError("sparse-sparse matmul")
    other = jnp.asarray(other)
    if self.ndim == 0 or other.ndim == 0:
      raise ValueError("matmul inputs cannot be zero-dimensional.")
    if self.ndim > 2 or other.ndim > 2:
      raise NotImplementedError("sparse matmul for dimensions larger than 2")
    dtype = jnp.promote_types(self.dtype, other.dtype)
    return bcoo_dot_general(self.data.astype(dtype), self.indices, other.astype(dtype),
                            lhs_shape=self.shape,
                            dimension_numbers=(([self.ndim - 1], [0]), ([], [])))

  def __rmatmul__(self, other):
    if isinstance(other, ops.JAXSparse):
      raise NotImplementedError("sparse-sparse matmul")
    other = jnp.asarray(other)
    if self.ndim == 0 or other.ndim == 0:
      raise ValueError("matmul inputs cannot be zero-dimensional.")
    if self.ndim > 2 or other.ndim > 2:
      raise NotImplementedError("sparse matmul for dimensions larger than 2")
    dtype = jnp.promote_types(self.dtype, other.dtype)
    return bcoo_rdot_general(other.astype(dtype), self.data.astype(dtype), self.indices,
                             rhs_shape=self.shape,
                             dimension_numbers=(([other.ndim - 1], [0]), ([], [])))

  def transpose(self, axes=None):
    """Create a new array containing the transpose."""
    axes = np.arange(self.ndim)[::-1] if axes is None else axes
    data_T, indices_T = bcoo_transpose(self.data, self.indices, shape=self.shape, permutation=axes)
    shape_T = [self.shape[i] for i in axes]
    return BCOO((data_T, indices_T), shape=shape_T)

  def tree_flatten(self):
    children = (self.data, self.indices)
    # pytree sometimes creates placeholder objects & we need to handle that.
    sparse_shape = self.shape if _is_placeholder(*children) else self._sparse_shape
    # We serialize the sparse shape only to support batching.
    return children, {"sparse_shape": sparse_shape}

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    data, indices = children
    sparse_shape = aux_data["sparse_shape"]
    # pytree sometimes creates placeholder objects & we need to handle that.
    if _is_placeholder(data, indices):
      shape = sparse_shape
    else:
      if np.ndim(indices) < 2 or len(sparse_shape) != np.shape(indices)[-1]:
        raise ValueError(f"Invalid sparse representation: got indices.shape={np.shape(indices)}, "
                         f"data.shape={np.shape(data)}, sparse_shape={sparse_shape}")
      n_batch = indices.ndim - 2
      shape = (
          tuple(np.maximum(data.shape[:n_batch], indices.shape[:n_batch]))
          + tuple(sparse_shape)
          + tuple(data.shape[n_batch + 1:]))
    return cls(children, shape=shape)

  # TODO(jakevdp): refactor to avoid circular imports - we can use the same strategy
  #                we use when adding methods to DeviceArray within lax_numpy.py
  def __neg__(self):
    from jax.experimental.sparse import sparsify
    return sparsify(jnp.negative)(self)

  def __mul__(self, other):
    from jax.experimental.sparse import sparsify
    return sparsify(jnp.multiply)(self, other)

  def __rmul__(self, other):
    from jax.experimental.sparse import sparsify
    return sparsify(jnp.multiply)(other, self)

  def __add__(self, other):
    from jax.experimental.sparse import sparsify
    return sparsify(jnp.add)(self, other)

  def __radd__(self, other):
    from jax.experimental.sparse import sparsify
    return sparsify(jnp.add)(other, self)

  def sum(self, *args, **kwargs):
    """Sum array along axis."""
    from jax.experimental.sparse import sparsify
    return sparsify(lambda x: x.sum(*args, **kwargs))(self)
