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
"""
JAX implementation of a general N-dimensional sparse format being implemented
in MLIR; see https://llvm.discourse.group/t/mlir-support-for-sparse-tensors/2020
"""
from typing import Any, IO, List, Optional, Tuple

import numpy as np
import jax.numpy as jnp
from jax import core
from jax import tree_util
from jax import xla

Array = Any


def _compress_indices_sparse(iptr, ind, size):
  """Utility to compress a series of indices

  Args:
    iptr : array of N + 1 pointers into ind array. The indices associated with
      dimension ``i`` are given by ``ind[iptr[i]:iptr[i + 1]]``
    ind : array of indices within each dimension
    size : (unused) integer size of the leading dimension.

  Returns:
    positions : array of N + 1 pointers into indices array. The indices associated
      with dimension ``i`` are given by ``indices[positions[i]:positions[i + 1]]``.
    indices : array of length M, with N <= M <= (N * size), containing all non-empty
      unique indices within each dimension.
    new_iptr : array of (M + 1) pointers into ind array, indicating the input indices
      associated with the output indices.

  Example:
    >>> iptr = jnp.array([0, 3, 7])
    >>> ind = jnp.array([0, 1, 1, 0, 0, 2, 2])
    >>> positions, indices, new_iptr = _compress_indices_sparse(iptr, ind, None)
    >>> positions
    DeviceArray([0, 2, 4], dtype=int32)
    >>> indices
    DeviceArray([0, 1, 0, 2], dtype=int32)
    >>> new_iptr
    DeviceArray([0, 1, 3, 5, 7], dtype=int32)
  """
  del size  # unused, for now

  # offset each segment so that consecutive indices cannot be equal
  offset = jnp.cumsum(jnp.zeros(len(ind) + 1).at[iptr].add(ind.max() + 1))
  offset_ind = offset.at[:-1].add(ind)

  # Use the cumulative sum to identify indices of unique values
  diffs = offset_ind[1:] != offset_ind[:-1]
  unique_ind = jnp.cumsum(jnp.concatenate([jnp.array([0]), diffs]))

  # generate the de-duplicated index with zero padding at the end
  index = jnp.zeros_like(ind).at[unique_ind[:-1]].set(ind)
  positions = unique_ind[iptr]

  # TODO: need a size on _nonzero here
  new_iptr = jnp.nonzero(jnp.concatenate([jnp.array([1]), diffs]))[0]
  # TODO: allow passing the full index through
  return positions, index[:positions[-1]], new_iptr


def _compress_indices_dense(iptr, ind, size, *, full_outputs=False):
  """Utility for compressing dense indices.

  Args:
    iptr : array of N + 1 pointers into ind array. The indices associated with
      dimension ``i`` are given by ``ind[iptr[i]:iptr[i + 1]]``
    ind : array of indices within each dimension
    size : integer size of the leading dimension.
    full_outputs : if True, compute and return `positions` and `indices`. Otherwise
      return None for these quantities (default=False).

  Returns:
    positions : None if full_outputs is False, otherwise a length (N + 1) array
      of pointers into indices array. The indices associated with dimension ``i`` are
      given by ``indices[positions[i]:positions[i + `]]``.
    indices : None if full_outputs is False, otherwise an array of length M = N * size
      containing (possibly empty) unique indices within each dimension.
    new_iptr : array of (M + 1) pointers into ind array, indicating the input indices
      associated with the output indices.

  Examples:
    >>> iptr = jnp.array([0, 3, 7])
    >>> ind = jnp.array([0, 1, 1, 0, 0, 2, 2])
    >>> positions, indices, new_iptr = _compress_indices_dense(iptr, ind, 3, full_outputs=True)
    >>> positions
    DeviceArray([0, 3, 6], dtype=int32)
    >>> indices
    DeviceArray([0, 1, 2, 0, 1, 2], dtype=int32)
    >>> new_iptr
    DeviceArray([0, 1, 3, 3, 5, 5, 7], dtype=int32)
  """
  # TODO: only use ind[iptr[0]:iptr[-1]]
  if full_outputs:
    N = len(iptr) - 1
    positions = jnp.arange(0, (N + 1) * size, size)
    indices = jnp.tile(jnp.arange(size), N)
  else:
    positions = indices = None
  step = jnp.cumsum(jnp.zeros_like(ind).at[iptr[1:]].add(size))
  counts = jnp.bincount(ind + step, length=size * (len(iptr) - 1))
  new_iptr = jnp.cumsum(jnp.concatenate([jnp.array([0]), counts]))
  return positions, indices, new_iptr


def _uncompress_indices_sparse(positions, indices, iptr, *, nnz=None):
  """
  Inverse of _compress_indices_sparse

  Examples:
    >>> iptr = jnp.array([0, 3, 7])
    >>> ind = jnp.array([0, 1, 1, 0, 0, 2, 2])
    >>> size = 3

    >>> args = _compress_indices_sparse(iptr, ind, size)
    >>> _uncompress_indices_sparse(*args)
    (DeviceArray([0, 3, 7], dtype=int32), DeviceArray([0, 1, 1, 0, 0, 2, 2], dtype=int32))
  """
  sizes = jnp.diff(iptr)
  ind = jnp.repeat(indices, sizes, total_repeat_length=nnz)
  iptr = iptr[positions]
  return iptr, ind


def _uncompress_indices_dense(positions, indices, iptr, *, size, N, nnz=None):
  """
  Inverse of _compress_indices_dense.

  Examples:
    >>> iptr = jnp.array([0, 3, 7])
    >>> ind = jnp.array([0, 1, 1, 0, 0, 2, 2])
    >>> size = 3
    >>> N = len(iptr) - 1

    >>> args = _compress_indices_dense(iptr, ind, size, full_outputs=False)
    >>> _uncompress_indices_dense(*args, size=size, N=N)
    (DeviceArray([0, 3, 7], dtype=int32), DeviceArray([0, 1, 1, 0, 0, 2, 2], dtype=int32))

    >>> args = _compress_indices_dense(iptr, ind, size, full_outputs=True)
    >>> _uncompress_indices_dense(*args, size=size, N=N)
    (DeviceArray([0, 3, 7], dtype=int32), DeviceArray([0, 1, 1, 0, 0, 2, 2], dtype=int32))
  """
  assert (positions is None) == (indices is None)
  if positions is None:
    positions = jnp.arange(0, (N + 1) * size, size)
    indices = jnp.tile(jnp.arange(size), N)
  return _uncompress_indices_sparse(positions, indices, iptr, nnz=nnz)


def read_frostt(f: IO) -> np.ndarray:
  """Read a matrix in extended FROSTT format, returning a dense representation.

  Example:
    >>> import io
    >>> f = io.StringIO('''
    ... 3 7
    ... 3 3 4
    ... 1 1 1  1.0
    ... 1 1 4  2.0
    ... 1 2 1  3.0
    ... 1 2 2  4.0
    ... 3 1 2  5.0
    ... 3 2 3  6.0
    ... 3 2 4  7.0
    ... ''')
    >>> read_frostt(f)
    array([[[1., 0., 0., 2.],
            [3., 4., 0., 0.],
            [0., 0., 0., 0.]],
           [[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]],
           [[0., 5., 0., 0.],
            [0., 0., 6., 7.],
            [0., 0., 0., 0.]]])
  """
  for line in f:
    line = line.split('#')[0]
    if line.strip():
      break
  ndim, nnz = map(int, line.split())

  line = f.readline().split('#')[0]
  shape = tuple(map(int, line.split()))
  assert len(shape) == ndim

  mat = np.zeros(shape, dtype=float)
  for i in range(nnz):
    line = f.readline().split('#')[0]
    *indices, val = line.split()
    mat[tuple(i - 1 for i in map(int, indices))] = float(val)
  return mat


def mlir_fromdense(mat, *, format):
  """Convert a dense matrix to a sparse representation with the given format.

  Args:
    mat : N-dimensional numpy array.
    format : length-N tuple with "S" for sparse dimensions and "D" for dense.

  Returns:
    positions, indices, values : general sparse representation
  """
  return mlir_fromdense_p.bind(mat, format=format)

mlir_fromdense_p = core.Primitive('mlir_fromdense')

@mlir_fromdense_p.def_abstract_eval
def _mlir_fromdense_abstract_eval(mat, *, format):
  # TODO: allow passing static metadata (e.g. nnz) to allow this to be
  # evaluated in abstract.
  raise NotImplementedError("mlir_fromdense_abstract_eval")

@mlir_fromdense_p.def_impl
def _mlir_fromdense_impl(mat, *, format):
  mat = jnp.asarray(mat)
  format = tuple(format)

  assert set(format).issubset({"D", "S"})
  assert len(format) == mat.ndim

  if "S" not in format:
    return [], [], mat

  last_s = format[::-1].index("S")
  axis = tuple(range(len(format) - last_s, len(format)))

  ind_tup = jnp.nonzero(mat.any(axis))
  values = mat[ind_tup]

  positions = []
  indices = []
  iptr = jnp.array([0, len(values)])

  for ind, fmt, size in zip(ind_tup, format, mat.shape):
    if fmt == "S":
      position, index, iptr = _compress_indices_sparse(iptr, ind, size)
      indices.append(index)
      positions.append(position)
    else:
      position, index, iptr = _compress_indices_dense(iptr, ind, size)
  return positions, indices, values.ravel()


xla.translations[mlir_fromdense_p] = xla.lower_fun(
    _mlir_fromdense_impl, multiple_results=False)


def _mlir_tocoo(positions, indices, values, *, shape, format):
  assert len(format) == len(shape)
  assert len(positions) == len(indices) == format.count("S")
  if "S" not in format:
    return (), values

  positions = [None if f == "D" else next(it) for it in [iter(positions)] for f in format]
  indices = [None if f == "D" else next(it) for it in [iter(indices)] for f in format]

  last_s = len(shape) - format[::-1].index("S")
  values = values.reshape((-1,) + shape[last_s:])
  ind = last_s * [None]
  iptr = jnp.arange(values.shape[0] + 1, dtype=indices[last_s - 1].dtype)
  nnz = values.shape[0]
  for i in range(last_s)[::-1]:
    size = shape[i]
    if format[i] == "S":
      iptr, ind[i] = _uncompress_indices_sparse(positions[i], indices[i], iptr, nnz=nnz)
    else:
      N = 1 if i == 0 else shape[i - 1] if format[i - 1] == "D" else len(indices[i - 1])
      iptr, ind[i] = _uncompress_indices_dense(positions[i], indices[i], iptr,
                                               size=size, N=N, nnz=nnz)
  return tuple(ind), values


def mlir_todense(positions, indices, values, *, shape, format):
  """Convert a sparse representation to a dense matrix.

  Args:
    positions : list containing None for dense dimensions and arrays of
      positions for sparse dimensions.
    indices : list containing None for dense dimentions and arrays of
      positions for sparse dimensions.
    values : array of nonzero values in the sparse representation
    shape : tuple representing the matrix shape.
    format : tuple representing the format
  Returns:
    mat : dense matrix representation of specified shape.
  """
  return mlir_todense_p.bind(*positions, *indices, values, shape=shape, format=format)

mlir_todense_p = core.Primitive('mlir_todense')

@mlir_todense_p.def_abstract_eval
def _mlir_todense_abstract_eval(*args, shape, format):
  *pos_ind, values = args
  assert len(pos_ind) == 2 * format.count('S')
  assert len(shape) == len(format)
  return core.ShapedArray(shape, values.dtype)

@mlir_todense_p.def_impl
def _mlir_todense_impl(*args, shape, format):
  *pos_ind, values = args
  positions, indices = pos_ind[:len(pos_ind) // 2], pos_ind[len(pos_ind) // 2:]
  ind, values = _mlir_tocoo(positions, indices, values, shape=shape, format=format)
  if not ind:
    return values
  return jnp.zeros(shape, values.dtype).at[ind].set(values)

xla.translations[mlir_todense_p] = xla.lower_fun(
    _mlir_todense_impl, multiple_results=False)

def mlir_matvec(positions, indices, values, v, *, shape, format):
  return mlir_matvec_p.bind(*positions, *indices, values, v, shape=shape, format=format)

mlir_matvec_p = core.Primitive('mlir_matvec')

@mlir_matvec_p.def_abstract_eval
def _mlir_matvec_abstract_eval(*args, shape, format):
  *pos_ind, values, v = args
  assert len(pos_ind) == 2 * format.count('S')
  assert len(shape) == len(format)
  # TODO(jakevdp): relax this
  assert values.dtype == v.dtype
  assert v.shape == shape[-1:]
  return core.ShapedArray(shape[:-1], values.dtype)

@mlir_matvec_p.def_impl
def _mlir_matvec_impl(*args, shape, format):
  *pos_ind, values, v = args
  positions, indices = pos_ind[:len(pos_ind) // 2], pos_ind[len(pos_ind) // 2:]
  v = jnp.asarray(v)
  if v.ndim != 1:
    raise NotImplementedError("mlir_matvec only supports 1-dimensional `v`")
  assert v.shape == shape[-1:]
  ind, values = _mlir_tocoo(positions, indices, values, shape=shape, format=format)
  if format[-1] == "D":
    return mlir_todense(positions, indices, values @ v,
                        shape=shape[:-1], format=format[:-1])
  if len(ind) == 1:
    return values @ v[ind[0]]
  elif len(ind) == 2:
    row, col = ind
    dv = values * v[col]
    return jnp.zeros(shape[0], dv.dtype).at[row].add(dv)
  else:
    # TODO: implement this case.
    raise NotImplementedError("mlir_matvec only supports 1 or 2 dimensional matrices.")

xla.translations[mlir_matvec_p] = xla.lower_fun(
    _mlir_matvec_impl, multiple_results=False)

@tree_util.register_pytree_node_class
class MLIRSparse:
  positions: List[Optional[Array]]
  indices: List[Optional[Array]]
  values: Array
  shape: Tuple[int, ...]
  format: str

  dtype = property(lambda self: self.values.dtype)
  nnz = property(lambda self: self.values.size)
  ndim = property(lambda self: len(self.shape))

  def __init__(self, args, *, shape, format):
    self.positions, self.indices, self.values = args
    self.shape = shape
    self.format = format

  @classmethod
  def fromfile(cls, f, *, format=None):
    # TODO: avoid trip through dense & pass correct nnz
    return cls.fromdense(read_frostt(f), format=format)

  @classmethod
  def fromdense(cls, mat, *, format=None):
    if format is None:
      format = "S" * mat.ndim
    assert isinstance(format, str)
    return cls(mlir_fromdense(mat, format=format), shape=mat.shape, format=format)

  def todense(self):
    return mlir_todense(self.positions, self.indices, self.values,
                        shape=self.shape, format=self.format)

  def __matmul__(self, v):
    return mlir_matvec(self.positions, self.indices, self.values, v,
                       shape=self.shape, format=self.format)

  def tree_flatten(self):
    children = (self.positions, self.indices, self.values)
    aux_data = {"shape": self.shape, "format": self.format}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(children, **aux_data)
