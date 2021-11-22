# Copyright 2019 Google LLC
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
cusparse wrappers for performing sparse matrix computations in JAX
"""

import numpy as np

from jax._src.lib import xla_client

try:
  from . import _cusparse
except ImportError:
  _cusparse = None
else:
  for _name, _value in _cusparse.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="CUDA")


is_supported : bool = _cusparse and _cusparse.cusparse_supported


_ops = xla_client.ops
_Shape = xla_client.Shape

def _validate_csr(c, data, indices, indptr, shape):
  data_dtype = np.dtype(c.get_shape(data).element_type())
  index_dtype = np.dtype(c.get_shape(indices).element_type())
  nnz, = c.get_shape(data).dimensions()
  assert c.get_shape(indices).dimensions() == (nnz,)
  assert c.get_shape(indptr).element_type() == index_dtype
  assert c.get_shape(indptr).dimensions() == (shape[0] + 1,)
  return data_dtype, index_dtype, nnz

def _validate_coo(c, data, row, col, shape):
  data_dtype = np.dtype(c.get_shape(data).element_type())
  index_dtype = np.dtype(c.get_shape(row).element_type())
  nnz, = c.get_shape(data).dimensions()
  assert c.get_shape(row).dimensions() == (nnz,)
  assert c.get_shape(col).element_type() == index_dtype
  assert c.get_shape(col).dimensions() == (nnz,)
  return data_dtype, index_dtype, nnz

def csr_todense(c, data, indices, indptr, *, shape):
  """CSR to dense matrix."""
  data_dtype, index_dtype, nnz = _validate_csr(c, data, indices, indptr, shape)
  rows, cols = shape

  buffer_size, opaque = _cusparse.build_csr_todense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  out = xla_client.ops.CustomCallWithLayout(
      c,
      b"cusparse_csr_todense",
      operands=(data, indices, indptr),
      operand_shapes_with_layout=(
          _Shape.array_shape(data_dtype, (nnz,), (0,)),
          _Shape.array_shape(index_dtype, (nnz,), (0,)),
          _Shape.array_shape(index_dtype, (rows + 1,), (0,)),
      ),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(data_dtype, shape, (1, 0)),
          _Shape.array_shape(np.dtype(np.int8), (buffer_size,), (0,)),
      )),
      opaque=opaque,
      api_version=xla_client.ops.CustomCallApiVersion
      .API_VERSION_STATUS_RETURNING,
  )
  return _ops.GetTupleElement(out, 0)


def csr_fromdense(c, mat, *, nnz, index_dtype):
  """CSR from dense matrix."""
  data_dtype = np.dtype(c.get_shape(mat).element_type())
  shape = c.get_shape(mat).dimensions()
  rows, cols = shape

  buffer_size, opaque = _cusparse.build_csr_fromdense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  out = xla_client.ops.CustomCallWithLayout(
      c,
      b"cusparse_csr_fromdense",
      operands=(mat,),
      operand_shapes_with_layout=(
          _Shape.array_shape(data_dtype, shape, (1, 0)),
      ),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(data_dtype, (nnz,), (0,)),
          _Shape.array_shape(index_dtype, (nnz,), (0,)),
          _Shape.array_shape(index_dtype, (shape[0] + 1,), (0,)),
          _Shape.array_shape(np.dtype(np.int8), (buffer_size,), (0,)),
      )),
      opaque=opaque,
      api_version=xla_client.ops.CustomCallApiVersion
      .API_VERSION_STATUS_RETURNING,
  )

  return tuple(_ops.GetTupleElement(out, i) for i in range(3))


def csr_matvec(c, data, indices, indptr, x, *, shape, transpose=False, compute_dtype=None):
  """CSR matrix/vector multiply."""
  data_dtype, index_dtype, nnz = _validate_csr(c, data, indices, indptr, shape)
  rows, cols = shape
  x_dtype = np.dtype(c.get_shape(x).element_type())
  x_shape = c.get_shape(x).dimensions()

  if compute_dtype is None:
    compute_dtype = data_dtype

  buffer_size, opaque = _cusparse.build_csr_matvec_descriptor(
      data_dtype, x_dtype, compute_dtype, index_dtype,
      rows, cols, nnz, transpose)
  out_size = cols if transpose else rows

  out = xla_client.ops.CustomCallWithLayout(
      c,
      b"cusparse_csr_matvec",
      operands=(data, indices, indptr, x),
      operand_shapes_with_layout=(
          _Shape.array_shape(data_dtype, (nnz,), (0,)),
          _Shape.array_shape(index_dtype, (nnz,), (0,)),
          _Shape.array_shape(index_dtype, (rows + 1,), (0,)),
          _Shape.array_shape(x_dtype, x_shape, (0,))
      ),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(compute_dtype, (out_size,), (0,)),
          _Shape.array_shape(np.dtype(np.uint8), (buffer_size,), (0,)))),
      opaque=opaque,
      api_version=xla_client.ops.CustomCallApiVersion
      .API_VERSION_STATUS_RETURNING,
  )
  return _ops.GetTupleElement(out, 0)


def csr_matmat(c, data, indices, indptr, B, *, shape, transpose=False, compute_dtype=None):
  """CSR from dense matrix."""
  data_dtype, index_dtype, nnz = _validate_csr(c, data, indices, indptr, shape)
  rows, cols = shape
  B_dtype = np.dtype(c.get_shape(B).element_type())
  B_shape = c.get_shape(B).dimensions()
  _, Ccols = B_shape

  if compute_dtype is None:
    compute_dtype = data_dtype

  buffer_size, opaque = _cusparse.build_csr_matmat_descriptor(
      data_dtype, B_dtype, compute_dtype, index_dtype,
      rows, cols, Ccols, nnz, transpose)
  out_size = cols if transpose else rows

  out = xla_client.ops.CustomCallWithLayout(
      c,
      b"cusparse_csr_matmat",
      operands=(data, indices, indptr, B),
      operand_shapes_with_layout=(
          _Shape.array_shape(data_dtype, (nnz,), (0,)),
          _Shape.array_shape(index_dtype, (nnz,), (0,)),
          _Shape.array_shape(index_dtype, (rows + 1,), (0,)),
          _Shape.array_shape(B_dtype, B_shape, (1, 0)),
      ),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(compute_dtype, (out_size, Ccols), (1, 0)),
          _Shape.array_shape(np.dtype(np.uint8), (buffer_size,), (0,)))),
      opaque=opaque,
      api_version=xla_client.ops.CustomCallApiVersion
      .API_VERSION_STATUS_RETURNING,
  )
  return _ops.GetTupleElement(out, 0)


def coo_todense(c, data, row, col, *, shape):
  """COO to dense matrix."""
  data_dtype, index_dtype, nnz = _validate_coo(c, data, row, col, shape)
  rows, cols = shape

  buffer_size, opaque = _cusparse.build_coo_todense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  out = xla_client.ops.CustomCallWithLayout(
      c,
      b"cusparse_coo_todense",
      operands=(data, row, col),
      operand_shapes_with_layout=(
          _Shape.array_shape(data_dtype, (nnz,), (0,)),
          _Shape.array_shape(index_dtype, (nnz,), (0,)),
          _Shape.array_shape(index_dtype, (nnz,), (0,)),
      ),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(data_dtype, shape, (1, 0)),
          _Shape.array_shape(np.dtype(np.int8), (buffer_size,), (0,)),
      )),
      opaque=opaque,
      api_version=xla_client.ops.CustomCallApiVersion
      .API_VERSION_STATUS_RETURNING,
  )
  return _ops.GetTupleElement(out, 0)


def coo_fromdense(c, mat, *, nnz, index_dtype):
  """COO from dense matrix."""
  data_dtype = np.dtype(c.get_shape(mat).element_type())
  shape = c.get_shape(mat).dimensions()
  rows, cols = shape

  buffer_size, opaque = _cusparse.build_coo_fromdense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  out = xla_client.ops.CustomCallWithLayout(
      c,
      b"cusparse_coo_fromdense",
      operands=(mat,),
      operand_shapes_with_layout=(
          _Shape.array_shape(data_dtype, shape, (1, 0)),
      ),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(data_dtype, (nnz,), (0,)),
          _Shape.array_shape(index_dtype, (nnz,), (0,)),
          _Shape.array_shape(index_dtype, (nnz,), (0,)),
          _Shape.array_shape(np.dtype(np.int8), (buffer_size,), (0,)),
      )),
      opaque=opaque,
      api_version=xla_client.ops.CustomCallApiVersion
      .API_VERSION_STATUS_RETURNING,
  )

  return tuple(_ops.GetTupleElement(out, i) for i in range(3))

def coo_matvec(c, data, row, col, x, *, shape, transpose=False, compute_dtype=None):
  """COO matrix/vector multiply."""
  data_dtype, index_dtype, nnz = _validate_coo(c, data, row, col, shape)
  rows, cols = shape
  x_dtype = np.dtype(c.get_shape(x).element_type())
  x_shape = c.get_shape(x).dimensions()

  if compute_dtype is None:
    compute_dtype = data_dtype

  buffer_size, opaque = _cusparse.build_coo_matvec_descriptor(
      data_dtype, x_dtype, compute_dtype, index_dtype,
      rows, cols, nnz, transpose)
  out_size = cols if transpose else rows

  out = xla_client.ops.CustomCallWithLayout(
      c,
      b"cusparse_coo_matvec",
      operands=(data, row, col, x),
      operand_shapes_with_layout=(
          _Shape.array_shape(data_dtype, (nnz,), (0,)),
          _Shape.array_shape(index_dtype, (nnz,), (0,)),
          _Shape.array_shape(index_dtype, (nnz,), (0,)),
          _Shape.array_shape(x_dtype, x_shape, (0,)),
      ),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(compute_dtype, (out_size,), (0,)),
          _Shape.array_shape(np.dtype(np.uint8), (buffer_size,), (0,)))),
      opaque=opaque,
      api_version=xla_client.ops.CustomCallApiVersion
      .API_VERSION_STATUS_RETURNING,
  )
  return _ops.GetTupleElement(out, 0)


def coo_matmat(c, data, row, col, B, *, shape, transpose=False, compute_dtype=None):
  """COO from dense matrix."""
  data_dtype, index_dtype, nnz = _validate_coo(c, data, row, col, shape)
  rows, cols = shape
  B_dtype = np.dtype(c.get_shape(B).element_type())
  B_shape = c.get_shape(B).dimensions()
  _, Ccols = B_shape

  if compute_dtype is None:
    compute_dtype = data_dtype

  buffer_size, opaque = _cusparse.build_coo_matmat_descriptor(
      data_dtype, B_dtype, compute_dtype, index_dtype,
      rows, cols, Ccols, nnz, transpose)
  out_size = cols if transpose else rows

  out = xla_client.ops.CustomCallWithLayout(
      c,
      b"cusparse_coo_matmat",
      operands=(data, row, col, B),
      operand_shapes_with_layout=(
          _Shape.array_shape(data_dtype, (nnz,), (0,)),
          _Shape.array_shape(index_dtype, (nnz,), (0,)),
          _Shape.array_shape(index_dtype, (nnz,), (0,)),
          _Shape.array_shape(B_dtype, B_shape, (1, 0)),
      ),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(compute_dtype, (out_size, Ccols), (1, 0)),
          _Shape.array_shape(np.dtype(np.uint8), (buffer_size,), (0,)))),
      opaque=opaque,
      api_version=xla_client.ops.CustomCallApiVersion
      .API_VERSION_STATUS_RETURNING,
  )
  return _ops.GetTupleElement(out, 0)


def gtsv2(c, dl, d, du, B, *, m, n, ldb, t):
  """Calls `cusparse<t>gtsv2(dl, d, du, B, m, n, ldb)`."""
  f32 = (t == np.float32)
  dl_shape, d_shape, du_shape, B_shape = map(c.get_shape, (dl, d, du, B))
  if f32:
    buffer_size = _cusparse.gtsv2_f32_buffer_size(m, n, ldb)
  else:
    buffer_size = _cusparse.gtsv2_f64_buffer_size(m, n, ldb)
  out = xla_client.ops.CustomCallWithLayout(
      c,
      b"cusparse_gtsv2_" + (b"f32" if f32 else b"f64"),
      operands=(dl, d, du, B),
      operand_shapes_with_layout=(dl_shape, d_shape, du_shape, B_shape),
      shape_with_layout=_Shape.tuple_shape(
          (_Shape.array_shape(np.dtype(t), (ldb, n), (1, 0)),
           _Shape.array_shape(np.dtype(np.uint8), (buffer_size,), (0,)))),
      opaque=_cusparse.build_gtsv2_descriptor(m, n, ldb),
      has_side_effect=False,
      api_version=xla_client.ops.CustomCallApiVersion
      .API_VERSION_STATUS_RETURNING)
  return _ops.GetTupleElement(out, 0)
