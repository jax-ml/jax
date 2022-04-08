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

import jaxlib.mlir.ir as ir
import jaxlib.mlir.dialects.mhlo as mhlo

import numpy as np

from jaxlib import xla_client

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

def _validate_csr_mhlo(data, indices, indptr, shape):
  data_type = ir.RankedTensorType(data.type)
  indices_type = ir.RankedTensorType(indices.type)
  indptr_type = ir.RankedTensorType(indptr.type)

  nnz, = data_type.shape
  assert indices_type.shape == [nnz]
  assert indptr_type.element_type == indices_type.element_type
  assert indptr_type.shape == [shape[0] + 1]
  return data_type.element_type, indices_type.element_type, nnz


def _validate_coo(c, data, row, col, shape):
  data_dtype = np.dtype(c.get_shape(data).element_type())
  index_dtype = np.dtype(c.get_shape(row).element_type())
  nnz, = c.get_shape(data).dimensions()
  assert c.get_shape(row).dimensions() == (nnz,)
  assert c.get_shape(col).element_type() == index_dtype
  assert c.get_shape(col).dimensions() == (nnz,)
  return data_dtype, index_dtype, nnz

def _validate_coo_mhlo(data, row, col, shape):
  data_type = ir.RankedTensorType(data.type)
  row_type = ir.RankedTensorType(row.type)
  col_type = ir.RankedTensorType(col.type)

  nnz, = data_type.shape
  assert row_type.shape == [nnz]
  assert col_type.element_type == row_type.element_type
  assert col_type.shape == [nnz]
  return data_type.element_type, row_type.element_type, nnz

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


def csr_todense_mhlo(data, indices, indptr, *, shape, data_dtype, index_dtype):
  """CSR to dense matrix."""
  data_type, index_type, nnz = _validate_csr_mhlo(data, indices, indptr, shape)
  rows, cols = shape

  buffer_size, opaque = _cusparse.build_csr_todense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  i32_type = ir.IntegerType.get_signless(32)
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
          ir.RankedTensorType.get(shape, data_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ])],
      [data, indices, indptr],
      call_target_name=ir.StringAttr.get("cusparse_csr_todense"),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(opaque),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ] * 3),
      result_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([1, 0]),
                                      type=ir.IndexType.get()),
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ]))
  return mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, 0)).result


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


def csr_fromdense_mhlo(mat, *, nnz, index_dtype, data_dtype, index_type):
  """CSR from dense matrix."""
  mat_type = ir.RankedTensorType(mat.type)
  rows, cols = mat_type.shape

  buffer_size, opaque = _cusparse.build_csr_fromdense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  i32_type = ir.IntegerType.get_signless(32)
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
          ir.RankedTensorType.get([nnz], mat_type.element_type),
          ir.RankedTensorType.get([nnz], index_type),
          ir.RankedTensorType.get([rows + 1], index_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ])],
      [mat],
      call_target_name=ir.StringAttr.get("cusparse_csr_fromdense"),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(opaque),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([1, 0]),
                                      type=ir.IndexType.get()),
      ]),
      result_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ] * 4))
  return [
      mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, i)).result
      for i in range(3)
  ]

def csr_matvec(c, data, indices, indptr, x, *, shape, transpose=False,
               compute_dtype=None):
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

def csr_matvec_mhlo(data, indices, indptr, x, *, shape, transpose=False,
                    compute_dtype=None, compute_type=None, data_dtype,
                    index_dtype, x_dtype):
  """CSR matrix/vector multiply."""
  data_type, index_type, nnz = _validate_csr_mhlo(data, indices, indptr, shape)
  rows, cols = shape

  if compute_dtype is None:
    compute_dtype = data_dtype
    compute_type = data_type

  buffer_size, opaque = _cusparse.build_csr_matvec_descriptor(
      data_dtype, x_dtype, compute_dtype, index_dtype,
      rows, cols, nnz, transpose)
  out_size = cols if transpose else rows

  i32_type = ir.IntegerType.get_signless(32)
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
          ir.RankedTensorType.get([out_size], compute_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ])],
      [data, indices, indptr, x],
      call_target_name=ir.StringAttr.get("cusparse_csr_matvec"),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(opaque),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ] * 4),
      result_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ] * 2))
  return mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, 0)).result


def csr_matmat(c, data, indices, indptr, B, *, shape, transpose=False,
               compute_dtype=None):
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

def csr_matmat_mhlo(data, indices, indptr, B, *, shape, transpose=False,
                    compute_dtype=None, compute_type=None, index_dtype,
                    data_dtype, B_dtype):
  """CSR from dense matrix."""
  data_type, index_type, nnz = _validate_csr_mhlo(data, indices, indptr, shape)
  rows, cols = shape
  B_shape = ir.RankedTensorType(B.type).shape
  _, Ccols = B_shape

  if compute_dtype is None:
    compute_dtype = data_dtype
    compute_type = data_type

  buffer_size, opaque = _cusparse.build_csr_matmat_descriptor(
      data_dtype, B_dtype, compute_dtype, index_dtype,
      rows, cols, Ccols, nnz, transpose)
  out_size = cols if transpose else rows

  i32_type = ir.IntegerType.get_signless(32)
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
          ir.RankedTensorType.get([out_size, Ccols], compute_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ])],
      [data, indices, indptr, B],
      call_target_name=ir.StringAttr.get("cusparse_csr_matmat"),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(opaque),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
          ir.DenseIntElementsAttr.get(np.array([1, 0]),
                                      type=ir.IndexType.get()),
      ]),
      result_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([1, 0]), type=ir.IndexType.get()),
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ]))
  return mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, 0)).result


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

def coo_todense_mhlo(data, row, col, *, shape, data_dtype, index_dtype):
  """COO to dense matrix."""
  data_type, _, nnz = _validate_coo_mhlo(data, row, col, shape)
  rows, cols = shape

  buffer_size, opaque = _cusparse.build_coo_todense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  i32_type = ir.IntegerType.get_signless(32)
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
          ir.RankedTensorType.get(shape, data_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ])],
      [data, row, col],
      call_target_name=ir.StringAttr.get("cusparse_coo_todense"),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(opaque),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ] * 3),
      result_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([1, 0]),
                                      type=ir.IndexType.get()),
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ]))
  return mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, 0)).result


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

def coo_fromdense_mhlo(mat, *, nnz, data_dtype, index_dtype,
                       index_type):
  """COO from dense matrix."""
  mat_type = ir.RankedTensorType(mat.type)
  rows, cols = mat_type.shape

  buffer_size, opaque = _cusparse.build_coo_fromdense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  i32_type = ir.IntegerType.get_signless(32)
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
          ir.RankedTensorType.get([nnz], mat_type.element_type),
          ir.RankedTensorType.get([nnz], index_type),
          ir.RankedTensorType.get([nnz], index_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ])],
      [mat],
      call_target_name=ir.StringAttr.get("cusparse_coo_fromdense"),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(opaque),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([1, 0]),
                                      type=ir.IndexType.get()),
      ]),
      result_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ] * 4))
  return [
      mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, i)).result
      for i in range(3)
  ]

def coo_matvec(c, data, row, col, x, *, shape, transpose=False,
               compute_dtype=None):
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


def coo_matvec_mhlo(data, row, col, x, *, shape, transpose=False,
                    compute_dtype=None,
                    compute_type=None, index_dtype, data_dtype, x_dtype):
  """COO matrix/vector multiply."""
  data_type, index_type, nnz = _validate_coo_mhlo(data, row, col, shape)
  rows, cols = shape

  if compute_dtype is None:
    compute_dtype = data_dtype
    compute_type = data_type

  buffer_size, opaque = _cusparse.build_coo_matvec_descriptor(
      data_dtype, x_dtype, compute_dtype, index_dtype,
      rows, cols, nnz, transpose)
  out_size = cols if transpose else rows

  i32_type = ir.IntegerType.get_signless(32)
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
          ir.RankedTensorType.get([out_size], compute_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ])],
      [data, row, col, x],
      call_target_name=ir.StringAttr.get("cusparse_coo_matvec"),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(opaque),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ] * 4),
      result_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ] * 2))
  return mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, 0)).result


def coo_matmat(c, data, row, col, B, *, shape, transpose=False,
               compute_dtype=None):
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

def coo_matmat_mhlo(data, row, col, B, *, shape, transpose=False,
                    compute_dtype=None, compute_type=None, x_dtype,
                    data_dtype, index_dtype):
  """COO from dense matrix."""
  data_type, index_type, nnz = _validate_coo_mhlo(data, row, col, shape)
  rows, cols = shape
  B_shape = ir.RankedTensorType(B.type).shape
  _, Ccols = B_shape

  if compute_dtype is None:
    compute_dtype = data_dtype
    compute_type = data_type

  buffer_size, opaque = _cusparse.build_coo_matmat_descriptor(
      data_dtype, x_dtype, compute_dtype, index_dtype,
      rows, cols, Ccols, nnz, transpose)
  out_size = cols if transpose else rows

  i32_type = ir.IntegerType.get_signless(32)
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
          ir.RankedTensorType.get([out_size, Ccols], compute_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ])],
      [data, row, col, B],
      call_target_name=ir.StringAttr.get("cusparse_coo_matmat"),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(opaque),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
          ir.DenseIntElementsAttr.get(np.array([1, 0]),
                                      type=ir.IndexType.get()),
      ]),
      result_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([1, 0]),
                                      type=ir.IndexType.get()),
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ]))
  return mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, 0)).result


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


def gtsv2_mhlo(dl, d, du, B, *, m, n, ldb, t):
  """Calls `cusparse<t>gtsv2(dl, d, du, B, m, n, ldb)`."""
  f32 = (t == np.float32)
  if f32:
    buffer_size = _cusparse.gtsv2_f32_buffer_size(m, n, ldb)
  else:
    buffer_size = _cusparse.gtsv2_f64_buffer_size(m, n, ldb)
  i32_type = ir.IntegerType.get_signless(32)
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
          ir.RankedTensorType.get(
              [ldb, n], ir.F32Type.get() if f32 else ir.F64Type.get()),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ])],
      [dl, d, du, B],
      call_target_name = ir.StringAttr.get(
          "cusparse_gtsv2_" + ("f32" if f32 else "f64")),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(
          _cusparse.build_gtsv2_descriptor(m, n, ldb)),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ] * 3 + [
          ir.DenseIntElementsAttr.get(np.array([1, 0]), type=ir.IndexType.get())
      ]),
      result_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([1, 0]),
                                      type=ir.IndexType.get()),
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ]))
  return mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, 0)).result
