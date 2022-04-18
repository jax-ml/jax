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
hipsparse wrappers for performing sparse matrix computations in JAX on ROCM
"""

import jaxlib.mlir.ir as ir
import jaxlib.mlir.dialects.mhlo as mhlo

import numpy as np

from jaxlib import xla_client

try:
  from . import _hipsparse
except ImportError:
  _hipsparse = None
else:
  for _name, _value in _hipsparse.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="ROCM")


is_supported : bool = _hipsparse and _hipsparse.hipsparse_supported


def _validate_csr_mhlo(data, indices, indptr, shape):
  data_type = ir.RankedTensorType(data.type)
  indices_type = ir.RankedTensorType(indices.type)
  indptr_type = ir.RankedTensorType(indptr.type)

  nnz, = data_type.shape
  assert indices_type.shape == [nnz]
  assert indptr_type.element_type == indices_type.element_type
  assert indptr_type.shape == [shape[0] + 1]
  return data_type.element_type, indices_type.element_type, nnz

def _validate_coo_mhlo(data, row, col, shape):
  data_type = ir.RankedTensorType(data.type)
  row_type = ir.RankedTensorType(row.type)
  col_type = ir.RankedTensorType(col.type)

  nnz, = data_type.shape
  assert row_type.shape == [nnz]
  assert col_type.element_type == row_type.element_type
  assert col_type.shape == [nnz]
  return data_type.element_type, row_type.element_type, nnz


def csr_todense_mhlo(data, indices, indptr, *, shape, data_dtype, index_dtype):
  """CSR to dense matrix."""
  data_type, index_type, nnz = _validate_csr_mhlo(data, indices, indptr, shape)
  rows, cols = shape

  buffer_size, opaque = _hipsparse.build_csr_todense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  i32_type = ir.IntegerType.get_signless(32)
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
          ir.RankedTensorType.get(shape, data_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ])],
      [data, indices, indptr],
      call_target_name=ir.StringAttr.get("hipsparse_csr_todense"),
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


def csr_fromdense_mhlo(mat, *, nnz, index_dtype, data_dtype, index_type):
  """CSR from dense matrix."""
  mat_type = ir.RankedTensorType(mat.type)
  rows, cols = mat_type.shape

  buffer_size, opaque = _hipsparse.build_csr_fromdense_descriptor(
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
      call_target_name=ir.StringAttr.get("hipsparse_csr_fromdense"),
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


def csr_matvec_mhlo(data, indices, indptr, x, *, shape, transpose=False,
                    compute_dtype=None, compute_type=None, data_dtype,
                    index_dtype, x_dtype):
  """CSR matrix/vector multiply."""
  data_type, index_type, nnz = _validate_csr_mhlo(data, indices, indptr, shape)
  rows, cols = shape

  if compute_dtype is None:
    compute_dtype = data_dtype
    compute_type = data_type

  buffer_size, opaque = _hipsparse.build_csr_matvec_descriptor(
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
      call_target_name=ir.StringAttr.get("hipsparse_csr_matvec"),
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

  buffer_size, opaque = _hipsparse.build_csr_matmat_descriptor(
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
      call_target_name=ir.StringAttr.get("hipsparse_csr_matmat"),
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


def coo_todense_mhlo(data, row, col, *, shape, data_dtype, index_dtype):
  """COO to dense matrix."""
  data_type, _, nnz = _validate_coo_mhlo(data, row, col, shape)
  rows, cols = shape

  buffer_size, opaque = _hipsparse.build_coo_todense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  i32_type = ir.IntegerType.get_signless(32)
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
          ir.RankedTensorType.get(shape, data_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ])],
      [data, row, col],
      call_target_name=ir.StringAttr.get("hipsparse_coo_todense"),
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


def coo_fromdense_mhlo(mat, *, nnz, data_dtype, index_dtype,
                       index_type):
  """COO from dense matrix."""
  mat_type = ir.RankedTensorType(mat.type)
  rows, cols = mat_type.shape

  buffer_size, opaque = _hipsparse.build_coo_fromdense_descriptor(
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
      call_target_name=ir.StringAttr.get("hipsparse_coo_fromdense"),
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


def coo_matvec_mhlo(data, row, col, x, *, shape, transpose=False,
                    compute_dtype=None,
                    compute_type=None, index_dtype, data_dtype, x_dtype):
  """COO matrix/vector multiply."""
  data_type, index_type, nnz = _validate_coo_mhlo(data, row, col, shape)
  rows, cols = shape

  if compute_dtype is None:
    compute_dtype = data_dtype
    compute_type = data_type

  buffer_size, opaque = _hipsparse.build_coo_matvec_descriptor(
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
      call_target_name=ir.StringAttr.get("hipsparse_coo_matvec"),
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

  buffer_size, opaque = _hipsparse.build_coo_matmat_descriptor(
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
      call_target_name=ir.StringAttr.get("hipsparse_coo_matmat"),
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


def gtsv2_mhlo(dl, d, du, B, *, m, n, ldb, t):
  """Calls `hipsparse<t>gtsv2(dl, d, du, B, m, n, ldb)`."""
  f32 = (t == np.float32)
  if f32:
    buffer_size = _hipsparse.gtsv2_f32_buffer_size(m, n, ldb)
  else:
    buffer_size = _hipsparse.gtsv2_f64_buffer_size(m, n, ldb)
  i32_type = ir.IntegerType.get_signless(32)
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
          ir.RankedTensorType.get(
              [ldb, n], ir.F32Type.get() if f32 else ir.F64Type.get()),
          ir.RankedTensorType.get([buffer_size], ir.IntegerType.get_signless(8)),
      ])],
      [dl, d, du, B],
      call_target_name = ir.StringAttr.get(
          "hipsparse_gtsv2_" + ("f32" if f32 else "f64")),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(
          _hipsparse.build_gtsv2_descriptor(m, n, ldb)),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ] * 3 + [
          ir.DenseIntElementsAttr.get(np.array([1, 0]), type=ir.IndexType.get())
      ]),
      result_layouts=ir.ArrayAttr.get([
          ir.DenseIntElementsAttr.get(np.array([1, 0]), type=ir.IndexType.get()),
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ]))
  return mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, 0)).result
