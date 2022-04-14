# Copyright 2018 Google LLC
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

# Shims that allow the XLA CPU backend to call scipy-provided LAPACK kernels
# via CustomCallWithLayout.

import jaxlib.mlir.ir as ir
import jaxlib.mlir.dialects.mhlo as mhlo

import numpy as np
from jaxlib import xla_client

from . import _lapack
for _name, _value in _lapack.registrations().items():
  xla_client.register_custom_call_target(_name, _value, platform="cpu")

_ops = xla_client.ops

Shape = xla_client.Shape


def _mhlo_u8(x):
  typ = ir.RankedTensorType.get([], ir.IntegerType.get_unsigned(8))
  return mhlo.ConstOp(
      typ,
      ir.DenseElementsAttr.get(np.array(x, dtype=np.uint8),
                               type=typ.element_type)).result

def _mhlo_s32(x):
  typ = ir.RankedTensorType.get([], ir.IntegerType.get_signless(32))
  return mhlo.ConstOp(
      typ, ir.DenseElementsAttr.get(np.array(x, dtype=np.int32))).result

def _constant_s32_scalar(c, x):
  return _ops.Constant(c, np.int32(x))

# TODO(phawkins): it would be nice to avoid duplicating code for each type.

# ?trsm(left_side, lower, trans_a, diag, m, n, alpha, a, b):
# triangular solve

def trsm(c, alpha, a, b, left_side=False, lower=False, trans_a=False,
         conj_a=False, diag=False):
  a_shape = c.get_shape(a)
  b_shape = c.get_shape(b)
  dtype = b_shape.element_type()

  dims = b_shape.dimensions()

  m, n = dims[-2:]
  k = m if left_side else n

  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  num_b = 1
  for d in batch_dims:
    num_b *= d

  if batch_dims + (k, k) != a_shape.dimensions() or a_shape.element_type() != dtype:
    raise ValueError("Argument mismatch for trsm, got {} and {}".format(
      a_shape, b_shape))

  if dtype == np.float32:
    fn = b"blas_strsm"
  elif dtype == np.float64:
    fn = b"blas_dtrsm"
  elif dtype == np.complex64:
    fn = b"blas_ctrsm"
  elif dtype == np.complex128:
    fn = b"blas_ztrsm"
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  if conj_a and not trans_a:
    raise NotImplementedError("Conjugation without transposition not supported")

  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  return _ops.CustomCallWithLayout(
    c, fn,
    operands=(
      _constant_s32_scalar(c, int(left_side)),
      _constant_s32_scalar(c, int(lower)),
      _constant_s32_scalar(c, (2 if conj_a else 1) if trans_a else 0),
      _constant_s32_scalar(c, int(diag)),
      _constant_s32_scalar(c, m),
      _constant_s32_scalar(c, n),
      _constant_s32_scalar(c, num_b),
      alpha, a, b),
    shape_with_layout=Shape.array_shape(dtype, b_shape.dimensions(), layout),
    operand_shapes_with_layout=(
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(dtype, (), ()),
        Shape.array_shape(dtype, a_shape.dimensions(), layout),
        Shape.array_shape(dtype, b_shape.dimensions(), layout),
    ),
    api_version=xla_client.ops.CustomCallApiVersion
    .API_VERSION_STATUS_RETURNING)
jax_trsm = trsm

def trsm_mhlo(dtype, alpha, a, b, left_side=False, lower=False, trans_a=False,
               conj_a=False, diag=False):
  a_type = ir.RankedTensorType(a.type)
  b_type = ir.RankedTensorType(b.type)

  dims = b_type.shape
  m, n = dims[-2:]
  k = m if left_side else n

  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  num_b = 1
  for d in batch_dims:
    num_b *= d

  if (batch_dims + (k, k) != tuple(a_type.shape) or
      a_type.element_type != b_type.element_type):
    raise ValueError("Argument mismatch for trsm, got {} and {}".format(
      a_type, b_type))

  if dtype == np.float32:
    fn = "blas_strsm"
  elif dtype == np.float64:
    fn = "blas_dtrsm"
  elif dtype == np.complex64:
    fn = "blas_ctrsm"
  elif dtype == np.complex128:
    fn = "blas_ztrsm"
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  if conj_a and not trans_a:
    raise NotImplementedError("Conjugation without transposition not supported")
  scalar_layout = ir.DenseIntElementsAttr.get(np.zeros((0,), np.int64),
                                              type=ir.IndexType.get())
  layout = ir.DenseIntElementsAttr.get(
      np.array((num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
      type=ir.IndexType.get())
  return mhlo.CustomCallOp(
      [b.type],
      [_mhlo_s32(int(left_side)), _mhlo_s32(int(lower)),
       _mhlo_s32((2 if conj_a else 1) if trans_a else 0), _mhlo_s32(int(diag)),
       _mhlo_s32(m), _mhlo_s32(n), _mhlo_s32(num_b),
       alpha, a, b],
      call_target_name = ir.StringAttr.get(fn),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(""),
      api_version=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([scalar_layout] * 8 + [layout] * 2),
      result_layouts=ir.ArrayAttr.get([layout])).result


# # ?getrf: LU decomposition

def getrf(c, a):
  a_shape = c.get_shape(a)
  dtype = a_shape.element_type()
  dims = a_shape.dimensions()
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  b = 1
  for d in batch_dims:
    b *= d

  if dtype == np.float32:
    fn = b"lapack_sgetrf"
  elif dtype == np.float64:
    fn = b"lapack_dgetrf"
  elif dtype == np.complex64:
    fn = b"lapack_cgetrf"
  elif dtype == np.complex128:
    fn = b"lapack_zgetrf"
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  out = _ops.CustomCallWithLayout(
    c, fn,
    operands=(
      _constant_s32_scalar(c, b),
      _constant_s32_scalar(c, m),
      _constant_s32_scalar(c, n),
      a),
    shape_with_layout=Shape.tuple_shape((
        Shape.array_shape(
          dtype,
          batch_dims + (m, n),
          (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
        Shape.array_shape(
          np.dtype(np.int32),
          batch_dims + (min(m, n),),
          tuple(range(num_bd, -1, -1))),
        Shape.array_shape(np.dtype(np.int32), batch_dims,
          tuple(range(num_bd - 1, -1, -1))),
    )),
    operand_shapes_with_layout=(
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(
          dtype,
          batch_dims + (m, n),
          (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
    ),
    api_version=xla_client.ops.CustomCallApiVersion
    .API_VERSION_STATUS_RETURNING)
  return tuple(_ops.GetTupleElement(out, i) for i in range(3))

def getrf_mhlo(dtype, a):
  dims = ir.RankedTensorType(a.type).shape
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  b = 1
  for d in batch_dims:
    b *= d

  if dtype == np.float32:
    fn = b"lapack_sgetrf"
  elif dtype == np.float64:
    fn = b"lapack_dgetrf"
  elif dtype == np.complex64:
    fn = b"lapack_cgetrf"
  elif dtype == np.complex128:
    fn = b"lapack_zgetrf"
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  scalar_layout = ir.DenseIntElementsAttr.get(np.zeros((0,), np.int64),
                                              type=ir.IndexType.get())
  layout = ir.DenseIntElementsAttr.get(
      np.array((num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
      type=ir.IndexType.get())
  i32_type = ir.IntegerType.get_signless(32)
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
        a.type,
        ir.RankedTensorType.get(batch_dims + (min(m, n),), i32_type),
        ir.RankedTensorType.get(batch_dims, i32_type),
      ])],
      [_mhlo_s32(int(b)), _mhlo_s32(m), _mhlo_s32(n), a],
      call_target_name = ir.StringAttr.get(fn),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(""),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([scalar_layout] * 3 + [layout]),
      result_layouts=ir.ArrayAttr.get([
        layout,
        ir.DenseIntElementsAttr.get(np.arange(num_bd, -1, -1),
                                    type=ir.IndexType.get()),
        ir.DenseIntElementsAttr.get(np.arange(num_bd - 1, -1, -1),
                                    type=ir.IndexType.get()),
      ]))
  return [
    mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, i)).result
    for i in range(3)
  ]


# # ?geqrf: QR decomposition

def geqrf(c, a):
  a_shape = c.get_shape(a)
  dtype = a_shape.element_type()
  dims = a_shape.dimensions()
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  b = 1
  for d in batch_dims:
    b *= d

  if dtype == np.float32:
    fn = b"lapack_sgeqrf"
    lwork = _lapack.lapack_sgeqrf_workspace(m, n)
  elif dtype == np.float64:
    fn = b"lapack_dgeqrf"
    lwork = _lapack.lapack_dgeqrf_workspace(m, n)
  elif dtype == np.complex64:
    fn = b"lapack_cgeqrf"
    lwork = _lapack.lapack_cgeqrf_workspace(m, n)
  elif dtype == np.complex128:
    fn = b"lapack_zgeqrf"
    lwork = _lapack.lapack_zgeqrf_workspace(m, n)
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  out = _ops.CustomCallWithLayout(
    c, fn,
    operands=(
      _constant_s32_scalar(c, b),
      _constant_s32_scalar(c, m),
      _constant_s32_scalar(c, n),
      _constant_s32_scalar(c, lwork),
      a,
    ),
    shape_with_layout=Shape.tuple_shape((
        Shape.array_shape(
          dtype,
          batch_dims + (m, n),
          (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
        Shape.array_shape(
          np.dtype(dtype),
          batch_dims + (min(m, n),),
          tuple(range(num_bd, -1, -1))),
        Shape.array_shape(np.dtype(np.int32), batch_dims,
          tuple(range(num_bd - 1, -1, -1))),
        Shape.array_shape(np.dtype(dtype), (lwork,), (0,)),
    )),
    operand_shapes_with_layout=(
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(
          dtype,
          batch_dims + (m, n),
          (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
    ),
    api_version=xla_client.ops.CustomCallApiVersion
    .API_VERSION_STATUS_RETURNING)
  return tuple(_ops.GetTupleElement(out, i) for i in range(3))

def geqrf_mhlo(dtype, a):
  a_type = ir.RankedTensorType(a.type)
  dims = a_type.shape
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  b = 1
  for d in batch_dims:
    b *= d

  if dtype == np.float32:
    fn = b"lapack_sgeqrf"
    lwork = _lapack.lapack_sgeqrf_workspace(m, n)
  elif dtype == np.float64:
    fn = b"lapack_dgeqrf"
    lwork = _lapack.lapack_dgeqrf_workspace(m, n)
  elif dtype == np.complex64:
    fn = b"lapack_cgeqrf"
    lwork = _lapack.lapack_cgeqrf_workspace(m, n)
  elif dtype == np.complex128:
    fn = b"lapack_zgeqrf"
    lwork = _lapack.lapack_zgeqrf_workspace(m, n)
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  scalar_layout = ir.DenseIntElementsAttr.get(np.zeros((0,), np.int64),
                                              type=ir.IndexType.get())
  layout = ir.DenseIntElementsAttr.get(
      np.array((num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
      type=ir.IndexType.get())
  i32_type = ir.IntegerType.get_signless(32)
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
        a.type,
        ir.RankedTensorType.get(batch_dims + (min(m, n),), a_type.element_type),
        ir.RankedTensorType.get(batch_dims, i32_type),
        ir.RankedTensorType.get([lwork], a_type.element_type),
      ])],
      [_mhlo_s32(int(b)), _mhlo_s32(m), _mhlo_s32(n), _mhlo_s32(lwork), a],
      call_target_name = ir.StringAttr.get(fn),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(""),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([scalar_layout] * 4 + [layout]),
      result_layouts=ir.ArrayAttr.get([
        layout,
        ir.DenseIntElementsAttr.get(np.arange(num_bd, -1, -1),
                                    type=ir.IndexType.get()),
        ir.DenseIntElementsAttr.get(np.arange(num_bd - 1, -1, -1),
                                    type=ir.IndexType.get()),
        ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ]))
  return [
    mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, i)).result
    for i in range(3)
  ]


# # ?orgqr: product of elementary Householder reflectors:

def orgqr(c, a, tau):
  a_shape = c.get_shape(a)
  dtype = a_shape.element_type()
  dims = a_shape.dimensions()
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  b = 1
  for d in batch_dims:
    b *= d

  tau_dims = c.get_shape(tau).dimensions()
  assert tau_dims[:-1] == dims[:-2]
  k = tau_dims[-1]

  if dtype == np.float32:
    fn = b"lapack_sorgqr"
    lwork = _lapack.lapack_sorgqr_workspace(m, n, k)
  elif dtype == np.float64:
    fn = b"lapack_dorgqr"
    lwork = _lapack.lapack_dorgqr_workspace(m, n, k)
  elif dtype == np.complex64:
    fn = b"lapack_cungqr"
    lwork = _lapack.lapack_cungqr_workspace(m, n, k)
  elif dtype == np.complex128:
    fn = b"lapack_zungqr"
    lwork = _lapack.lapack_zungqr_workspace(m, n, k)
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  out = _ops.CustomCallWithLayout(
    c, fn,
    operands=(
      _constant_s32_scalar(c, b),
      _constant_s32_scalar(c, m),
      _constant_s32_scalar(c, n),
      _constant_s32_scalar(c, k),
      _constant_s32_scalar(c, lwork),
      a,
      tau,
    ),
    shape_with_layout=Shape.tuple_shape((
        Shape.array_shape(
          dtype,
          batch_dims + (m, n),
          (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
        Shape.array_shape(np.dtype(np.int32), batch_dims,
          tuple(range(num_bd - 1, -1, -1))),
        Shape.array_shape(dtype, (lwork,), (0,)),
    )),
    operand_shapes_with_layout=(
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(
          dtype,
          batch_dims + (m, n),
          (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
        Shape.array_shape(
          dtype,
          batch_dims + (k,),
          tuple(range(num_bd, -1, -1))),
    ),
    api_version=xla_client.ops.CustomCallApiVersion
    .API_VERSION_STATUS_RETURNING)
  return tuple(_ops.GetTupleElement(out, i) for i in range(2))


def orgqr_mhlo(dtype, a, tau):
  a_type = ir.RankedTensorType(a.type)
  dims = a_type.shape
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  b = 1
  for d in batch_dims:
    b *= d

  tau_dims = ir.RankedTensorType(tau.type).shape
  assert tau_dims[:-1] == dims[:-2]
  k = tau_dims[-1]

  if dtype == np.float32:
    fn = b"lapack_sorgqr"
    lwork = _lapack.lapack_sorgqr_workspace(m, n, k)
  elif dtype == np.float64:
    fn = b"lapack_dorgqr"
    lwork = _lapack.lapack_dorgqr_workspace(m, n, k)
  elif dtype == np.complex64:
    fn = b"lapack_cungqr"
    lwork = _lapack.lapack_cungqr_workspace(m, n, k)
  elif dtype == np.complex128:
    fn = b"lapack_zungqr"
    lwork = _lapack.lapack_zungqr_workspace(m, n, k)
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  scalar_layout = ir.DenseIntElementsAttr.get(np.zeros((0,), np.int64),
                                              type=ir.IndexType.get())
  layout = ir.DenseIntElementsAttr.get(
      np.array((num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
      type=ir.IndexType.get())
  i32_type = ir.IntegerType.get_signless(32)
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
        a.type,
        ir.RankedTensorType.get(batch_dims, i32_type),
        ir.RankedTensorType.get([lwork], a_type.element_type),
      ])],
      [_mhlo_s32(int(b)), _mhlo_s32(m), _mhlo_s32(n), _mhlo_s32(k),
       _mhlo_s32(lwork), a, tau],
      call_target_name = ir.StringAttr.get(fn),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(""),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([scalar_layout] * 5 + [
          layout,
          ir.DenseIntElementsAttr.get(np.arange(num_bd, -1, -1),
                                type=ir.IndexType.get()),
      ]),
      result_layouts=ir.ArrayAttr.get([
        layout,
        ir.DenseIntElementsAttr.get(np.arange(num_bd - 1, -1, -1),
                                    type=ir.IndexType.get()),
        ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ]))
  return [
    mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, i)).result
    for i in range(2)
  ]


# ?potrf: Cholesky decomposition

def potrf(c, a, lower=False):
  a_shape = c.get_shape(a)
  dtype = a_shape.element_type()
  dims = a_shape.dimensions()
  m, n = dims[-2:]
  if m != n:
    raise ValueError("potrf expects a square matrix, got {}".format(a_shape))
  if dtype == np.float32:
    fn = b"lapack_spotrf"
  elif dtype == np.float64:
    fn = b"lapack_dpotrf"
  elif dtype == np.complex64:
    fn = b"lapack_cpotrf"
  elif dtype == np.complex128:
    fn = b"lapack_zpotrf"
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  b = 1
  for d in batch_dims:
    b *= d

  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  out = _ops.CustomCallWithLayout(
    c, fn,
    operands=(_constant_s32_scalar(c, int(lower)),
              _constant_s32_scalar(c, b), _constant_s32_scalar(c, n), a),
    shape_with_layout=Shape.tuple_shape((
        Shape.array_shape(dtype, dims, layout),
        Shape.array_shape(
            np.dtype(np.int32), batch_dims, tuple(range(num_bd - 1, -1, -1))),
    )),
    operand_shapes_with_layout=(
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(dtype, dims, layout),
    ),
    api_version=xla_client.ops.CustomCallApiVersion
    .API_VERSION_STATUS_RETURNING)
  return tuple(_ops.GetTupleElement(out, i) for i in range(2))


def potrf_mhlo(dtype, a, lower=False):
  a_type = ir.RankedTensorType(a.type)
  dims = a_type.shape
  m, n = dims[-2:]
  if m != n:
    raise ValueError("potrf expects a square matrix, got {}".format(a_type))
  if dtype == np.float32:
    fn = b"lapack_spotrf"
  elif dtype == np.float64:
    fn = b"lapack_dpotrf"
  elif dtype == np.complex64:
    fn = b"lapack_cpotrf"
  elif dtype == np.complex128:
    fn = b"lapack_zpotrf"
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  b = 1
  for d in batch_dims:
    b *= d

  scalar_layout = ir.DenseIntElementsAttr.get(np.zeros((0,), np.int64),
                                              type=ir.IndexType.get())
  layout = ir.DenseIntElementsAttr.get(
      np.array((num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
      type=ir.IndexType.get())
  i32_type = ir.IntegerType.get_signless(32)
  info_layout = ir.DenseIntElementsAttr.get(
      np.array(range(num_bd - 1, -1, -1)), type=ir.IndexType.get())
  tup = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
          a.type,
          ir.RankedTensorType.get(batch_dims, i32_type),
      ])],
      [_mhlo_s32(int(lower)), _mhlo_s32(b), _mhlo_s32(n), a],
      call_target_name = ir.StringAttr.get(fn),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(""),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([scalar_layout] * 3 + [layout]),
      result_layouts=ir.ArrayAttr.get([layout, info_layout])).result
  return [
    mhlo.GetTupleElementOp(tup, ir.IntegerAttr.get(i32_type, 0)).result,
    mhlo.GetTupleElementOp(tup, ir.IntegerAttr.get(i32_type, 1)).result,
  ]



# # ?gesdd: Singular value decomposition

def gesdd(c, a, full_matrices=True, compute_uv=True):
  a_shape = c.get_shape(a)
  dtype = a_shape.element_type()
  dims = a_shape.dimensions()
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  b = 1
  for d in batch_dims:
    b *= d

  if dtype == np.float32:
    fn = b"lapack_sgesdd"
    singular_vals_dtype = np.float32
    lwork = _lapack.sgesdd_work_size(m, n, compute_uv, full_matrices)
    workspace = (
      Shape.array_shape(np.dtype(np.int32),
                        (_lapack.gesdd_iwork_size(m, n),), (0,)),
      Shape.array_shape(dtype, (lwork,), (0,)),
    )
  elif dtype == np.float64:
    fn = b"lapack_dgesdd"
    singular_vals_dtype = np.float64
    lwork = _lapack.dgesdd_work_size(m, n, compute_uv, full_matrices)
    workspace = (
      Shape.array_shape(np.dtype(np.int32),
                        (_lapack.gesdd_iwork_size(m, n),), (0,)),
      Shape.array_shape(dtype, (lwork,), (0,)),
    )
  elif dtype == np.complex64:
    fn = b"lapack_cgesdd"
    singular_vals_dtype = np.float32
    lwork = _lapack.cgesdd_work_size(m, n, compute_uv, full_matrices)
    workspace = (
      Shape.array_shape(np.dtype(np.int32), (_lapack.gesdd_iwork_size(m, n),),
                        (0,)),
      Shape.array_shape(np.dtype(np.float32),
                        (_lapack.cgesdd_rwork_size(m, n, int(compute_uv)),),
                        (0,)),
      Shape.array_shape(dtype, (lwork,), (0,)),
    )
  elif dtype == np.complex128:
    fn = b"lapack_zgesdd"
    singular_vals_dtype = np.float64
    lwork = _lapack.zgesdd_work_size(m, n, compute_uv, full_matrices)
    workspace = (
      Shape.array_shape(np.dtype(np.int32), (_lapack.gesdd_iwork_size(m, n),),
                        (0,)),
      Shape.array_shape(np.dtype(np.float64),
                        (_lapack.cgesdd_rwork_size(m, n, int(compute_uv)),),
                        (0,)),
      Shape.array_shape(dtype, (lwork,), (0,)),
    )
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  scalar_layout = tuple(range(num_bd - 1, -1, -1))
  vector_layout = (num_bd,) + scalar_layout
  matrix_layout = (num_bd, num_bd + 1) + scalar_layout
  out = _ops.CustomCallWithLayout(
    c, fn,
    operands=(_constant_s32_scalar(c, int(full_matrices)),
              _constant_s32_scalar(c, int(compute_uv)),
              _constant_s32_scalar(c, b),
              _constant_s32_scalar(c, m), _constant_s32_scalar(c, n),
              _constant_s32_scalar(c, lwork), a),
    shape_with_layout=Shape.tuple_shape((
        Shape.array_shape(dtype, batch_dims + (m, n), matrix_layout),
        Shape.array_shape(np.dtype(singular_vals_dtype),
                          batch_dims + (min(m, n),), vector_layout),
        Shape.array_shape(dtype,
                          batch_dims + (m, m if full_matrices else min(m, n)),
                          matrix_layout),
        Shape.array_shape(dtype,
                          batch_dims + (n if full_matrices else min(m, n), n),
                          matrix_layout),
        Shape.array_shape(np.dtype(np.int32), batch_dims, scalar_layout),
      ) + workspace
    ),
    operand_shapes_with_layout=(
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(dtype, batch_dims + (m, n), matrix_layout),
    ),
    api_version=xla_client.ops.CustomCallApiVersion
    .API_VERSION_STATUS_RETURNING)
  return (_ops.GetTupleElement(out, 1), _ops.GetTupleElement(out, 2),
          _ops.GetTupleElement(out, 3), _ops.GetTupleElement(out, 4))


def gesdd_mhlo(dtype, a, full_matrices=True, compute_uv=True):
  a_type = ir.RankedTensorType(a.type)
  dims = a_type.shape
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  b = 1
  for d in batch_dims:
    b *= d

  i32_type = ir.IntegerType.get_signless(32)
  if dtype == np.float32:
    fn = b"lapack_sgesdd"
    singular_vals_type = ir.F32Type.get()
    lwork = _lapack.sgesdd_work_size(m, n, compute_uv, full_matrices)
    workspace = [
        ir.RankedTensorType.get([_lapack.gesdd_iwork_size(m, n)], i32_type),
        ir.RankedTensorType.get([lwork], a_type.element_type),
    ]
    workspace_layouts = [
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
    ]
  elif dtype == np.float64:
    fn = b"lapack_dgesdd"
    singular_vals_type = ir.F64Type.get()
    lwork = _lapack.dgesdd_work_size(m, n, compute_uv, full_matrices)
    workspace = [
        ir.RankedTensorType.get([_lapack.gesdd_iwork_size(m, n)], i32_type),
        ir.RankedTensorType.get([lwork], a_type.element_type),
    ]
    workspace_layouts = [
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
    ]
  elif dtype == np.complex64:
    fn = b"lapack_cgesdd"
    singular_vals_type = ir.F32Type.get()
    lwork = _lapack.cgesdd_work_size(m, n, compute_uv, full_matrices)
    workspace = [
        ir.RankedTensorType.get([_lapack.gesdd_iwork_size(m, n)], i32_type),
        ir.RankedTensorType.get(
            [_lapack.cgesdd_rwork_size(m, n, int(compute_uv))],
            ir.F32Type.get()),
        ir.RankedTensorType.get([lwork], a_type.element_type),
    ]
    workspace_layouts = [
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
    ]
  elif dtype == np.complex128:
    fn = b"lapack_zgesdd"
    singular_vals_type = ir.F64Type.get()
    lwork = _lapack.zgesdd_work_size(m, n, compute_uv, full_matrices)
    workspace = [
        ir.RankedTensorType.get([_lapack.gesdd_iwork_size(m, n)], i32_type),
        ir.RankedTensorType.get(
            [_lapack.cgesdd_rwork_size(m, n, int(compute_uv))],
            ir.F64Type.get()),
        ir.RankedTensorType.get([lwork], a_type.element_type),
    ]
    workspace_layouts = [
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
    ]
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  scalar_layout = ir.DenseIntElementsAttr.get(np.zeros((0,), np.int64),
                                              type=ir.IndexType.get())
  layout = ir.DenseIntElementsAttr.get(
      np.array((num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
      type=ir.IndexType.get())
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
          a.type,
          ir.RankedTensorType.get(batch_dims + (min(m, n),), singular_vals_type),
          ir.RankedTensorType.get(
            batch_dims + (m, m if full_matrices else min(m, n)),
            a_type.element_type),
          ir.RankedTensorType.get(
            batch_dims + (n if full_matrices else min(m, n), n),
            a_type.element_type),
          ir.RankedTensorType.get(batch_dims, i32_type),
      ] + workspace)],
      [_mhlo_s32(int(full_matrices)), _mhlo_s32(int(compute_uv)), _mhlo_s32(b),
       _mhlo_s32(m), _mhlo_s32(n), _mhlo_s32(lwork), a],
      call_target_name = ir.StringAttr.get(fn),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(""),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([scalar_layout] * 6 + [layout]),
      result_layouts=ir.ArrayAttr.get([
          layout,
          ir.DenseIntElementsAttr.get(
              np.array((num_bd,) + tuple(range(num_bd - 1, -1, -1))),
              type=ir.IndexType.get()),
          layout,
          layout,
          ir.DenseIntElementsAttr.get(np.array(range(num_bd - 1, -1, -1)),
                                      type=ir.IndexType.get()),
      ] + workspace_layouts))
  return [
      mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, i)).result
      for i in range(1, 5)
  ]


# # syevd: Symmetric eigendecomposition

def syevd(c, a, lower=False):
  a_shape = c.get_shape(a)
  dtype = a_shape.element_type()
  dims = a_shape.dimensions()
  assert len(dims) >= 2
  m, n = dims[-2:]
  assert m == n
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  b = 1
  for d in batch_dims:
    b *= d
  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))

  if dtype == np.float32:
    fn = b"lapack_ssyevd"
    eigvals_type = np.float32
    workspace = (Shape.array_shape(dtype, (_lapack.syevd_work_size(n),), (0,)),
                 Shape.array_shape(np.dtype(np.int32),
                                   (_lapack.syevd_iwork_size(n),), (0,)))
  elif dtype == np.float64:
    fn = b"lapack_dsyevd"
    eigvals_type = np.float64
    workspace = (Shape.array_shape(dtype, (_lapack.syevd_work_size(n),), (0,)),
                 Shape.array_shape(np.dtype(np.int32),
                                   (_lapack.syevd_iwork_size(n),), (0,)))
  elif dtype == np.complex64:
    fn = b"lapack_cheevd"
    eigvals_type = np.float32
    workspace = (Shape.array_shape(dtype, (_lapack.heevd_work_size(n),), (0,)),
                 Shape.array_shape(np.dtype(np.float32),
                                   (_lapack.heevd_rwork_size(n),), (0,)),
                 Shape.array_shape(np.dtype(np.int32),
                                   (_lapack.syevd_iwork_size(n),), (0,)))
  elif dtype == np.complex128:
    fn = b"lapack_zheevd"
    eigvals_type = np.float64
    workspace = (Shape.array_shape(dtype, (_lapack.heevd_work_size(n),), (0,)),
                 Shape.array_shape(np.dtype(np.float64),
                                   (_lapack.heevd_rwork_size(n),), (0,)),
                 Shape.array_shape(np.dtype(np.int32),
                                   (_lapack.syevd_iwork_size(n),), (0,)))
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  out = _ops.CustomCallWithLayout(
    c, fn,
    operands=(_constant_s32_scalar(c, 1 if lower else 0),
              _constant_s32_scalar(c, b),
              _constant_s32_scalar(c, n),
              a),
    shape_with_layout=Shape.tuple_shape((
        Shape.array_shape(dtype, dims, layout),
        Shape.array_shape(np.dtype(eigvals_type), batch_dims + (n,),
                          tuple(range(num_bd, -1, -1))),
        Shape.array_shape(np.dtype(np.int32), batch_dims,
                          tuple(range(num_bd - 1, -1, -1))))
        + workspace
    ),
    operand_shapes_with_layout=(
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(dtype, dims, layout),
    ),
    api_version=xla_client.ops.CustomCallApiVersion
    .API_VERSION_STATUS_RETURNING)
  return (_ops.GetTupleElement(out, 0), _ops.GetTupleElement(out, 1),
          _ops.GetTupleElement(out, 2))

def syevd_mhlo(dtype, a, lower=False):
  a_type = ir.RankedTensorType(a.type)
  dims = a_type.shape
  assert len(dims) >= 2
  m, n = dims[-2:]
  assert m == n
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  b = 1
  for d in batch_dims:
    b *= d
  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))

  i32_type = ir.IntegerType.get_signless(32)
  if dtype == np.float32:
    fn = b"lapack_ssyevd"
    eigvals_type = ir.F32Type.get()
    workspace = [
        ir.RankedTensorType.get([_lapack.syevd_work_size(n)],
                                a_type.element_type),
        ir.RankedTensorType.get([_lapack.syevd_iwork_size(n)], i32_type),
    ]
    workspace_layouts = [
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
    ]
  elif dtype == np.float64:
    fn = b"lapack_dsyevd"
    eigvals_type = ir.F64Type.get()
    workspace = [
        ir.RankedTensorType.get([_lapack.syevd_work_size(n)],
                                a_type.element_type),
        ir.RankedTensorType.get([_lapack.syevd_iwork_size(n)], i32_type),
    ]
    workspace_layouts = [
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
    ]
  elif dtype == np.complex64:
    fn = b"lapack_cheevd"
    eigvals_type = ir.F32Type.get()
    workspace = [
        ir.RankedTensorType.get([_lapack.heevd_work_size(n)],
                                a_type.element_type),
        ir.RankedTensorType.get([_lapack.heevd_rwork_size(n)], eigvals_type),
        ir.RankedTensorType.get([_lapack.syevd_iwork_size(n)], i32_type),
    ]
    workspace_layouts = [
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
    ]
  elif dtype == np.complex128:
    fn = b"lapack_zheevd"
    eigvals_type = ir.F64Type.get()
    workspace = [
        ir.RankedTensorType.get([_lapack.heevd_work_size(n)],
                                a_type.element_type),
        ir.RankedTensorType.get([_lapack.heevd_rwork_size(n)], eigvals_type),
        ir.RankedTensorType.get([_lapack.syevd_iwork_size(n)], i32_type),
    ]
    workspace_layouts = [
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
    ]
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  scalar_layout = ir.DenseIntElementsAttr.get(np.zeros((0,), np.int64),
                                              type=ir.IndexType.get())
  layout = ir.DenseIntElementsAttr.get(
      np.array((num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
      type=ir.IndexType.get())
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
          a.type,
          ir.RankedTensorType.get(batch_dims + (n,), eigvals_type),
          ir.RankedTensorType.get(batch_dims, i32_type),
      ] + workspace)],
      [_mhlo_s32(1 if lower else 0), _mhlo_s32(b), _mhlo_s32(n), a],
      call_target_name = ir.StringAttr.get(fn),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(""),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([scalar_layout] * 3 + [layout]),
      result_layouts=ir.ArrayAttr.get([
          layout,
          ir.DenseIntElementsAttr.get(np.array(range(num_bd, -1, -1)),
                                      type=ir.IndexType.get()),
          ir.DenseIntElementsAttr.get(np.array(range(num_bd - 1, -1, -1)),
                                      type=ir.IndexType.get()),
      ] + workspace_layouts))
  return [
      mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, i)).result
      for i in range(3)
  ]


# # geev: Nonsymmetric eigendecomposition

def geev(c, a, jobvl=True, jobvr=True):
  a_shape = c.get_shape(a)
  dtype = a_shape.element_type()
  dims = a_shape.dimensions()
  assert len(dims) >= 2
  m, n = dims[-2:]
  assert m == n
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  b = 1
  for d in batch_dims:
    b *= d
  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))

  jobvl_c = ord('V' if jobvl else 'N')
  jobvr_c = ord('V' if jobvr else 'N')

  if dtype == np.float32:
    fn = b"lapack_sgeev"
    real = True
    eigvecs_type = np.complex64
    workspaces = (Shape.array_shape(np.dtype(np.float32), (n, n), (0, 1)),
                  Shape.array_shape(np.dtype(np.float32), (n, n), (0, 1)),
                  Shape.array_shape(np.dtype(np.float32), (n, n), (0, 1)))
    eigvals = (Shape.array_shape(np.dtype(np.float32), batch_dims + (n,),
                                 tuple(range(num_bd, -1, -1))),
               Shape.array_shape(np.dtype(np.float32), batch_dims + (n,),
                                 tuple(range(num_bd, -1, -1))))
  elif dtype == np.float64:
    fn = b"lapack_dgeev"
    real = True
    eigvecs_type = np.complex128
    workspaces = (Shape.array_shape(np.dtype(np.float64), (n, n), (0, 1)),
                  Shape.array_shape(np.dtype(np.float64), (n, n), (0, 1)),
                  Shape.array_shape(np.dtype(np.float64), (n, n), (0, 1)))
    eigvals = (Shape.array_shape(np.dtype(np.float64), batch_dims + (n,),
                                 tuple(range(num_bd, -1, -1))),
               Shape.array_shape(np.dtype(np.float64), batch_dims + (n,),
                                 tuple(range(num_bd, -1, -1))))
  elif dtype == np.complex64:
    fn = b"lapack_cgeev"
    real = False
    eigvecs_type = np.complex64
    workspaces = (Shape.array_shape(np.dtype(np.complex64), (n, n), (0, 1)),
                  Shape.array_shape(np.dtype(np.float32), (2 * n,), (0,)))
    eigvals = (Shape.array_shape(np.dtype(np.complex64), batch_dims + (n,),
                                 tuple(range(num_bd, -1, -1))),)
  elif dtype == np.complex128:
    fn = b"lapack_zgeev"
    real = False
    eigvecs_type = np.complex128
    workspaces = (Shape.array_shape(np.dtype(np.complex128), (n, n), (0, 1)),
                  Shape.array_shape(np.dtype(np.float64), (2 * n,), (0,)))
    eigvals = (Shape.array_shape(np.dtype(np.complex128), batch_dims + (n,),
                                 tuple(range(num_bd, -1, -1))),)
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  out = _ops.CustomCallWithLayout(
    c, fn,
    operands=(_constant_s32_scalar(c, b),
              _constant_s32_scalar(c, n),
              _ops.Constant(c, np.uint8(jobvl_c)),
              _ops.Constant(c, np.uint8(jobvr_c)),
              a),
    shape_with_layout=Shape.tuple_shape(workspaces + eigvals + (
        Shape.array_shape(np.dtype(eigvecs_type), dims, layout),
        Shape.array_shape(np.dtype(eigvecs_type), dims, layout),
        Shape.array_shape(np.dtype(np.int32), batch_dims,
                          tuple(range(num_bd - 1, -1, -1))))
    ),
    operand_shapes_with_layout=(
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.int32), (), ()),
        Shape.array_shape(np.dtype(np.uint8), (), ()),
        Shape.array_shape(np.dtype(np.uint8), (), ()),
        Shape.array_shape(dtype, dims, layout),
    ),
    api_version=xla_client.ops.CustomCallApiVersion
    .API_VERSION_STATUS_RETURNING)
  if real:
    return (_ops.Complex(_ops.GetTupleElement(out, 3),
                         _ops.GetTupleElement(out, 4)),
            _ops.GetTupleElement(out, 5), _ops.GetTupleElement(out, 6),
            _ops.GetTupleElement(out, 7))
  else:
    return (_ops.GetTupleElement(out, 2), _ops.GetTupleElement(out, 3),
            _ops.GetTupleElement(out, 4), _ops.GetTupleElement(out, 5))


def geev_mhlo(dtype, a, jobvl=True, jobvr=True):
  dims = ir.RankedTensorType(a.type).shape
  assert len(dims) >= 2
  m, n = dims[-2:]
  assert m == n
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  b = 1
  for d in batch_dims:
    b *= d
  layout = ir.DenseIntElementsAttr.get(
      np.array((num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
      type=ir.IndexType.get())

  jobvl_c = ord('V' if jobvl else 'N')
  jobvr_c = ord('V' if jobvr else 'N')

  if dtype == np.float32:
    fn = b"lapack_sgeev"
    real = True
    eigvecs_type = ir.ComplexType.get(ir.F32Type.get())
    workspaces = [ir.RankedTensorType.get([n, n], ir.F32Type.get()),
                  ir.RankedTensorType.get([n, n], ir.F32Type.get()),
                  ir.RankedTensorType.get([n, n], ir.F32Type.get())]
    workspace_layouts = [
      ir.DenseIntElementsAttr.get(np.array([0, 1]), type=ir.IndexType.get())
    ] * 3
    eigvals = [ir.RankedTensorType.get(batch_dims + (n,), ir.F32Type.get()),
               ir.RankedTensorType.get(batch_dims + (n,), ir.F32Type.get())]
    eigvals_layouts = [
      ir.DenseIntElementsAttr.get(np.arange(num_bd, -1, -1),
                                  type=ir.IndexType.get())
    ] * 2
  elif dtype == np.float64:
    fn = b"lapack_dgeev"
    real = True
    eigvecs_type = ir.ComplexType.get(ir.F64Type.get())
    workspaces = [ir.RankedTensorType.get([n, n], ir.F64Type.get()),
                  ir.RankedTensorType.get([n, n], ir.F64Type.get()),
                  ir.RankedTensorType.get([n, n], ir.F64Type.get())]
    workspace_layouts = [
      ir.DenseIntElementsAttr.get(np.array([0, 1]), type=ir.IndexType.get())
    ] * 3
    eigvals = [ir.RankedTensorType.get(batch_dims + (n,), ir.F64Type.get()),
               ir.RankedTensorType.get(batch_dims + (n,), ir.F64Type.get())]
    eigvals_layouts = [
      ir.DenseIntElementsAttr.get(np.arange(num_bd, -1, -1),
                                  type=ir.IndexType.get())
    ] * 2
  elif dtype == np.complex64:
    fn = b"lapack_cgeev"
    real = False
    eigvecs_type = ir.ComplexType.get(ir.F32Type.get())
    workspaces = [ir.RankedTensorType.get([n, n],
                                          ir.ComplexType.get(ir.F32Type.get())),
                  ir.RankedTensorType.get([2 * n], ir.F32Type.get())]
    workspace_layouts = [
      ir.DenseIntElementsAttr.get(np.array([0, 1]), type=ir.IndexType.get()),
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get())
    ]
    eigvals = [ir.RankedTensorType.get(batch_dims + (n,),
                                       ir.ComplexType.get(ir.F32Type.get()))]
    eigvals_layouts = [
      ir.DenseIntElementsAttr.get(np.arange(num_bd, -1, -1),
                                  type=ir.IndexType.get())
    ]
  elif dtype == np.complex128:
    fn = b"lapack_zgeev"
    real = False
    eigvecs_type = ir.ComplexType.get(ir.F64Type.get())
    workspaces = [ir.RankedTensorType.get([n, n],
                                          ir.ComplexType.get(ir.F64Type.get())),
                  ir.RankedTensorType.get([2 * n], ir.F64Type.get())]
    workspace_layouts = [
      ir.DenseIntElementsAttr.get(np.array([0, 1]), type=ir.IndexType.get()),
      ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get())
    ]
    eigvals = [ir.RankedTensorType.get(batch_dims + (n,),
                                       ir.ComplexType.get(ir.F64Type.get()))]
    eigvals_layouts = [
      ir.DenseIntElementsAttr.get(np.arange(num_bd, -1, -1),
                                  type=ir.IndexType.get())
    ]
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  i32_type = ir.IntegerType.get_signless(32)
  scalar_layout = ir.DenseIntElementsAttr.get(np.zeros((0,), np.int64),
                                              type=ir.IndexType.get())
  info_layout = ir.DenseIntElementsAttr.get(np.arange(num_bd - 1, -1, -1),
                                            type=ir.IndexType.get())
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple(workspaces + eigvals + [
        ir.RankedTensorType.get(dims, eigvecs_type),
        ir.RankedTensorType.get(dims, eigvecs_type),
        ir.RankedTensorType.get(batch_dims, i32_type),
      ])],
      [_mhlo_s32(b), _mhlo_s32(n), _mhlo_u8(jobvl_c), _mhlo_u8(jobvr_c), a],
      call_target_name = ir.StringAttr.get(fn),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(""),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([scalar_layout] * 4 + [layout]),
      result_layouts=ir.ArrayAttr.get(
          workspace_layouts + eigvals_layouts + [layout] * 2 + [info_layout])
  ).result
  i32_attr = lambda i: ir.IntegerAttr.get(i32_type, i)
  if real:
    return (mhlo.ComplexOp(mhlo.GetTupleElementOp(out, i32_attr(3)),
                           mhlo.GetTupleElementOp(out, i32_attr(4))).result,
            mhlo.GetTupleElementOp(out, i32_attr(5)).result,
            mhlo.GetTupleElementOp(out, i32_attr(6)).result,
            mhlo.GetTupleElementOp(out, i32_attr(7)).result)
  else:
    return (mhlo.GetTupleElementOp(out, i32_attr(2)).result,
            mhlo.GetTupleElementOp(out, i32_attr(3)).result,
            mhlo.GetTupleElementOp(out, i32_attr(4)).result,
            mhlo.GetTupleElementOp(out, i32_attr(5)).result)

# # gees : Schur factorization

def gees(c, a, jobvs=True, sort=False, select=None):
  a_shape = c.get_shape(a)
  dtype = a_shape.element_type()
  dims = a_shape.dimensions()
  assert len(dims) >= 2
  m, n = dims[-2:]
  assert m == n
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  b = 1
  for d in batch_dims:
    b *= d
  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))

  if sort:
    raise NotImplementedError(
        "The sort feature of LAPACK's gees routine is not implemented.")

  jobvs = ord('V' if jobvs else 'N')
  sort = ord('S' if sort else 'N')

  if dtype == np.float32 or dtype == np.float64:
    fn = b"lapack_sgees" if dtype == np.float32 else b"lapack_dgees"
    schurvecs_type = dtype
    workspaces = (Shape.array_shape(np.dtype(schurvecs_type), dims, layout),)
    eigvals = (Shape.array_shape(
        np.dtype(dtype), batch_dims + (n,), tuple(range(num_bd, -1, -1))),
               Shape.array_shape(
                   np.dtype(dtype), batch_dims + (n,),
                   tuple(range(num_bd, -1, -1))))
  elif dtype == np.complex64 or dtype == np.complex128:
    fn = b"lapack_cgees" if dtype == np.complex64 else b"lapack_zgees"
    schurvecs_type = dtype
    workspaces = (
        Shape.array_shape(np.dtype(schurvecs_type), dims, layout),
        Shape.array_shape(
            np.dtype(np.float32 if dtype == np.complex64 else np.float64),
            (n,), (0,)))
    eigvals = (Shape.array_shape(
        np.dtype(dtype), batch_dims + (n,), tuple(range(num_bd, -1, -1))),)
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  out = _ops.CustomCallWithLayout(
      c,
      fn,
      operands=(
          _constant_s32_scalar(c, b),
          _constant_s32_scalar(c, n),
          _ops.Constant(c, np.uint8(jobvs)),
          _ops.Constant(c, np.uint8(sort)),
          #figure out how to put the callable select function here
          a),
      shape_with_layout=Shape.tuple_shape(workspaces + eigvals + (
          Shape.array_shape(np.dtype(schurvecs_type), dims, layout),
          Shape.array_shape(
              np.dtype(np.int32), batch_dims, tuple(range(num_bd - 1, -1, -1))),
          Shape.array_shape(
              np.dtype(np.int32), batch_dims, tuple(range(num_bd -
                                                          1, -1, -1))))),
      operand_shapes_with_layout=(
          Shape.array_shape(np.dtype(np.int32), (), ()),
          Shape.array_shape(np.dtype(np.int32), (), ()),
          Shape.array_shape(np.dtype(np.uint8), (), ()),
          Shape.array_shape(np.dtype(np.uint8), (), ()),
          Shape.array_shape(dtype, dims, layout),
      ),
      api_version=xla_client.ops.CustomCallApiVersion
      .API_VERSION_STATUS_RETURNING)
  if sort == ord('S'):
    return (_ops.GetTupleElement(out, 0), _ops.GetTupleElement(out, 3),
            _ops.GetTupleElement(out, 4), _ops.GetTupleElement(out, 5))
  else:
    return (_ops.GetTupleElement(out, 0), _ops.GetTupleElement(out, 3),
            _ops.GetTupleElement(out, 5))

def gees_mhlo(a, jobvs=True, sort=False, select=None):
  a_type = ir.RankedTensorType(a.type)
  etype = a_type.element_type
  dims = a_type.shape
  assert len(dims) >= 2
  m, n = dims[-2:]
  assert m == n
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  b = 1
  for d in batch_dims:
    b *= d
  layout = ir.DenseIntElementsAttr.get(
      np.array((num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
      type=ir.IndexType.get())

  if sort:
    raise NotImplementedError(
        "The sort feature of LAPACK's gees routine is not implemented.")

  jobvs = ord('V' if jobvs else 'N')
  sort = ord('S' if sort else 'N')

  if not ir.ComplexType.isinstance(etype):
    fn = "lapack_sgees" if etype == ir.F32Type.get() else "lapack_dgees"
    schurvecs_type = etype
    workspaces = [ir.RankedTensorType.get(dims, schurvecs_type)]
    workspace_layouts = [layout]
    eigvals = [ir.RankedTensorType.get(batch_dims + (n,), etype)] * 2
    eigvals_layouts = [
        ir.DenseIntElementsAttr.get(np.arange(num_bd, -1, -1),
                                    type=ir.IndexType.get())
    ] * 2
  else:
    fn = ("lapack_cgees" if etype == ir.ComplexType.get(ir.F32Type.get())
          else "lapack_zgees")
    schurvecs_type = etype
    workspaces = [
        ir.RankedTensorType.get(dims, schurvecs_type),
        ir.RankedTensorType.get([n], ir.ComplexType(etype).element_type),
    ]
    workspace_layouts = [
        layout,
        ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
    ]
    eigvals = [ir.RankedTensorType.get(batch_dims + (n,), etype)]
    eigvals_layouts = [
        ir.DenseIntElementsAttr.get(np.arange(num_bd, -1, -1),
                                    type=ir.IndexType.get())
    ]

  i32_type = ir.IntegerType.get_signless(32)

  scalar_layout = ir.DenseIntElementsAttr.get(np.zeros((0,), np.int64),
                                              type=ir.IndexType.get())
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple(workspaces + eigvals + [
        ir.RankedTensorType.get(dims, schurvecs_type),
        ir.RankedTensorType.get(batch_dims, i32_type),
        ir.RankedTensorType.get(batch_dims, i32_type),
      ])],
      [
          _mhlo_s32(b),
          _mhlo_s32(n),
          _mhlo_u8(np.uint8(jobvs)),
          _mhlo_u8(np.uint8(sort)),
          # TODO: figure out how to put the callable select function here
          a
      ],
      call_target_name = ir.StringAttr.get(fn),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(""),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([scalar_layout] * 4 + [layout]),
      result_layouts=ir.ArrayAttr.get(workspace_layouts + eigvals_layouts + [
          layout,
          ir.DenseIntElementsAttr.get(np.arange(num_bd - 1, -1, -1),
                                      type=ir.IndexType.get()),
          ir.DenseIntElementsAttr.get(np.arange(num_bd - 1, -1, -1),
                                      type=ir.IndexType.get()),
      ])
  )
  i32_attr = lambda i: ir.IntegerAttr.get(i32_type, i)
  if sort == ord('S'):
    return (mhlo.GetTupleElementOp(out, i32_attr(0)).result,
            mhlo.GetTupleElementOp(out, i32_attr(3)).result,
            mhlo.GetTupleElementOp(out, i32_attr(4)).result,
            mhlo.GetTupleElementOp(out, i32_attr(5)).result)
  else:
    return (mhlo.GetTupleElementOp(out, i32_attr(0)).result,
            mhlo.GetTupleElementOp(out, i32_attr(3)).result,
            mhlo.GetTupleElementOp(out, i32_attr(5)).result)
