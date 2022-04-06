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

import functools
import operator

import jaxlib.mlir.ir as ir
import jaxlib.mlir.dialects.mhlo as mhlo

import numpy as np

from jaxlib import xla_client

try:
  from . import _hipblas
  for _name, _value in _hipblas.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="ROCM")
except ImportError:
  pass

try:
  from . import _hipsolver
  for _name, _value in _hipsolver.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="ROCM")
except ImportError:
  pass

_ops = xla_client.ops
_Shape = xla_client.Shape


def _real_type(dtype):
  """Returns the real equivalent of 'dtype'."""
  return np.finfo(dtype).dtype


_prod = lambda xs: functools.reduce(operator.mul, xs, 1)


def trsm(c,
         a,
         b,
         left_side=False,
         lower=False,
         trans_a=False,
         conj_a=False,
         diag=False):
  """Batched triangular solve.

  XLA implements unbatched triangular solve directly, so we need only implement
  the batched case."""
  b_shape = c.get_shape(b)
  dtype = b_shape.element_type()
  dims = b_shape.dimensions()
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  batch = _prod(batch_dims)
  k = m if left_side else n

  a_shape = c.get_shape(a)
  if (batch_dims + (k, k) != a_shape.dimensions()
      or a_shape.element_type() != dtype):
    raise ValueError("Argument mismatch for trsm, got {} and {}".format(
        a_shape, b_shape))

  if conj_a and not trans_a:
    raise NotImplementedError(
        "Conjugation without transposition not supported")

  lwork, opaque = _hipblas.build_trsm_batched_descriptor(
      np.dtype(dtype), batch, m, n, left_side, lower, trans_a, conj_a, diag)
  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  out = _ops.CustomCallWithLayout(
      c,
      b"hipblas_trsm_batched",
      operands=(a, b),
      shape_with_layout=_Shape.tuple_shape(
          (_Shape.array_shape(dtype, b_shape.dimensions(), layout),
           _Shape.array_shape(np.dtype(np.int8), (lwork, ), (0, )),
           _Shape.array_shape(np.dtype(np.int8), (lwork, ), (0, )))),
      operand_shapes_with_layout=(
          _Shape.array_shape(dtype, a_shape.dimensions(), layout),
          _Shape.array_shape(dtype, b_shape.dimensions(), layout),
      ),
      opaque=opaque,
      api_version=xla_client.ops.CustomCallApiVersion.
      API_VERSION_STATUS_RETURNING)
  return _ops.GetTupleElement(out, 0)

def trsm_mhlo(dtype, a, b, left_side=False, lower=False, trans_a=False,
              conj_a=False, diag=False):
  """Batched triangular solve.

  XLA implements unbatched triangular solve directly, so we need only implement
  the batched case."""
  b_type = ir.RankedTensorType(b.type)
  dims = b_type.shape
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  batch = _prod(batch_dims)
  k = m if left_side else n

  a_type = ir.RankedTensorType(a.type)
  if (batch_dims + (k, k) != tuple(a_type.shape) or
      a_type.element_type != b_type.element_type):
    raise ValueError("Argument mismatch for trsm, got {} and {}".format(
      a_type, b_type))

  if conj_a and not trans_a:
    raise NotImplementedError("Conjugation without transposition not supported")

  lwork, opaque = _hipblas.build_trsm_batched_descriptor(
    np.dtype(dtype), batch, m, n, left_side, lower, trans_a, conj_a, diag)
  layout = ir.DenseIntElementsAttr.get(
      np.array((num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
      type=ir.IndexType.get())
  work_type = ir.RankedTensorType.get([lwork], ir.IntegerType.get_signless(8))
  work_layout = ir.DenseIntElementsAttr.get(np.array([0]),
                                            type=ir.IndexType.get())
  i32_type = ir.IntegerType.get_signless(32)
  tup = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([b_type, work_type, work_type])],
      [a, b],
      call_target_name = ir.StringAttr.get("hipblas_trsm_batched"),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(opaque),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([layout] * 2),
      result_layouts=ir.ArrayAttr.get([layout, work_layout, work_layout])).result
  return mhlo.GetTupleElementOp(
      tup, ir.IntegerAttr.get(i32_type, 0)).result

def potrf(c, a, lower):
  """Cholesky decomposition."""
  a_shape = c.get_shape(a)
  dtype = a_shape.element_type()
  dims = a_shape.dimensions()
  m, n = dims[-2:]
  assert m == n
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  batch = _prod(batch_dims)

  lwork, opaque = _hipsolver.build_potrf_descriptor(np.dtype(dtype), lower,
                                                    batch, n)
  kernel = b"hipsolver_potrf"

  out = _ops.CustomCallWithLayout(
      c,
      kernel,
      operands=(a, ),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(dtype, batch_dims + (n, n), (num_bd, num_bd + 1) +
                             tuple(range(num_bd - 1, -1, -1))),
          _Shape.array_shape(np.dtype(np.int32), batch_dims,
                             tuple(range(num_bd - 1, -1, -1))),
          _Shape.array_shape(np.dtype(np.int8), (lwork, ), (0, )),
      )),
      operand_shapes_with_layout=(_Shape.array_shape(
          dtype, batch_dims + (n, n),
          (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))), ),
      opaque=opaque,
      api_version=xla_client.ops.CustomCallApiVersion.
      API_VERSION_STATUS_RETURNING)
  return _ops.GetTupleElement(out, 0), _ops.GetTupleElement(out, 1)


def potrf_mhlo(dtype, a, lower):
  """Cholesky decomposition."""
  a_type = ir.RankedTensorType(a.type)
  dims = a_type.shape
  m, n = dims[-2:]
  assert m == n
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  batch = _prod(batch_dims)

  lwork, opaque = _hipsolver.build_potrf_descriptor(
      np.dtype(dtype), lower, batch, n)
  kernel = b"hipsolver_potrf"

  layout = ir.DenseIntElementsAttr.get(
      np.array((num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
      type=ir.IndexType.get())
  info_layout = ir.DenseIntElementsAttr.get(
      np.array(range(num_bd - 1, -1, -1)), type=ir.IndexType.get())
  i32_type = ir.IntegerType.get_signless(32)
  info_type = ir.RankedTensorType.get(batch_dims, i32_type)
  work_layout = ir.DenseIntElementsAttr.get(np.array([0]),
                                            type=ir.IndexType.get())
  tup = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
        a.type,
        info_type,
        ir.RankedTensorType.get([lwork], ir.IntegerType.get_signless(8)),
      ])],
      [a],
      call_target_name = ir.StringAttr.get(kernel),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(opaque),
      api_version=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([layout]),
      result_layouts=ir.ArrayAttr.get([layout, info_layout, work_layout])).result
  return [
    mhlo.GetTupleElementOp(tup, ir.IntegerAttr.get(i32_type, 0)).result,
    mhlo.GetTupleElementOp(tup, ir.IntegerAttr.get(i32_type, 1)).result,
  ]


def getrf(c, a):
  """LU decomposition."""
  a_shape = c.get_shape(a)
  dtype = a_shape.element_type()
  dims = a_shape.dimensions()
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  batch = _prod(batch_dims)

  if batch > 1 and m == n and m // batch <= 128:
    lwork, opaque = _hipblas.build_getrf_batched_descriptor(
        np.dtype(dtype), batch, m)
    workspace = _Shape.array_shape(np.dtype(np.int8), (lwork, ), (0, ))
    kernel = b"hipblas_getrf_batched"
  else:
    lwork, opaque = _hipsolver.build_getrf_descriptor(np.dtype(dtype), batch,
                                                      m, n)
    workspace = _Shape.array_shape(dtype, (lwork, ), (0, ))
    kernel = b"hipsolver_getrf"

  out = _ops.CustomCallWithLayout(
      c,
      kernel,
      operands=(a, ),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(dtype, batch_dims + (m, n), (num_bd, num_bd + 1) +
                             tuple(range(num_bd - 1, -1, -1))),
          _Shape.array_shape(np.dtype(np.int32), batch_dims + (min(m, n), ),
                             tuple(range(num_bd, -1, -1))),
          _Shape.array_shape(np.dtype(np.int32), batch_dims,
                             tuple(range(num_bd - 1, -1, -1))),
          workspace,
      )),
      operand_shapes_with_layout=(_Shape.array_shape(
          dtype, batch_dims + (m, n),
          (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))), ),
      opaque=opaque,
      api_version=xla_client.ops.CustomCallApiVersion.
      API_VERSION_STATUS_RETURNING)
  return (_ops.GetTupleElement(out, 0), _ops.GetTupleElement(out, 1),
          _ops.GetTupleElement(out, 2))


def getrf_mhlo(dtype, a):
  """LU decomposition."""
  a_type = ir.RankedTensorType(a.type)
  dims = a_type.shape
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  batch = _prod(batch_dims)

  if batch > 1 and m == n and m // batch <= 128:
    lwork, opaque = _hipblas.build_getrf_batched_descriptor(
      np.dtype(dtype), batch, m)
    workspace = ir.RankedTensorType.get([lwork], ir.IntegerType.get_signless(8))
    kernel = "hipblas_getrf_batched"
  else:
    lwork, opaque = _hipsolver.build_getrf_descriptor(
        np.dtype(dtype), batch, m, n)
    workspace = ir.RankedTensorType.get([lwork], a_type.element_type)
    kernel = "hipsolver_getrf"

  layout = ir.DenseIntElementsAttr.get(
      np.array((num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
      type=ir.IndexType.get())
  i32_type = ir.IntegerType.get_signless(32)
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
        a.type,
        ir.RankedTensorType.get(batch_dims + (min(m, n),), i32_type),
        ir.RankedTensorType.get(batch_dims, i32_type),
        workspace,
      ])],
      [a],
      call_target_name = ir.StringAttr.get(kernel),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(opaque),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([layout]),
      result_layouts=ir.ArrayAttr.get([
        layout,
        ir.DenseIntElementsAttr.get(np.arange(num_bd, -1, -1),
                                    type=ir.IndexType.get()),
        ir.DenseIntElementsAttr.get(np.arange(num_bd - 1, -1, -1),
                                    type=ir.IndexType.get()),
        ir.DenseIntElementsAttr.get(np.array([0]),
                                    type=ir.IndexType.get()),
      ]))
  return [
    mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, i)).result
    for i in range(3)
  ]


def geqrf(c, a):
  """QR decomposition."""
  a_shape = c.get_shape(a)
  dtype = a_shape.element_type()
  dims = a_shape.dimensions()
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  batch = _prod(batch_dims)

  lwork, opaque = _hipsolver.build_geqrf_descriptor(np.dtype(dtype), batch, m,
                                                    n)
  workspace = _Shape.array_shape(dtype, (lwork, ), (0, ))
  kernel = b"hipsolver_geqrf"

  out = _ops.CustomCallWithLayout(
      c,
      kernel,
      operands=(a, ),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(dtype, batch_dims + (m, n), (num_bd, num_bd + 1) +
                             tuple(range(num_bd - 1, -1, -1))),
          _Shape.array_shape(dtype, batch_dims + (min(m, n), ),
                             tuple(range(num_bd, -1, -1))),
          _Shape.array_shape(np.dtype(np.int32), batch_dims,
                             tuple(range(num_bd - 1, -1, -1))),
          workspace,
      )),
      operand_shapes_with_layout=(_Shape.array_shape(
          dtype, batch_dims + (m, n),
          (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))), ),
      opaque=opaque,
      api_version=xla_client.ops.CustomCallApiVersion.
      API_VERSION_STATUS_RETURNING)
  return (_ops.GetTupleElement(out, 0), _ops.GetTupleElement(out, 1),
          _ops.GetTupleElement(out, 2))

def geqrf_mhlo(dtype, a):
  """QR decomposition."""
  a_type = ir.RankedTensorType(a.type)
  dims = a_type.shape
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  batch = _prod(batch_dims)

  lwork, opaque = _hipsolver.build_geqrf_descriptor(
      np.dtype(dtype), batch, m, n)

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
      [a],
      call_target_name = ir.StringAttr.get(b"hipsolver_geqrf"),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(opaque),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([layout]),
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


def orgqr(c, a, tau):
  """Product of elementary Householder reflections."""
  a_shape = c.get_shape(a)
  dtype = a_shape.element_type()
  dims = a_shape.dimensions()
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  batch = _prod(batch_dims)

  tau_dims = c.get_shape(tau).dimensions()
  assert tau_dims[:-1] == dims[:-2]
  k = tau_dims[-1]

  lwork, opaque = _hipsolver.build_orgqr_descriptor(np.dtype(dtype), batch, m,
                                                    n, k)
  workspace = _Shape.array_shape(dtype, (lwork, ), (0, ))
  kernel = b"hipsolver_orgqr"

  out = _ops.CustomCallWithLayout(
      c,
      kernel,
      operands=(a, tau),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(dtype, batch_dims + (m, n), (num_bd, num_bd + 1) +
                             tuple(range(num_bd - 1, -1, -1))),
          _Shape.array_shape(np.dtype(np.int32), batch_dims,
                             tuple(range(num_bd - 1, -1, -1))),
          workspace,
      )),
      operand_shapes_with_layout=(
          _Shape.array_shape(dtype, batch_dims + (m, n), (num_bd, num_bd + 1) +
                             tuple(range(num_bd - 1, -1, -1))),
          _Shape.array_shape(dtype, batch_dims + (k, ),
                             tuple(range(num_bd, -1, -1))),
      ),
      opaque=opaque,
      api_version=xla_client.ops.CustomCallApiVersion.
      API_VERSION_STATUS_RETURNING)
  return (_ops.GetTupleElement(out, 0), _ops.GetTupleElement(out, 1))


def orgqr_mhlo(dtype, a, tau):
  """Product of elementary Householder reflections."""
  a_type = ir.RankedTensorType(a.type)
  dims = a_type.shape
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  batch = _prod(batch_dims)

  tau_dims = ir.RankedTensorType(tau.type).shape
  assert tau_dims[:-1] == dims[:-2]
  k = tau_dims[-1]

  lwork, opaque = _hipsolver.build_orgqr_descriptor(
      np.dtype(dtype), batch, m, n, k)

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
      [a, tau],
      call_target_name = ir.StringAttr.get(b"hipsolver_orgqr"),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(opaque),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([
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



def syevd(c, a, lower=False):
  """Symmetric (Hermitian) eigendecomposition."""
  a_shape = c.get_shape(a)
  dtype = a_shape.element_type()
  dims = a_shape.dimensions()
  assert len(dims) >= 2
  m, n = dims[-2:]
  assert m == n
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  batch = _prod(batch_dims)
  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))

  # TODO(rocm): rocm does not support jacobian method.
  kernel = b"hipsolver_syevd"
  lwork, opaque = _hipsolver.build_syevd_descriptor(np.dtype(dtype), lower,
                                                    batch, n)
  eigvals_type = _real_type(dtype)

  out = _ops.CustomCallWithLayout(
      c,
      kernel,
      operands=(a, ),
      shape_with_layout=_Shape.tuple_shape(
          (_Shape.array_shape(dtype, dims, layout),
           _Shape.array_shape(np.dtype(eigvals_type), batch_dims + (n, ),
                              tuple(range(num_bd, -1, -1))),
           _Shape.array_shape(np.dtype(np.int32), batch_dims,
                              tuple(range(num_bd - 1, -1, -1))),
           _Shape.array_shape(dtype, (lwork, ), (0, )))),
      operand_shapes_with_layout=(_Shape.array_shape(dtype, dims, layout), ),
      opaque=opaque,
      api_version=xla_client.ops.CustomCallApiVersion.
      API_VERSION_STATUS_RETURNING)
  return (_ops.GetTupleElement(out, 0), _ops.GetTupleElement(out, 1),
          _ops.GetTupleElement(out, 2))

def syevd_mhlo(dtype, a, lower=False):
  """Symmetric (Hermitian) eigendecomposition."""
  a_type = ir.RankedTensorType(a.type)
  dims = a_type.shape
  assert len(dims) >= 2
  m, n = dims[-2:]
  assert m == n
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  batch = _prod(batch_dims)
  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))

  kernel = b"hipsolver_syevd"
  lwork, opaque = _hipsolver.build_syevd_descriptor(
      np.dtype(dtype), lower, batch, n)

  if ir.ComplexType.isinstance(a_type.element_type):
    eigvals_type = ir.ComplexType(a_type.element_type).element_type
  else:
    eigvals_type = a_type.element_type

  layout = ir.DenseIntElementsAttr.get(
      np.array((num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
      type=ir.IndexType.get())
  i32_type = ir.IntegerType.get_signless(32)
  out = mhlo.CustomCallOp(
      [ir.TupleType.get_tuple([
          a.type,
          ir.RankedTensorType.get(batch_dims + (n,), eigvals_type),
          ir.RankedTensorType.get(batch_dims, i32_type),
          ir.RankedTensorType.get([lwork], a_type.element_type),
      ])],
      [a],
      call_target_name = ir.StringAttr.get(kernel),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(opaque),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      operand_layouts=ir.ArrayAttr.get([layout]),
      result_layouts=ir.ArrayAttr.get([
          layout,
          ir.DenseIntElementsAttr.get(np.array(range(num_bd, -1, -1)),
                                      type=ir.IndexType.get()),
          ir.DenseIntElementsAttr.get(np.array(range(num_bd - 1, -1, -1)),
                                      type=ir.IndexType.get()),
          ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
      ]))
  return [
      mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, i)).result
      for i in range(3)
  ]

def gesvd(c, a, full_matrices=True, compute_uv=True):
  """Singular value decomposition."""
  a_shape = c.get_shape(a)
  dims = a_shape.dimensions()
  dtype = a_shape.element_type()
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  b = _prod(batch_dims)
  singular_vals_dtype = np.dtype(_real_type(dtype))

  # TODO(rocm): rocm does not support jacobian method.
  # for cuda, jax uses jacobian method for small size matrixes
  if m < n:
    lwork, opaque = _hipsolver.build_gesvd_descriptor(np.dtype(dtype), b, n, m,
                                                      compute_uv,
                                                      full_matrices)
    scalar_layout = tuple(range(num_bd - 1, -1, -1))
    vector_layout = (num_bd, ) + scalar_layout
    matrix_layout = (num_bd + 1, num_bd) + scalar_layout
    out = _ops.CustomCallWithLayout(
        c,
        b"hipsolver_gesvd",
        operands=(a, ),
        shape_with_layout=_Shape.tuple_shape((
            _Shape.array_shape(dtype, batch_dims + (m, n), matrix_layout),
            _Shape.array_shape(singular_vals_dtype, batch_dims + (min(m, n), ),
                               vector_layout),
            _Shape.array_shape(dtype, batch_dims + (n, n), matrix_layout),
            _Shape.array_shape(dtype, batch_dims + (m, m), matrix_layout),
            _Shape.array_shape(np.dtype(np.int32), batch_dims, scalar_layout),
            _Shape.array_shape(dtype, (lwork, ), (0, )),
        )),
        operand_shapes_with_layout=(_Shape.array_shape(dtype,
                                                       batch_dims + (m, n),
                                                       matrix_layout), ),
        opaque=opaque,
        api_version=xla_client.ops.CustomCallApiVersion.
        API_VERSION_STATUS_RETURNING)
    s = _ops.GetTupleElement(out, 1)
    vt = _ops.GetTupleElement(out, 2)
    u = _ops.GetTupleElement(out, 3)
    info = _ops.GetTupleElement(out, 4)
  else:
    lwork, opaque = _hipsolver.build_gesvd_descriptor(np.dtype(dtype), b, m, n,
                                                      compute_uv,
                                                      full_matrices)

    scalar_layout = tuple(range(num_bd - 1, -1, -1))
    vector_layout = (num_bd, ) + scalar_layout
    matrix_layout = (num_bd, num_bd + 1) + scalar_layout
    out = _ops.CustomCallWithLayout(
        c,
        b"hipsolver_gesvd",
        operands=(a, ),
        shape_with_layout=_Shape.tuple_shape((
            _Shape.array_shape(dtype, batch_dims + (m, n), matrix_layout),
            _Shape.array_shape(singular_vals_dtype, batch_dims + (min(m, n), ),
                               vector_layout),
            _Shape.array_shape(dtype, batch_dims + (m, m), matrix_layout),
            _Shape.array_shape(dtype, batch_dims + (n, n), matrix_layout),
            _Shape.array_shape(np.dtype(np.int32), batch_dims, scalar_layout),
            _Shape.array_shape(dtype, (lwork, ), (0, )),
        )),
        operand_shapes_with_layout=(_Shape.array_shape(dtype,
                                                       batch_dims + (m, n),
                                                       matrix_layout), ),
        opaque=opaque,
        api_version=xla_client.ops.CustomCallApiVersion.
        API_VERSION_STATUS_RETURNING)
    s = _ops.GetTupleElement(out, 1)
    u = _ops.GetTupleElement(out, 2)
    vt = _ops.GetTupleElement(out, 3)
    info = _ops.GetTupleElement(out, 4)
  if not full_matrices:
    u = _ops.Slice(u, (0, ) * len(dims), batch_dims + (m, min(m, n)),
                   (1, ) * len(dims))
    vt = _ops.Slice(vt, (0, ) * len(dims), batch_dims + (min(m, n), n),
                    (1, ) * len(dims))
  return s, u, vt, info


def gesvd_mhlo(dtype, a, full_matrices=True, compute_uv=True):
  """Singular value decomposition."""
  a_type = ir.RankedTensorType(a.type)
  dims = a_type.shape
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  b = _prod(batch_dims)
  if ir.ComplexType.isinstance(a_type.element_type):
    singular_vals_type = ir.ComplexType(a_type.element_type).element_type
  else:
    singular_vals_type = a_type.element_type

  scalar_layout = ir.DenseIntElementsAttr.get(
    np.array(tuple(range(num_bd - 1, -1, -1))),
    type=ir.IndexType.get())
  vector_layout = ir.DenseIntElementsAttr.get(
    np.array((num_bd,) + tuple(range(num_bd - 1, -1, -1))),
    type=ir.IndexType.get())
  matrix_layout = ir.DenseIntElementsAttr.get(
    np.array((num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
    type=ir.IndexType.get())
  i32_type = ir.IntegerType.get_signless(32)

  if m < n:
    lwork, opaque = _hipsolver.build_gesvd_descriptor(
        np.dtype(dtype), b, n, m, compute_uv, full_matrices)
    k = n if full_matrices else m
    out = mhlo.CustomCallOp(
        [ir.TupleType.get_tuple([
          a.type,
          ir.RankedTensorType.get(batch_dims + (min(m, n),), singular_vals_type),
          ir.RankedTensorType.get(batch_dims + (k, n), a_type.element_type),
          ir.RankedTensorType.get(batch_dims + (m, m), a_type.element_type),
          ir.RankedTensorType.get(batch_dims, i32_type),
          ir.RankedTensorType.get([lwork], a_type.element_type),
        ])],
        [a],
        call_target_name = ir.StringAttr.get("hipsolver_gesvd"),
        has_side_effect=ir.BoolAttr.get(False),
        backend_config=ir.StringAttr.get(opaque),
        api_version=ir.IntegerAttr.get(i32_type, 2),
        called_computations=ir.ArrayAttr.get([]),
        operand_layouts=ir.ArrayAttr.get([matrix_layout]),
        result_layouts=ir.ArrayAttr.get([
            matrix_layout,
            vector_layout,
            matrix_layout,
            matrix_layout,
            scalar_layout,
            ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
        ]))
    s = mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, 1)).result
    vt = mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, 2)).result
    u = mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, 3)).result
    info = mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, 4)).result
  else:
    lwork, opaque = _hipsolver.build_gesvd_descriptor(
        np.dtype(dtype), b, m, n, compute_uv, full_matrices)
    k = m if full_matrices else n
    out = mhlo.CustomCallOp(
        [ir.TupleType.get_tuple([
          a.type,
          ir.RankedTensorType.get(batch_dims + (min(m, n),), singular_vals_type),
          ir.RankedTensorType.get(batch_dims + (m, k), a_type.element_type),
          ir.RankedTensorType.get(batch_dims + (n, n), a_type.element_type),
          ir.RankedTensorType.get(batch_dims, i32_type),
          ir.RankedTensorType.get([lwork], a_type.element_type),
        ])],
        [a],
        call_target_name = ir.StringAttr.get("hipsolver_gesvd"),
        has_side_effect=ir.BoolAttr.get(False),
        backend_config=ir.StringAttr.get(opaque),
        api_version=ir.IntegerAttr.get(i32_type, 2),
        called_computations=ir.ArrayAttr.get([]),
        operand_layouts=ir.ArrayAttr.get([matrix_layout]),
        result_layouts=ir.ArrayAttr.get([
            matrix_layout,
            vector_layout,
            matrix_layout,
            matrix_layout,
            scalar_layout,
            ir.DenseIntElementsAttr.get(np.array([0]), type=ir.IndexType.get()),
        ]))
    s = mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, 1)).result
    u = mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, 2)).result
    vt = mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, 3)).result
    info = mhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, 4)).result
  return s, u, vt, info
