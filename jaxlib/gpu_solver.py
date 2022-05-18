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


import functools
from functools import partial
import operator

import jaxlib.mlir.ir as ir
import jaxlib.mlir.dialects.mhlo as mhlo

import numpy as np

from jaxlib import xla_client

from .mhlo_helpers import custom_call

try:
  from .cuda import _cublas
  for _name, _value in _cublas.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="CUDA")
except ImportError:
  _cublas = None

try:
  from .cuda import _cusolver
  for _name, _value in _cusolver.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="CUDA")
except ImportError:
  _cusolver = None


try:
  from .rocm import _hipblas
  for _name, _value in _hipblas.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="ROCM")
except ImportError:
  _hipblas = None

try:
  from .rocm import _hipsolver
  for _name, _value in _hipsolver.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="ROCM")
except ImportError:
  _hipsolver = None


def _real_type(dtype):
  """Returns the real equivalent of 'dtype'."""
  return np.finfo(dtype).dtype

_prod = lambda xs: functools.reduce(operator.mul, xs, 1)


def _trsm_mhlo(platform, gpu_blas, dtype, a, b, left_side=False, lower=False,
               trans_a=False, conj_a=False, diag=False):
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

  lwork, opaque = gpu_blas.build_trsm_batched_descriptor(
    np.dtype(dtype), batch, m, n, left_side, lower, trans_a, conj_a, diag)
  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  work_type = ir.RankedTensorType.get([lwork], ir.IntegerType.get_signless(8))
  work_layout = [0]
  out = custom_call(
      f"{platform}blas_trsm_batched",
      [b_type, work_type, work_type],
      [a, b],
      backend_config=opaque,
      operand_layouts=[layout] * 2,
      result_layouts=[layout, work_layout, work_layout])
  return out[0]

cuda_trsm = partial(_trsm_mhlo, "cu", _cublas)
rocm_trsm = partial(_trsm_mhlo, "hip", _hipblas)


def _potrf_mhlo(platform, gpu_solver, dtype, a, lower):
  """Cholesky decomposition."""
  a_type = ir.RankedTensorType(a.type)
  dims = a_type.shape
  m, n = dims[-2:]
  assert m == n
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  batch = _prod(batch_dims)

  lwork, opaque = gpu_solver.build_potrf_descriptor(
      np.dtype(dtype), lower, batch, n)

  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  info_layout = tuple(range(num_bd - 1, -1, -1))
  i32_type = ir.IntegerType.get_signless(32)
  info_type = ir.RankedTensorType.get(batch_dims, i32_type)
  work_layout = [0]
  out = custom_call(
      f"{platform}solver_potrf",
      [
        a.type,
        info_type,
        ir.RankedTensorType.get([lwork], ir.IntegerType.get_signless(8)),
      ],
      [a],
      backend_config=opaque,
      operand_layouts=[layout],
      result_layouts=[layout, info_layout, work_layout])
  return out[:2]

cuda_potrf = partial(_potrf_mhlo, "cu", _cusolver)
rocm_potrf = partial(_potrf_mhlo, "hip", _hipsolver)


def _getrf_mhlo(platform, gpu_blas, gpu_solver, dtype, a):
  """LU decomposition."""
  a_type = ir.RankedTensorType(a.type)
  dims = a_type.shape
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  batch = _prod(batch_dims)

  if batch > 1 and m == n and m // batch <= 128:
    lwork, opaque = gpu_blas.build_getrf_batched_descriptor(
      np.dtype(dtype), batch, m)
    workspace = ir.RankedTensorType.get([lwork], ir.IntegerType.get_signless(8))
    kernel = f"{platform}blas_getrf_batched"
  else:
    lwork, opaque = gpu_solver.build_getrf_descriptor(
        np.dtype(dtype), batch, m, n)
    workspace = ir.RankedTensorType.get([lwork], a_type.element_type)
    kernel = f"{platform}solver_getrf"

  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  i32_type = ir.IntegerType.get_signless(32)
  out = custom_call(
      kernel,
      [
        a.type,
        ir.RankedTensorType.get(batch_dims + (min(m, n),), i32_type),
        ir.RankedTensorType.get(batch_dims, i32_type),
        workspace,
      ],
      [a],
      backend_config=opaque,
      operand_layouts=[layout],
      result_layouts=[
        layout,
        tuple(range(num_bd, -1, -1)),
        tuple(range(num_bd - 1, -1, -1)),
        [0],
      ])
  return out[:3]

cuda_getrf = partial(_getrf_mhlo, "cu", _cublas, _cusolver)
rocm_getrf = partial(_getrf_mhlo, "hip", _hipblas, _hipsolver)


def _geqrf_mhlo(platform, gpu_solver, dtype, a):
  """QR decomposition."""
  a_type = ir.RankedTensorType(a.type)
  dims = a_type.shape
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  batch = _prod(batch_dims)

  lwork, opaque = gpu_solver.build_geqrf_descriptor(
      np.dtype(dtype), batch, m, n)

  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  i32_type = ir.IntegerType.get_signless(32)
  out = custom_call(
      f"{platform}solver_geqrf",
      [
        a.type,
        ir.RankedTensorType.get(batch_dims + (min(m, n),), a_type.element_type),
        ir.RankedTensorType.get(batch_dims, i32_type),
        ir.RankedTensorType.get([lwork], a_type.element_type),
      ],
      [a],
      backend_config=opaque,
      operand_layouts=[layout],
      result_layouts=[
        layout,
        tuple(range(num_bd, -1, -1)),
        tuple(range(num_bd - 1, -1, -1)),
        [0],
      ])
  return out[:3]

cuda_geqrf = partial(_geqrf_mhlo, "cu", _cusolver)
rocm_geqrf = partial(_geqrf_mhlo, "hip", _hipsolver)

def _geqrf_batched_mhlo(platform, gpu_blas, dtype, a):
  """Batched QR decomposition."""
  a_type = ir.RankedTensorType(a.type)
  dims = a_type.shape
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  batch = _prod(batch_dims)

  lwork, opaque = gpu_blas.build_geqrf_batched_descriptor(
      np.dtype(dtype), batch, m, n)

  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  out = custom_call(
      f"{platform}blas_geqrf_batched",
      [
        a.type,
        ir.RankedTensorType.get(batch_dims + (min(m, n),), a_type.element_type),
        ir.RankedTensorType.get([lwork], ir.IntegerType.get_signless(8)),
        ir.RankedTensorType.get([lwork], ir.IntegerType.get_signless(8)),
      ],
      [a],
      backend_config=opaque,
      operand_layouts=[layout],
      result_layouts=[
        layout,
        tuple(range(num_bd, -1, -1)),
        [0],
        [0],
      ])
  return out[:2]

cuda_geqrf_batched = partial(_geqrf_batched_mhlo, "cu", _cublas)
rocm_geqrf_batched = partial(_geqrf_batched_mhlo, "hip", _hipblas)


def _orgqr_mhlo(platform, gpu_solver, dtype, a, tau):
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

  lwork, opaque = gpu_solver.build_orgqr_descriptor(
      np.dtype(dtype), batch, m, n, k)

  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  i32_type = ir.IntegerType.get_signless(32)
  out = custom_call(
      f"{platform}solver_orgqr",
      [
        a.type,
        ir.RankedTensorType.get(batch_dims, i32_type),
        ir.RankedTensorType.get([lwork], a_type.element_type),
      ],
      [a, tau],
      backend_config=opaque,
      operand_layouts=[
          layout,
          tuple(range(num_bd, -1, -1)),
      ],
      result_layouts=[
        layout,
        tuple(range(num_bd - 1, -1, -1)),
        [0],
      ])
  return out[:2]

cuda_orgqr = partial(_orgqr_mhlo, "cu", _cusolver)
rocm_orgqr = partial(_orgqr_mhlo, "hip", _hipsolver)


def _syevd_mhlo(platform, gpu_solver, have_jacobi_solver, dtype, a,
                lower=False):
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

  if have_jacobi_solver and n <= 32:
    kernel = f"{platform}solver_syevj"
    lwork, opaque = gpu_solver.build_syevj_descriptor(
        np.dtype(dtype), lower, batch, n)
  else:
    kernel = f"{platform}solver_syevd"
    lwork, opaque = gpu_solver.build_syevd_descriptor(
        np.dtype(dtype), lower, batch, n)

  if ir.ComplexType.isinstance(a_type.element_type):
    eigvals_type = ir.ComplexType(a_type.element_type).element_type
  else:
    eigvals_type = a_type.element_type

  i32_type = ir.IntegerType.get_signless(32)
  out = custom_call(
      kernel,
      [
          a.type,
          ir.RankedTensorType.get(batch_dims + (n,), eigvals_type),
          ir.RankedTensorType.get(batch_dims, i32_type),
          ir.RankedTensorType.get([lwork], a_type.element_type),
      ],
      [a],
      backend_config=opaque,
      operand_layouts=[layout],
      result_layouts=[
          layout,
          tuple(range(num_bd, -1, -1)),
          tuple(range(num_bd - 1, -1, -1)),
          [0],
      ])
  return out[:3]

cuda_syevd = partial(_syevd_mhlo, "cu", _cusolver, True)
rocm_syevd = partial(_syevd_mhlo, "hip", _hipsolver, False)


def _gesvd_mhlo(platform, gpu_solver, have_jacobi_solver, dtype, a,
                full_matrices=True, compute_uv=True):
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

  scalar_layout = tuple(range(num_bd - 1, -1, -1))
  vector_layout = (num_bd,) + tuple(range(num_bd - 1, -1, -1))
  i32_type = ir.IntegerType.get_signless(32)

  if have_jacobi_solver and m < 32 and n < 32:
    # The batched kernel doesn't support "econ" mode.
    econ = not full_matrices and b == 1
    lwork, opaque = gpu_solver.build_gesvdj_descriptor(
        np.dtype(dtype), b, m, n, compute_uv, 1 if econ else 0)
    k = min(m, n)
    matrix_layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
    _, s, u, v, info, _ = custom_call(
        f"{platform}solver_gesvdj",
        [
          a.type,
          ir.RankedTensorType.get(batch_dims + (min(m, n),), singular_vals_type),
          ir.RankedTensorType.get(batch_dims + (m, k if econ else m),
                                  a_type.element_type),
          ir.RankedTensorType.get(batch_dims + (n, k if econ else n),
                                  a_type.element_type),
          ir.RankedTensorType.get(batch_dims, i32_type),
          ir.RankedTensorType.get([lwork], a_type.element_type),
        ],
        [a],
        backend_config=opaque,
        operand_layouts=[matrix_layout],
        result_layouts=[
            matrix_layout,
            vector_layout,
            matrix_layout,
            matrix_layout,
            scalar_layout,
            [0],
        ])
    vt = mhlo.TransposeOp(
        v,
        ir.DenseIntElementsAttr.get(np.array(tuple(range(num_bd)) + (num_bd + 1, num_bd)))).result
    if np.issubdtype(dtype, np.complexfloating):
      vt = mhlo.ComplexOp(mhlo.RealOp(vt), mhlo.NegOp(mhlo.ImagOp(vt))).result
    if not full_matrices and not econ:
      u = mhlo.SliceOp(
          u,
          ir.DenseIntElementsAttr.get(np.zeros([len(dims)], np.int64)),
          ir.DenseIntElementsAttr.get(np.array(batch_dims + (m, min(m, n)))),
          ir.DenseIntElementsAttr.get(np.ones([len(dims)], np.int64))).result
      vt = mhlo.SliceOp(
          vt,
          ir.DenseIntElementsAttr.get(np.zeros([len(dims)], np.int64)),
          ir.DenseIntElementsAttr.get(np.array(batch_dims + (min(m, n), n))),
          ir.DenseIntElementsAttr.get(np.ones([len(dims)], np.int64))).result
  elif m < n:
    lwork, opaque = gpu_solver.build_gesvd_descriptor(
        np.dtype(dtype), b, n, m, compute_uv, full_matrices)
    k = n if full_matrices else m
    matrix_layout = (num_bd + 1, num_bd) + tuple(range(num_bd - 1, -1, -1))
    _, s, vt, u, info, _ = custom_call(
        f"{platform}solver_gesvd",
        [
          a.type,
          ir.RankedTensorType.get(batch_dims + (min(m, n),), singular_vals_type),
          ir.RankedTensorType.get(batch_dims + (k, n), a_type.element_type),
          ir.RankedTensorType.get(batch_dims + (m, m), a_type.element_type),
          ir.RankedTensorType.get(batch_dims, i32_type),
          ir.RankedTensorType.get([lwork], a_type.element_type),
        ],
        [a],
        backend_config=opaque,
        operand_layouts=[matrix_layout],
        result_layouts=[
          matrix_layout,
          vector_layout,
          matrix_layout,
          matrix_layout,
          scalar_layout,
          [0],
        ])
  else:
    lwork, opaque = gpu_solver.build_gesvd_descriptor(
        np.dtype(dtype), b, m, n, compute_uv, full_matrices)
    k = m if full_matrices else n
    matrix_layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
    _, s, u, vt, info, _ = custom_call(
        f"{platform}solver_gesvd",
        [
          a.type,
          ir.RankedTensorType.get(batch_dims + (min(m, n),), singular_vals_type),
          ir.RankedTensorType.get(batch_dims + (m, k), a_type.element_type),
          ir.RankedTensorType.get(batch_dims + (n, n), a_type.element_type),
          ir.RankedTensorType.get(batch_dims, i32_type),
          ir.RankedTensorType.get([lwork], a_type.element_type),
        ],
        [a],
        backend_config=opaque,
        operand_layouts=[matrix_layout],
        result_layouts=[
          matrix_layout,
          vector_layout,
          matrix_layout,
          matrix_layout,
          scalar_layout,
          [0],
        ])
  return s, u, vt, info

cuda_gesvd = partial(_gesvd_mhlo, "cu", _cusolver, True)
rocm_gesvd = partial(_gesvd_mhlo, "hip", _hipsolver, False)
