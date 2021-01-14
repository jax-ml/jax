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
import operator

import numpy as np

from jaxlib import xla_client

try:
  from jaxlib import rocblas_kernels
  for _name, _value in rocblas_kernels.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="ROCM")
except ImportError:
  pass

# we have a single module for both rocsolver and rocblas functions
rocsolver_kernels = rocblas_kernels

_ops = xla_client.ops
_Shape = xla_client.Shape


def _real_type(dtype):
  """Returns the real equivalent of 'dtype'."""
  if dtype == np.float32:
    return np.float32
  elif dtype == np.float64:
    return np.float64
  elif dtype == np.complex64:
    return np.float32
  elif dtype == np.complex128:
    return np.float64
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))


_prod = lambda xs: functools.reduce(operator.mul, xs, 1)


def trsm(c,
         a,
         b,
         left_side=False,
         lower=False,
         trans_a=False,
         conj_a=False,
         diag=False):
  """triangular solve"""
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
  if (batch_dims + (k, k) != a_shape.dimensions() or a_shape.element_type() != dtype):
    raise ValueError("Argument mismatch for trsm, got {} and {}".format(
        a_shape, b_shape))

  if conj_a and not trans_a:
    raise NotImplementedError("Conjugation without transposition not supported")

  lwork, opaque = rocblas_kernels.build_trsm_descriptor(np.dtype(dtype), batch, m, n,
                                                        left_side, lower, trans_a,
                                                        conj_a, diag)
  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  out = _ops.CustomCallWithLayout(
      c,
      b"rocblas_trsm",
      operands=(a, b),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(dtype, b_shape.dimensions(),
                             layout),  # buffers[2] (b, OUT)
          _Shape.array_shape(np.dtype(np.int8), (lwork,),
                             (0,)),  # buffers[3] (a batch pointers)
          _Shape.array_shape(np.dtype(np.int8), (lwork,),
                             (0,)))),  # buffers[4] (b batch pointers)
      operand_shapes_with_layout=(
          _Shape.array_shape(dtype, a_shape.dimensions(), layout),  # buffers[0] (a)
          _Shape.array_shape(dtype, b_shape.dimensions(), layout),  # buffers[1] (b, IN)
      ),
      opaque=opaque)
  return _ops.GetTupleElement(out, 0)


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

  lwork, opaque = rocsolver_kernels.build_potrf_descriptor(np.dtype(dtype), lower,
                                                           batch, n)
  kernel = b"rocsolver_potrf"

  out = _ops.CustomCallWithLayout(
      c,
      kernel,
      operands=(a,),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(dtype, batch_dims + (n, n),
                             (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
                             ),  # buffers[1] (a, OUT)
          _Shape.array_shape(np.dtype(np.int32), batch_dims,
                             tuple(range(num_bd - 1, -1, -1))),  # buffers[2] (info)
          _Shape.array_shape(np.dtype(np.int8), (lwork,),
                             (0,)),  # buffers[3] (a batch pointers)
      )),
      operand_shapes_with_layout=(
          _Shape.array_shape(dtype, batch_dims + (n, n),
                             (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
                             ),  # buffers[0] (a, IN)
      ),
      opaque=opaque)
  return _ops.GetTupleElement(out, 0), _ops.GetTupleElement(out, 1)


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

  lwork, opaque = rocsolver_kernels.build_getrf_descriptor(np.dtype(dtype), batch, m, n)
  kernel = b"rocsolver_getrf"

  out = _ops.CustomCallWithLayout(
      c,
      kernel,
      operands=(a,),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(dtype, batch_dims + (m, n),
                             (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
                             ),  # buffers[1] (a, OUT)
          _Shape.array_shape(np.dtype(np.int32), batch_dims + (min(m, n),),
                             tuple(range(num_bd, -1, -1))),  # buffers[2] (ipiv)
          _Shape.array_shape(np.dtype(np.int32), batch_dims,
                             tuple(range(num_bd - 1, -1, -1))),  # buffers[3] (info)
          _Shape.array_shape(np.dtype(np.int8), (lwork,),
                             (0,)),  # buffers[4] (a batch pointers)
      )),
      operand_shapes_with_layout=(
          _Shape.array_shape(dtype, batch_dims + (m, n),
                             (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
                             ),  # buffers[0] (a, IN)
      ),
      opaque=opaque)
  return (_ops.GetTupleElement(out, 0), _ops.GetTupleElement(out, 1),
          _ops.GetTupleElement(out, 2))


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

  lwork, opaque = rocsolver_kernels.build_geqrf_descriptor(np.dtype(dtype), batch, m, n)
  kernel = b"rocsolver_geqrf"

  out = _ops.CustomCallWithLayout(
      c,
      kernel,
      operands=(a,),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(dtype, batch_dims + (m, n),
                             (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
                             ),  # buffers[1] (a, OUT)
          _Shape.array_shape(dtype, batch_dims + (min(m, n),),
                             tuple(range(num_bd, -1, -1))),  # buffers[2] (tau)
          # buffers[3]  (a batch pointers)
          _Shape.array_shape(np.dtype(np.int8), (lwork,), (0,)),
      )),
      operand_shapes_with_layout=(
          _Shape.array_shape(dtype, batch_dims + (m, n),
                             (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
                             ),  # buffers[0] (a, IN)
      ),
      opaque=opaque)
  # rocsolver geqrf does not return info
  return (_ops.GetTupleElement(out, 0), _ops.GetTupleElement(out, 1), None)


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

  _, opaque = rocsolver_kernels.build_orgqr_descriptor(np.dtype(dtype), batch, m, n, k)
  kernel = b"rocsolver_orgqr"

  out = _ops.CustomCallWithLayout(
      c,
      kernel,
      operands=(a, tau),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(dtype, batch_dims + (m, n),
                             (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
                             ),  # buffers[2]  (a OUT)
      )),
      operand_shapes_with_layout=(
          _Shape.array_shape(dtype, batch_dims + (m, n),
                             (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
                             ),  # buffers[0]  (a, IN)
          _Shape.array_shape(dtype, batch_dims + (k,),
                             tuple(range(num_bd, -1, -1))),  # buffers[1]  (tau IN)
      ),
      opaque=opaque)
  return (_ops.GetTupleElement(out, 0), None)  # ROCSolver orgqr does not return info


def syevd(c, a, lower=False):
  raise NotImplementedError(
      "Symmetric (Hermitian) eigendecomposition is not yet implemented in ROCSolver")


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

  # gesvdj is not yet implemented in ROCSolver
  # if m < 32 and n < 32:
  # ...
  # elif m < n:

  if m < n:
    lwork, opaque = rocsolver_kernels.build_gesvd_descriptor(np.dtype(dtype), b, n, m,
                                                             compute_uv, full_matrices)
    scalar_layout = tuple(range(num_bd - 1, -1, -1))
    vector_layout = (num_bd,) + scalar_layout
    matrix_layout = (num_bd + 1, num_bd) + scalar_layout
    out = _ops.CustomCallWithLayout(
        c,
        b"rocsolver_gesvd",
        operands=(a,),
        shape_with_layout=_Shape.tuple_shape((
            _Shape.array_shape(dtype, batch_dims + (m, n),
                               matrix_layout),  # buffers[1] (a, OUT)
            _Shape.array_shape(singular_vals_dtype, batch_dims + (min(m, n),),
                               vector_layout),  # buffers[2] (s)
            # buffers[3] (u; actually vt)
            _Shape.array_shape(dtype, batch_dims + (n, n), matrix_layout),
            # buffers[4] (vt; actually u)
            _Shape.array_shape(dtype, batch_dims + (m, m), matrix_layout),
            _Shape.array_shape(singular_vals_dtype, (min(m, n) - 1,),
                               (0,)),  # buffers[5] (e)
            _Shape.array_shape(np.dtype(np.int32), batch_dims,
                               scalar_layout),  # buffers[6] (info)
            _Shape.array_shape(np.dtype(np.int8), (lwork,),
                               (0,)),  # buffers[7] (a batch pointers)
        )),
        operand_shapes_with_layout=(
            _Shape.array_shape(dtype, batch_dims + (m, n),
                               matrix_layout),  # a (buffers[0]))
        ),
        opaque=opaque)
    s = _ops.GetTupleElement(out, 1)
    vt = _ops.GetTupleElement(out, 2)
    u = _ops.GetTupleElement(out, 3)
    info = _ops.GetTupleElement(out, 5)
  else:
    lwork, opaque = rocsolver_kernels.build_gesvd_descriptor(np.dtype(dtype), b, m, n,
                                                             compute_uv, full_matrices)
    scalar_layout = tuple(range(num_bd - 1, -1, -1))
    vector_layout = (num_bd,) + scalar_layout
    matrix_layout = (num_bd, num_bd + 1) + scalar_layout
    out = _ops.CustomCallWithLayout(
        c,
        b"rocsolver_gesvd",
        operands=(a,),
        shape_with_layout=_Shape.tuple_shape((
            _Shape.array_shape(dtype, batch_dims + (m, n),
                               matrix_layout),  # buffers[1] (a, OUT)
            _Shape.array_shape(singular_vals_dtype, batch_dims + (min(m, n),),
                               vector_layout),  # buffers[2] (s)
            _Shape.array_shape(dtype, batch_dims + (m, m),
                               matrix_layout),  # buffers[3] (u)
            _Shape.array_shape(dtype, batch_dims + (n, n),
                               matrix_layout),  # buffers[4] (vt)
            _Shape.array_shape(singular_vals_dtype, (min(m, n) - 1,),
                               (0,)),  # buffers[5] (e)
            _Shape.array_shape(np.dtype(np.int32), batch_dims,
                               scalar_layout),  # buffers[6] (info)
            _Shape.array_shape(np.dtype(np.int8), (lwork,),
                               (0,)),  # buffers[7] (a batch pointers)
        )),
        operand_shapes_with_layout=(
            _Shape.array_shape(dtype, batch_dims + (m, n),
                               matrix_layout),  # buffers[0] (a, IN)
        ),
        opaque=opaque)
    s = _ops.GetTupleElement(out, 1)
    u = _ops.GetTupleElement(out, 2)
    vt = _ops.GetTupleElement(out, 3)
    info = _ops.GetTupleElement(out, 5)
  if not full_matrices:
    u = _ops.Slice(u, (0,) * len(dims), batch_dims + (m, min(m, n)), (1,) * len(dims))
    vt = _ops.Slice(vt, (0,) * len(dims), batch_dims + (min(m, n), n), (1,) * len(dims))
  return s, u, vt, info
