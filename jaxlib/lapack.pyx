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

# distutils: language = c++

# Shims that allow the XLA CPU backend to call scipy-provided LAPACK kernels
# via CustomCall.

from __future__ import print_function

from libc.stdint cimport int32_t
from libc.string cimport memcpy
from libcpp.string cimport string
from cpython.pycapsule cimport PyCapsule_New

from scipy.linalg.cython_blas cimport strsm, dtrsm
from scipy.linalg.cython_lapack cimport sgetrf, dgetrf, cgetrf, spotrf, dpotrf

import numpy as np
from jaxlib import xla_client

Shape = xla_client.Shape


cdef register_cpu_custom_call_target(fn_name, void* fn):
  cdef const char* name = "xla._CPU_CUSTOM_CALL_TARGET"
  xla_client.register_cpu_custom_call_target(
    fn_name, PyCapsule_New(fn, name, NULL))

# TODO(phawkins): it would be nice to avoid duplicating code for each type.

# ?trsm(left_side, lower, trans_a, diag, m, n, alpha, a, b):
# triangular solve

cdef void blas_strsm(void* out, void** data) nogil:
  cdef int32_t left_side = (<int32_t*>(data[0]))[0]
  cdef int32_t lower = (<int32_t*>(data[1]))[0]
  cdef int32_t trans_a = (<int32_t*>(data[2]))[0]
  cdef int32_t diag = (<int32_t*>(data[3]))[0]
  cdef int m = (<int32_t*>(data[4]))[0]
  cdef int n = (<int32_t*>(data[5]))[0]
  cdef float* alpha = <float*>(data[6])
  cdef float* a = <float*>(data[7])
  cdef float* b = <float*>(data[8])

  cdef float* x = <float*>(out)
  if x != b:
    memcpy(x, b, m * n * sizeof(float))

  cdef char cside = 'L' if left_side else 'R'
  cdef char cuplo = 'L' if lower else 'U'
  cdef char ctransa = 'N'
  if trans_a == 1:
    ctransa = 'T'
  elif trans_a == 2:
    ctransa = 'C'
  cdef char cdiag = 'U' if diag else 'N'
  cdef int lda = m
  cdef int ldb = m if left_side else n
  strsm(&cside, &cuplo, &ctransa, &cdiag, &m, &n, alpha, a, &lda, x, &ldb)

register_cpu_custom_call_target(b"blas_strsm", <void*>(blas_strsm))

cdef void blas_dtrsm(void* out, void** data) nogil:
  cdef int32_t left_side = (<int32_t*>(data[0]))[0]
  cdef int32_t lower = (<int32_t*>(data[1]))[0]
  cdef int32_t trans_a = (<int32_t*>(data[2]))[0]
  cdef int32_t diag = (<int32_t*>(data[3]))[0]
  cdef int m = (<int32_t*>(data[4]))[0]
  cdef int n = (<int32_t*>(data[5]))[0]
  cdef double* alpha = <double*>(data[6])
  cdef double* a = <double*>(data[7])
  cdef double* b = <double*>(data[8])

  cdef double* x = <double*>(out)
  if x != b:
    memcpy(x, b, m * n * sizeof(double))

  cdef char cside = 'L' if left_side else 'R'
  cdef char cuplo = 'L' if lower else 'U'
  cdef char ctransa = 'N'
  if trans_a == 1:
    ctransa = 'T'
  elif trans_a == 2:
    ctransa = 'C'
  cdef char cdiag = 'U' if diag else 'N'
  cdef int lda = m
  cdef int ldb = m if left_side else n
  dtrsm(&cside, &cuplo, &ctransa, &cdiag, &m, &n, alpha, a, &lda, x, &ldb)

register_cpu_custom_call_target(b"blas_dtrsm", <void*>(blas_dtrsm))


def jax_trsm(c, alpha, a, b, left_side=False, lower=False, trans_a=False,
             conj_a=False, diag=False):
  b_shape = c.GetShape(b)
  dtype = b_shape.element_type()
  #if left_side:
  m, n = b_shape.dimensions()
  #else:
  #  n, m = b_shape.dimensions()

  a_shape = c.GetShape(a)
  if (m, m) != a_shape.dimensions() or a_shape.element_type() != dtype:
    raise ValueError("Argument mismatch for trsm, got {} and {}".format(
      a_shape, b_shape))

  if dtype == np.float32:
    fn = b"blas_strsm"
  elif dtype == np.float64: 
    fn = b"blas_dtrsm"
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  return c.CustomCall(
      fn,
      operands=(
        c.ConstantS32Scalar(int(left_side)),
        c.ConstantS32Scalar(int(lower)),
        c.ConstantS32Scalar(1 if trans_a else 0),
        c.ConstantS32Scalar(int(diag)),
        c.ConstantS32Scalar(m),
        c.ConstantS32Scalar(n),
        alpha, a, b),
      shape_with_layout=Shape.array_shape(dtype, b_shape.dimensions(), (0, 1)),
      operand_shapes_with_layout=(
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(dtype, (), ()),
          Shape.array_shape(dtype, a_shape.dimensions(), (0, 1)),
          Shape.array_shape(dtype, b_shape.dimensions(), (0, 1)),
      ))


# ?getrf: LU decomposition

cdef void lapack_sgetrf(void* out_tuple, void** data) nogil:
  cdef int m = (<int32_t*>(data[0]))[0]
  cdef int n = (<int32_t*>(data[1]))[0]
  cdef const float* a_in = <float*>(data[2])

  cdef void** out = <void**>(out_tuple)
  cdef float* a_out = <float*>(out[0])
  cdef int* ipiv = <int*>(out[1])
  cdef int* info = <int*>(out[2])
  if a_out != a_in:
    memcpy(a_out, a_in, m * n * sizeof(float))

  sgetrf(&m, &n, a_out, &m, ipiv, info)

register_cpu_custom_call_target(b"lapack_sgetrf", <void*>(lapack_sgetrf))


cdef void lapack_dgetrf(void* out_tuple, void** data) nogil:
  cdef int m = (<int32_t*>(data[0]))[0]
  cdef int n = (<int32_t*>(data[1]))[0]
  cdef const double* a_in = <double*>(data[2])

  cdef void** out = <void**>(out_tuple)
  cdef double* a_out = <double*>(out[0])
  cdef int* ipiv = <int*>(out[1])
  cdef int* info = <int*>(out[2])
  if a_out != a_in:
    memcpy(a_out, a_in, m * n * sizeof(double))

  dgetrf(&m, &n, a_out, &m, ipiv, info)

register_cpu_custom_call_target(b"lapack_dgetrf", <void*>(lapack_dgetrf))


cdef void lapack_cgetrf(void* out_tuple, void** data) nogil:
  cdef int m = (<int32_t*>(data[0]))[0]
  cdef int n = (<int32_t*>(data[1]))[0]
  cdef const float complex* a_in = <float complex*>(data[2])

  cdef void** out = <void**>(out_tuple)
  cdef float complex* a_out = <float complex*>(out[0])
  cdef int* ipiv = <int*>(out[1])
  cdef int* info = <int*>(out[2])
  if a_out != a_in:
    memcpy(a_out, a_in, m * n * sizeof(float complex))

  cgetrf(&m, &n, a_out, &m, ipiv, info)

register_cpu_custom_call_target(b"lapack_cgetrf", <void*>(lapack_cgetrf))


def jax_getrf(c, a):
  assert sizeof(int32_t) == sizeof(int)

  a_shape = c.GetShape(a)
  dtype = a_shape.element_type()
  m, n = a_shape.dimensions()
  if dtype == np.float32:
    fn = b"lapack_sgetrf"
  elif dtype == np.float64:
    fn = b"lapack_dgetrf"
  elif dtype == np.complex64:
    fn = b"lapack_cgetrf"
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  return c.CustomCall(
      fn,
      operands=(c.ConstantS32Scalar(m), c.ConstantS32Scalar(n), a),
      shape_with_layout=Shape.tuple_shape((
          Shape.array_shape(dtype, (m, n), (0, 1)),
          Shape.array_shape(np.int32, (min(m, n),), (0,)),
          Shape.array_shape(np.int32, (), ()),
      )),
      operand_shapes_with_layout=(
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(dtype, (m, n), (0, 1)),
      ))



# ?potrf: Cholesky decomposition

cdef void lapack_spotrf(void* out_tuple, void** data) nogil:
  cdef int32_t lower = (<int32_t*>(data[0]))[0]
  cdef int n = (<int32_t*>(data[1]))[0]
  cdef const float* a_in = <float*>(data[2])
  cdef char uplo = 'L' if lower else 'U'

  cdef void** out = <void**>(out_tuple)
  cdef float* a_out = <float*>(out[0])
  cdef int* info = <int*>(out[1])
  if a_out != a_in:
    memcpy(a_out, a_in, n * n * sizeof(float))

  spotrf(&uplo, &n, a_out, &n, info)

  # spotrf leaves junk in the part of the triangle that is not written; zero it.
  cdef int i
  cdef int j
  if lower:
    for i in range(n):
      for j in range(i):
        a_out[i * n + j] = 0
  else:
    for i in range(n):
      for j in range(i, n):
        a_out[i * n + j] = 0

register_cpu_custom_call_target(b"lapack_spotrf", <void*>(lapack_spotrf))


cdef void lapack_dpotrf(void* out_tuple, void** data) nogil:
  cdef int32_t lower = (<int32_t*>(data[0]))[0]
  cdef int n = (<int32_t*>(data[1]))[0]
  cdef const double* a_in = <double*>(data[2])
  cdef char uplo = 'L' if lower else 'U'

  cdef void** out = <void**>(out_tuple)
  cdef double* a_out = <double*>(out[0])
  cdef int* info = <int*>(out[1])
  if a_out != a_in:
    memcpy(a_out, a_in, n * n * sizeof(double))

  dpotrf(&uplo, &n, a_out, &n, info)

  # dpotrf leaves junk in the part of the triangle that is not written; zero it.
  cdef int i
  cdef int j
  if lower:
    for i in range(n):
      for j in range(i):
        a_out[i * n + j] = 0
  else:
    for i in range(n):
      for j in range(i, n):
        a_out[i * n + j] = 0

register_cpu_custom_call_target(b"lapack_dpotrf", <void*>(lapack_dpotrf))

def jax_potrf(c, a, lower=False):
  assert sizeof(int32_t) == sizeof(int)

  a_shape = c.GetShape(a)
  dtype = a_shape.element_type()
  m, n = a_shape.dimensions()
  if m != n:
    raise ValueError("potrf expects a square matrix, got {}".format(a_shape))
  if dtype == np.float32:
    fn = b"lapack_spotrf"
  elif dtype == np.float64: 
    fn = b"lapack_dpotrf"
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  return c.CustomCall(
      fn,
      operands=(c.ConstantS32Scalar(int(lower)), c.ConstantS32Scalar(n), a),
      shape_with_layout=Shape.tuple_shape((
          Shape.array_shape(dtype, (n, n), (0, 1)),
          Shape.array_shape(np.int32, (), ()),
      )),
      operand_shapes_with_layout=(
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(dtype, (n, n), (0, 1)),
      ))
