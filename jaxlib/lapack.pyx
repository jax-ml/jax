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
#
# cython: language_level=2
# distutils: language = c++

# Shims that allow the XLA CPU backend to call scipy-provided LAPACK kernels
# via CustomCallWithLayout.

from __future__ import print_function

cdef extern from "<cmath>" namespace "std":
  bint isnan(float x) nogil
  bint isnan(double x) nogil

from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t, int64_t
from libc.string cimport memcpy
from libcpp cimport bool as bool_t
from libcpp.string cimport string
from cpython.pycapsule cimport PyCapsule_New

from scipy.linalg.cython_blas cimport strsm, dtrsm, ctrsm, ztrsm
from scipy.linalg.cython_lapack cimport sgetrf, dgetrf, cgetrf, zgetrf
from scipy.linalg.cython_lapack cimport sgeqrf, dgeqrf, cgeqrf, zgeqrf
from scipy.linalg.cython_lapack cimport sorgqr, dorgqr, cungqr, zungqr
from scipy.linalg.cython_lapack cimport spotrf, dpotrf, cpotrf, zpotrf
from scipy.linalg.cython_lapack cimport sgesdd, dgesdd, cgesdd, zgesdd
from scipy.linalg.cython_lapack cimport ssyevd, dsyevd, cheevd, zheevd
from scipy.linalg.cython_lapack cimport sgeev, dgeev, cgeev, zgeev

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
  cdef int batch = (<int32_t*>(data[6]))[0]
  cdef float* alpha = <float*>(data[7])
  cdef float* a = <float*>(data[8])
  cdef float* b = <float*>(data[9])

  cdef float* x = <float*>(out)
  if x != b:
    memcpy(x, b, <int64_t>(batch) * <int64_t>(m) * <int64_t>(n) * sizeof(float))

  cdef char cside = 'L' if left_side else 'R'
  cdef char cuplo = 'L' if lower else 'U'
  cdef char ctransa = 'N'
  if trans_a == 1:
    ctransa = 'T'
  elif trans_a == 2:
    ctransa = 'C'
  cdef char cdiag = 'U' if diag else 'N'
  cdef int lda = m if left_side else n
  cdef int ldb = m

  cdef int64_t x_plus = <int64_t>(m) * <int64_t>(n)
  cdef int64_t a_plus = <int64_t>(lda) * <int64_t>(lda)

  for _ in range(batch):
    strsm(&cside, &cuplo, &ctransa, &cdiag, &m, &n, alpha, a, &lda, x, &ldb)
    x += x_plus
    a += a_plus

register_cpu_custom_call_target(b"blas_strsm", <void*>(blas_strsm))

cdef void blas_dtrsm(void* out, void** data) nogil:
  cdef int32_t left_side = (<int32_t*>(data[0]))[0]
  cdef int32_t lower = (<int32_t*>(data[1]))[0]
  cdef int32_t trans_a = (<int32_t*>(data[2]))[0]
  cdef int32_t diag = (<int32_t*>(data[3]))[0]
  cdef int m = (<int32_t*>(data[4]))[0]
  cdef int n = (<int32_t*>(data[5]))[0]
  cdef int batch = (<int32_t*>(data[6]))[0]
  cdef double* alpha = <double*>(data[7])
  cdef double* a = <double*>(data[8])
  cdef double* b = <double*>(data[9])

  cdef double* x = <double*>(out)
  if x != b:
    memcpy(x, b, <int64_t>(batch) * <int64_t>(m) * <int64_t>(n) * sizeof(double))

  cdef char cside = 'L' if left_side else 'R'
  cdef char cuplo = 'L' if lower else 'U'
  cdef char ctransa = 'N'
  if trans_a == 1:
    ctransa = 'T'
  elif trans_a == 2:
    ctransa = 'C'
  cdef char cdiag = 'U' if diag else 'N'
  cdef int lda = m if left_side else n
  cdef int ldb = m

  cdef int64_t x_plus = <int64_t>(m) * <int64_t>(n)
  cdef int64_t a_plus = <int64_t>(lda) * <int64_t>(lda)

  for _ in range(batch):
    dtrsm(&cside, &cuplo, &ctransa, &cdiag, &m, &n, alpha, a, &lda, x, &ldb)
    x += x_plus
    a += a_plus


register_cpu_custom_call_target(b"blas_dtrsm", <void*>(blas_dtrsm))


cdef void blas_ctrsm(void* out, void** data) nogil:
  cdef int32_t left_side = (<int32_t*>(data[0]))[0]
  cdef int32_t lower = (<int32_t*>(data[1]))[0]
  cdef int32_t trans_a = (<int32_t*>(data[2]))[0]
  cdef int32_t diag = (<int32_t*>(data[3]))[0]
  cdef int m = (<int32_t*>(data[4]))[0]
  cdef int n = (<int32_t*>(data[5]))[0]
  cdef int batch = (<int32_t*>(data[6]))[0]
  cdef float complex* alpha = <float complex*>(data[7])
  cdef float complex* a = <float complex*>(data[8])
  cdef float complex* b = <float complex*>(data[9])

  cdef float complex* x = <float complex*>(out)
  if x != b:
    memcpy(x, b, <int64_t>(batch) * <int64_t>(m) * <int64_t>(n) * sizeof(float complex))

  cdef char cside = 'L' if left_side else 'R'
  cdef char cuplo = 'L' if lower else 'U'
  cdef char ctransa = 'N'
  if trans_a == 1:
    ctransa = 'T'
  elif trans_a == 2:
    ctransa = 'C'
  cdef char cdiag = 'U' if diag else 'N'
  cdef int lda = m if left_side else n
  cdef int ldb = m

  cdef int64_t x_plus = <int64_t>(m) * <int64_t>(n)
  cdef int64_t a_plus = <int64_t>(lda) * <int64_t>(lda)

  for _ in range(batch):
    ctrsm(&cside, &cuplo, &ctransa, &cdiag, &m, &n, alpha, a, &lda, x, &ldb)
    x += x_plus
    a += a_plus


register_cpu_custom_call_target(b"blas_ctrsm", <void*>(blas_ctrsm))

cdef void blas_ztrsm(void* out, void** data) nogil:
  cdef int32_t left_side = (<int32_t*>(data[0]))[0]
  cdef int32_t lower = (<int32_t*>(data[1]))[0]
  cdef int32_t trans_a = (<int32_t*>(data[2]))[0]
  cdef int32_t diag = (<int32_t*>(data[3]))[0]
  cdef int m = (<int32_t*>(data[4]))[0]
  cdef int n = (<int32_t*>(data[5]))[0]
  cdef int batch = (<int32_t*>(data[6]))[0]
  cdef double complex* alpha = <double complex*>(data[7])
  cdef double complex* a = <double complex*>(data[8])
  cdef double complex* b = <double complex*>(data[9])

  cdef double complex* x = <double complex*>(out)
  if x != b:
    memcpy(x, b, <int64_t>(batch) * <int64_t>(m) * <int64_t>(n) * sizeof(double complex))

  cdef char cside = 'L' if left_side else 'R'
  cdef char cuplo = 'L' if lower else 'U'
  cdef char ctransa = 'N'
  if trans_a == 1:
    ctransa = 'T'
  elif trans_a == 2:
    ctransa = 'C'
  cdef char cdiag = 'U' if diag else 'N'
  cdef int lda = m if left_side else n
  cdef int ldb = m

  cdef int64_t x_plus = <int64_t>(m) * <int64_t>(n)
  cdef int64_t a_plus = <int64_t>(lda) * <int64_t>(lda)

  for _ in range(batch):
    ztrsm(&cside, &cuplo, &ctransa, &cdiag, &m, &n, alpha, a, &lda, x, &ldb)
    x += x_plus
    a += a_plus

register_cpu_custom_call_target(b"blas_ztrsm", <void*>(blas_ztrsm))


def trsm(c, alpha, a, b, left_side=False, lower=False, trans_a=False,
             conj_a=False, diag=False):
  a_shape = c.GetShape(a)
  b_shape = c.GetShape(b)
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
  return c.CustomCallWithLayout(
      fn,
      operands=(
        c.ConstantS32Scalar(int(left_side)),
        c.ConstantS32Scalar(int(lower)),
        c.ConstantS32Scalar((2 if conj_a else 1) if trans_a else 0),
        c.ConstantS32Scalar(int(diag)),
        c.ConstantS32Scalar(m),
        c.ConstantS32Scalar(n),
        c.ConstantS32Scalar(num_b),
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
      ))
jax_trsm = trsm

# ?getrf: LU decomposition

cdef void lapack_sgetrf(void* out_tuple, void** data) nogil:
  cdef int b = (<int32_t*>(data[0]))[0]
  cdef int m = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef const float* a_in = <float*>(data[3])

  cdef void** out = <void**>(out_tuple)
  cdef float* a_out = <float*>(out[0])
  cdef int* ipiv = <int*>(out[1])
  cdef int* info = <int*>(out[2])
  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(m) * <int64_t>(n) * sizeof(float))

  for i in range(b):
    sgetrf(&m, &n, a_out, &m, ipiv, info)
    a_out += m * n
    ipiv += min(m, n)
    info += 1

register_cpu_custom_call_target(b"lapack_sgetrf", <void*>(lapack_sgetrf))


cdef void lapack_dgetrf(void* out_tuple, void** data) nogil:
  cdef int b = (<int32_t*>(data[0]))[0]
  cdef int m = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef const double* a_in = <double*>(data[3])

  cdef void** out = <void**>(out_tuple)
  cdef double* a_out = <double*>(out[0])
  cdef int* ipiv = <int*>(out[1])
  cdef int* info = <int*>(out[2])
  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(m) * <int64_t>(n) * sizeof(double))

  for i in range(b):
    dgetrf(&m, &n, a_out, &m, ipiv, info)
    a_out += m * n
    ipiv += min(m, n)
    info += 1

register_cpu_custom_call_target(b"lapack_dgetrf", <void*>(lapack_dgetrf))


cdef void lapack_cgetrf(void* out_tuple, void** data) nogil:
  cdef int b = (<int32_t*>(data[0]))[0]
  cdef int m = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef const float complex* a_in = <float complex*>(data[3])

  cdef void** out = <void**>(out_tuple)
  cdef float complex* a_out = <float complex*>(out[0])
  cdef int* ipiv = <int*>(out[1])
  cdef int* info = <int*>(out[2])
  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(m) * <int64_t>(n) * sizeof(float complex))

  for i in range(b):
    cgetrf(&m, &n, a_out, &m, ipiv, info)
    a_out += m * n
    ipiv += min(m, n)
    info += 1

register_cpu_custom_call_target(b"lapack_cgetrf", <void*>(lapack_cgetrf))


cdef void lapack_zgetrf(void* out_tuple, void** data) nogil:
  cdef int b = (<int32_t*>(data[0]))[0]
  cdef int m = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef const double complex* a_in = <double complex*>(data[3])

  cdef void** out = <void**>(out_tuple)
  cdef double complex* a_out = <double complex*>(out[0])
  cdef int* ipiv = <int*>(out[1])
  cdef int* info = <int*>(out[2])
  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(m) * <int64_t>(n) * sizeof(double complex))

  for i in range(b):
    zgetrf(&m, &n, a_out, &m, ipiv, info)
    a_out += m * n
    ipiv += min(m, n)
    info += 1

register_cpu_custom_call_target(b"lapack_zgetrf", <void*>(lapack_zgetrf))

def getrf(c, a):
  assert sizeof(int32_t) == sizeof(int)

  a_shape = c.GetShape(a)
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

  out = c.CustomCallWithLayout(
      fn,
      operands=(
        c.ConstantS32Scalar(b),
        c.ConstantS32Scalar(m),
        c.ConstantS32Scalar(n),
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
      ))
  return tuple(c.GetTupleElement(out, i) for i in range(3))

def jax_getrf(c, a):
  return c.Tuple(*getrf(c, a))

# ?geqrf: QR decomposition

cdef int lapack_sgeqrf_workspace(int m, int n):
  cdef float work
  cdef int lwork = -1
  cdef int info
  sgeqrf(&m, &n, NULL, &m, NULL, &work, &lwork, &info)
  return <int>(work) if info == 0 else -1

cdef void lapack_sgeqrf(void* out_tuple, void** data) nogil:
  cdef int b = (<int32_t*>(data[0]))[0]
  cdef int m = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef int lwork = (<int32_t*>(data[3]))[0]
  cdef const float* a_in = <float*>(data[4])

  cdef void** out = <void**>(out_tuple)
  cdef float* a_out = <float*>(out[0])
  cdef float* tau = <float*>(out[1])
  cdef int* info = <int*>(out[2])
  cdef float* work = <float*>(out[3])

  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(m) * <int64_t>(n) * sizeof(float))

  for i in range(b):
    sgeqrf(&m, &n, a_out, &m, tau, work, &lwork, info)
    a_out += m * n
    tau += min(m, n)
    info += 1

register_cpu_custom_call_target(b"lapack_sgeqrf", <void*>(lapack_sgeqrf))

cdef int lapack_dgeqrf_workspace(int m, int n):
  cdef double work
  cdef int lwork = -1
  cdef int info
  dgeqrf(&m, &n, NULL, &m, NULL, &work, &lwork, &info)
  return <int>(work) if info == 0 else -1

cdef void lapack_dgeqrf(void* out_tuple, void** data) nogil:
  cdef int b = (<int32_t*>(data[0]))[0]
  cdef int m = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef int lwork = (<int32_t*>(data[3]))[0]
  cdef const double* a_in = <double*>(data[4])

  cdef void** out = <void**>(out_tuple)
  cdef double* a_out = <double*>(out[0])
  cdef double* tau = <double*>(out[1])
  cdef int* info = <int*>(out[2])
  cdef double* work = <double*>(out[3])

  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(m) * <int64_t>(n) * sizeof(double))

  for i in range(b):
    dgeqrf(&m, &n, a_out, &m, tau, work, &lwork, info)
    a_out += m * n
    tau += min(m, n)
    info += 1

register_cpu_custom_call_target(b"lapack_dgeqrf", <void*>(lapack_dgeqrf))

cdef int lapack_cgeqrf_workspace(int m, int n):
  cdef float complex work
  cdef int lwork = -1
  cdef int info
  cgeqrf(&m, &n, NULL, &m, NULL, &work, &lwork, &info)
  return <int>(work.real) if info == 0 else -1

cdef void lapack_cgeqrf(void* out_tuple, void** data) nogil:
  cdef int b = (<int32_t*>(data[0]))[0]
  cdef int m = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef int lwork = (<int32_t*>(data[3]))[0]
  cdef const float complex* a_in = <float complex*>(data[4])

  cdef void** out = <void**>(out_tuple)
  cdef float complex* a_out = <float complex*>(out[0])
  cdef float complex* tau = <float complex*>(out[1])
  cdef int* info = <int*>(out[2])
  cdef float complex* work = <float complex*>(out[3])

  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(m) * <int64_t>(n) * sizeof(float complex))

  for i in range(b):
    cgeqrf(&m, &n, a_out, &m, tau, work, &lwork, info)
    a_out += m * n
    tau += min(m, n)
    info += 1

register_cpu_custom_call_target(b"lapack_cgeqrf", <void*>(lapack_cgeqrf))

cdef int lapack_zgeqrf_workspace(int m, int n):
  cdef double complex work
  cdef int lwork = -1
  cdef int info
  zgeqrf(&m, &n, NULL, &m, NULL, &work, &lwork, &info)
  return <int>(work.real) if info == 0 else -1

cdef void lapack_zgeqrf(void* out_tuple, void** data) nogil:
  cdef int b = (<int32_t*>(data[0]))[0]
  cdef int m = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef int lwork = (<int32_t*>(data[3]))[0]
  cdef const double complex* a_in = <double complex*>(data[4])

  cdef void** out = <void**>(out_tuple)
  cdef double complex* a_out = <double complex*>(out[0])
  cdef double complex* tau = <double complex*>(out[1])
  cdef int* info = <int*>(out[2])
  cdef double complex* work = <double complex*>(out[3])

  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(m) * <int64_t>(n) * sizeof(double complex))

  for i in range(b):
    zgeqrf(&m, &n, a_out, &m, tau, work, &lwork, info)
    a_out += m * n
    tau += min(m, n)
    info += 1

register_cpu_custom_call_target(b"lapack_zgeqrf", <void*>(lapack_zgeqrf))

def geqrf(c, a):
  assert sizeof(int32_t) == sizeof(int)

  a_shape = c.GetShape(a)
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
    lwork = lapack_sgeqrf_workspace(m, n)
  elif dtype == np.float64:
    fn = b"lapack_dgeqrf"
    lwork = lapack_dgeqrf_workspace(m, n)
  elif dtype == np.complex64:
    fn = b"lapack_cgeqrf"
    lwork = lapack_cgeqrf_workspace(m, n)
  elif dtype == np.complex128:
    fn = b"lapack_zgeqrf"
    lwork = lapack_zgeqrf_workspace(m, n)
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  out = c.CustomCallWithLayout(
      fn,
      operands=(
        c.ConstantS32Scalar(b),
        c.ConstantS32Scalar(m),
        c.ConstantS32Scalar(n),
        c.ConstantS32Scalar(lwork),
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
      ))
  return tuple(c.GetTupleElement(out, i) for i in range(3))

# ?orgqr: product of elementary Householder reflectors:

cdef int lapack_sorgqr_workspace(int m, int n, int k):
  cdef float work
  cdef int lwork = -1
  cdef int info
  sorgqr(&m, &n, &k, NULL, &m, NULL, &work, &lwork, &info)
  return <int>(work) if info == 0 else -1

cdef void lapack_sorgqr(void* out_tuple, void** data) nogil:
  cdef int b = (<int32_t*>(data[0]))[0]
  cdef int m = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef int k = (<int32_t*>(data[3]))[0]
  cdef int lwork = (<int32_t*>(data[4]))[0]
  cdef const float* a_in = <float*>(data[5])
  cdef float* tau = <float*>(data[6])

  cdef void** out = <void**>(out_tuple)
  cdef float* a_out = <float*>(out[0])
  cdef int* info = <int*>(out[1])
  cdef float* work = <float*>(out[2])

  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(m) * <int64_t>(n) * sizeof(float))

  for i in range(b):
    sorgqr(&m, &n, &k, a_out, &m, tau, work, &lwork, info)
    a_out += m * n
    tau += k
    info += 1

register_cpu_custom_call_target(b"lapack_sorgqr", <void*>(lapack_sorgqr))

cdef int lapack_dorgqr_workspace(int m, int n, int k):
  cdef double work
  cdef int lwork = -1
  cdef int info
  dorgqr(&m, &n, &k, NULL, &m, NULL, &work, &lwork, &info)
  return <int>(work) if info == 0 else -1

cdef void lapack_dorgqr(void* out_tuple, void** data) nogil:
  cdef int b = (<int32_t*>(data[0]))[0]
  cdef int m = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef int k = (<int32_t*>(data[3]))[0]
  cdef int lwork = (<int32_t*>(data[4]))[0]
  cdef const double* a_in = <double*>(data[5])
  cdef double* tau = <double*>(data[6])

  cdef void** out = <void**>(out_tuple)
  cdef double* a_out = <double*>(out[0])
  cdef int* info = <int*>(out[1])
  cdef double* work = <double*>(out[2])

  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(m) * <int64_t>(n) * sizeof(double))

  for i in range(b):
    dorgqr(&m, &n, &k, a_out, &m, tau, work, &lwork, info)
    a_out += m * n
    tau += k
    info += 1

register_cpu_custom_call_target(b"lapack_dorgqr", <void*>(lapack_dorgqr))

cdef int lapack_cungqr_workspace(int m, int n, int k):
  cdef float complex work
  cdef int lwork = -1
  cdef int info
  cungqr(&m, &n, &k, NULL, &m, NULL, &work, &lwork, &info)
  return <int>(work.real) if info == 0 else -1

cdef void lapack_cungqr(void* out_tuple, void** data) nogil:
  cdef int b = (<int32_t*>(data[0]))[0]
  cdef int m = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef int k = (<int32_t*>(data[3]))[0]
  cdef int lwork = (<int32_t*>(data[4]))[0]
  cdef const float complex* a_in = <float complex*>(data[5])
  cdef float complex* tau = <float complex*>(data[6])

  cdef void** out = <void**>(out_tuple)
  cdef float complex* a_out = <float complex*>(out[0])
  cdef int* info = <int*>(out[1])
  cdef float complex* work = <float complex*>(out[2])

  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(m) * <int64_t>(n) * sizeof(float complex))

  for i in range(b):
    cungqr(&m, &n, &k, a_out, &m, tau, work, &lwork, info)
    a_out += m * n
    tau += k
    info += 1

register_cpu_custom_call_target(b"lapack_cungqr", <void*>(lapack_cungqr))

cdef int lapack_zungqr_workspace(int m, int n, int k):
  cdef double complex work
  cdef int lwork = -1
  cdef int info
  zungqr(&m, &n, &k, NULL, &m, NULL, &work, &lwork, &info)
  return <int>(work.real) if info == 0 else -1

cdef void lapack_zungqr(void* out_tuple, void** data) nogil:
  cdef int b = (<int32_t*>(data[0]))[0]
  cdef int m = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef int k = (<int32_t*>(data[3]))[0]
  cdef int lwork = (<int32_t*>(data[4]))[0]
  cdef const double complex* a_in = <double complex*>(data[5])
  cdef double complex* tau = <double complex*>(data[6])

  cdef void** out = <void**>(out_tuple)
  cdef double complex* a_out = <double complex*>(out[0])
  cdef int* info = <int*>(out[1])
  cdef double complex* work = <double complex*>(out[2])

  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(m) * <int64_t>(n) * sizeof(double complex))

  for i in range(b):
    zungqr(&m, &n, &k, a_out, &m, tau, work, &lwork, info)
    a_out += m * n
    tau += k
    info += 1

register_cpu_custom_call_target(b"lapack_zungqr", <void*>(lapack_zungqr))

def orgqr(c, a, tau):
  assert sizeof(int32_t) == sizeof(int)

  a_shape = c.GetShape(a)
  dtype = a_shape.element_type()
  dims = a_shape.dimensions()
  assert len(dims) >= 2
  m, n = dims[-2:]
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)
  b = 1
  for d in batch_dims:
    b *= d

  tau_dims = c.GetShape(tau).dimensions()
  assert tau_dims[:-1] == dims[:-2]
  k = tau_dims[-1]

  if dtype == np.float32:
    fn = b"lapack_sorgqr"
    lwork = lapack_sorgqr_workspace(m, n, k)
  elif dtype == np.float64:
    fn = b"lapack_dorgqr"
    lwork = lapack_dorgqr_workspace(m, n, k)
  elif dtype == np.complex64:
    fn = b"lapack_cungqr"
    lwork = lapack_cungqr_workspace(m, n, k)
  elif dtype == np.complex128:
    fn = b"lapack_zungqr"
    lwork = lapack_zungqr_workspace(m, n, k)
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  out = c.CustomCallWithLayout(
      fn,
      operands=(
        c.ConstantS32Scalar(b),
        c.ConstantS32Scalar(m),
        c.ConstantS32Scalar(n),
        c.ConstantS32Scalar(k),
        c.ConstantS32Scalar(lwork),
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
      ))
  return tuple(c.GetTupleElement(out, i) for i in range(2))


# ?potrf: Cholesky decomposition

cdef void lapack_spotrf(void* out_tuple, void** data) nogil:
  cdef int32_t lower = (<int32_t*>(data[0]))[0]
  cdef int b = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef const float* a_in = <float*>(data[3])
  cdef char uplo = 'L' if lower else 'U'

  cdef void** out = <void**>(out_tuple)
  cdef float* a_out = <float*>(out[0])
  cdef int* info = <int*>(out[1])
  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(n) * <int64_t>(n) * sizeof(float))

  for i in range(b):
    spotrf(&uplo, &n, a_out, &n, info)
    a_out += <int64_t>(n) * <int64_t>(n)
    info += 1

register_cpu_custom_call_target(b"lapack_spotrf", <void*>(lapack_spotrf))


cdef void lapack_dpotrf(void* out_tuple, void** data) nogil:
  cdef int32_t lower = (<int32_t*>(data[0]))[0]
  cdef int b = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef const double* a_in = <double*>(data[3])
  cdef char uplo = 'L' if lower else 'U'

  cdef void** out = <void**>(out_tuple)
  cdef double* a_out = <double*>(out[0])
  cdef int* info = <int*>(out[1])
  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(n) * <int64_t>(n) * sizeof(double))

  for i in range(b):
    dpotrf(&uplo, &n, a_out, &n, info)
    a_out += <int64_t>(n) * <int64_t>(n)
    info += 1

register_cpu_custom_call_target(b"lapack_dpotrf", <void*>(lapack_dpotrf))


cdef void lapack_cpotrf(void* out_tuple, void** data) nogil:
  cdef int32_t lower = (<int32_t*>(data[0]))[0]
  cdef int b = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef const float complex* a_in = <float complex*>(data[3])
  cdef char uplo = 'L' if lower else 'U'

  cdef void** out = <void**>(out_tuple)
  cdef float complex* a_out = <float complex*>(out[0])
  cdef int* info = <int*>(out[1])
  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(n) * <int64_t>(n) * sizeof(float complex))

  for i in range(b):
    cpotrf(&uplo, &n, a_out, &n, info)
    a_out += <int64_t>(n) * <int64_t>(n)
    info += 1

register_cpu_custom_call_target(b"lapack_cpotrf", <void*>(lapack_cpotrf))

cdef void lapack_zpotrf(void* out_tuple, void** data) nogil:
  cdef int32_t lower = (<int32_t*>(data[0]))[0]
  cdef int b = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef const double complex* a_in = <double complex*>(data[3])
  cdef char uplo = 'L' if lower else 'U'

  cdef void** out = <void**>(out_tuple)
  cdef double complex* a_out = <double complex*>(out[0])
  cdef int* info = <int*>(out[1])
  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(n) * <int64_t>(n) * sizeof(double complex))

  for i in range(b):
    zpotrf(&uplo, &n, a_out, &n, info)
    a_out += <int64_t>(n) * <int64_t>(n)
    info += 1

register_cpu_custom_call_target(b"lapack_zpotrf", <void*>(lapack_zpotrf))

def potrf(c, a, lower=False):
  assert sizeof(int32_t) == sizeof(int)

  a_shape = c.GetShape(a)
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
  out = c.CustomCallWithLayout(
      fn,
      operands=(c.ConstantS32Scalar(int(lower)),
                c.ConstantS32Scalar(b), c.ConstantS32Scalar(n), a),
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
      ))
  return tuple(c.GetTupleElement(out, i) for i in range(2))


# ?gesdd: Singular value decomposition

cdef int gesdd_iwork_size(int m, int n) nogil:
  return 8 * min(m, n)

cdef int cgesdd_rwork_size(int m, int n, int compute_uv) nogil:
  cdef int mn = min(m, n)
  if compute_uv == 0:
    return 7 * mn
  cdef int mx = max(m, n)
  return max(5 * mn * mn + 5 * mn, 2 * mx * mn + 2 * mn * mn + mn)

cdef char gesdd_jobz(bool_t job_opt_compute_uv,
                     bool_t job_opt_full_matrices) nogil:
  # define appropriate job code
  cdef char jobz = 'A'
  if job_opt_compute_uv == 0:
    jobz = 'N'
  else:
    if job_opt_full_matrices == 0:
      jobz = 'S'
  return jobz

cdef int sgesdd_work_size(int m, int n, bool_t job_opt_compute_uv,
                          bool_t job_opt_full_matrices):
  cdef float work
  cdef int lwork = -1
  cdef int info
  cdef int ldvt = min(m, n) if job_opt_full_matrices == 0 else n
  cdef char jobz = gesdd_jobz(job_opt_compute_uv, job_opt_full_matrices)
  sgesdd(&jobz, &m, &n, NULL, &m, NULL, NULL, &m, NULL, &ldvt, &work,
         &lwork, NULL, &info)
  return <int>(work) if info == 0 else -1

cdef void lapack_sgesdd(void* out_tuple, void** data) nogil:
  cdef int32_t job_opt_full_matrices = (<int32_t*>(data[0]))[0]
  cdef int32_t job_opt_compute_uv = (<int32_t*>(data[1]))[0]
  cdef int b = (<int32_t*>(data[2]))[0]
  cdef int m = (<int32_t*>(data[3]))[0]
  cdef int n = (<int32_t*>(data[4]))[0]
  cdef int lwork = (<int32_t*>(data[5]))[0]
  cdef float* a_in = <float*>(data[6])

  cdef void** out = <void**>(out_tuple)
  cdef float* a_out = <float*>(out[0])
  cdef float* s = <float*>(out[1])
  cdef float* u = <float*>(out[2])
  cdef float* vt = <float*>(out[3])
  cdef int* info = <int*>(out[4])
  cdef int* iwork = <int*>(out[5])
  cdef float* work = <float*>(out[6])

  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(m) * <int64_t>(n) * sizeof(float))

  cdef char jobz = gesdd_jobz(job_opt_compute_uv, job_opt_full_matrices)

  cdef int lda = m
  cdef int ldu = m
  cdef int tdu = min(m, n) if job_opt_full_matrices == 0 else m
  cdef int ldvt = min(m, n) if job_opt_full_matrices == 0 else n

  for i in range(b):
    sgesdd(&jobz, &m, &n, a_out, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
           iwork, info)
    a_out += m * n
    s += min(m, n)
    u += m * tdu
    vt += ldvt * n
    info += 1

register_cpu_custom_call_target(b"lapack_sgesdd", <void*>(lapack_sgesdd))


cdef int dgesdd_work_size(int m, int n, bool_t job_opt_compute_uv,
                          bool_t job_opt_full_matrices):
  cdef double work
  cdef int lwork = -1
  cdef int info
  cdef int ldvt = min(m, n) if job_opt_full_matrices == 0 else n
  cdef char jobz = gesdd_jobz(job_opt_compute_uv, job_opt_full_matrices)
  dgesdd(&jobz, &m, &n, NULL, &m, NULL, NULL, &m, NULL, &ldvt, &work,
         &lwork, NULL, &info)
  return <int>(work) if info == 0 else -1

cdef void lapack_dgesdd(void* out_tuple, void** data) nogil:
  cdef int32_t job_opt_full_matrices = (<int32_t*>(data[0]))[0]
  cdef int32_t job_opt_compute_uv = (<int32_t*>(data[1]))[0]
  cdef int b = (<int32_t*>(data[2]))[0]
  cdef int m = (<int32_t*>(data[3]))[0]
  cdef int n = (<int32_t*>(data[4]))[0]
  cdef int lwork = (<int32_t*>(data[5]))[0]
  cdef double* a_in = <double*>(data[6])

  cdef void** out = <void**>(out_tuple)
  cdef double* a_out = <double*>(out[0])
  cdef double* s = <double*>(out[1])
  cdef double* u = <double*>(out[2])
  cdef double* vt = <double*>(out[3])
  cdef int* info = <int*>(out[4])
  cdef int* iwork = <int*>(out[5])
  cdef double* work = <double*>(out[6])

  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(m) * <int64_t>(n) * sizeof(double))

  cdef char jobz = gesdd_jobz(job_opt_compute_uv, job_opt_full_matrices)

  cdef int lda = m
  cdef int ldu = m
  cdef int tdu = min(m, n) if job_opt_full_matrices == 0 else m
  cdef int ldvt = min(m, n) if job_opt_full_matrices == 0 else n

  for i in range(b):
    dgesdd(&jobz, &m, &n, a_out, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
           iwork, info)
    a_out += m * n
    s += min(m, n)
    u += m * tdu
    vt += ldvt * n
    info += 1

register_cpu_custom_call_target(b"lapack_dgesdd", <void*>(lapack_dgesdd))

cdef int cgesdd_work_size(int m, int n, bool_t job_opt_compute_uv,
                          bool_t job_opt_full_matrices):
  cdef float complex work
  cdef int lwork = -1
  cdef int info
  cdef int ldvt = min(m, n) if job_opt_full_matrices == 0 else n
  cdef char jobz = gesdd_jobz(job_opt_compute_uv, job_opt_full_matrices)
  cgesdd(&jobz, &m, &n, NULL, &m, NULL, NULL, &m, NULL, &ldvt, &work,
         &lwork, NULL, NULL, &info)
  return <int>(work.real) if info == 0 else -1

cdef void lapack_cgesdd(void* out_tuple, void** data) nogil:
  cdef int32_t job_opt_full_matrices = (<int32_t*>(data[0]))[0]
  cdef int32_t job_opt_compute_uv = (<int32_t*>(data[1]))[0]
  cdef int b = (<int32_t*>(data[2]))[0]
  cdef int m = (<int32_t*>(data[3]))[0]
  cdef int n = (<int32_t*>(data[4]))[0]
  cdef int lwork = (<int32_t*>(data[5]))[0]
  cdef float complex* a_in = <float complex*>(data[6])

  cdef void** out = <void**>(out_tuple)
  cdef float complex* a_out = <float complex*>(out[0])
  cdef float* s = <float*>(out[1])
  cdef float complex* u = <float complex*>(out[2])
  cdef float complex* vt = <float complex*>(out[3])
  cdef int* info = <int*>(out[4])
  cdef int* iwork = <int*>(out[5])
  cdef float* rwork = <float*>(out[6])
  cdef float complex* work = <float complex*>(out[7])

  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(m) * <int64_t>(n) * sizeof(float complex))

  cdef char jobz = gesdd_jobz(job_opt_compute_uv, job_opt_full_matrices)

  cdef int lda = m
  cdef int ldu = m
  cdef int tdu = min(m, n) if job_opt_full_matrices == 0 else m
  cdef int ldvt = min(m, n) if job_opt_full_matrices == 0 else n

  for i in range(b):
    cgesdd(&jobz, &m, &n, a_out, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
           rwork, iwork, info)
    a_out += m * n
    s += min(m, n)
    u += m * tdu
    vt += ldvt * n
    info += 1

register_cpu_custom_call_target(b"lapack_cgesdd", <void*>(lapack_cgesdd))


cdef int zgesdd_work_size(int m, int n, bool_t job_opt_compute_uv,
                          bool_t job_opt_full_matrices):
  cdef double complex work
  cdef int lwork = -1
  cdef int info
  cdef int ldvt = min(m, n) if job_opt_full_matrices == 0 else n
  cdef char jobz = gesdd_jobz(job_opt_compute_uv, job_opt_full_matrices)
  zgesdd(&jobz, &m, &n, NULL, &m, NULL, NULL, &m, NULL, &ldvt, &work,
         &lwork, NULL, NULL, &info)
  return <int>(work.real) if info == 0 else -1

cdef void lapack_zgesdd(void* out_tuple, void** data) nogil:
  cdef int32_t job_opt_full_matrices = (<int32_t*>(data[0]))[0]
  cdef int32_t job_opt_compute_uv = (<int32_t*>(data[1]))[0]
  cdef int b = (<int32_t*>(data[2]))[0]
  cdef int m = (<int32_t*>(data[3]))[0]
  cdef int n = (<int32_t*>(data[4]))[0]
  cdef int lwork = (<int32_t*>(data[5]))[0]
  cdef double complex* a_in = <double complex*>(data[6])

  cdef void** out = <void**>(out_tuple)
  cdef double complex* a_out = <double complex*>(out[0])
  cdef double* s = <double*>(out[1])
  cdef double complex* u = <double complex*>(out[2])
  cdef double complex* vt = <double complex*>(out[3])
  cdef int* info = <int*>(out[4])
  cdef int* iwork = <int*>(out[5])
  cdef double* rwork = <double*>(out[6])
  cdef double complex* work = <double complex*>(out[7])

  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(m) * <int64_t>(n) * sizeof(double complex))

  cdef char jobz = gesdd_jobz(job_opt_compute_uv, job_opt_full_matrices)

  cdef int lda = m
  cdef int ldu = m
  cdef int tdu = min(m, n) if job_opt_full_matrices == 0 else m
  cdef int ldvt = min(m, n) if job_opt_full_matrices == 0 else n

  for i in range(b):
    zgesdd(&jobz, &m, &n, a_out, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
           rwork, iwork, info)
    a_out += m * n
    s += min(m, n)
    u += m * tdu
    vt += ldvt * n
    info += 1

register_cpu_custom_call_target(b"lapack_zgesdd", <void*>(lapack_zgesdd))

def gesdd(c, a, full_matrices=True, compute_uv=True):
  assert sizeof(int32_t) == sizeof(int)

  a_shape = c.GetShape(a)
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
    lwork = sgesdd_work_size(m, n, compute_uv, full_matrices)
    workspace = (
      Shape.array_shape(np.dtype(np.int32), (gesdd_iwork_size(m, n),), (0,)),
      Shape.array_shape(dtype, (lwork,), (0,)),
    )
  elif dtype == np.float64:
    fn = b"lapack_dgesdd"
    singular_vals_dtype = np.float64
    lwork = dgesdd_work_size(m, n, compute_uv, full_matrices)
    workspace = (
      Shape.array_shape(np.dtype(np.int32), (gesdd_iwork_size(m, n),), (0,)),
      Shape.array_shape(dtype, (lwork,), (0,)),
    )
  elif dtype == np.complex64:
    fn = b"lapack_cgesdd"
    singular_vals_dtype = np.float32
    lwork = cgesdd_work_size(m, n, compute_uv, full_matrices)
    workspace = (
      Shape.array_shape(np.dtype(np.int32), (gesdd_iwork_size(m, n),), (0,)),
      Shape.array_shape(np.dtype(np.float32),
                        (cgesdd_rwork_size(m, n, int(compute_uv)),), (0,)),
      Shape.array_shape(dtype, (lwork,), (0,)),
    )
  elif dtype == np.complex128:
    fn = b"lapack_zgesdd"
    singular_vals_dtype = np.float64
    lwork = zgesdd_work_size(m, n, compute_uv, full_matrices)
    workspace = (
      Shape.array_shape(np.dtype(np.int32), (gesdd_iwork_size(m, n),), (0,)),
      Shape.array_shape(np.dtype(np.float64),
                        (cgesdd_rwork_size(m, n, int(compute_uv)),), (0,)),
      Shape.array_shape(dtype, (lwork,), (0,)),
    )
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  scalar_layout = tuple(range(num_bd - 1, -1, -1))
  vector_layout = (num_bd,) + scalar_layout
  matrix_layout = (num_bd, num_bd + 1) + scalar_layout
  out = c.CustomCallWithLayout(
      fn,
      operands=(c.ConstantS32Scalar(int(full_matrices)),
                c.ConstantS32Scalar(int(compute_uv)),
                c.ConstantS32Scalar(b),
                c.ConstantS32Scalar(m), c.ConstantS32Scalar(n),
                c.ConstantS32Scalar(lwork), a),
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
      ))
  return (c.GetTupleElement(out, 1), c.GetTupleElement(out, 2),
          c.GetTupleElement(out, 3), c.GetTupleElement(out, 4))

def jax_gesdd(c, a, full_matrices=True, compute_uv=True):
  return c.Tuple(*gesdd(c, a, full_matrices, compute_uv))


# syevd: Symmetric eigendecomposition

# Workspace sizes, taken from the LAPACK documentation.
cdef int syevd_work_size(int n) nogil:
  return 1 + 6 * n + 2 * n * n

cdef int syevd_iwork_size(int n) nogil:
  return 3 + 5 * n

cdef void lapack_ssyevd(void* out_tuple, void** data) nogil:
  cdef int32_t lower = (<int32_t*>(data[0]))[0]
  cdef int b = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef const float* a_in = <float*>(data[3])
  cdef void** out = <void**>(out_tuple)
  cdef float* a_out = <float*>(out[0])
  cdef float* w_out = <float*>(out[1])
  cdef int* info_out = <int*>(out[2])
  cdef float* work = <float*>(out[3])
  cdef int* iwork = <int*>(out[4])
  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(n) * <int64_t>(n) * sizeof(float))

  cdef char jobz = 'V'
  cdef char uplo = 'L' if lower else 'U'

  cdef int lwork = syevd_work_size(n)
  cdef int liwork = syevd_iwork_size(n)
  for i in range(b):
    ssyevd(&jobz, &uplo, &n, a_out, &n, w_out, work, &lwork, iwork, &liwork,
           info_out)
    a_out += n * n
    w_out += n
    info_out += 1

register_cpu_custom_call_target(b"lapack_ssyevd", <void*>(lapack_ssyevd))

cdef void lapack_dsyevd(void* out_tuple, void** data) nogil:
  cdef int32_t lower = (<int32_t*>(data[0]))[0]
  cdef int b = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef const double* a_in = <double*>(data[3])

  cdef void** out = <void**>(out_tuple)
  cdef double* a_out = <double*>(out[0])
  cdef double* w_out = <double*>(out[1])
  cdef int* info_out = <int*>(out[2])
  cdef double* work = <double*>(out[3])
  cdef int* iwork = <int*>(out[4])
  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(n) * <int64_t>(n) * sizeof(double))

  cdef char jobz = 'V'
  cdef char uplo = 'L' if lower else 'U'

  cdef int lwork = syevd_work_size(n)
  cdef int liwork = syevd_iwork_size(n)
  for i in range(b):
    dsyevd(&jobz, &uplo, &n, a_out, &n, w_out, work, &lwork, iwork, &liwork,
           info_out)
    a_out += n * n
    w_out += n
    info_out += 1

register_cpu_custom_call_target(b"lapack_dsyevd", <void*>(lapack_dsyevd))

# Workspace sizes, taken from the LAPACK documentation.
cdef int heevd_work_size(int n) nogil:
  return 1 + 2 * n + n * n

cdef int heevd_rwork_size(int n) nogil:
  return 1 + 5 * n + 2 * n * n


cdef void lapack_cheevd(void* out_tuple, void** data) nogil:
  cdef int32_t lower = (<int32_t*>(data[0]))[0]
  cdef int b = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef const float complex* a_in = <float complex*>(data[3])

  cdef void** out = <void**>(out_tuple)
  cdef float complex* a_out = <float complex*>(out[0])
  cdef float* w_out = <float*>(out[1])
  cdef int* info_out = <int*>(out[2])
  cdef float complex* work = <float complex*>(out[3])
  cdef float* rwork = <float*>(out[4])
  cdef int* iwork = <int*>(out[5])
  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(n) * <int64_t>(n) * sizeof(float complex))

  cdef char jobz = 'V'
  cdef char uplo = 'L' if lower else 'U'

  cdef int lwork = heevd_work_size(n)
  cdef int lrwork = heevd_rwork_size(n)
  cdef int liwork = syevd_iwork_size(n)
  for i in range(b):
    cheevd(&jobz, &uplo, &n, a_out, &n, w_out, work, &lwork, rwork, &lrwork,
           iwork, &liwork, info_out)
    a_out += n * n
    w_out += n
    info_out += 1

register_cpu_custom_call_target(b"lapack_cheevd", <void*>(lapack_cheevd))


cdef void lapack_zheevd(void* out_tuple, void** data) nogil:
  cdef int32_t lower = (<int32_t*>(data[0]))[0]
  cdef int b = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef const double complex* a_in = <double complex*>(data[3])

  cdef void** out = <void**>(out_tuple)
  cdef double complex* a_out = <double complex*>(out[0])
  cdef double* w_out = <double*>(out[1])
  cdef int* info_out = <int*>(out[2])
  cdef double complex* work = <double complex*>(out[3])
  cdef double* rwork = <double*>(out[4])
  cdef int* iwork = <int*>(out[5])
  if a_out != a_in:
    memcpy(a_out, a_in,
           <int64_t>(b) * <int64_t>(n) * <int64_t>(n) * sizeof(double complex))

  cdef char jobz = 'V'
  cdef char uplo = 'L' if lower else 'U'

  cdef int lwork = heevd_work_size(n)
  cdef int lrwork = heevd_rwork_size(n)
  cdef int liwork = syevd_iwork_size(n)
  for i in range(b):
    zheevd(&jobz, &uplo, &n, a_out, &n, w_out, work, &lwork, rwork, &lrwork,
           iwork, &liwork, info_out)
    a_out += n * n
    w_out += n
    info_out += 1

register_cpu_custom_call_target(b"lapack_zheevd", <void*>(lapack_zheevd))

def syevd(c, a, lower=False):
  assert sizeof(int32_t) == sizeof(int)

  a_shape = c.GetShape(a)
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
    workspace = (Shape.array_shape(dtype, (syevd_work_size(n),), (0,)),
                 Shape.array_shape(np.dtype(np.int32),
                                   (syevd_iwork_size(n),), (0,)))
  elif dtype == np.float64:
    fn = b"lapack_dsyevd"
    eigvals_type = np.float64
    workspace = (Shape.array_shape(dtype, (syevd_work_size(n),), (0,)),
                 Shape.array_shape(np.dtype(np.int32),
                                   (syevd_iwork_size(n),), (0,)))
  elif dtype == np.complex64:
    fn = b"lapack_cheevd"
    eigvals_type = np.float32
    workspace = (Shape.array_shape(dtype, (heevd_work_size(n),), (0,)),
                 Shape.array_shape(np.dtype(np.float32),
                                   (heevd_rwork_size(n),), (0,)),
                 Shape.array_shape(np.dtype(np.int32),
                                   (syevd_iwork_size(n),), (0,)))
  elif dtype == np.complex128:
    fn = b"lapack_zheevd"
    eigvals_type = np.float64
    workspace = (Shape.array_shape(dtype, (heevd_work_size(n),), (0,)),
                 Shape.array_shape(np.dtype(np.float64),
                                   (heevd_rwork_size(n),), (0,)),
                 Shape.array_shape(np.dtype(np.int32),
                                   (syevd_iwork_size(n),), (0,)))
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  out = c.CustomCallWithLayout(
      fn,
      operands=(c.ConstantS32Scalar(1 if lower else 0),
                c.ConstantS32Scalar(b),
                c.ConstantS32Scalar(n),
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
      ))
  return (c.GetTupleElement(out, 0), c.GetTupleElement(out, 1),
          c.GetTupleElement(out, 2))

def jax_syevd(c, a, lower=False):
  return c.Tuple(*syevd(c, a, lower))


# geev: Nonsymmetric eigendecomposition

# LAPACK uses a packed representation to represent a mixture of real
# eigenvectors and complex conjugate pairs. This helper unpacks the
# representation into regular complex matrices.
cdef void _unpack_float_eigenvectors(
    int n, const float* im_eigenvalues, const float* packed,
    float complex* unpacked) nogil:
  cdef float re, im
  cdef int j, k
  j = 0
  while j < n:
    if im_eigenvalues[j] == 0. or isnan(im_eigenvalues[j]):
      for k in range(n):
        unpacked[j*n + k].real = packed[j*n + k]
        unpacked[j*n + k].imag = 0.
      j += 1
    else:
      for k in range(n):
        re = packed[j*n + k]
        im = packed[(j+1)*n + k]
        unpacked[j*n + k].real = unpacked[(j + 1)*n + k].real = re
        unpacked[j*n + k].imag = im
        unpacked[(j + 1)*n + k].imag = -im
      j += 2

cdef void lapack_sgeev(void* out_tuple, void** data) nogil:
  cdef int b = (<int32_t*>(data[0]))[0]
  cdef int n = (<int32_t*>(data[1]))[0]
  cdef const float* a_in = <float*>(data[2])

  cdef void** out = <void**>(out_tuple)
  cdef float* a_work = <float*>(out[0])
  cdef float* vl_work = <float*>(out[1])
  cdef float* vr_work = <float*>(out[2])

  cdef float* wr_out = <float*>(out[3])
  cdef float* wi_out = <float*>(out[4])
  cdef float complex* vl_out = <float complex*>(out[5])
  cdef float complex* vr_out = <float complex*>(out[6])
  cdef int* info_out = <int*>(out[7])

  cdef char jobvlr = 'V'
  cdef float work_query
  cdef int lwork = -1
  sgeev(&jobvlr, &jobvlr, &n, a_work, &n, wr_out, wi_out, vl_work, &n,
        vr_work, &n, &work_query, &lwork, info_out)
  lwork = <int>(work_query)
  cdef float* work = <float*> malloc(lwork * sizeof(float))

  for i in range(b):
    memcpy(a_work, a_in, <int64_t>(n) * <int64_t>(n) * sizeof(float))
    sgeev(&jobvlr, &jobvlr, &n, a_work, &n, wr_out, wi_out, vl_work, &n,
          vr_work, &n, work, &lwork, info_out)
    if info_out[0] == 0:
      _unpack_float_eigenvectors(n, wi_out, vl_work, vl_out)
      _unpack_float_eigenvectors(n, wi_out, vr_work, vr_out)

    a_in += n * n
    wr_out += n
    wi_out += n
    vl_out += n * n
    vr_out += n * n
    info_out += 1
  free(work)

register_cpu_custom_call_target(b"lapack_sgeev", <void*>(lapack_sgeev))

cdef void _unpack_double_eigenvectors(
    int n, const double* im_eigenvalues, const double* packed,
    double complex* unpacked) nogil:
  cdef double re, im
  cdef int j, k
  j = 0
  while j < n:
    if im_eigenvalues[j] == 0. or isnan(im_eigenvalues[j]):
      for k in range(n):
        unpacked[j*n + k].real = packed[j*n + k]
        unpacked[j*n + k].imag = 0.
      j += 1
    else:
      for k in range(n):
        re = packed[j*n + k]
        im = packed[(j+1)*n + k]
        unpacked[j*n + k].real = unpacked[(j + 1)*n + k].real = re
        unpacked[j*n + k].imag = im
        unpacked[(j + 1)*n + k].imag = -im
      j += 2


cdef void lapack_dgeev(void* out_tuple, void** data) nogil:
  cdef int b = (<int32_t*>(data[0]))[0]
  cdef int n = (<int32_t*>(data[1]))[0]
  cdef const double* a_in = <double*>(data[2])

  cdef void** out = <void**>(out_tuple)
  cdef double* a_work = <double*>(out[0])
  cdef double* vl_work = <double*>(out[1])
  cdef double* vr_work = <double*>(out[2])

  cdef double* wr_out = <double*>(out[3])
  cdef double* wi_out = <double*>(out[4])
  cdef double complex* vl_out = <double complex*>(out[5])
  cdef double complex* vr_out = <double complex*>(out[6])
  cdef int* info_out = <int*>(out[7])

  cdef char jobvlr = 'V'
  cdef double work_query
  cdef int lwork = -1
  dgeev(&jobvlr, &jobvlr, &n, a_work, &n, wr_out, wi_out, vl_work, &n,
        vr_work, &n, &work_query, &lwork, info_out)
  lwork = <int>(work_query)
  cdef double* work = <double*> malloc(lwork * sizeof(double))

  for i in range(b):
    memcpy(a_work, a_in, <int64_t>(n) * <int64_t>(n) * sizeof(double))
    dgeev(&jobvlr, &jobvlr, &n, a_work, &n, wr_out, wi_out, vl_work, &n,
          vr_work, &n, work, &lwork, info_out)
    if info_out[0] == 0:
      _unpack_double_eigenvectors(n, wi_out, vl_work, vl_out)
      _unpack_double_eigenvectors(n, wi_out, vr_work, vr_out)

    a_in += n * n
    wr_out += n
    wi_out += n
    vl_out += n * n
    vr_out += n * n
    info_out += 1
  free(work)

register_cpu_custom_call_target(b"lapack_dgeev", <void*>(lapack_dgeev))


cdef void lapack_cgeev(void* out_tuple, void** data) nogil:
  cdef int b = (<int32_t*>(data[0]))[0]
  cdef int n = (<int32_t*>(data[1]))[0]
  cdef const float complex* a_in = <float complex*>(data[2])

  cdef void** out = <void**>(out_tuple)
  cdef float complex* a_work = <float complex*>(out[0])
  cdef float* r_work = <float*>(out[1])

  cdef float complex* w_out = <float complex*>(out[2])
  cdef float complex* vl_out = <float complex*>(out[3])
  cdef float complex* vr_out = <float complex*>(out[4])
  cdef int* info_out = <int*>(out[5])

  cdef char jobvlr = 'V'
  cdef float complex work_query
  cdef int lwork = -1
  cgeev(&jobvlr, &jobvlr, &n, a_work, &n, w_out, vl_out, &n,
        vr_out, &n, &work_query, &lwork, r_work, info_out)
  lwork = <int>(work_query.real)
  cdef float complex* work = <float complex*>malloc(
      lwork * sizeof(float complex))

  for i in range(b):
    memcpy(a_work, a_in, <int64_t>(n) * <int64_t>(n) * sizeof(float complex))
    cgeev(&jobvlr, &jobvlr, &n, a_work, &n, w_out, vl_out, &n, vr_out, &n,
          work, &lwork, r_work, info_out)

    a_in += n * n
    w_out += n
    vl_out += n * n
    vr_out += n * n
    info_out += 1
  free(work)

register_cpu_custom_call_target(b"lapack_cgeev", <void*>(lapack_cgeev))


cdef void lapack_zgeev(void* out_tuple, void** data) nogil:
  cdef int b = (<int32_t*>(data[0]))[0]
  cdef int n = (<int32_t*>(data[1]))[0]
  cdef const double complex* a_in = <double complex*>(data[2])

  cdef void** out = <void**>(out_tuple)
  cdef double complex* a_work = <double complex*>(out[0])
  cdef double* r_work = <double*>(out[1])

  cdef double complex* w_out = <double complex*>(out[2])
  cdef double complex* vl_out = <double complex*>(out[3])
  cdef double complex* vr_out = <double complex*>(out[4])
  cdef int* info_out = <int*>(out[5])

  cdef char jobvlr = 'V'
  cdef double complex work_query
  cdef int lwork = -1
  zgeev(&jobvlr, &jobvlr, &n, a_work, &n, w_out, vl_out, &n,
        vr_out, &n, &work_query, &lwork, r_work, info_out)
  lwork = <int>(work_query.real)
  cdef double complex* work = <double complex*>malloc(
      lwork * sizeof(double complex))

  for i in range(b):
    memcpy(a_work, a_in, <int64_t>(n) * <int64_t>(n) * sizeof(double complex))
    zgeev(&jobvlr, &jobvlr, &n, a_work, &n, w_out, vl_out, &n, vr_out, &n,
          work, &lwork, r_work, info_out)

    a_in += n * n
    w_out += n
    vl_out += n * n
    vr_out += n * n
    info_out += 1
  free(work)

register_cpu_custom_call_target(b"lapack_zgeev", <void*>(lapack_zgeev))



def geev(c, a):
  assert sizeof(int32_t) == sizeof(int)

  a_shape = c.GetShape(a)
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

  out = c.CustomCallWithLayout(
      fn,
      operands=(c.ConstantS32Scalar(b), c.ConstantS32Scalar(n), a),
      shape_with_layout=Shape.tuple_shape(workspaces + eigvals + (
          Shape.array_shape(np.dtype(eigvecs_type), dims, layout),
          Shape.array_shape(np.dtype(eigvecs_type), dims, layout),
          Shape.array_shape(np.dtype(np.int32), batch_dims,
                            tuple(range(num_bd - 1, -1, -1))))
      ),
      operand_shapes_with_layout=(
          Shape.array_shape(np.dtype(np.int32), (), ()),
          Shape.array_shape(np.dtype(np.int32), (), ()),
          Shape.array_shape(dtype, dims, layout),
      ))
  if real:
    return (c.Complex(c.GetTupleElement(out, 3), c.GetTupleElement(out, 4)),
            c.GetTupleElement(out, 5), c.GetTupleElement(out, 6),
            c.GetTupleElement(out, 7))
  else:
    return (c.GetTupleElement(out, 2), c.GetTupleElement(out, 3),
            c.GetTupleElement(out, 4), c.GetTupleElement(out, 5))

def jax_geev(c, a):
  return c.Tuple(*geev(c, a))
