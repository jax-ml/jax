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

from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t
from libc.string cimport memcpy
from libcpp.string cimport string
from cpython.pycapsule cimport PyCapsule_New

from scipy.linalg.cython_blas cimport strsm, dtrsm, ctrsm, ztrsm
from scipy.linalg.cython_lapack cimport sgetrf, dgetrf, cgetrf, zgetrf
from scipy.linalg.cython_lapack cimport spotrf, dpotrf, cpotrf, zpotrf
from scipy.linalg.cython_lapack cimport sgesdd, dgesdd, cgesdd, zgesdd
from scipy.linalg.cython_lapack cimport ssyevd, dsyevd, cheevd, zheevd

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
  cdef int lda = m if left_side else n
  cdef int ldb = m
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
  cdef int lda = m if left_side else n
  cdef int ldb = m
  dtrsm(&cside, &cuplo, &ctransa, &cdiag, &m, &n, alpha, a, &lda, x, &ldb)

register_cpu_custom_call_target(b"blas_dtrsm", <void*>(blas_dtrsm))


cdef void blas_ctrsm(void* out, void** data) nogil:
  cdef int32_t left_side = (<int32_t*>(data[0]))[0]
  cdef int32_t lower = (<int32_t*>(data[1]))[0]
  cdef int32_t trans_a = (<int32_t*>(data[2]))[0]
  cdef int32_t diag = (<int32_t*>(data[3]))[0]
  cdef int m = (<int32_t*>(data[4]))[0]
  cdef int n = (<int32_t*>(data[5]))[0]
  cdef float complex* alpha = <float complex*>(data[6])
  cdef float complex* a = <float complex*>(data[7])
  cdef float complex* b = <float complex*>(data[8])

  cdef float complex* x = <float complex*>(out)
  if x != b:
    memcpy(x, b, m * n * sizeof(float complex))

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
  ctrsm(&cside, &cuplo, &ctransa, &cdiag, &m, &n, alpha, a, &lda, x, &ldb)

register_cpu_custom_call_target(b"blas_ctrsm", <void*>(blas_ctrsm))

cdef void blas_ztrsm(void* out, void** data) nogil:
  cdef int32_t left_side = (<int32_t*>(data[0]))[0]
  cdef int32_t lower = (<int32_t*>(data[1]))[0]
  cdef int32_t trans_a = (<int32_t*>(data[2]))[0]
  cdef int32_t diag = (<int32_t*>(data[3]))[0]
  cdef int m = (<int32_t*>(data[4]))[0]
  cdef int n = (<int32_t*>(data[5]))[0]
  cdef double complex* alpha = <double complex*>(data[6])
  cdef double complex* a = <double complex*>(data[7])
  cdef double complex* b = <double complex*>(data[8])

  cdef double complex* x = <double complex*>(out)
  if x != b:
    memcpy(x, b, m * n * sizeof(double complex))

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
  ztrsm(&cside, &cuplo, &ctransa, &cdiag, &m, &n, alpha, a, &lda, x, &ldb)

register_cpu_custom_call_target(b"blas_ztrsm", <void*>(blas_ztrsm))


def jax_trsm(c, alpha, a, b, left_side=False, lower=False, trans_a=False,
             conj_a=False, diag=False):
  b_shape = c.GetShape(b)
  dtype = b_shape.element_type()
  m, n = b_shape.dimensions()
  k = m if left_side else n

  a_shape = c.GetShape(a)
  if (k, k) != a_shape.dimensions() or a_shape.element_type() != dtype:
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

  return c.CustomCall(
      fn,
      operands=(
        c.ConstantS32Scalar(int(left_side)),
        c.ConstantS32Scalar(int(lower)),
        c.ConstantS32Scalar((2 if conj_a else 1) if trans_a else 0),
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
  cdef int b = (<int32_t*>(data[0]))[0]
  cdef int m = (<int32_t*>(data[1]))[0]
  cdef int n = (<int32_t*>(data[2]))[0]
  cdef const float* a_in = <float*>(data[3])

  cdef void** out = <void**>(out_tuple)
  cdef float* a_out = <float*>(out[0])
  cdef int* ipiv = <int*>(out[1])
  cdef int* info = <int*>(out[2])
  if a_out != a_in:
    memcpy(a_out, a_in, b * m * n * sizeof(float))

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
    memcpy(a_out, a_in, b * m * n * sizeof(double))

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
    memcpy(a_out, a_in, b * m * n * sizeof(float complex))

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
    memcpy(a_out, a_in, b * m * n * sizeof(double complex))

  for i in range(b):
    zgetrf(&m, &n, a_out, &m, ipiv, info)
    a_out += m * n
    ipiv += min(m, n)
    info += 1

register_cpu_custom_call_target(b"lapack_zgetrf", <void*>(lapack_zgetrf))

def jax_getrf(c, a):
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

  return c.CustomCall(
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
            np.int32,
            batch_dims + (min(m, n),),
            tuple(range(num_bd, -1, -1))),
          Shape.array_shape(np.int32, batch_dims,
            tuple(range(num_bd - 1, -1, -1))),
      )),
      operand_shapes_with_layout=(
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(
            dtype,
            batch_dims + (m, n),
            (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
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

register_cpu_custom_call_target(b"lapack_dpotrf", <void*>(lapack_dpotrf))


cdef void lapack_cpotrf(void* out_tuple, void** data) nogil:
  cdef int32_t lower = (<int32_t*>(data[0]))[0]
  cdef int n = (<int32_t*>(data[1]))[0]
  cdef const float complex* a_in = <float complex*>(data[2])
  cdef char uplo = 'L' if lower else 'U'

  cdef void** out = <void**>(out_tuple)
  cdef float complex* a_out = <float complex*>(out[0])
  cdef int* info = <int*>(out[1])
  if a_out != a_in:
    memcpy(a_out, a_in, n * n * sizeof(float complex))

  cpotrf(&uplo, &n, a_out, &n, info)

register_cpu_custom_call_target(b"lapack_cpotrf", <void*>(lapack_cpotrf))

cdef void lapack_zpotrf(void* out_tuple, void** data) nogil:
  cdef int32_t lower = (<int32_t*>(data[0]))[0]
  cdef int n = (<int32_t*>(data[1]))[0]
  cdef const double complex* a_in = <double complex*>(data[2])
  cdef char uplo = 'L' if lower else 'U'

  cdef void** out = <void**>(out_tuple)
  cdef double complex* a_out = <double complex*>(out[0])
  cdef int* info = <int*>(out[1])
  if a_out != a_in:
    memcpy(a_out, a_in, n * n * sizeof(double complex))

  zpotrf(&uplo, &n, a_out, &n, info)

register_cpu_custom_call_target(b"lapack_zpotrf", <void*>(lapack_zpotrf))

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
  elif dtype == np.complex64:
    fn = b"lapack_cpotrf"
  elif dtype == np.complex128:
    fn = b"lapack_zpotrf"
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


# ?gesdd: Singular value decomposition

cdef int gesdd_iwork_size(int m, int n) nogil:
  return 8 * min(m, n)

cdef int cgesdd_rwork_size(int m, int n, int compute_uv) nogil:
  cdef int mn = min(m, n)
  if compute_uv == 0:
    return 7 * mn
  cdef int mx = max(m, n)
  return max(5 * mn * mn + 5 * mn, 2 * mx * mn + 2 * mn * mn + mn)

cdef void lapack_sgesdd(void* out_tuple, void** data) nogil:
  cdef int32_t job_opt_full_matrices = (<int32_t*>(data[0]))[0]
  cdef int32_t job_opt_compute_uv = (<int32_t*>(data[1]))[0]
  cdef int m = (<int32_t*>(data[2]))[0]
  cdef int n = (<int32_t*>(data[3]))[0]
  cdef float* a_in = <float*>(data[4])

  cdef void** out = <void**>(out_tuple)
  cdef float* a_out = <float*>(out[0])
  cdef float* s = <float*>(out[1])
  cdef float* u = <float*>(out[2])
  cdef float* vt = <float*>(out[3])
  cdef int* info = <int*>(out[4])
  cdef int* iwork = <int*>(out[5])

  if a_out != a_in:
    memcpy(a_out, a_in, m * n * sizeof(float))

  # define appropriate job code
  cdef char jobz = 'A'
  if job_opt_compute_uv == 0:
    jobz = 'N'
  else:
    if job_opt_full_matrices == 0:
      jobz = 'S'

  cdef int lda = m
  cdef int ldu = m
  cdef int ldvt = n
  if job_opt_full_matrices == 0:
    ldvt = min(m, n)

  # First perform a workspace query to get the optimal lwork
  # NB: We perform a workspace query with malloc and free for the work array, 
  # because it is officially recommended in the LAPACK documentation
  cdef float wkopt = 0
  cdef int lwork = -1
  sgesdd(&jobz, &m, &n, a_out, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork, iwork, info)
  lwork = <int> wkopt

  # Now get the actual SVD
  cdef float* work = <float *> malloc(lwork * sizeof(float))
  sgesdd(&jobz, &m, &n, a_out, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, info)
  free(work)

register_cpu_custom_call_target(b"lapack_sgesdd", <void*>(lapack_sgesdd))


cdef void lapack_dgesdd(void* out_tuple, void** data) nogil:
  cdef int32_t job_opt_full_matrices = (<int32_t*>(data[0]))[0]
  cdef int32_t job_opt_compute_uv = (<int32_t*>(data[1]))[0]
  cdef int m = (<int32_t*>(data[2]))[0]
  cdef int n = (<int32_t*>(data[3]))[0]
  cdef double* a_in = <double*>(data[4])

  cdef void** out = <void**>(out_tuple)
  cdef double* a_out = <double*>(out[0])
  cdef double* s = <double*>(out[1])
  cdef double* u = <double*>(out[2])
  cdef double* vt = <double*>(out[3])
  cdef int* info = <int*>(out[4])
  cdef int* iwork = <int*>(out[5])

  if a_out != a_in:
    memcpy(a_out, a_in, m * n * sizeof(double))

  # define appropriate job code
  cdef char jobz = 'A'
  if job_opt_compute_uv == 0:
    jobz = 'N'
  else:
    if job_opt_full_matrices == 0:
      jobz = 'S'

  cdef int lda = m
  cdef int ldu = m
  cdef int ldvt = n
  if job_opt_full_matrices == 0:
    ldvt = min(m, n)

  # First perform a workspace query to get the optimal lwork
  # NB: We perform a workspace query with malloc and free for the work array, 
  # because it is officially recommended in the LAPACK documentation
  cdef double wkopt = 0
  cdef int lwork = -1
  dgesdd(&jobz, &m, &n, a_out, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork, iwork, info)
  lwork = <int> wkopt

  # Now get the actual SVD
  cdef double* work = <double *> malloc(lwork * sizeof(double))
  dgesdd(&jobz, &m, &n, a_out, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, info)
  free(work)

register_cpu_custom_call_target(b"lapack_dgesdd", <void*>(lapack_dgesdd))


cdef void lapack_cgesdd(void* out_tuple, void** data) nogil:
  cdef int32_t job_opt_full_matrices = (<int32_t*>(data[0]))[0]
  cdef int32_t job_opt_compute_uv = (<int32_t*>(data[1]))[0]
  cdef int m = (<int32_t*>(data[2]))[0]
  cdef int n = (<int32_t*>(data[3]))[0]
  cdef float complex* a_in = <float complex*>(data[4])

  cdef void** out = <void**>(out_tuple)
  cdef float complex* a_out = <float complex*>(out[0])
  cdef float* s = <float*>(out[1])
  cdef float complex* u = <float complex*>(out[2])
  cdef float complex* vt = <float complex*>(out[3])
  cdef int* info = <int*>(out[4])
  cdef int* iwork = <int*>(out[5])
  cdef float* rwork = <float*>(out[6])

  if a_out != a_in:
    memcpy(a_out, a_in, m * n * sizeof(float complex))

  # define appropriate job code
  cdef char jobz = 'A'
  if job_opt_compute_uv == 0:
    jobz = 'N'
  else:
    if job_opt_full_matrices == 0:
      jobz = 'S'

  cdef int lda = m
  cdef int ldu = m
  cdef int ldvt = n
  if job_opt_full_matrices == 0:
    ldvt = min(m, n)

  # First perform a workspace query to get the optimal lwork
  # NB: We perform a workspace query with malloc and free for the work array,
  # because it is officially recommended in the LAPACK documentation
  cdef float complex wkopt = 0
  cdef int lwork = -1
  cgesdd(&jobz, &m, &n, a_out, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork, rwork, iwork, info)
  lwork = <int>(wkopt.real)

  # Now get the actual SVD
  cdef float complex* work = <float complex*> malloc(lwork * sizeof(float complex))
  cgesdd(&jobz, &m, &n, a_out, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, iwork, info)
  free(work)

register_cpu_custom_call_target(b"lapack_cgesdd", <void*>(lapack_cgesdd))


cdef void lapack_zgesdd(void* out_tuple, void** data) nogil:
  cdef int32_t job_opt_full_matrices = (<int32_t*>(data[0]))[0]
  cdef int32_t job_opt_compute_uv = (<int32_t*>(data[1]))[0]
  cdef int m = (<int32_t*>(data[2]))[0]
  cdef int n = (<int32_t*>(data[3]))[0]
  cdef double complex* a_in = <double complex*>(data[4])

  cdef void** out = <void**>(out_tuple)
  cdef double complex* a_out = <double complex*>(out[0])
  cdef double* s = <double*>(out[1])
  cdef double complex* u = <double complex*>(out[2])
  cdef double complex* vt = <double complex*>(out[3])
  cdef int* info = <int*>(out[4])
  cdef int* iwork = <int*>(out[5])
  cdef double* rwork = <double*>(out[6])

  if a_out != a_in:
    memcpy(a_out, a_in, m * n * sizeof(double complex))

  # define appropriate job code
  cdef char jobz = 'A'
  if job_opt_compute_uv == 0:
    jobz = 'N'
  else:
    if job_opt_full_matrices == 0:
      jobz = 'S'

  cdef int lda = m
  cdef int ldu = m
  cdef int ldvt = n
  if job_opt_full_matrices == 0:
    ldvt = min(m, n)

  # First perform a workspace query to get the optimal lwork
  # NB: We perform a workspace query with malloc and free for the work array,
  # because it is officially recommended in the LAPACK documentation
  cdef double complex wkopt = 0
  cdef int lwork = -1
  zgesdd(&jobz, &m, &n, a_out, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork, rwork, iwork, info)
  lwork = <int>(wkopt.real)

  # Now get the actual SVD
  cdef double complex* work = <double complex*> malloc(lwork * sizeof(double complex))
  zgesdd(&jobz, &m, &n, a_out, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, iwork, info)
  free(work)

register_cpu_custom_call_target(b"lapack_zgesdd", <void*>(lapack_zgesdd))

def jax_gesdd(c, a, full_matrices=True, compute_uv=True):
  assert sizeof(int32_t) == sizeof(int)

  a_shape = c.GetShape(a)
  dtype = a_shape.element_type()
  m, n = a_shape.dimensions()
  if dtype == np.float32:
    fn = b"lapack_sgesdd"
    singular_vals_dtype = np.float32
    workspace = (Shape.array_shape(np.int32, (gesdd_iwork_size(m, n),), (0,)),)
  elif dtype == np.float64:
    fn = b"lapack_dgesdd"
    singular_vals_dtype = np.float64
    workspace = (Shape.array_shape(np.int32, (gesdd_iwork_size(m, n),), (0,)),)
  elif dtype == np.complex64:
    fn = b"lapack_cgesdd"
    singular_vals_dtype = np.float32
    workspace = (Shape.array_shape(np.int32, (gesdd_iwork_size(m, n),), (0,)),
                 Shape.array_shape(np.float32, (cgesdd_rwork_size(m, n, int(compute_uv)),), (0,)))
  elif dtype == np.complex128:
    fn = b"lapack_zgesdd"
    singular_vals_dtype = np.float64
    workspace = (Shape.array_shape(np.int32, (gesdd_iwork_size(m, n),), (0,)),
                 Shape.array_shape(np.float64, (cgesdd_rwork_size(m, n, int(compute_uv)),), (0,)))
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  out = c.CustomCall(
      fn,
      operands=(c.ConstantS32Scalar(int(full_matrices)), c.ConstantS32Scalar(int(compute_uv)),
                c.ConstantS32Scalar(m), c.ConstantS32Scalar(n), a),
      shape_with_layout=Shape.tuple_shape((
          Shape.array_shape(dtype, (m, n), (0, 1)),
          Shape.array_shape(singular_vals_dtype, (min(m, n),), (0,)),
          Shape.array_shape(dtype, (m, m if full_matrices else min(m, n)), (0, 1)),
          Shape.array_shape(dtype, (n if full_matrices else min(m, n), n), (0, 1)),
          Shape.array_shape(np.int32, (), ())) + workspace
      ),
      operand_shapes_with_layout=(
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(dtype, (m, n), (0, 1)),
      ))
  return c.Tuple(c.GetTupleElement(out, 1), c.GetTupleElement(out, 2),
                 c.GetTupleElement(out, 3), c.GetTupleElement(out, 4))


# syevd: Symmetric eigendecomposition

# Workspace sizes, taken from the LAPACK documentation.
cdef int syevd_work_size(int n) nogil:
  return 1 + 6 * n + 2 * n * n

cdef int syevd_iwork_size(int n) nogil:
  return 3 + 5 * n

cdef void lapack_ssyevd(void* out_tuple, void** data) nogil:
  cdef int32_t lower = (<int32_t*>(data[0]))[0]
  cdef int n = (<int32_t*>(data[1]))[0]
  cdef const float* a_in = <float*>(data[2])

  cdef void** out = <void**>(out_tuple)
  cdef float* a_out = <float*>(out[0])
  cdef float* w_out = <float*>(out[1])
  cdef int* info_out = <int*>(out[2])
  cdef float* work = <float*>(out[3])
  cdef int* iwork = <int*>(out[4])
  if a_out != a_in:
    memcpy(a_out, a_in, n * n * sizeof(float))

  cdef char jobz = 'V'
  cdef char uplo = 'L' if lower else 'U'

  cdef int lwork = syevd_work_size(n)
  cdef int liwork = syevd_iwork_size(n)
  ssyevd(&jobz, &uplo, &n, a_out, &n, w_out, work, &lwork, iwork, &liwork,
         info_out)

register_cpu_custom_call_target(b"lapack_ssyevd", <void*>(lapack_ssyevd))

cdef void lapack_dsyevd(void* out_tuple, void** data) nogil:
  cdef int32_t lower = (<int32_t*>(data[0]))[0]
  cdef int n = (<int32_t*>(data[1]))[0]
  cdef const double* a_in = <double*>(data[2])

  cdef void** out = <void**>(out_tuple)
  cdef double* a_out = <double*>(out[0])
  cdef double* w_out = <double*>(out[1])
  cdef int* info_out = <int*>(out[2])
  cdef double* work = <double*>(out[3])
  cdef int* iwork = <int*>(out[4])
  if a_out != a_in:
    memcpy(a_out, a_in, n * n * sizeof(double))

  cdef char jobz = 'V'
  cdef char uplo = 'L' if lower else 'U'

  cdef int lwork = syevd_work_size(n)
  cdef int liwork = syevd_iwork_size(n)
  dsyevd(&jobz, &uplo, &n, a_out, &n, w_out, work, &lwork, iwork, &liwork,
         info_out)

register_cpu_custom_call_target(b"lapack_dsyevd", <void*>(lapack_dsyevd))

# Workspace sizes, taken from the LAPACK documentation.
cdef int heevd_work_size(int n) nogil:
  return 1 + 2 * n + n * n

cdef int heevd_rwork_size(int n) nogil:
  return 1 + 5 * n + 2 * n * n


cdef void lapack_cheevd(void* out_tuple, void** data) nogil:
  cdef int32_t lower = (<int32_t*>(data[0]))[0]
  cdef int n = (<int32_t*>(data[1]))[0]
  cdef const float complex* a_in = <float complex*>(data[2])

  cdef void** out = <void**>(out_tuple)
  cdef float complex* a_out = <float complex*>(out[0])
  cdef float* w_out = <float*>(out[1])
  cdef int* info_out = <int*>(out[2])
  cdef float complex* work = <float complex*>(out[3])
  cdef float* rwork = <float*>(out[4])
  cdef int* iwork = <int*>(out[5])
  if a_out != a_in:
    memcpy(a_out, a_in, n * n * sizeof(float complex))

  cdef char jobz = 'V'
  cdef char uplo = 'L' if lower else 'U'

  cdef int lwork = heevd_work_size(n)
  cdef int lrwork = heevd_rwork_size(n)
  cdef int liwork = syevd_iwork_size(n)
  cheevd(&jobz, &uplo, &n, a_out, &n, w_out, work, &lwork, rwork, &lrwork,
         iwork, &liwork, info_out)

register_cpu_custom_call_target(b"lapack_cheevd", <void*>(lapack_cheevd))


cdef void lapack_zheevd(void* out_tuple, void** data) nogil:
  cdef int32_t lower = (<int32_t*>(data[0]))[0]
  cdef int n = (<int32_t*>(data[1]))[0]
  cdef const double complex* a_in = <double complex*>(data[2])

  cdef void** out = <void**>(out_tuple)
  cdef double complex* a_out = <double complex*>(out[0])
  cdef double* w_out = <double*>(out[1])
  cdef int* info_out = <int*>(out[2])
  cdef double complex* work = <double complex*>(out[3])
  cdef double* rwork = <double*>(out[4])
  cdef int* iwork = <int*>(out[5])
  if a_out != a_in:
    memcpy(a_out, a_in, n * n * sizeof(double complex))

  cdef char jobz = 'V'
  cdef char uplo = 'L' if lower else 'U'

  cdef int lwork = heevd_work_size(n)
  cdef int lrwork = heevd_rwork_size(n)
  cdef int liwork = syevd_iwork_size(n)
  zheevd(&jobz, &uplo, &n, a_out, &n, w_out, work, &lwork, rwork, &lrwork,
         iwork, &liwork, info_out)

register_cpu_custom_call_target(b"lapack_zheevd", <void*>(lapack_zheevd))

def jax_syevd(c, a, lower=False):
  assert sizeof(int32_t) == sizeof(int)

  a_shape = c.GetShape(a)
  dtype = a_shape.element_type()
  m, n = a_shape.dimensions()
  if dtype == np.float32:
    fn = b"lapack_ssyevd"
    eigvals_type = np.float32
    workspace = (Shape.array_shape(dtype, (syevd_work_size(n),), (0,)),
                 Shape.array_shape(np.int32, (syevd_iwork_size(n),), (0,)))
  elif dtype == np.float64:
    fn = b"lapack_dsyevd"
    eigvals_type = np.float64
    workspace = (Shape.array_shape(dtype, (syevd_work_size(n),), (0,)),
                 Shape.array_shape(np.int32, (syevd_iwork_size(n),), (0,)))
  elif dtype == np.complex64:
    fn = b"lapack_cheevd"
    eigvals_type = np.float32
    workspace = (Shape.array_shape(dtype, (heevd_work_size(n),), (0,)),
                 Shape.array_shape(np.float32, (heevd_rwork_size(n),), (0,)),
                 Shape.array_shape(np.int32, (syevd_iwork_size(n),), (0,)))
  elif dtype == np.complex128:
    fn = b"lapack_zheevd"
    eigvals_type = np.float64
    workspace = (Shape.array_shape(dtype, (heevd_work_size(n),), (0,)),
                 Shape.array_shape(np.float64, (heevd_rwork_size(n),), (0,)),
                 Shape.array_shape(np.int32, (syevd_iwork_size(n),), (0,)))
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  out = c.CustomCall(
      fn,
      operands=(c.ConstantS32Scalar(1 if lower else 0),
                c.ConstantS32Scalar(n),
                a),
      shape_with_layout=Shape.tuple_shape((
          Shape.array_shape(dtype, (n, n), (0, 1)),
          Shape.array_shape(eigvals_type, (n,), (0,)),
          Shape.array_shape(np.int32, (), ())) + workspace
      ),
      operand_shapes_with_layout=(
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(dtype, (n, n), (0, 1)),
      ))
  return c.Tuple(c.GetTupleElement(out, 0), c.GetTupleElement(out, 1),
                 c.GetTupleElement(out, 2))
