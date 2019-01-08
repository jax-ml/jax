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

from scipy.linalg.cython_blas cimport strsm, dtrsm, ctrsm
from scipy.linalg.cython_lapack cimport sgetrf, dgetrf, cgetrf
from scipy.linalg.cython_lapack cimport spotrf, dpotrf, cpotrf
from scipy.linalg.cython_lapack cimport sgesdd, dgesdd, cgesdd
from scipy.linalg.cython_lapack cimport ssyevd, dsyevd, cheevd

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
  cdef int lda = m
  cdef int ldb = m if left_side else n
  ctrsm(&cside, &cuplo, &ctransa, &cdiag, &m, &n, alpha, a, &lda, x, &ldb)

register_cpu_custom_call_target(b"blas_ctrsm", <void*>(blas_ctrsm))

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
  elif dtype == np.complex64:
    fn = b"blas_ctrsm"
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

  # cpotrf leaves junk in the part of the triangle that is not written; zero it.
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

register_cpu_custom_call_target(b"lapack_cpotrf", <void*>(lapack_cpotrf))

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


# ?gesdd: SVD decomposition

cdef void lapack_sgesdd(void* out_tuple, void** data) nogil:
  cdef int32_t job_opt_some = (<int32_t*>(data[0]))[0]
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

  if a_out != a_in:
    memcpy(a_out, a_in, m * n * sizeof(float))

  # define appropriate job code
  cdef char jobz = 'A'
  if job_opt_compute_uv == 0:
    jobz = 'N'
  else:
    if job_opt_some == 1:
      jobz = 'S'

  cdef int lda = m
  cdef int ldu = m
  cdef int ldvt = n

  cdef int* iwork = <int *> malloc(min(m, n) * sizeof(int))

  # First perform a workspace query to get the optimal lwork
  cdef float wkopt = 0
  cdef int lwork = -1
  sgesdd(&jobz, &m, &n, a_out, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork, iwork, info)
  lwork = <int> wkopt

  # Now get the actual SVD
  cdef work = <float *> malloc(lwork * sizeof(float))
  sgesdd(&jobz, &m, &n, a_out, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, info)

register_cpu_custom_call_target(b"lapack_sgesdd", <void*>(lapack_sgesdd))


cdef void lapack_dgesdd(void* out_tuple, void** data) nogil:
  cdef int32_t job_opt_some = (<int32_t*>(data[0]))[0]
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

  if a_out != a_in:
    memcpy(a_out, a_in, m * n * sizeof(double))

  # define appropriate job code
  cdef char jobz = 'A'
  if job_opt_compute_uv == 0:
    jobz = 'N'
  else:
    if job_opt_some == 1:
      jobz = 'S'

  cdef int lda = m
  cdef int ldu = m
  cdef int ldvt = n

  cdef int* iwork = <int *> malloc(min(m, n) * sizeof(int))

  # First perform a workspace query to get the optimal lwork
  cdef double wkopt = 0
  cdef int lwork = -1
  dgesdd(&jobz, &m, &n, a_out, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork, iwork, info)
  lwork = <int> wkopt

  # Now get the actual SVD
  cdef work = <double *> malloc(lwork * sizeof(double))
  dgesdd(&jobz, &m, &n, a_out, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, info)

register_cpu_custom_call_target(b"lapack_dgesdd", <void*>(lapack_dgesdd))

def jax_gesdd(c, a, full_matrices=True, compute_uv=True):
  assert sizeof(int32_t) == sizeof(int)

  a_shape = c.GetShape(a)
  dtype = a_shape.element_type()
  m, n = a_shape.dimensions()
  if dtype == np.float32:
    fn = b"lapack_sgesdd"
  elif dtype == np.float64:
    fn = b"lapack_dgesdd"
  else:
    raise NotImplementedError("Unsupported dtype {}".format(dtype))

  return c.CustomCall(
      fn,
      operands=(c.ConstantS32Scalar(int(not full_matrices)), c.ConstantS32Scalar(int(compute_uv)),
                c.ConstantS32Scalar(m), c.ConstantS32Scalar(n), a),
      shape_with_layout=Shape.tuple_shape((
          Shape.array_shape(dtype, (m, n), (0, 1)),
          Shape.array_shape(dtype, (min(m, n),), (0,)),
          Shape.array_shape(dtype, (m, m if full_matrices else min(m, n)), (0, 1)),
          Shape.array_shape(dtype, (n if full_matrices else min(m, n), n), (0, 1)),
          Shape.array_shape(np.int32, (), ()),
      )),
      operand_shapes_with_layout=(
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(np.int32, (), ()),
          Shape.array_shape(dtype, (m, n), (0, 1)),
      ))


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
