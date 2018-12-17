# distutils: language = c++

# Shims that allow the XLA CPU backend to call scipy-provided LAPACK kernels
# via CustomCall.

from __future__ import print_function

from libc.string cimport memcpy
from libcpp.string cimport string
from cpython.pycapsule cimport PyCapsule_New

from scipy.linalg.cython_lapack cimport spotrf, dpotrf

import numpy as np
from jaxlib import xla_client


cdef register_cpu_custom_call_target(fn_name, void* fn):
  cdef const char* name = "xla._CPU_CUSTOM_CALL_TARGET"
  xla_client.register_cpu_custom_call_target(
    fn_name, PyCapsule_New(fn, name, NULL))

# ?potrf (Cholesky decomposition)

cdef void lapack_spotrf(void* out_tuple, void** data) nogil:
  cdef bint lower = (<bint*>(data[0]))[0]
  cdef int n = (<int*>(data[1]))[0]
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

def jax_spotrf(c, a, lower=False):
  a_shape = c.GetShape(a)
  m, n = a_shape.dimensions()
  if m != n:
    raise ValueError("spotrf expects a square matrix, got {}".format(a_shape))
  return c.CustomCall(
      b"lapack_spotrf",
      operands=(c.ConstantPredScalar(lower), c.ConstantS32Scalar(n), a),
      shape_with_layout=xla_client.Shape.tuple_shape((
          xla_client.Shape.array_shape(np.float32, (n, n), (0, 1)),
          xla_client.Shape.array_shape(np.int32, (), ()),
      )),
      operand_shapes_with_layout=(
          xla_client.Shape.array_shape(np.bool, (), ()),
          xla_client.Shape.array_shape(np.int32, (), ()),
          xla_client.Shape.array_shape(np.float32, (n, n), (0, 1)),
      ))


cdef void lapack_dpotrf(void* out_tuple, void** data) nogil:
  cdef bint lower = (<bint*>(data[0]))[0]
  cdef int n = (<int*>(data[1]))[0]
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

def jax_dpotrf(c, a, lower=False):
  a_shape = c.GetShape(a)
  m, n = a_shape.dimensions()
  if m != n:
    raise ValueError("dpotrf expects a square matrix, got {}".format(a_shape))
  return c.CustomCall(
      b"lapack_dpotrf",
      operands=(c.ConstantPredScalar(lower), c.ConstantS32Scalar(n), a),
      shape_with_layout=xla_client.Shape.tuple_shape((
          xla_client.Shape.array_shape(np.float64, (n, n), (0, 1)),
          xla_client.Shape.array_shape(np.int32, (), ()),
      )),
      operand_shapes_with_layout=(
          xla_client.Shape.array_shape(np.bool, (), ()),
          xla_client.Shape.array_shape(np.int32, (), ()),
          xla_client.Shape.array_shape(np.float64, (n, n), (0, 1)),
      ))
