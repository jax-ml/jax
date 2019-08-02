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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from jaxlib import xla_client

try:
  from jaxlib import cusolver_kernels
  for _name, _value in cusolver_kernels.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")
except ImportError:
  pass

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


def getrf(c, a):
  """LU decomposition."""
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

  lwork, opaque = cusolver_kernels.build_getrf_descriptor(
      np.dtype(dtype), b, m, n)
  out = c.CustomCall(
      b"cusolver_getrf",
      operands=(a,),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(
              dtype, batch_dims + (m, n),
              (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),
          _Shape.array_shape(dtype, (lwork,), (0,)),
          _Shape.array_shape(
              np.dtype(np.int32), batch_dims + (min(m, n),),
              tuple(range(num_bd, -1, -1))),
          _Shape.array_shape(
              np.dtype(np.int32), batch_dims, tuple(range(num_bd - 1, -1, -1))),
      )),
      operand_shapes_with_layout=(_Shape.array_shape(
          dtype, batch_dims + (m, n),
          (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))),),
      opaque=opaque)
  return (c.GetTupleElement(out, 0), c.GetTupleElement(out, 2),
          c.GetTupleElement(out, 3))


def syevd(c, a, lower=False):
  """Symmetric (Hermitian) eigendecomposition."""

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

  lwork, opaque = cusolver_kernels.build_syevd_descriptor(
      np.dtype(dtype), lower, b, n)
  eigvals_type = _real_type(dtype)

  out = c.CustomCall(
      b"cusolver_syevd",
      operands=(a,),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(dtype, dims, layout),
          _Shape.array_shape(
              np.dtype(eigvals_type), batch_dims + (n,),
              tuple(range(num_bd, -1, -1))),
          _Shape.array_shape(
              np.dtype(np.int32), batch_dims,
              tuple(range(num_bd - 1, -1, -1))),
          _Shape.array_shape(dtype, (lwork,), (0,))
      )),
      operand_shapes_with_layout=(
          _Shape.array_shape(dtype, dims, layout),
      ),
      opaque=opaque)
  return (c.GetTupleElement(out, 0), c.GetTupleElement(out, 1),
          c.GetTupleElement(out, 2))


def gesvd(c, a, full_matrices=True, compute_uv=True):
  """Singular value decomposition."""

  a_shape = c.GetShape(a)
  dtype = a_shape.element_type()
  b = 1
  m, n = a_shape.dimensions()
  singular_vals_dtype = _real_type(dtype)

  if m < n:
    lwork, opaque = cusolver_kernels.build_gesvd_descriptor(
        np.dtype(dtype), b, n, m, compute_uv, full_matrices)
    out = c.CustomCall(
        b"cusolver_gesvd",
        operands=(a,),
        shape_with_layout=_Shape.tuple_shape((
            _Shape.array_shape(dtype, (m, n), (1, 0)),
            _Shape.array_shape(np.dtype(singular_vals_dtype), (min(m, n),), (0,)),
            _Shape.array_shape(dtype, (n, n), (1, 0)),
            _Shape.array_shape(dtype, (m, m), (1, 0)),
            _Shape.array_shape(np.dtype(np.int32), (), ()),
            _Shape.array_shape(dtype, (lwork,), (0,)),
        )),
        operand_shapes_with_layout=(
            _Shape.array_shape(dtype, (m, n), (1, 0)),
        ),
        opaque=opaque)
    s = c.GetTupleElement(out, 1)
    vt = c.GetTupleElement(out, 2)
    u = c.GetTupleElement(out, 3)
    info = c.GetTupleElement(out, 4)
  else:
    lwork, opaque = cusolver_kernels.build_gesvd_descriptor(
        np.dtype(dtype), b, m, n, compute_uv, full_matrices)

    out = c.CustomCall(
        b"cusolver_gesvd",
        operands=(a,),
        shape_with_layout=_Shape.tuple_shape((
            _Shape.array_shape(dtype, (m, n), (0, 1)),
            _Shape.array_shape(np.dtype(singular_vals_dtype), (min(m, n),), (0,)),
            _Shape.array_shape(dtype, (m, m), (0, 1)),
            _Shape.array_shape(dtype, (n, n), (0, 1)),
            _Shape.array_shape(np.dtype(np.int32), (), ()),
            _Shape.array_shape(dtype, (lwork,), (0,)),
        )),
        operand_shapes_with_layout=(
            _Shape.array_shape(dtype, (m, n), (0, 1)),
        ),
        opaque=opaque)
    s = c.GetTupleElement(out, 1)
    u = c.GetTupleElement(out, 2)
    vt = c.GetTupleElement(out, 3)
    info = c.GetTupleElement(out, 4)
  if not full_matrices:
    u = c.Slice(u, (0, 0), (m, min(m, n)))
    vt = c.Slice(vt, (0, 0), (min(m, n), n))
  return s, u, vt, info
