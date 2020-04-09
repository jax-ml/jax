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


import numpy as onp
from .. import lax
from . import lax_numpy as np

from jax import jit
from .lax_numpy import _wraps
from .linalg import eigvals as _eigvals
from .. import ops as jaxops


def _to_inexact_type(type):
  return type if np.issubdtype(type, np.inexact) else np.float_


def _promote_inexact(arr):
  return lax.convert_element_type(arr, _to_inexact_type(arr.dtype))


@jit
def _roots_no_zeros(p):
  # assume: p does not have leading zeros and has length > 1
  p = _promote_inexact(p)

  # build companion matrix and find its eigenvalues (the roots)
  A = np.diag(np.ones((p.size - 2,), p.dtype), -1)
  A = jaxops.index_update(A, jaxops.index[0, :], -p[1:] / p[0])
  roots = _eigvals(A)
  return roots


@jit
def _nonzero_range(arr):
  # return start and end s.t. arr[:start] = 0 = arr[end:] padding zeros
  is_zero = arr == 0
  start = np.argmin(is_zero)
  end = is_zero.size - np.argmin(is_zero[::-1])
  return start, end


@_wraps(onp.roots, lax_description="""\
If the input polynomial coefficients of length n do not start with zero,
the polynomial is of degree n - 1 leading to n - 1 roots. 
If the coefficients do have leading zeros, the polynomial they define
has a smaller degree and the number of roots (and thus the output shape) 
is value dependent.

The general implementation can therefore not be transformed with jit.
If the coefficients are guaranteed to have no leading zeros, use the 
keyword argument `strip_zeros=False` to get a jit-compatible variant::

    >>> roots_unsafe = jax.jit(jax.partial(np.roots, strip_zeros=False))
    >>> roots_unsafe([1, 2])     # ok
    DeviceArray([-2.+0.j], dtype=complex64)
    >>> roots_unsafe([0, 1, 2])  # problem
    DeviceArray([nan+nanj, nan+nanj], dtype=complex64)
    >>> np.roots([0, 1, 2])         # use the no-jit version instead
    DeviceArray([-2.+0.j], dtype=complex64)
""")
def roots(p, *, strip_zeros=True):
  # ported from https://github.com/numpy/numpy/blob/v1.17.0/numpy/lib/polynomial.py#L168-L251
  p = np.atleast_1d(p)
  if p.ndim != 1:
    raise ValueError("Input must be a rank-1 array.")

  # strip_zeros=False is unsafe because leading zeros aren't removed
  if not strip_zeros:
    if p.size > 1:
      return _roots_no_zeros(p)
    else:
      return np.array([])

  if np.all(p == 0):
    return np.array([])

  # factor out trivial roots
  start, end = _nonzero_range(p)
  # number of trailing zeros = number of roots at 0
  trailing_zeros = p.size - end

  # strip leading and trailing zeros
  p = p[start:end]

  if p.size < 2:
    return np.zeros(trailing_zeros, p.dtype)
  else:
    roots = _roots_no_zeros(p)
    # combine roots and zero roots
    roots = np.hstack((roots, np.zeros(trailing_zeros, p.dtype)))
    return roots
