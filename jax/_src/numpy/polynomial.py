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


from functools import partial
import operator

from jax import core
from jax import jit
from jax import lax
from jax._src.numpy.lax_numpy import (
    all, arange, argmin, array, asarray, atleast_1d, concatenate, convolve, diag, dot, finfo,
    full, hstack, maximum, ones, outer, sqrt, trim_zeros, trim_zeros_tol, true_divide, vander, zeros)
from jax._src.numpy import linalg
from jax._src.numpy.util import _check_arraylike, _promote_dtypes, _promote_dtypes_inexact, _wraps
import numpy as np


@jit
def _roots_no_zeros(p):
  # assume: p does not have leading zeros and has length > 1
  p, = _promote_dtypes_inexact(p)

  # build companion matrix and find its eigenvalues (the roots)
  A = diag(ones((p.size - 2,), p.dtype), -1)
  A = A.at[0, :].set(-p[1:] / p[0])
  roots = linalg.eigvals(A)
  return roots


@jit
def _nonzero_range(arr):
  # return start and end s.t. arr[:start] = 0 = arr[end:] padding zeros
  is_zero = arr == 0
  start = argmin(is_zero)
  end = is_zero.size - argmin(is_zero[::-1])
  return start, end


@_wraps(np.roots, lax_description="""\
If the input polynomial coefficients of length n do not start with zero,
the polynomial is of degree n - 1 leading to n - 1 roots.
If the coefficients do have leading zeros, the polynomial they define
has a smaller degree and the number of roots (and thus the output shape)
is value dependent.

The general implementation can therefore not be transformed with jit.
If the coefficients are guaranteed to have no leading zeros, use the
keyword argument `strip_zeros=False` to get a jit-compatible variant:

>>> from functools import partial
>>> roots_unsafe = jax.jit(partial(jnp.roots, strip_zeros=False))
>>> roots_unsafe([1, 2])     # ok
DeviceArray([-2.+0.j], dtype=complex64)
>>> roots_unsafe([0, 1, 2])  # problem
DeviceArray([nan+nanj, nan+nanj], dtype=complex64)
>>> jnp.roots([0, 1, 2])     # use the no-jit version instead
DeviceArray([-2.+0.j], dtype=complex64)
""")
def roots(p, *, strip_zeros=True):
  # ported from https://github.com/numpy/numpy/blob/v1.17.0/numpy/lib/polynomial.py#L168-L251
  p = atleast_1d(p)
  if p.ndim != 1:
    raise ValueError("Input must be a rank-1 array.")

  # strip_zeros=False is unsafe because leading zeros aren't removed
  if not strip_zeros:
    if p.size > 1:
      return _roots_no_zeros(p)
    else:
      return array([])

  if all(p == 0):
    return array([])

  # factor out trivial roots
  start, end = _nonzero_range(p)
  # number of trailing zeros = number of roots at 0
  trailing_zeros = p.size - end

  # strip leading and trailing zeros
  p = p[start:end]

  if p.size < 2:
    return zeros(trailing_zeros, p.dtype)
  else:
    roots = _roots_no_zeros(p)
    # combine roots and zero roots
    roots = hstack((roots, zeros(trailing_zeros, p.dtype)))
    return roots


_POLYFIT_DOC = """\
Unlike NumPy's implementation of polyfit, :py:func:`jax.numpy.polyfit` will not warn on rank reduction, which indicates an ill conditioned matrix
Also, it works best on rcond <= 10e-3 values.
"""
@_wraps(np.polyfit, lax_description=_POLYFIT_DOC)
@partial(jit, static_argnames=('deg', 'rcond', 'full', 'cov'))
def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
  _check_arraylike("polyfit", x, y)
  deg = core.concrete_or_error(int, deg, "deg must be int")
  order = deg + 1
  # check arguments
  if deg < 0:
    raise ValueError("expected deg >= 0")
  if x.ndim != 1:
    raise TypeError("expected 1D vector for x")
  if x.size == 0:
    raise TypeError("expected non-empty vector for x")
  if y.ndim < 1 or y.ndim > 2:
    raise TypeError("expected 1D or 2D array for y")
  if x.shape[0] != y.shape[0]:
    raise TypeError("expected x and y to have same length")

  # set rcond
  if rcond is None:
    rcond = len(x) * finfo(x.dtype).eps
  rcond = core.concrete_or_error(float, rcond, "rcond must be float")
  # set up least squares equation for powers of x
  lhs = vander(x, order)
  rhs = y

  # apply weighting
  if w is not None:
    _check_arraylike("polyfit", w)
    w, = _promote_dtypes_inexact(w)
    if w.ndim != 1:
      raise TypeError("expected a 1-d array for weights")
    if w.shape[0] != y.shape[0]:
      raise TypeError("expected w and y to have the same length")
    lhs *= w[:, np.newaxis]
    if rhs.ndim == 2:
      rhs *= w[:, np.newaxis]
    else:
      rhs *= w

  # scale lhs to improve condition number and solve
  scale = sqrt((lhs*lhs).sum(axis=0))
  lhs /= scale[np.newaxis,:]
  c, resids, rank, s = linalg.lstsq(lhs, rhs, rcond)
  c = (c.T/scale).T  # broadcast scale coefficients

  if full:
    return c, resids, rank, s, rcond
  elif cov:
    Vbase = linalg.inv(dot(lhs.T, lhs))
    Vbase /= outer(scale, scale)
    if cov == "unscaled":
      fac = 1
    else:
      if len(x) <= order:
        raise ValueError("the number of data points must exceed order "
                            "to scale the covariance matrix")
      fac = resids / (len(x) - order)
      fac = fac[0] #making np.array() of shape (1,) to int
    if y.ndim == 1:
      return c, Vbase * fac
    else:
      return c, Vbase[:, :, np.newaxis] * fac
  else:
    return c


_POLY_DOC = """\
This differs from np.poly when an integer array is given.
np.poly returns a result with dtype float64 in this case.
jax returns a result with an inexact type, but not necessarily
float64.

This also differs from np.poly when the input array strictly
contains pairs of complex conjugates, e.g. [1j, -1j, 1-1j, 1+1j].
np.poly returns an array with a real dtype in such cases.
jax returns an array with a complex dtype in such cases.
"""

@_wraps(np.poly, lax_description=_POLY_DOC)
@jit
def poly(seq_of_zeros):
  _check_arraylike('poly', seq_of_zeros)
  seq_of_zeros, = _promote_dtypes_inexact(seq_of_zeros)
  seq_of_zeros = atleast_1d(seq_of_zeros)

  sh = seq_of_zeros.shape
  if len(sh) == 2 and sh[0] == sh[1] and sh[0] != 0:
    # import at runtime to avoid circular import
    from jax._src.numpy import linalg
    seq_of_zeros = linalg.eigvals(seq_of_zeros)

  if seq_of_zeros.ndim != 1:
    raise ValueError("input must be 1d or non-empty square 2d array.")

  dt = seq_of_zeros.dtype
  if len(seq_of_zeros) == 0:
    return ones((), dtype=dt)

  a = ones((1,), dtype=dt)
  for k in range(len(seq_of_zeros)):
    a = convolve(a, array([1, -seq_of_zeros[k]], dtype=dt), mode='full')

  return a


@_wraps(np.polyval, lax_description="""\
The ``unroll`` parameter is JAX specific. It does not effect correctness but can
have a major impact on performance for evaluating high-order polynomials. The
parameter controls the number of unrolled steps with ``lax.scan`` inside the
``polyval`` implementation. Consider setting ``unroll=128`` (or even higher) to
improve runtime performance on accelerators, at the cost of increased
compilation time.
""")
@partial(jit, static_argnames=['unroll'])
def polyval(p, x, *, unroll=16):
  _check_arraylike("polyval", p, x)
  p, x = _promote_dtypes_inexact(p, x)
  shape = lax.broadcast_shapes(p.shape[1:], x.shape)
  y = lax.full_like(x, 0, shape=shape, dtype=x.dtype)
  y, _ = lax.scan(lambda y, p: (y * x + p, None), y, p, unroll=unroll)
  return y

@_wraps(np.polyadd)
@jit
def polyadd(a1, a2):
  _check_arraylike("polyadd", a1, a2)
  a1, a2 = _promote_dtypes(a1, a2)
  if a2.shape[0] <= a1.shape[0]:
    return a1.at[-a2.shape[0]:].add(a2)
  else:
    return a2.at[-a1.shape[0]:].add(a1)


@_wraps(np.polyint)
@partial(jit, static_argnames=('m',))
def polyint(p, m=1, k=None):
  m = core.concrete_or_error(operator.index, m, "'m' argument of jnp.polyint")
  k = 0 if k is None else k
  _check_arraylike("polyint", p, k)
  p, k = _promote_dtypes_inexact(p, k)
  if m < 0:
    raise ValueError("Order of integral must be positive (see polyder)")
  k = atleast_1d(k)
  if len(k) == 1:
    k = full((m,), k[0])
  if k.shape != (m,):
    raise ValueError("k must be a scalar or a rank-1 array of length 1 or m.")
  if m == 0:
    return p
  else:
    coeff = maximum(1, arange(len(p) + m, 0, -1)[np.newaxis, :] - 1 - arange(m)[:, np.newaxis]).prod(0)
    return true_divide(concatenate((p, k)), coeff)


@_wraps(np.polyder)
@partial(jit, static_argnames=('m',))
def polyder(p, m=1):
  _check_arraylike("polyder", p)
  m = core.concrete_or_error(operator.index, m, "'m' argument of jnp.polyder")
  p, = _promote_dtypes_inexact(p)
  if m < 0:
    raise ValueError("Order of derivative must be positive")
  if m == 0:
    return p
  coeff = (arange(len(p), m, -1)[np.newaxis, :] - 1 - arange(m)[:, np.newaxis]).prod(0)
  return p[:-m] * coeff


_LEADING_ZEROS_DOC = """\
Setting trim_leading_zeros=True makes the output match that of numpy.
But prevents the function from being able to be used in compiled code.
Due to differences in accumulation of floating point arithmetic errors, the cutoff for values to be
considered zero may lead to inconsistent results between NumPy and JAX, and even between different
JAX backends. The result may lead to inconsistent output shapes when trim_leading_zeros=True.
"""

@_wraps(np.polymul, lax_description=_LEADING_ZEROS_DOC)
def polymul(a1, a2, *, trim_leading_zeros=False):
  _check_arraylike("polymul", a1, a2)
  a1, a2 = _promote_dtypes_inexact(a1, a2)
  if trim_leading_zeros and (len(a1) > 1 or len(a2) > 1):
    a1, a2 = trim_zeros(a1, trim='f'), trim_zeros(a2, trim='f')
  if len(a1) == 0:
    a1 = asarray([0.])
  if len(a2) == 0:
    a2 = asarray([0.])
  val = convolve(a1, a2, mode='full')
  return val

@_wraps(np.polydiv, lax_description=_LEADING_ZEROS_DOC)
def polydiv(u, v, *, trim_leading_zeros=False):
  _check_arraylike("polydiv", u, v)
  u, v = _promote_dtypes_inexact(u, v)
  m = len(u) - 1
  n = len(v) - 1
  scale = 1. / v[0]
  q = zeros(max(m - n + 1, 1), dtype = u.dtype) # force same dtype
  for k in range(0, m-n+1):
    d = scale * u[k]
    q = q.at[k].set(d)
    u = u.at[k:k+n+1].add(-d*v)
  if trim_leading_zeros:
    # use the square root of finfo(dtype) to approximate the absolute tolerance used in numpy
    return q, trim_zeros_tol(u, tol=sqrt(finfo(u.dtype).eps), trim='f')
  else:
    return q, u

@_wraps(np.polysub)
@jit
def polysub(a1, a2):
  _check_arraylike("polysub", a1, a2)
  a1, a2 = _promote_dtypes(a1, a2)
  return polyadd(a1, -a2)
