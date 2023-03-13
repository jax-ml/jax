# Copyright 2018 The JAX Authors.
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
from typing import Optional, Tuple, Union

import numpy as np

from jax import jit
from jax import lax
from jax._src import dtypes
from jax._src import core
from jax._src.numpy.lax_numpy import (
    arange, argmin, array, asarray, atleast_1d, concatenate, convolve,
    diag, dot, finfo, full, ones, outer, roll, trim_zeros,
    trim_zeros_tol, vander, zeros)
from jax._src.numpy.ufuncs import maximum, true_divide, sqrt
from jax._src.numpy.reductions import all
from jax._src.numpy import linalg
from jax._src.numpy.util import (
    check_arraylike, promote_dtypes, promote_dtypes_inexact, _where, _wraps)
from jax._src.typing import Array, ArrayLike


@jit
def _roots_no_zeros(p: Array) -> Array:
  # build companion matrix and find its eigenvalues (the roots)
  if p.size < 2:
    return array([], dtype=dtypes.to_complex_dtype(p.dtype))
  A = diag(ones((p.size - 2,), p.dtype), -1)
  A = A.at[0, :].set(-p[1:] / p[0])
  return linalg.eigvals(A)


@jit
def _roots_with_zeros(p: Array, num_leading_zeros: int) -> Array:
  # Avoid lapack errors when p is all zero
  p = _where(len(p) == num_leading_zeros, 1.0, p)
  # Roll any leading zeros to the end & compute the roots
  roots = _roots_no_zeros(roll(p, -num_leading_zeros))
  # Sort zero roots to the end.
  roots = lax.sort_key_val(roots == 0, roots)[1]
  # Set roots associated with num_leading_zeros to NaN
  return _where(arange(roots.size) < roots.size - num_leading_zeros, roots, complex(np.nan, np.nan))


@_wraps(np.roots, lax_description="""\
Unlike the numpy version of this function, the JAX version returns the roots in
a complex array regardless of the values of the roots. Additionally, the jax
version of this function adds the ``strip_zeros`` function which must be set to
False for the function to be compatible with JIT and other JAX transformations.
With ``strip_zeros=False``, if your coefficients have leading zeros, the
roots will be padded with NaN values:

>>> coeffs = jnp.array([0, 1, 2])

# The default behavior matches numpy and strips leading zeros:
>>> jnp.roots(coeffs)
Array([-2.+0.j], dtype=complex64)

# With strip_zeros=False, extra roots are set to NaN:
>>> jnp.roots(coeffs, strip_zeros=False)
Array([-2. +0.j, nan+nanj], dtype=complex64)
""",
extra_params="""
strip_zeros : bool, default=True
    If set to True, then leading zeros in the coefficients will be stripped, similar
    to :func:`numpy.roots`. If set to False, leading zeros will not be stripped, and
    undefined roots will be represented by NaN values in the function output.
    ``strip_zeros`` must be set to ``False`` for the function to be compatible with
    :func:`jax.jit` and other JAX transformations.
""")
def roots(p: ArrayLike, *, strip_zeros: bool = True) -> Array:
  check_arraylike("roots", p)
  p_arr = atleast_1d(*promote_dtypes_inexact(p))
  if p_arr.ndim != 1:
    raise ValueError("Input must be a rank-1 array.")
  if p_arr.size < 2:
    return array([], dtype=dtypes.to_complex_dtype(p_arr.dtype))
  num_leading_zeros = _where(all(p_arr == 0), len(p_arr), argmin(p_arr == 0))

  if strip_zeros:
    num_leading_zeros = core.concrete_or_error(int, num_leading_zeros,
      "The error occurred in the jnp.roots() function. To use this within a "
      "JIT-compiled context, pass strip_zeros=False, but be aware that leading zeros "
      "will be result in some returned roots being set to NaN.")
    return _roots_no_zeros(p_arr[num_leading_zeros:])
  else:
    return _roots_with_zeros(p_arr, num_leading_zeros)


_POLYFIT_DOC = """\
Unlike NumPy's implementation of polyfit, :py:func:`jax.numpy.polyfit` will not warn on rank reduction, which indicates an ill conditioned matrix
Also, it works best on rcond <= 10e-3 values.
"""
@_wraps(np.polyfit, lax_description=_POLYFIT_DOC)
@partial(jit, static_argnames=('deg', 'rcond', 'full', 'cov'))
def polyfit(x: Array, y: Array, deg: int, rcond: Optional[float] = None,
            full: bool = False, w: Optional[Array] = None, cov: bool = False
            ) -> Union[Array, Tuple[Array, ...]]:
  check_arraylike("polyfit", x, y)
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
    check_arraylike("polyfit", w)
    w, = promote_dtypes_inexact(w)
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
    return c, resids, rank, s, asarray(rcond)
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
def poly(seq_of_zeros: Array) -> Array:
  check_arraylike('poly', seq_of_zeros)
  seq_of_zeros, = promote_dtypes_inexact(seq_of_zeros)
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
def polyval(p: Array, x: Array, *, unroll: int = 16) -> Array:
  check_arraylike("polyval", p, x)
  p, x = promote_dtypes_inexact(p, x)
  shape = lax.broadcast_shapes(p.shape[1:], x.shape)
  y = lax.full_like(x, 0, shape=shape, dtype=x.dtype)
  y, _ = lax.scan(lambda y, p: (y * x + p, None), y, p, unroll=unroll)
  return y

@_wraps(np.polyadd)
@jit
def polyadd(a1: Array, a2: Array) -> Array:
  check_arraylike("polyadd", a1, a2)
  a1, a2 = promote_dtypes(a1, a2)
  if a2.shape[0] <= a1.shape[0]:
    return a1.at[-a2.shape[0]:].add(a2)
  else:
    return a2.at[-a1.shape[0]:].add(a1)


@_wraps(np.polyint)
@partial(jit, static_argnames=('m',))
def polyint(p: Array, m: int = 1, k: Optional[int] = None) -> Array:
  m = core.concrete_or_error(operator.index, m, "'m' argument of jnp.polyint")
  k = 0 if k is None else k
  check_arraylike("polyint", p, k)
  p, k_arr = promote_dtypes_inexact(p, k)
  if m < 0:
    raise ValueError("Order of integral must be positive (see polyder)")
  k_arr = atleast_1d(k_arr)
  if len(k_arr) == 1:
    k_arr = full((m,), k_arr[0])
  if k_arr.shape != (m,):
    raise ValueError("k must be a scalar or a rank-1 array of length 1 or m.")
  if m == 0:
    return p
  else:
    grid = (arange(len(p) + m, dtype=p.dtype)[np.newaxis]
            - arange(m, dtype=p.dtype)[:, np.newaxis])
    coeff = maximum(1, grid).prod(0)[::-1]
    return true_divide(concatenate((p, k_arr)), coeff)


@_wraps(np.polyder)
@partial(jit, static_argnames=('m',))
def polyder(p: Array, m: int = 1) -> Array:
  check_arraylike("polyder", p)
  m = core.concrete_or_error(operator.index, m, "'m' argument of jnp.polyder")
  p, = promote_dtypes_inexact(p)
  if m < 0:
    raise ValueError("Order of derivative must be positive")
  if m == 0:
    return p
  coeff = (arange(m, len(p), dtype=p.dtype)[np.newaxis]
          - arange(m, dtype=p.dtype)[:, np.newaxis]).prod(0)
  return p[:-m] * coeff[::-1]


_LEADING_ZEROS_DOC = """\
Setting trim_leading_zeros=True makes the output match that of numpy.
But prevents the function from being able to be used in compiled code.
Due to differences in accumulation of floating point arithmetic errors, the cutoff for values to be
considered zero may lead to inconsistent results between NumPy and JAX, and even between different
JAX backends. The result may lead to inconsistent output shapes when trim_leading_zeros=True.
"""

@_wraps(np.polymul, lax_description=_LEADING_ZEROS_DOC)
def polymul(a1: ArrayLike, a2: ArrayLike, *, trim_leading_zeros: bool = False) -> Array:
  check_arraylike("polymul", a1, a2)
  a1_arr, a2_arr = promote_dtypes_inexact(a1, a2)
  if trim_leading_zeros and (len(a1_arr) > 1 or len(a2_arr) > 1):
    a1_arr, a2_arr = trim_zeros(a1_arr, trim='f'), trim_zeros(a2_arr, trim='f')
  if len(a1_arr) == 0:
    a1_arr = asarray([0], dtype=a2_arr.dtype)
  if len(a2_arr) == 0:
    a2_arr = asarray([0], dtype=a1_arr.dtype)
  return convolve(a1_arr, a2_arr, mode='full')

@_wraps(np.polydiv, lax_description=_LEADING_ZEROS_DOC)
def polydiv(u: ArrayLike, v: ArrayLike, *, trim_leading_zeros: bool = False) -> Tuple[Array, Array]:
  check_arraylike("polydiv", u, v)
  u_arr, v_arr = promote_dtypes_inexact(u, v)
  m = len(u_arr) - 1
  n = len(v_arr) - 1
  scale = 1. / v_arr[0]
  q: Array = zeros(max(m - n + 1, 1), dtype = u_arr.dtype) # force same dtype
  for k in range(0, m-n+1):
    d = scale * u_arr[k]
    q = q.at[k].set(d)
    u_arr = u_arr.at[k:k+n+1].add(-d*v_arr)
  if trim_leading_zeros:
    # use the square root of finfo(dtype) to approximate the absolute tolerance used in numpy
    u_arr = trim_zeros_tol(u_arr, tol=sqrt(finfo(u_arr.dtype).eps), trim='f')
  return q, u_arr

@_wraps(np.polysub)
@jit
def polysub(a1: Array, a2: Array) -> Array:
  check_arraylike("polysub", a1, a2)
  a1, a2 = promote_dtypes(a1, a2)
  return polyadd(a1, -a2)
