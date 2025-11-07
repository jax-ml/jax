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

import numpy as np

from jax._src import lax
from jax._src import numpy as jnp
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import promote_args_inexact
from jax._src.scipy.special import betaln, betainc, xlogy, xlog1py
from jax._src.typing import Array, ArrayLike
from jax._src.scipy.special import erfinv
import jax


def logpdf(x: ArrayLike, a: ArrayLike, b: ArrayLike,
           loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Beta log probability distribution function.

  JAX implementation of :obj:`scipy.stats.beta` ``logpdf``.

  The pdf of the beta function is:

  .. math::

    f(x, a, b) = \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} x^{a-1}(1-x)^{b-1}

  where :math:`\Gamma` is the :func:`~jax.scipy.special.gamma` function,
  It is defined for :math:`0\le x\le 1` and :math:`b>0`.

  Args:
    x: arraylike, value at which to evaluate the PDF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logpdf values

  See Also:
    - :func:`jax.scipy.stats.beta.cdf`
    - :func:`jax.scipy.stats.beta.pdf`
    - :func:`jax.scipy.stats.beta.sf`
    - :func:`jax.scipy.stats.beta.logcdf`
    - :func:`jax.scipy.stats.beta.logsf`
  """
  x, a, b, loc, scale = promote_args_inexact("beta.logpdf", x, a, b, loc, scale)
  one = _lax_const(x, 1)
  zero = _lax_const(a, 0)
  shape_term = lax.neg(betaln(a, b))
  y = lax.div(lax.sub(x, loc), scale)
  log_linear_term = lax.add(xlogy(lax.sub(a, one), y),
                            xlog1py(lax.sub(b, one), lax.neg(y)))
  log_probs = lax.sub(lax.add(shape_term, log_linear_term), lax.log(scale))
  result = jnp.where(jnp.logical_or(lax.gt(x, lax.add(loc, scale)),
                                    lax.lt(x, loc)), -np.inf, log_probs)
  result_positive_constants = jnp.where(jnp.logical_or(jnp.logical_or(lax.le(a, zero), lax.le(b, zero)),
                                                       lax.le(scale, zero)), np.nan, result)
  return result_positive_constants


def pdf(x: ArrayLike, a: ArrayLike, b: ArrayLike,
        loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Beta probability distribution function.

  JAX implementation of :obj:`scipy.stats.beta` ``pdf``.

  The pdf of the beta function is:

  .. math::

    f(x, a, b) = \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} x^{a-1}(1-x)^{b-1}

  where :math:`\Gamma` is the :func:`~jax.scipy.special.gamma` function.
  It is defined for :math:`0\le x\le 1` and :math:`b>0`.

  Args:
    x: arraylike, value at which to evaluate the PDF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of pdf values

  See Also:
    - :func:`jax.scipy.stats.beta.cdf`
    - :func:`jax.scipy.stats.beta.sf`
    - :func:`jax.scipy.stats.beta.logcdf`
    - :func:`jax.scipy.stats.beta.logpdf`
    - :func:`jax.scipy.stats.beta.logsf`
  """
  return lax.exp(logpdf(x, a, b, loc, scale))


def cdf(x: ArrayLike, a: ArrayLike, b: ArrayLike,
        loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Beta cumulative distribution function

  JAX implementation of :obj:`scipy.stats.beta` ``cdf``.

  The cdf is defined as

  .. math::

     f_{cdf}(x, a, b) = \int_{-\infty}^x f_{pdf}(y, a, b)\mathrm{d}y

  where :math:`f_{pdf}` is the beta distribution probability density function,
  :func:`jax.scipy.stats.beta.pdf`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of cdf values

  See Also:
    - :func:`jax.scipy.stats.beta.pdf`
    - :func:`jax.scipy.stats.beta.sf`
    - :func:`jax.scipy.stats.beta.logcdf`
    - :func:`jax.scipy.stats.beta.logpdf`
    - :func:`jax.scipy.stats.beta.logsf`
  """
  x, a, b, loc, scale = promote_args_inexact("beta.cdf", x, a, b, loc, scale)
  return betainc(
    a,
    b,
    lax.clamp(
      _lax_const(x, 0),
      lax.div(lax.sub(x, loc), scale),
      _lax_const(x, 1),
    )
  )


def logcdf(x: ArrayLike, a: ArrayLike, b: ArrayLike,
           loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Beta log cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.beta` ``logcdf``.

  The cdf is defined as

  .. math::

     f_{cdf}(x, a, b) = \int_{-\infty}^x f_{pdf}(y, a, b)\mathrm{d}y

  where :math:`f_{pdf}` is the beta distribution probability density function,
  :func:`jax.scipy.stats.beta.pdf`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logcdf values

  See Also:
    - :func:`jax.scipy.stats.beta.cdf`
    - :func:`jax.scipy.stats.beta.pdf`
    - :func:`jax.scipy.stats.beta.sf`
    - :func:`jax.scipy.stats.beta.logpdf`
    - :func:`jax.scipy.stats.beta.logsf`
  """
  return lax.log(cdf(x, a, b, loc, scale))


def sf(x: ArrayLike, a: ArrayLike, b: ArrayLike,
       loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Beta distribution survival function.

  JAX implementation of :obj:`scipy.stats.beta` ``sf``.

  The survival function is defined as

  .. math::

     f_{sf}(x, a, b) = 1 - f_{cdf}(x, a, b)

  where :math:`f_{cdf}(x, a, b)` is the beta cumulative distribution function,
  :func:`jax.scipy.stats.beta.cdf`.

  Args:
    x: arraylike, value at which to evaluate the SF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of sf values.

  See Also:
    - :func:`jax.scipy.stats.beta.cdf`
    - :func:`jax.scipy.stats.beta.pdf`
    - :func:`jax.scipy.stats.beta.logcdf`
    - :func:`jax.scipy.stats.beta.logpdf`
    - :func:`jax.scipy.stats.beta.logsf`
  """
  x, a, b, loc, scale = promote_args_inexact("beta.sf", x, a, b, loc, scale)
  return betainc(
    b,
    a,
    1 - lax.clamp(
      _lax_const(x, 0),
      lax.div(lax.sub(x, loc), scale),
      _lax_const(x, 1),
    )
  )


def logsf(x: ArrayLike, a: ArrayLike, b: ArrayLike,
          loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Beta distribution log survival function.

  JAX implementation of :obj:`scipy.stats.beta` ``logsf``.

  The survival function is defined as

  .. math::

     f_{sf}(x, a, b) = 1 - f_{cdf}(x, a, b)

  where :math:`f_{cdf}(x, a, b)` is the beta cumulative distribution function,
  :func:`jax.scipy.stats.beta.cdf`.

  Args:
    x: arraylike, value at which to evaluate the SF
    a: arraylike, distribution shape parameter
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logsf values.

  See Also:
    - :func:`jax.scipy.stats.beta.cdf`
    - :func:`jax.scipy.stats.beta.pdf`
    - :func:`jax.scipy.stats.beta.sf`
    - :func:`jax.scipy.stats.beta.logcdf`
    - :func:`jax.scipy.stats.beta.logpdf`
  """
  return lax.log(sf(x, a, b, loc, scale))


def ppf(q: ArrayLike, a: ArrayLike, b: ArrayLike,
        loc: ArrayLike = 0, scale: ArrayLike = 1,
        tol: float = 1e-8, maxiter: int = 60) -> Array:
  r"""Percent point function (inverse of cdf) for the Beta distribution.

  This implements a robust bisection solver in standardized space [0,1].
  The function exposes a custom VJP so that gradients match the implicit
  relation dx/dq = 1 / pdf(x).
  """
  q, a, b, loc, scale = promote_args_inexact("beta.ppf", q, a, b, loc, scale)
  q = jnp.asarray(q)
  zero = _lax_const(q, 0)
  one = _lax_const(q, 1)

  is_zero = lax.eq(q, zero)
  is_one = lax.eq(q, one)

  q_clamped = lax.clamp(zero, q, one)

  def _bisection_std(q, a, b, maxiter):
    # ensure lo/hi have the same shape as q so the loop carry types don't change
    lo = jnp.full_like(q, 1e-12)
    hi = jnp.full_like(q, 1.0) - jnp.full_like(q, 1e-12)

    def body(i, state):
      lo, hi = state
      mid = (lo + hi) * 0.5
      c = betainc(a, b, mid)
      lo = jnp.where(lax.lt(c, q), mid, lo)
      hi = jnp.where(lax.lt(c, q), hi, mid)
      return (lo, hi)

    lo, hi = jax.lax.fori_loop(0, int(maxiter), body, (lo, hi))
    return (lo + hi) * 0.5

  @jax.custom_vjp
  def _ppf_std(q, a, b, loc, scale):
    x = _bisection_std(q, a, b, maxiter)
    return loc + scale * x

  def _ppf_std_fwd(q, a, b, loc, scale):
    x = _bisection_std(q, a, b, maxiter)
    return loc + scale * x, (x, a, b, scale)

  def _ppf_std_bwd(res, g):
    x, a, b, scale = res
    # pdf in standardized space
    safe_x = jnp.clip(x, 1e-12, 1.0 - 1e-12)
    log_pdf = lax.sub(lax.add(lax.mul(lax.sub(a, _lax_const(a, 1)), lax.log(safe_x)),
                              lax.mul(lax.sub(b, _lax_const(b, 1)), lax.log(lax.sub(_lax_const(safe_x, 1), safe_x)))), betaln(a, b))
    pdf_std = lax.exp(log_pdf)
    # d(res)/dq = scale * (1 / pdf_std)
    dq = g * (scale / jnp.maximum(pdf_std, _lax_const(pdf_std, 1e-300)))
    return (dq, jnp.zeros_like(a), jnp.zeros_like(b), jnp.zeros_like(loc), jnp.zeros_like(scale))

  _ppf_std.defvjp(_ppf_std_fwd, _ppf_std_bwd)

  res = _ppf_std(q_clamped, a, b, loc, scale)
  res = jnp.where(is_zero, loc, res)
  res = jnp.where(is_one, loc + scale, res)
  return res
