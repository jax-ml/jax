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


import jax.numpy as jnp
from jax import lax
from jax._src.numpy.util import promote_args_inexact
from jax._src.typing import Array, ArrayLike


def logpdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Exponential log probability distribution function.

  JAX implementation of :obj:`scipy.stats.expon` ``logpdf``.

  The Exponential probability distribution function is

  .. math::

     f(x) = \begin{cases}
       e^{-x} & x \ge 0 \\
       0 & \mathrm{otherwise}
     \end{cases}

  Args:
    x: arraylike, value at which to evaluate the PDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logpdf values.

  See Also:
    :func:`jax.scipy.stats.expon.cdf`
    :func:`jax.scipy.stats.expon.pdf`
    :func:`jax.scipy.stats.expon.ppf`
    :func:`jax.scipy.stats.expon.sf`
    :func:`jax.scipy.stats.expon.logcdf`
    :func:`jax.scipy.stats.expon.logpdf`
    :func:`jax.scipy.stats.expon.logsf`
  """
  x, loc, scale = promote_args_inexact("expon.logpdf", x, loc, scale)
  log_scale = lax.log(scale)
  linear_term = lax.div(lax.sub(x, loc), scale)
  log_probs = lax.neg(lax.add(linear_term, log_scale))
  return jnp.where(lax.lt(x, loc), -jnp.inf, log_probs)


def pdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Exponential probability distribution function.

  JAX implementation of :obj:`scipy.stats.expon` ``pdf``.

  The Exponential probability distribution function is

  .. math::

     f(x) = \begin{cases}
       e^{-x} & x \ge 0 \\
       0 & \mathrm{otherwise}
     \end{cases}

  Args:
    x: arraylike, value at which to evaluate the PDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of pdf values.

  See Also:
    :func:`jax.scipy.stats.expon.cdf`
    :func:`jax.scipy.stats.expon.pdf`
    :func:`jax.scipy.stats.expon.ppf`
    :func:`jax.scipy.stats.expon.sf`
    :func:`jax.scipy.stats.expon.logcdf`
    :func:`jax.scipy.stats.expon.logpdf`
    :func:`jax.scipy.stats.expon.logsf`
  """
  return lax.exp(logpdf(x, loc, scale))


def cdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Exponential cumulative density function.

  JAX implementation of :obj:`scipy.stats.expon` ``cdf``.

  The cdf is defined as

  .. math::

     f_{cdf}(x) = \int_{-\infty}^x f_{pdf}(y)\mathrm{d}y

  where :math:`f_{pdf}` is the exponential distribution probability density function,
  :func:`jax.scipy.stats.expon.pdf`.

  Args:
    x: arraylike, value at which to evaluate the PDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of pdf values.

  See Also:
    :func:`jax.scipy.stats.expon.cdf`
    :func:`jax.scipy.stats.expon.pdf`
    :func:`jax.scipy.stats.expon.ppf`
    :func:`jax.scipy.stats.expon.sf`
    :func:`jax.scipy.stats.expon.logcdf`
    :func:`jax.scipy.stats.expon.logpdf`
    :func:`jax.scipy.stats.expon.logsf`
  """
  x, loc, scale = promote_args_inexact("expon.cdf", x, loc, scale)
  neg_scaled_x = lax.div(lax.sub(loc, x), scale)
  return jnp.where(
    lax.lt(x, loc),
    jnp.zeros_like(neg_scaled_x),
    lax.neg(lax.expm1(neg_scaled_x)),
  )


def logcdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Exponential log cumulative density function.

  JAX implementation of :obj:`scipy.stats.expon` ``logcdf``.

  The cdf is defined as

  .. math::

     f_{cdf}(x) = \int_{-\infty}^x f_{pdf}(y)\mathrm{d}y

  where :math:`f_{pdf}` is the exponential distribution probability density function,
  :func:`jax.scipy.stats.expon.pdf`.

  Args:
    x: arraylike, value at which to evaluate the PDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of pdf values.

  See Also:
    :func:`jax.scipy.stats.expon.cdf`
    :func:`jax.scipy.stats.expon.pdf`
    :func:`jax.scipy.stats.expon.ppf`
    :func:`jax.scipy.stats.expon.sf`
    :func:`jax.scipy.stats.expon.logcdf`
    :func:`jax.scipy.stats.expon.logpdf`
    :func:`jax.scipy.stats.expon.logsf`
  """
  return lax.log1p(lax.neg(sf(x, loc, scale)))


def logsf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Exponential log survival function.

  JAX implementation of :obj:`scipy.stats.expon` ``logsf``.

  The survival function is defined as

  .. math::

     f_{sf}(x) = 1 - f_{cdf}(x)

  where :math:`f_{cdf}(x)` is the exponential cumulative distribution function,
  :func:`jax.scipy.stats.expon.cdf`.

  Args:
    x: arraylike, value at which to evaluate the PDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of pdf values.

  See Also:
    :func:`jax.scipy.stats.expon.cdf`
    :func:`jax.scipy.stats.expon.pdf`
    :func:`jax.scipy.stats.expon.ppf`
    :func:`jax.scipy.stats.expon.sf`
    :func:`jax.scipy.stats.expon.logcdf`
    :func:`jax.scipy.stats.expon.logpdf`
    :func:`jax.scipy.stats.expon.logsf`
  """
  x, loc, scale = promote_args_inexact("expon.sf", x, loc, scale)
  neg_scaled_x = lax.div(lax.sub(loc, x), scale)
  return jnp.where(lax.lt(x, loc), jnp.zeros_like(neg_scaled_x), neg_scaled_x)


def sf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Exponential survival function.

  JAX implementation of :obj:`scipy.stats.expon` ``sf``.

  The survival function is defined as

  .. math::

     f_{sf}(x) = 1 - f_{cdf}(x)

  where :math:`f_{cdf}(x)` is the exponential cumulative distribution function,
  :func:`jax.scipy.stats.expon.cdf`.

  Args:
    x: arraylike, value at which to evaluate the PDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of pdf values.

  See Also:
    :func:`jax.scipy.stats.expon.cdf`
    :func:`jax.scipy.stats.expon.pdf`
    :func:`jax.scipy.stats.expon.ppf`
    :func:`jax.scipy.stats.expon.sf`
    :func:`jax.scipy.stats.expon.logcdf`
    :func:`jax.scipy.stats.expon.logpdf`
    :func:`jax.scipy.stats.expon.logsf`
  """
  return lax.exp(logsf(x, loc, scale))


def ppf(q: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Exponential survival function.

  JAX implementation of :obj:`scipy.stats.expon` ``ppf``.

  The percent point function is defined as the inverse of the
  cumulative distribution function, :func:`jax.scipy.stats.expon.cdf`.

  Args:
    x: arraylike, value at which to evaluate the PDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of pdf values.

  See Also:
    :func:`jax.scipy.stats.expon.cdf`
    :func:`jax.scipy.stats.expon.pdf`
    :func:`jax.scipy.stats.expon.ppf`
    :func:`jax.scipy.stats.expon.sf`
    :func:`jax.scipy.stats.expon.logcdf`
    :func:`jax.scipy.stats.expon.logpdf`
    :func:`jax.scipy.stats.expon.logsf`
  """
  q, loc, scale = promote_args_inexact("expon.ppf", q, loc, scale)
  neg_scaled_q = lax.div(lax.sub(loc, q), scale)
  return jnp.where(
    jnp.isnan(q) | (q < 0) | (q > 1),
    jnp.nan,
    lax.neg(lax.log1p(neg_scaled_q)),
  )
