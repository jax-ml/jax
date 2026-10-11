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
from jax._src.numpy.ufuncs import isneginf, isposinf
from jax._src.numpy.util import promote_args_inexact
from jax._src.scipy.special import stdtr
from jax._src.typing import Array, ArrayLike


def logpdf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Student's T log probability distribution function.

  JAX implementation of :obj:`scipy.stats.t` ``logpdf``.

  The Student's T probability distribution function is given by

  .. math::

     f(x, \nu) = \frac{\Gamma((\nu + 1)/2)}{\sqrt{\pi\nu}\Gamma(\nu/2)}(1 + x^2/\nu)^{(\nu+1)/2}

  Where :math:`\Gamma` is the :func:`~jax.scipy.special.gamma` function, and :math:`\nu > 0`
  is the degrees of freedom (JAX follows the scipy convention of naming this ``df``).

  Args:
    x: arraylike, value at which to evaluate the PDF
    df: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logpdf values.

  See Also:
    - :func:`jax.scipy.stats.t.cdf`
    - :func:`jax.scipy.stats.t.pdf`
    - :func:`jax.scipy.stats.t.sf`
    - :func:`jax.scipy.stats.t.logcdf`
    - :func:`jax.scipy.stats.t.logsf`
  """
  x, df, loc, scale = promote_args_inexact("t.logpdf", x, df, loc, scale)
  two = _lax_const(x, 2)
  scaled_x = lax.div(lax.sub(x, loc), scale)
  df_over_two = lax.div(df, two)
  df_plus_one_over_two = lax.add(df_over_two, _lax_const(x, 0.5))
  normalize_term_const = lax.mul(lax.mul(scale, scale), _lax_const(x, np.pi))
  normalize_term_tmp = lax.div(lax.log(lax.mul(normalize_term_const, df)), two)
  normalize_term = lax.sub(lax.add(lax.lgamma(df_over_two), normalize_term_tmp),
                           lax.lgamma(df_plus_one_over_two))
  quadratic = lax.div(lax.mul(scaled_x, scaled_x), df)
  return lax.neg(lax.add(normalize_term, lax.mul(df_plus_one_over_two, lax.log1p(quadratic))))


def pdf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Student's T probability distribution function.

  JAX implementation of :obj:`scipy.stats.t` ``pdf``.

  The Student's T probability distribution function is given by

  .. math::

     f(x, \nu) = \frac{\Gamma((\nu + 1)/2)}{\sqrt{\pi\nu}\Gamma(\nu/2)}(1 + x^2/\nu)^{(\nu+1)/2}

  Where :math:`\Gamma` is the :func:`~jax.scipy.special.gamma` function, and :math:`\nu > 0`
  is the degrees of freedom (JAX follows the scipy convention of naming this ``df``).

  Args:
    x: arraylike, value at which to evaluate the PDF
    df: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array

  See Also:
    - :func:`jax.scipy.stats.t.cdf`
    - :func:`jax.scipy.stats.t.logpdf`
    - :func:`jax.scipy.stats.t.sf`
    - :func:`jax.scipy.stats.t.logcdf`
    - :func:`jax.scipy.stats.t.logsf`
  """
  return lax.exp(logpdf(x, df, loc, scale))


def cdf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Student's T cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.t` ``cdf``.

  The cumulative distribution function is defined as

  .. math::

     F(x, \nu) = \int_{-\infty}^x f(u, \nu)\mathrm{d}u

  where :math:`f(u, \nu)` is the probability density function,
  :func:`jax.scipy.stats.t.pdf`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    df: arraylike, distribution shape parameter (degrees of freedom)
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of cdf values.

  See Also:
    - :func:`jax.scipy.stats.t.pdf`
    - :func:`jax.scipy.stats.t.logpdf`
    - :func:`jax.scipy.stats.t.sf`
    - :func:`jax.scipy.stats.t.logcdf`
    - :func:`jax.scipy.stats.t.logsf`
    - :func:`jax.scipy.special.stdtr`
  """
  x, df, loc, scale = promote_args_inexact("t.cdf", x, df, loc, scale)
  scaled_x = lax.div(lax.sub(x, loc), scale)
  return stdtr(df, scaled_x)


def sf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Student's T survival function.

  JAX implementation of :obj:`scipy.stats.t` ``sf``.

  The survival function is defined as

  .. math::

     S(x, \nu) = 1 - F(x, \nu)

  where :math:`F(x, \nu)` is the cumulative distribution function,
  :func:`jax.scipy.stats.t.cdf`.

  Args:
    x: arraylike, value at which to evaluate the SF
    df: arraylike, distribution shape parameter (degrees of freedom)
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of sf values.

  See Also:
    - :func:`jax.scipy.stats.t.cdf`
    - :func:`jax.scipy.stats.t.pdf`
    - :func:`jax.scipy.stats.t.logpdf`
    - :func:`jax.scipy.stats.t.logcdf`
    - :func:`jax.scipy.stats.t.logsf`
  """
  x, df, loc, scale = promote_args_inexact("t.sf", x, df, loc, scale)
  return cdf(-x, df, -loc, scale)


def logcdf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Student's T log cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.t` ``logcdf``.

  The log cumulative distribution function is defined as

  .. math::

     \log F(x, \nu) = \log\left(\int_{-\infty}^x f(u, \nu)\mathrm{d}u\right)

  where :math:`f(u, \nu)` is the probability density function,
  :func:`jax.scipy.stats.t.pdf`.

  Args:
    x: arraylike, value at which to evaluate the log CDF
    df: arraylike, distribution shape parameter (degrees of freedom)
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logcdf values.

  See Also:
    - :func:`jax.scipy.stats.t.cdf`
    - :func:`jax.scipy.stats.t.pdf`
    - :func:`jax.scipy.stats.t.logpdf`
    - :func:`jax.scipy.stats.t.sf`
    - :func:`jax.scipy.stats.t.logsf`
  """
  x, df, loc, scale = promote_args_inexact("t.logcdf", x, df, loc, scale)
  scaled_x = lax.div(lax.sub(x, loc), scale)
  zero = _lax_const(scaled_x, 0.0)
  left_tail = stdtr(df, scaled_x)
  right_tail = stdtr(df, lax.neg(scaled_x))
  res = jnp.where(
      lax.lt(scaled_x, zero),
      lax.log(left_tail),
      lax.log1p(lax.neg(right_tail)),
  )
  res = jnp.where(isneginf(scaled_x), -np.inf, res)
  res = jnp.where(isposinf(scaled_x), zero, res)
  res = jnp.where(lax.le(df, zero), np.nan, res)
  return res


def logsf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Student's T log survival function.

  JAX implementation of :obj:`scipy.stats.t` ``logsf``.

  The log survival function is defined as

  .. math::

     \log S(x, \nu) = \log(1 - F(x, \nu))

  where :math:`F(x, \nu)` is the cumulative distribution function,
  :func:`jax.scipy.stats.t.cdf`.

  Args:
    x: arraylike, value at which to evaluate the log SF
    df: arraylike, distribution shape parameter (degrees of freedom)
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logsf values.

  See Also:
    - :func:`jax.scipy.stats.t.cdf`
    - :func:`jax.scipy.stats.t.pdf`
    - :func:`jax.scipy.stats.t.logpdf`
    - :func:`jax.scipy.stats.t.sf`
    - :func:`jax.scipy.stats.t.logcdf`
  """
  x, df, loc, scale = promote_args_inexact("t.logsf", x, df, loc, scale)
  return logcdf(-x, df, -loc, scale)

