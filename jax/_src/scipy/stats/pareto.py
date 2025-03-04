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
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import promote_args_inexact
from jax._src.typing import Array, ArrayLike


def logpdf(
  x: ArrayLike, b: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1
) -> Array:
  r"""Pareto log probability distribution function.

  JAX implementation of :obj:`scipy.stats.pareto` ``logpdf``.

  The Pareto probability density function is given by

  .. math::

     f(x, b) = \begin{cases}
       bx^{-(b+1)} & x \ge 1\\
       0 & x < 1
     \end{cases}

  and is defined for :math:`b > 0`.

  Args:
    x: arraylike, value at which to evaluate the PDF
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logpdf values.

  See Also:
    :func:`jax.scipy.stats.pareto.logcdf`
    :func:`jax.scipy.stats.pareto.logppf`
    :func:`jax.scipy.stats.pareto.logsf`
    :func:`jax.scipy.stats.pareto.cdf`
    :func:`jax.scipy.stats.pareto.pdf`
    :func:`jax.scipy.stats.pareto.ppf`
    :func:`jax.scipy.stats.pareto.sf`
  """
  x, b, loc, scale = promote_args_inexact("pareto.logpdf", x, b, loc, scale)
  one = _lax_const(x, 1)
  scaled_x = lax.div(lax.sub(x, loc), scale)
  normalize_term = lax.log(lax.div(scale, b))
  log_probs = lax.neg(
    lax.add(normalize_term, lax.mul(lax.add(b, one), lax.log(scaled_x)))
  )
  return jnp.where(lax.lt(x, lax.add(loc, scale)), -jnp.inf, log_probs)


def pdf(x: ArrayLike, b: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Pareto probability distribution function.

  JAX implementation of :obj:`scipy.stats.pareto` ``pdf``.

  The Pareto probability density function is given by

  .. math::

     f(x, b) = \begin{cases}
       bx^{-(b+1)} & x \ge 1\\
       0 & x < 1
     \end{cases}

  and is defined for :math:`b > 0`.

  Args:
    x: arraylike, value at which to evaluate the PDF
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of pdf values.

  See Also:
    :func:`jax.scipy.stats.pareto.logcdf`
    :func:`jax.scipy.stats.pareto.logpdf`
    :func:`jax.scipy.stats.pareto.logppf`
    :func:`jax.scipy.stats.pareto.logsf`
    :func:`jax.scipy.stats.pareto.cdf`
    :func:`jax.scipy.stats.pareto.ppf`
    :func:`jax.scipy.stats.pareto.sf`
  """
  return lax.exp(logpdf(x, b, loc, scale))


def cdf(x: ArrayLike, b: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Pareto cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.pareto` ``cdf``.

  The Pareto cumulative distribution function is given by

  .. math::

     F(x, b) = \begin{cases}
       1 - x^{-b} & x \ge 1\\
       0 & x < 1
     \end{cases}

  and is defined for :math:`b > 0`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of CDF values.

  See Also:
    :func:`jax.scipy.stats.pareto.logcdf`
    :func:`jax.scipy.stats.pareto.logpdf`
    :func:`jax.scipy.stats.pareto.logppf`
    :func:`jax.scipy.stats.pareto.logsf`
    :func:`jax.scipy.stats.pareto.pdf`
    :func:`jax.scipy.stats.pareto.ppf`
    :func:`jax.scipy.stats.pareto.sf`
  """
  x, b, loc, scale = promote_args_inexact("pareto.cdf", x, b, loc, scale)
  one = _lax_const(x, 1)
  zero = _lax_const(x, 0)
  scaled_x = lax.div(lax.sub(x, loc), scale)
  cdf = lax.sub(one, lax.pow(scaled_x, lax.neg(b)))
  return jnp.where(lax.lt(x, lax.add(loc, scale)), zero, cdf)


def logcdf(
  x: ArrayLike, b: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1
) -> Array:
  r"""Pareto log cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.pareto` ``logcdf``.

  The Pareto cumulative distribution function is given by

  .. math::

     F(x, b) = \begin{cases}
       1 - x^{-b} & x \ge 1\\
       0 & x < 1
     \end{cases}

  and is defined for :math:`b > 0`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logCDF values.

  See Also:
    :func:`jax.scipy.stats.pareto.logpdf`
    :func:`jax.scipy.stats.pareto.logppf`
    :func:`jax.scipy.stats.pareto.logsf`
    :func:`jax.scipy.stats.pareto.cdf`
    :func:`jax.scipy.stats.pareto.pdf`
    :func:`jax.scipy.stats.pareto.ppf`
    :func:`jax.scipy.stats.pareto.sf`
  """
  return lax.log(cdf(x, b, loc, scale))


def logsf(
  x: ArrayLike, b: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1
) -> Array:
  r"""Pareto log survival function.

  JAX implementation of :obj:`scipy.stats.pareto` ``logsf``.

  The Pareto survival function is given by

  .. math::

     S(x, b) = \begin{cases}
       x^{-b} & x \ge 1\\
       1 & x < 1
     \end{cases}

  and is defined for :math:`b > 0`.

  Args:
    x: arraylike, value at which to evaluate the survival function
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of log survival function values.

  See Also:
    :func:`jax.scipy.stats.pareto.logcdf`
    :func:`jax.scipy.stats.pareto.logpdf`
    :func:`jax.scipy.stats.pareto.logppf`
    :func:`jax.scipy.stats.pareto.cdf`
    :func:`jax.scipy.stats.pareto.pdf`
    :func:`jax.scipy.stats.pareto.ppf`
    :func:`jax.scipy.stats.pareto.sf`
  """
  x, b, loc, scale = promote_args_inexact("pareto.logsf", x, b, loc, scale)
  one = _lax_const(x, 1)
  scaled_x = lax.div(lax.sub(x, loc), scale)
  logsf_val = lax.neg(lax.mul(b, lax.log(scaled_x)))
  return jnp.where(lax.lt(x, lax.add(loc, scale)), one, logsf_val)


def sf(x: ArrayLike, b: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Pareto survival function.

  JAX implementation of :obj:`scipy.stats.pareto` ``sf``.

  The Pareto survival function is given by

  .. math::

     S(x, b) = \begin{cases}
       x^{-b} & x \ge 1\\
       1 & x < 1
     \end{cases}

  and is defined for :math:`b > 0`.

  Args:
    x: arraylike, value at which to evaluate the survival function
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of survival function values.

  See Also:
    :func:`jax.scipy.stats.pareto.logcdf`
    :func:`jax.scipy.stats.pareto.logpdf`
    :func:`jax.scipy.stats.pareto.logppf`
    :func:`jax.scipy.stats.pareto.logsf`
    :func:`jax.scipy.stats.pareto.cdf`
    :func:`jax.scipy.stats.pareto.pdf`
    :func:`jax.scipy.stats.pareto.ppf`
  """
  return lax.exp(logsf(x, b, loc, scale))


def logppf(
  q: ArrayLike, b: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1
) -> Array:
  r"""Pareto log percent point function (inverse CDF).

  JAX implementation of :obj:`scipy.stats.pareto` ``logppf``.

  The Pareto percent point function is the inverse of the Pareto CDF, and is
  given by

  .. math::

     F^{-1}(q, b) = \begin{cases}
       (1 - q)^{-1/b} & 0 \le q < 1\\
       \text{NaN} & \text{otherwise}
     \end{cases}

  and is defined for :math:`b > 0`.

  Args:
    q: arraylike, value at which to evaluate the inverse CDF
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of log percent point function values.

  See Also:
    :func:`jax.scipy.stats.pareto.logcdf`
    :func:`jax.scipy.stats.pareto.logpdf`
    :func:`jax.scipy.stats.pareto.logsf`
    :func:`jax.scipy.stats.pareto.cdf`
    :func:`jax.scipy.stats.pareto.pdf`
    :func:`jax.scipy.stats.pareto.ppf`
    :func:`jax.scipy.stats.pareto.sf`
  """
  q, b, loc, scale = promote_args_inexact("pareto.logppf", q, b, loc, scale)
  return jnp.where(
    jnp.isnan(q) | (q < 0) | (q > 1),
    jnp.nan,
    lax.neg(lax.div(lax.log1p(lax.neg(q)), b)),
  )


def ppf(q: ArrayLike, b: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Pareto percent point function (inverse CDF).

  JAX implementation of :obj:`scipy.stats.pareto` ``ppf``.

  The Pareto percent point function is the inverse of the Pareto CDF, and is
  given by

  .. math::

     F^{-1}(q, b) = \begin{cases}
       (1 - q)^{-1/b} & 0 \le q < 1\\
       \text{NaN} & \text{otherwise}
     \end{cases}

  and is defined for :math:`b > 0`.

  Args:
    q: arraylike, value at which to evaluate the inverse CDF
    b: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of percent point function values.

  See Also:
    :func:`jax.scipy.stats.pareto.logcdf`
    :func:`jax.scipy.stats.pareto.logpdf`
    :func:`jax.scipy.stats.pareto.logppf`
    :func:`jax.scipy.stats.pareto.logsf`
    :func:`jax.scipy.stats.pareto.cdf`
    :func:`jax.scipy.stats.pareto.pdf`
    :func:`jax.scipy.stats.pareto.sf`
  """
  return lax.exp(logppf(q, b, loc, scale))
