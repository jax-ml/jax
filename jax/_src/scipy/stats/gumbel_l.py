# Copyright 2025 The JAX Authors.
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
from jax._src.typing import Array, ArrayLike
from jax._src.scipy.special import xlogy, xlog1py


def logpdf(x: ArrayLike,
           loc: ArrayLike = 0,
           scale: ArrayLike = 1) -> Array:
  r"""
  Gumbel Distribution (Left Skewed) log probability distribution function.

  JAX implementation of :obj:`scipy.stats.gumbel_l` ``logpdf``.

   .. math::

      f_{pdf}(x; \mu, \beta) = \frac{1}{\beta} \exp\left( \frac{x - \mu}{\beta} - \exp\left( \frac{x - \mu}{\beta} \right) \right)

  Args:
    x: ArrayLike, value at which to evaluate log(pdf)
    loc: ArrayLike, distribution offset (:math:`\mu`) (defaulted to 0)
    scale: ArrayLike, distribution scaling (:math:`\beta`) (defaulted to 1)

  Returns:
    array of logpdf values

  See Also:
    - :func:`jax.scipy.stats.gumbel_l.pdf`
    - :func:`jax.scipy.stats.gumbel_l.logcdf`
    - :func:`jax.scipy.stats.gumbel_l.cdf`
    - :func:`jax.scipy.stats.gumbel_l.ppf`
    - :func:`jax.scipy.stats.gumbel_l.logsf`
    - :func:`jax.scipy.stats.gumbel_l.sf`
  """

  x, loc, scale = promote_args_inexact("gumbel_l.logpdf", x, loc, scale)
  ok = lax.gt(scale, _lax_const(scale, 0))
  # logpdf = -log(scale) + z - exp(z)
  z = lax.div(lax.sub(x, loc), scale)
  neg_log_scale = xlogy(-1, scale)
  t2 = lax.sub(z, lax.exp(z))
  log_pdf = lax.add(neg_log_scale, t2)
  return jnp.where(ok, log_pdf, np.nan)


def pdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""
  Gumbel Distribution (Left Skewed) probability distribution function.

  JAX implementation of :obj:`scipy.stats.gumbel_l` ``pdf``.

  .. math::

      f_{pdf}(x; \mu, \beta) = \frac{1}{\beta} \exp\left( \frac{x - \mu}{\beta} - \exp\left( \frac{x - \mu}{\beta} \right) \right)

  Args:
    x: ArrayLike, value at which to evaluate pdf
    loc: ArrayLike, distribution offset (:math:`\mu`) (defaulted to 0)
    scale: ArrayLike, distribution scaling (:math:`\beta`) (defaulted to 1)

  Returns:
    array of pdf values

  See Also:
    - :func:`jax.scipy.stats.gumbel_l.logpdf`
    - :func:`jax.scipy.stats.gumbel_l.logcdf`
    - :func:`jax.scipy.stats.gumbel_l.cdf`
    - :func:`jax.scipy.stats.gumbel_l.ppf`
    - :func:`jax.scipy.stats.gumbel_l.logsf`
    - :func:`jax.scipy.stats.gumbel_l.sf`
  """
  return lax.exp(logpdf(x, loc, scale))


def logcdf(x: ArrayLike,
           loc: ArrayLike = 0,
           scale: ArrayLike = 1) -> Array:
  r"""
  Gumbel Distribution (Left Skewed) log cumulative density function.

  JAX implementation of :obj:`scipy.stats.gumbel_l` ``logcdf``.

  .. math::

      f_{cdf}(x; \mu, \beta) = 1 - \exp\left( -\exp\left( \frac{x - \mu}{\beta} \right) \right)

  Args:
    x: ArrayLike, value at which to evaluate log(cdf)
    loc: ArrayLike, distribution offset (:math:`\mu`) (defaulted to 0)
    scale: ArrayLike, distribution scaling (:math:`\beta`) (defaulted to 1)

  Returns:
    array of logcdf values

  See Also:
    - :func:`jax.scipy.stats.gumbel_l.logpdf`
    - :func:`jax.scipy.stats.gumbel_l.pdf`
    - :func:`jax.scipy.stats.gumbel_l.cdf`
    - :func:`jax.scipy.stats.gumbel_l.ppf`
    - :func:`jax.scipy.stats.gumbel_l.logsf`
    - :func:`jax.scipy.stats.gumbel_l.sf`
  """
  x, loc, scale = promote_args_inexact("gumbel_l.logcdf", x, loc, scale)
  ok = lax.gt(scale, _lax_const(scale, 0))
  z = lax.div(lax.sub(x, loc), scale)
  neg_exp_z = lax.neg(lax.exp(z))
  # xlog1p fails here, that's why log1p is used here
  # even log1p fails for some cases when using float64 mode
  # so we're using this formula which is stable
  log_cdf = lax.log(-lax.expm1(neg_exp_z))
  return jnp.where(ok, log_cdf, np.nan)


def cdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""
  Gumbel Distribution (Left Skewed) cumulative density function.

  JAX implementation of :obj:`scipy.stats.gumbel_l` ``cdf``.

  .. math::

      f_{cdf}(x; \mu, \beta) = 1 - \exp\left( -\exp\left( \frac{x - \mu}{\beta} \right) \right)

  Args:
    x: ArrayLike, value at which to evaluate cdf
    loc: ArrayLike, distribution offset (:math:`\mu`) (defaulted to 0)
    scale: ArrayLike, distribution scaling (:math:`\beta`) (defaulted to 1)

  Returns:
    array of cdf values

  See Also:
    - :func:`jax.scipy.stats.gumbel_l.logpdf`
    - :func:`jax.scipy.stats.gumbel_l.pdf`
    - :func:`jax.scipy.stats.gumbel_l.logcdf`
    - :func:`jax.scipy.stats.gumbel_l.ppf`
    - :func:`jax.scipy.stats.gumbel_l.logsf`
    - :func:`jax.scipy.stats.gumbel_l.sf`
  """
  return lax.exp(logcdf(x, loc, scale))


def ppf(p: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""
  Gumbel Distribution (Left Skewed) percent point function (inverse of CDF)

  JAX implementation of :obj:`scipy.stats.gumbel_l` ``ppf``.

  .. math::

      F_{ppf}}(p; \mu, \beta) = \mu + \beta \log\left( -\log(1 - p) \right)

  Args:
    p: ArrayLike, probability value (quantile) at which to evaluate ppf
    loc: ArrayLike, distribution offset (:math:`\mu`) (defaulted to 0)
    scale: ArrayLike, distribution scaling (:math:`\beta`) (defaulted to 1)

  Returns:
    array of ppf values

  See Also:
    - :func:`jax.scipy.stats.gumbel_l.logpdf`
    - :func:`jax.scipy.stats.gumbel_l.pdf`
    - :func:`jax.scipy.stats.gumbel_l.logcdf`
    - :func:`jax.scipy.stats.gumbel_l.cdf`
    - :func:`jax.scipy.stats.gumbel_l.logsf`
    - :func:`jax.scipy.stats.gumbel_l.sf`
  """
  p, loc, scale = promote_args_inexact("gumbel_l.ppf", p, loc, scale)
  ok = lax.bitwise_and(lax.gt(p, _lax_const(p, 0)),
                       lax.lt(p, _lax_const(p, 1)))
  # quantile = loc + (scale)*log(-log(1 - p))
  t1 = xlog1py(-1, lax.neg(p))
  # xlogp failed here too, that's why log is used
  t = lax.mul(scale, lax.log(t1))
  quantile = lax.add(loc, t)
  return jnp.where(ok, quantile, np.nan)


def sf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""
  Gumbel Distribution (Left Skewed) survival function.

  JAX implementation of :obj:`scipy.stats.gumbel_l` ``sf``.

  .. math::

      f_{sf}(x; \mu, \beta) = 1 - f_{cdf}(x, \mu, \beta)

  Args:
    x: ArrayLike, value at which to evaluate survival function
    loc: ArrayLike, distribution offset (:math:`\mu`) (defaulted to 0)
    scale: ArrayLike, distribution scaling (:math:`\beta`) (defaulted to 1)

  Returns:
    array of sf values (1 - cdf)

  See Also:
    - :func:`jax.scipy.stats.gumbel_l.logpdf`
    - :func:`jax.scipy.stats.gumbel_l.pdf`
    - :func:`jax.scipy.stats.gumbel_l.logcdf`
    - :func:`jax.scipy.stats.gumbel_l.cdf`
    - :func:`jax.scipy.stats.gumbel_l.logsf`
  """
  return jnp.exp(logsf(x, loc, scale))


def logsf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""
  Gumbel Distribution (Left Skewed) log survival function.

  JAX implementation of :obj:`scipy.stats.gumbel_l` ``logsf``.

  .. math::

      f_{sf}(x; \mu, \beta) = 1 - f_{cdf}(x, \mu, \beta)

  Args:
    x: ArrayLike, value at which to evaluate log survival function
    loc: ArrayLike, distribution offset (:math:`\mu`) (defaulted to 0)
    scale: ArrayLike, distribution scaling (:math:`\beta`) (defaulted to 1)

  Returns:
    array of logsf values

  See Also:
    - :func:`jax.scipy.stats.gumbel_l.logpdf`
    - :func:`jax.scipy.stats.gumbel_l.pdf`
    - :func:`jax.scipy.stats.gumbel_l.logcdf`
    - :func:`jax.scipy.stats.gumbel_l.cdf`
    - :func:`jax.scipy.stats.gumbel_l.sf`
  """
  x, loc, scale = promote_args_inexact("gumbel_l.logsf", x, loc, scale)
  ok = lax.gt(scale, _lax_const(scale, 0))
  # logsf = -exp(z)
  z = lax.div(lax.sub(x, loc), scale)
  log_sf = lax.neg(lax.exp(z))
  return jnp.where(ok, log_sf, np.nan)
