# Copyright 2020 The JAX Authors.
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
from jax._src.scipy.special import xlog1py
from jax._src.typing import Array, ArrayLike


def logpmf(k: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Geometric log probability mass function.

  JAX implementation of :obj:`scipy.stats.geom` ``logpmf``.

  The Geometric probability mass function is given by

  .. math::

     f(k) = (1 - p)^{k-1}p

  for :math:`k\ge 1` and :math:`0 \le p \le 1`.

  Args:
    k: arraylike, value at which to evaluate the PMF
    p: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of logpmf values.

  See Also:
    :func:`jax.scipy.stats.geom.pmf`
  """
  k, p, loc = promote_args_inexact("geom.logpmf", k, p, loc)
  zero = _lax_const(k, 0)
  one = _lax_const(k, 1)
  x = lax.sub(k, loc)
  log_probs = xlog1py(lax.sub(x, one), -p) + lax.log(p)
  return jnp.where(lax.le(x, zero), -np.inf, log_probs)


def pmf(k: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Geometric probability mass function.

  JAX implementation of :obj:`scipy.stats.geom` ``pmf``.

  The Geometric probability mass function is given by

  .. math::

     f(k) = (1 - p)^{k-1}p

  for :math:`k\ge 1` and :math:`0 \le p \le 1`.

  Args:
    k: arraylike, value at which to evaluate the PMF
    p: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of pmf values.

  See Also:
    - :func:`jax.scipy.stats.geom.cdf`
    - :func:`jax.scipy.stats.geom.logcdf`
    - :func:`jax.scipy.stats.geom.logpmf`
    - :func:`jax.scipy.stats.geom.ppf`
  """
  return jnp.exp(logpmf(k, p, loc))


def cdf(k: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Geometric cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.geom` ``cdf``.

  The cumulative distribution function is defined as:

  .. math::

     f_{cdf}(k, p) = \sum_{i=1}^{\lfloor k \rfloor} (1 - p)^{i - 1} p = 1 - (1 - p)^{\lfloor k \rfloor}

  for :math:`\lfloor k \rfloor \ge 1` and :math:`0 \le p \le 1`.

  Args:
    k: arraylike, value at which to evaluate the CDF
    p: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of cdf values.

  See Also:
    - :func:`jax.scipy.stats.geom.logcdf`
    - :func:`jax.scipy.stats.geom.logpmf`
    - :func:`jax.scipy.stats.geom.pmf`
    - :func:`jax.scipy.stats.geom.ppf`
    - :func:`jax.scipy.stats.geom.sf`
  """
  k, p, loc = promote_args_inexact("geom.cdf", k, p, loc)
  zero = _lax_const(k, 0)
  one = _lax_const(k, 1)
  x = lax.floor(lax.sub(k, loc))
  x_safe = jnp.where(lax.lt(x, one), zero, x)
  cdf_vals = lax.neg(lax.expm1(lax.mul(x_safe, lax.log1p(lax.neg(p)))))
  return jnp.where(lax.lt(x, one), zero, cdf_vals)


def logcdf(k: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Geometric log cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.geom` ``logcdf``.

  Args:
    k: arraylike, value at which to evaluate the log CDF
    p: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of logcdf values.

  See Also:
    - :func:`jax.scipy.stats.geom.cdf`
    - :func:`jax.scipy.stats.geom.logpmf`
    - :func:`jax.scipy.stats.geom.pmf`
    - :func:`jax.scipy.stats.geom.ppf`
  """
  k, p, loc = promote_args_inexact("geom.logcdf", k, p, loc)
  one = _lax_const(k, 1)
  x = lax.floor(lax.sub(k, loc))
  x_safe = jnp.where(lax.lt(x, one), one, x)
  logcdf_vals = lax.log1p(lax.neg(lax.exp(lax.mul(x_safe, lax.log1p(lax.neg(p))))))
  return jnp.where(lax.lt(x, one), -np.inf, logcdf_vals)


def sf(k: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Geometric survival function.

  JAX implementation of :obj:`scipy.stats.geom` ``sf``.

  The survival function is defined as :math:`1 - f_{cdf}(k, p) = (1 - p)^{\lfloor k \rfloor}`
  for :math:`\lfloor k \rfloor \ge 1`.

  Args:
    k: arraylike, value at which to evaluate the SF
    p: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of sf values.

  See Also:
    - :func:`jax.scipy.stats.geom.cdf`
    - :func:`jax.scipy.stats.geom.logsf`
  """
  k, p, loc = promote_args_inexact("geom.sf", k, p, loc)
  zero = _lax_const(k, 0)
  one = _lax_const(k, 1)
  x = lax.floor(lax.sub(k, loc))
  x_safe = jnp.where(lax.lt(x, one), zero, x)
  sf_vals = lax.exp(lax.mul(x_safe, lax.log1p(lax.neg(p))))
  return jnp.where(lax.lt(x, one), one, sf_vals)


def logsf(k: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Geometric log survival function.

  JAX implementation of :obj:`scipy.stats.geom` ``logsf``.

  Args:
    k: arraylike, value at which to evaluate the log SF
    p: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of logsf values.

  See Also:
    - :func:`jax.scipy.stats.geom.sf`
  """
  k, p, loc = promote_args_inexact("geom.logsf", k, p, loc)
  zero = _lax_const(k, 0)
  one = _lax_const(k, 1)
  x = lax.floor(lax.sub(k, loc))
  x_safe = jnp.where(lax.lt(x, one), zero, x)
  logsf_vals = lax.mul(x_safe, lax.log1p(lax.neg(p)))
  return jnp.where(lax.lt(x, one), zero, logsf_vals)


def ppf(q: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Geometric percent point function.

  JAX implementation of :obj:`scipy.stats.geom` ``ppf``.

  The percent point function is defined as the inverse of the
  cumulative distribution function, :func:`jax.scipy.stats.geom.cdf`.

  Args:
    q: arraylike, value at which to evaluate the PPF
    p: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of ppf values.

  See Also:
    - :func:`jax.scipy.stats.geom.cdf`
    - :func:`jax.scipy.stats.geom.logpmf`
    - :func:`jax.scipy.stats.geom.pmf`
  """
  q, p, loc = promote_args_inexact("geom.ppf", q, p, loc)
  zero = _lax_const(q, 0)
  one = _lax_const(q, 1)

  q_safe = jnp.where(jnp.logical_and(lax.gt(q, zero), lax.lt(q, one)), q, _lax_const(q, 0.5))
  ratio = lax.div(lax.log1p(lax.neg(q_safe)), lax.log1p(lax.neg(p)))
  res = lax.add(loc, lax.ceil(ratio))

  res = jnp.where(lax.eq(q, zero), loc, res)
  res = jnp.where(lax.eq(q, one), np.inf, res)
  return jnp.where(
      jnp.isnan(q) | jnp.isnan(p) | lax.lt(q, zero) | lax.gt(q, one) | lax.le(p, zero) | lax.gt(p, one),
      np.nan,
      res,
  )

