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

from jax._src import lax
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import promote_args_inexact
from jax._src.typing import Array, ArrayLike


def logpdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Laplace log probability distribution function.

  JAX implementation of :obj:`scipy.stats.laplace` ``logpdf``.

  The Laplace probability distribution function is given by

  .. math::

     f(x) = \frac{1}{2} e^{-|x|}

  Args:
    x: arraylike, value at which to evaluate the PDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logpdf values.

  See Also:
    - :func:`jax.scipy.stats.laplace.cdf`
    - :func:`jax.scipy.stats.laplace.pdf`
  """
  x, loc, scale = promote_args_inexact("laplace.logpdf", x, loc, scale)
  two = _lax_const(x, 2)
  linear_term = lax.div(lax.abs(lax.sub(x, loc)), scale)
  return lax.neg(lax.add(linear_term, lax.log(lax.mul(two, scale))))


def pdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Laplace probability distribution function.

  JAX implementation of :obj:`scipy.stats.laplace` ``pdf``.

  The Laplace probability distribution function is given by

  .. math::

     f(x) = \frac{1}{2} e^{-|x|}

  Args:
    x: arraylike, value at which to evaluate the PDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of pdf values.

  See Also:
    - :func:`jax.scipy.stats.laplace.cdf`
    - :func:`jax.scipy.stats.laplace.logpdf`
  """
  return lax.exp(logpdf(x, loc, scale))


def cdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Laplace cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.laplace` ``cdf``.

  The cdf is defined as

  .. math::

     f_{cdf}(x, k) = \int_{-\infty}^x f_{pdf}(y, k)\mathrm{d}y

  where :math:`f_{pdf}` is the probability density function,
  :func:`jax.scipy.stats.laplace.pdf`.

  Args:
    x: arraylike, value at which to evaluate the CDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of cdf values.

  See Also:
    - :func:`jax.scipy.stats.laplace.pdf`
    - :func:`jax.scipy.stats.laplace.logpdf`
  """
  x, loc, scale = promote_args_inexact("laplace.cdf", x, loc, scale)
  half = _lax_const(x, 0.5)
  one = _lax_const(x, 1)
  zero = _lax_const(x, 0)
  diff = lax.div(lax.sub(x, loc), scale)
  return lax.select(lax.le(diff, zero),
                    lax.mul(half, lax.exp(diff)),
                    lax.sub(one, lax.mul(half, lax.exp(lax.neg(diff)))))


def logcdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Laplace log cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.laplace` ``logcdf``.

  The logcdf is defined as

  .. math::

     f_{logcdf}(x) = \log f_{cdf}(x)

  where :math:`f_{cdf}` is the cumulative distribution function,
  :func:`jax.scipy.stats.laplace.cdf`. The :math:`x \ge \mathrm{loc}` branch
  uses ``log1p`` to avoid catastrophic cancellation.

  Args:
    x: arraylike, value at which to evaluate the log-CDF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logcdf values.

  See Also:
    - :func:`jax.scipy.stats.laplace.cdf`
    - :func:`jax.scipy.stats.laplace.logpdf`
    - :func:`jax.scipy.stats.laplace.logsf`
  """
  x, loc, scale = promote_args_inexact("laplace.logcdf", x, loc, scale)
  half = _lax_const(x, 0.5)
  log_half = lax.log(half)
  zero = _lax_const(x, 0)
  diff = lax.div(lax.sub(x, loc), scale)
  return lax.select(lax.le(diff, zero),
                    lax.add(log_half, diff),
                    lax.log1p(lax.neg(lax.mul(half, lax.exp(lax.neg(diff))))))


def sf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Laplace survival function.

  JAX implementation of :obj:`scipy.stats.laplace` ``sf``.

  The survival function is defined as

  .. math::

     f_{sf}(x) = 1 - f_{cdf}(x)

  where :math:`f_{cdf}` is the cumulative distribution function,
  :func:`jax.scipy.stats.laplace.cdf`. The Laplace distribution is symmetric
  around ``loc``, so ``sf(x) == cdf(2 * loc - x)``.

  Args:
    x: arraylike, value at which to evaluate the SF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of sf values.

  See Also:
    - :func:`jax.scipy.stats.laplace.cdf`
    - :func:`jax.scipy.stats.laplace.logsf`
    - :func:`jax.scipy.stats.laplace.isf`
  """
  x, loc, scale = promote_args_inexact("laplace.sf", x, loc, scale)
  return cdf(lax.sub(lax.mul(_lax_const(x, 2), loc), x), loc, scale)


def logsf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Laplace log survival function.

  JAX implementation of :obj:`scipy.stats.laplace` ``logsf``.

  The log survival function is defined as

  .. math::

     f_{logsf}(x) = \log(1 - f_{cdf}(x))

  where :math:`f_{cdf}` is the cumulative distribution function,
  :func:`jax.scipy.stats.laplace.cdf`. The :math:`x \le \mathrm{loc}` branch
  uses ``log1p`` to avoid catastrophic cancellation.

  Args:
    x: arraylike, value at which to evaluate the log-SF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logsf values.

  See Also:
    - :func:`jax.scipy.stats.laplace.sf`
    - :func:`jax.scipy.stats.laplace.logcdf`
    - :func:`jax.scipy.stats.laplace.isf`
  """
  x, loc, scale = promote_args_inexact("laplace.logsf", x, loc, scale)
  return logcdf(lax.sub(lax.mul(_lax_const(x, 2), loc), x), loc, scale)


def ppf(q: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Laplace percent point function.

  JAX implementation of :obj:`scipy.stats.laplace` ``ppf``.

  The percent point function is the inverse of the cumulative distribution
  function, :func:`jax.scipy.stats.laplace.cdf`. In closed form,

  .. math::

     f_{ppf}(q) = \begin{cases}
       \mathrm{loc} + \mathrm{scale} \cdot \log(2 q) & q \le 1/2 \\
       \mathrm{loc} - \mathrm{scale} \cdot \log(2 - 2 q) & q > 1/2
     \end{cases}

  Args:
    q: arraylike, value at which to evaluate the PPF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of ppf values.

  See Also:
    - :func:`jax.scipy.stats.laplace.cdf`
    - :func:`jax.scipy.stats.laplace.pdf`
  """
  q, loc, scale = promote_args_inexact("laplace.ppf", q, loc, scale)
  half = _lax_const(q, 0.5)
  two = _lax_const(q, 2)
  centered = lax.sub(q, half)
  # ppf(q) = loc - scale * sign(q - 1/2) * log(1 - 2|q - 1/2|)
  log_term = lax.log1p(lax.neg(lax.mul(two, lax.abs(centered))))
  return lax.sub(loc, lax.mul(scale, lax.mul(lax.sign(centered), log_term)))


def isf(q: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Laplace inverse survival function.

  JAX implementation of :obj:`scipy.stats.laplace` ``isf``.

  The inverse survival function is the inverse of the survival function,
  :func:`jax.scipy.stats.laplace.sf`. By symmetry about ``loc``,
  ``isf(q) == 2 * loc - ppf(q)``.

  Args:
    q: arraylike, value at which to evaluate the ISF
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of isf values.

  See Also:
    - :func:`jax.scipy.stats.laplace.sf`
    - :func:`jax.scipy.stats.laplace.ppf`
  """
  q, loc, scale = promote_args_inexact("laplace.isf", q, loc, scale)
  return lax.sub(lax.mul(_lax_const(q, 2), loc), ppf(q, loc, scale))
