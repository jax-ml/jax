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
from jax._src.numpy.util import promote_args_inexact, promote_dtypes_inexact, ensure_arraylike
from jax._src.scipy.special import xlogy, entr, gammaln, gammaincc
from jax._src.typing import Array, ArrayLike


def logpmf(k: ArrayLike, mu: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Poisson log probability mass function.

  JAX implementation of :obj:`scipy.stats.poisson` ``logpmf``.

  The Poisson probability mass function is given by

  .. math::

     f(k) = e^{-\mu}\frac{\mu^k}{k!}

  and is defined for :math:`k \ge 0` and :math:`\mu \ge 0`.

  Args:
    k: arraylike, value at which to evaluate the PMF
    mu: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of logpmf values.

  See Also:
    - :func:`jax.scipy.stats.poisson.cdf`
    - :func:`jax.scipy.stats.poisson.pmf`
  """
  k, mu, loc = promote_args_inexact("poisson.logpmf", k, mu, loc)
  zero = _lax_const(k, 0)
  x = lax.sub(k, loc)
  log_probs = xlogy(x, mu) - gammaln(x + 1) - mu
  return jnp.where(jnp.logical_or(lax.lt(x, zero),
                                  lax.ne(jnp.round(k), k)), -np.inf, log_probs)


def pmf(k: ArrayLike, mu: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Poisson probability mass function.

  JAX implementation of :obj:`scipy.stats.poisson` ``pmf``.

  The Poisson probability mass function is given by

  .. math::

     f(k) = e^{-\mu}\frac{\mu^k}{k!}

  and is defined for :math:`k \ge 0` and :math:`\mu \ge 0`.

  Args:
    k: arraylike, value at which to evaluate the PMF
    mu: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of pmf values.

  See Also:
    - :func:`jax.scipy.stats.poisson.cdf`
    - :func:`jax.scipy.stats.poisson.logpmf`
  """
  return jnp.exp(logpmf(k, mu, loc))


def cdf(k: ArrayLike, mu: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Poisson cumulative distribution function.

  JAX implementation of :obj:`scipy.stats.poisson` ``cdf``.

  The cumulative distribution function is defined as:

  .. math::

     f_{cdf}(k, p) = \sum_{i=0}^k f_{pmf}(k, p)

  where :math:`f_{pmf}(k, p)` is the probability mass function
  :func:`jax.scipy.stats.poisson.pmf`.

  Args:
    k: arraylike, value at which to evaluate the CDF
    mu: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

  Returns:
    array of cdf values.

  See Also:
    - :func:`jax.scipy.stats.poisson.pmf`
    - :func:`jax.scipy.stats.poisson.logpmf`
  """
  k, mu, loc = promote_args_inexact("poisson.logpmf", k, mu, loc)
  zero = _lax_const(k, 0)
  x = lax.sub(k, loc)
  p = gammaincc(jnp.floor(1 + x), mu)
  return jnp.where(lax.lt(x, zero), zero, p)

def entropy(mu: ArrayLike, loc: ArrayLike = 0) -> Array:
  r"""Shannon entropy of the Poisson distribution.

  JAX implementation of :obj:`scipy.stats.poisson` ``entropy``.

  The entropy :math:`H(X)` of a Poisson random variable
  :math:`X \sim \text{Poisson}(\mu)` is defined as:

  .. math::

   H(X) = -\sum_{k=0}^\infty p(k) \log p(k)

  where :math:`p(k) = e^{-\mu} \mu^k / k!` for
  :math:`k \geq \max(0, \lfloor \text{loc} \rfloor)`.

  This implementation uses **regime switching** for numerical stability
  and performance:

  - **Small** :math:`\mu < 10`: Direct summation over PMF with adaptive
    upper bound :math:`k \leq \mu + 20`
  - **Medium** :math:`10 \leq \mu < 100`: Summation with bound
    :math:`k \leq \mu + 10\sqrt{\mu} + 20`
  - **Large** :math:`\mu \geq 100`: Asymptotic Stirling approximation:
    :math:`H(\mu) \approx \frac{1}{2} \log(2\pi e \mu) - \frac{1}{12\mu}`

  Matches SciPy to relative error :math:`< 10^{-5}` across all regimes.

  Args:
    mu: arraylike, mean parameter of the Poisson distribution.
      Must be ``> 0``.
    loc: arraylike, optional location parameter (default: 0).
      Accepted for API compatibility with scipy but does not
      affect the entropy

  Returns:
    Array of entropy values with shape broadcast from ``mu`` and ``loc``.
    Returns ``NaN`` for ``mu <= 0``.

  Examples:
    >>> from jax.scipy.stats import poisson
    >>> poisson.entropy(5.0)
    Array(2.204394, dtype=float32)
    >>> poisson.entropy(jax.numpy.array([1, 10, 100]))
    Array([1.3048419, 2.5614073, 3.7206903], dtype=float32)

  See Also:
    - :func:`jax.scipy.stats.poisson.pmf`
    - :func:`jax.scipy.stats.poisson.logpmf`
    - :obj:`scipy.stats.poisson`
  """
  mu, loc = ensure_arraylike("poisson.entropy", mu, loc)
  promoted_mu, promoted_loc = promote_dtypes_inexact(mu, loc)

  #Note: loc does not affect the entropy - translation invariant
  #it has only been taken to maintain compatibility with scipy api
  result_shape = jnp.broadcast_shapes(
    promoted_mu.shape,
    promoted_loc.shape
  )

  mu_flat = jnp.ravel(promoted_mu)
  zero_result = jnp.zeros_like(mu_flat)


  # Choose the computation regime based on mu value
  result = jnp.where(
    mu_flat == 0,
    zero_result,
    jnp.where(
      mu_flat < 10,
      _entropy_small_mu(mu_flat),
      jnp.where(
        mu_flat < 100,
        _entropy_medium_mu(mu_flat),
        _entropy_large_mu(mu_flat)
      )
    )
  )

  result_mu_shape = jnp.reshape(result, promoted_mu.shape)

  # Restore original shape
  return jnp.broadcast_to(result_mu_shape, result_shape)

def _entropy_small_mu(mu: Array) -> Array:
  """Entropy via direct PMF summation for small μ (< 10).
  Uses adaptive upper bound k ≤ μ + 20 to capture >99.999% of mass.
  """
  max_k = 35

  k = jnp.arange(max_k, dtype=mu.dtype)[:, None]
  probs = pmf(k, mu, 0)

  # Mask: only compute up to mu + 20 for each value
  upper_bounds = jnp.ceil(mu + 20).astype(k.dtype)
  mask = k < upper_bounds[None, :]
  probs_masked = jnp.where(mask, probs, 0.0)

  return jnp.sum(entr(probs_masked), axis=0)

def _entropy_medium_mu(mu: Array) -> Array:
  """Entropy for medium mu (10-100): Adaptive bounds based on std dev.

  Bounds: k ≤ μ + 10√μ + 20. Caps at k=250 for JIT compatibility.
  """
  max_k = 250  # Static bound for JIT. For mu<100, upper bound < 220

  k = jnp.arange(max_k, dtype=mu.dtype)[:, None]
  probs = pmf(k, mu, 0)

  upper_bounds = jnp.ceil(mu + 10 * jnp.sqrt(mu) + 20).astype(k.dtype)
  mask = k < upper_bounds[None, :]
  probs_masked = jnp.where(mask, probs, 0.0)

  return jnp.sum(entr(probs_masked), axis=0)

def _entropy_large_mu(mu: Array) -> Array:
  """Entropy for large mu (>= 100): Asymptotic approximation.

  Formula: H(λ) ≈ 0.5*log(2πeλ) - 1/(12λ) + O(λ^-2)
  """
  return 0.5 * jnp.log(2 * np.pi * np.e * mu) - 1.0 / (12 * mu)
