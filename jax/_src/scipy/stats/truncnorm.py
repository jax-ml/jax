# Copyright 2022 The JAX Authors.
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


import scipy.stats as osp_stats

from jax import lax
import jax.numpy as jnp
from jax._src.numpy.util import implements, promote_args_inexact
from jax._src.scipy.stats import norm
from jax._src.scipy.special import logsumexp, log_ndtr, ndtr


def _log_diff(x, y):
  return logsumexp(
    jnp.array([x, y]),
    b=jnp.array([jnp.ones_like(x), -jnp.ones_like(y)]),
    axis=0
  )


def _log_gauss_mass(a, b):
  """Log of Gaussian probability mass within an interval"""
  a, b = jnp.array(a), jnp.array(b)
  a, b = jnp.broadcast_arrays(a, b)

  # Note: Docstring carried over from scipy
  # Calculations in right tail are inaccurate, so we'll exploit the
  # symmetry and work only in the left tail
  case_left = b <= 0
  case_right = a > 0
  case_central = ~(case_left | case_right)

  def mass_case_left(a, b):
    return _log_diff(log_ndtr(b), log_ndtr(a))

  def mass_case_right(a, b):
    return mass_case_left(-b, -a)

  def mass_case_central(a, b):
    # Note: Docstring carried over from scipy
    # Previously, this was implemented as:
    # left_mass = mass_case_left(a, 0)
    # right_mass = mass_case_right(0, b)
    # return _log_sum(left_mass, right_mass)
    # Catastrophic cancellation occurs as np.exp(log_mass) approaches 1.
    # Correct for this with an alternative formulation.
    # We're not concerned with underflow here: if only one term
    # underflows, it was insignificant; if both terms underflow,
    # the result can't accurately be represented in logspace anyway
    # because sc.log1p(x) ~ x for small x.
    return jnp.log1p(-ndtr(a) - ndtr(-b))

  out = jnp.select(
    [case_left, case_right, case_central],
    [mass_case_left(a, b), mass_case_right(a, b), mass_case_central(a, b)]
  )
  return out


@implements(osp_stats.truncnorm.logpdf, update_doc=False)
def logpdf(x, a, b, loc=0, scale=1):
  x, a, b, loc, scale = promote_args_inexact("truncnorm.logpdf", x, a, b, loc, scale)
  val = lax.sub(norm.logpdf(x, loc, scale), _log_gauss_mass(a, b))

  x_scaled = lax.div(lax.sub(x, loc), scale)
  val = jnp.where((x_scaled < a) | (x_scaled > b), -jnp.inf, val)
  val = jnp.where(a >= b, jnp.nan, val)
  return val


@implements(osp_stats.truncnorm.pdf, update_doc=False)
def pdf(x, a, b, loc=0, scale=1):
  return lax.exp(logpdf(x, a, b, loc, scale))


@implements(osp_stats.truncnorm.logsf, update_doc=False)
def logsf(x, a, b, loc=0, scale=1):
  x, a, b, loc, scale = promote_args_inexact("truncnorm.logsf", x, a, b, loc, scale)
  return logcdf(-x, -b, -a, -loc, scale)


@implements(osp_stats.truncnorm.sf, update_doc=False)
def sf(x, a, b, loc=0, scale=1):
  return lax.exp(logsf(x, a, b, loc, scale))


@implements(osp_stats.truncnorm.logcdf, update_doc=False)
def logcdf(x, a, b, loc=0, scale=1):
  x, a, b, loc, scale = promote_args_inexact("truncnorm.logcdf", x, a, b, loc, scale)
  x, a, b = jnp.broadcast_arrays(x, a, b)
  x = lax.div(lax.sub(x, loc), scale)
  logcdf = _log_gauss_mass(a, x) - _log_gauss_mass(a, b)
  logsf = _log_gauss_mass(x, b) - _log_gauss_mass(a, b)

  logcdf = jnp.select(
    # third condition: avoid catastrophic cancellation (from scipy)
    [x >= b, x <= a, logcdf > -0.1, x > a],
    [0, -jnp.inf, jnp.log1p(-jnp.exp(logsf)), logcdf]
  )
  logcdf = jnp.where(a >= b, jnp.nan, logcdf)
  return logcdf


@implements(osp_stats.truncnorm.cdf, update_doc=False)
def cdf(x, a, b, loc=0, scale=1):
  return lax.exp(logcdf(x, a, b, loc, scale))
