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


import scipy.stats as osp_stats

from jax import lax
import jax.numpy as jnp
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import _wraps, promote_args_inexact
from jax._src.typing import Array, ArrayLike
from jax.scipy.special import xlogy, xlog1py


@_wraps(osp_stats.bernoulli.logpmf, update_doc=False)
def logpmf(k: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
  k, p, loc = promote_args_inexact("bernoulli.logpmf", k, p, loc)
  zero = _lax_const(k, 0)
  one = _lax_const(k, 1)
  x = lax.sub(k, loc)
  log_probs = xlogy(x, p) + xlog1py(lax.sub(one, x), -p)
  return jnp.where(jnp.logical_or(lax.lt(x, zero), lax.gt(x, one)),
                  -jnp.inf, log_probs)

@_wraps(osp_stats.bernoulli.pmf, update_doc=False)
def pmf(k: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
  return jnp.exp(logpmf(k, p, loc))

@_wraps(osp_stats.bernoulli.cdf, update_doc=False)
def cdf(k: ArrayLike, p: ArrayLike) -> Array:
  k, p = promote_args_inexact('bernoulli.cdf', k, p)
  zero, one = _lax_const(k, 0), _lax_const(k, 1)
  conds = [
    jnp.isnan(k) | jnp.isnan(p) | (p < zero) | (p > one),
    lax.lt(k, zero),
    jnp.logical_and(lax.ge(k, zero), lax.lt(k, one)),
    lax.ge(k, one)
    ]
  vals = [jnp.nan, zero, one - p, one]
  return jnp.select(conds, vals)

@_wraps(osp_stats.bernoulli.ppf, update_doc=False)
def ppf(q: ArrayLike, p: ArrayLike) -> Array:
  q, p = promote_args_inexact('bernoulli.ppf', q, p)
  zero, one = _lax_const(q, 0), _lax_const(q, 1)
  return jnp.where(
    jnp.isnan(q) | jnp.isnan(p) | (p < zero) | (p > one) | (q < zero) | (q > one),
    jnp.nan,
    jnp.where(lax.le(q, one - p), zero, one)
  )
