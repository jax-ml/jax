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
from jax._src.numpy.util import implements, promote_args_inexact
from jax._src.typing import Array, ArrayLike
from jax.scipy.special import xlogy, gammaln, gammaincc


@implements(osp_stats.poisson.logpmf, update_doc=False)
def logpmf(k: ArrayLike, mu: ArrayLike, loc: ArrayLike = 0) -> Array:
  k, mu, loc = promote_args_inexact("poisson.logpmf", k, mu, loc)
  zero = _lax_const(k, 0)
  x = lax.sub(k, loc)
  log_probs = xlogy(x, mu) - gammaln(x + 1) - mu
  return jnp.where(lax.lt(x, zero), -jnp.inf, log_probs)

@implements(osp_stats.poisson.pmf, update_doc=False)
def pmf(k: ArrayLike, mu: ArrayLike, loc: ArrayLike = 0) -> Array:
  return jnp.exp(logpmf(k, mu, loc))

@implements(osp_stats.poisson.cdf, update_doc=False)
def cdf(k: ArrayLike, mu: ArrayLike, loc: ArrayLike = 0) -> Array:
  k, mu, loc = promote_args_inexact("poisson.logpmf", k, mu, loc)
  zero = _lax_const(k, 0)
  x = lax.sub(k, loc)
  p = gammaincc(jnp.floor(1 + x), mu)
  return jnp.where(lax.lt(x, zero), zero, p)
