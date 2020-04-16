# Copyright 2018 Google LLC
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
from jax.numpy import lax_numpy as jnp
from jax.scipy import special
from jax.scipy.stats._freezing import Freezer, _required


freeze = Freezer(__name__.split(".")[-1], p=_required, loc=0)


@freeze.wrap
@jnp._wraps(osp_stats.bernoulli.logpmf, update_doc=False)
def logpmf(k, p, loc=0):
  k, p, loc = jnp._promote_args_inexact("bernoulli.logpmf", k, p, loc)
  zero = jnp._constant_like(k, 0)
  one = jnp._constant_like(k, 1)
  x = lax.sub(k, loc)
  log_probs = special.xlogy(x, p) + special.xlog1py(lax.sub(one, x), -p)
  return jnp.where(jnp.logical_or(lax.lt(x, zero), lax.gt(x, one)),
                   -jnp.inf, log_probs)


@freeze.wrap
@jnp._wraps(osp_stats.bernoulli.pmf, update_doc=False)
def pmf(k, p, loc=0):
  return jnp.exp(logpmf(k, p, loc))
