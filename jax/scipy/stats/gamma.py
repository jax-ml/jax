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


freeze = Freezer(__name__.split(".")[-1], a=_required, loc=0, scale=1)


@freeze.wrap
@jnp._wraps(osp_stats.gamma.logpdf, update_doc=False)
def logpdf(x, a, loc=0, scale=1):
  x, a, loc, scale = jnp._promote_args_inexact("gamma.logpdf", x, a, loc, scale)
  one = jnp._constant_like(x, 1)
  y = lax.div(lax.sub(x, loc), scale)
  log_linear_term = lax.sub(lax.mul(lax.sub(a, one), lax.log(y)), y)
  shape_terms = lax.add(special.gammaln(a), lax.log(scale))
  log_probs = lax.sub(log_linear_term, shape_terms)
  return jnp.where(lax.lt(x, loc), -jnp.inf, log_probs)


@freeze.wrap
@jnp._wraps(osp_stats.gamma.pdf, update_doc=False)
def pdf(x, a, loc=0, scale=1):
  return lax.exp(logpdf(x, a, loc, scale))
