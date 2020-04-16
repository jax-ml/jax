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


freeze = Freezer(__name__.split(".")[-1], a=_required, b=_required, loc=0, scale=1)


@freeze.wrap
@jnp._wraps(osp_stats.beta.logpdf, update_doc=False)
def logpdf(x, a, b, loc=0, scale=1):
  x, a, b, loc, scale = jnp._promote_args_inexact("beta.logpdf", x, a, b, loc, scale)
  one = jnp._constant_like(x, 1)
  shape_term = lax.neg(special.betaln(a, b))
  y = lax.div(lax.sub(x, loc), scale)
  log_linear_term = lax.add(lax.mul(lax.sub(a, one), lax.log(y)),
                            lax.mul(lax.sub(b, one), lax.log1p(lax.neg(y))))
  log_probs = lax.sub(lax.add(shape_term, log_linear_term), lax.log(scale))
  return jnp.where(jsp.logical_or(lax.gt(x, lax.add(loc, scale)),
                                  lax.lt(x, loc)), -jnp.inf, log_probs)


@freeze.wrap
@jnp._wraps(osp_stats.beta.pdf, update_doc=False)
def pdf(x, a, b, loc=0, scale=1):
  return lax.exp(logpdf(x, a, b, loc, scale))
