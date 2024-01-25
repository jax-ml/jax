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
from jax import numpy as jnp
from jax.numpy import where, inf, logical_or
from jax._src.typing import Array, ArrayLike
from jax._src.numpy.util import implements, promote_args_inexact


@implements(osp_stats.uniform.logpdf, update_doc=False)
def logpdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, loc, scale = promote_args_inexact("uniform.logpdf", x, loc, scale)
  log_probs = lax.neg(lax.log(scale))
  return where(logical_or(lax.gt(x, lax.add(loc, scale)),
                          lax.lt(x, loc)),
               -inf, log_probs)

@implements(osp_stats.uniform.pdf, update_doc=False)
def pdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  return lax.exp(logpdf(x, loc, scale))

@implements(osp_stats.uniform.cdf, update_doc=False)
def cdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, loc, scale = promote_args_inexact("uniform.cdf", x, loc, scale)
  zero, one = jnp.array(0, x.dtype), jnp.array(1, x.dtype)
  conds = [lax.lt(x, loc), lax.gt(x, lax.add(loc, scale)), lax.ge(x, loc) & lax.le(x, lax.add(loc, scale))]
  vals = [zero, one, lax.div(lax.sub(x, loc), scale)]

  return jnp.select(conds, vals)

@implements(osp_stats.uniform.ppf, update_doc=False)
def ppf(q: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  q, loc, scale = promote_args_inexact("uniform.ppf", q, loc, scale)
  return where(
    jnp.isnan(q) | (q < 0) | (q > 1),
    jnp.nan,
    lax.add(loc, lax.mul(scale, q))
  )
