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
from jax.scipy.special import gammaln, xlogy, gammainc, gammaincc


@_wraps(osp_stats.gamma.logpdf, update_doc=False)
def logpdf(x: ArrayLike, a: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, a, loc, scale = promote_args_inexact("gamma.logpdf", x, a, loc, scale)
  one = _lax_const(x, 1)
  y = lax.div(lax.sub(x, loc), scale)
  log_linear_term = lax.sub(xlogy(lax.sub(a, one), y), y)
  shape_terms = lax.add(gammaln(a), lax.log(scale))
  log_probs = lax.sub(log_linear_term, shape_terms)
  return jnp.where(lax.lt(x, loc), -jnp.inf, log_probs)

@_wraps(osp_stats.gamma.pdf, update_doc=False)
def pdf(x: ArrayLike, a: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  return lax.exp(logpdf(x, a, loc, scale))


@_wraps(osp_stats.gamma.cdf, update_doc=False)
def cdf(x: ArrayLike, a: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, a, loc, scale = promote_args_inexact("gamma.cdf", x, a, loc, scale)
  return gammainc(
    a,
    lax.clamp(
      _lax_const(x, 0),
      lax.div(lax.sub(x, loc), scale),
      _lax_const(x, jnp.inf),
    )
  )


@_wraps(osp_stats.gamma.logcdf, update_doc=False)
def logcdf(x: ArrayLike, a: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  return lax.log(cdf(x, a, loc, scale))


@_wraps(osp_stats.gamma.sf, update_doc=False)
def sf(x: ArrayLike, a: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, a, loc, scale = promote_args_inexact("gamma.sf", x, a, loc, scale)
  return gammaincc(a, lax.div(lax.sub(x, loc), scale))
