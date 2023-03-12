# Copyright 2020 The JAX Authors.
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
from jax.scipy.special import expit, logit

from jax import lax
import jax.numpy as jnp
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import _wraps, promote_args_inexact
from jax._src.typing import Array, ArrayLike


@_wraps(osp_stats.logistic.logpdf, update_doc=False)
def logpdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, loc, scale = promote_args_inexact("logistic.logpdf", x, loc, scale)
  x = lax.div(lax.sub(x, loc), scale)
  two = _lax_const(x, 2)
  half_x = lax.div(x, two)
  return lax.sub(lax.mul(lax.neg(two), jnp.logaddexp(half_x, lax.neg(half_x))), lax.log(scale))


@_wraps(osp_stats.logistic.pdf, update_doc=False)
def pdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  return lax.exp(logpdf(x, loc, scale))


@_wraps(osp_stats.logistic.ppf, update_doc=False)
def ppf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, loc, scale = promote_args_inexact("logistic.ppf", x, loc, scale)
  return lax.add(lax.mul(logit(x), scale), loc)


@_wraps(osp_stats.logistic.sf, update_doc=False)
def sf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, loc, scale = promote_args_inexact("logistic.sf", x, loc, scale)
  return expit(lax.neg(lax.div(lax.sub(x, loc), scale)))


@_wraps(osp_stats.logistic.isf, update_doc=False)
def isf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, loc, scale = promote_args_inexact("logistic.isf", x, loc, scale)
  return lax.add(lax.mul(lax.neg(logit(x)), scale), loc)


@_wraps(osp_stats.logistic.cdf, update_doc=False)
def cdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, loc, scale = promote_args_inexact("logistic.cdf", x, loc, scale)
  return expit(lax.div(lax.sub(x, loc), scale))
