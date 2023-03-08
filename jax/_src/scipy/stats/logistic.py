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
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import _wraps
from jax._src.numpy.lax_numpy import _promote_args_inexact
from jax._src.numpy import lax_numpy as jnp
from jax._src.typing import Array, ArrayLike


@_wraps(osp_stats.logistic.logpdf, update_doc=False)
def logpdf(x: ArrayLike) -> Array:
  x, = _promote_args_inexact("logistic.logpdf", x)
  two = _lax_const(x, 2)
  half_x = lax.div(x, two)
  return lax.mul(lax.neg(two), jnp.logaddexp(half_x, lax.neg(half_x)))


@_wraps(osp_stats.logistic.pdf, update_doc=False)
def pdf(x: ArrayLike) -> Array:
  return lax.exp(logpdf(x))

@_wraps(osp_stats.logistic.ppf, update_doc=False)
def ppf(x):
  return logit(x)

@_wraps(osp_stats.logistic.sf, update_doc=False)
def sf(x):
  return expit(lax.neg(x))

@_wraps(osp_stats.logistic.isf, update_doc=False)
def isf(x):
  return -logit(x)

@_wraps(osp_stats.logistic.cdf, update_doc=False)
def cdf(x):
  return expit(x)
