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
from jax._src.numpy.util import implements, promote_args_inexact
from jax._src.typing import Array, ArrayLike


@implements(osp_stats.gennorm.logpdf, update_doc=False)
def logpdf(x: ArrayLike, p: ArrayLike) -> Array:
  x, p = promote_args_inexact("gennorm.logpdf", x, p)
  return lax.log(.5 * p) - lax.lgamma(1/p) - lax.abs(x)**p

@implements(osp_stats.gennorm.cdf, update_doc=False)
def cdf(x: ArrayLike, p: ArrayLike) -> Array:
  x, p = promote_args_inexact("gennorm.cdf", x, p)
  return .5 * (1 + lax.sign(x) * lax.igamma(1/p, lax.abs(x)**p))

@implements(osp_stats.gennorm.pdf, update_doc=False)
def pdf(x: ArrayLike, p: ArrayLike) -> Array:
  return lax.exp(logpdf(x, p))
