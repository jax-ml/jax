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
from jax._src.numpy.util import _wraps
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.lax_numpy import _promote_args_inexact, where, inf, logical_or
from jax._src.typing import Array, ArrayLike


@_wraps(osp_stats.uniform.logpdf, update_doc=False)
def logpdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, loc, scale = _promote_args_inexact("uniform.logpdf", x, loc, scale)
  log_probs = lax.neg(lax.log(scale))
  return where(logical_or(lax.gt(x, lax.add(loc, scale)),
                          lax.lt(x, loc)),
               -inf, log_probs)

@_wraps(osp_stats.uniform.pdf, update_doc=False)
def pdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  return lax.exp(logpdf(x, loc, scale))

@_wraps(osp_stats.uniform.cdf, update_doc=False)
def cdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, loc, scale = _promote_args_inexact("uniform.cdf", x, loc, scale)
  zero = _lax_const(x, 0)
  one = _lax_const(x, 1)
  if lax.gt(x, lax.add(loc, scale)):
    return one
  elif lax.lt(x, loc):
    return zero
  else:
    return lax.div(lax.sub(x, loc), scale)

@_wraps(osp_stats.uniform.ppf, update_doc=False)
def ppf(q: ArrayLike, loc : ArrayLike = 0, scale : ArrayLike = 1) -> Array:
  q, loc, scale = _promote_args_inexact("uniform.ppf", q, loc, scale)
  zero = _lax_const(q, 0)
  return where(logical_or(lax.gt(q, lax.add(loc, scale)),lax.lt(q, loc)),zero, lax.add(lax.mul(q, scale), loc))
