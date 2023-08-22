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

from typing import cast

import numpy as np
import scipy.stats as osp_stats

from jax import lax
import jax.numpy as jnp
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import _wraps, promote_args_inexact
from jax._src.typing import Array, ArrayLike
from jax.scipy import special


@_wraps(osp_stats.norm.logpdf, update_doc=False)
def logpdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, loc, scale = promote_args_inexact("norm.logpdf", x, loc, scale)
  scale_sqrd = lax.square(scale)
  log_normalizer = lax.log(lax.mul(_lax_const(x, 2 * np.pi), scale_sqrd))
  quadratic = lax.div(lax.square(lax.sub(x, loc)), scale_sqrd)
  return lax.div(lax.add(log_normalizer, quadratic), _lax_const(x, -2))


@_wraps(osp_stats.norm.pdf, update_doc=False)
def pdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  return lax.exp(logpdf(x, loc, scale))


@_wraps(osp_stats.norm.cdf, update_doc=False)
def cdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, loc, scale = promote_args_inexact("norm.cdf", x, loc, scale)
  return special.ndtr(lax.div(lax.sub(x, loc), scale))


@_wraps(osp_stats.norm.logcdf, update_doc=False)
def logcdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, loc, scale = promote_args_inexact("norm.logcdf", x, loc, scale)
  # Cast required because custom_jvp return type is broken.
  return cast(Array, special.log_ndtr(lax.div(lax.sub(x, loc), scale)))


@_wraps(osp_stats.norm.ppf, update_doc=False)
def ppf(q: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  return jnp.asarray(special.ndtri(q) * scale + loc, float)


@_wraps(osp_stats.norm.logsf, update_doc=False)
def logsf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, loc, scale = promote_args_inexact("norm.logsf", x, loc, scale)
  return logcdf(-x, -loc, scale)


@_wraps(osp_stats.norm.sf, update_doc=False)
def sf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, loc, scale = promote_args_inexact("norm.sf", x, loc, scale)
  return cdf(-x, -loc, scale)


@_wraps(osp_stats.norm.isf, update_doc=False)
def isf(q: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  return ppf(lax.sub(_lax_const(q, 1), q), loc, scale)
