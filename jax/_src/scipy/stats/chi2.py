# Copyright 2021 The JAX Authors.
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
# limitations under the License


import scipy.stats as osp_stats

from jax import lax
import jax.numpy as jnp
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import _wraps, promote_args_inexact
from jax._src.typing import Array, ArrayLike
from jax.scipy.special import gammainc, gammaincc


@_wraps(osp_stats.chi2.logpdf, update_doc=False)
def logpdf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, df, loc, scale = promote_args_inexact("chi2.logpdf", x, df, loc, scale)
  one = _lax_const(x, 1)
  two = _lax_const(x, 2)
  y = lax.div(lax.sub(x, loc), scale)
  df_on_two = lax.div(df, two)

  kernel = lax.sub(lax.mul(lax.sub(df_on_two, one), lax.log(y)), lax.div(y,two))

  nrml_cnst = lax.neg(lax.add(lax.lgamma(df_on_two),lax.div(lax.mul(lax.log(two), df),two)))

  log_probs = lax.add(lax.sub(nrml_cnst, lax.log(scale)), kernel)
  return jnp.where(lax.lt(x, loc), -jnp.inf, log_probs)

@_wraps(osp_stats.chi2.pdf, update_doc=False)
def pdf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  return lax.exp(logpdf(x, df, loc, scale))


@_wraps(osp_stats.chi2.cdf, update_doc=False)
def cdf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, df, loc, scale = promote_args_inexact("chi2.cdf", x, df, loc, scale)
  two = _lax_const(scale, 2)
  return gammainc(
    lax.div(df, two),
    lax.clamp(
      _lax_const(x, 0),
      lax.div(
        lax.sub(x, loc),
        lax.mul(scale, two),
      ),
      _lax_const(x, jnp.inf),
    ),
  )


@_wraps(osp_stats.chi2.logcdf, update_doc=False)
def logcdf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  return lax.log(cdf(x, df, loc, scale))


@_wraps(osp_stats.chi2.sf, update_doc=False)
def sf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, df, loc, scale = promote_args_inexact("chi2.sf", x, df, loc, scale)
  two = _lax_const(scale, 2)
  return gammaincc(
    lax.div(df, two),
    lax.clamp(
      _lax_const(x, 0),
      lax.div(
        lax.sub(x, loc),
        lax.mul(scale, two),
      ),
      _lax_const(x, jnp.inf),
    ),
  )


@_wraps(osp_stats.chi2.logsf, update_doc=False)
def logsf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  return lax.log(sf(x, df, loc, scale))
