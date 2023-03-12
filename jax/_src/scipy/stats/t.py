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


import numpy as np
import scipy.stats as osp_stats

from jax import lax
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import _wraps, promote_args_inexact
from jax._src.typing import Array, ArrayLike


@_wraps(osp_stats.t.logpdf, update_doc=False)
def logpdf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, df, loc, scale = promote_args_inexact("t.logpdf", x, df, loc, scale)
  two = _lax_const(x, 2)
  scaled_x = lax.div(lax.sub(x, loc), scale)
  df_over_two = lax.div(df, two)
  df_plus_one_over_two = lax.add(df_over_two, _lax_const(x, 0.5))
  normalize_term_const = lax.mul(lax.mul(scale, scale), _lax_const(x, np.pi))
  normalize_term_tmp = lax.div(lax.log(lax.mul(normalize_term_const, df)), two)
  normalize_term = lax.sub(lax.add(lax.lgamma(df_over_two), normalize_term_tmp),
                           lax.lgamma(df_plus_one_over_two))
  quadratic = lax.div(lax.mul(scaled_x, scaled_x), df)
  return lax.neg(lax.add(normalize_term, lax.mul(df_plus_one_over_two, lax.log1p(quadratic))))


@_wraps(osp_stats.t.pdf, update_doc=False)
def pdf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  return lax.exp(logpdf(x, df, loc, scale))
