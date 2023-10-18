# Copyright 2023 The JAX Authors.
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


@_wraps(osp_stats.wrapcauchy.logpdf, update_doc=False)
def logpdf(x: ArrayLike, c: ArrayLike) -> Array:
  x, c = promote_args_inexact('wrapcauchy.logpdf', x, c)
  return jnp.where(
    lax.gt(c, _lax_const(c, 0)) & lax.lt(c, _lax_const(c, 1)),
    jnp.where(
      lax.ge(x, _lax_const(x, 0)) & lax.le(x, _lax_const(x, jnp.pi * 2)),
      jnp.log(1 - c * c) - jnp.log(2 * jnp.pi) - jnp.log(1 + c * c - 2 * c * jnp.cos(x)),
      -jnp.inf,
    ),
    jnp.nan,
  )

@_wraps(osp_stats.wrapcauchy.pdf, update_doc=False)
def pdf(x: ArrayLike, c: ArrayLike) -> Array:
  return lax.exp(logpdf(x, c))
