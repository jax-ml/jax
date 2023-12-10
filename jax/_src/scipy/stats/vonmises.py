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
import jax.numpy as jnp
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import _wraps, promote_args_inexact
from jax._src.typing import Array, ArrayLike

@_wraps(osp_stats.vonmises.logpdf, update_doc=False)
def logpdf(x: ArrayLike, kappa: ArrayLike) -> Array:
  x, kappa = promote_args_inexact('vonmises.logpdf', x, kappa)
  zero = _lax_const(kappa, 0)
  return jnp.where(lax.gt(kappa, zero), kappa * (jnp.cos(x) - 1) - jnp.log(2 * jnp.pi * lax.bessel_i0e(kappa)), jnp.nan)

@_wraps(osp_stats.vonmises.pdf, update_doc=False)
def pdf(x: ArrayLike, kappa: ArrayLike) -> Array:
  return lax.exp(logpdf(x, kappa))
