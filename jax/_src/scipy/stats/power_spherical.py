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


import jax.numpy as jnp

from jax import lax
from jax._src.lax.lax import _const as _lax_const
from jax._src.typing import Array, ArrayLike


def logpdf(mu: ArrayLike, kappa: ArrayLike) -> Array:
  t = jnp.promote_types(mu.dtype, kappa.dtype)
  mu, kappa = mu.astype(t), kappa.astype(t)
  zero, two = _lax_const(kappa, 0), _lax_const(kappa, 2)
  n = mu.shape[-1]
  b = (n - 1) / 2
  a = b + kappa
  logpdf = (lax.add(a, b) * lax.log(two)) + (lax.lgamma(a) - lax.lgamma(a + b)) + (b * lax.log(jnp.pi))
  return jnp.where(lax.ge(kappa, zero), logpdf, jnp.inf)

def pdf(mu: ArrayLike, kappa: ArrayLike) -> Array:
  return lax.exp(logpdf(mu, kappa))
