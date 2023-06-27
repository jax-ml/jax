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
# limitations under the License


import scipy.stats as osp_stats

from jax import lax
import jax.numpy as jnp
from jax._src.numpy.util import _wraps, promote_args_inexact
from jax._src.scipy.special import gammaln, xlogy, xlog1py
from jax._src.typing import Array, ArrayLike


@_wraps(osp_stats.nbinom.logpmf, update_doc=False)
def logpmf(k: ArrayLike, n: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
    """JAX implementation of scipy.stats.binom.logpmf."""
    k, n, p, loc = promote_args_inexact("binom.logpmf", k, n, p, loc)
    y = lax.sub(k, loc)
    comb_term = lax.sub(
        gammaln(n + 1),
        lax.add(gammaln(y + 1), gammaln(n - y + 1))
    )
    log_linear_term = lax.add(xlogy(y, p), xlog1py(lax.sub(n, y), lax.neg(p)))
    log_probs = lax.add(comb_term, log_linear_term)
    return jnp.where(lax.lt(k, loc), -jnp.inf, log_probs)


@_wraps(osp_stats.nbinom.pmf, update_doc=False)
def pmf(k: ArrayLike, n: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
    """JAX implementation of scipy.stats.binom.pmf."""
    return lax.exp(logpmf(k, n, p, loc))
