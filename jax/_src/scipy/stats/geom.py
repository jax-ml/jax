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

from jax import lax
import jax.numpy as jnp
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import implements, promote_args_inexact
from jax.scipy.special import xlog1py
from jax._src.typing import Array, ArrayLike


@implements(osp_stats.geom.logpmf, update_doc=False)
def logpmf(k: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
    k, p, loc = promote_args_inexact("geom.logpmf", k, p, loc)
    zero = _lax_const(k, 0)
    one = _lax_const(k, 1)
    x = lax.sub(k, loc)
    log_probs = xlog1py(lax.sub(x, one), -p) + lax.log(p)
    return jnp.where(lax.le(x, zero), -jnp.inf, log_probs)


@implements(osp_stats.geom.pmf, update_doc=False)
def pmf(k: ArrayLike, p: ArrayLike, loc: ArrayLike = 0) -> Array:
  return jnp.exp(logpmf(k, p, loc))
