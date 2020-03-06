# Copyright 2018 Google LLC
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

from ... import lax
from ...numpy import lax_numpy as jnp
from ..special import gammaln, xlogy


def _is_simplex(x):
    x_sum = jnp.sum(x, axis=-1)
    return jnp.all(x > 0, axis=-1) & (x_sum <= 1) & (x_sum > 1 - 1e-6)


@jnp._wraps(osp_stats.dirichlet.logpdf, update_doc=False)
def logpdf(x, alpha):
    args = (np.ones((0,), lax.dtype(x)), np.ones((1,), lax.dtype(alpha)))
    to_dtype = lax.dtype(osp_stats.dirichlet.logpdf(*args))
    x, alpha = [lax.convert_element_type(arg, to_dtype) for arg in (x, alpha)]
    one = jnp._constant_like(x, 1)
    normalize_term = jnp.sum(gammaln(alpha), axis=-1) - gammaln(jnp.sum(alpha, axis=-1))
    log_probs = lax.sub(jnp.sum(xlogy(lax.sub(alpha, one), x), axis=-1), normalize_term)
    return jnp.where(_is_simplex(x), log_probs, -jnp.inf)


@jnp._wraps(osp_stats.dirichlet.pdf, update_doc=False)
def pdf(x, alpha):
  return lax.exp(logpdf(x, alpha))
