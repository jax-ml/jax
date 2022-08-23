# Copyright 2022 Google LLC
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
from jax._src.numpy import lax_numpy as jnp
from jax._src.numpy.lax_numpy import _promote_args_inexact
from jax._src.numpy.util import _wraps
from jax._src.scipy.special import gammaln, xlogy

@_wraps(osp_stats.multinomial.logpmf, update_doc=False)
def logpmf(x, n, p):
  """JAX implementation of scipy.stats.multinomial.logpmf."""
  x, p = _promote_args_inexact("multinomial.logpmf", x, p)
  logprobs = gammaln(n + 1) + jnp.sum(xlogy(x, p) - gammaln(x + 1), axis=-1)
  return jnp.where(jnp.equal(jnp.sum(x), n), logprobs, -jnp.inf)

@_wraps(osp_stats.multinomial.pmf, update_doc=False)
def pmf(x, n, p):
  """JAX implementation of scipy.stats.multinomial.pmf."""
  x, p = _promote_args_inexact("multinomial.pmf", x, p)
  return lax.exp(logpmf(x, n, p))
