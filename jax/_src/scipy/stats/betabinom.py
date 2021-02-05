# Copyright 2021 Google LLC
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


import scipy
import scipy.stats as osp_stats

from jax import lax
from jax._src.numpy.util import _wraps
from jax._src.numpy.lax_numpy import _promote_args_inexact, _constant_like, where, inf, logical_or, nan
from jax._src.scipy.special import betaln

scipy_version = tuple(map(int, scipy.version.version.split('.')[:2]))


def logpmf(k, n, a, b, loc=0):
  """JAX implementation of scipy.stats.betabinom.logpmf."""
  k, n, a, b, loc = _promote_args_inexact("betabinom.logpmf", k, n, a, b, loc)
  y = lax.sub(lax.floor(k), loc)
  one = _constant_like(y, 1)
  zero = _constant_like(y, 0)
  combiln = lax.neg(lax.add(lax.log1p(n), betaln(lax.add(lax.sub(n,y), one), lax.add(y,one))))
  beta_lns = lax.sub(betaln(lax.add(y,a), lax.add(lax.sub(n,y),b)), betaln(a,b))
  log_probs = lax.add(combiln, beta_lns)
  y_cond = logical_or(lax.lt(y, lax.neg(loc)), lax.gt(y, lax.sub(n, loc)))
  log_probs = where(y_cond, -inf, log_probs)
  n_a_b_cond = logical_or(logical_or(lax.lt(n, one), lax.lt(a, zero)), lax.lt(b, zero))
  return where(n_a_b_cond, nan, log_probs)


def pmf(k, n, a, b, loc=0):
  """JAX implementation of scipy.stats.betabinom.pmf."""
  return lax.exp(logpmf(k, n, a, b, loc))


# betabinom was added in scipy 1.4.0
if scipy_version >= (1, 4):
  logpmf = _wraps(osp_stats.betabinom.logpmf, update_doc=False)(logpmf)
  pmf = _wraps(osp_stats.betabinom.pmf, update_doc=False)(pmf)
