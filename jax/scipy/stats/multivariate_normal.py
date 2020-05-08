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
from ...lax_linalg import cholesky, triangular_solve
from ... import numpy as jnp
from jax._src.numpy._util import _wraps
from jax._src.numpy.lax_numpy import _promote_dtypes_inexact, _constant_like


@_wraps(osp_stats.multivariate_normal.logpdf, update_doc=False)
def logpdf(x, mean, cov):
  x, mean, cov = _promote_dtypes_inexact(x, mean, cov)
  if not mean.shape:
    return -1/2 * (x - mean) ** 2 / cov - 1/2 * (np.log(2*np.pi) + jnp.log(cov))
  else:
    n = mean.shape[-1]
    if not np.shape(cov):
      y = x - mean
      return (-1/2 * jnp.einsum('...i,...i->...', y, y) / cov
              - n/2 * (np.log(2*np.pi) + jnp.log(cov)))
    else:
      if cov.ndim < 2 or cov.shape[-2:] != (n, n):
        raise ValueError("multivariate_normal.logpdf got incompatible shapes")
      L = cholesky(cov)
      y = triangular_solve(L, x - mean, lower=True, transpose_a=True)
      return (-1/2 * jnp.einsum('...i,...i->...', y, y) - n/2*np.log(2*np.pi)
              - jnp.log(L.diagonal()).sum())

@_wraps(osp_stats.multivariate_normal.pdf, update_doc=False)
def pdf(x, mean, cov):
  return lax.exp(logpdf(x, mean, cov))
