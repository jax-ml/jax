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

from functools import partial

import numpy as np
import scipy.stats as osp_stats

from jax import lax
from jax import numpy as jnp
from jax._src.numpy.util import _wraps, promote_dtypes_inexact
from jax._src.typing import Array, ArrayLike


@_wraps(osp_stats.multivariate_normal.logpdf, update_doc=False, lax_description="""
In the JAX version, the `allow_singular` argument is not implemented.
""")
def logpdf(x: ArrayLike, mean: ArrayLike, cov: ArrayLike, allow_singular: None = None) -> ArrayLike:
  if allow_singular is not None:
    raise NotImplementedError("allow_singular argument of multivariate_normal.logpdf")
  x, mean, cov = promote_dtypes_inexact(x, mean, cov)
  if not mean.shape:
    return (-1/2 * jnp.square(x - mean) / cov
            - 1/2 * (jnp.log(2*np.pi) + jnp.log(cov)))
  else:
    n = mean.shape[-1]
    if not np.shape(cov):
      y = x - mean
      return (-1/2 * jnp.einsum('...i,...i->...', y, y) / cov
              - n/2 * (jnp.log(2*np.pi) + jnp.log(cov)))
    else:
      if cov.ndim < 2 or cov.shape[-2:] != (n, n):
        raise ValueError("multivariate_normal.logpdf got incompatible shapes")
      L = lax.linalg.cholesky(cov)
      y = jnp.vectorize(
        partial(lax.linalg.triangular_solve, lower=True, transpose_a=True),
        signature="(n,n),(n)->(n)"
      )(L, x - mean)
      return (-1/2 * jnp.einsum('...i,...i->...', y, y) - n/2 * jnp.log(2*np.pi)
              - jnp.log(L.diagonal(axis1=-1, axis2=-2)).sum(-1))

@_wraps(osp_stats.multivariate_normal.pdf, update_doc=False)
def pdf(x: ArrayLike, mean: ArrayLike, cov: ArrayLike) -> Array:
  return lax.exp(logpdf(x, mean, cov))
