# Copyright 2020 Google LLC
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

from functools import wraps
from typing import Sequence, Union

import numpy as np
from .lax import SparseArray, BSR, COO, CSR, ELL
import jax.random as _random
from jax import dtypes
import jax.numpy as jnp

_FMTS = {'bsr': BSR, 'coo': COO, 'csr': CSR, 'ell': ELL}

PRNGKey = _random.PRNGKey

def _sparse_rng(func):
  @wraps(func)
  def wrapped(key, *args, nnz=0.1, fmt="csr", **kwargs):
    # TODO(jakevdp): avoid creating dense array.
    key, key2 = _random.split(key)
    mat = func(key, *args, **kwargs)
    nnz = int(nnz * mat.size if 0 < nnz < 1 else nnz)

    if nnz <= 0:
      mat = np.zeros_like(mat)
    elif nnz < mat.size:
      mask = _random.shuffle(key2, jnp.arange(mat.size)).reshape(mat.shape)
      mat = jnp.where(mask < nnz, mat, 0)

    if fmt == 'dense':
      return mat
    elif fmt in _FMTS:
      return _FMTS[fmt].fromdense(mat)
    else:
      raise ValueError(f"Unrecognized format: {fmt}")
  return wrapped


@_sparse_rng
def uniform(key: jnp.ndarray,
            shape: Sequence[int] = (),
            dtype: np.dtype = dtypes.float_,
            minval: Union[float, jnp.ndarray] = 0.,
            maxval: Union[float, jnp.ndarray] = 1.,
            *,
            nnz: Union[int, float] = 0.1,
            fmt: str = "csr") -> SparseArray:
  """Sparse Uniform Array"""
  return _random.uniform(key, shape=shape, dtype=dtype, minval=minval, maxval=maxval)
