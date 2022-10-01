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

"""BCSR (Bached compressed row) matrix object and associated primitives."""

from typing import Tuple

from jax import core
from jax.experimental.sparse._base import JAXSparse
from jax.experimental.sparse.util import _safe_asarray
import jax.numpy as jnp
from jax.util import split_list

Shape = Tuple[int, ...]


class BCSR(JAXSparse):
  """Experimental batched CSR matrix implemented in JAX."""

  data: jnp.ndarray
  indices: jnp.ndarray
  indptr: jnp.ndarray
  shape: Shape
  nse = property(lambda self: self.indices.shape[-1])
  dtype = property(lambda self: self.data.dtype)
  n_batch = property(lambda self: self.indices.ndim - 1)
  n_sparse = property(lambda _: 2)
  n_dense = property(lambda self: self.data.ndim - self.indices.ndim)

  @property
  def _sparse_shape(self):
    return tuple(self.shape[self.n_batch:self.n_batch + 2])

  def __init__(self, args, *, shape):
    # JAX transforms will sometimes instantiate pytrees with null values, so we
    # must catch that in the initialization of inputs.
    self.data, self.indices, self.indptr = _safe_asarray(args)
    super().__init__(args, shape=shape)

  def __repr__(self):
    name = self.__class__.__name__
    try:
      nse = self.nse
      n_batch = self.n_batch
      n_dense = self.n_dense
      dtype = self.dtype
      shape = list(self.shape)
    except Exception:  # pylint: disable=broad-except
      repr_ = f"{name}(<invalid>)"
    else:
      extra = f", nse={nse}"
      if n_batch: extra += f", n_batch={n_batch}"
      if n_dense: extra += f", n_dense={n_dense}"
      repr_ = f"{name}({dtype}{shape}{extra})"
    if isinstance(self.data, core.Tracer):
      repr_ = f"{type(self.data).__name__}[{repr_}]"
    return repr_

  def transpose(self, *args, **kwargs):
    raise NotImplementedError("Tranpose is not implemented.")

  def tree_flatten(self):
    return (self.data, self.indices, self.indptr), {}

  @classmethod
  def _empty(cls, shape, *, dtype=None, index_dtype='int32', n_dense=0,
             n_batch=0, nse=0):
    """Create an empty BCSR instance. Public method is sparse.empty()."""
    shape = tuple(shape)
    if n_dense < 0 or n_batch < 0 or nse < 0:
      raise ValueError(f"Invalid inputs: shape={shape}, n_dense={n_dense},"
                       f"n_batch={n_batch}, nse={nse}")
    n_sparse = len(shape) - n_dense - n_batch
    if n_sparse != 2:
      raise ValueError("BCSR sparse.empty: must have 2 sparse dimensions.")
    batch_shape, sparse_shape, dense_shape = split_list(shape,
                                                        [n_batch, n_sparse])
    data = jnp.zeros((*batch_shape, nse, *dense_shape), dtype)
    indices = jnp.full((*batch_shape, nse), jnp.array(sparse_shape[1]),
                       index_dtype)
    indptr = jnp.zeros((*batch_shape, sparse_shape[0] + 1), index_dtype)
    return cls((data, indices, indptr), shape=shape)
