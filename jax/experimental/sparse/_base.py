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
# limitations under the License.

"""Base JAX Sparse object."""
from typing import Tuple

from jax import core
import jax.numpy as jnp


class JAXSparse:
  """Base class for high-level JAX sparse objects."""
  data: jnp.ndarray
  shape: Tuple[int, ...]
  nse: property
  dtype: property

  @property
  def ndim(self):
    return len(self.shape)

  def __init__(self, args, *, shape):
    self.shape = tuple(shape)

  def __repr__(self):
    name = self.__class__.__name__
    try:
      nse = self.nse
      dtype = self.dtype
      shape = list(self.shape)
    except:
      repr_ = f"{name}(<invalid>)"
    else:
      repr_ = f"{name}({dtype}{shape}, nse={nse})"
    if isinstance(self.data, core.Tracer):
      repr_ = f"{type(self.data).__name__}[{repr_}]"
    return repr_

  def tree_flatten(self):
    raise NotImplementedError("tree_flatten")

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(children, **aux_data)

  def matvec(self, v):
    raise NotImplementedError("matvec")

  def matmat(self, B):
    raise NotImplementedError("matmat")

  def transpose(self, axes=None):
    raise NotImplementedError()

  @property
  def T(self):
    return self.transpose()

  def __matmul__(self, other):
    if isinstance(other, JAXSparse):
      raise NotImplementedError("matmul between two sparse objects.")
    other = jnp.asarray(other)
    if other.ndim == 1:
      return self.matvec(other)
    elif other.ndim == 2:
      return self.matmat(other)
    else:
      raise NotImplementedError(f"matmul with object of shape {other.shape}")

  def block_until_ready(self):
    for arg in self.tree_flatten()[0]:
      arg.block_until_ready()
    return self
