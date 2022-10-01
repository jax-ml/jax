# Copyright 2021 The JAX Authors.
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
import abc
from typing import Tuple

from jax import core
import jax.numpy as jnp
from jax._src import util


class JAXSparse(abc.ABC):
  """Base class for high-level JAX sparse objects."""
  data: jnp.ndarray
  shape: Tuple[int, ...]
  nse: property
  dtype: property

  # Ignore type because of https://github.com/python/mypy/issues/4266.
  __hash__ = None  # type: ignore

  @property
  def size(self):
    return util.prod(self.shape)

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

  @abc.abstractmethod
  def tree_flatten(self):
    ...

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(children, **aux_data)

  @abc.abstractmethod
  def transpose(self, axes=None):
    ...

  @property
  def T(self):
    return self.transpose()

  def block_until_ready(self):
    for arg in self.tree_flatten()[0]:
      arg.block_until_ready()
    return self

  # Not abstract methods because not all sparse classes implement them

  def sum(self, *args, **kwargs):
    raise NotImplementedError(f"{self.__class__}.sum")

  def __neg__(self):
    raise NotImplementedError(f"{self.__class__}.__neg__")

  def __pos__(self):
    raise NotImplementedError(f"{self.__class__}.__pos__")

  def __matmul__(self, other):
    raise NotImplementedError(f"{self.__class__}.__matmul__")

  def __rmatmul__(self, other):
    raise NotImplementedError(f"{self.__class__}.__rmatmul__")

  def __mul__(self, other):
    raise NotImplementedError(f"{self.__class__}.__mul__")

  def __rmul__(self, other):
    raise NotImplementedError(f"{self.__class__}.__rmul__")

  def __add__(self, other):
    raise NotImplementedError(f"{self.__class__}.__add__")

  def __radd__(self, other):
    raise NotImplementedError(f"{self.__class__}.__radd__")

  def __sub__(self, other):
    raise NotImplementedError(f"{self.__class__}.__sub__")

  def __rsub__(self, other):
    raise NotImplementedError(f"{self.__class__}.__rsub__")

  def __getitem__(self, item):
    raise NotImplementedError(f"{self.__class__}.__getitem__")
