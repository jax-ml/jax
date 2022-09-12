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

"""JAX implementation of 1D Series objects."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

from jax import tree_util
import jax.numpy as jnp
import numpy as np
import pandas as pd

from jax.experimental.pandax.index import Index
from jax.experimental.pandax.index import RangeIndex
from jax.experimental.pandax.stringarray import asarray_or_stringarray
from jax.experimental.pandax.stringarray import StringArray


class Series:
  """Series implementation built on JAX."""
  _index: Index
  _data: Union[jnp.ndarray, StringArray]
  _pd_value: Optional[pd.Series] = None

  index = property(lambda self: self._index)
  dtype = property(lambda self: self._data.dtype)
  shape = property(lambda self: self._data.shape)
  size = property(lambda self: self._data.size)
  ndim = property(lambda self: self._data.ndim)

  # TODO(jakevdp): shape, size, ndim, etc.

  def __init__(self, data, index=None, dtype=None):
    # TODO(jakevdp): allow instantiation from pandas Series
    if isinstance(data, Series):
      if index is None:
        index = data._index
      data = data._data
    data = asarray_or_stringarray(data, dtype=dtype)
    if data.ndim != 1:
      raise ValueError("Series data must be 1-dimensional")
    self._data = data
    self._index = RangeIndex(len(self._data)) if index is None else Index(index)

  @property
  def _value(self) -> pd.Series:
    if self._pd_value is None:
      self._pd_value = pd.Series(
          np.asarray(self._data), index=np.asarray(self._index))
    return self._pd_value

  def to_pandas(self) -> pd.Series:
    return self._value

  def __array__(self) -> np.ndarray:
    return np.array(self._data)

  def __repr__(self) -> str:
    return repr(self._value)

  def __iter__(self):
    return iter(self.index)

  def __len__(self) -> int:
    return len(self._data)

  def __getitem__(self, ind):
    if isinstance(ind, slice):
      return Series(self._data[ind], self._index[ind])
    ind = asarray_or_stringarray(ind)
    if ind.ndim > 1:
      raise ValueError(f"Too many indices for Series: {ind}")

    indexer = self._index.get_indexer(ind.ravel())
    if ind.ndim == 0:
      return self._data[indexer[0]]
    return Series(self._data[indexer], self._index[indexer])

  def groupby(self, by: Any) -> SeriesGroupBy:
    return SeriesGroupBy(self, by)

  def _tree_flatten(self) -> Tuple[List[jnp.ndarray], tree_util.PyTreeDef]:
    return tree_util.tree_flatten([self._index, self._data])

  @classmethod
  def _tree_unflatten(cls, aux_data: tree_util.PyTreeDef,
                      children: List[jnp.ndarray]) -> Series:
    obj = cls([])
    obj._index, obj._data = tree_util.tree_unflatten(aux_data, children)
    return obj


class SeriesGroupBy:
  """Intermediate object for groupby() operations on Series."""
  _ser: Series
  _by: Any

  def __init__(self, ser: Series, by: Any):
    self._ser = ser
    self._by = by

  def count(self):
    from jax.experimental.pandax import ops  # pylint: disable=g-import-not-at-top
    return ops.groupby_count(self._by)

  def max(self):
    from jax.experimental.pandax import ops  # pylint: disable=g-import-not-at-top
    return ops.groupby_max(self._by, [self._ser])[0]

  def min(self):
    from jax.experimental.pandax import ops  # pylint: disable=g-import-not-at-top
    return ops.groupby_min(self._by, [self._ser])[0]

  def sum(self):
    from jax.experimental.pandax import ops  # pylint: disable=g-import-not-at-top
    return ops.groupby_sum(self._by, [self._ser])[0]

  def prod(self):
    from jax.experimental.pandax import ops  # pylint: disable=g-import-not-at-top
    return ops.groupby_prod(self._by, [self._ser])[0]


tree_util.register_pytree_node(
    Series,
    lambda obj: obj._tree_flatten(),  # pylint: disable=protected-access
    Series._tree_unflatten)  # pylint: disable=protected-access
