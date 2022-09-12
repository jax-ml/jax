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

"""JAX implementation of 2D DataFrame objects."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from jax import tree_util
import jax.numpy as jnp
import numpy as np
import pandas as pd

from jax.experimental.pandax.index import Index
from jax.experimental.pandax.index import RangeIndex
from jax.experimental.pandax.series import Series
from jax.experimental.pandax.stringarray import StringArray

IndexLike = Union[Any, Index]
SeriesLike = Union[Any, Series]


class DataFrame:
  """DataFrame implementation built on JAX."""
  _column_index: Index
  _row_index: Index
  _columns: Tuple[Union[jnp.ndarray, StringArray]]
  _pd_value: Optional[pd.DataFrame] = None

  shape = property(lambda self: (len(self._row_index), len(self._column_index)))
  size = property(lambda self: len(self._row_index) * len(self._column_index))
  ndim = property(lambda self: 2)
  index = property(lambda self: self._row_index)
  columns = property(lambda self: self._column_index)

  def __init__(self,
               data: Dict[Any, SeriesLike],
               *,
               index: Optional[IndexLike] = None,
               columns: Optional[IndexLike] = None):
    # TODO(jakevdp): add more instantiation options
    if not isinstance(data, dict):
      raise NotImplementedError(
          "Dataframes can only be instantiated from dicts")
    if columns is not None:
      raise ValueError("columns cannot be specified when input is a dict.")

    columns = data.keys()
    data = tuple(Series(val, index=index) for val in data.values())
    assert all(col.ndim == 1 for col in data)
    assert len({len(col) for col in data}) < 2

    if index is None:
      # TODO(jakevdp) check index for other columns?
      index = data[0].index if data else RangeIndex(0)

    if data:
      assert len(index) == len(data[0])

    self._columns = tuple(col._data for col in data)
    self._column_index = Index(list(columns))
    self._row_index = Index(index)

  @property
  def _value(self) -> pd.DataFrame:
    if self._pd_value is None:
      self._pd_value = pd.DataFrame(
          dict(zip(self._column_index, self._columns)), self._row_index._data)  # pylint: disable=protected-access
    return self._pd_value

  def to_pandas(self) -> pd.DataFrame:
    return self._value

  def __repr__(self) -> str:
    return repr(self._value)

  def __len__(self) -> int:
    return len(self._row_index)

  def __iter__(self):
    return iter(self.columns)

  def __getitem__(self, item) -> Series:
    if isinstance(item, str):
      idx, = self._column_index.get_indexer([item])
      return Series(self._columns[idx], index=self._row_index)
    else:
      raise NotImplementedError(f"getitem {item}")

  def _tree_flatten(self) -> Tuple[List[jnp.ndarray], tree_util.PyTreeDef]:
    return tree_util.tree_flatten(
        [self._column_index, self._row_index, self._columns])

  def drop(self, labels, axis=0):
    if axis == 0:
      raise NotImplementedError("DataFrame.drop() along axis=0")
    if np.ndim(labels) > 0:
      raise NotImplementedError("DataFrame.drop() with multiple labels")

    if labels not in self.columns:
      raise KeyError(f"{labels} not found in {self.columns}")

    data = dict((key, col)
                for key, col in zip(self.columns, self._columns)
                if key != labels)
    return self.__class__(data, index=self.index)

  @classmethod
  def _tree_unflatten(cls, aux_data: tree_util.PyTreeDef,
                      children: List[jnp.ndarray]) -> DataFrame:
    obj = object.__new__(cls)
    obj._column_index, obj._row_index, obj._columns = tree_util.tree_unflatten(
        aux_data, children)
    return obj

  def groupby(self, by: Any) -> DataFrameGroupBy:
    return DataFrameGroupBy(self, by)


class DataFrameGroupBy:
  """Intermediate object for groupby() operations on DataFrame."""
  _df: DataFrame
  _by: Any

  def __init__(self, df: DataFrame, by: Any):
    if not np.shape(by):
      df, by = df.drop(by, axis=1), df[by]
    self._df = df
    self._by = by

  def count(self) -> DataFrame:
    from jax.experimental.pandax import ops  # pylint: disable=g-import-not-at-top
    counts = ops.groupby_count(self._by)
    return DataFrame({col: counts for col in self._df.columns})  # pylint: disable=protected-access

  def max(self):
    from jax.experimental.pandax import ops  # pylint: disable=g-import-not-at-top
    grouped = ops.groupby_max(
        self._by,  # pylint: disable=protected-access
        self._df._columns)  # pylint: disable=protected-access
    return DataFrame(dict(zip(self._df.columns, grouped)))  # pylint: disable=protected-access

  def min(self):
    from jax.experimental.pandax import ops  # pylint: disable=g-import-not-at-top
    grouped = ops.groupby_min(
        self._by,  # pylint: disable=protected-access
        self._df._columns)  # pylint: disable=protected-access
    return DataFrame(dict(zip(self._df.columns, grouped)))  # pylint: disable=protected-access

  def sum(self):
    from jax.experimental.pandax import ops  # pylint: disable=g-import-not-at-top
    grouped = ops.groupby_sum(
        self._by,  # pylint: disable=protected-access
        self._df._columns)  # pylint: disable=protected-access
    return DataFrame(dict(zip(self._df.columns, grouped)))  # pylint: disable=protected-access

  def prod(self):
    from jax.experimental.pandax import ops  # pylint: disable=g-import-not-at-top
    grouped = ops.groupby_prod(
        self._by,  # pylint: disable=protected-access
        self._df._columns)  # pylint: disable=protected-access
    return DataFrame(dict(zip(self._df.columns, grouped)))  # pylint: disable=protected-access


tree_util.register_pytree_node(
    DataFrame,
    lambda obj: obj._tree_flatten(),  # pylint: disable=protected-access
    DataFrame._tree_unflatten)  # pylint: disable=protected-access
