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

"""Index objects used by pandax Series & Dataframes."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple, Union

from jax import tree_util
from jax._src import dtypes
import jax.numpy as jnp
import numpy as np

from jax.experimental.pandax.stringarray import asarray_or_stringarray
from jax.experimental.pandax.stringarray import StringArray


def find_in_list(list_to_search: Sequence[Any], item: Any) -> int:
  if isinstance(item, StringArray):
    item = str(item)
  try:
    return list_to_search.index(item)
  except ValueError:
    return -1

class Index:
  """Index object used by pandax Series & Dataframes."""
  _data: Union[jnp.ndarray, StringArray]

  dtype = property(lambda self: self._data.dtype)

  def __new__(cls, *args, **kwargs) -> Index:
    if len(args) == 1 and isinstance(args[0], (range, RangeIndex)):
      return RangeIndex(*args, **kwargs)
    return object.__new__(cls)

  def __init__(self, data, *, dtype=None):
    # TODO(jakevdp): allow instantiation from Pandas Index
    if isinstance(data, Index):
      data = data._data
    self._data = asarray_or_stringarray(data)

  def __repr__(self):
    return f"Index{repr(self._data)[11:]}"

  def __array__(self):
    return np.asarray(self._data)

  def __iter__(self):
    return iter(np.asarray(self))

  def __len__(self):
    return len(self._data)

  def __getitem__(self, idx):
    new_data = self._data[idx]
    if new_data.ndim == 0:
      return new_data
    elif new_data.ndim == 1:
      return Index(new_data)
    else:
      raise NotImplementedError("multi-dimensional indexing on Index objects")

  def __setitem__(self, idx, values):
    raise ValueError("Index objects are immutable.")

  def get_indexer(self, indices: Union[jnp.ndarray, Index, StringArray, Sequence[Any]]) -> jnp.ndarray:
    """Get a data-space indexer given an index-space indexer."""
    indices = asarray_or_stringarray(indices)
    # TODO(jakevdp): should this kind of logic be pushed into a method of
    # StringArray itsef? Or perhaps we need a PandaxArray abstraction that
    # wraps both jnp.ndarray and StringArray and provides a common API for
    # use in pandax functions?
    if isinstance(self._data, StringArray):
      indices = [find_in_list(self._data._labels, i) for i in indices]  # pylint: disable=protected-access
      data = self._data._data  # pylint: disable=protected-access
    else:
      data = self._data

    mask = jnp.array(data) == jnp.asarray(indices)[:, None]
    ind = jnp.argmin(~mask, axis=1)
    return jnp.where(mask.any(axis=1), ind, -1)

  def _tree_flatten(self) -> Tuple[List[jnp.ndarray], tree_util.PyTreeDef]:
    return tree_util.tree_flatten(self._data)

  @classmethod
  def _tree_unflatten(cls, aux_data: tree_util.PyTreeDef,
                      children: List[jnp.ndarray]) -> Index:
    return cls(tree_util.tree_unflatten(aux_data, children))


tree_util.register_pytree_node(
    Index,
    lambda obj: obj._tree_flatten(),  # pylint: disable=protected-access
    Index._tree_unflatten)  # pylint: disable=protected-access


class RangeIndex(Index):
  """Index backed by a simple integer range()."""
  _data: range
  _dtype: np.dtype

  start = property(lambda self: self._data.start)
  stop = property(lambda self: self._data.stop)
  step = property(lambda self: self._data.step)
  dtype = property(lambda self: self._dtype)

  def __new__(cls, *args, **kwargs):
    return object.__new__(cls)

  def __init__(self, start, stop=None, step=None, *, dtype=None):  # pylint: disable=super-init-not-called
    if isinstance(start, RangeIndex):
      assert stop is None and step is None
      self._data = start._data
      self._dtype = dtypes.canonicalize_dtype(dtype or start.dtype)
    elif isinstance(start, range):
      assert stop is None and step is None
      self._data = start
      self._dtype = dtypes.canonicalize_dtype(dtype or jnp.int_)
    else:
      if not isinstance(start, int):
        raise ValueError("RangeIndex: expected start argument to be an int, "
                         f"RangeIndex, or range. Got {start!r}.")
      if stop is None:
        start, stop = 0, start
      if step is None:
        step = 1
      self._data = range(start, stop, step)
      self._dtype = dtypes.canonicalize_dtype(dtype or jnp.int_)
    if not jnp.issubdtype(self._dtype, jnp.integer):
      raise ValueError(
          "RangeIndex requires integer dtype; got {self._dtype.name}")

  def __repr__(self):
    return f"RangeIndex(start={self.start}, stop={self.stop}, step={self.step})"

  def __array__(self):
    return np.asarray(self._data)

  def __iter__(self):
    return iter(np.asarray(self))

  def __len__(self):
    return len(self._data)

  def __getitem__(self, idx):
    if isinstance(idx, int):
      return jnp.array(self._data[idx], dtype=self.dtype)
    elif isinstance(idx, slice):
      return RangeIndex(self._data[idx], dtype=self.dtype)
    else:
      data = jnp.arange(self.start, self.stop, self.step, dtype=self.dtype)
      return Index(data)[idx]

  def get_indexer(self, indices: jnp.ndarray) -> jnp.ndarray:
    """Get a data-space indexer given an index-space indexer."""
    indices = jnp.asarray(indices)
    assert jnp.issubdtype(indices.dtype, jnp.integer)
    indexer = indices - self.start
    indexer = jnp.where(indexer % self.step == 0, indexer // self.step, -1)
    return jnp.where(((0 <= indexer) & (indexer < len(self))), indexer, -1)

  def _tree_flatten(self) -> Tuple[List[str], Tuple[range, np.dtype]]:
    return [], (self._data, self._dtype)

  @classmethod
  def _tree_unflatten(cls, aux_data: Tuple[range, np.dtype],
                      children: List[str]) -> RangeIndex:
    assert not children
    rng, dtype = aux_data
    return cls(rng, dtype=dtype)


tree_util.register_pytree_node(
    RangeIndex,
    lambda obj: obj._tree_flatten(),  # pylint: disable=protected-access
    RangeIndex._tree_unflatten)  # pylint: disable=protected-access
