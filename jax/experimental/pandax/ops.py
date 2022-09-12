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

"""pandax Dataframe & Series operations."""

from typing import Any, List, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np

from jax.experimental.pandax.index import Index
from jax.experimental.pandax.series import Series
from jax.experimental.pandax.stringarray import asarray_or_stringarray
from jax.experimental.pandax.stringarray import StringArray


def _validate_shapes(key: Union[jnp.ndarray, StringArray],
                     arrs: List[Union[jnp.ndarray, StringArray]]):
  """Utility to validate shapes for groupby operations."""
  if not arrs:
    return
  arr_shapes = [arr.shape for arr in arrs]
  if len(set(arr_shapes)) != 1:
    raise ValueError(f"array shapes must match; got shapes={arr_shapes}")
  if key.shape != arrs[0].shape:
    raise ValueError("key shape must match array shapes; "
                     f"got key.shape={key.shape}, shapes={arr_shapes}")
  if arrs[0].ndim != 1:
    raise ValueError(f"arrays must be one dimensional; got shapes={arr_shapes}")


def _identity(op: str, size: int, dtype: np.dtype) -> jnp.array:
  """Helper function to compute the identity array for a given aggregation."""
  if op == "add":
    ident = 0
  elif op == "mul":
    ident = 1
  elif op == "min":
    if dtype == jnp.bool_:
      ident = True
    elif jnp.issubdtype(dtype, jnp.integer):
      ident = jnp.iinfo(dtype).max
    ident = jnp.inf
  elif op == "max":
    if dtype == jnp.bool_:
      ident = False
    elif jnp.issubdtype(dtype, jnp.integer):
      ident = jnp.iinfo(dtype).min
    ident = -jnp.inf
  else:
    raise ValueError(f"Unrecognized op: {op}")
  return jnp.full(size, ident, dtype=dtype)


def _grouper_to_indices(key: Any) -> Tuple[Index, jnp.ndarray]:
  key = asarray_or_stringarray(key)
  if isinstance(key, StringArray):
    keys, indices = key._labels, key._data  # pylint: disable=protected-access
  else:
    keys, indices = jnp.unique(key, return_inverse=True)
  return Index(keys), jnp.asarray(indices)


# TODO(jakevdp): these groupby operations could become primitives


def groupby_count(grouper: Any) -> Series:
  """Groupby count operation."""
  keys, indices = _grouper_to_indices(grouper)
  index = Index(list(keys))
  return Series(jnp.zeros(len(keys), jnp.int32).at[indices].add(1), index=index)


def groupby_max(grouper: Any, arrs: Sequence[Series]) -> List[Series]:
  """Groupby maximum operation."""
  keys, indices = _grouper_to_indices(grouper)
  arrs = [asarray_or_stringarray(arr) for arr in arrs]
  _validate_shapes(indices, arrs)
  index = Index(list(keys))
  return [
      Series(
          _identity("max", len(keys), arr.dtype).at[indices].max(arr),
          index=index) for arr in arrs
  ]


def groupby_min(grouper: Any, arrs: Sequence[Series]) -> List[Series]:
  """Groupby minimum operation."""
  keys, indices = _grouper_to_indices(grouper)
  arrs = [asarray_or_stringarray(arr) for arr in arrs]
  _validate_shapes(indices, arrs)
  index = Index(list(keys))
  return [
      Series(
          _identity("min", len(keys), arr.dtype).at[indices].min(arr),
          index=index) for arr in arrs
  ]


def groupby_prod(grouper: Any, arrs: Sequence[Series]) -> List[Series]:
  """Groupby multiplication operation."""
  keys, indices = _grouper_to_indices(grouper)
  arrs = [asarray_or_stringarray(arr) for arr in arrs]
  _validate_shapes(indices, arrs)
  index = Index(list(keys))
  return [
      Series(
          _identity("mul", len(keys), arr.dtype).at[indices].mul(arr),
          index=index) for arr in arrs
  ]


def groupby_sum(grouper: Any, arrs: Sequence[Series]) -> List[Series]:
  """Groupby summation operation."""
  keys, indices = _grouper_to_indices(grouper)
  arrs = [asarray_or_stringarray(arr) for arr in arrs]
  _validate_shapes(indices, arrs)
  index = Index(list(keys))
  return [
      Series(
          _identity("add", len(keys), arr.dtype).at[indices].add(arr),
          index=index) for arr in arrs
  ]
