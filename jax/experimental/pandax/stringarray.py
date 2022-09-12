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

"""stringarray: JAX-compatible object array for manipulating strings."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

from jax import tree_util
import jax.numpy as jnp
import numpy as np


class StringArray:
  """StringArray: JAX-compatible representation of an array of strings.

  Examples:

    >>> arr = StringArray(['A', 'B', 'C', 'D'])
    >>> arr
    StringArray(['A', 'B', 'C', 'D'], dtype=object)
    >>> len(arr)
    4
    >>> arr.shape
    (4,)
    >>> arr.dtype
    dtype('O')
    >>> arr[0]
    StringArray('A', dtype=object)
  """
  _labels: Tuple[str, ...]
  _data: jnp.ndarray
  _npy_value: Optional[np.ndarray] = None

  dtype = property(lambda self: np.dtype(object))
  shape = property(lambda self: self._data.shape)
  size = property(lambda self: self._data.size)
  ndim = property(lambda self: self._data.ndim)
  T = property(lambda self: self.transpose())

  def __init__(self, data: Union[str, List[str], np.ndarray, StringArray]):
    data = np.asarray(data)
    labels, values = np.unique(data, return_inverse=True)
    self._labels = tuple(map(str, labels))
    self._data = jnp.asarray(values).reshape(data.shape)

  @classmethod
  def _simple_new(cls, labels: Tuple[str, ...], data: jnp.ndarray):
    obj = object.__new__(cls)
    obj._labels = labels
    obj._data = data
    return obj

  def _forward_to_data(self, name: str, *args, **kwargs):
    return self._simple_new(self._labels,
                            getattr(self._data, name)(*args, **kwargs))

  @property
  def _value(self) -> np.ndarray:
    if self._npy_value is None:
      self._npy_value = np.asarray(
          np.asarray(self._labels, dtype=object)[np.asarray(self._data)],
          dtype=object)
    return self._npy_value

  def __array__(self) -> np.ndarray:
    return self._value

  def __iter__(self):
    return (self._simple_new(self._labels, i) for i in self._data)

  def __len__(self) -> int:
    return len(self._data)

  def __repr__(self) -> str:
    arr_repr = repr(self._value)
    if arr_repr.startswith('array'):
      return f'StringArray{arr_repr[5:]}'
    return arr_repr

  def __str__(self) -> str:
    if self.ndim == 0:
      return self._labels[int(self._data)]
    raise ValueError('Can only convert scalar StringArray to string.')

  def __getitem__(self, item) -> StringArray:
    return self._forward_to_data('__getitem__', item)

  def transpose(self, *axes: int) -> StringArray:
    return self._forward_to_data('transpose', *axes)

  def ravel(self) -> StringArray:
    return self._forward_to_data('ravel')

  def reshape(self, *args, **kwargs) -> StringArray:
    return self._forward_to_data('reshape', *args, **kwargs)

  def astype(self, dtype: np.dtype) -> StringArray:
    if dtype != self.dtype:
      raise ValueError(f'cannot convert StringArray to dtype={dtype}; '
                       f'only dtype={self.dtype} is supported.')
    return self

  def _tree_flatten(self) -> Tuple[List[jnp.ndarray], Tuple[str, ...]]:
    return [self._data], self._labels

  @classmethod
  def _tree_unflatten(cls, aux_data: Tuple[str, ...],
                      children: List[jnp.ndarray]):
    return cls._simple_new(aux_data, children[0])


tree_util.register_pytree_node(
    StringArray,
    lambda obj: obj._tree_flatten(),  # pylint: disable=protected-access
    StringArray._tree_unflatten)  # pylint: disable=protected-access


def asarray_or_stringarray(
    val: Any,
    dtype: Optional[np.dtype] = None) -> Union[jnp.ndarray, StringArray]:
  """Convert val to a DeviceArray or StringArray as appropriate."""
  out_types = (jnp.ndarray, StringArray)
  if isinstance(val, out_types):
    out = val
  elif hasattr(val, '_data') and isinstance(val._data, out_types):  # pylint: disable=protected-access
    # properly convert Series-like and Index-like objects.
    out = val._data  # pylint: disable=protected-access
  else:
    try:
      val = jnp.asarray(val)
    except (ValueError, TypeError):
      val = np.asarray(val)
    if np.issubdtype(val.dtype, str) or np.issubdtype(val.dtype, object):
      out = StringArray(val)
    else:
      out = jnp.asarray(val)
  if dtype is not None:
    out = out.astype(dtype)
  return out
