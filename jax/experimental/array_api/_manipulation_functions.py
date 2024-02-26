# Copyright 2023 The JAX Authors.
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

from __future__ import annotations

import jax
from jax import Array
from jax.experimental.array_api._data_type_functions import result_type as _result_type


def broadcast_arrays(*arrays: Array) -> list[Array]:
  """Broadcasts one or more arrays against one another."""
  return jax.numpy.broadcast_arrays(*arrays)


def broadcast_to(x: Array, /, shape: tuple[int, ...]) -> Array:
  """Broadcasts an array to a specified shape."""
  return jax.numpy.broadcast_to(x, shape=shape)


def concat(arrays: tuple[Array, ...] | list[Array], /, *, axis: int | None = 0) -> Array:
  """Joins a sequence of arrays along an existing axis."""
  dtype = _result_type(*arrays)
  return jax.numpy.concat([arr.astype(dtype) for arr in arrays], axis=axis)


def expand_dims(x: Array, /, *, axis: int = 0) -> Array:
  """Expands the shape of an array by inserting a new axis (dimension) of size one at the position specified by axis."""
  if axis < -x.ndim - 1 or axis > x.ndim:
    raise IndexError(f"{axis=} is out of bounds for array of dimension {x.ndim}")
  return jax.numpy.expand_dims(x, axis=axis)


def flip(x: Array, /, *, axis: int | tuple[int, ...] | None = None) -> Array:
  """Reverses the order of elements in an array along the given axis."""
  return jax.numpy.flip(x, axis=axis)


def permute_dims(x: Array, /, axes: tuple[int, ...]) -> Array:
  """Permutes the axes (dimensions) of an array x."""
  return jax.numpy.permute_dims(x, axes=axes)


def reshape(x: Array, /, shape: tuple[int, ...], *, copy: bool | None = None) -> Array:
  """Reshapes an array without changing its data."""
  del copy  # unused
  return jax.numpy.reshape(x, shape)


def roll(x: Array, /, shift: int | tuple[int, ...], *, axis: int | tuple[int, ...] | None = None) -> Array:
  """Rolls array elements along a specified axis."""
  return jax.numpy.roll(x, shift=shift, axis=axis)


def squeeze(x: Array, /, axis: int | tuple[int, ...]) -> Array:
  """Removes singleton dimensions (axes) from x."""
  return jax.numpy.squeeze(x, axis=axis)


def stack(arrays: tuple[Array, ...] | list[Array], /, *, axis: int = 0) -> Array:
  """Joins a sequence of arrays along a new axis."""
  dtype = _result_type(*arrays)
  return jax.numpy.stack(arrays, axis=axis, dtype=dtype)
