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

import jax
from jax.experimental.array_api._data_type_functions import result_type as _result_type


def argmax(x, /, *, axis=None, keepdims=False):
  """Returns the indices of the maximum values along a specified axis."""
  return jax.numpy.argmax(x, axis=axis, keepdims=keepdims)


def argmin(x, /, *, axis=None, keepdims=False):
  """Returns the indices of the minimum values along a specified axis."""
  return jax.numpy.argmin(x, axis=axis, keepdims=keepdims)


def nonzero(x, /):
  """Returns the indices of the array elements which are non-zero."""
  if jax.numpy.ndim(x) == 0:
    raise ValueError("inputs to nonzero() must have at least one dimension.")
  return jax.numpy.nonzero(x)


def where(condition, x1, x2, /):
  """Returns elements chosen from x1 or x2 depending on condition."""
  dtype = _result_type(x1, x2)
  return jax.numpy.where(condition, x1.astype(dtype), x2.astype(dtype))
