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
from jax.experimental.array_api._data_type_functions import (
    _promote_to_default_dtype,
)


def max(x, /, *, axis=None, keepdims=False):
  """Calculates the maximum value of the input array x."""
  return jax.numpy.max(x, axis=axis, keepdims=keepdims)


def mean(x, /, *, axis=None, keepdims=False):
  """Calculates the arithmetic mean of the input array x."""
  return jax.numpy.mean(x, axis=axis, keepdims=keepdims)


def min(x, /, *, axis=None, keepdims=False):
  """Calculates the minimum value of the input array x."""
  return jax.numpy.min(x, axis=axis, keepdims=keepdims)


def prod(x, /, *, axis=None, dtype=None, keepdims=False):
  """Calculates the product of input array x elements."""
  x = _promote_to_default_dtype(x)
  return jax.numpy.prod(x, axis=axis, dtype=dtype, keepdims=keepdims)


def std(x, /, *, axis=None, correction=0.0, keepdims=False):
  """Calculates the standard deviation of the input array x."""
  return jax.numpy.std(x, axis=axis, ddof=correction, keepdims=keepdims)


def sum(x, /, *, axis=None, dtype=None, keepdims=False):
  """Calculates the sum of the input array x."""
  x = _promote_to_default_dtype(x)
  return jax.numpy.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)


def var(x, /, *, axis=None, correction=0.0, keepdims=False):
  """Calculates the variance of the input array x."""
  return jax.numpy.var(x, axis=axis, ddof=correction, keepdims=keepdims)
