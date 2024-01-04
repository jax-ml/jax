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
from jax import Array


def argsort(x: Array, /, *, axis: int = -1, descending: bool = False,
            stable: bool = True) -> Array:
  """Returns the indices that sort an array x along a specified axis."""
  return jax.numpy.argsort(x, axis=axis, descending=descending, stable=stable)


def sort(x: Array, /, *, axis: int = -1, descending: bool = False,
         stable: bool = True) -> Array:
  """Returns a sorted copy of an input array x."""
  return jax.numpy.sort(x, axis=axis, descending=descending, stable=stable)
