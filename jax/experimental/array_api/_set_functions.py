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


def unique_all(x, /):
  """Returns the unique elements of an input array x, the first occurring indices for each unique element in x, the indices from the set of unique elements that reconstruct x, and the corresponding counts for each unique element in x."""
  return jax.numpy.unique_all(x)


def unique_counts(x, /):
  """Returns the unique elements of an input array x and the corresponding counts for each unique element in x."""
  return jax.numpy.unique_counts(x)


def unique_inverse(x, /):
  """Returns the unique elements of an input array x and the indices from the set of unique elements that reconstruct x."""
  return jax.numpy.unique_inverse(x)


def unique_values(x, /):
  """Returns the unique elements of an input array x."""
  return jax.numpy.unique_values(x)
