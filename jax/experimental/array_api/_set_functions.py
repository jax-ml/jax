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

from typing import NamedTuple
import jax


class UniqueAllResult(NamedTuple):
  values: jax.Array
  indices: jax.Array
  inverse_indices: jax.Array
  counts: jax.Array


class UniqueCountsResult(NamedTuple):
    values: jax.Array
    counts: jax.Array


class UniqueInverseResult(NamedTuple):
    values: jax.Array
    inverse_indices: jax.Array


def unique_all(x, /):
  """Returns the unique elements of an input array x, the first occurring indices for each unique element in x, the indices from the set of unique elements that reconstruct x, and the corresponding counts for each unique element in x."""
  values, indices, inverse_indices, counts = jax.numpy.unique(
    x, return_index=True, return_inverse=True, return_counts=True)
  # jnp.unique() flattens inverse indices
  inverse_indices = inverse_indices.reshape(x.shape)
  return UniqueAllResult(values=values, indices=indices, inverse_indices=inverse_indices, counts=counts)


def unique_counts(x, /):
  """Returns the unique elements of an input array x and the corresponding counts for each unique element in x."""
  values, counts = jax.numpy.unique(x, return_counts=True)
  return UniqueCountsResult(values=values, counts=counts)


def unique_inverse(x, /):
  """Returns the unique elements of an input array x and the indices from the set of unique elements that reconstruct x."""
  values, inverse_indices = jax.numpy.unique(x, return_inverse=True)
  inverse_indices = inverse_indices.reshape(x.shape)
  return UniqueInverseResult(values=values, inverse_indices=inverse_indices)


def unique_values(x, /):
  """Returns the unique elements of an input array x."""
  return jax.numpy.unique(x)
