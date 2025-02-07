# Copyright 2024 The JAX Authors.
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
"""Helper functions for PRNG kernels."""
from typing import Sequence
from jax import lax
import jax.numpy as jnp

Shape = Sequence[int]

round_up = lambda x, y: (x + y - 1) // y * y

def blocked_iota(block_shape: Shape,
                 total_shape: Shape):
  """Computes a sub-block of a larger shaped iota.

  Args:
    block_shape: The output block shape of the iota.
    total_shape: The total shape of the input tensor.
  Returns:
    Result of the blocked iota.
  """
  iota_data = jnp.zeros(block_shape, dtype=jnp.uint32)
  multiplier = 1
  for dim in range(len(block_shape)-1, -1, -1):
    block_mult = 1
    counts_lo = lax.broadcasted_iota(
        dtype=jnp.uint32, shape=block_shape, dimension=dim
    )
    iota_data += counts_lo * multiplier * block_mult
    multiplier *= total_shape[dim]
  return iota_data


def compute_scalar_offset(iteration_index,
                          total_size: Shape,
                          block_size: Shape):
  ndims = len(iteration_index)
  dim_size = 1
  total_idx = 0
  for i in range(ndims-1, -1, -1):
    dim_idx = iteration_index[i] * block_size[i]
    total_idx += dim_idx * dim_size
    dim_size *= total_size[i]
  return total_idx
