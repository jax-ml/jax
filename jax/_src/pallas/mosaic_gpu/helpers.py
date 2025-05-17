# Copyright 2025 The JAX Authors.
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

"""Helpers for Pallas Mosaic GPU kernels."""

from collections.abc import Callable, Hashable, Sequence
import math
from typing import TypeVar

import jax
from jax import lax
import jax.numpy as jnp

_T = TypeVar("_T")


def partitioned_nd_loop(
    grid: Sequence[int],
    body: Callable[[Sequence[jax.Array], _T], _T],
    init_val: _T,
    *,
    axis_name: Hashable
) -> _T:
  """A loop over a multi-dimensional grid partitioned along the given axis.

  For example, if the axis size is 3 and the grid is (2, 3), the loop
  body will be called with

      axis index      indices
          0        (0, 0) (1, 0)
          1        (0, 1) (1, 1)
          2        (0, 2) (1, 2)

  See also:
    - :func:`jax.lax.fori_loop`: A single-dimensional indexed loop.
  """
  axis_index = lax.axis_index(axis_name)
  axis_size = lax.axis_size(axis_name)
  grid_size = math.prod(grid)

  def wrapper(step, carry):
    step = step * axis_size + axis_index
    # The loop below is conceptually ``jnp.unravel_index``, but it uses
    # ``lax`` APIs instead of ``jax.numpy`` to minimize the number of
    # primitives used.
    index = []
    for grid_dim in reversed(grid):
      index.append(lax.rem(step, grid_dim))
      step = lax.div(step, grid_dim)
    index.reverse()
    return body(tuple(index), carry)

  return lax.fori_loop(
      0,
      grid_size // axis_size + jnp.int32(axis_index < grid_size % axis_size),
      wrapper,
      init_val,
  )
