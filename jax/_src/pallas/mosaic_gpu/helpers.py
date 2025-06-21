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

_T = TypeVar("_T")


def nd_loop(
    grid: Sequence[int],
    *,
    collective_axes: Sequence[Hashable] | Hashable,
) -> Callable[[Callable[[Sequence[jax.Array]], None]], None]:
  """A loop over a multi-dimensional grid partitioned along the given axes.

  For example, if ``collective_axes`` is ``"x"`` with :func:`lax.axis_size`
  equal to 4 and the grid is (2, 3), the implementation would produce the
  following iteration order

      loop step    index    axis index

          0        (0, 0)       0
          1        (0, 1)       1
          2        (0, 2)       2
          3        (1, 0)       3
          4        (1, 1)       0
          5        (1, 2)       1

  which comes from partitioning the flat iteration space into chunks in an
  interleaved fashion wrt the ``"x"`` axis index.

  Note that in the example the total number of loop steps is not divisible
  by the axis size of ``"x"``, and thus for some ``"x"`` axis indices the
  loop will do one iteration less.

      axis index       indices

          0         (0, 0), (1, 1)
          1         (0, 1), (1, 2)
          2         (0, 2)
          3         (1, 0)

  See also:
    - :func:`jax.experimental.pallas.loop`: A loop over a single dimension.
  """
  axis_index = lax.axis_index(collective_axes)
  axis_size = lax.axis_size(collective_axes)
  grid_size = math.prod(grid)

  def decorator(body):
    def wrapper(step, _):
      step = step * axis_size + axis_index
      # The loop below is conceptually ``jnp.unravel_index``, but it uses
      # ``lax`` APIs instead of ``jax.numpy`` to minimize the number of
      # primitives used.
      index = []
      for grid_dim in reversed(grid):
        grid_dim = lax.convert_element_type(grid_dim, step.dtype)
        index.append(lax.rem(step, grid_dim))
        step = lax.div(step, grid_dim)
      index.reverse()
      return body(tuple(index))

    upper = lax.div(grid_size, axis_size) + lax.convert_element_type(
        axis_index < grid_size % axis_size, axis_index.dtype
    )
    return lax.fori_loop(0, upper, wrapper, None)

  return decorator
