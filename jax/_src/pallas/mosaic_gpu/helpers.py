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
import functools
import math
from typing import TypeVar, overload

import jax
from jax import lax

_T = TypeVar("_T")


@overload
def nd_loop(
    grid: Sequence[int],
    *,
    collective_axes: Sequence[Hashable] | Hashable,
    tiling: Sequence[int] | None = None,
    init_carry: None = None
) -> Callable[[Callable[[Sequence[jax.Array]], None]], None]:
  ...


@overload
def nd_loop(
    grid: Sequence[int],
    *,
    collective_axes: Sequence[Hashable] | Hashable,
    tiling: Sequence[int] | None = None,
    init_carry: _T
) -> Callable[[Callable[[Sequence[jax.Array], _T], _T]], _T]:
  ...


# TODO(justinfu): Fix the type signature to include both carry and wave_step.
@overload
def nd_loop(
    grid: Sequence[int],
    *,
    collective_axes: Sequence[Hashable] | Hashable,
    tiling: Sequence[int] | None = None,
    include_wave_step: bool
) -> Callable[[Callable[[Sequence[jax.Array], jax.Array], None]], None]:
  ...


def nd_loop(grid, *, collective_axes,
            tiling=None,
            init_carry=None,
            include_wave_step=False):
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

  If ``init_carry`` is passed then ``nd_loop()`` will expect the body to
  take and return the carry. If it's ``None`` then no carry argument is
  expected.

  If ``include_wave_step`` is True then the body will be called with an
  additional ``wave_step`` keyword argument that specifies the current
  iteration local to the thread.

  See also:
    - :func:`jax.experimental.pallas.loop`: A loop over a single dimension.
  """

  axis_index = lax.axis_index(collective_axes)
  axis_size = lax.axis_size(collective_axes)
  if tiling:
    if len(grid) != len(tiling):
      raise ValueError(f"{tiling=} and {grid=} must have same length.")
    if any(dim % tile != 0 for dim, tile in zip(grid, tiling, strict=True)):
      raise ValueError(f"Tiling {tiling} does not divide grid {grid}.")
    tile_grid = tuple(
        dim // tile for dim, tile in zip(grid, tiling, strict=True))
    grid = (*tile_grid, *tiling)
  grid_size = math.prod(grid)

  def decorator(body):
    def wrapper(wave_step, carry):
      nonlocal body
      step = wave_step * axis_size + axis_index
      # The loop below is conceptually ``jnp.unravel_index``, but it uses
      # ``lax`` APIs instead of ``jax.numpy`` to minimize the number of
      # primitives used.
      index = []
      for grid_dim in reversed(grid):
        grid_dim = lax.convert_element_type(grid_dim, step.dtype)
        index.append(lax.rem(step, grid_dim))
        step = lax.div(step, grid_dim)
      index.reverse()

      if tiling:
        # Recompute index as if the grid was not tiled.
        tile_indices, subtile_indices = index[:len(tiling)], index[len(tiling):]
        untiled_index = []
        for sub_idx, tile_idx, tile_dim in zip(
            subtile_indices, tile_indices, tiling, strict=True):
          untiled_index.append(sub_idx + tile_idx * tile_dim)
        index = untiled_index

      if include_wave_step:
        body = functools.partial(body, wave_step=wave_step)
      if init_carry is None:
        body(tuple(index))
      else:
        return body(tuple(index), carry=carry)

    upper = lax.div(grid_size, axis_size) + lax.convert_element_type(
        axis_index < grid_size % axis_size, axis_index.dtype
    )
    return lax.fori_loop(0, upper, wrapper, init_carry)
  return decorator
