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
import dataclasses
import functools
import math
from typing import TypeVar, overload

import jax
from jax import numpy as jnp
from jax import lax
from jax._src import dtypes
from jax._src.pallas.mosaic_gpu import primitives as gpu_primitives
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas import primitives as pallas_primitives
import numpy as np

_T = TypeVar("_T")


@dataclasses.dataclass(frozen=True, eq=False)
class NDLoopInfo:
  """Container dataclass for loop iteration information.

  Attributes:
    index: The grid indices corresponding to the current loop iteration.
    local_index: The local iteration index.
    num_local_steps: The total number of local iterations to run. None
      if unknown.
  """
  index: tuple[jax.Array, ...]
  local_index: jax.Array | int
  num_local_steps: jax.Array | int | None


@overload
def nd_loop(
    grid: Sequence[int],
    *,
    collective_axes: Sequence[Hashable] | Hashable,
    tiling: Sequence[int] | None = None,
    init_carry: None = None
) -> Callable[[Callable[[NDLoopInfo], None]], None]:
  ...


@overload
def nd_loop(
    grid: Sequence[int],
    *,
    collective_axes: Sequence[Hashable] | Hashable,
    tiling: Sequence[int] | None = None,
    init_carry: _T
) -> Callable[[Callable[[NDLoopInfo, _T], _T]], _T]:
  ...


def nd_loop(grid, *, collective_axes, tiling=None, init_carry=None):
  """A loop over a multi-dimensional grid partitioned along the given axes.

  The body of the loop a single argument ``loop_info`` which is an NDLoopInfo
  object containing index and iteration information. However if a carry is
  specified, the body will expect a second keyword argument `carry` containing
  the loop carry.

  For example, if ``collective_axes`` is ``"x"`` with :func:`lax.axis_size`
  equal to 4 and the grid is (2, 3), the implementation would produce the
  following iteration order

  +-----------+--------+------------+
  | loop step | index  | axis index |
  +===========+========+============+
  |     0     | (0, 0) |     0      |
  +-----------+--------+------------+
  |     1     | (0, 1) |     1      |
  +-----------+--------+------------+
  |     2     | (0, 2) |     2      |
  +-----------+--------+------------+
  |     3     | (1, 0) |     3      |
  +-----------+--------+------------+
  |     4     | (1, 1) |     0      |
  +-----------+--------+------------+
  |     5     | (1, 2) |     1      |
  +-----------+--------+------------+

  which comes from partitioning the flat iteration space into chunks in an
  interleaved fashion wrt the ``"x"`` axis index.

  Note that in the example the total number of loop steps is not divisible
  by the axis size of ``"x"``, and thus for some ``"x"`` axis indices the
  loop will do one iteration less.

  +------------+------------------+
  | axis index | indices          |
  +============+==================+
  |     0      | (0, 0), (1, 1)   |
  +------------+------------------+
  |     1      | (0, 1), (1, 2)   |
  +------------+------------------+
  |     2      | (0, 2)           |
  +------------+------------------+
  |     3      | (1, 0)           |
  +------------+------------------+

  If ``init_carry`` is passed then ``nd_loop()`` will expect the body to
  take and return the carry. If it's ``None`` then no carry argument is
  expected.

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

      loop_info = NDLoopInfo(
          index=tuple(index),
          local_index=wave_step,
          num_local_steps=upper
      )
      if init_carry is None:
        body(loop_info)
      else:
        return body(loop_info, carry=carry)

    upper = lax.div(grid_size, axis_size) + lax.convert_element_type(
        axis_index < grid_size % axis_size, axis_index.dtype
    )
    return lax.fori_loop(0, upper, wrapper, init_carry)
  return decorator


def format_tcgen05_sparse_metadata(meta):
  """Formats the sparse metadata for tcgen05.mma into the expected format.

  See https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-sparse-matrices-sparsity-selector-kind-f16-m128-256
  for the documentation of the required layouts. The array can be copied into
  SMEM, from where ``plgpu.async_copy_sparse_metadata_to_tmem`` can be used to
  copy it over to TMEM.
  """
  if meta.dtype != dtypes.uint2:
    raise ValueError(f"Expected metadata dtype to be uint2, got: {meta.dtype}")
  if meta.ndim != 3:
    raise ValueError(
        "Expected metadata to be 3-dimensional (M, K // 4, 2), but it is"
        f" {meta.ndim}D"
    )
  m, k, _2 = meta.shape
  if _2 != 2:
    raise ValueError(
        "Expected the trailing dimension of the metadata to be 2, got:"
        f" {meta.shape[-1]}"
    )
  k *= 2
  return (
      meta.reshape(m // 128, 8, 2, 8, k // 64, 4, 2, 8)
      .transpose(0, 4, 1, 6, 3, 5, 2, 7)
      .reshape(m // 128, k // 64, 128, 64)
  )


def find_swizzle(minor_dim_bits: int, what: str = ""):
  """Returns the largest swizzle that can be applied to a memory region.

  Swizzling is usually necessary when dealing with 2D data in SMEM, especially
  if the reference is used as an MMA operand. The returned swizzle is usually
  applied as ``plgpu`` transform:

    transforms = (
        plgpu.TilingTransform((8, 8 * swizzle // elem_bits)),
        plgpu.SwizzleTransform(swizzle))
    )

  Args:
    minor_dim_bits: The number of bits in the minor (last) dimension of the
      memory region. Usually computed as ``dim_size * jnp.finfo(dtype).bits``.
    what: A string describing the operand for which the swizzle is being
      computed. Improves the error message if specified.
  """
  for swizzle_bytes in (128, 64, 32, 16):
    if minor_dim_bits % (swizzle_bytes * 8) == 0:
      return swizzle_bytes
  if what:
    what = " for " + what
  raise ValueError(
      f"No valid out swizzle{what}: minor dimension has"
      f" {minor_dim_bits} bits, which is not a multiple of 128 (16 bytes)"
  )


def planar_snake(
    lin_idx: jax.Array, shape: tuple[int, int], minor_dim: int, tile_width: int
):
  """Converts a linear index into an index into shape, trying to optimize locality.

  The "space filling curve" this function computes splits the minor dimension
  into tiles of length ``tile_width``. Every other tile has its major dimension
  inverted, so that the iteration order "snakes around" when going from one tile
  to another.

  For a shape of (8, 8), ``minor_dim=0`` and ``tile_width=2``, the iteration
  order is::

       0   2   4   6   8  10  12  14
       1   3   5   7   9  11  13  15
      30  28  26  24  22  20  18  16
      31  29  27  25  23  21  19  17
      32  34  36  38  40  42  44  46
      33  35  37  39  41  43  45  47
      62  60  58  56  54  52  50  48
      63  61  59  57  55  53  51  49

  Notice how each pair of rows forms a tile (``minor_dim=0``, ``tile_width=2``)
  and when moving from one tile to another, the indices increase along columns
  in one of them and decrease in the other.
  """
  tile_width = np.int32(tile_width)
  major_size = np.int32(shape[1 - minor_dim])
  minor_size = np.int32(shape[minor_dim])
  minor_tile_idx = lax.div(lin_idx, tile_width * major_size)

  def tile_coordinates(lin_idx, width):
    # if minor_dim == 0 then tiles are (tile_width, major_size) else (major_size, tile_width)
    minor_within_tile = lax.rem(lin_idx, width)
    major_within_tile = lax.rem(lax.div(lin_idx, width), major_size)
    minor = minor_tile_idx * tile_width + minor_within_tile
    major = lax.select(
      lax.rem(minor_tile_idx, np.int32(2)) == 0,
      major_within_tile,
      major_size - 1 - major_within_tile,
    )
    return (minor, major) if minor_dim == 0 else (major, minor)

  num_full_tiles = shape[minor_dim] // tile_width
  full_tiles_minor_size = num_full_tiles * tile_width
  num_full_tiles_elements = num_full_tiles * tile_width * major_size
  is_full_tile = lin_idx < num_full_tiles_elements
  return jax.tree.map(
      functools.partial(jax.lax.select, is_full_tile),
      tile_coordinates(lin_idx, tile_width),
      tile_coordinates(lin_idx - num_full_tiles_elements, minor_size - full_tiles_minor_size)
  )


@overload
def dynamic_scheduling_loop(
    grid_names: Sequence[Hashable],
    *,
    thread_axis: Hashable | None = None,
    init_carry: None = None
) -> Callable[[Callable[[NDLoopInfo], None]], None]:
  ...


@overload
def dynamic_scheduling_loop(
    grid_names: Sequence[Hashable],
    *,
    thread_axis: Hashable | None = None,
    init_carry: _T
) -> Callable[[Callable[[NDLoopInfo, _T], _T]], _T]:
  ...


def dynamic_scheduling_loop(
    grid_names,
    thread_axis = None,
    init_carry = None):
  """A loop over program instances using dynamic work scheduling.

  This loop will iterate through available program instances until all
  work has been scheduled. The kernel should be instantiated with a grid
  equal to the logical amount of work to be done (as opposed to a persistent
  kernel where the grid is set to the number of cores). Each core running
  this loop will continuously query the next available block of work and
  the loop will terminate when the entire grid has been scheduled.

  Example usage::

    @plgpu.dynamic_scheduling_loop(grid_names)
    def body(loop_info):
      work(loop_info.index)  # do work...

  Args:
    grid_names: The names of the axes in the grid.
    thread_axis: The name of the thread axis. This must be passed in if
      the kernel uses multiple threads.
    init_carry: An optional initial carry for the loop. If passed in, the
      body function should expect a ``carry`` keyword argument and return
      the next carry value.
  """
  if thread_axis is not None:
    num_threads = lax.axis_size(thread_axis)
  else:
    num_threads = 1
  user_carry = init_carry

  def decorator(body):
    grid_idx = tuple(lax.axis_index(axis_name) for axis_name in grid_names)
    success = True
    def _scoped(try_cancel_buffer, try_cancel_barrier):
      def try_cancel_cond(carry):
        _, success, _, _ = carry
        return success
      def try_cancel_body(carry):
        grid_idx, _, wave_step, user_carry = carry
        slot = lax.rem(wave_step, jnp.int32(2))
        gpu_primitives.try_cluster_cancel(try_cancel_buffer.at[slot],
                                          try_cancel_barrier.at[slot])
        loop_info = NDLoopInfo(
            index=grid_idx,
            local_index=wave_step,
            num_local_steps=None,
        )
        if user_carry is None:
          body(loop_info)
        else:
          user_carry = body(loop_info, carry=user_carry)
        gpu_primitives.barrier_wait(try_cancel_barrier.at[slot])
        grid_idx, success = gpu_primitives.query_cluster_cancel(
            try_cancel_buffer.at[slot],
            grid_names=grid_names)
        return (grid_idx, success, wave_step + jnp.int32(1), user_carry)
      init_carry = (grid_idx, success, jnp.int32(0), user_carry)
      final_carry = lax.while_loop(
          try_cancel_cond,
          try_cancel_body,
          init_carry,
      )
      if user_carry is not None:
        return final_carry[-1]
    return pallas_primitives.run_scoped(
        _scoped,
        try_cancel_buffer=gpu_core.TryClusterCancelResult(2),
        try_cancel_barrier=gpu_core.Barrier(num_arrivals=num_threads,
                                            num_barriers=2),
        collective_axes=thread_axis,
    )
  return decorator
