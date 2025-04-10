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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax._src import blocked_sampler
from jax._src import config
from jax._src import test_util as jtu
import numpy as np


config.parse_flags_with_absl()


def call_kernel(
  kernel,
  grid: tuple[int, int],
  transpose_grid: bool,
  key: jax.Array,
  total_size: tuple[int, int],
  block_size: tuple[int, int],
  tile_size: tuple[int, int],
  ):
  """Calls a kernel over a grid and concatenates results to a single array."""
  if transpose_grid:
    grid = (grid[1], grid[0])
  m, n = grid
  samples = jnp.concatenate([
              jnp.concatenate([
                kernel((i, j), key, total_size, block_size, tile_size)
                  for j in range(n)], axis=1)
                    for i in range(m)], axis=0)
  # Slice out the padding.
  samples = samples[:total_size[0], :total_size[1]]
  return samples


def call_kernel_3d(
  kernel,
  grid: tuple[int, int],
  *args
  ):
  """Calls a kernel over a 3D grid and concatenates results to a single array."""
  depth, rows, cols = grid
  return jnp.concatenate([
          jnp.concatenate([
            jnp.concatenate([
              jnp.array(kernel((i, j, k), *args))
                for k in range(cols)], axis=2)
                  for j in range(rows)], axis=1)
                    for i in range(depth)], axis=0)


def blocked_fold_in(block_index, key, total_size, block_size, tile_size):
  """Folds in block_index into global_key."""
  return blocked_sampler.blocked_fold_in(key,
                                         total_size=total_size,
                                         block_size=block_size,
                                         tile_size=tile_size,
                                         block_index=block_index)


def uniform_kernel(block_index, key, total_size, block_size, tile_size):
  """Uniform random sampling kernel function."""
  keys = blocked_fold_in(block_index, key,
                         total_size=total_size,
                         block_size=block_size,
                         tile_size=tile_size)
  return blocked_sampler.sample_block(jax.random.uniform,
                                         keys,
                                         block_size=block_size,
                                         tile_size=tile_size,
                                         minval=0.0, maxval=1.0)


class BlockedSamplerTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
    dict(testcase_name='8x128_vs_16x256', total_size=(32, 256),
         block_size_a=(8, 128), block_size_b=(16, 256),
         tile_size=(8, 128), transpose_grid=False),
    dict(testcase_name='transpose_8x128_vs_16x256', total_size=(32, 256),
         block_size_a=(8, 128), block_size_b=(16, 256),
         tile_size=(8, 128), transpose_grid=True),
    dict(testcase_name='8x128_vs_32x128', total_size=(32, 128),
         block_size_a=(8, 128), block_size_b=(32, 128),
         tile_size=(8, 128), transpose_grid=False),
    dict(testcase_name='16x256_vs_32x128', total_size=(32, 256),
         block_size_a=(16, 256), block_size_b=(32, 128),
         tile_size=(8, 128), transpose_grid=False),
    dict(testcase_name='128x128_vs_128x256_padding',
         total_size=(256, 128), block_size_a=(128, 128),
         block_size_b=(128, 256), tile_size=(128, 128), transpose_grid=False),
    dict(testcase_name='128x128_vs_128x256_padding2',
         total_size=(257, 129), block_size_a=(128, 128),
         block_size_b=(128, 256), tile_size=(128, 128), transpose_grid=False),
  )
  def test_block_shape_invariance(self, total_size, block_size_a,
                          block_size_b, tile_size, transpose_grid):
    global_key = jax.random.key(0)
    ceil_div = lambda x, y: (x + y - 1) // y
    grid_a = tuple(ceil_div(_tot, _blk)
                   for _tot, _blk in zip(total_size, block_size_a))
    result_a = call_kernel(
        uniform_kernel, grid_a, transpose_grid, global_key,
        total_size, block_size_a, tile_size)

    grid_b = tuple(ceil_div(_tot, _blk)
                   for _tot, _blk in zip(total_size, block_size_b))
    result_b = call_kernel(
        uniform_kernel, grid_b, transpose_grid, global_key,
        total_size, block_size_b, tile_size)
    np.testing.assert_array_equal(result_a, result_b)


class BlockedFoldInTest(jtu.JaxTestCase):
  @parameterized.named_parameters(
  # Check that sampling a tensor of total size > jnp.iinfo(jnp.uint32).max works
  # as expected. Specifically, blocked key folding does not depend on the total
  # size of the tensor, but only the total number of tiles.
  # Using a 3D grid (with very large inner dimensions) triggers an overflow in a
  # previous implementation of blocked_fold_in.
  dict(testcase_name='4096x512_vs_1024x2048',
         total_size=(2, 64 * 1024, 64 * 1024), block_size_a=(1, 4096, 512),
         block_size_b=(1, 1024, 2048), tile_size=(1, 1024, 512)),
  )
  def test_blocked_fold_in_shape_invariance(self, total_size, block_size_a,
                                            block_size_b, tile_size):
    global_key = jax.random.key(0)
    grid_a = tuple(_tot // _blk for _tot, _blk in zip(total_size, block_size_a))
    result_a = call_kernel_3d(
        blocked_fold_in, grid_a, global_key, total_size,
        block_size_a, tile_size)

    grid_b = tuple(_tot // _blk for _tot, _blk in zip(total_size, block_size_b))
    result_b = call_kernel_3d(
        blocked_fold_in, grid_b, global_key, total_size,
        block_size_b, tile_size)
    np.testing.assert_array_equal(jax.random.key_data(result_a),
                                  jax.random.key_data(result_b))



if __name__ == "__main__":
  absltest.main()
