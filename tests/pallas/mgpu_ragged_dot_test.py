# Copyright 2025 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test different parameterizations of our Mosaic GPU ragged dot kernel."""

import os

from absl.testing import absltest, parameterized  # pylint: disable=g-multiple-import
import jax
from jax import random
from jax._src import config
from jax._src import test_util as jtu
from jax._src.pallas import pallas_call
from jax.experimental.pallas.ops.gpu import blackwell_ragged_dot_mgpu
from jax.experimental.pallas.ops.gpu import ragged_dot_mgpu
from jax.experimental.pallas.ops.gpu import transposed_ragged_dot_mgpu
import jax.numpy as jnp
import numpy as np


config.parse_flags_with_absl()


# TODO(justinfu): Test empty groups
def sample_inputs(
    key, m, k, n, num_groups, dtype=jnp.float16, transposed=False,
):
  kx, ky, kz = random.split(key, num=3)
  if transposed:
    lhs = jax.random.normal(kx, (k, m), dtype)
    rhs = jax.random.normal(ky, (k, n), dtype)
    batch_size = k
  else:
    lhs = jax.random.normal(kx, (m, k), dtype)
    rhs = jax.random.normal(ky, (num_groups, k, n), dtype)
    batch_size = m
  group_boundaries = jax.lax.sort(
      jax.random.randint(kz, (num_groups - 1,), 0, batch_size, jnp.int32)
  )
  group_starts = jax.lax.concatenate(
      [jnp.array([0], dtype=jnp.int32), group_boundaries], 0
  )
  group_ends = jax.lax.concatenate(
      [group_boundaries, jnp.array([batch_size], dtype=jnp.int32)], 0
  )
  group_sizes = group_ends - group_starts
  assert group_sizes.shape == (num_groups,)
  return lhs, rhs, group_sizes


@jtu.with_config(jax_traceback_filtering="off")
class RaggedDotTestCase(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if ragged_dot_mgpu is None:
      self.skipTest("Mosaic GPU not available.")
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_equal("9.0")):
      self.skipTest("Only works on GPU with capability sm90a")
    self.enter_context(pallas_call._PALLAS_USE_MOSAIC_GPU(True))

  @parameterized.product(
      block_m=(64, 128),
      block_n=(64, 128, 192),
      block_k=(64, 128),
      grid_block_n=(2, 4),
      max_concurrent_steps=(2, 4),
      num_groups=(1, 3, 16),
      transpose_rhs=(False, True),
  )
  def test_ragged_dot(
      self,
      block_m,
      block_n,
      block_k,
      grid_block_n,
      max_concurrent_steps,
      num_groups,
      transpose_rhs,
  ):
    dtype = jnp.float16
    lhs_smem_size = block_m * block_k * max_concurrent_steps * 2
    rhs_smem_size = block_k * block_n * max_concurrent_steps * 2
    # H100 SMEM limit is 228kB.
    if lhs_smem_size + rhs_smem_size > 228_000:
      self.skipTest("This configuration requires too much SMEM.")

    m, k, n = 16 * 1024, 2048, 16 * 1024
    lhs, rhs, group_sizes = sample_inputs(
        random.key(1234), m, k, n, num_groups, dtype
    )

    out = ragged_dot_mgpu.ragged_dot(
        lhs,
        jnp.transpose(rhs, (0, 2, 1)) if transpose_rhs else rhs,
        group_sizes=group_sizes,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        max_concurrent_steps=max_concurrent_steps,
        grid_block_n=grid_block_n,
        transpose_rhs=transpose_rhs,
    )
    out_ref = jax.lax.ragged_dot(lhs, rhs, group_sizes=group_sizes)
    np.testing.assert_allclose(out, out_ref, atol=1e-3, rtol=1e-3)

  @parameterized.product(
      block_m=(64, 128),
      block_n=(64, 128),
      block_k=(64, 128),
      grid_block_n=(2, 4),
      max_concurrent_steps=(1,),
      num_groups=(1, 3, 16),
  )
  def test_ragged_dot_transposed(
      self,
      block_m,
      block_n,
      block_k,
      grid_block_n,
      max_concurrent_steps,
      num_groups,
  ):
    dtype = jnp.float16
    lhs_smem_size = block_m * block_k * max_concurrent_steps * 2
    rhs_smem_size = block_k * block_n * max_concurrent_steps * 2
    # H100 SMEM limit is 228kB.
    if lhs_smem_size + rhs_smem_size > 228_000:
      self.skipTest("This configuration requires too much SMEM.")

    k, m, n, num_groups = 16 * 1024, 2048, 2048, 16
    lhs, rhs, group_sizes = sample_inputs(
        random.key(1234), m, k, n, num_groups,
        dtype=dtype, transposed=True,
    )

    with jax.numpy_dtype_promotion("standard"):
      # We need standard dtype promotion for dynamic grid size to work, because
      # python integers are treated as int64, and some of the dtypes inside
      # emit_pipeline are hardcoded to use int32.
      out = transposed_ragged_dot_mgpu.transposed_ragged_dot(
          lhs,
          rhs,
          group_sizes=group_sizes,
          block_m=block_m,
          block_n=block_n,
          block_k=block_k,
          max_concurrent_steps=max_concurrent_steps,
          grid_block_n=grid_block_n,
      )
    out_ref = jax.lax.ragged_dot_general(
        lhs, rhs, group_sizes,
        ragged_dot_dimension_numbers=jax.lax.RaggedDotDimensionNumbers(
            dot_dimension_numbers=(((0,), (0,)), ((), ())),
            lhs_ragged_dimensions=[0],
            rhs_group_dimensions=[],
        )
    )
    np.testing.assert_allclose(out, out_ref, atol=1e-3, rtol=1e-3)


@jtu.with_config(jax_traceback_filtering="off")
class RaggedDotSm100aTestCase(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if blackwell_ragged_dot_mgpu is None:
      self.skipTest("Mosaic GPU not available.")
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_equal("10.0")):
      self.skipTest("Only works on GPU with capability sm100a")
    self.enter_context(pallas_call._PALLAS_USE_MOSAIC_GPU(True))

  @parameterized.product(
      grid_tile_width=(1, 8, 16),
      grid_minor_dim=(0, 1),
      max_concurrent_steps=(2, 4),
      num_groups=(1, 3, 16),
      tile_k=(64, 128)
  )
  def test_ragged_dot(
      self,
      grid_tile_width,
      grid_minor_dim,
      max_concurrent_steps,
      num_groups,
      tile_k,
  ):
    # Kernel does not support other tiling on M and N dimensions currently.
    tile_m = 128
    tile_n = 128

    lhs_smem_size = tile_m * tile_k * max_concurrent_steps * 2
    rhs_smem_size = tile_k * tile_n * max_concurrent_steps * 2
    # B200 SMEM limit is 228kB.
    if lhs_smem_size + rhs_smem_size > 228_000:
      self.skipTest("This configuration requires too much SMEM.")

    dtype = jnp.float16
    m, k, n = 16 * 1024, 2048, 16 * 1024
    lhs, rhs, group_sizes = sample_inputs(
        random.key(1234), m, k, n, num_groups, dtype
    )
    tuning_config = blackwell_ragged_dot_mgpu.TuningConfig(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        grid_tile_width=grid_tile_width,
        grid_minor_dim=grid_minor_dim,
        max_concurrent_steps=max_concurrent_steps,
        collective=True,
    )
    out = blackwell_ragged_dot_mgpu.ragged_dot_kernel(
        lhs,
        rhs,
        group_sizes=group_sizes,
        config=tuning_config,
    )
    out_ref = jax.lax.ragged_dot(lhs, rhs, group_sizes=group_sizes,
                                 preferred_element_type=dtype)
    np.testing.assert_allclose(out, out_ref, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
  os.environ["XLA_FLAGS"] = (
      os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0"
  )
  absltest.main(testLoader=jtu.JaxTestLoader())
