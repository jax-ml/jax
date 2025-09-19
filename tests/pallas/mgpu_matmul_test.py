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
"""Test different parameterizations of matrix multiplication."""

import os

from absl.testing import absltest
from absl.testing import parameterized
from jax._src import config
from jax._src import test_util as jtu
from jax._src.pallas import pallas_call
import jax.experimental.mosaic.gpu  # noqa: F401
from jax.experimental.pallas.ops.gpu import blackwell_matmul_mgpu
from jax.experimental.pallas.ops.gpu import hopper_matmul_mgpu
import jax.numpy as jnp
import numpy as np


config.parse_flags_with_absl()
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0")


@jtu.with_config(jax_traceback_filtering="off")
class MatrixMultiplicationSm100ATest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cuda"]):
      self.skipTest("Test requires an NVIDIA GPU")
    self.enter_context(pallas_call._PALLAS_USE_MOSAIC_GPU(True))

  @parameterized.product(
      m=(1024, 4096),
      k=(1024, 4096),
      n=(1024, 4096),
      dtype=(jnp.float16,),
  )
  def test_blackwell_matmul(
      self,
      m,
      n,
      k,
      dtype,
  ):
    if not jtu.is_cuda_compute_capability_equal("10.0"):
      self.skipTest("Only works on GPU with capability sm100a")
    k1, k2, = jax.random.split(jax.random.key(42), 2)
    a = jax.random.normal(k1, (m, k), dtype)
    b = jax.random.normal(k2, (k, n), dtype)

    out = blackwell_matmul_mgpu.matmul_kernel(
        a,
        b,
        blackwell_matmul_mgpu.TuningConfig(
            tile_m=128, tile_n=128, tile_k=128,
            max_concurrent_steps=2,
            collective=False,
        ),
    )
    out_ref = a @ b
    np.testing.assert_allclose(out, out_ref, atol=2e-3, rtol=1e-3)

@jtu.with_config(jax_traceback_filtering="off")
class MatrixMultiplicationSm90ATest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cuda"]):
      self.skipTest("Test requires an NVIDIA GPU")
    self.enter_context(pallas_call._PALLAS_USE_MOSAIC_GPU(True))

  @parameterized.product(
      m=(4096,),
      k=(4096,),
      n=(4096,),
      tile_m=(64, 128),
      tile_n=(64, 128),
      tile_k=(64, 128),
      max_concurrent_steps=(2, 4),
      dtype=(jnp.float16,),
      epi_tile_n=(None, 64),
      epi_tile_m=(None, 64),
      wg_dimension=tuple(hopper_matmul_mgpu.MatmulDimension),
  )
  def test_hopper_matmul(self, *args, **kwargs):
    self.check_hopper_matmul(*args, **kwargs)

  # Grid tiling doesn't really interact with many other options so we can test
  # it separately.
  @parameterized.product(
      grid_minor_dim=tuple(hopper_matmul_mgpu.MatmulDimension),
      grid_tile_width=(1, 3, 4),
  )
  def test_hopper_matmul_grid_tiling(self, grid_minor_dim, grid_tile_width):
    self.check_hopper_matmul(
        m=4096,
        k=4096,
        n=4096,
        dtype=jnp.float16,
        tile_m=64,
        tile_n=64,
        tile_k=64,
        max_concurrent_steps=2,
        epi_tile_m=64,
        epi_tile_n=64,
        wg_dimension=hopper_matmul_mgpu.MatmulDimension.M,
        grid_minor_dim=grid_minor_dim,
        grid_tile_width=grid_tile_width,
    )

  @parameterized.product(
    tile_m=(64, 128),
    tile_n=(64, 128),
    wg_dimension=tuple(hopper_matmul_mgpu.MatmulDimension),
    cluster_dimension=tuple(hopper_matmul_mgpu.MatmulDimension),
  )
  def test_hopper_matmul_cluster(self, tile_m, tile_n, wg_dimension, cluster_dimension):
    self.check_hopper_matmul(
        m=4096,
        k=4096,
        n=4096,
        dtype=jnp.float16,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=64,
        max_concurrent_steps=4,
        epi_tile_m=64,
        epi_tile_n=64,
        wg_dimension=wg_dimension,
        cluster_dimension=cluster_dimension,
    )

  def check_hopper_matmul(
      self,
      m,
      n,
      k,
      dtype,
      tile_m,
      tile_n,
      tile_k,
      max_concurrent_steps,
      epi_tile_m,
      epi_tile_n,
      wg_dimension,
      **kwargs
  ):
    if not jtu.is_cuda_compute_capability_equal("9.0"):
      self.skipTest("Only works on GPU with capability sm90a")

    epi_tile_size = (epi_tile_m or tile_m) * (epi_tile_n or tile_n)
    num_epi_tiles = tile_m * tile_n // epi_tile_size
    cta_tile_m = tile_m * (1 + (wg_dimension == hopper_matmul_mgpu.MatmulDimension.M))
    cta_tile_n = tile_n * (1 + (wg_dimension == hopper_matmul_mgpu.MatmulDimension.N))
    if (
        (cta_tile_m + cta_tile_n) * tile_k * max_concurrent_steps
        + 2 * min(2, num_epi_tiles) * epi_tile_size
    ) * 2 > 228000:
      self.skipTest("Tile too big to fit into SMEM")
    k1, k2, = jax.random.split(jax.random.key(42), 2)
    a = jax.random.normal(k1, (m, k), dtype)
    b = jax.random.normal(k2, (k, n), dtype)

    spec = hopper_matmul_mgpu.TuningConfig(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        max_concurrent_steps=max_concurrent_steps,
        epi_tile_m=epi_tile_m,
        epi_tile_n=epi_tile_n,
        wg_dimension=wg_dimension,
        **kwargs,
    )
    out = hopper_matmul_mgpu.matmul_kernel(a, b, spec)
    out_ref = jnp.dot(a, b, precision=jax.lax.DotAlgorithmPreset.F16_F16_F32)
    np.testing.assert_allclose(out, out_ref)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
