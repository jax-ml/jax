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
"""Test different parameterizations of our Mosaic GPU collective matmul."""

import functools
import os

from absl.testing import parameterized  # pylint: disable=g-multiple-import
import jax
from jax import lax
from jax import random
from jax._src import test_multiprocess as jt_multiprocess
from jax._src import test_util as jtu
from jax._src.pallas import pallas_call
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.pallas.ops.gpu import all_gather_lhs_matmul_mgpu
import jax.numpy as jnp
import numpy as np


P = jax.sharding.PartitionSpec


@jtu.with_config(jax_traceback_filtering="off")
class CollectiveMatmulTestCase(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if all_gather_lhs_matmul_mgpu is None:
      self.skipTest("Mosaic GPU not available.")
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_equal("9.0")):
      self.skipTest("Only works on GPU with capability sm90a")
    if not mgpu.supports_cross_device_collectives():
      if "FAIL_ON_NVSHMEM_UNAVAILABLE" in os.environ:
        raise ValueError("NVSHMEM library unavailable.")
      else:
        self.skipTest("NVSHMEM library unavailable.")
    if jax.process_count() == 1:
      self.skipTest("Test requires multiple processes.")
    if os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "") == "platform":
      self.skipTest("NVSHMEM doesn't work with the platform allocator.")
    self.enter_context(pallas_call._PALLAS_USE_MOSAIC_GPU(True))
    num_devices = jax.device_count()
    mesh = jax.make_mesh(
        (num_devices,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,)
    )
    self.enter_context(jax.set_mesh(mesh))

  @parameterized.product(
      m_shard=(3072,),
      n_shard=(256, 576),
      k=(4096,),
      tile_m=(64, 128, 192),
      tile_n=(64, 128, 192),
      tile_k=(64, 128),
      grid_minor_dim=(all_gather_lhs_matmul_mgpu.MatmulDimension.N,),
      grid_tile_width=(1,),
      wg_dimension=(all_gather_lhs_matmul_mgpu.MatmulDimension.N,),
      max_concurrent_steps=(2, 4),
      dtype=(jnp.bfloat16,),
  )
  def test_all_gather_lhs_matmul(
      self,
      m_shard,
      n_shard,
      k,
      tile_m,
      tile_n,
      tile_k,
      max_concurrent_steps,
      grid_minor_dim,
      grid_tile_width,
      wg_dimension,
      dtype,
  ):
    num_devices = jax.device_count()
    epi_tile_size = 64 * 64
    num_epi_tiles = tile_m * tile_n // epi_tile_size
    cta_tile_m = tile_m * (1 + (wg_dimension == all_gather_lhs_matmul_mgpu.MatmulDimension.M))
    cta_tile_n = tile_n * (1 + (wg_dimension == all_gather_lhs_matmul_mgpu.MatmulDimension.N))
    if (
        (cta_tile_m + cta_tile_n) * tile_k * max_concurrent_steps
        + 2 * min(2, num_epi_tiles) * epi_tile_size
    ) * 2 > 228000:
      self.skipTest("Tile too big to fit into SMEM")
    if n_shard % cta_tile_n:
      self.skipTest("n_shard must be divisible by block_n for now.")
    if m_shard % cta_tile_m:
      self.skipTest("m_shard must be divisible by block_m for now.")

    k1, k2 = random.split(random.key(1234), num=2)
    lhs = random.normal(k1, (num_devices * m_shard, k), dtype)
    rhs = random.normal(k2, (k, num_devices * n_shard), dtype)
    lhs = jax.sharding.reshard(lhs, P("x", None))
    rhs = jax.sharding.reshard(rhs, P(None, "x"))

    def run(body):
      out = jax.jit(
          jax.shard_map(body, out_specs=P(None, "x"), check_vma=False)
      )(lhs, rhs)
      # Gather output, for NumPy comparison on the host.
      out = jax.shard_map(
          lambda x: lax.all_gather(x, "x", axis=1, tiled=True),
          out_specs=P(None), check_vma=False,
      )(out)
      return out

    ref_out = run(lambda x, y: lax.all_gather(x, "x", axis=0, tiled=True) @ y)
    config = all_gather_lhs_matmul_mgpu.TuningConfig(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        max_concurrent_steps=max_concurrent_steps,
        grid_minor_dim=grid_minor_dim,
        grid_tile_width=grid_tile_width,
        wg_dimension=wg_dimension,
    )
    out = run(
        functools.partial(
            all_gather_lhs_matmul_mgpu.all_gather_lhs_matmul,
            axis_name="x",
            config=config,
            dtype=dtype,
        )
    )
    np.testing.assert_allclose(out, ref_out)


if __name__ == "__main__":
  # This test doesn't work with the platform allocator, so we override it
  # if it's ran alone. If it's part of a larger test suite and the platform
  # allocator is used, setUp will skip the test.
  os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
  os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "default"
  os.environ["XLA_FLAGS"] = (
      os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0"
  )
  jt_multiprocess.main()
