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

import contextlib
import os

from absl.testing import absltest, parameterized  # pylint: disable=g-multiple-import
from jax import random
from jax._src import config
from jax._src import test_util as jtu
from jax._src.pallas import pallas_call
import jax.numpy as jnp
import numpy as np

# pylint: disable=g-import-not-at-top
try:
  # We only import this to see if Mosaic is available.
  import jax.experimental.mosaic.gpu  # noqa: F401
except ImportError:
  ragged_dot = None
else:
  from jax.experimental.pallas.ops.gpu import ragged_dot_mgpu


config.parse_flags_with_absl()


@jtu.with_config(jax_traceback_filtering="off")
class RaggedDotTestCase(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if ragged_dot_mgpu is None:
      self.skipTest("Mosaic GPU not available.")
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_equal("9.0")):
      self.skipTest("Only works on GPU with capability sm90a")
    context_stack = contextlib.ExitStack()
    context_stack.enter_context(pallas_call._PALLAS_USE_MOSAIC_GPU(True))
    self.addCleanup(context_stack.close)

  @parameterized.product(
      block_m=(64, 128, 192),
      block_n=(64, 128, 192),
      block_k=(64, 128),
      grid_block_n=(2, 4),
      max_concurrent_steps=(2, 4),
      num_groups=(1, 3, 16),
  )
  def test_ragged_dot(
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

    m, k, n = 16 * 1024, 2048, 16 * 1024
    kx, ky, kz = random.split(random.key(1234), num=3)

    lhs = jax.random.normal(kx, (m, k), dtype)
    rhs = jax.random.normal(ky, (num_groups, k, n), dtype)
    group_boundaries = jax.lax.sort(
        jax.random.randint(kz, (num_groups - 1,), 0, m, jnp.int32)
    )
    group_starts = jax.lax.concatenate(
        [jnp.array([0], dtype=jnp.int32), group_boundaries], 0
    )
    group_ends = jax.lax.concatenate(
        [group_boundaries, jnp.array([m], dtype=jnp.int32)], 0
    )
    group_sizes = group_ends - group_starts
    assert group_sizes.shape == (num_groups,)

    out = ragged_dot_mgpu.ragged_dot(
        lhs,
        rhs,
        group_sizes=group_sizes,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        max_concurrent_steps=max_concurrent_steps,
        grid_block_n=grid_block_n,
    )
    out_ref = jax.lax.ragged_dot(lhs, rhs, group_sizes=group_sizes)
    np.testing.assert_allclose(out, out_ref, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
  os.environ["XLA_FLAGS"] = (
      os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0"
  )
  absltest.main(testLoader=jtu.JaxTestLoader())
