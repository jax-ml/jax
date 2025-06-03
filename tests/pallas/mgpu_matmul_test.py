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

import contextlib
import os

from absl.testing import absltest
from absl.testing import parameterized
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
  blackwell_matmul_mgpu = None
else:
  from jax.experimental.pallas.ops.gpu import blackwell_matmul_mgpu


config.parse_flags_with_absl()
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0")


@jtu.with_config(jax_traceback_filtering="off")
class MatrixMultiplicationSm100ATest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if blackwell_matmul_mgpu is None:
      self.skipTest("Mosaic GPU not available.")
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_equal("10.0")):
      self.skipTest("Only works on GPU with capability sm100a")
    context_stack = contextlib.ExitStack()
    context_stack.enter_context(pallas_call._PALLAS_USE_MOSAIC_GPU(True))
    self.addCleanup(context_stack.close)

  @parameterized.product(
      m=(1024, 4096),
      k=(1024, 4096),
      n=(1024, 4096),
      dtype=(jnp.float16,),
  )
  def test_matmul(
      self,
      m,
      n,
      k,
      dtype,
  ):
    k1, k2, = jax.random.split(jax.random.key(42), 2)
    a = jax.random.normal(k1, (m, k), dtype)
    b = jax.random.normal(k2, (k, n), dtype)

    out = blackwell_matmul_mgpu.matmul_kernel(
        a,
        b,
        blackwell_matmul_mgpu.TuningConfig(
            block_m=128, block_n=128, block_k=128,
            max_concurrent_steps=2,
            collective=False,
        ),
    )
    out_ref = a @ b
    np.testing.assert_allclose(out, out_ref, atol=2e-3, rtol=1e-3)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
