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

import functools

from absl.testing import absltest
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.pallas import pallas_call
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np

try:
  import torch
except ImportError:
  torch = None

# pylint: disable=g-import-not-at-top
try:
  # We only import this to see if Mosaic is available.
  import jax.experimental.mosaic.gpu  # noqa: F401
except ImportError:
  attention_mgpu = None
else:
  from jax.experimental.pallas.ops.gpu import attention_mgpu


config.parse_flags_with_absl()


class TorchTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if torch is None:
      self.skipTest("Test requires PyTorch")
    if attention_mgpu is None:
      self.skipTest("Mosaic GPU not available.")
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_equal("9.0")):
      self.skipTest("Only works on GPU with capability sm90a")
    self.enter_context(pallas_call._PALLAS_USE_MOSAIC_GPU(True))

  def test_simple(self):
    @plgpu.as_torch_kernel
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([128], jnp.int32),
    )
    def kernel(x_ref, y_ref, o_ref):
      o_ref[...] = x_ref[...] + y_ref[0]

    x = torch.arange(128, dtype=torch.int32, device="cuda")
    y = torch.arange(128, dtype=torch.int32, device="cuda")
    np.testing.assert_array_equal(kernel(x, y).cpu(), (x + y[0]).cpu())

  def test_attention(self):
    batch_size = 1
    q_seq_len = 4096
    kv_seq_len = 4096
    head_dim = 64
    num_q_heads, num_kv_heads = 4, 1
    block_q = block_kv = 64
    q = torch.randn(
        (batch_size, q_seq_len, num_q_heads, head_dim),
        dtype=torch.float16,
        device="cuda",
    )
    k = torch.randn(
        (batch_size, kv_seq_len, num_kv_heads, head_dim),
        dtype=torch.float16,
        device="cuda",
    )
    v = torch.randn(
        (batch_size, kv_seq_len, num_kv_heads, head_dim),
        dtype=torch.float16,
        device="cuda",
    )
    kernel_fn = functools.partial(
        attention_mgpu.attention,
        config=attention_mgpu.TuningConfig(
            block_q=block_q,
            block_kv=block_kv,
            max_concurrent_steps=2,
        ),
    )
    # Just make sure it computes something.
    _ = plgpu.as_torch_kernel(kernel_fn)(q, k, v)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
