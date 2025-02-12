# Copyright 2024 The JAX Authors. All Rights Reserved.
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
"""Test different parameterizations of FlashAttention."""

import os

import numpy as np
from absl.testing import absltest, parameterized
from jax._src import config
from jax._src import test_util as jtu
import jax.numpy as jnp

# pylint: disable=g-import-not-at-top
try:
  # We only import this to see if Mosaic is available.
  import jax.experimental.mosaic.gpu  # noqa: F401
except ImportError:
  attention_mgpu = None
else:
  from jax.experimental.pallas.ops.gpu import attention_mgpu


config.parse_flags_with_absl()
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0")


@jtu.with_config(jax_traceback_filtering="off")
class FlashAttentionTestCase(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if attention_mgpu is None:
      self.skipTest("Mosaic GPU not available.")
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_equal("9.0")):
      self.skipTest("Only works on GPU with capability sm90a")

  @parameterized.product(
      batch_size=(1, 4),
      q_seq_len=(4096,),
      kv_seq_len=(4096,),
      num_q_and_kv_heads=((4, 1),    # MQA
                          (6, 3),    # GQA
                          (4, 4),),  # MHA
      head_dim=(64, 128, 256),
  )
  def test_flash_attention(
      self, batch_size, q_seq_len, kv_seq_len, num_q_and_kv_heads, head_dim
  ):
    num_q_heads, num_kv_heads = num_q_and_kv_heads
    k1, k2, k3 = jax.random.split(jax.random.key(42), 3)
    q = jax.random.normal(k1, (batch_size, q_seq_len, num_q_heads, head_dim), jnp.float16)
    k = jax.random.normal(k2, (batch_size, kv_seq_len, num_kv_heads, head_dim), jnp.float16)
    v = jax.random.normal(k3, (batch_size, kv_seq_len, num_kv_heads, head_dim), jnp.float16)
    out = attention_mgpu.attention(
        q, k, v, attention_mgpu.TuningConfig(block_q=64, block_kv=64, max_concurrent_steps=2)
    )
    out_ref = attention_mgpu.attention_reference(q, k, v)
    np.testing.assert_allclose(out, out_ref, atol=2e-3, rtol=1e-3)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
