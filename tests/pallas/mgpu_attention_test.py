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

import contextlib
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import config
from jax._src import test_util as jtu
from jax._src.pallas import pallas_call
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
    context_stack = contextlib.ExitStack()
    context_stack.enter_context(pallas_call._PALLAS_USE_MOSAIC_GPU(True))
    self.addCleanup(context_stack.close)

  @parameterized.product(
      batch_size=(1, 4),
      q_seq_len=(4096,),
      kv_seq_len=(4096,),
      num_q_and_kv_heads=(
          (4, 1),  # MQA
          (6, 3),  # GQA
          (4, 4),
      ),  # MHA
      head_dim=(64, 128, 256),
      attention_impl=(
          attention_mgpu.attention,
          attention_mgpu.attention_with_pipeline_emitter,
      ),
      save_residuals=(True,),
  )
  def test_flash_attention(
      self,
      batch_size,
      q_seq_len,
      kv_seq_len,
      num_q_and_kv_heads,
      head_dim,
      attention_impl,
      save_residuals,
  ):
    num_q_heads, num_kv_heads = num_q_and_kv_heads
    k1, k2, k3 = jax.random.split(jax.random.key(42), 3)
    q = jax.random.normal(k1, (batch_size, q_seq_len, num_q_heads, head_dim), jnp.float16)
    k = jax.random.normal(k2, (batch_size, kv_seq_len, num_kv_heads, head_dim), jnp.float16)
    v = jax.random.normal(k3, (batch_size, kv_seq_len, num_kv_heads, head_dim), jnp.float16)
    out, *res = attention_impl(
        q,
        k,
        v,
        attention_mgpu.TuningConfig(
            block_q=64, block_kv=64, max_concurrent_steps=2
        ),
        save_residuals=save_residuals,
    )
    out_ref, *res_ref = attention_mgpu.attention_reference(q, k, v, save_residuals=save_residuals)
    np.testing.assert_allclose(out, out_ref, atol=2e-3, rtol=1e-3)
    if save_residuals:
      (lse,) = res[0]
      (lse_ref,) = res_ref[0]
      np.testing.assert_allclose(lse, lse_ref, atol=2e-3, rtol=1e-3)

  @parameterized.product(
      batch_size=(3,),
      seq_lens=((512, 512), (3584, 4096)),
      num_q_and_kv_heads=(
          (4, 4),  # MHA
          (4, 1),  # MQA
          (6, 3),  # GQA
          ),
      bwd_blocks = (
          (64, 64, 64, 64),
          (64, 128, 128, 64),
          (128, 128, 128, 128),
      ),
      head_dim=(64, 128, 256),
  )
  def test_bwd_flash_attention(
      self,
      batch_size,
      seq_lens,
      num_q_and_kv_heads,
      bwd_blocks,
      head_dim,
  ):
    num_q_heads, num_kv_heads = num_q_and_kv_heads
    kv_seq_len, q_seq_len = seq_lens
    block_q_dq, block_kv_dq, block_q_dkv, block_kv_dkv = bwd_blocks
    compute_wgs = 2 if head_dim <= 128 else 1
    k1, k2, k3 = jax.random.split(jax.random.key(42), 3)
    q = jax.random.normal(k1, (batch_size, q_seq_len, num_q_heads, head_dim), jnp.float16)
    k = jax.random.normal(k2, (batch_size, kv_seq_len, num_kv_heads, head_dim), jnp.float16)
    v = jax.random.normal(k3, (batch_size, kv_seq_len, num_kv_heads, head_dim), jnp.float16)

    def f(q, k, v):
      return attention_mgpu.attention(
          q,
          k,
          v,
          attention_mgpu.TuningConfig(
              block_q=block_q_dq, block_kv=block_kv_dq,
              max_concurrent_steps=2, compute_wgs_bwd=compute_wgs,
              block_q_dkv=block_q_dkv, block_kv_dkv=block_kv_dkv,
              block_q_dq=block_q_dq, block_kv_dq=block_kv_dq,
          )
      ).sum()

    def f_ref(q, k, v):
      return attention_mgpu.attention_reference(q, k, v).sum()

    try:
      # TODO(pobudzey): Replace with `jtu.check_grads` when it's fixed.
      dq, dk, dv = jax.grad(f, argnums=(0, 1, 2))(q, k, v)
      dq_ref, dk_ref, dv_ref = jax.grad(f_ref, argnums=(0, 1, 2))(q, k, v)

      self.assertAllClose(dq, dq_ref, atol=7e-2)
      self.assertAllClose(dk, dk_ref, atol=7e-2)
      self.assertAllClose(dv, dv_ref, atol=5e-2)

    except ValueError as e:
      if "exceeds available shared memory" in e.args[0]:
        self.skipTest("Not enough SMEM for this configuration.")

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
