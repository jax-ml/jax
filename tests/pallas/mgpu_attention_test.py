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

import math
import os
from typing import Literal

import numpy as np
from absl.testing import absltest, parameterized
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lib import cuda_versions
from jax._src.pallas import pallas_call
import jax.numpy as jnp

# pylint: disable=g-import-not-at-top
try:
  # We only import this to see if Mosaic is available.
  import jax.experimental.mosaic.gpu  # noqa: F401
except ImportError:
  attention_mgpu = None
  attention_mask = None
  hopper_splash_attention = None
else:
  from jax.experimental.pallas.ops.gpu import attention_mgpu
  from jax.experimental.pallas.ops.gpu.splash_attention import attention_mask
  from jax.experimental.pallas.ops.gpu.splash_attention import hopper_splash_attention

config.parse_flags_with_absl()
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0")

MaskType = Literal[
    "full", "zeros", "causal", "random", "multidoc", "sliding_window", "zeroed_partial_blocks", "realistic"
]



@jtu.with_config(jax_traceback_filtering="off")
class FlashAttentionTestCase(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if attention_mgpu is None:
      self.skipTest("Mosaic GPU not available.")
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_equal("9.0")):
      self.skipTest("Only works on GPU with capability sm90a")
    self.enter_context(pallas_call._PALLAS_USE_MOSAIC_GPU(True))

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
      blocks=((64, 64), (64, 128), (128, 64)),
      attention_impl=(
          attention_mgpu.attention,
          attention_mgpu.attention_with_pipeline_emitter,
      ),
      save_residuals=(True,),
      causal=(True, False,),
  )
  def test_flash_attention(
      self,
      batch_size,
      q_seq_len,
      kv_seq_len,
      num_q_and_kv_heads,
      head_dim,
      blocks,
      attention_impl,
      save_residuals,
      causal,
  ):
    assert cuda_versions is not None
    cuda_runtime_version = cuda_versions.cuda_runtime_get_version()
    # TODO(pobudzey): Undo when we upgrade to cuda 12.9.1.
    if causal and (cuda_runtime_version >= 12080 and cuda_runtime_version < 12091):
      self.skipTest("Skipping because of ptxas miscompilation.")

    if causal and attention_impl == attention_mgpu.attention_with_pipeline_emitter:
      self.skipTest("Pipeline emitter does not support causal attention.")

    if head_dim >= 256 and max(blocks) >= 128:
      self.skipTest("Head dim too large for block sizes.")

    num_q_heads, num_kv_heads = num_q_and_kv_heads
    block_q, block_kv = blocks
    k1, k2, k3 = jax.random.split(jax.random.key(42), 3)
    q = jax.random.normal(k1, (batch_size, q_seq_len, num_q_heads, head_dim), jnp.float16)
    k = jax.random.normal(k2, (batch_size, kv_seq_len, num_kv_heads, head_dim), jnp.float16)
    v = jax.random.normal(k3, (batch_size, kv_seq_len, num_kv_heads, head_dim), jnp.float16)
    out, *res = attention_impl(
        q,
        k,
        v,
        attention_mgpu.TuningConfig(
            block_q=block_q, block_kv=block_kv, max_concurrent_steps=2, causal=causal
        ),
        save_residuals=save_residuals,
    )
    out_ref, *res_ref = attention_mgpu.attention_reference(
        q, k, v, causal=causal, save_residuals=save_residuals)
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


@jtu.with_config(jax_traceback_filtering="off")
class AttentionMaskTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if attention_mgpu is None:
      self.skipTest("Mosaic GPU not available.")

  def test_fwd_mask_2_tiling(self):
    mask = jnp.array(
        [
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        ]
    )
    mask = mask.reshape((1, 1, mask.shape[0], mask.shape[1])).astype(bool)
    mask_info = attention_mask.process_dynamic_mask(
        mask=mask,
        block_shape=(2, 2),
        is_dkv=False,
        data_tiling=2,
    )
    np.testing.assert_array_equal(
        mask_info.block_mask[0, 0],
        jnp.array(
            [
                [2, 1, 2, 0, 0],
                [2, 2, 0, 0, 0],
                [2, 1, 0, 1, 0],
                [0, 0, 1, 0, 0],
            ]
        ),
    )
    np.testing.assert_array_equal(
        mask_info.data_next[0, 0],
        jnp.array(
            [
                [0, 1, 4, -1, -1],
                [0, 1, 2, 4, -1],
            ]
        ),
    )
    np.testing.assert_array_equal(
        mask_info.mask_next[0, 0],
        jnp.array(
            [
                [-1, 1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, 11, -1, 14, -1],
                [-1, -1, 17, -1, -1],
            ]
        ),
    )

  def test_dkv_mask_2_tiling(self):
    mask = jnp.array(
        [
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0],
        ]
    )
    mask = mask.reshape((1, 1, mask.shape[0], mask.shape[1])).astype(bool)
    mask_info = attention_mask.process_dynamic_mask(
        mask=mask,
        block_shape=(2, 2),
        is_dkv=True,
        data_tiling=2,
    )
    np.testing.assert_array_equal(
        mask_info.block_mask[0, 0],
        jnp.array(
            [
                [0, 0, 0, 0],
                [0, 0, 2, 0],
                [2, 2, 1, 0],
                [1, 2, 0, 1],
                [2, 0, 1, 0],
            ]
        ),
    )
    np.testing.assert_array_equal(
        mask_info.data_next[0, 0],
        jnp.array(
            [
                [-1, -1],
                [-1, 0],
                [0, 1],
                [1, 2],
                [4, 4],
            ]
        ),
    )
    np.testing.assert_array_equal(
        mask_info.mask_next[0, 0],
        jnp.array(
            [
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, 6, -1],
                [4, -1, -1, 11],
                [-1, -1, 18, -1],
            ]
        ),
    )

  def test_compress(self):
    x = jnp.array(
        [
            [0, 1, 2, -1, -1, 10, 6, 7, -1, -1, -1, 9, -1],
            [-1, -1, -1, -1, 0, 1, 2, -1, -1, -1, -1, -1, -1],
        ]
    )
    expected = jnp.array(
        [
            [0, 1, 2, 10, 6, 7, 9, -1, -1, -1, -1, -1, -1],
            [0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        ]
    )
    np.testing.assert_array_equal(
        attention_mask.compress_array(x, placeholder=-1)[0], expected)

  def test_tiled_max(self):
    x = jnp.array([[0, 1, 2, -1, -1, -1], [-1, -1, 2, 3, 4, -1],
                  [-1, 1, 2, 3, -1, -1], [-1, -1, 2, 3, 4, -1]])
    expected = jnp.array([[0, 1, 2, 3, 4, -1], [-1, 1, 2, 3, 4, -1]])
    np.testing.assert_array_equal(
        attention_mask.tiled_max(x, tile_size=2, axis=-2), expected)


@jtu.with_config(jax_traceback_filtering="off")
class SplashAttentionTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if attention_mgpu is None:
      self.skipTest("Mosaic GPU not available.")
    if (not jtu.test_device_matches(["cuda"]) or
            not jtu.is_cuda_compute_capability_equal("9.0")):
      self.skipTest("Only works on GPU with capability sm90a")

    self.tuning_config = hopper_splash_attention.TuningConfig(
        block_q=64,
        block_kv=64,
        max_concurrent_steps=2,
        block_q_dkv=64,
        block_kv_dkv=64,
        block_q_dq=64,
        block_kv_dq=64,
    )

  def _generate_mask(self, mask_type: MaskType, batch_size: int, seq_len: int) -> jax.Array:
    def build_mask(
        *,
        position_ids: jax.Array,
        segment_ids: jax.Array,
        kv_token_mask: jax.Array,
        kv_position_ids: jax.Array,
        kv_segment_ids: jax.Array,
        causal: bool,
        sliding_window_length: int | None = None,
    ) -> jax.Array:
      mask = jnp.ones((1, 1, 1, 1), dtype=bool)
      if causal:
        position_ids_ = position_ids[:, None, :, None]
        kv_position_ids_ = kv_position_ids[:, None, None, :]
        causal_mask = position_ids_ >= kv_position_ids_
        mask = jnp.logical_and(mask, causal_mask)

      if sliding_window_length:
        if not causal:
          raise NotImplementedError("Sliding window mask must also be causal")
        position_ids_ = position_ids[:, None, :, None]
        kv_position_ids_ = kv_position_ids[:, None, None, :]
        local_window_mask = jnp.abs(
            position_ids_ - kv_position_ids_) < sliding_window_length
        mask = jnp.logical_and(mask, local_window_mask)

      segment_ids = segment_ids[:, None, :, None]
      kv_segment_ids = kv_segment_ids[:, None, None, :]
      document_mask = segment_ids == kv_segment_ids
      mask = jnp.logical_and(mask, document_mask)

      kv_token_mask = kv_token_mask[:, None, None, :]
      mask = jnp.logical_and(mask, kv_token_mask)
      return mask

    if mask_type == "full":
      mask = jnp.ones((batch_size, 1, seq_len, seq_len), dtype=jnp.bool_)
    elif mask_type == "zeros":
      mask = jnp.zeros((batch_size, 1, seq_len, seq_len), dtype=jnp.bool_)
    elif mask_type == "random":
      mask = jax.random.bernoulli(jax.random.key(
          0), 128.0 / seq_len, shape=(batch_size, 1, seq_len, seq_len))
    elif mask_type == "causal":
      position_ids = jnp.arange(seq_len).reshape((1, seq_len))
      kv_position_ids = jnp.arange(seq_len).reshape((1, seq_len))
      segment_ids = jnp.zeros((1, seq_len), dtype=jnp.int32)
      kv_token_mask = jnp.ones((1, seq_len), dtype=jnp.bool_)
      mask = build_mask(
          position_ids=position_ids,
          segment_ids=segment_ids,
          kv_token_mask=kv_token_mask,
          kv_position_ids=kv_position_ids,
          kv_segment_ids=segment_ids,
          causal=True,
      )
      mask = jnp.tile(mask, (batch_size, 1, 1, 1))
    elif mask_type == "multidoc":
      masks = []
      for batch_idx in range(batch_size):
        position_ids = jnp.arange(seq_len).reshape((1, seq_len))
        kv_position_ids = jnp.arange(seq_len).reshape((1, seq_len))
        num_docs = 2 ** (batch_idx + 1)
        segment_ids = np.repeat(jnp.arange(num_docs), seq_len // num_docs)
        segment_ids = jnp.reshape(segment_ids, (1, seq_len))
        kv_token_mask = jnp.ones((1, seq_len), dtype=jnp.bool_)
        mask = build_mask(
            position_ids=position_ids,
            segment_ids=segment_ids,
            kv_token_mask=kv_token_mask,
            kv_position_ids=kv_position_ids,
            kv_segment_ids=segment_ids,
            causal=True,
        )
        mask = jnp.tile(mask, (batch_size, 1, 1, 1))
        masks.append(mask)
      mask = jnp.concatenate(masks, axis=0)
    elif mask_type == "sliding_window":
      # Create sliding window mask
      masks = []
      for batch_idx in range(batch_size):
        position_ids = jnp.arange(seq_len).reshape((1, seq_len))
        kv_position_ids = position_ids
        segment_ids = jnp.zeros((1, seq_len), dtype=jnp.int32)
        kv_token_mask = jnp.ones((1, seq_len), dtype=jnp.bool_)

        mask = build_mask(
            position_ids=position_ids,
            segment_ids=segment_ids,
            kv_token_mask=kv_token_mask,
            kv_position_ids=kv_position_ids,
            kv_segment_ids=segment_ids,
            causal=True,
        )
        position_ids_ = position_ids[:, None, :, None]
        kv_position_ids_ = kv_position_ids[:, None, None, :]
        sliding_window_mask = jnp.abs(
            position_ids_ - kv_position_ids_) < (2 ** (6 + batch_idx))
        mask = jnp.logical_and(mask, sliding_window_mask)
        masks.append(mask)
      mask = jnp.concatenate(masks, axis=0)
    elif mask_type == "zeroed_partial_blocks":
      mask = jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_)
      mask = mask.at[:, :, 0:96, 0:96].set(False)
      mask = jnp.tile(mask, (batch_size, 1, 1, 1))
    elif mask_type == "realistic":
      # Generate a "realistic" mask with multiple documents of varying sizes that are also "masked" with padding tokens
      # at the end of each document.
      masks = []
      temp = 10.0
      for batch in range(batch_size):
        num_docs = 2 * (batch + 2)
        probs_key, sample_key, pad_key = jax.random.split(
            jax.random.key(batch), 3)
        position_ids = jnp.arange(seq_len).reshape((1, seq_len))
        kv_position_ids = jnp.arange(seq_len).reshape((1, seq_len))
        probs = jax.random.dirichlet(probs_key, jnp.ones((num_docs,)) * temp)
        doc_sizes = jax.random.multinomial(
            sample_key, seq_len, probs).astype(jnp.int32)
        segment_ids = jnp.concatenate([jnp.array([i] * int(n)) for (i, n) in enumerate(doc_sizes)]).reshape(
            (1, seq_len)
        )

        n_keep = [
            int(jax.random.binomial(jax.random.fold_in(pad_key, i), doc_sizes[i], 0.8)) for i in range(num_docs)
        ]
        kv_token_mask = jnp.concatenate(
            [
                jnp.concatenate([jnp.ones((n_keep[i],)), jnp.zeros(
                    (int(doc_sizes[i]) - n_keep[i],))])
                for i in range(num_docs)
            ]
        )
        kv_token_mask = kv_token_mask.reshape((1, seq_len)).astype(jnp.bool_)
        mask = build_mask(
            position_ids=position_ids,
            segment_ids=segment_ids,
            kv_token_mask=kv_token_mask,
            kv_position_ids=kv_position_ids,
            kv_segment_ids=segment_ids,
            causal=True,
        )
        masks.append(mask)
      mask = jnp.concatenate(masks, axis=0)
    assert mask.shape == (batch_size, 1, seq_len, seq_len)
    return mask

  def _generate_qkv(self, shape, seed=0, q_heads_per_kv_head=1):
    q_rng, k_rng, v_rng = jax.random.split(jax.random.key(seed), 3)
    q_shape = shape
    if q_heads_per_kv_head != 1:
      q_shape = (shape[0], shape[1], shape[2] * q_heads_per_kv_head, shape[3])
    q = jax.random.normal(q_rng, q_shape, dtype=jnp.float32)
    k = jax.random.normal(k_rng, shape, dtype=jnp.float32)
    v = jax.random.normal(v_rng, shape, dtype=jnp.float32)
    # RMS norm
    q = q / jnp.sqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    k = k / jnp.sqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    return q, k, v

  @parameterized.product(
      batch_size=(1, 2),
      q_heads_per_kv_head=(1, 2),
      mask_type=("realistic", "full", "causal", "random",
                 "sliding_window", "zeroed_partial_blocks")
  )
  def test_mgpu_masked_fwd(self, batch_size: int, q_heads_per_kv_head: int, mask_type: MaskType):
    if not jtu.is_cuda_compute_capability_equal("9.0"):
      self.skipTest("Test only runs on H100 GPUs.")
    seq_len = 2 * 1024
    q, k, v = self._generate_qkv(
        (batch_size, seq_len, 1, 128), q_heads_per_kv_head=q_heads_per_kv_head)
    q = q.astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    v = v.astype(jnp.bfloat16)
    scale = 1.0 / np.sqrt(q.shape[-1])
    mask = self._generate_mask(mask_type, batch_size, seq_len)
    mask_info = attention_mask.process_dynamic_mask(
        mask=mask,
        block_shape=(self.tuning_config.block_q, self.tuning_config.block_kv),
        is_dkv=False,
        data_tiling=2,
    )

    ref_result, ref_lse = jax.nn.dot_product_attention(
        q.astype(jnp.float32),
        k.astype(jnp.float32),
        v.astype(jnp.float32),
        mask=mask,
        scale=scale,
        return_residual=True,
    )
    ref_lse = ref_lse.swapaxes(-1, -2)  # Swap from BTN -> BNT
    # MGPU scales lse by log2(e) but JAX does not.
    ref_lse *= math.log2(math.e)
    mgpu_result, (lse,) = hopper_splash_attention.attention(
        q, k, v, mask_info=mask_info, config=self.tuning_config, scale=scale, save_residuals=True
    )
    np.testing.assert_allclose(mgpu_result, ref_result, atol=1e-2)
    np.testing.assert_allclose(lse, ref_lse, atol=5e-4)

  @parameterized.product(
      batch_size=(1, 2),
      q_heads_per_kv_head=(1, 2),
      mask_type=("realistic", "full", "causal", "random",
                 "sliding_window", "zeroed_partial_blocks")
  )
  def test_mgpu_masked_bwd(self, batch_size: int, q_heads_per_kv_head: int, mask_type: MaskType):
    if not jtu.is_cuda_compute_capability_equal("9.0"):
      self.skipTest("Test only runs on H100 GPUs.")
    seq_len = 2 * 1024
    q, k, v = self._generate_qkv(
        (batch_size, seq_len, 1, 128), q_heads_per_kv_head=q_heads_per_kv_head)
    q = q.astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    v = v.astype(jnp.bfloat16)
    scale = 1.0 / np.sqrt(q.shape[-1])
    mask = self._generate_mask(mask_type, batch_size, seq_len)
    mask_info = attention_mask.process_dynamic_mask(
        mask=mask,
        block_shape=(self.tuning_config.block_q, self.tuning_config.block_kv),
        is_dkv=False,
        data_tiling=2,
    )
    mask_info_dkv = attention_mask.process_dynamic_mask(
        mask=mask,
        block_shape=(self.tuning_config.block_q, self.tuning_config.block_kv),
        is_dkv=True,
        data_tiling=2,
    )

    def run_mgpu(q, k, v):
      result = hopper_splash_attention.attention(
          q,
          k,
          v,
          mask_info=mask_info,
          mask_info_dkv=mask_info_dkv,
          config=self.tuning_config,
          scale=scale,
          save_residuals=False,
      )
      return jnp.sum(result)

    @jax.jit
    def run_reference(q, k, v):
      result = jax.nn.dot_product_attention(
          q.astype(jnp.float32),
          k.astype(jnp.float32),
          v.astype(jnp.float32),
          mask=mask,
          scale=scale,
          return_residual=False,
      )
      return jnp.sum(result)

    result_grad = jax.grad(run_mgpu, argnums=[0, 1, 2])(q, k, v)
    ref_grad = jax.grad(run_reference, argnums=[0, 1, 2])(q, k, v)

    np.testing.assert_allclose(result_grad[0], ref_grad[0], atol=2e-3)  # dq
    np.testing.assert_allclose(result_grad[1], ref_grad[1], atol=2e-3)  # dk
    if mask_type == "zeros":
      # For zeros mask, reference outputs 1 but the kernel outputs 0, so
      # we manually check against zero.
      np.testing.assert_array_equal(result_grad[2], 0)
    else:
      np.testing.assert_allclose(
          result_grad[2], ref_grad[2], atol=5e-2, rtol=5e-2)  # dv


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
