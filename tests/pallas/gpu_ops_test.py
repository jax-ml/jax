# Copyright 2023 The JAX Authors.
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

import functools
import os
import sys

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax import random
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lax.control_flow.for_loop import for_loop
from jax._src.pallas.pallas_call import _trace_kernel_to_jaxpr
from jax.experimental import pallas as pl
if sys.platform != "win32":
  from jax.experimental.pallas.ops.gpu import attention
  from jax.experimental.pallas.ops.gpu import layer_norm
  from jax.experimental.pallas.ops.gpu import rms_norm
  from jax.experimental.pallas.ops.gpu import softmax
  BlockSizes = attention.BlockSizes
else:
  attention = None
  layer_norm = None
  rms_norm = None
  softmax = None
  BlockSizes = None
import jax.numpy as jnp
import numpy as np

# TODO(sharadmv): Update signatures of pallas_call to correct inputs/outputs.
# pylint: disable=no-value-for-parameter

config.parse_flags_with_absl()


@functools.partial(jax.jit, static_argnames=["bm", "bn", "gm", "bk",
                                             "interpret", "debug"])
def matmul(x, y, *, bm, bn, gm, bk, interpret, debug=False):
  m, n, k = x.shape[0], y.shape[1], x.shape[1]
  @functools.partial(
      pl.pallas_call, out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
      interpret=interpret,
      debug=debug,
      grid=pl.cdiv(m, bm) * pl.cdiv(n, bn))
  def matmul_kernel(x_ref, y_ref, o_ref):
    pid = pl.program_id(axis=0)
    num_pid_m = m // bm
    num_pid_n = n // bn
    num_pid_in_group = gm * num_pid_n
    group_id = lax.div(pid, num_pid_in_group)
    first_pid_m = group_id * gm
    group_size_m = jnp.minimum(num_pid_m - first_pid_m, gm)
    pid_m = first_pid_m + lax.rem(pid, group_size_m)
    pid_n = lax.div(lax.rem(pid, num_pid_in_group), group_size_m)
    idx_m = pid_m * bm + jnp.arange(bm)
    idx_n = pid_n * bn + jnp.arange(bn)
    idx_m = pl.max_contiguous(pl.multiple_of(idx_m, bm), bm)
    idx_n = pl.max_contiguous(pl.multiple_of(idx_n, bn), bn)
    acc = jnp.zeros((bm, bn), dtype=jnp.float32)
    def body(i, acc_ref):
      idx_k = i * bk + jnp.arange(bk)
      x_idx = (
          jax.lax.broadcast_in_dim(idx_m, (bm, bk), (0,)),
          jax.lax.broadcast_in_dim(idx_k, (bm, bk), (1,)))
      y_idx = (
          jax.lax.broadcast_in_dim(idx_k, (bk, bn), (0,)),
          jax.lax.broadcast_in_dim(idx_n, (bk, bn), (1,)))
      x_block, y_block = x_ref[x_idx], y_ref[y_idx]
      out = pl.dot(x_block, y_block)
      acc_ref[:, :] += out
    acc = for_loop(k // bk, body, acc).astype(o_ref.dtype)
    o_idx = (
        jax.lax.broadcast_in_dim(idx_m, (bm, bn), (0,)),
        jax.lax.broadcast_in_dim(idx_n, (bm, bn), (1,)),
        )
    o_ref[o_idx] = acc
  return matmul_kernel(x, y)


@functools.partial(jax.jit, static_argnames=["bm", "bn", "bk",
                                             "interpret", "debug"])
def matmul_block_spec(x, y, *, bm, bn, bk, interpret, debug=False):
  m, n, k = x.shape[0], y.shape[1], x.shape[1]
  @functools.partial(
      pl.pallas_call,
      out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
      interpret=interpret,
      debug=debug,
      in_specs=[
          pl.BlockSpec((bm, x.shape[1]), lambda i, _: (i, 0)),
          pl.BlockSpec((y.shape[0], bn), lambda _, j: (0, j)),
      ],
      out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
      grid=(pl.cdiv(m, bm), pl.cdiv(n, bn)),
  )
  def matmul_kernel(x_ref, y_ref, o_ref):
    acc = jnp.zeros(o_ref.shape, dtype=jnp.float32)
    def body(i, acc_ref):
      x_block = pl.load(x_ref, (slice(None), pl.ds(i * bk, bk)))
      y_block = pl.load(y_ref, (pl.ds(i * bk, bk), slice(None)))
      acc_ref[:, :] += pl.dot(x_block, y_block)
    acc = for_loop(k // bk, body, acc).astype(o_ref.dtype)
    o_ref[:, :] = acc
  return matmul_kernel(x, y)


@jtu.with_config(jax_traceback_filtering="off")
class PallasBaseTest(jtu.JaxTestCase):
  INTERPRET = False

  def setUp(self):
    print("platform: ", sys.platform)
    if jtu.test_device_matches(["cpu"]) and not self.INTERPRET:
      self.skipTest("On CPU the test works only in interpret mode")
    if jtu.test_device_matches(["cpu", "gpu"]) and jax.config.x64_enabled:
      self.skipTest("On CPU and GPU the test works only in 32-bit")
    if (jtu.test_device_matches(["cuda"]) and
        not jtu.is_cuda_compute_capability_at_least("8.0")):
      self.skipTest("Only works on GPU with capability >= sm80")
    if sys.platform == "win32":
      self.skipTest("Only works on non-Windows platforms")

    super().setUp()
    _trace_kernel_to_jaxpr.cache_clear()

  def pallas_call(self, *args, **kwargs):
    return pl.pallas_call(*args, **kwargs, interpret=self.INTERPRET)


class FusedAttentionTest(PallasBaseTest):

  def setUp(self):
    super().setUp()
    if jtu.test_device_matches(["tpu"]):
      self.skipTest("Not intended for TPU")

  @jtu.sample_product(
      batch_size=(1, 2),
      seq_len=(128, 384),
      num_heads=(1, 2, 8),
      head_dim=(32, 64, 128),
      block_sizes=(
        BlockSizes(block_q=128,block_k=128),
        BlockSizes(block_q=64,block_k=64),
        BlockSizes(block_q=64,block_k=128),
      ),
      causal=(True, False),
      use_fwd=(True, False),
      use_segment_ids=(True, False),
  )
  def test_fused_attention_fwd(
      self,
      *,
      batch_size,
      seq_len,
      num_heads,
      head_dim,
      block_sizes,
      causal,
      use_fwd,
      use_segment_ids,
  ):
    k1, k2, k3 = random.split(random.key(0), 3)
    q = random.normal(
        k1, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16
    )
    k = random.normal(
        k2, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16
    )
    v = random.normal(
        k3, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16
    )
    if use_segment_ids:
      segment_ids_1 = jnp.zeros((batch_size, seq_len // 2), dtype=jnp.int32)
      segment_ids_2 = jnp.ones((batch_size, seq_len // 2), dtype=jnp.int32)
      segment_ids = jnp.concatenate((segment_ids_1, segment_ids_2), axis=-1)
    else:
      segment_ids = None

    if use_fwd:

      @jax.jit
      def impl(q, k, v):
        v, _ = jax.vjp(
            functools.partial(
                attention.mha,
                block_sizes=block_sizes,
                causal=causal,
                segment_ids=segment_ids,
                interpret=self.INTERPRET,
            ),
            q,
            k,
            v,
        )
        return v

    else:
      impl = functools.partial(
          attention.mha,
          block_sizes=block_sizes,
          causal=causal,
          segment_ids=segment_ids,
          interpret=self.INTERPRET,
      )
    o = impl(q, k, v)
    o_ref = attention.mha_reference(q, k, v, segment_ids, causal=causal)
    np.testing.assert_allclose(o, o_ref, atol=0.05)

  @jtu.sample_product(
      batch_size=(1, 2),
      seq_len=(128, 384),
      num_heads=(1, 2),
      head_dim=(32, 64, 128,),
      block_sizes=(
        BlockSizes(
          block_q=128,
          block_k=128,
          block_q_dkv=128,
          block_kv_dkv=128,
          block_q_dq=128,
          block_kv_dq=128,
        ),
        BlockSizes(
          block_q=128,
          block_k=128,
          block_q_dkv=64,
          block_kv_dkv=64,
          block_q_dq=64,
          block_kv_dq=64,
        ),
        BlockSizes(
          block_q=128,
          block_k=128,
          block_q_dkv=64,
          block_kv_dkv=128,
          block_q_dq=128,
          block_kv_dq=64,
        ),
      ),
      causal=(True, False),
      use_segment_ids=(True, False),
  )
  def test_fused_attention_bwd(
      self,
      *,
      batch_size,
      seq_len,
      num_heads,
      head_dim,
      block_sizes,
      causal,
      use_segment_ids,
  ):
    k1, k2, k3 = random.split(random.key(0), 3)
    q = random.normal(
        k1, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16
    )
    k = random.normal(
        k2, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16
    )
    v = random.normal(
        k3, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16
    )
    if use_segment_ids:
      segment_ids_1 = jnp.zeros((batch_size, seq_len // 2), dtype=jnp.int32)
      segment_ids_2 = jnp.ones((batch_size, seq_len // 2), dtype=jnp.int32)
      segment_ids = jnp.concatenate((segment_ids_1, segment_ids_2), axis=-1)
    else:
      segment_ids = None

    def f(q, k, v):
      return attention.mha(
          q, k, v,
          block_sizes=block_sizes,
          causal=causal,
          segment_ids=segment_ids,
          interpret=self.INTERPRET).sum()

    def f_ref(q, k, v):
      return attention.mha_reference(q, k, v, segment_ids, causal=causal).sum()

    dq, dk, dv = jax.grad(f, argnums=(0, 1, 2))(q, k, v)
    dq_ref, dk_ref, dv_ref = jax.grad(f_ref, argnums=(0, 1, 2))(q, k, v)
    # TODO(sharadmv): Fix test.
    self.assertAllClose(dq, dq_ref, atol=5e-2)
    self.assertAllClose(dk, dk_ref, atol=5e-2)
    self.assertAllClose(dv, dv_ref, atol=1e-3)


class FusedAttentionInterpretTest(FusedAttentionTest):
  INTERPRET = True


class FusedLayerNormTest(PallasBaseTest):

  def setUp(self):
    super().setUp()
    if jtu.test_device_matches(["cpu", "tpu"]):
      self.skipTest("Works only on GPU")

  @parameterized.parameters(*[
    (1, 384, 192),
    (2, 384, 192),
  ])
  def test_fused_layernorm_fwd(self, batch_size, seq_len, embed_dim):
    k1, k2, k3 = random.split(random.key(0), 3)
    x = random.normal(k1, (batch_size, seq_len, embed_dim), dtype=jnp.float32)
    w = jax.random.normal(k2, (embed_dim,), dtype=jnp.float32)
    b = jax.random.normal(k3, (embed_dim,), dtype=jnp.float32)

    o = layer_norm.layer_norm(x, w, b)
    o_ref = layer_norm.layer_norm_reference(x, w, b)
    np.testing.assert_allclose(o, o_ref, atol=1e-5)

  @parameterized.parameters(*[
    (1, 384, 192),
    (2, 384, 192),
  ])
  def test_fused_layernorm_bwd(self, batch_size, seq_len, embed_dim):
    k1, k2, k3 = random.split(random.key(0), 3)
    x = random.normal(k1, (batch_size, seq_len, embed_dim), dtype=jnp.float32)
    w = jax.random.normal(k2, (embed_dim,), dtype=jnp.float32)
    b = jax.random.normal(k3, (embed_dim,), dtype=jnp.float32)

    def f(x, w, b):
      return layer_norm.layer_norm(x, w, b).sum()

    def f_ref(x, w, b):
      return layer_norm.layer_norm_reference(x, w, b).sum()

    dx, dw, db = jax.grad(f, argnums=(0, 1, 2))(x, w, b)
    dx_ref, dw_ref, db_ref = jax.grad(f_ref, argnums=(0, 1, 2))(x, w, b)
    np.testing.assert_allclose(dx, dx_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dw, dw_ref, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(db, db_ref, rtol=1e-2, atol=1e-2)


class FusedLayerNormInterpretTest(FusedLayerNormTest):
  INTERPRET = True


class RmsNormTest(PallasBaseTest):

  def setUp(self):
    super().setUp()
    if jtu.test_device_matches(["cpu", "tpu"]):
      self.skipTest("Works only on GPU")

  @parameterized.parameters(*[
    (1, 384, 192),
    (2, 384, 192),
  ])
  def test_rms_fwd(self, batch_size, seq_len, embed_dim):
    k1, k2, k3 = random.split(random.key(0), 3)
    x = random.normal(k1, (batch_size, seq_len, embed_dim), dtype=jnp.float32)
    w = jax.random.normal(k2, (embed_dim,), dtype=jnp.float32)
    b = jax.random.normal(k3, (embed_dim,), dtype=jnp.float32)

    o = rms_norm.rms_norm(x, w, b)
    o_ref = rms_norm.rms_norm_reference(x, w, b)
    np.testing.assert_allclose(o, o_ref, atol=1e-5)

  @parameterized.parameters(*[
    (1, 384, 192),
    (2, 384, 192),
  ])
  def test_rms_norm_bwd(self, batch_size, seq_len, embed_dim):
    k1, k2, k3 = random.split(random.key(0), 3)
    x = random.normal(k1, (batch_size, seq_len, embed_dim), dtype=jnp.float32)
    w = jax.random.normal(k2, (embed_dim,), dtype=jnp.float32)
    b = jax.random.normal(k3, (embed_dim,), dtype=jnp.float32)

    def f(x, w, b):
      return rms_norm.rms_norm(x, w, b).sum()

    def f_ref(x, w, b):
      return rms_norm.rms_norm_reference(x, w, b).sum()

    dx, dw, db = jax.grad(f, argnums=(0, 1, 2))(x, w, b)
    dx_ref, dw_ref, db_ref = jax.grad(f_ref, argnums=(0, 1, 2))(x, w, b)
    np.testing.assert_allclose(dx, dx_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dw, dw_ref, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(db, db_ref, rtol=1e-2, atol=1e-2)


class RmsNormInterpretTest(RmsNormTest):
  INTERPRET = True


class SoftmaxTest(PallasBaseTest):

  def setUp(self):
    super().setUp()
    if jtu.test_device_matches(["cpu", "tpu"]):
      self.skipTest("Works only on GPU")

  @parameterized.product(
      shape=[(1024, 125), (4, 1024, 125)],
      dtype=[jnp.bfloat16, jnp.float16, jnp.float32]
  )
  def test_softmax(self, shape, dtype):
    x = jax.random.normal(random.key(0), shape, dtype=dtype)

    atol, rtol = {
        jnp.bfloat16: (1e-2, 1e-4),
        jnp.float16: (1e-2, 1e-4),
        jnp.float32: (1e-7, 1e-6),
    }[dtype]

    # We upcast to float32 because NumPy <2.0 does not handle custom dtypes
    # properly. See https://github.com/jax-ml/jax/issues/11014.
    np.testing.assert_allclose(
        softmax.softmax(x, axis=-1).astype(jnp.float32),
        jax.nn.softmax(x, axis=-1).astype(jnp.float32),
        atol=atol,
        rtol=rtol,
    )


class SoftmaxInterpretTest(SoftmaxTest):
  INTERPRET = True


if __name__ == "__main__":
  absltest.main()
