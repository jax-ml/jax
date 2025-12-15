# Copyright 2025 The JAX Authors.
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

"""Tests for partitioning splash_attention."""

import functools
import math
from absl.testing import absltest, parameterized
import jax
from jax import random
from jax._src import test_util as jtu
from jax._src.pallas import pallas_test_util as ptu
from jax._src.shard_map import shard_map
from jax.experimental.pallas.ops.tpu.splash_attention import (
    CausalMask,
    MultiHeadMask,
    SegmentIds,
    make_splash_mha,
)
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
import jax.numpy as jnp
from jax.sharding import PartitionSpec
import numpy as np

partial = functools.partial

jax.config.parse_flags_with_absl()


@jtu.with_config(jax_traceback_filtering="off")
class PallasBaseTest(ptu.PallasTPUTest):

  def setUp(self):
    super().setUp()
    if len(jax.devices()) < 4:
      self.skipTest("This test requires at least 4 devices.")

  def _assert_allclose(self, x, y, **kwargs):
    if x.dtype == np.dtype(jnp.bfloat16):
      x = x.astype(np.float32)
    if y.dtype == np.dtype(jnp.bfloat16):
      y = y.astype(np.float32)
    self.assertEqual(x.dtype, y.dtype)
    self.assertTupleEqual(x.shape, y.shape)
    np.testing.assert_allclose(x, y, **kwargs)


def generate_mask(shape, num_heads, seed) -> np.ndarray:
  assert num_heads >= 2
  assert shape > (64, 64)

  masks = [
      mask_lib.make_causal_mask(shape),
      mask_lib.make_local_attention_mask(shape, window_size=(64, 64)),
  ]
  masks += [mask_lib.make_random_mask(shape, 0.8, seed)] * (num_heads - 2)
  return np.stack(masks, axis=0)


class SplashAttentionShardingTest(PallasBaseTest):

  @parameterized.product(
      topology=[(1, 1), (2, 1), (2, 2), (1, 2), (1, 4), (4, 1)],
      num_heads=[2, 4, 16],
      dtype=[jnp.bfloat16],
      is_dynamic_mask=[False, True],
  )
  def test_dynamic_mask_manual_partitioning_mha(
      self, topology, num_heads, dtype, is_dynamic_mask
  ):
    k1, k2, k3 = random.split(random.key(0), 3)
    seq_len = 1024
    head_dim = 128

    head_shards, q_seq_shards = topology
    num_devices = math.prod(topology)

    if head_shards > num_heads:
      self.skipTest(
          f"This test requires {num_heads} heads, but has only"
          f" {head_shards} head shards available."
      )

    if len(jax.devices()) < num_devices:
      self.skipTest(
          f"This test requires {num_devices} devices, but has only"
          f" {len(jax.devices())} devices available."
      )

    q = random.uniform(k1, (num_heads, seq_len, head_dim), dtype=dtype)
    k = random.uniform(k2, (num_heads, seq_len, head_dim), dtype=dtype)
    v = random.uniform(k3, (num_heads, seq_len, head_dim), dtype=dtype)

    mask = generate_mask((seq_len, seq_len), num_heads, seed=0)
    if is_dynamic_mask:
      mask = jnp.array(mask)

    devices = np.asarray(jax.devices()[:num_devices]).reshape(
        head_shards, q_seq_shards
    )

    mesh = jax.sharding.Mesh(devices, ("heads", "q_seq"))
    q_spec = PartitionSpec(
        "heads" if head_shards > 1 else None,
        "q_seq" if q_seq_shards > 1 else None,
    )
    kv_spec = PartitionSpec("heads" if head_shards > 1 else None, None)
    kernel = splash.make_splash_mha(
        mask, head_shards=head_shards, q_seq_shards=q_seq_shards
    )
    kernel_spec = kernel.manual_sharding_spec(
        jax.sharding.NamedSharding(mesh, q_spec)
    )

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            kernel_spec,
            q_spec,
            kv_spec,
            kv_spec,
        ),
        out_specs=q_spec,
        check_vma=False,
    )
    def f(kernel, q, k, v):
      return kernel(q, k, v)

    out = f(kernel, q, k, v)
    out_ref = jax.vmap(splash.attention_reference)(mask, q, k, v, None)
    self._assert_allclose(out, out_ref, rtol=3e-3, atol=3e-3)

  @parameterized.product(
      topology=[(1, 1), (2, 1), (2, 2), (1, 2), (1, 4), (4, 1)],
      num_heads=[2, 4],
      dtype=[jnp.bfloat16],
      is_dynamic_mask=[False, True],
  )
  def test_dynamic_mask_manual_partitioning_mha_bwd(
      self, topology, num_heads, dtype, is_dynamic_mask
  ):
    assert num_heads % 2 == 0
    k1, k2, k3, k4 = random.split(random.key(0), 4)
    seq_len = 1024
    head_dim = 128

    head_shards, q_seq_shards = topology
    num_devices = math.prod(topology)

    if head_shards > num_heads:
      self.skipTest(
          f"This test requires {num_heads} heads, but has only"
          f" {head_shards} head shards available."
      )

    q = random.uniform(k1, (num_heads, seq_len, head_dim), dtype=dtype)
    k = random.uniform(k2, (num_heads, seq_len, head_dim), dtype=dtype)
    v = random.uniform(k3, (num_heads, seq_len, head_dim), dtype=dtype)

    mask = generate_mask((seq_len, seq_len), num_heads, seed=0)
    if is_dynamic_mask:
      mask = jnp.array(mask)

    devices = np.asarray(jax.devices()[:num_devices]).reshape(
        head_shards, q_seq_shards
    )

    mesh = jax.sharding.Mesh(devices, ("heads", "q_seq"))
    q_spec = PartitionSpec(
        "heads" if head_shards > 1 else None,
        "q_seq" if q_seq_shards > 1 else None,
    )
    kv_spec = PartitionSpec("heads" if head_shards > 1 else None, None)

    kernel = splash.make_splash_mha(
        mask, head_shards=head_shards, q_seq_shards=q_seq_shards
    )
    kernel_spec = kernel.manual_sharding_spec(
        jax.sharding.NamedSharding(mesh, q_spec)
    )

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            kernel_spec,
            q_spec,
            kv_spec,
            kv_spec,
        ),
        out_specs=q_spec,
        check_vma=False,
    )
    def f(kernel, q, k, v):
      return kernel(q, k, v)

    f_ref = jax.vmap(splash.attention_reference)

    out, out_vjp = jax.vjp(f, kernel, q, k, v)
    out_ref, out_vjp_ref = jax.vjp(f_ref, mask, q, k, v, None)
    self._assert_allclose(out, out_ref, rtol=3e-3, atol=3e-3)

    do = random.uniform(k4, out.shape, dtype=out.dtype)
    _, dq, dk, dv = out_vjp(do)
    _, dq_ref, dk_ref, dv_ref, _ = out_vjp_ref(do.astype(jnp.float32))

    self.assertAllClose(dq, dq_ref, atol=5e-2)
    self.assertAllClose(dk, dk_ref, atol=5e-2)
    self.assertAllClose(dv, dv_ref, atol=5e-2)

  def test_splash_explicit_vmap_one_mesh_axis(self):
    mesh = jax.make_mesh((4,), ("dp",))

    NUM_HEADS = 4
    SEQ_LEN = 256
    HEAD_DIM = 64
    d_model = NUM_HEADS * HEAD_DIM

    key = jax.random.key(0)
    input_sharding = jax.NamedSharding(mesh, jax.P("dp", None, None))
    x_seq = jax.random.normal(key, (4, SEQ_LEN, d_model), dtype=jnp.bfloat16)
    x_seq = jax.device_put(x_seq, input_sharding)

    def make_splash_kernel_with_shard_map(mesh):
      mask = MultiHeadMask([CausalMask(shape=(SEQ_LEN, SEQ_LEN))
                            for _ in range(NUM_HEADS)])
      splash_spec = jax.P(None, None)
      sspec = jax.NamedSharding(mesh, splash_spec)

      kernel = make_splash_mha(mask, head_shards=1, q_seq_shards=1)
      kspec = kernel.manual_sharding_spec(sspec)

      @jax.shard_map(
          mesh=mesh,
          in_specs=(kspec, splash_spec, splash_spec, splash_spec, jax.P()),
          out_specs=splash_spec,
          check_vma=False,
      )
      def splash_sharded(kernel, q, k, v, segment_ids):
        return kernel(q, k, v, segment_ids=segment_ids)

      return splash_sharded, kernel

    def attention_fn_with_shmap(splash_sharded, kernel, x_seq):
      s = x_seq.shape[0]
      q, k, v = jnp.ones((3, NUM_HEADS, s, HEAD_DIM), out_sharding=jax.P())
      segment_ids = SegmentIds(q=jnp.zeros((s,)), kv=jnp.zeros((s,)))
      scale = HEAD_DIM ** -0.25
      out = splash_sharded(kernel, q * scale, k * scale, v, segment_ids)
      return out

    splash_sharded, kernel = make_splash_kernel_with_shard_map(mesh)

    @jax.jit
    def step(x_seq):
      def loss_fn(x_seq):
        attn_fn = partial(attention_fn_with_shmap, splash_sharded, kernel)
        out = jax.vmap(attn_fn)(x_seq)
        return out.sum()

      loss, grads = jax.value_and_grad(loss_fn)(x_seq)
      return loss, grads

    with jax.set_mesh(mesh):
      step(x_seq)  # doesn't crash


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
