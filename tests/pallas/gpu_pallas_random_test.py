# Copyright 2024 The JAX Authors.
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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random as jax_random
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np
from jax._src import shard_map

import jax.experimental.pallas.ops.gpu.random.xoshiro
import jax.experimental.pallas.ops.gpu.random.threefry

P = jax.sharding.PartitionSpec

jax.config.parse_flags_with_absl()

class XoshiroTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.test_device_matches(["gpu"]):
      self.skipTest("Need GPU devices")
    super().setUp()

  def test_deterministic(self):
    key = jax_random.key(42, impl="pallas_xoshiro128pp")
    out1 = jax_random.bits(key, shape=(512,), dtype=jnp.uint32)
    out2 = jax_random.bits(key, shape=(512,), dtype=jnp.uint32)
    np.testing.assert_array_equal(out1, out2)

  def test_large_shape_raises(self):
    key = jax_random.key(0, impl="pallas_xoshiro128pp")
    with self.assertRaisesRegex(ValueError, "too large"):
      jax_random.bits(key, shape=(2**32,), dtype=jnp.uint32)

  def test_different_keys_produce_different_output(self):
    key1 = jax_random.key(0, impl="pallas_xoshiro128pp")
    key2 = jax_random.key(1, impl="pallas_xoshiro128pp")
    out1 = jax_random.bits(key1, shape=(512,), dtype=jnp.uint32)
    out2 = jax_random.bits(key2, shape=(512,), dtype=jnp.uint32)
    with self.assertRaises(AssertionError):
        np.testing.assert_array_equal(out1, out2)

  def test_kernel_matches_python_reference(self):
    # Pure Python reference implementation of the kernel logic
    def _mix32_py(z):
        z = (int(z) + 1) & 0xFFFFFFFF
        z = ((z ^ (z >> 16)) * 0x85ebca6b) & 0xFFFFFFFF
        z = ((z ^ (z >> 13)) * 0xc2b2ae35) & 0xFFFFFFFF
        return z ^ (z >> 16)

    def _rotl32_py(x, k):
        x = int(x)
        return ((x << k) & 0xFFFFFFFF) | (x >> (32 - k))

    def get_expected_for_thread(global_idx, key_data, elements_per_thread=4):
        k0_mixed = _mix32_py(key_data[0])
        k1_mixed = (_mix32_py(key_data[1]) * 0x6C62272E) & 0xFFFFFFFF
        thread_state = global_idx ^ k0_mixed ^ k1_mixed

        weyl = 0x9E3779B9
        s0 = _mix32_py((thread_state + weyl) & 0xFFFFFFFF)
        s0 = 1 if s0 == 0 else s0
        s1 = _mix32_py((thread_state + weyl * 2) & 0xFFFFFFFF)
        s2 = _mix32_py((thread_state + weyl * 3) & 0xFFFFFFFF)
        s3 = _mix32_py((thread_state + weyl * 4) & 0xFFFFFFFF)

        expected = []
        for _ in range(elements_per_thread):
            result = (_rotl32_py((s0 + s3) & 0xFFFFFFFF, 7) + s0) & 0xFFFFFFFF
            expected.append(result)

            t = (s1 << 9) & 0xFFFFFFFF
            s2 ^= s0
            s3 ^= s1
            s1 ^= s2
            s0 ^= s3
            s2 ^= t
            s3 = _rotl32_py(s3, 11)
        return np.array(expected, dtype=np.uint32)

    key = jax_random.key(42, impl="pallas_xoshiro128pp")
    key_data = jax_random.key_data(key)

    # Generate 3000 elements to guarantee we spill over into Block 1
    # (Block 0 holds 256 threads * 8 elements = 2048 elements).
    out = jax_random.bits(key, shape=(3000,), dtype=jnp.uint32)

    expected_t0 = get_expected_for_thread(0, key_data, elements_per_thread=4)
    np.testing.assert_array_equal(out[0:4], expected_t0)

    expected_t256 = get_expected_for_thread(256, key_data, elements_per_thread=4)
    np.testing.assert_array_equal(out[2048:2052], expected_t256)

  @parameterized.parameters(
      ((256,),),
      ((4, 128),),
  )
  def test_generate_bits_64(self, shape):
    if not jax.config.jax_enable_x64:
      self.skipTest("Requires JAX_ENABLE_X64=1")

    key = jax_random.key(0, impl="pallas_xoshiro128pp")
    out = jax_random.bits(key, shape=shape, dtype=jnp.uint64)
    self.assertEqual(out.dtype, jnp.uint64)
    self.assertEqual(out.shape, shape)

  @parameterized.parameters(
      ((512, 512),),
      ((137, 275),),
      ((4, 512, 512),),
      ((34,),),
      ((),),
      ((0,),),
  )
  def test_generate_bits(self, shape):
    key_pl = jax_random.key(0, impl="pallas_xoshiro128pp")
    pl_gen = jax_random.bits(key_pl, shape=shape, dtype=jnp.uint32)
    self.assertEqual(pl_gen.shape, shape)
    self.assertEqual(pl_gen.dtype, jnp.uint32)

    if pl_gen.size > 1:
      unique_frac = float(len(jnp.unique(pl_gen))) / pl_gen.size
      self.assertGreater(unique_frac, 0.99)

  def test_split(self):
    key = jax_random.key(42, impl="pallas_xoshiro128pp")
    key1, key2 = jax_random.split(key)
    with self.assertRaises(AssertionError):
      np.testing.assert_array_equal(jax_random.key_data(key), jax_random.key_data(key1))
    with self.assertRaises(AssertionError):
      np.testing.assert_array_equal(jax_random.key_data(key), jax_random.key_data(key2))
    with self.assertRaises(AssertionError):
      np.testing.assert_array_equal(jax_random.key_data(key1), jax_random.key_data(key2))

  def test_foldin(self):
    key = jax_random.key(42, impl="pallas_xoshiro128pp")
    key_f0 = jax_random.fold_in(key, 0)
    key_f1 = jax_random.fold_in(key, 1)
    key_f2 = jax_random.fold_in(key, 2)

    keys = [key, key_f0, key_f1, key_f2]
    for i in range(len(keys)):
      for j in range(i + 1, len(keys)):
        with self.assertRaises(AssertionError):
          np.testing.assert_array_equal(jax_random.key_data(keys[i]), jax_random.key_data(keys[j]))

  def test_vmap(self):
    key = jax_random.key(42, impl="pallas_xoshiro128pp")
    keys = jax_random.split(key, 10)

    @jax.vmap
    def generate(k):
        return jax_random.bits(k, shape=(128,), dtype=jnp.uint32)

    vmapped_out = generate(keys)
    self.assertEqual(vmapped_out.shape, (10, 128))
    unique_frac = float(len(jnp.unique(vmapped_out))) / vmapped_out.size
    self.assertGreater(unique_frac, 0.99)

  @parameterized.parameters(
      ((512, 512),),
      ((137, 275),),
  )
  def test_generate_uniform(self, shape):
    key = jax_random.key(0, impl="pallas_xoshiro128pp")
    values = jax_random.uniform(key, shape=shape)
    if values.size > 0:
        self.assertGreaterEqual(jnp.min(values), 0.0)
        self.assertLessEqual(jnp.max(values), 1.0)

  @parameterized.parameters(
      ((256, 256),),
      ((35, 113),),
  )
  def test_xoshiro_sharded(self, shape):
    if jax.device_count() < 2:
      self.skipTest("Need at least 2 devices")

    num_devices = jax.device_count()
    partition = P("x")
    mesh = jax.make_mesh(
        (num_devices,),
        ("x",),
        axis_types=(jax.sharding.AxisType.Auto,),
    )
    sharding = jax.sharding.NamedSharding(mesh, partition)

    key_pallas = jax_random.split(
        jax_random.key(0, impl="pallas_xoshiro128pp"), num_devices)
    key_pallas = jax.device_put(key_pallas, sharding)

    generate = shard_map.shard_map(
        lambda x: jax_random.bits(x[0], shape=shape),
        mesh=mesh,
        in_specs=partition,
        out_specs=partition,
        check_rep=False,
    )
    pl_gen = generate(key_pallas)

    self.assertEqual(pl_gen.shape, shape)

    shard_outputs = jnp.reshape(pl_gen, (num_devices, -1))
    for i in range(num_devices):
        for j in range(i + 1, num_devices):
            with self.assertRaises(AssertionError):
                np.testing.assert_array_equal(shard_outputs[i], shard_outputs[j])


class ThreefryTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.test_device_matches(["gpu"]):
      self.skipTest("Need GPU devices")
    super().setUp()

  def test_deterministic(self):
    key = jax_random.key(42, impl="pallas_threefry2x32_gpu")
    out1 = jax_random.bits(key, shape=(512,), dtype=jnp.uint32)
    out2 = jax_random.bits(key, shape=(512,), dtype=jnp.uint32)
    np.testing.assert_array_equal(out1, out2)

  def test_large_shape_raises(self):
    key = jax_random.key(0, impl="pallas_threefry2x32_gpu")
    with self.assertRaisesRegex(ValueError, "too large"):
      jax_random.bits(key, shape=(2**32,), dtype=jnp.uint32)

  @parameterized.parameters(
      ((256,),),
      ((4, 128),),
  )
  def test_generate_bits_64(self, shape):
    if not jax.config.jax_enable_x64:
      self.skipTest("Requires JAX_ENABLE_X64=1")

    with jax.threefry_partitionable(True):
      key_jax = jax_random.key(0, impl="threefry2x32")
      jax_gen = jax_random.bits(key_jax, shape=shape, dtype=jnp.uint64)

      key_pl = jax_random.key(0, impl="pallas_threefry2x32_gpu")
      pl_gen = jax_random.bits(key_pl, shape=shape, dtype=jnp.uint64)

    self.assertEqual(pl_gen.dtype, jnp.uint64)
    self.assertEqual(pl_gen.shape, shape)
    np.testing.assert_array_equal(jax_gen, pl_gen)

  @parameterized.parameters(
      ((512, 512),),
      ((137, 275),),
      ((4, 512, 512),),
      ((34,),),
      ((),),
  )
  def test_pallas_matches_jax_threefry(self, shape):
    with jax.threefry_partitionable(True):
      key_jax = jax_random.key(42, impl="threefry2x32")
      jax_gen = jax_random.bits(key_jax, shape=shape)

      key_pl = jax_random.key(42, impl="pallas_threefry2x32_gpu")
      pl_gen = jax_random.bits(key_pl, shape=shape)

    np.testing.assert_array_equal(jax_gen, pl_gen)

  def test_split(self):
    key = jax_random.key(42, impl="pallas_threefry2x32_gpu")
    key1, key2 = jax_random.split(key)
    with self.assertRaises(AssertionError):
      np.testing.assert_array_equal(jax_random.key_data(key), jax_random.key_data(key1))
    with self.assertRaises(AssertionError):
      np.testing.assert_array_equal(jax_random.key_data(key), jax_random.key_data(key2))
    with self.assertRaises(AssertionError):
      np.testing.assert_array_equal(jax_random.key_data(key1), jax_random.key_data(key2))

  def test_foldin(self):
    key = jax_random.key(42, impl="pallas_threefry2x32_gpu")
    key_f0 = jax_random.fold_in(key, 0)
    key_f1 = jax_random.fold_in(key, 1)
    key_f2 = jax_random.fold_in(key, 2)

    keys = [key, key_f0, key_f1, key_f2]
    for i in range(len(keys)):
      for j in range(i + 1, len(keys)):
        with self.assertRaises(AssertionError):
          np.testing.assert_array_equal(jax_random.key_data(keys[i]), jax_random.key_data(keys[j]))

  def test_vmap(self):
    key = jax_random.key(42, impl="pallas_threefry2x32_gpu")
    keys = jax_random.split(key, 10)

    @jax.vmap
    def generate(k):
        return jax_random.bits(k, shape=(128,), dtype=jnp.uint32)

    vmapped_out = generate(keys)
    self.assertEqual(vmapped_out.shape, (10, 128))
    unique_frac = float(len(jnp.unique(vmapped_out))) / vmapped_out.size
    self.assertGreater(unique_frac, 0.99)

  @parameterized.parameters(
      ((512, 512),),
      ((137, 275),),
  )
  def test_generate_uniform(self, shape):
    key = jax_random.key(0, impl="pallas_threefry2x32_gpu")
    values = jax_random.uniform(key, shape=shape)
    self.assertGreaterEqual(jnp.min(values), 0.0)
    self.assertLessEqual(jnp.max(values), 1.0)

  @parameterized.parameters(
      ((256, 256),),
      ((35, 113),),
      ((331,),),
  )
  def test_threefry_kernel_matches_jax_threefry_sharded(self, shape):
    if jax.device_count() < 2:
      self.skipTest("Need at least 2 devices")
    num_devices = jax.device_count()
    partition = P("x")
    mesh = jax.make_mesh(
        (num_devices,),
        ("x",),
        axis_types=(jax.sharding.AxisType.Auto,),
    )
    sharding = jax.sharding.NamedSharding(mesh, partition)

    with jax.threefry_partitionable(True):
      key_jax = jax_random.split(
          jax_random.key(0, impl="threefry2x32"), num_devices)
      key_pallas = jax_random.split(
          jax_random.key(0, impl="pallas_threefry2x32_gpu"), num_devices)

      key_jax = jax.device_put(key_jax, sharding)
      key_pallas = jax.device_put(key_pallas, sharding)

      generate = shard_map.shard_map(
          lambda x: jax_random.bits(x[0], shape=shape),
          mesh=mesh,
          in_specs=partition,
          out_specs=partition,
          check_rep=False,
      )
      jax_gen = generate(key_jax)
      pl_gen = generate(key_pallas)

    np.testing.assert_array_equal(jax_gen, pl_gen)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
