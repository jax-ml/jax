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
"""Tests for random ops in Pallas + Mosaic."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random as jax_random
from jax._src import test_util as jtu
from jax._src.pallas.mosaic import random as plrandom
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.random import threefry  # pylint: disable=unused-import  # noqa: F401
import jax.numpy as jnp
import numpy as np

P = jax.sharding.PartitionSpec


jax.config.parse_flags_with_absl()


class PRNGTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Need TPU devices")
    super().setUp()

  def test_to_pallas_key_under_vmap(self):
    key = jax.random.key(42, impl="rbg")
    key = jax.random.split(key, 10)
    batched_key = plrandom.to_pallas_key(key)
    batched_key_data = jax.random.key_data(batched_key)
    vmapped_key = jax.vmap(plrandom.to_pallas_key)(key)
    vmapped_key_data = jax.random.key_data(vmapped_key)
    np.testing.assert_array_equal(batched_key_data, vmapped_key_data)

  def test_pallas_key_raise_not_implemented_outside_of_kernel(self):
    key = jax_random.key(0, impl="rbg")
    pallas_key = plrandom.to_pallas_key(key)
    # Using a pallas key outside of a kernel should raise an error when
    # trying to lower TPU-specific ops to XLA.
    # TODO(justinfu): Make this error more specific to pallas PRNG usage.
    with self.assertRaisesRegex(NotImplementedError,
                                "MLIR translation rule .* not found"):
      jax.random.uniform(
          pallas_key, shape=(1,), minval=0.0, maxval=1.0)

  def test_seeded_reproducibility(self):
    # Test whether generating random bits with the same seed
    # produces the same result (and different seeds produce
    # different results).
    def seeded_body(seed: int):
      def body(o_ref):
        pltpu.prng_seed(seed)
        o_ref[...] = pltpu.prng_random_bits(o_ref[...].shape)
      return body

    out = jax.ShapeDtypeStruct((8, 128), jnp.int32)
    result_1a = pl.pallas_call(seeded_body(0), out_shape=out)()
    result_1b = pl.pallas_call(seeded_body(0), out_shape=out)()
    result_2 = pl.pallas_call(seeded_body(1), out_shape=out)()
    with self.subTest("same_seed_same_result"):
      np.testing.assert_array_equal(result_1a, result_1b)
    with self.subTest("diff_seed_diff_result"):
      np.testing.assert_array_compare(np.not_equal, result_1a, result_2)

  @parameterized.parameters(
      ((32, 256),),
      ((8, 16),),
  )
  def test_prng_non_vreg_shape_output(self, shape):
    # Tests that RNG generation works with output shapes
    # not equal to a native-sized VREG.
    # This test makes sure that vector layout tiling
    # is implemented correctly.
    def body(o_ref):
      pltpu.prng_seed(0)
      samples = pltpu.prng_random_bits(o_ref[...].shape)
      o_ref[...] = samples

    o_shape = jax.ShapeDtypeStruct(shape, jnp.int32)
    result = pl.pallas_call(body, out_shape=o_shape)()
    # Check that random_bits generates (mostly) unique values.
    unique_frac = float(len(jnp.unique(result))) / np.prod(shape)
    self.assertGreater(unique_frac, 0.99)
    self.assertLessEqual(jnp.max(result), np.iinfo(jnp.int32).max)
    self.assertGreaterEqual(jnp.min(result), np.iinfo(jnp.int32).min)

  def test_stateful_uniform_sample(self):
    # Test stateful RNG using the jax.random API wrappers.
    def body(key_ref, o_ref):
      plrandom.set_seed(key_ref[...])
      o_ref[...] = plrandom.uniform(
          shape=o_ref[...].shape, minval=0.0, maxval=1.0)

    rbg_key = jax_random.key(0, impl="rbg")
    key = plrandom.to_pallas_key(rbg_key)
    o_shape = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    result = pl.pallas_call(
        body,
        in_specs=[pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.SMEM)],
        out_shape=o_shape,
    )(key)
    self.assertGreaterEqual(jnp.min(result), 0)
    self.assertLessEqual(jnp.max(result), 1.0)

  def test_stateless_uniform_sample(self):
    # Test keyed RNG using the jax.random API.
    def body(key_ref, o_ref):
      o_ref[...] = jax_random.uniform(
          key_ref[...], shape=o_ref[...].shape, minval=0.0, maxval=1.0
      )

    rbg_key = jax_random.key(0, impl="rbg")
    key = plrandom.to_pallas_key(rbg_key)
    o_shape = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    result = pl.pallas_call(
        body,
        in_specs=[pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.SMEM)],
        out_shape=o_shape,
    )(key)
    self.assertGreaterEqual(jnp.min(result), 0)
    self.assertLessEqual(jnp.max(result), 1.0)

  def test_key_data(self):
    def body(key_ref, o_ref):
      o_ref[...] = jax.random.key_data(key_ref[...])
    rbg_key = jax_random.key(0, impl="rbg")
    key = plrandom.to_pallas_key(rbg_key)
    expected_key_data = jax.random.key_data(key)
    o_shape = jax.ShapeDtypeStruct(expected_key_data.shape,
                                   expected_key_data.dtype)
    result = pl.pallas_call(
        body,
        in_specs=[pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.SMEM)],
        out_shape=o_shape,
    )(key)
    self.assertEqual(result, expected_key_data)

  def test_fold_in(self):
    # Test that folding in a value results in different random numbers.
    def body(key_ref, o_ref):
      key = key_ref[...]
      o_ref[0, ...] = jax_random.uniform(
          key, shape=o_ref[0, ...].shape, minval=0.0, maxval=1.0
      )

      key = jax_random.fold_in(key, 2)
      o_ref[1, ...] = jax_random.uniform(
          key, shape=o_ref[1, ...].shape, minval=0.0, maxval=1.0
      )

    rbg_key = jax_random.key(0, impl="rbg")
    key = plrandom.to_pallas_key(rbg_key)
    o_shape = jax.ShapeDtypeStruct((2, 8, 128), jnp.float32)
    result = pl.pallas_call(
        body,
        in_specs=[pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.SMEM)],
        out_shape=o_shape,
    )(key)
    result_a = result[0]
    result_b = result[1]
    np.testing.assert_array_compare(np.not_equal, result_a, result_b)


class BlockInvarianceTest(parameterized.TestCase):

  def setUp(self):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Need TPU devices")
    super().setUp()

  def test_block_invariance(self):

    def make_kernel_body(index_map):
      def body(key_ref, o_ref):
        key = key_ref[...]
        samples = plrandom.sample_block(
            jax.random.uniform,
            key,
            block_size=o_ref[...].shape,
            tile_size=(16, 128),
            total_size=(64, 512),
            block_index=index_map(pl.program_id(0), pl.program_id(1)),
            minval=0.0,
            maxval=1.0)
        o_ref[...] = samples
      return body

    global_key = jax_random.key(0, impl="pallas_tpu")
    o_shape = jnp.ones((64, 512), dtype=jnp.float32)
    key_spec = pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.SMEM)
    out_spec = pl.BlockSpec((16, 128), lambda i, j: (i, j))
    result_16x128 = pl.pallas_call(
        make_kernel_body(index_map=lambda i, j: (i, j)),
        out_shape=o_shape,
        in_specs=[key_spec],
        out_specs=out_spec,
        grid=(4, 4),
    )(global_key)

    out_spec = pl.BlockSpec((32, 256), lambda i, j: (j, i))
    result_32x256 = pl.pallas_call(
        make_kernel_body(index_map=lambda i, j: (j, i)),
        in_specs=[key_spec],
        out_shape=o_shape,
        out_specs=out_spec,
        grid=(2, 2),
    )(global_key)
    np.testing.assert_array_equal(result_16x128, result_32x256)


class ThreefryTest(parameterized.TestCase):

  def setUp(self):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Need TPU devices")
    super().setUp()

  @parameterized.parameters(
      ((8, 128),),
      ((32, 256),),
      ((4, 16, 128),),
  )
  def test_uniform_matches_jax_threefry(self, shape):
    def body(key_ref, o_ref):
      key = jax.random.wrap_key_data(key_ref[0, ...], impl='threefry2x32')
      o_ref[...] = jax_random.uniform(
          key, shape=o_ref[...].shape, minval=0.0, maxval=1.0
      )

    threefry_key = jax_random.key(0, impl="threefry2x32").reshape((1,))
    o_shape = jax.ShapeDtypeStruct(shape, jnp.float32)
    with jax.threefry_partitionable(True):
      # TODO(justinfu): support passing keys into VMEM.
      result = pl.pallas_call(
          body,
          in_specs=[pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM)],
          out_shape=o_shape,
      )(jax.random.key_data(threefry_key))
      jax_result = jax_random.uniform(
          threefry_key[0], shape=o_shape.shape, minval=0.0, maxval=1.0
      )
    np.testing.assert_array_equal(result, jax_result)

  @parameterized.parameters(
      ((512, 512),),
      ((137, 275),),  # Non block-aligned shape
      ((4, 512, 512),),  # Greater than 2D shape
      ((34,),),  # 1D
      (tuple(),),  # 0D
  )
  def test_threefry_kernel_matches_jax_threefry(self, shape):
    with jax.threefry_partitionable(True):
      key_jax = jax_random.key(0, impl="threefry2x32")
      jax_gen = jax_random.bits(key_jax, shape=shape)
      key_pl = jax_random.key(0, impl="pallas_threefry2x32")
      pl_gen = jax_random.bits(key_pl, shape=shape)

    np.testing.assert_array_equal(jax_gen, pl_gen)

  @parameterized.parameters(
      ((256, 256),),
      ((35, 113),),  # Non block-aligned shape
      ((331,),),  # 1D
  )
  def test_threefry_kernel_matches_jax_threefry_sharded(self, shape):
    if jax.device_count() < 2:
      self.skipTest("Need at least 2 devices")
    num_devices = jax.device_count()
    partition = P("x")
    mesh = jax.make_mesh((num_devices,), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, partition)

    with jax.threefry_partitionable(True):
      key_jax = jax_random.split(
          jax_random.key(0, impl="threefry2x32"), num_devices)
      key_pallas = jax_random.split(
          jax_random.key(0, impl="pallas_threefry2x32"), num_devices)
      key_jax = jax.device_put(key_jax, sharding)
      key_pallas = jax.device_put(key_pallas, sharding)
      generate = shard_map.shard_map(
          lambda x: jax_random.bits(x[0], shape=shape),
          mesh=mesh,
          in_specs=partition,
          out_specs=partition,
      )
      jax_gen = generate(key_jax)
      pl_gen = generate(key_pallas)

    np.testing.assert_array_equal(jax_gen, pl_gen)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
