# Copyright 2026 The JAX Authors.
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

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import random as exp_random
from jax._src import config
from jax._src import test_util as jtu


config.parse_flags_with_absl()

class StatefulRNGTest(jtu.JaxTestCase):

  def test_stateful_rng_instantiation(self, seed=547389):
    rng = exp_random.stateful_rng(seed)
    key = jax.random.key(seed)

    self.assertEqual(key, rng._base_key)
    self.assertEqual(rng._counter.shape, ())
    self.assertEqual(0, rng._counter[...])

  def test_stateful_rng_counter_increment(self, seed=7865943):
    rng = exp_random.stateful_rng(seed)
    original_key = rng._base_key
    self.assertEqual(0, rng._counter[...])

    _ = jax.jit(rng.key)()  # implicit update

    self.assertEqual(original_key, rng._base_key)  # base key does not change
    self.assertEqual(1, rng._counter[...])  # counter is incremented

  def test_stateful_rng_invalid_instantiation(self):
    valid_key = jax.random.key(0)
    valid_counter = jax.new_ref(0)
    invalid_key = jax.numpy.array([0, 1], dtype='uint32')
    invalid_counter = 0
    with self.assertRaisesRegex(ValueError, "Expected base_key to be a typed PRNG key"):
      exp_random.StatefulPRNG(invalid_key, valid_counter)
    with self.assertRaisesRegex(ValueError, "Expected counter to be a scalar integer ref"):
      exp_random.StatefulPRNG(valid_key, invalid_counter)

  def testRepeatedKeys(self, seed=578543):
    prng = exp_random.stateful_rng(seed)
    self.assertNotEqual(prng.key(), prng.key())

  def testShapedKeys(self, seed=7589432):
    prng = exp_random.stateful_rng(seed)

    keys1 = prng.key(10)
    self.assertEqual(keys1.shape, (10,))
    self.assertTrue(jax.dtypes.issubdtype(keys1.dtype, jax.dtypes.prng_key))

    keys2 = prng.key(10)
    self.assertEqual(keys1.shape, (10,))
    self.assertTrue(jax.dtypes.issubdtype(keys2.dtype, jax.dtypes.prng_key))

    self.assertFalse((keys1 == keys2).any())

  def testRepeatedDraws(self, seed=328090):
    prng = exp_random.stateful_rng(seed)
    vals1 = prng.uniform(size=10)
    vals2 = prng.uniform(size=10)
    self.assertTrue((vals1 != vals2).all())

  def testRepeatedDrawsJIT(self, seed=328090):
    prng = exp_random.stateful_rng(seed)
    @jax.jit
    def get_values(prng):
      return prng.uniform(size=10)
    vals1 = get_values(prng)
    vals2 = get_values(prng)
    self.assertTrue((vals1 != vals2).all())

  @jtu.sample_product(
      size=[None, 2, (5, 2)],
      dtype=jtu.dtypes.floating,
  )
  def testRandom(self, size, dtype):
    rng = exp_random.stateful_rng(578943)
    vals = rng.random(size, dtype)
    shape = np.broadcast_shapes(size or ())

    self.assertEqual(vals.shape, shape)
    self.assertEqual(vals.dtype, dtype)
    self.assertTrue((vals < 1).all())
    self.assertTrue((vals >= 0).all())

  @jtu.sample_product(
      low=[0, 1, np.array([0, 1])],
      high=[2, 3, np.array([2, 3])],
      size=[None, 2, (5, 2)],
      dtype=jtu.dtypes.floating,
  )
  @jax.numpy_dtype_promotion('standard')
  @jax.numpy_rank_promotion('allow')
  def testUniform(self, low, high, size, dtype):
    rng = exp_random.stateful_rng(473289)
    vals = rng.uniform(low, high, size, dtype=dtype)
    shape = np.broadcast_shapes(np.shape(low), np.shape(high), size or ())

    self.assertEqual(vals.shape, shape)
    self.assertEqual(vals.dtype, dtype)
    self.assertTrue((vals < high).all())
    self.assertTrue((vals >= low).all())

  @jtu.sample_product(
      loc=[0, 1, np.array([0, 1])],
      scale=[2, 3, np.array([2, 3])],
      size=[None, 2, (5, 2)],
      dtype=jtu.dtypes.floating,
  )
  @jax.numpy_dtype_promotion('standard')
  @jax.numpy_rank_promotion('allow')
  def testNormal(self, loc, scale, size, dtype):
    rng = exp_random.stateful_rng(473289)
    vals = rng.normal(loc, scale, size, dtype=dtype)
    shape = np.broadcast_shapes(np.shape(loc), np.shape(scale), size or ())

    self.assertEqual(vals.shape, shape)
    self.assertEqual(vals.dtype, dtype)

  @jtu.sample_product(
      low=[0, 1, np.array([0, 1])],
      high=[10, 15, np.array([10, 15])],
      size=[None, 2, (5, 2)],
      dtype=jtu.dtypes.integer,
  )
  @jax.numpy_dtype_promotion('standard')
  @jax.numpy_rank_promotion('allow')
  def testIntegers(self, low, high, size, dtype):
    rng = exp_random.stateful_rng(473289)
    vals = rng.integers(low, high, size, dtype=dtype)
    shape = np.broadcast_shapes(np.shape(low), np.shape(high), size or ())

    self.assertEqual(vals.shape, shape)
    self.assertEqual(vals.dtype, dtype)
    self.assertTrue((vals < high).all())
    self.assertTrue((vals >= low).all())

  def testSpawn(self):
    rng = exp_random.stateful_rng(758943)
    rngs = rng.spawn(4)

    for child_rng in rngs:
      self.assertNotEqual(rng._base_key, child_rng._base_key)
      self.assertEqual(0, child_rng._counter[...])

  @jtu.sample_product(shape=[4, (5,), (2, 3)])
  def testSplit(self, shape):
    rng = exp_random.stateful_rng(758943)
    rng_split = rng.split(shape)

    expected_shape = (shape,) if isinstance(shape, int) else shape

    self.assertEqual(rng_split._base_key.dtype, rng._base_key.dtype)
    self.assertEqual(rng_split._base_key.shape, expected_shape)

    self.assertIsInstance(rng_split._counter, jax.Ref)
    self.assertEqual(rng_split._counter.shape, expected_shape)

  def testVmapMapped(self):
    seed = 758943
    N = 4
    x = np.arange(N, dtype=float)
    def f(rng, x):
      return x + rng.uniform()

    rng = exp_random.stateful_rng(seed)
    expected = x + jnp.array([rng.uniform() for rng in rng.spawn(N)])

    rng = exp_random.stateful_rng(seed)
    actual = jax.vmap(f)(rng.split(N), x)

    self.assertArraysEqual(actual, expected)

  def testVmapUnmapped(self):
    seed = 758943
    x = np.arange(4, dtype=float)
    rng = exp_random.stateful_rng(seed)
    def f(rng, x):
      return x + rng.uniform()
    with self.assertRaisesRegex(Exception, "performing an addupdate operation with vmapped value"):
      jax.vmap(f, in_axes=(None, 0))(rng, x)

  def testScanClosure(self):
    seed = 432932
    def f1(seed):
      rng = exp_random.stateful_rng(seed)
      def scan_f(_, __):
        return None, rng.uniform()
      return jax.lax.scan(scan_f, None, length=10)[1]

    def f2(seed):
      rng = exp_random.stateful_rng(seed)
      return jax.numpy.array([rng.uniform() for i in range(10)])

    self.assertArraysAllClose(f1(seed), f2(seed))

  def testDefaultSeed(self):
    rng = exp_random.stateful_rng()
    x = rng.uniform(size=10)
    self.assertEqual(x.shape, (10,))

  def testDefaultSeedErrorUnderJIT(self):
    def f():
      return exp_random.stateful_rng().uniform(size=10)
    with self.assertRaisesRegex(TypeError, "When used within transformed code"):
      jax.jit(f)()

  def testDefaultSeedErrorUnderGrad(self):
    def f(x):
      return x + exp_random.stateful_rng().uniform()
    with self.assertRaisesRegex(TypeError, "When used within transformed code"):
      jax.grad(f)(1.0)

  def testDefaultSeedErrorUnderVmap(self):
    def f(x):
      return x + exp_random.stateful_rng().uniform()
    with self.assertRaisesRegex(TypeError, "When used within transformed code"):
      jax.vmap(f)(jnp.arange(5.0))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
