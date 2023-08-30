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

from absl.testing import absltest

import jax
import jax.extend as jex

from jax._src import linear_util
from jax._src import prng
from jax._src import test_util as jtu

from jax import config
config.parse_flags_with_absl()


class ExtendTest(jtu.JaxTestCase):
  def test_symbols(self):
    # Assume these are tested in random_test.py, only check equivalence
    self.assertIs(jex.random.PRNGImpl, prng.PRNGImpl)
    self.assertIs(jex.random.seed_with_impl, prng.seed_with_impl)
    self.assertIs(jex.random.threefry2x32_p, prng.threefry2x32_p)
    self.assertIs(jex.random.threefry_2x32, prng.threefry_2x32)
    self.assertIs(jex.random.threefry_prng_impl, prng.threefry_prng_impl)
    self.assertIs(jex.random.rbg_prng_impl, prng.rbg_prng_impl)
    self.assertIs(jex.random.unsafe_rbg_prng_impl, prng.unsafe_rbg_prng_impl)

    # Assume these are tested elsewhere, only check equivalence
    self.assertIs(jex.linear_util.StoreException, linear_util.StoreException)
    self.assertIs(jex.linear_util.WrappedFun, linear_util.WrappedFun)
    self.assertIs(jex.linear_util.cache, linear_util.cache)
    self.assertIs(jex.linear_util.merge_linear_aux, linear_util.merge_linear_aux)
    self.assertIs(jex.linear_util.transformation, linear_util.transformation)
    self.assertIs(jex.linear_util.transformation_with_aux, linear_util.transformation_with_aux)
    self.assertIs(jex.linear_util.wrap_init, linear_util.wrap_init)


class RandomTest(jtu.JaxTestCase):
  def test_wrap_key_default(self):
    key1 = jax.random.key(17)
    data = jax.random.key_data(key1)
    key2 = jex.random.wrap_key_data(data)
    self.assertEqual(key1.dtype, key2.dtype)
    self.assertArraysEqual(jax.random.key_data(key1),
                           jax.random.key_data(key2))

    impl = config.jax_default_prng_impl
    key3 = jex.random.wrap_key_data(data, impl=impl)
    self.assertEqual(key1.dtype, key3.dtype)
    self.assertArraysEqual(jax.random.key_data(key1),
                           jax.random.key_data(key3))

  def test_wrap_key_explicit(self):
    key1 = jax.random.key(17, impl='rbg')
    data = jax.random.key_data(key1)
    key2 = jex.random.wrap_key_data(data, impl='rbg')
    self.assertEqual(key1.dtype, key2.dtype)
    self.assertArraysEqual(jax.random.key_data(key1),
                           jax.random.key_data(key2))

    key3 = jex.random.wrap_key_data(data, impl='unsafe_rbg')
    self.assertNotEqual(key1.dtype, key3.dtype)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
