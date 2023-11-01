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
import jax.numpy as jnp

from jax._src import abstract_arrays
from jax._src import linear_util
from jax._src import prng
from jax._src import test_util as jtu

from jax import config
config.parse_flags_with_absl()


class ExtendTest(jtu.JaxTestCase):

  def test_symbols(self):
    # Assume these are tested in random_test.py, only check equivalence
    self.assertIs(jex.random.seed_with_impl, prng.seed_with_impl)
    self.assertIs(jex.random.threefry2x32_p, prng.threefry2x32_p)
    self.assertIs(jex.random.threefry_2x32, prng.threefry_2x32)
    self.assertIs(jex.random.threefry_prng_impl, prng.threefry_prng_impl)
    self.assertIs(jex.random.rbg_prng_impl, prng.rbg_prng_impl)
    self.assertIs(jex.random.unsafe_rbg_prng_impl, prng.unsafe_rbg_prng_impl)

    # Assume these are tested elsewhere, only check equivalence
    self.assertIs(jex.core.array_types, abstract_arrays.array_types)
    self.assertIs(jex.linear_util.StoreException, linear_util.StoreException)
    self.assertIs(jex.linear_util.WrappedFun, linear_util.WrappedFun)
    self.assertIs(jex.linear_util.cache, linear_util.cache)
    self.assertIs(jex.linear_util.merge_linear_aux, linear_util.merge_linear_aux)
    self.assertIs(jex.linear_util.transformation, linear_util.transformation)
    self.assertIs(jex.linear_util.transformation_with_aux, linear_util.transformation_with_aux)
    self.assertIs(jex.linear_util.wrap_init, linear_util.wrap_init)


class RandomTest(jtu.JaxTestCase):

  def test_key_make_with_custom_impl(self):
    shape = (4, 2, 7)

    def seed_rule(_):
      return jnp.ones(shape, dtype=jnp.dtype('uint32'))

    def no_rule(*args, **kwargs):
      assert False, 'unreachable'

    impl = jex.random.define_prng_impl(
        key_shape=shape, seed=seed_rule, split=no_rule, fold_in=no_rule,
        random_bits=no_rule)
    k = jax.random.key(42, impl=impl)
    self.assertEqual(k.shape, ())
    self.assertEqual(impl, jax.random.key_impl(k))

  def test_key_wrap_with_custom_impl(self):
    def no_rule(*args, **kwargs):
      assert False, 'unreachable'

    shape = (4, 2, 7)
    impl = jex.random.define_prng_impl(
        key_shape=shape, seed=no_rule, split=no_rule, fold_in=no_rule,
        random_bits=no_rule)
    data = jnp.ones((3, *shape), dtype=jnp.dtype('uint32'))
    k = jax.random.wrap_key_data(data, impl=impl)
    self.assertEqual(k.shape, (3,))
    self.assertEqual(impl, jax.random.key_impl(k))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
