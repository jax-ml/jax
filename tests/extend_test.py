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
from jax._src import api
from jax._src import linear_util
from jax._src import prng
from jax._src import test_util as jtu
from jax._src import xla_bridge
from jax._src.interpreters import mlir

jax.config.parse_flags_with_absl()


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
    self.assertIs(jex.backend.backends, xla_bridge.backends)
    self.assertIs(jex.backend.backend_xla_version, xla_bridge.backend_xla_version)
    self.assertIs(jex.backend.clear_backends, api.clear_backends)
    self.assertIs(jex.backend.get_backend, xla_bridge.get_backend)
    self.assertIs(jex.backend.register_backend_factory, xla_bridge.register_backend_factory)
    self.assertIs(jex.core.array_types, abstract_arrays.array_types)
    self.assertIs(jex.linear_util.StoreException, linear_util.StoreException)
    self.assertIs(jex.linear_util.WrappedFun, linear_util.WrappedFun)
    self.assertIs(jex.linear_util.cache, linear_util.cache)
    self.assertIs(jex.linear_util.merge_linear_aux, linear_util.merge_linear_aux)
    self.assertIs(jex.linear_util.transformation, linear_util.transformation)
    self.assertIs(jex.linear_util.transformation_with_aux, linear_util.transformation_with_aux)
    # TODO(necula): revert this change once we deprecate the old wrap_init
    # self.assertIs(jex.linear_util.wrap_init, linear_util.wrap_init)


class RandomTest(jtu.JaxTestCase):

  def make_custom_impl(self, shape, seed=False, split=False, fold_in=False,
                       random_bits=False):
    assert not split and not fold_in and not random_bits  # not yet implemented
    def seed_rule(_):
      return jnp.ones(shape, dtype=jnp.dtype('uint32'))

    def no_rule(*args, **kwargs):
      assert False, 'unreachable'

    return jex.random.define_prng_impl(
        key_shape=shape, seed=seed_rule if seed else no_rule, split=no_rule,
        fold_in=no_rule, random_bits=no_rule)

  def test_key_make_with_custom_impl(self):
    impl = self.make_custom_impl(shape=(4, 2, 7), seed=True)
    k = jax.random.key(42, impl=impl)
    self.assertEqual(k.shape, ())
    self.assertEqual(impl, jax.random.key_impl(k))

  def test_key_wrap_with_custom_impl(self):
    shape = (4, 2, 7)
    impl = self.make_custom_impl(shape=shape)
    data = jnp.ones((3, *shape), dtype=jnp.dtype('uint32'))
    k = jax.random.wrap_key_data(data, impl=impl)
    self.assertEqual(k.shape, (3,))
    self.assertEqual(impl, jax.random.key_impl(k))

  def test_key_impl_is_spec(self):
    # this is counterpart to random_test.py:
    # KeyArrayTest.test_key_impl_builtin_is_string_name
    spec_ref = self.make_custom_impl(shape=(4, 2, 7), seed=True)
    key = jax.random.key(42, impl=spec_ref)
    spec = jax.random.key_impl(key)
    self.assertEqual(repr(spec), f"PRNGSpec({spec_ref._impl.name!r})")


class MlirRegisterLoweringTest(jtu.JaxTestCase):

  def test_unknown_platform_error(self):
    with self.assertRaisesRegex(
        NotImplementedError,
        "Registering an MLIR lowering rule for primitive .+ for an unknown "
        "platform foo. Known platforms are: .+."):
      mlir.register_lowering(prim=None, rule=None, platform="foo")


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
