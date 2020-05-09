# Copyright 2020 Google LLC
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

from jax import test_util as jtu
from jax.experimental.callback import (
    callback_transform, find_by_value, rewrite, FoundValue)
import jax.numpy as jnp
from jax import lax
from jax import jit

from jax.config import config
config.parse_flags_with_absl()

class CallbackTest(jtu.JaxTestCase):
  @parameterized.named_parameters(jtu.cases_from_list(
      {'testcase_name': '_value={}'.format(value), 'value': value}
      for value in [jnp.inf, jnp.nan]))
  def testFindByValueFound(self, value):
    def f(x):
      y = x ** 2
      z = 1 - y
      r = 1 / z
      return r * 0

    with self.assertRaises(FoundValue):
      find_by_value(f, value)(jnp.array([1.0, 2.0, 3.0]))

  @parameterized.named_parameters(jtu.cases_from_list(
      {'testcase_name': '_value={}'.format(value), 'value': value}
      for value in [jnp.inf, jnp.nan]))
  def testFindByValueFoundJIT(self, value):
    def f(x):
      @jit
      def g(x):
        y = x ** 2
        z = 1 - y
        r = 1 / z
        return r * 0
      return g(x)
    with self.assertRaises(FoundValue):
      find_by_value(f, value)(jnp.array([1.0, 2.0, 3.0]))

  @parameterized.named_parameters(jtu.cases_from_list(
      {'testcase_name': '_value={}'.format(value), 'value': value}
      for value in [jnp.inf, jnp.nan]))
  def testFindByValueNotFound(self, value):
    def f(x):
      y = x ** 2
      z = 1 - y
      return z

    find_by_value(f, value)(jnp.array([1.0, 2.0, 3.0]))

  def testRewrite(self):
    def f(x):
      return x * 2

    x = jnp.array([2.0, 4.0])
    self.assertAllClose(f(x), jnp.array([4.0, 8.0]), True)

    self.assertAllClose(
        rewrite(f, {lax.mul_p: lambda x, y: x + y})(x),
        jnp.array([4.0, 6.0]), True)

  def testRewriteJIT(self):
    def f(x):
      @jit
      def g(x):
        return x * 2
      return g(x)

    x = jnp.array([2.0, 4.0])
    self.assertAllClose(f(x), jnp.array([4.0, 8.0]), True)

    self.assertAllClose(
        rewrite(f, {lax.mul_p: lambda x, y: x + y})(x),
        jnp.array([4.0, 6.0]), True)

if __name__ == "__main__":
  absltest.main()
