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


from absl.testing import absltest
import jax
from jax import numpy as jnp
from jax._src import cost_analysis
import jax._src.test_util as jtu


jax.config.parse_flags_with_absl()


class CostAnalysisTest(jtu.JaxTestCase):
  """Tests cost_analysis.py."""

  def setUp(self):
    self._bytes_per_word = 8 if jax.config.read('jax_enable_x64') else 4

  def test_matmul(self):
    f = lambda x, y: x @ y
    jaxpr = jax.make_jaxpr(f)(jnp.zeros((3, 10)), jnp.ones((10, 7)))
    self.assertEqual(cost_analysis.count_flops(jaxpr), 2 * 3 * 10 * 7)
    self.assertEqual(
        cost_analysis.count_bytes_accessed(jaxpr),
        self._bytes_per_word * (3 * 10 + 10 * 7 + 3 * 7),
    )

  def test_maximum(self):
    f = lambda x, y: jnp.maximum(x, y)
    jaxpr = jax.make_jaxpr(f)(jnp.zeros((11, 3)), jnp.ones((11, 3)))
    self.assertEqual(cost_analysis.count_flops(jaxpr), 2 * 11 * 3 * 3 + 11 * 3)
    self.assertEqual(
        cost_analysis.count_bytes_accessed(jaxpr),
        self._bytes_per_word * (11 * 3 + 11 * 3 + 5 * (11 * 3) + 1),
    )

  def test_add(self):
    f = lambda x, y: x + y
    jaxpr = jax.make_jaxpr(f)(jnp.zeros((11, 4)), jnp.ones((11, 4)))
    self.assertEqual(cost_analysis.count_flops(jaxpr), 3 * 11 * 4)
    self.assertEqual(
        cost_analysis.count_bytes_accessed(jaxpr),
        self._bytes_per_word * 3 * 11 * 4,
    )

  def test_mul(self):
    f = lambda x, y: x * y
    jaxpr = jax.make_jaxpr(f)(jnp.zeros((11, 5)), jnp.ones((11, 5)))
    self.assertEqual(cost_analysis.count_flops(jaxpr), 3 * 11 * 5)
    self.assertEqual(
        cost_analysis.count_bytes_accessed(jaxpr),
        self._bytes_per_word * 3 * 11 * 5,
    )

  def test_reduce_sum(self):
    f = lambda x: jnp.sum(x)
    jaxpr = jax.make_jaxpr(f)(jnp.zeros((11, 4)))
    self.assertEqual(cost_analysis.count_flops(jaxpr), 11 * 4 - 1)
    self.assertEqual(
        cost_analysis.count_bytes_accessed(jaxpr),
        self._bytes_per_word * (11 * 4 + 2),
    )

  def test_sin(self):
    f = lambda x: jnp.sin(x)
    jaxpr = jax.make_jaxpr(f)(jnp.zeros((11, 4)))
    self.assertEqual(cost_analysis.count_flops(jaxpr), 11 * 4)
    self.assertEqual(
        cost_analysis.count_bytes_accessed(jaxpr),
        2 * self._bytes_per_word * (11 * 4),
    )

  def test_nested(self):
    def f(x, y):
      @jax.jit
      def f(x):
        return x * y

      return f(x) + f(y)

    jaxpr = jax.make_jaxpr(f)(jnp.zeros((11, 4)), jnp.ones((11, 4)))
    self.assertEqual(cost_analysis.count_flops(jaxpr), 3 * 3 * 11 * 4)
    self.assertEqual(
        cost_analysis.count_bytes_accessed(jaxpr),
        3 * self._bytes_per_word * (3 * 11 * 4),
    )


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
