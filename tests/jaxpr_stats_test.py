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

from jax import jit, make_jaxpr, numpy as jnp, test_util as jtu
from jax.config import config
from jax.experimental import jaxpr_stats as js


config.parse_flags_with_absl()
FLAGS = config.FLAGS


class JaxprStatsTest(jtu.JaxTestCase):

  def test_primitives(self):
    def f(x, y):
      s = jit(jnp.sin)(x)
      return jnp.sin(s) + jnp.cos(y)

    hist = js.primitives(make_jaxpr(f)(1., 1.).jaxpr)

    for k in ['add', 'sin', 'cos', 'xla_call']:
      assert k in hist, k
    self.assertEqual(hist['sin'], 2)
    self.assertTrue(all(count == 1 for k, count in hist.items() if k != 'sin'))

  def test_primitives_by_source(self):
    def f(x, y):
      s = jnp.sin(x)
      return jnp.sin(s) + jnp.cos(y)

    hist = js.primitives_by_source(make_jaxpr(f)(1., 1.).jaxpr)

    sin_keys = [k for k in hist.keys() if k.startswith('sin @ ')]
    self.assertEqual(len(sin_keys), 2)
    self.assertTrue(all(count == 1 for count in hist.values()))

  def test_primitives_by_shape(self):
    def f(x, y):
      def sub(x, y):
        return jnp.sum(jnp.array([x, y])), y
      s, _ = jit(sub)(x, y)
      return jnp.sin(s) + jnp.cos(y)

    hist = js.primitives_by_shape(make_jaxpr(f)(1., 1.).jaxpr)

    shapes = [
        'add :: float32[]',
        'sin :: float32[]',
        'cos :: float32[]',
        'reduce_sum :: float32[]',
        'concatenate :: float32[2]',
        'xla_call :: float32[] *',
    ]
    for k in shapes:
      self.assertEqual(hist[k], 1)

  def test_source_locations(self):
    def f(x, y):
      s = jnp.sin(x)                  # sin
      return jnp.sin(s) + jnp.cos(y)  # sin, cos, add

    hist = js.source_locations(make_jaxpr(f)(1., 1.).jaxpr)
    self.assertEqual(set(hist.values()), set([1, 3]))

  def test_print_histogram(self):
    def f(x, y):
      s = jit(jnp.sin)(x)
      return jnp.sin(s) + jnp.cos(y)
    hist = js.primitives_by_source(make_jaxpr(f)(1., 1.).jaxpr)
    js.print_histogram(hist)


if __name__ == "__main__":
  absltest.main()
