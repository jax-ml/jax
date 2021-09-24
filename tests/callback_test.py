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

import jax
from jax._src import test_util as jtu
from jax.experimental.callback import find_by_value, rewrite, FoundValue
import jax.numpy as jnp
from jax import lax
from jax import jit
from jax import grad

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
    self.assertAllClose(f(x), jnp.array([4.0, 8.0]))

    self.assertAllClose(
        rewrite(f, {lax.mul_p: lambda x, y: x + y})(x),
        jnp.array([4.0, 6.0]))

  def testRewriteJIT(self):
    def f(x):
      @jit
      def g(x):
        return x * 2
      return g(x)

    x = jnp.array([2.0, 4.0])
    self.assertAllClose(f(x), jnp.array([4.0, 8.0]))

    self.assertAllClose(
        rewrite(f, {lax.mul_p: lambda x, y: x + y})(x),
        jnp.array([4.0, 6.0]))

  def testRewriteWithCustomGradients(self):
    def f(x):
      return jax.nn.relu(x)

    x = jnp.array([2.0, 4.0])
    self.assertAllClose(f(x), jnp.array([2.0, 4.0]))

    self.assertAllClose(
        rewrite(f, {})(x),
        jnp.array([2.0, 4.0]))

  def testRewriteThroughScan(self):
    def f(xs):
      def body(carry, x):
        carry = carry * 2.
        return carry, x - 2.
      return lax.scan(body, 1., xs)

    xs = jnp.arange(4.)
    carry, ys = f(xs)
    self.assertAllClose(carry, 16.)
    self.assertAllClose(ys, jnp.arange(4.) - 2.)

    rewrites = {
        lax.mul_p: lambda x, y: x + y,
        lax.sub_p: lambda x, y: x / y
    }
    carry, ys = rewrite(f, rewrites)(xs)
    self.assertAllClose(carry, 1. + 8.)
    self.assertAllClose(ys, jnp.arange(4.) / 2.)


  def testRewriteThroughWhile(self):
    def f(x):
      def cond(x):
        return x < 5
      def body(x):
        return x + 1
      return lax.while_loop(cond, body, x)

    x = 0
    self.assertAllClose(f(x), 5)

    rewrites = {
        lax.add_p: lambda x, y: x + y + 100,
    }
    self.assertAllClose(rewrite(f, rewrites)(x), 101)

    rewrites = {
        lax.lt_p: lambda x, y: x < y + 5
    }
    self.assertAllClose(rewrite(f, rewrites)(x), 10)


  def testRewriteThroughForLoop(self):
    def f(x):
      def body(i, x):
        return x * i
      return lax.fori_loop(1, 5, body, x)

    x = 1
    self.assertAllClose(f(x), 24)

    rewrites = {
        lax.mul_p: lambda x, y: x + y
    }
    self.assertAllClose(rewrite(f, rewrites)(x), 11)

  def testRewriteThroughCustomVJP(self):

    @jax.custom_gradient
    def f(x):
      return x * 2, lambda g: g + x

    x = 2.
    self.assertAllClose(f(x), 4.)
    self.assertAllClose(grad(f)(x), 3.)

    rewrites = {
        lax.mul_p: lambda x, y: x / y
    }
    g = rewrite(f, rewrites)

    self.assertAllClose(g(x), 1.)
    self.assertAllClose(grad(g)(x), 3.)

    rewrites = {
        lax.add_p: lambda x, y: x - y
    }
    g = rewrite(f, rewrites)

    self.assertAllClose(g(x), 4.)
    self.assertAllClose(grad(g)(x), -1.)

  def testRewriteThroughCustomVJPInScan(self):

    @jax.custom_gradient
    def foo(x):
      return x * 2, lambda g: g + x

    def f(x):
      out, _ = lax.scan(lambda c, _: (foo(c), None), x, None, length=1)
      return out

    x = 2.
    self.assertAllClose(f(x), 4.)
    self.assertAllClose(grad(f)(x), 3.)

    rewrites = {
        lax.mul_p: lambda x, y: x / y
    }
    g = rewrite(f, rewrites)

    self.assertAllClose(g(x), 1.)
    self.assertAllClose(grad(g)(x), 3.)

    rewrites = {
        lax.add_p: lambda x, y: x * y
    }
    g = rewrite(f, rewrites)

    self.assertAllClose(g(x), 4.)
    self.assertAllClose(grad(g)(x), 2.)

  def testRewriteThroughCustomJVP(self):

    @jax.custom_jvp
    def f(x):
      return x + 2

    @f.defjvp
    def f_jvp(primals, tangents):
      x, = primals
      d, = tangents
      return f(x), x * d

    x = 2.
    self.assertAllClose(f(x), 4.)
    f_primal, jvp = jax.jvp(f, (x,), (1.,))
    self.assertAllClose(f_primal, 4.)
    self.assertAllClose(jvp, 2.)
    self.assertAllClose(grad(f)(x), 2.)

    rewrites = {
        lax.add_p: lambda x, y: x - y
    }
    g = rewrite(f, rewrites)

    self.assertAllClose(g(x), 0.)
    g_primal, jvp = jax.jvp(g, (x,), (1.,))
    self.assertAllClose(g_primal, 0.)
    self.assertAllClose(jvp, 2.)
    self.assertAllClose(grad(g)(x), 2.)

  def testRewriteThroughCustomJVPInScan(self):

    @jax.custom_jvp
    def foo(x):
      return x + 2

    @foo.defjvp
    def foo_jvp(primals, tangents):
      x, = primals
      d, = tangents
      return f(x), x * d
    def f(x):
      out, _ = lax.scan(lambda c, _: (foo(c), None), x, None, length=1)
      return out

    x = 2.
    self.assertAllClose(f(x), 4.)
    f_primal, jvp = jax.jvp(f, (x,), (1.,))
    self.assertAllClose(f_primal, 4.)
    self.assertAllClose(jvp, 2.)
    self.assertAllClose(grad(f)(x), 2.)

    rewrites = {
        lax.add_p: lambda x, y: x - y
    }
    g = rewrite(f, rewrites)

    self.assertAllClose(g(x), 0.)
    g_primal, jvp = jax.jvp(g, (x,), (1.,))
    self.assertAllClose(g_primal, 0.)
    self.assertAllClose(jvp, 2.)
    self.assertAllClose(grad(g)(x), 2.)

    rewrites = {
        lax.mul_p: lambda x, y: x + y
    }
    g = rewrite(f, rewrites)

    self.assertAllClose(g(x), 4.)
    g_primal, jvp = jax.jvp(g, (x,), (1.,))
    self.assertAllClose(g_primal, 4.)
    self.assertAllClose(jvp, 3.)
    self.assertAllClose(grad(g)(x), 1.)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
