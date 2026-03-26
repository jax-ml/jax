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
from jax import shard_map
from jax._src import config
from jax._src import test_util as jtu
from jax._src.pallas.fuser import fuser_utils
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np


jax.config.parse_flags_with_absl()


class FuserUtilsTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if config.enable_x64.value:
      self.skipTest('x64 not supported')

  def test_topo_identity(self):
    # traces as: sin -> cos -> mul -> sub
    def func1(x, y):
      a = jnp.sin(x)
      b = jnp.cos(y)
      return (a * b) - x

    # traces as: cos -> sin -> mul -> sub
    def func2(x, y):
      b = jnp.cos(y)
      a = jnp.sin(x)
      return (a * b) - x

    jaxpr1 = jax.make_jaxpr(func1)(1.0, 1.0)
    jaxpr2 = jax.make_jaxpr(func2)(1.0, 1.0)
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr1))
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr2, jaxpr2))
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr2))

  def test_commutative_nonidentity(self):
    # traces as: sin -> cos -> mul -> sub
    def func1(x, y):
      a = jnp.sin(x)
      b = jnp.cos(y)
      return (a * b) - x

    # traces as: cos -> sin -> mul -> sub
    def func2(x, y):
      a = jnp.sin(x)
      b = jnp.cos(y)
      return (b * a) - x

    jaxpr1 = jax.make_jaxpr(func1)(1.0, 1.0)
    jaxpr2 = jax.make_jaxpr(func2)(1.0, 1.0)
    self.assertFalse(fuser_utils.compare_jaxprs(jaxpr1, jaxpr2))

  def test_vmap_topo_identity(self):
    def func1(x, y):
      a = jnp.sin(x)
      b = jnp.cos(y)
      return a * b

    def func2(x, y):
      b = jnp.cos(y)
      a = jnp.sin(x)
      return a * b

    v_func1 = jax.vmap(func1)
    v_func2 = jax.vmap(func2)
    jaxpr1 = jax.make_jaxpr(v_func1)(jnp.ones(4), jnp.ones(4))
    jaxpr2 = jax.make_jaxpr(v_func2)(jnp.ones(4), jnp.ones(4))
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr1))
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr2))

  def test_shard_map_topo_identity(self):
    mesh = Mesh(np.array(jax.devices()[:1]), ('x',))

    def func1(x, y):
      def smap_f(a, b):
        c = jnp.sin(a)
        d = jnp.cos(b)
        return c * d

      return shard_map(
          smap_f, mesh=mesh, in_specs=(P('x'), P('x')), out_specs=P('x')
      )(x, y)

    def func2(x, y):
      def smap_g(a, b):
        d = jnp.cos(b)
        c = jnp.sin(a)
        return c * d

      return shard_map(
          smap_g, mesh=mesh, in_specs=(P('x'), P('x')), out_specs=P('x')
      )(x, y)

    jaxpr1 = jax.make_jaxpr(func1)(jnp.ones((4, 4)), jnp.ones((4, 4)))
    jaxpr2 = jax.make_jaxpr(func2)(jnp.ones((4, 4)), jnp.ones((4, 4)))
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr1))
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr2))

  def test_scan_topo_identity(self):
    def func1(xs):
      def f(c, x):
        a = c + 1
        b = x * 2.0
        return a, b

      return jax.lax.scan(f, 0, xs)[1]

    def func2(ys):
      def g(d, y):
        b = y * 2.0
        a = d + 1
        return a, b

      return jax.lax.scan(g, 0, ys)[1]

    jaxpr1 = jax.make_jaxpr(func1)(jnp.arange(10.0))
    jaxpr2 = jax.make_jaxpr(func2)(jnp.arange(10.0))
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr1))
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr2))

  def test_cond_topo_identity(self):
    def func1(pred, x):
      def true_fn(a):
        b = jnp.sin(a)
        c = jnp.cos(a)
        return b + c

      def false_fn(a):
        b = a - 1.0
        return b / 2.0

      return jax.lax.cond(pred, true_fn, false_fn, x)

    def func2(p, y):
      def t_fn(val):
        c = jnp.cos(val)
        b = jnp.sin(val)
        return b + c

      def f_fn(v):
        tmp = v - 1.0
        return tmp / 2.0

      return jax.lax.cond(p, t_fn, f_fn, y)

    jaxpr1 = jax.make_jaxpr(func1)(True, 1.0)
    jaxpr2 = jax.make_jaxpr(func2)(True, 1.0)
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr1))
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr2))

  def test_while_topo_identity(self):
    def func1(x):
      def cond_fn(val):
        return val[0] < 10

      def body_fn(val):
        a = val[0] + 1
        b = val[1] * 2
        return (a, b)

      return jax.lax.while_loop(cond_fn, body_fn, x)

    def func2(y):
      def c_fn(v):
        return v[0] < 10

      def b_fn(v):
        b = v[1] * 2
        a = v[0] + 1
        return (a, b)

      return jax.lax.while_loop(c_fn, b_fn, y)

    jaxpr1 = jax.make_jaxpr(func1)((0, 1))
    jaxpr2 = jax.make_jaxpr(func2)((0, 1))
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr1))
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr2))

  def test_jvp_topo_identity(self):
    def func1(x, y):
      a = jnp.sin(x)
      b = jnp.cos(y)
      return a * b

    def func2(x, y):
      b = jnp.cos(y)
      a = jnp.sin(x)
      return a * b

    def test1(x, y):
      return jax.jvp(func1, (x, y), (1.0, 1.0))[1]

    def test2(x, y):
      return jax.jvp(func2, (x, y), (1.0, 1.0))[1]

    jaxpr1 = jax.make_jaxpr(test1)(1.0, 1.0)
    jaxpr2 = jax.make_jaxpr(test2)(1.0, 1.0)
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr1))
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr2))

  def test_vjp_topo_identity(self):
    def func1(x, y):
      a = jnp.sin(x)
      b = jnp.cos(y)
      return a * b

    def func2(x, y):
      b = jnp.cos(y)
      a = jnp.sin(x)
      return a * b

    def test1(x, y):
      return jax.grad(func1, argnums=(0, 1))(x, y)

    def test2(x, y):
      return jax.grad(func2, argnums=(0, 1))(x, y)

    jaxpr1 = jax.make_jaxpr(test1)(1.0, 1.0)
    jaxpr2 = jax.make_jaxpr(test2)(1.0, 1.0)
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr1))
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr2))

  def test_embedded_constant_identity(self):
    c_arr = jnp.array([1.0, 2.0])

    def func1(x, y):
      a = x * 2.0  # scalar const
      b = y + c_arr  # array const
      return a + b

    def func2(x, y):
      b = y + c_arr
      a = x * 2.0
      return a + b

    jaxpr1 = jax.make_jaxpr(func1)(jnp.ones(2), jnp.ones(2))
    jaxpr2 = jax.make_jaxpr(func2)(jnp.ones(2), jnp.ones(2))
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr1))
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr2))

  def test_embedded_constant_nonidentity(self):
    c_arr1 = jnp.array([1.0, 2.0])
    c_arr2 = jnp.array([1.0, 3.0])

    def func1(x, y):
      a = x * 2.0
      b = y + c_arr1
      return a + b

    def func2(x, y):
      b = y + c_arr2
      a = x * 2.0
      return a + b

    jaxpr1 = jax.make_jaxpr(func1)(jnp.ones(2), jnp.ones(2))
    jaxpr2 = jax.make_jaxpr(func2)(jnp.ones(2), jnp.ones(2))
    self.assertFalse(fuser_utils.compare_jaxprs(jaxpr1, jaxpr2))

  def test_ref_effects_topo_identity(self):
    def func1(x, y):
      ref1 = jax.new_ref(x)
      ref2 = jax.new_ref(y)
      jax.ref.set(ref1, (), x * 2.0)
      jax.ref.set(ref2, (), y + 1.0)
      a = jax.ref.get(ref1, ())
      b = jax.ref.get(ref2, ())
      return a + b

    def func2(x, y):
      ref2 = jax.new_ref(y)
      ref1 = jax.new_ref(x)
      jax.ref.set(ref2, (), y + 1.0)
      jax.ref.set(ref1, (), x * 2.0)
      b = jax.ref.get(ref2, ())
      a = jax.ref.get(ref1, ())
      return a + b

    jaxpr1 = jax.make_jaxpr(func1)(1.0, 1.0)
    jaxpr2 = jax.make_jaxpr(func2)(1.0, 1.0)
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr1))
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr2))

  def test_ref_effects_nonidentity(self):
    def func1(x, y):
      ref1 = jax.new_ref(x)
      ref2 = jax.new_ref(y)
      jax.ref.set(ref1, (), x * 2.0)
      jax.ref.set(ref2, (), y + 1.0)
      a = jax.ref.get(ref1, ())
      b = jax.ref.get(ref2, ())
      return a + b

    def func2(x, y):
      ref1 = jax.new_ref(x)
      ref2 = jax.new_ref(y)
      jax.ref.set(ref1, (), y + 1.0)
      jax.ref.set(ref2, (), x * 2.0)
      b = jax.ref.get(ref1, ())
      a = jax.ref.get(ref2, ())
      return a + b

    jaxpr1 = jax.make_jaxpr(func1)(1.0, 1.0)
    jaxpr2 = jax.make_jaxpr(func2)(1.0, 1.0)
    self.assertFalse(fuser_utils.compare_jaxprs(jaxpr1, jaxpr2))

  def test_ref_effects_outside_topo_identity(self):
    ref1 = jax.new_ref(1.0)
    ref2 = jax.new_ref(1.0)

    def func1(x, y):
      jax.ref.set(ref1, (), x * 2.0)
      jax.ref.set(ref2, (), y + 1.0)
      a = jax.ref.get(ref1, ())
      b = jax.ref.get(ref2, ())
      return a + b

    def func2(x, y):
      jax.ref.set(ref2, (), y + 1.0)
      jax.ref.set(ref1, (), x * 2.0)
      b = jax.ref.get(ref2, ())
      a = jax.ref.get(ref1, ())
      return a + b

    jaxpr1 = jax.make_jaxpr(func1)(1.0, 1.0)
    jaxpr2 = jax.make_jaxpr(func2)(1.0, 1.0)
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr1))
    self.assertTrue(fuser_utils.compare_jaxprs(jaxpr1, jaxpr2))

  def test_ref_effects_outside_nonidentity(self):
    ref1 = jax.new_ref(1.0)
    ref2 = jax.new_ref(1.0)

    def func1(x, y):
      jax.ref.set(ref1, (), x * 2.0)
      jax.ref.set(ref2, (), y + 1.0)
      a = jax.ref.get(ref1, ())
      b = jax.ref.get(ref2, ())
      return a + b

    def func2(x, y):
      jax.ref.set(ref1, (), y + 1.0)
      jax.ref.set(ref2, (), x * 2.0)
      b = jax.ref.get(ref1, ())
      a = jax.ref.get(ref2, ())
      return a + b

    jaxpr1 = jax.make_jaxpr(func1)(1.0, 1.0)
    jaxpr2 = jax.make_jaxpr(func2)(1.0, 1.0)
    self.assertFalse(fuser_utils.compare_jaxprs(jaxpr1, jaxpr2))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
