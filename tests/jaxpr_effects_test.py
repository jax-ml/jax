# Copyright 2022 Google LLC
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
import jax.numpy as jnp
from jax import core
from jax import lax
from jax.config import config
from jax._src import test_util as jtu

config.parse_flags_with_absl()

effect_p = core.Primitive('effect')
effect_p.multiple_results = True

@effect_p.def_effectful_abstract_eval
def _(*, effect):
  return [], {effect}


class JaxprEffectsTest(jtu.JaxTestCase):

  def test_trivial_jaxpr_has_no_effects(self):
    def f(x):
      return x + 1.
    jaxpr = jax.make_jaxpr(f)(2.)
    self.assertEqual(core.no_effects, jaxpr.effects)

  def test_effectful_primitive_in_jaxpr_creates_effects(self):
    def f(x):
      effect_p.bind(effect='foo')
      return x + 1.
    jaxpr = jax.make_jaxpr(f)(2.)
    self.assertEqual({'foo'}, jaxpr.jaxpr.eqns[0].effects)
    self.assertEqual({'foo'}, jaxpr.effects)

  def test_different_effects_in_jaxpr(self):
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='bar')
      return x + 1.
    jaxpr = jax.make_jaxpr(f)(2.)
    self.assertEqual({'foo'}, jaxpr.jaxpr.eqns[0].effects)
    self.assertEqual({'bar'}, jaxpr.jaxpr.eqns[1].effects)
    self.assertEqual({'foo', 'bar'}, jaxpr.effects)

  def test_jaxpr_typecheck_should_verify_eqn_effects_are_subset(self):
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='bar')
      return x + 1.
    jaxpr = jax.make_jaxpr(f)(2.).jaxpr

    # Edit jaxpr to make its type wrong
    jaxpr = jaxpr.replace(effects={'foo'})

    with self.assertRaisesRegex(core.JaxprTypeError,
        'Equation effects are not subset of Jaxpr effects.'):
      core.check_jaxpr(jaxpr)

class ControlFlowEffectsTest(jtu.JaxTestCase):

  def test_effects_disallowed_in_cond(self):
    def f1(x):
      def true_fun(x):
        effect_p.bind(effect='foo')
        return x
      def false_fun(x):
        return x
      return lax.cond(True, true_fun, false_fun, x)

    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f1)(2.)

    def f2(x):
      def true_fun(x):
        return x
      def false_fun(x):
        effect_p.bind(effect='foo')
        return x
      return lax.cond(True, true_fun, false_fun, x)

    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f2)(2.)

  def test_effects_disallowed_in_while(self):
    def f1(x):
      def cond_fun(x):
        effect_p.bind(effect='foo')
        return False
      def body_fun(x):
        return x
      return lax.while_loop(cond_fun, body_fun, x)

    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f1)(2.)

    def f2(x):
      def cond_fun(x):
        return False
      def body_fun(x):
        effect_p.bind(effect='foo')
        return x
      return lax.while_loop(cond_fun, body_fun, x)

    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f2)(2.)

  def test_effects_disallowed_in_scan(self):

    def f(x):
      def body(carry, x):
        effect_p.bind(effect='foo')
        return carry, x
      return lax.scan(body, x, jnp.arange(4))

    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f)(2.)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
