# Copyright 2021 Google LLC
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
import functools

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax import core
from jax import lax
from jax import linear_util as lu
from jax.config import config
from jax._src import test_util as jtu
from jax._src import source_info_util
from jax._src.lib import xla_client

config.parse_flags_with_absl()
extend_name_stack = source_info_util.extend_name_stack

def _get_hlo(f):
  def wrapped(*args, **kwargs):
    c = jax.xla_computation(f)(*args, **kwargs)
    print_opts = xla_client._xla.HloPrintOptions.short_parsable()
    print_opts.print_metadata = True
    return c.as_hlo_module().to_string(print_opts)
  return wrapped

class _EnableNameStackTestCase(jtu.JaxTestCase):

  def setUp(self):
    self.cfg = config._read("jax_experimental_name_stack")
    config.update("jax_experimental_name_stack", True)

  def tearDown(self):
    config.update("jax_experimental_name_stack", self.cfg)


class NameStackTest(_EnableNameStackTestCase):

  def test_trivial_name_stack(self):

    def f(x):
      return x + 1
    jaxpr = jax.make_jaxpr(f)(2).jaxpr
    for eqn in jaxpr.eqns:
      self.assertEqual(str(eqn.source_info.name_stack), '')

  def test_name_call_name_stack(self):

    @jax.named_call
    def f(x):
      return x + 1
    jaxpr = jax.make_jaxpr(f)(2).jaxpr
    for eqn in jaxpr.eqns:
      self.assertEqual(str(eqn.source_info.name_stack), 'f')

  def test_manual_name_stack(self):

    @extend_name_stack('foo')
    def f(x):
      return x + 1
    jaxpr = jax.make_jaxpr(f)(2).jaxpr
    for eqn in jaxpr.eqns:
      self.assertEqual(str(eqn.source_info.name_stack), 'foo')

  def test_nested_name_stack(self):

    @extend_name_stack('foo')
    def f(x):
      with extend_name_stack('bar'):
        return x + 1
    jaxpr = jax.make_jaxpr(f)(2).jaxpr
    for eqn in jaxpr.eqns:
      self.assertEqual(str(eqn.source_info.name_stack), 'foo/bar')

  def test_multiple_name_stack(self):

    def f(x):
      with extend_name_stack('foo'):
        y = x + 1
      with extend_name_stack('bar'):
        with extend_name_stack('baz'):
          return y + 1
    jaxpr = jax.make_jaxpr(f)(2).jaxpr
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'foo')
    self.assertEqual(str(jaxpr.eqns[1].source_info.name_stack), 'bar/baz')

  def test_call_primitive_jaxpr_should_not_store_outer_name_stack(self):
    @extend_name_stack('foo')
    def f(x):
      @lu.wrap_init
      @extend_name_stack('bar')
      def _f(x):
        return [x + 1]
      return core.call(_f, x)[0]

    jaxpr = jax.make_jaxpr(f)(2).jaxpr
    self.assertEqual(str(jaxpr.eqns[0].params['call_jaxpr'].eqns[0].source_info.name_stack), 'bar')

    hlo_text = _get_hlo(f)(2)
    self.assertIn('foo/jit(core_call)/bar', hlo_text)

  def test_xla_call_primitive_jaxpr_should_not_store_outer_name_stack(self):
    @extend_name_stack('foo')
    def f(x):
      @jax.jit
      @extend_name_stack('bar')
      def _f(x):
        return x + 1
      return _f(x)

    jaxpr = jax.make_jaxpr(f)(2).jaxpr
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'foo')
    self.assertEqual(str(jaxpr.eqns[0].params['call_jaxpr'].eqns[0].source_info.name_stack), 'bar')

    hlo_text = _get_hlo(f)(2)
    self.assertIn('foo/jit(_f)/bar', hlo_text)

  def test_pmap_call_primitive_jaxpr_should_not_store_outer_name_stack(self):
    @extend_name_stack('foo')
    @jax.pmap
    def f(x):
      with extend_name_stack('bar'):
        return x + 1
    jaxpr = jax.make_jaxpr(f)(jnp.ones(1)).jaxpr
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'foo')
    self.assertEqual(str(jaxpr.eqns[0].params['call_jaxpr'].eqns[0].source_info.name_stack), 'bar')


class NameStackTransformationTest(_EnableNameStackTestCase):

  def test_vmap_should_transform_name_stack(self):
    @jax.vmap
    def f(x):
      with extend_name_stack('foo'):
        return x + 1
    jaxpr = jax.make_jaxpr(f)(jnp.ones(2)).jaxpr
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'vmap(foo)')

  def test_vmap_should_transform_inner_name_stacks(self):
    @extend_name_stack('foo')
    @jax.vmap
    def f(x):
      with extend_name_stack('bar'):
        with extend_name_stack('baz'):
          return x + 1
    jaxpr = jax.make_jaxpr(f)(jnp.ones(2)).jaxpr
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'foo/vmap(bar)/vmap(baz)')

  def test_vmap_should_apply_to_call_jaxpr(self):
    @extend_name_stack('foo')
    @jax.vmap
    def f(x):
      @jax.jit
      @extend_name_stack('bar')
      def _f(x):
        return x + 1
      return _f(x)

    jaxpr = jax.make_jaxpr(f)(jnp.ones(2)).jaxpr
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'foo')
    self.assertEqual(str(jaxpr.eqns[0].params['call_jaxpr'].eqns[0].source_info.name_stack), 'bar')

    hlo_text = _get_hlo(f)(jnp.ones(2))
    self.assertIn('foo/vmap(jit(_f))/vmap(bar)', hlo_text)

  def test_jvp_should_transform_stacks(self):
    def f(x):
      with extend_name_stack('bar'):
        with extend_name_stack('baz'):
          return jnp.square(x)
    g = extend_name_stack('foo')(lambda x, t: jax.jvp(f, (x,), (t,)))
    jaxpr = jax.make_jaxpr(g)(1., 1.).jaxpr
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack),
                     'foo/jvp(bar)/jvp(baz)')

  def test_jvp_should_apply_to_call_jaxpr(self):
    @jax.jit
    def f(x):
      with extend_name_stack('bar'):
        with extend_name_stack('baz'):
          return jnp.square(x)
    g = extend_name_stack('foo')(lambda x, t: jax.jvp(f, (x,), (t,)))
    jaxpr = jax.make_jaxpr(g)(1., 1.).jaxpr
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'foo')
    self.assertEqual(
        str(jaxpr.eqns[0].params['call_jaxpr'].eqns[0].source_info.name_stack),
        'bar/baz')

    hlo_text = _get_hlo(g)(1., 1.)
    self.assertIn('foo/jvp(jit(f))/jvp(bar)', hlo_text)

  def test_grad_should_add_jvp_and_transpose_to_name_stack(self):
    @jax.value_and_grad
    def f(x):
      with extend_name_stack('foo'):
        return 2 * jnp.sin(x)
    jaxpr = jax.make_jaxpr(f)(1.).jaxpr
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'jvp(foo)')
    self.assertEqual(str(jaxpr.eqns[1].source_info.name_stack), 'jvp(foo)')
    self.assertEqual(str(jaxpr.eqns[2].source_info.name_stack), 'jvp(foo)')
    self.assertEqual(str(jaxpr.eqns[4].source_info.name_stack),
        'transpose(jvp(foo))')

    hlo_text = _get_hlo(f)(1.)
    self.assertIn('jvp(foo)/sin', hlo_text)
    self.assertIn('jvp(foo)/cos', hlo_text)
    self.assertIn('transpose(jvp(foo))/mul', hlo_text)

  def test_grad_should_add_jvp_and_transpose_to_call_jaxpr(self):
    @jax.grad
    @extend_name_stack('foo')
    @jax.jit
    def f(x):
      with extend_name_stack('bar'):
        return jnp.sin(x)
    jaxpr = jax.make_jaxpr(f)(1.).jaxpr
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'jvp(foo)')
    self.assertEqual(str(jaxpr.eqns[1].source_info.name_stack), 'transpose(jvp(foo))')
    self.assertEqual(str(
      jaxpr.eqns[0].params['call_jaxpr'].eqns[0].source_info.name_stack), 'bar')
    self.assertEqual(str(
      jaxpr.eqns[0].params['call_jaxpr'].eqns[1].source_info.name_stack), 'bar')
    self.assertEqual(str(
      jaxpr.eqns[1].params['call_jaxpr'].eqns[0].source_info.name_stack), 'bar')

    hlo_text = _get_hlo(f)(1.)
    self.assertIn('jvp(foo)/jvp(jit(f))/jvp(bar)/sin', hlo_text)
    self.assertIn('jvp(foo)/jvp(jit(f))/jvp(bar)/cos', hlo_text)
    self.assertIn(
        'transpose(jvp(foo))/transpose(jvp(jit(f)))/transpose(jvp(bar))/mul',
        hlo_text)


class NameStackControlFlowTest(_EnableNameStackTestCase):

  def test_while_loop_body_should_not_have_name_stack(self):

    @extend_name_stack('foo')
    def f(x):
      @extend_name_stack('bar')
      def body(x):
        return x + 1
      @extend_name_stack('bar_cond')
      def cond(x):
        return x < 5
      return lax.while_loop(cond, body, x)
    jaxpr = jax.make_jaxpr(f)(0)
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'foo')
    self.assertEqual(str(
      jaxpr.eqns[0].params['body_jaxpr'].eqns[0].source_info.name_stack),
      'bar')
    self.assertEqual(str(
      jaxpr.eqns[0].params['cond_jaxpr'].eqns[0].source_info.name_stack),
      'bar_cond')

    hlo_text = _get_hlo(f)(1.)
    self.assertIn('foo/while/body/bar', hlo_text)
    self.assertIn('foo/while/cond/bar_cond', hlo_text)

  def test_vmap_of_while_loop_should_transform_name_stack(self):

    @jax.vmap
    @extend_name_stack('foo')
    def f(x):
      @extend_name_stack('bar')
      def body(x):
        return x + 1
      @extend_name_stack('bar_cond')
      def cond(x):
        return x < 5
      return lax.while_loop(cond, body, x)
    jaxpr = jax.make_jaxpr(f)(jnp.arange(2))
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'vmap(foo)')
    self.assertEqual(str(
      jaxpr.eqns[0].params['body_jaxpr'].eqns[0].source_info.name_stack),
      'bar')
    self.assertEqual(str(
      jaxpr.eqns[0].params['cond_jaxpr'].eqns[0].source_info.name_stack),
      'bar_cond')

    hlo_text = _get_hlo(f)(jnp.arange(2.))
    self.assertIn('vmap(foo)/vmap(while)/vmap(body)/vmap(bar)', hlo_text)
    self.assertIn('vmap(foo)/vmap(while)/vmap(cond)/vmap(bar_cond)', hlo_text)

  def test_jvp_of_while_loop_transforms_name_stack(self):

    @extend_name_stack('foo')
    def f(x):
      @extend_name_stack('bar')
      def body(x):
        return x + 1.
      @extend_name_stack('bar_cond')
      def cond(x):
        return x < 5.
      return lax.while_loop(cond, body, x)
    g = lambda x, t: jax.jvp(f, (x,), (t,))
    jaxpr = jax.make_jaxpr(g)(1., 1.)
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'jvp(foo)')
    self.assertEqual(str(
      jaxpr.eqns[0].params['body_jaxpr'].eqns[0].source_info.name_stack),
      'bar')
    self.assertEqual(str(
      jaxpr.eqns[0].params['cond_jaxpr'].eqns[0].source_info.name_stack),
      'bar_cond')

    hlo_text = _get_hlo(g)(1., 1.)
    self.assertIn('jvp(foo)/jvp(while)/jvp(body)/jvp(bar)', hlo_text)
    self.assertIn('jvp(foo)/jvp(while)/jvp(cond)/jvp(bar_cond)', hlo_text)

  def test_vmap_of_jvp_of_while_loop_transforms_name_stack(self):

    @extend_name_stack('foo')
    def f(x):
      @extend_name_stack('bar')
      def body(x):
        return x + 1.
      @extend_name_stack('bar_cond')
      def cond(x):
        return x < 5.
      return lax.while_loop(cond, body, x)
    g = jax.vmap(lambda x, t: jax.jvp(f, (x,), (t,)))
    jaxpr = jax.make_jaxpr(g)(jnp.arange(2.), jnp.ones(2))
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'vmap(jvp(foo))')
    self.assertEqual(str(
      jaxpr.eqns[0].params['body_jaxpr'].eqns[0].source_info.name_stack),
      'bar')
    self.assertEqual(str(
      jaxpr.eqns[0].params['cond_jaxpr'].eqns[0].source_info.name_stack),
      'bar_cond')

    hlo_text = _get_hlo(g)(jnp.arange(2.), jnp.ones(2))
    self.assertIn(
        'vmap(jvp(foo))/vmap(jvp(while))/vmap(jvp(body))/vmap(jvp(bar))',
        hlo_text)
    self.assertIn(
        'vmap(jvp(foo))/vmap(jvp(while))/vmap(jvp(cond))/vmap(jvp(bar_cond))',
        hlo_text)

  def test_cond_body_should_not_have_name_stack(self):

    @extend_name_stack('foo')
    def f(x, y):
      @extend_name_stack('true')
      def true_fn(x):
        return x + 1
      @extend_name_stack('false')
      def false_fn(x):
        return x - 1
      return lax.cond(y, true_fn, false_fn, x)
    jaxpr = jax.make_jaxpr(f)(0, True)
    for eqn in jaxpr.eqns:
      self.assertEqual(str(eqn.source_info.name_stack), 'foo')
      if eqn.primitive is lax.cond_p:
        self.assertEqual(str(
          eqn.params['branches'][0].eqns[0].source_info.name_stack),
          'false')
        self.assertEqual(str(
          eqn.params['branches'][1].eqns[0].source_info.name_stack),
          'true')

    hlo_text = _get_hlo(f)(1, True)
    self.assertIn('foo/cond/branch_0_fun/false/sub', hlo_text)
    self.assertIn('foo/cond/branch_1_fun/true/add', hlo_text)

  def test_vmap_of_cond_should_transform_name_stack(self):

    @extend_name_stack('foo')
    @functools.partial(jax.vmap, in_axes=(0, None))
    def f(x, y):
      @extend_name_stack('true')
      def true_fn(x):
        return x + 1
      @extend_name_stack('false')
      def false_fn(x):
        return x - 1
      return lax.cond(y, true_fn, false_fn, x)
    jaxpr = jax.make_jaxpr(f)(jnp.arange(2), True)
    for eqn in jaxpr.eqns:
      self.assertIn('foo', str(eqn.source_info.name_stack))
      if eqn.primitive is lax.cond_p:
        self.assertEqual(str(
          eqn.params['branches'][0].eqns[0].source_info.name_stack),
          'false')
        self.assertEqual(str(
          eqn.params['branches'][1].eqns[0].source_info.name_stack),
          'true')

    hlo_text = _get_hlo(f)(jnp.arange(2.), True)
    self.assertIn('foo/vmap(cond)/vmap(branch_0_fun)/vmap(false)/sub', hlo_text)
    self.assertIn('foo/vmap(cond)/vmap(branch_1_fun)/vmap(true)/add', hlo_text)

  def test_jvp_of_cond_transforms_name_stack(self):

    @extend_name_stack('foo')
    def f(x, y):
      @extend_name_stack('true')
      def true_fn(x):
        return x + 1
      @extend_name_stack('false')
      def false_fn(x):
        return x - 1
      return lax.cond(y, true_fn, false_fn, x)
    f_ = lambda x: jax.jit(f)(x, True)
    g = lambda x, t: jax.jvp(f_, (x,), (t,))
    jaxpr = jax.make_jaxpr(g)(jnp.arange(2.), jnp.ones(2))
    call_jaxpr = jaxpr.jaxpr.eqns[0].params['call_jaxpr']
    self.assertEqual(str(call_jaxpr.eqns[1].source_info.name_stack), 'foo')
    self.assertEqual(str(
      call_jaxpr.eqns[1].params['branches'][0].eqns[0].source_info.name_stack),
      'false')
    self.assertEqual(str(
      call_jaxpr.eqns[1].params['branches'][1].eqns[0].source_info.name_stack),
      'true')

    hlo_text = _get_hlo(g)(jnp.arange(2.), jnp.ones(2))
    self.assertIn('jvp(foo)/jvp(cond)/jvp(branch_0_fun)/jvp(false)/sub', hlo_text)
    self.assertIn('jvp(foo)/jvp(cond)/jvp(branch_1_fun)/jvp(true)/add', hlo_text)

  def test_vmap_of_jvp_of_cond_transforms_name_stack(self):

    @extend_name_stack('foo')
    def f(x, y):
      @extend_name_stack('true')
      def true_fn(x):
        return x + 1
      @extend_name_stack('false')
      def false_fn(x):
        return x - 1
      return lax.cond(y, true_fn, false_fn, x)
    f_ = lambda x: jax.jit(f)(x, True)
    g = jax.vmap(lambda x, t: jax.jvp(f_, (x,), (t,)))
    jaxpr = jax.make_jaxpr(g)(jnp.arange(2.), jnp.ones(2))
    call_jaxpr = jaxpr.jaxpr.eqns[0].params['call_jaxpr']
    self.assertEqual(str(
      call_jaxpr.eqns[1].params['branches'][0].eqns[0].source_info.name_stack),
      'false')
    self.assertEqual(str(
      call_jaxpr.eqns[1].params['branches'][1].eqns[0].source_info.name_stack),
      'true')

    hlo_text = _get_hlo(g)(jnp.arange(2.), jnp.ones(2))
    self.assertIn(
        'vmap(jvp(foo))/vmap(jvp(cond))/vmap(jvp(branch_0_fun))/vmap(jvp(false))/sub',
        hlo_text)
    self.assertIn(
        'vmap(jvp(foo))/vmap(jvp(cond))/vmap(jvp(branch_1_fun))/vmap(jvp(true))/add',
        hlo_text)

  def test_grad_of_cond_transforms_name_stack(self):

    @jax.grad
    @extend_name_stack('foo')
    def f(x, y):
      @extend_name_stack('true')
      def true_fn(x):
        return x * x * 2.
      @extend_name_stack('false')
      def false_fn(x):
        return x / jnp.square(x)
      return lax.cond(y, true_fn, false_fn, x)
    jaxpr = jax.make_jaxpr(f)(1., True)
    self.assertEqual(str(jaxpr.eqns[1].source_info.name_stack), 'jvp(foo)')
    self.assertEqual(str(jaxpr.eqns[2].source_info.name_stack),
        'transpose(jvp(foo))')

    hlo_text = _get_hlo(f)(1., True)
    self.assertIn(
        'jvp(foo)/jvp(cond)/jvp(branch_0_fun)/jvp(false)/div',
        hlo_text)
    self.assertIn(
        'jvp(foo)/jvp(cond)/jvp(branch_1_fun)/jvp(true)/mul',
        hlo_text)
    self.assertIn(
        'transpose(jvp(foo))/transpose(jvp(cond))/transpose(jvp(branch_0_fun))/transpose(jvp(false))/div',
        hlo_text)
    self.assertIn(
        'transpose(jvp(foo))/transpose(jvp(cond))/transpose(jvp(branch_1_fun))/transpose(jvp(true))/mul',
        hlo_text)

  def test_vmap_of_grad_of_cond_transforms_name_stack(self):

    @functools.partial(jax.vmap, in_axes=(0, None))
    @jax.grad
    @extend_name_stack('foo')
    def f(x, y):
      @extend_name_stack('true')
      def true_fn(x):
        return x * x * 2.
      @extend_name_stack('false')
      def false_fn(x):
        return x / x / 2.
      return lax.cond(y, true_fn, false_fn, x)
    jaxpr = jax.make_jaxpr(f)(jnp.arange(2.), True)
    self.assertEqual(str(jaxpr.eqns[1].source_info.name_stack), 'vmap(jvp(foo))')
    self.assertEqual(str(jaxpr.eqns[2].source_info.name_stack),
        'vmap(transpose(jvp(foo)))')

    hlo_text = _get_hlo(f)(jnp.arange(2.), True)
    self.assertIn(
        'vmap(jvp(foo))/vmap(jvp(cond))/vmap(jvp(branch_0_fun))/vmap(jvp(false))/div',
        hlo_text)
    self.assertIn(
        'vmap(jvp(foo))/vmap(jvp(cond))/vmap(jvp(branch_1_fun))/vmap(jvp(true))/mul',
        hlo_text)
    self.assertIn(
        'vmap(transpose(jvp(foo)))/vmap(transpose(jvp(cond)))/vmap(transpose(jvp(branch_0_fun)))/vmap(transpose(jvp(false)))/div',
        hlo_text)
    self.assertIn(
        'vmap(transpose(jvp(foo)))/vmap(transpose(jvp(cond)))/vmap(transpose(jvp(branch_1_fun)))/vmap(transpose(jvp(true)))/mul',
        hlo_text)

  def test_scan_body_should_not_have_name_stack(self):

    @extend_name_stack('foo')
    def f(x):
      @extend_name_stack('scan_body')
      def body(carry, x):
        return carry + x, carry + x
      return lax.scan(body, x, jnp.arange(5.))
    jaxpr = jax.make_jaxpr(f)(1.)
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'foo')
    self.assertEqual(str(
      jaxpr.eqns[1].params['jaxpr'].eqns[0].source_info.name_stack),
      'scan_body')

    hlo_text = _get_hlo(f)(1.)
    self.assertIn('foo/while/body/scan_body', hlo_text)

  def test_vmap_of_scan_should_transform_stack(self):

    @jax.vmap
    @extend_name_stack('foo')
    def f(x):
      @extend_name_stack('scan_body')
      def body(carry, x):
        return carry + x, carry + x
      return lax.scan(body, x, jnp.arange(8.))
    jaxpr = jax.make_jaxpr(f)(jnp.arange(2.))
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'vmap(foo)')
    self.assertEqual(str(
      jaxpr.eqns[1].params['jaxpr'].eqns[0].source_info.name_stack),
      'scan_body')

    hlo_text = _get_hlo(f)(jnp.arange(2.))
    self.assertIn('vmap(foo)/vmap(while)/vmap(body)/vmap(scan_body)/add', hlo_text)

  def test_jvp_of_scan_should_transform_stack(self):

    @extend_name_stack('foo')
    def f(x):
      @extend_name_stack('scan_body')
      def body(carry, x):
        return carry + x, carry + x
      return lax.scan(body, x, jnp.arange(8.))
    g = lambda x, t: jax.jvp(f, (x,), (t,))
    jaxpr = jax.make_jaxpr(g)(1., 1.)
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'jvp(foo)')
    self.assertEqual(str(
      jaxpr.eqns[1].params['jaxpr'].eqns[0].source_info.name_stack),
      'scan_body')

    hlo_text = _get_hlo(g)(1., 1.)
    self.assertIn('jvp(foo)/jvp(while)/jvp(body)/jvp(scan_body)/add', hlo_text)

  def test_grad_of_scan_should_transform_stack(self):

    @jax.value_and_grad
    @extend_name_stack('foo')
    def f(x):
      @extend_name_stack('scan_body')
      def body(carry, x):
        return 2 * carry * x, carry + x
      return lax.scan(body, x, jnp.arange(8.))[0]
    jaxpr = jax.make_jaxpr(f)(1.)
    self.assertEqual(str(jaxpr.eqns[1].source_info.name_stack), 'jvp(foo)')
    self.assertEqual(str(jaxpr.eqns[3].source_info.name_stack),
        'transpose(jvp(foo))')
    self.assertEqual(str(
      jaxpr.eqns[1].params['jaxpr'].eqns[0].source_info.name_stack),
      'scan_body')

    hlo_text = _get_hlo(f)(1.)
    self.assertIn('jvp(foo)/jvp(while)/jvp(body)/jvp(scan_body)/mul', hlo_text)
    self.assertIn('transpose(jvp(foo))/transpose(jvp(while))/transpose(jvp(body))/transpose(jvp(scan_body))/mul', hlo_text)

  def test_vmap_of_grad_of_scan_should_transform_stack(self):

    @jax.vmap
    @jax.value_and_grad
    @extend_name_stack('foo')
    def f(x):
      @extend_name_stack('scan_body')
      def body(carry, x):
        return carry * x, carry + x
      return lax.scan(body, x, jnp.arange(8.))[0]
    jaxpr = jax.make_jaxpr(f)(jnp.arange(2.))
    self.assertEqual(str(jaxpr.eqns[1].source_info.name_stack), 'vmap(jvp(foo))')
    self.assertEqual(str(jaxpr.eqns[3].source_info.name_stack),
        'vmap(transpose(jvp(foo)))')
    self.assertEqual(str(
      jaxpr.eqns[1].params['jaxpr'].eqns[0].source_info.name_stack),
      'scan_body')

    hlo_text = _get_hlo(f)(jnp.arange(2.))
    self.assertIn('vmap(jvp(foo))/vmap(jvp(while))/vmap(jvp(body))/vmap(jvp(scan_body))/mul', hlo_text)
    self.assertIn('vmap(transpose(jvp(foo)))/vmap(transpose(jvp(while)))/vmap(transpose(jvp(body)))/vmap(transpose(jvp(scan_body)))/mul', hlo_text)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
