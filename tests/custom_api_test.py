# Copyright 2018 The JAX Authors.
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

import collections
from collections.abc import Callable
import concurrent.futures
import functools
from functools import partial
import itertools as it
import re
import unittest
import textwrap

from absl.testing import absltest, parameterized
import numpy as np

import jax
import jax.numpy as jnp
from jax import float0, grad, jit
from jax import lax
from jax import tree_util
from jax.ad_checkpoint import checkpoint as new_checkpoint
import jax.custom_batching
import jax.custom_derivatives
import jax.custom_transpose
import jax.experimental.custom_dce
from jax.errors import UnexpectedTracerError

from jax._src import api
from jax._src import api_util
from jax._src import config
from jax._src import core
from jax._src import custom_derivatives
from jax._src import deprecations
from jax._src import test_util as jtu
from jax._src.interpreters import partial_eval as pe

config.parse_flags_with_absl()


class CustomJVPTest(jtu.JaxTestCase):

  def test_basic(self):
    @jax.custom_jvp
    def f(x):
      return jnp.sin(x)
    def f_jvp(primals, tangents):
      x, = primals
      g, = tangents
      return f(x), 2 * jnp.cos(x) * g
    f.defjvp(f_jvp)

    x = 3.
    self.assertAllClose(f(x), jnp.sin(x))
    self.assertAllClose(api.jvp(f, (x,), (1.,)),
                        (jnp.sin(x), 2 * jnp.cos(x)))
    self.assertAllClose(api.grad(f)(x), 2 * jnp.cos(x))

  def test_invariance(self):
    @jax.custom_jvp
    def f(x):
      return jnp.cos(2 * x) / 2.
    def f_jvp(primals, tangents):
      x, = primals
      g, = tangents
      return (f(x), 3 * g)
    f.defjvp(f_jvp)
    def f2(x):
      y, _ = api.jvp(f, (x,), (x,))
      return y
    def f3(x):
      y, _ = api.jvp(f2, (x,), (x,))
      return y
    x = 1.
    self.assertAllClose(api.jvp(f, (x,), (x,)),
                        api.jvp(f2, (x,), (x,)),
                        check_dtypes=False)
    self.assertAllClose(api.jvp(f, (x,), (x,)),
                        api.jvp(f3, (x,), (x,)),
                        check_dtypes=False)

  def test_python_control_flow(self):
    @jax.custom_jvp
    def f(x):
      if x > 0:
        return jnp.sin(x)
      else:
        return jnp.cos(x)
    def f_jvp(primals, tangents):
      x, = primals
      g, = tangents
      if x > 0:
        return f(x), 2 * g
      else:
        return f(x), 3 * g
    f.defjvp(f_jvp)
    x = 2.
    self.assertAllClose(f(x), jnp.sin(x))
    self.assertAllClose(f(-x), jnp.cos(-x))
    self.assertAllClose(api.jvp(f, (x,), (1.,)),
                        (jnp.sin(x), 2.),
                        check_dtypes=False)
    self.assertAllClose(api.jvp(f, (-x,), (1.,)),
                        (jnp.cos(-x), 3.),
                        check_dtypes=False)
    self.assertAllClose(api.grad(f)(x), 2., check_dtypes=False)
    self.assertAllClose(api.grad(f)(-x), 3., check_dtypes=False)

  def test_vmap(self):
    @jax.custom_jvp
    def f(x):
      assert jnp.ndim(x) == 0
      return jnp.sin(x)
    def f_jvp(primals, tangents):
      x, = primals
      g, = tangents
      assert jnp.ndim(x) == jnp.ndim(g) == 0
      return f(x), 2 * jnp.cos(x) * g
    f.defjvp(f_jvp)

    x = jnp.arange(3.)
    xx = jnp.arange(6.).reshape(2, 3)

    # vmap of f
    self.assertAllClose(api.vmap(f)(x), jnp.sin(x))
    self.assertAllClose(api.vmap(api.vmap(f))(xx), jnp.sin(xx))

    # vmap of jvp of f
    self.assertAllClose(api.vmap(lambda x: api.jvp(f, (x,), (x,)))(x),
                        (jnp.sin(x), 2 * jnp.cos(x) * x))
    self.assertAllClose(api.vmap(api.vmap(lambda x: api.jvp(f, (x,), (x,))))(xx),
                        (jnp.sin(xx), 2 * jnp.cos(xx) * xx))

    # jvp of vmap of f
    self.assertAllClose(api.jvp(api.vmap(f), (x,), (x,)),
                        (jnp.sin(x), 2 * jnp.cos(x) * x))
    self.assertAllClose(api.jvp(api.vmap(api.vmap(f)), (xx,), (xx,)),
                        (jnp.sin(xx), 2 * jnp.cos(xx) * xx))

    # vmap of jvp of vmap of f
    self.assertAllClose(api.vmap(lambda x: api.jvp(api.vmap(f), (x,), (x,)))(xx),
                        (jnp.sin(xx), 2 * jnp.cos(xx) * xx))

  def test_jit(self):
    @jax.custom_jvp
    def f(x):
      return jnp.sin(x)
    def f_jvp(primals, tangents):
      x, = primals
      g, = tangents
      return f(x), 2 * jnp.cos(x) * g
    f.defjvp(f_jvp)

    x = 3.

    # jit
    self.assertAllClose(api.jit(f)(x), jnp.sin(x))
    self.assertAllClose(api.jit(api.jit(f))(x), jnp.sin(x))

    # jit of jvp
    self.assertAllClose(api.jit(lambda x: api.jvp(f, (x,), (x,)))(x),
                        (jnp.sin(x), 2 * jnp.cos(x) * x),
                        check_dtypes=False)

    # jvp of jit
    self.assertAllClose(api.jvp(api.jit(f), (x,), (x,)),
                        (jnp.sin(x), 2 * jnp.cos(x) * x),
                        check_dtypes=False)

  def test_pytrees(self):
    @jax.custom_jvp
    def f(x):
      return {'b': jnp.sin(x['a'])}
    def f_jvp(primals, tangents):
      x, = primals
      g, = tangents
      return f(x), {'b': 2 * jnp.cos(x['a']) * g['a']}
    f.defjvp(f_jvp)
    x = {'a': 3.}
    self.assertAllClose(f(x)['b'], jnp.sin(x['a']))
    self.assertAllClose(api.jvp(f, (x,), (x,)),
                        ({'b': jnp.sin(x['a'])},
                         {'b': 2 * jnp.cos(x['a']) * x['a']}),
                        check_dtypes=False)

  def test_kwargs(self):
    # from https://github.com/jax-ml/jax/issues/1938
    @jax.custom_jvp
    def my_fun(x, y, c=1.):
      return c * (x + y)
    def my_jvp(primals, tangents):
      x, y, c = primals
      t_x, t_y, t_c = tangents
      return my_fun(x, y, c), t_c
    my_fun.defjvp(my_jvp)
    f = lambda x, y: jnp.square(my_fun(x, y, c=2.)).sum()
    f(10., 5.)  # doesn't crash
    api.jvp(f, (10., 5.), (1., 1.))  # doesn't crash

  def test_initial_style(self):
    @jax.custom_jvp
    def f(x):
      return 3 * x
    def f_jvp(primals, tangents):
      x, = primals
      g, = tangents
      return f(x), 2 * g
    f.defjvp(f_jvp)

    def foo(x):
      out, _  = lax.scan(lambda c, _: (f(c), None), x, None, length=1)
      return out

    ans = api.grad(foo)(3.)
    expected = 2.
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(api.jit(foo))(3.)
    expected = 2.
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.jit(api.grad(foo))(3.)
    expected = 2.
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(api.grad(foo))(3.)
    expected = 0.
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(api.grad(api.jit(foo)))(3.)
    expected = 0.
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(api.jit(api.grad(foo)))(3.)
    expected = 0.
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.jit(api.grad(api.grad(foo)))(3.)
    expected = 0.
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_initial_style_vmap(self):
    @jax.custom_jvp
    def f(x):
      assert jnp.ndim(x) == 0
      return 3 * x
    def f_jvp(primals, tangents):
      x, = primals
      g, = tangents
      return f(x), 2 * g
    f.defjvp(f_jvp)

    def foo(x):
      out, _  = lax.scan(lambda c, _: (f(c), None), x, None, length=1)
      return out

    ans = api.vmap(foo)(jnp.ones(3))
    expected = 3. * jnp.ones(3)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.vmap(api.jit(foo))(jnp.ones(3))
    expected = 3. * jnp.ones(3)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.jit(api.vmap(foo))(jnp.ones(3))
    expected = 3. * jnp.ones(3)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(lambda x: api.vmap(foo)(x).sum())(jnp.ones(3))
    expected = 2. * jnp.ones(3)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(lambda x: api.vmap(api.jit(foo))(x).sum())(jnp.ones(3))
    expected = 2. * jnp.ones(3)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(lambda x: api.jit(api.vmap(foo))(x).sum())(jnp.ones(3))
    expected = 2. * jnp.ones(3)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(api.jit(lambda x: api.vmap(foo)(x).sum()))(jnp.ones(3))
    expected = 2. * jnp.ones(3)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.jit(api.grad(lambda x: api.vmap(foo)(x).sum()))(jnp.ones(3))
    expected = 2. * jnp.ones(3)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_initial_style_vmap_with_collective(self):

    @jax.custom_jvp
    def f(x):
      return lax.psum(x, 'foo')

    @f.defjvp
    def f_jvp(xs, ts):
      x, = xs
      t, = ts
      return lax.psum(x, 'foo'), t

    def g(x):
      jaxpr = api.make_jaxpr(f)(x)
      return core.eval_jaxpr(jaxpr.jaxpr, [], x)[0]

    v = api.vmap(lambda _, x: g(x), axis_name='foo', in_axes=(0, None),
        out_axes=None)(jnp.arange(4.), 2.)
    self.assertAllClose(v, 8.)

  def test_closed_over_tracers_error_message(self):
    def f(x):
      @jax.custom_jvp
      def g(y):
        return x + y
      def g_jvp(primals, tangents):
        return g(x), 2 * primals[0]
      g.defjvp(g_jvp)
      return g(1.)

    self.assertRaises(UnexpectedTracerError, lambda: api.jvp(f, (3.,), (1.,)))
    self.assertRaises(UnexpectedTracerError, lambda: api.grad(f)(3.))

  def test_nondiff_argnums(self):
    @partial(jax.custom_jvp, nondiff_argnums=(0,))
    def app(f, x):
      return f(x)
    def app_jvp(f, primals, tangents):
      (x,), (t,) = primals, tangents
      return app(f, x), 3 * t
    app.defjvp(app_jvp)

    ans = app(lambda x: 2 * x, 1)
    expected = 2
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.jvp(lambda x: app(lambda y: 2 * y, x), (1.,), (1.,))
    expected = (2., 3.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_nondiff_argnames(self):
    @partial(jax.custom_jvp, nondiff_argnames=('f',))
    def app(f, x):
      return f(x)

    def app_jvp(f, primals, tangents):
      (x,), (t,) = primals, tangents
      return app(f, x), 3 * t

    app.defjvp(app_jvp)

    ans = app(lambda x: 2 * x, 1)
    expected = 2
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_nondiff_arg_jit_tracer(self):
    # This test would pass with "final-style" JIT tracing, but that was
    # misleading: it doesn't work with "initial-style" staging, i.e. control
    # flow primitives like jax.lax.scan or even pjit. The behavior isn't very
    # useful either: instead of using nondiff_argnums here, a user can just pass
    # such inputs as ordinary arguments, and ignore the corresponding tangents.
    # Then nondiff_argnums can be reserved for (1) non jaxtype data (like a
    # string- or callable-valued argument which parameterizes the function or
    # rule) or (2) static data (e.g. integers which parameterize shapes).
    raise unittest.SkipTest("behavior no longer supported")

    @partial(jax.custom_jvp, nondiff_argnums=(0,))
    def f(x, y):
      return x * y
    def f_jvp(x, primals, tangents):
      (y,), (t_y,) = primals, tangents
      return f(x, y), 5 * t_y
    f.defjvp(f_jvp)

    @jit
    def g(x, y):
      return f(x, y)

    ans = api.jvp(lambda y: g(2., y), (3.,), (1.,))
    expected = (6., 5.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_nondiff_arg_vmap_tracer(self):
    @partial(jax.custom_jvp, nondiff_argnums=(0,))
    def f(x, y):
      return x * y
    def f_jvp(x, primals, tangents):
      (y,), (t_y,) = primals, tangents
      return f(x, y), 5 * t_y
    f.defjvp(f_jvp)

    g = jax.vmap(f)

    ans = api.jvp(lambda y: g(jnp.array([2.]), y),
                  (jnp.array([3.]),), (jnp.array([1.]),))
    expected = (jnp.array([6.]), jnp.array([5.]))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_nondiff_arg_hiding_jvp_tracer(self):
    def f(x):
      @partial(jax.custom_jvp, nondiff_argnums=(0,))
      def g(h, x):
        return h(x)
      @g.defjvp
      def g_jvp(h, primals, tangents):
        x, = primals
        t, = tangents
        return g(h, x), 2. * t
      h = lambda y: x + y  # capture x
      return g(h, x)

    with self.assertRaises(UnexpectedTracerError):
      api.jvp(f, (2.,), (1.,))

  def test_vmap_axes(self):
    raise unittest.SkipTest("TODO")  # TODO(mattjj): write test

  def test_pmap(self):
    raise unittest.SkipTest("TODO")  # TODO(mattjj): write test

  def test_missing_jvp_rule_error_message(self):
    @jax.custom_jvp
    def foo(x):
      return x ** 2

    self.assertRaisesRegex(
        AttributeError,
        r"No JVP defined for custom_jvp function foo using defjvp.",
        lambda: foo(2))
    self.assertRaisesRegex(
        AttributeError,
        r"No JVP defined for custom_jvp function foo using defjvp.",
        lambda: api.jvp(foo, (2.,), (1.,)))
    self.assertRaisesRegex(
        AttributeError,
        r"No JVP defined for custom_jvp function foo using defjvp.",
        lambda: api.grad(foo)(2.))

  def test_jvp_rule_inconsistent_pytree_structures_error_message(self):
    @jax.custom_jvp
    def f(x):
      return (x**2,)

    @f.defjvp
    def foo_jvp(primals, tangents):
      x, = primals
      t, = tangents
      return f(x), [2 * x * t, x]

    f(2.)  # doesn't crash
    self.assertRaisesRegex(
        TypeError,
        re.escape(
            "Custom JVP rule foo_jvp for function f "
            "must produce primal and tangent outputs "
            "with equal container (pytree) structures, but got "
            "{} and {} respectively.".format(
                jax.tree.structure((1,)),
                jax.tree.structure([1, 2]))
        ),
        lambda: api.jvp(f, (2.,), (1.,)))

  def test_primal_tangent_aval_disagreement_error_message(self):
    @jax.custom_jvp
    def f(x):
      return x ** 2

    @f.defjvp
    def foo_jvp(primals, tangents):
      x, = primals
      t, = tangents
      return f(x), jnp.reshape(t, (1,))

    f(2.)  # doesn't crash
    self.assertRaisesRegex(
        TypeError,
        re.escape(
            "Custom JVP rule must produce primal and tangent outputs "
            "with corresponding shapes and dtypes. "
            "Expected float32[] (tangent type of float32[]) but got float32[1]."),
        lambda: api.jvp(f, (jnp.float32(2.),), (jnp.float32(1.),)))


  def test_jvp_rule_doesnt_return_pair_error_message(self):
    # https://github.com/jax-ml/jax/issues/2516

    @jax.custom_jvp
    def f(x):
      return x ** 2

    @f.defjvp
    def foo_jvp(primals, tangents):
      x, = primals
      t, = tangents
      return t

    f(2.)  # doesn't crash
    self.assertRaisesRegex(
        TypeError,
        re.escape(
            "Custom JVP rule foo_jvp for function f "
            "must produce a pair (list or tuple of length two) "
            "representing primal and tangent outputs, but got 1.0"),
        lambda: api.jvp(f, (2.,), (1.,)))

  def test_jvp_rule_primal_out_type_doesnt_match_primal_error_message(self):
    # https://github.com/lucidrains/flash-attention-jax/issues/7

    def scan_apply(f, x):
      y, _ = jax.lax.scan(lambda x, _: (f(x), None), x, None, length=1)
      return y

    @jax.custom_jvp
    def f(x):
      return x

    @f.defjvp
    def f_jvp(primals, tangents):
      (x,), (xdot,) = primals, tangents
      return (x, x), (xdot, xdot)

    x = jnp.float32(1.)
    self.assertRaisesRegex(
        TypeError,
        re.escape(
            "Custom JVP rule f_jvp for function f must produce a pair "
            "(list or tuple of length two) where the first element represents "
            "the primal output (equal in value to the output of the "
            "custom_jvp-decorated function f, and in particular of the "
            "same container/pytree structure), but instead the JVP rule "
            "output's first element had container/pytree structure:\n"
            "    (float32[], float32[])\n"
            "while the custom_jvp-decorated function f had output "
            "container/pytree structure:\n"
            "    float32[]."
        ),
        lambda: jax.jvp(lambda x: scan_apply(f, x), (x,), (x,)))

    @f.defjvp
    def f_jvp2(primals, tangents):
      (x,), (xdot,) = primals, tangents
      return jnp.zeros((3, *x.shape), x.dtype), xdot

    self.assertRaisesRegex(
        TypeError,
        re.escape(
            "Custom JVP rule f_jvp2 for function f must produce a pair "
            "(list or tuple of length two) where the first element represents "
            "the primal output (equal in value to the output of the "
            "custom_jvp-decorated function f, and in particular "
            "with leaves of the same shape/dtype), but instead the JVP rule "
            "output's first element had shapes/dtypes of:\n"
            "    float32[3]\n"
            "while the custom_jvp-decorated function f had output shapes/dtypes"
            " of:\n"
            "    float32[]"
        ),
        lambda: jax.jvp(lambda x: scan_apply(f, x), (x,), (x,)))

  def test_multiple_rule_invocations(self):
    @jax.custom_jvp
    def expit(x):
      return 1 / (1 + lax.exp(-x))

    @expit.defjvp
    def _expit_jvp(primals, tangents):
      (x,), (t,) = primals, tangents
      ans = expit(x)
      t_out = t * ans * (1 - ans)
      return ans, t_out

    def scanned_fun(c, _):
      return [expit(c[0])] + [c[i-1] + c[i] for i in range(1, len(c))], None

    def foo(x):
      zero = jnp.zeros_like(x)
      c, _ = lax.scan(scanned_fun, [x, zero, zero, zero, zero], None, length=10)
      return c[-1]

    # just make sure these don't crash
    foo(3.)
    grad(foo)(3.)
    grad(lambda x: jax.vmap(foo)(x).sum())(jnp.arange(3.))

  def test_hard_stuff(self):
    arr = jnp.ones((5, 2, 2))
    api.jit(jax.vmap(jnp.linalg.det))(arr)  # doesn't crash

  def test_hard_stuff2(self):
    @jax.custom_jvp
    def f(x):
      return np.zeros(x.shape, x.dtype)

    @f.defjvp
    def f_jvp(primals, tangents):
      x, = primals
      t, = tangents
      return f(x), t

    # don't crash
    jax.jit(jax.vmap(f))(jnp.arange(3.))
    jax.jit(jax.vmap(jax.grad(f)))(jnp.arange(3.))
    jax.jit(jax.grad(lambda x: jax.vmap(f)(x).sum()))(jnp.arange(3.))
    jax.grad(lambda x: jax.vmap(f)(x).sum())(jnp.arange(3.))
    jax.jvp(jax.vmap(f), (jnp.arange(3.),), (jnp.ones(3),))

  def test_hard_stuff3(self):
    @jax.custom_jvp
    def relu(x):
      return jnp.maximum(x, 0)

    @relu.defjvp
    def _relu_jvp(primals, tangents):
      x, = primals
      t, = tangents
      return relu(x), lax.select(x > 0, t, lax.full_like(t, 0))

    def scanned_fun(c, _):
      return [relu(c[0])] + [c[i-1] + c[i] for i in range(1, len(c))], None

    def f(x):
      zero = jnp.zeros_like(x)
      c, _ = lax.scan(scanned_fun, [x, zero, zero, zero, zero], None, length=10)
      return c[-1]

    # don't crash
    jax.jit(jax.vmap(f))(jnp.arange(3.))
    jax.jit(jax.vmap(jax.grad(f)))(jnp.arange(3.))
    jax.jit(jax.grad(lambda x: jax.vmap(f)(x).sum()))(jnp.arange(3.))
    jax.grad(lambda x: jax.vmap(f)(x).sum())(jnp.arange(3.))
    jax.jvp(jax.jit(jax.vmap(f)), (jnp.arange(3.),), (jnp.ones(3),))

  def test_eval_shape(self):
    @jax.custom_jvp
    def expit(x):
      return 1 / (1 + lax.exp(-x))

    @expit.defjvp
    def _expit_jvp(primals, tangents):
      (x,), (t,) = primals, tangents
      ans = expit(x)
      t_out = t * ans * (1 - ans)
      return ans, t_out

    # don't crash
    api.eval_shape(expit, jnp.ones((2, 3)))
    api.eval_shape(api.grad(lambda x: expit(x).sum()), jnp.ones((2, 3)))

  def test_jaxpr_zeros(self):
    # from https://github.com/jax-ml/jax/issues/2657
    @jax.custom_jvp
    def f(A, b):
      return A @ b

    def f_jvp(primals, tangents):
      A, b = primals
      dA, db = tangents
      z = f(A, b)
      dz = A @ db + dA @ b
      return z, dz

    f.defjvp(f_jvp)

    def experiment(theta):
      def step(q, _):
        z = f(jnp.eye(3), jnp.ones(3) * theta)
        q += z[0]
        return q, q

      q = 0.
      q, _ = lax.scan(step, q, None, 4)
      return q

    grad(experiment)(1.)  # doesn't crash

  def test_linear_in_scan(self):
    @jax.custom_jvp
    def f(x):
      return -x

    @f.defjvp
    def f_jvp(primals, tangents):
      x, = primals
      x_dot, = tangents
      return f(x), f(x_dot)

    def foo(x):
      out, _  = lax.scan(lambda c, _: (f(c), None), x, None, length=1)
      return out

    ans = api.grad(foo)(3.)
    expected = -1.
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_custom_jvps_first_rule_is_none(self):
    # https://github.com/jax-ml/jax/issues/3389
    @jax.custom_jvp
    def f(x, y):
      return x ** 2 * y

    f.defjvps(None, lambda x_dot, primal_out, x, y: 2 * x * y * x_dot)
    ans = grad(f, 1)(2., 3.)  # doesn't crash
    expected = 12.
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_concurrent_initial_style(self):
    # https://github.com/jax-ml/jax/issues/3843
    def unroll(param, sequence):
      def scan_f(prev_state, inputs):
        return prev_state, jax.nn.sigmoid(param * inputs)
      return jnp.sum(jax.lax.scan(scan_f, None, sequence)[1])

    def run():
      return jax.grad(unroll)(jnp.array(1.0), jnp.array([1.0]))

    expected = run()

    # we just don't want this to crash
    n_workers = 2
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as e:
      futures = []
      for _ in range(n_workers):
        futures.append(e.submit(run))
      results = [f.result() for f in futures]
    for ans in results:
      self.assertAllClose(ans, expected)

  def test_nondiff_argnums_vmap_tracer(self):
    # https://github.com/jax-ml/jax/issues/3964
    @partial(jax.custom_jvp, nondiff_argnums=(0, 2))
    def sample(shape, param, seed):
      return jax.random.uniform(key=seed, shape=shape, minval=param)

    @sample.defjvp
    def sample_jvp(shape, seed, primals, tangents):
      param, = primals
      dparam, = tangents
      dparam = jnp.broadcast_to(dparam, shape)
      samples = sample(shape, param, seed)
      return samples, samples * dparam  # dummy jvp for proof of concept

    # check these don't crash
    jax.vmap(lambda seed: sample((2,3), 1., seed))(
        jax.random.split(jax.random.key(1), 10))
    jax.jvp(lambda x: sample((2, 3), x, jax.random.key(1)),
            (1.,), (1.,))

  def test_fun_with_nested_calls_2(self):
    def call(f, *args):
      f = jax.custom_jvp(f)
      f.defjvp(lambda primals, tangents: (f(*primals), sum(tangents)))
      return f(*args)

    def fun_with_nested_calls_2(x):
      def bar(y):
        def baz(w):
          q = call(lambda x: y, x)
          q = q + call(lambda: y)
          q = q + call(lambda y: w + y, y)
          q = call(lambda w: call(jnp.sin, x) * y, 1.0) + q
          return q
        return api.jit(baz)(x)
      return call(bar, x)

    # test these don't crash
    self.assertAllClose(api.jit(fun_with_nested_calls_2)(3.),
                        fun_with_nested_calls_2(3.))
    api.vmap(fun_with_nested_calls_2)(jnp.arange(3.))

  def test_closure_with_vmap(self):
    # https://github.com/jax-ml/jax/issues/3822
    alpha = np.float32(2.)

    def sample(seed):
      @jax.custom_jvp
      def f(alpha):
        return jax.random.gamma(seed, alpha, shape=[])

      @f.defjvp
      def f_jvp(primal, tangent):
        alpha = primal
        dalpha = tangent
        sample = f(alpha)
        partial_alpha = lax.random_gamma_grad(alpha, sample)
        return sample, partial_alpha * dalpha
      return f(alpha)

    api.vmap(sample)(jax.random.split(jax.random.key(1), 3))  # don't crash

  def test_closure_with_vmap2(self):
    # https://github.com/jax-ml/jax/issues/8783
    def h(z):
      def f(x):
        @jax.custom_jvp
        def g(y):
          return x * y

        # NOTE: rule closes over vmap tracer
        @g.defjvp
        def g_jvp(primals, tangents):
          (y,), (ydot,) = primals, tangents
          return x * y, x * ydot

        return g(z)  # NOTE: no vmapped arg

      return jax.vmap(f)(jnp.arange(3., dtype='float32'))

    primals, tangents = jax.jvp(h, (jnp.float32(1.),), (jnp.float32(2.),))
    self.assertAllClose(primals ,     jnp.arange(3., dtype='float32'))
    self.assertAllClose(tangents, 2 * jnp.arange(3., dtype='float32'))

  def test_float0(self):
    scalar_float0 = jnp.zeros((), dtype=float0)
    @jax.custom_jvp
    def f(x, y):
      return x, y
    def f_jvp(primals, _):
      x, y = primals
      return (x, y), (2., jax.custom_derivatives.zero_from_primal(y))
    f.defjvp(f_jvp)

    primals = (2., 3)
    tangents = (np.ones(()), scalar_float0)
    expected_tangents = (2., scalar_float0)
    self.assertAllClose(api.jvp(f, primals, tangents),
                        (primals, expected_tangents))

  def test_float0_initial_style(self):
    scalar_float0 = jnp.zeros((), dtype=float0)
    @jax.custom_jvp
    def f(x, y):
      return x, y
    def f_jvp(primals, _):
      x, y = primals
      return (x, y), (2., jax.custom_derivatives.zero_from_primal(y))
    f.defjvp(f_jvp)

    def foo(x, y):
      out, _ = lax.scan(lambda c, _: (f(*c), None), (x, y), None, length=1)
      return out

    primals = (2., 3)
    tangents = (np.ones(()), scalar_float0)
    expected_tangents = (2., scalar_float0)

    self.assertAllClose(api.jvp(foo, primals, tangents),
                        (primals, expected_tangents))

  def test_remat(self):
    @jax.custom_jvp
    def f(x):
      return jnp.sin(x)
    def f_jvp(primals, tangents):
      x, = primals
      g, = tangents
      return f(x), 2 * jnp.cos(x) * g
    f.defjvp(f_jvp)

    @jax.remat
    def g(x):
      return f(f(x))

    ans = g(2.)
    expected = np.sin(np.sin(2.))
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(g)(2.)
    expected = 4. * api.grad(lambda x: jnp.sin(jnp.sin(x)))(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_remat_higher_order(self):
    @jax.custom_jvp
    def f(x):
      return jnp.sin(x)
    def f_jvp(primals, tangents):
      x, = primals
      g, = tangents
      return f(x), 2 * jnp.cos(x) * g
    f.defjvp(f_jvp)

    def g(x):
      return f(f(x))

    ans = api.grad(api.grad(new_checkpoint(g)))(2.)
    expected = api.grad(api.grad(g))(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(new_checkpoint(api.grad(g)))(2.)
    expected = api.grad(api.grad(g))(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(api.grad(api.grad(new_checkpoint(g))))(2.)
    expected = api.grad(api.grad(api.grad(g)))(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_initial_style_vmap_2(self):
    # This is like test_initial_style_vmap except the primal function closes
    # over an array constant.
    y = jnp.arange(1., 4.)

    @jax.custom_jvp
    def f(x):
      assert jnp.ndim(x) == 0
      return 3 * x * jnp.sum(y)
    def f_jvp(primals, tangents):
      x, = primals
      g, = tangents
      return f(x), 2 * g
    f.defjvp(f_jvp)

    def foo(x):
      out, _  = lax.scan(lambda c, _: (f(c), None), x, None, length=1)
      return out

    ans = api.grad(lambda x: api.vmap(foo)(x).sum())(jnp.ones(3))
    expected = 2. * jnp.ones(3)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(lambda x: api.vmap(api.jit(foo))(x).sum())(jnp.ones(3))
    expected = 2. * jnp.ones(3)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(lambda x: api.jit(api.vmap(foo))(x).sum())(jnp.ones(3))
    expected = 2. * jnp.ones(3)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(api.jit(lambda x: api.vmap(foo)(x).sum()))(jnp.ones(3))
    expected = 2. * jnp.ones(3)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.jit(api.grad(lambda x: api.vmap(foo)(x).sum()))(jnp.ones(3))
    expected = 2. * jnp.ones(3)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_custom_jvp_vmap_broadcasting_interaction(self):
    # https://github.com/jax-ml/jax/issues/6452
    def f2(y, z):
      v1 = z
      v2 = jnp.sum(y) + z
      return jnp.logaddexp(v1, v2)

    def f1(y, z):
      v = api.vmap(lambda _y: f2(_y, z))(y)
      return jnp.sum(v)

    y = jnp.ones((3, 2))
    f = lambda z: f1(y, z)
    z = 0.1
    val, g = api.value_and_grad(f)(z)
    self.assertEqual(val.shape, ())
    self.assertEqual(g.shape, ())

  def test_custom_jvp_vmap_broadcasting_interaction_2(self):
    # https://github.com/jax-ml/jax/issues/5849
    @jax.custom_jvp
    def transform(box, R):
      if jnp.isscalar(box) or box.size == 1:
        return R * box
      elif box.ndim == 2:
        return jnp.einsum('ij,j->i', box, R)
      raise ValueError()

    @transform.defjvp
    def transform_jvp(primals, tangents):
      box, R = primals
      dbox, dR = tangents
      return (transform(box, R), dR + transform(dbox, R))

    def periodic_general(box):
      def displacement_fn(Ra, Rb, **kwargs):
        _box = kwargs.get('box', box)
        return transform(_box, Ra - Rb)

      return displacement_fn

    N = 250

    scalar_box = 1.0
    displacement = periodic_general(scalar_box)

    key = jax.random.key(0)
    R = jax.random.uniform(key, (N, 2))

    def energy_fn(box):
      d = partial(displacement, box=box)
      d = api.vmap(api.vmap(d, (None, 0)), (0, None))
      return jnp.sum(d(R, R) ** 2)

    self.assertEqual(grad(energy_fn)(scalar_box).shape, ())

  def test_custom_jvp_implicit_broadcasting(self):
    # https://github.com/jax-ml/jax/issues/6357
    if config.enable_x64.value:
      raise unittest.SkipTest("test only applies when x64 is disabled")

    @jax.custom_jvp
    def projection_unit_simplex(x: jax.Array) -> jax.Array:
      """Projection onto the unit simplex."""
      s = 1.0
      n_features = x.shape[0]
      u = jnp.sort(x)[::-1]
      cssv = jnp.cumsum(u) - s
      ind = jnp.arange(n_features, dtype=x.dtype) + 1
      cond = u - cssv / ind > 0
      idx = jnp.count_nonzero(cond)
      threshold = cssv[idx - 1] / idx.astype(x.dtype)
      return jax.nn.relu(x - threshold)


    @projection_unit_simplex.defjvp
    def projection_unit_simplex_jvp(primals, tangents):
      x, = primals
      x_dot, = tangents
      primal_out = projection_unit_simplex(x)
      supp = (primal_out > 0).astype(x_dot.dtype)
      card = jnp.count_nonzero(supp).astype(x_dot.dtype)
      tangent_out = supp * x_dot - (jnp.dot(supp, x_dot) / card) * supp
      return primal_out, tangent_out

    rng = self.rng()
    x = rng.rand(5).astype(np.float32)

    J_rev = jax.jacrev(projection_unit_simplex)(x)
    J_fwd = jax.jacfwd(projection_unit_simplex)(x)

    p = projection_unit_simplex(x)
    support = (p > 0).astype(jnp.float32)
    cardinality = jnp.count_nonzero(support).astype(support.dtype)
    J_true = jnp.diag(support) - jnp.outer(support, support) / cardinality
    self.assertAllClose(J_true, J_fwd)
    self.assertAllClose(J_true, J_rev)

    proj = jax.vmap(projection_unit_simplex)

    def fun(X):
      return jnp.sum(proj(X) ** 2)

    rng = self.rng()
    X = rng.rand(4, 5).astype(np.float32)
    U = rng.rand(4, 5)
    U /= np.sqrt(np.sum(U ** 2))
    U = U.astype(np.float32)

    eps = 1e-3
    dir_deriv_num = (fun(X + eps * U) - fun(X - eps * U)) / (2 * eps)
    dir_deriv = jnp.vdot(jax.grad(fun)(X), U)
    self.assertAllClose(dir_deriv, dir_deriv_num, atol=1e-3)

  def test_vmap_inside_defjvp(self):
    # https://github.com/jax-ml/jax/issues/3201
    seed = 47
    key = jax.random.key(seed)
    mat = jax.random.normal(key, (2, 3))

    @jax.custom_jvp
    def f(mat, aux):
      num_rows, num_cols = mat.shape
      return jnp.ones((num_rows, 1)) / num_cols

    @f.defjvp
    def f_jvp(primals, tangents):
      mat, aux = primals
      vec, _ = tangents
      output = f(*primals)
      num_rows, num_cols = mat.shape
      size = num_rows * num_cols
      # -----
      bd_mat = mat.reshape(1, 1, num_rows, num_cols)
      bd_mat = jnp.tile(bd_mat, reps=(num_rows, num_cols))
      bd_mat = bd_mat.reshape(size, num_rows, num_cols)
      # -----
      rowsum = jnp.sum(mat, axis=1, keepdims=True)
      colsum = jnp.sum(mat, axis=0, keepdims=True)
      bd_rowsum = jnp.tile(rowsum, reps=(1, num_rows))
      bd_colsum = jnp.tile(colsum, reps=(num_cols, 1))
      # -----
      bd_vec = vec.reshape(size, 1)
      # -----
      def operate(mx, val):
        buf = 0
        for i in range(2):
          buf = buf + jnp.matmul(mx, bd_colsum) / jnp.power(aux, i)
        buf = jnp.matmul(bd_rowsum, buf)
        return buf * val[None, :]
      # -----
      # Vertorizing will raise shape error
      bd_buf = jax.vmap(operate, in_axes=(0, 0), out_axes=0)(bd_mat, bd_vec)
      # -----
      bd_buf = bd_buf / aux
      jvp = jnp.sum(bd_buf, axis=0)
      jvp = jnp.mean(jvp, axis=1, keepdims=True)
      # -----
      # JVP ends successfully, but still raise an error
      return (output, jvp)

    jax.grad(lambda mat, aux: jnp.sum(f(mat, aux)))(mat, 0.5)  # doesn't crash

  def test_custom_jvp_unbroadcasting(self):
    # https://github.com/jax-ml/jax/issues/3056
    a = jnp.array([1., 1.])

    @jax.custom_jvp
    def f(x):
      return a * x

    @f.defjvp
    def f_jvp(primals, tangents):
      x, = primals
      dx, = tangents
      return a * x, a * dx

    shape = grad(lambda x: jnp.sum(f(x)))(jnp.array(1.)).shape
    self.assertEqual(shape, ())

  def test_maybe_perturbed_internal_helper_function(self):
    # This is a unit test for an internal API. We include it so as not to
    # regress https://github.com/jax-ml/jax/issues/9567. For an explanation of
    # this helper function, see https://github.com/jax-ml/jax/issues/6415.
    def f(x):
      def g(y, _):
        z = y * x
        self.assertTrue(custom_derivatives._maybe_perturbed(z))
        return y, None
      g(1, None)
      return lax.scan(g, 1, xs=None, length=1)[0]

    jax.jvp(f, (1.0,), (1.0,))  # assertions inside f

  def test_maybe_perturbed_int_regression(self):
    # see https://github.com/jax-ml/jax/discussions/9951

    @jax.jit
    def f():
      x = jnp.array(1)
      _, aux_args = custom_derivatives.closure_convert(lambda: x)
      self.assertEmpty(aux_args)
    f()

  def test_sinc_constant_function_batching(self):
    # https://github.com/jax-ml/jax/pull/10756
    batch_data = jnp.arange(15.).reshape(5, 3)

    @jax.vmap
    def f(x):
      return jax.lax.map(jnp.sinc, x)
    g = lambda param: f(param * batch_data).sum()

    @jax.vmap
    def f_ref(x):
      return jnp.stack([jnp.sinc(x_) for x_ in x])
    g_ref = lambda param: f_ref(param * batch_data).sum()

    grad     = jax.grad(g    )(0.1)  # doesn't crash
    grad_ref = jax.grad(g_ref)(0.1)
    self.assertAllClose(grad, grad_ref, check_dtypes=False)

  @parameterized.named_parameters(
      ('jit_vmap', True, True),
      ('jit', True, False),
      ('vmap', False, True),
      ('', False, False),
  )
  def test_symbolic_zero_custom_jvp(self, maybe_jit, maybe_vmap):
    def f(static_scalar, static_array, dyn_scalar, dyn_array):
      out1 = static_scalar + dyn_scalar
      out2 = static_array + dyn_array
      return out1, out2

    def _pack(x):
      return lax.broadcast(x, (1,))

    def _unpack(x):
      (x,) = x
      return x

    def _vmap(fun):
      def _fun(*args):
        args = jax.tree.map(_pack, args)
        out = jax.vmap(fun)(*args)
        out = jax.tree.map(_unpack, out)
        return out
      return _fun

    f = jax.custom_jvp(f)

    @partial(f.defjvp, symbolic_zeros=True)
    def f_jvp(primals, tangents):
      static_scalar, *_ = primals
      t_static, t_static_arr, t_dyn_scalar, t_dyn_array = tangents
      self.assertIs(type(t_static)    , jax.custom_derivatives.SymbolicZero)
      self.assertIs(type(t_static_arr), jax.custom_derivatives.SymbolicZero)
      self.assertEqual(t_static.shape, ())
      self.assertEqual(t_static_arr.shape, (2,))
      return f(*primals), (static_scalar + 90, t_dyn_array + 91)

    def g(dyn_scalar, dyn_array):
      if maybe_vmap:
        f_ = _vmap(f)
      else:
        f_ = f
      return f_(1., jnp.array([2., 3.]), dyn_scalar, dyn_array)

    def run(primal_ins, tangent_ins):
      return jax.jvp(g, primal_ins, tangent_ins)

    if maybe_jit:
      run = jax.jit(run)

    primal_ins = (4., jnp.array([5., 6.]))
    tangent_ins = (7., jnp.array([8., 9.]))
    primal_outs, tangent_outs = run(primal_ins, tangent_ins)
    primal_out1, primal_out2 = primal_outs
    tangent_out1, tangent_out2 = tangent_outs
    scalar_type = jax.Array if maybe_jit or maybe_vmap else float
    self.assertIsInstance(primal_out1, scalar_type)
    self.assertAllClose(primal_out1, 5.)
    self.assertIsInstance(tangent_out1, scalar_type)
    self.assertAllClose(tangent_out1, 91.)
    self.assertIsInstance(primal_out2, jax.Array)
    self.assertArraysAllClose(primal_out2, jnp.array([7., 9.]))
    self.assertIsInstance(tangent_out2, jax.Array)
    self.assertArraysAllClose(tangent_out2, jnp.array([99., 100.]))

  def test_symbolic_zero_custom_jvp_vmap_output(self):
    @jax.custom_jvp
    def f(x, y):
      return x * y

    @partial(f.defjvp, symbolic_zeros=True)
    def f_jvp(primals, tangents):
      x, y = primals
      x_dot, y_dot = tangents
      self.assertIs(type(y_dot), jax.custom_derivatives.SymbolicZero)
      return f(x, y), y_dot

    jax.grad(lambda x, y: jax.vmap(f)(x, y).sum())(jnp.ones(3), jnp.ones(3))

  def test_symbolic_zeros_memoization_caching(self):
    # Tests multiple zero patterns for partial_eval._memoize, and also tests
    # that we're okay with stores being occupied with equal values.

    @jax.custom_jvp
    def f(x, y):
      return x * y

    @partial(f.defjvp, symbolic_zeros=True)
    def f_jvp(primals, tangents):
      x, y = primals
      x_dot, y_dot = tangents
      return f(x, y), y_dot

    f_ = core.jaxpr_as_fun(jax.make_jaxpr(f)(2., 3.))
    _ = jax.linearize(f_, 2., 3.)
    _ = jax.linearize(lambda x: f_(x, 3.), 2.)  # don't crash!

  def test_symbolic_zeros_under_jit(self):
    # https://github.com/jax-ml/jax/issues/14833
    Zero = jax.custom_derivatives.SymbolicZero

    @jax.custom_jvp
    def f(x, y):
        return x * y

    @partial(f.defjvp, symbolic_zeros=True)
    def fjvp(primals, tangents):
        x, y = primals
        tx, ty = tangents
        assert type(tx) is not Zero or type(ty) is not Zero
        return f(x, y), (
            ty if type(tx) is Zero else
            tx if type(ty) is Zero else
            tx + ty)

    jax.jacfwd(jax.jit(f))(0.1, 0.2)  # don't crash

  def test_custom_jvp_functools_partial(self):
    def fun(x, y, a):
      return x + y * a

    fun_wrapped = functools.partial(fun, a = 0.1)

    def jvp_fn(primals, tangents):
      return jax.jvp(fun_wrapped, primals, tangents)

    fn = jax.custom_jvp(fun_wrapped)
    fn.defjvp(jvp_fn)

    self.assertEqual((1.0, 0.1), jax.grad(lambda args: fn(*args))((1.0, 2.0)))

  def test_run_rules_more_than_once(self):
    # https://github.com/jax-ml/jax/issues/16614

    @jax.custom_jvp
    def f(x, y):
      return x

    @partial(f.defjvp, symbolic_zeros=True)
    def f_jvp(primals, tangents):
      x, _ = primals
      x_dot, _ = tangents
      return x, x_dot

    def body(x_y, _):
      x, y = x_y
      return (f(x, y), x), None

    @jax.grad
    def g(x):
      (out, _), _ = lax.scan(body, (x, 1.), xs=None, length=2)
      return out

    g(1.)  # doesn't crash

  def test_dce(self):
    @jax.custom_jvp
    def f(x, y):
      return jnp.sin(x), x + jnp.cos(y)

    @f.defjvp
    def f_jvp(primals, tangents):
      x, y = primals
      dx, dy = tangents
      return f(x, y), (2.0 * jnp.cos(x) * dx, 1.5 * dx - 0.5 * jnp.sin(y) * dy)

    def check_jaxpr(jaxpr, used_outs, includes, excludes):
      dce_jaxpr, _ = pe.dce_jaxpr(jaxpr, used_outs)
      if not dce_jaxpr.eqns:
        assert not includes
        return
      call_jaxpr = dce_jaxpr.eqns[0].params["call_jaxpr"]
      for prim in includes:
        assert any(eqn.primitive == prim for eqn in call_jaxpr.eqns)
      for prim in excludes:
        assert all(eqn.primitive != prim for eqn in call_jaxpr.eqns)

    x, y = 0.1, -1.3
    jaxpr = jax.make_jaxpr(f)(x, y).jaxpr
    check_jaxpr(jaxpr, [True, True], [lax.sin_p, lax.cos_p], [])
    check_jaxpr(jaxpr, [True, False], [lax.sin_p], [lax.cos_p])
    check_jaxpr(jaxpr, [False, True], [lax.cos_p], [lax.sin_p])
    check_jaxpr(jaxpr, [False, False], [], [lax.sin_p, lax.cos_p])

    def dce_jaxpr_as_fun(jaxpr, used_outs):
      jaxpr_, _ = pe.dce_jaxpr(jaxpr, used_outs)
      fun = core.jaxpr_as_fun(pe.close_jaxpr(jaxpr_))
      return lambda *args: fun(*args)[0]

    f0 = dce_jaxpr_as_fun(jaxpr, [True, False])
    f1 = dce_jaxpr_as_fun(jaxpr, [False, True])
    self.assertAllClose(
        api.jvp(f0, (x, y), (1.0, 0.0)), (f0(x, y), 2.0 * jnp.cos(x)))
    self.assertAllClose(
        api.jvp(f0, (x, y), (0.0, 1.0)), (f0(x, y), 0.0))
    self.assertAllClose(
        api.jvp(f1, (x, y), (1.0, 0.0)), (f1(x, y), 1.5))
    self.assertAllClose(
        api.jvp(f1, (x, y), (0.0, 1.0)), (f1(x, y), -0.5 * jnp.sin(y)))

  def test_resolve_kwargs_error_message(self):
    @jax.custom_jvp
    def f(x, y, *, z=None):
      return jnp.sin(x), x + jnp.cos(y)

    @f.defjvp
    def f_jvp(primals, tangents):
      self.fail("should not be executed")

    with self.assertRaisesRegex(
        TypeError,
        r"The input arguments to the custom_jvp-decorated function f(.*)\n"
        r"missing a required argument: 'y'"
    ):
      f(0.5)

    with self.assertRaisesRegex(
        TypeError,
        r"The input arguments to the custom_jvp-decorated function f(.*)\n"
        "The following keyword arguments could not be resolved to positions: z"
    ):
      f(0.5, 0.1, z=1.0)

  def test_symbolic_zero_custom_jvp_vmap_doesnt_instantiate(self):
    @jax.custom_jvp
    def f(x, y):
      return y

    def f_jvp(primals, tangents):
      (x, y), (x_dot, y_dot) = primals, tangents
      assert type(y_dot) is jax.custom_derivatives.SymbolicZero
      return y, y_dot

    f.defjvp(f_jvp, symbolic_zeros=True)

    def g(x):
      return f(x, f(x, 1.))

    jax.jvp(jax.vmap(g), (jnp.ones(3),), (jnp.ones(3),))  # don't crash

  def test_symbolic_zero_under_vmap_of_jit(self):
    # https://github.com/jax-ml/jax/issues/28144
    @jax.custom_jvp
    def f(x):
        return x + 1

    @f.defjvp
    def f_jvp(x, t):
        (x,) = x
        (t,) = t
        z = jax.custom_derivatives.zero_from_primal(x, symbolic_zeros=True)
        return f(x), z

    x = jnp.arange(3.0)
    jax.jvp(jax.vmap(jax.jit(f)), (x,), (x,))  # doesn't crash

  def test_pretty_print(self):
    @jax.custom_jvp
    def f(x):
      return x + 1

    @f.defjvp
    def f_jvp(primals, tangents):
      return f(*primals), tangents[0]

    x = jnp.array([4.2], dtype=jnp.float32)
    jaxpr = jax.make_jaxpr(f)(x)
    actual = jaxpr.pretty_print(use_color=False)
    expected = textwrap.dedent(
        """
        { lambda ; a:f32[1]. let
            b:f32[1] = custom_jvp_call[
              name=f
              call_jaxpr={ lambda ; c:f32[1]. let d:f32[1] = add c 1.0:f32[] in (d,) }
              jvp=f_jvp
              symbolic_zeros=False
            ] a
          in (b,) }
        """).strip()
    self.assertEqual(actual, expected)



class CustomVJPTest(jtu.JaxTestCase):

  def test_basic(self):
    @jax.custom_vjp
    def f(x):
      return jnp.sin(x)
    def f_fwd(x):
      return f(x), jnp.cos(x)
    def f_rev(cos_x, g):
      return (2 * cos_x * g,)
    f.defvjp(f_fwd, f_rev)

    x = 3.
    self.assertAllClose(f(x), jnp.sin(x))
    self.assertAllClose(api.grad(f)(x), 2 * jnp.cos(x))
    self.assertAllClose(api.value_and_grad(f)(x),
                        (jnp.sin(x), 2 * jnp.cos(x)))

  def test_invariance(self):
    @jax.custom_vjp
    def f(x):
      return jnp.cos(2 * x) / 2.
    def f_fwd(x):
      return (f(x), x)
    def f_rev(x, g):
      return (g * 3,)
    f.defvjp(f_fwd, f_rev)
    def f2(x):
      y, _ = api.value_and_grad(f)(x)
      return y
    def f3(x):
      y, _ = api.value_and_grad(f2)(x)
      return y
    x = 1.
    self.assertAllClose(f(x), f2(x), check_dtypes=False)
    self.assertAllClose(f(x), f3(x), check_dtypes=False)
    self.assertAllClose(api.grad(f)(x), api.grad(f2)(x),
                        check_dtypes=False)
    self.assertAllClose(api.grad(f)(x), api.grad(f3)(x),
                        check_dtypes=False)

  def test_python_control_flow(self):
    @jax.custom_vjp
    def f(x):
      if x > 0:
        return jnp.sin(x)
      else:
        return jnp.cos(x)
    def f_fwd(x):
      if x > 0:
        return f(x), x
      else:
        return f(x), x
    def f_rev(x, g):
      if x > 0:
        return (2 * g,)
      else:
        return (3 * g,)
    f.defvjp(f_fwd, f_rev)
    x = 2.
    self.assertAllClose(f(x), jnp.sin(x))
    self.assertAllClose(f(-x), jnp.cos(-x))
    self.assertAllClose(api.value_and_grad(f)(x), (jnp.sin(x), 2.),
                        check_dtypes=False)
    self.assertAllClose(api.value_and_grad(f)(-x), (jnp.cos(-x), 3.),
                        check_dtypes=False)

  def test_vmap(self):
    @jax.custom_vjp
    def f(x):
      assert jnp.ndim(x) == 0
      return jnp.sin(x)
    def f_fwd(x):
      assert jnp.ndim(x) == 0
      return f(x), jnp.cos(x)
    def f_rev(cos_x, g):
      return (2 * cos_x * g,)
    f.defvjp(f_fwd, f_rev)

    x = jnp.arange(3.)
    xx = jnp.arange(6.).reshape(2, 3)

    # vmap of f
    self.assertAllClose(api.vmap(f)(x), jnp.sin(x))
    self.assertAllClose(api.vmap(api.vmap(f))(xx), jnp.sin(xx))

    # vmap of grad of f
    self.assertAllClose(api.vmap(api.grad(f))(x), 2 * jnp.cos(x))
    self.assertAllClose(api.vmap(api.value_and_grad(f))(x),
                        (jnp.sin(x), 2 * jnp.cos(x)))
    self.assertAllClose(api.vmap(api.vmap(api.grad(f)))(xx), 2 * jnp.cos(xx))
    self.assertAllClose(api.vmap(api.vmap(api.value_and_grad(f)))(xx),
                        (jnp.sin(xx), 2 * jnp.cos(xx)))

    # grad of vmap of f
    self.assertAllClose(api.grad(lambda x: api.vmap(f)(x).sum())(x),
                        2 * jnp.cos(x))
    self.assertAllClose(api.grad(lambda x: api.vmap(api.vmap(f))(x).sum())(xx),
                        2 * jnp.cos(xx))

    # vmap of grad of vmap of f
    self.assertAllClose(api.vmap(api.grad(lambda x: api.vmap(f)(x).sum()))(xx),
                        2 * jnp.cos(xx))

  def test_jit(self):
    @jax.custom_vjp
    def f(x):
      return jnp.sin(x)
    def f_fwd(x):
      return f(x), jnp.cos(x)
    def f_rev(cos_x, g):
      return (2 * cos_x * g,)
    f.defvjp(f_fwd, f_rev)

    x = 3.

    # jit
    self.assertAllClose(api.jit(f)(x), jnp.sin(x))
    self.assertAllClose(api.jit(api.jit(f))(x), jnp.sin(x))

    # jit of grad
    self.assertAllClose(api.jit(api.grad(f))(x), 2 * jnp.cos(x),
                        check_dtypes=False)

    # grad of jit
    self.assertAllClose(api.grad(api.jit(f))(x), 2 * jnp.cos(x),
                        check_dtypes=False)

  def test_pytrees(self):
    @jax.custom_vjp
    def f(x):
      return {'b': jnp.sin(x['a'])}
    def f_fwd(x):
      return f(x), {'r': jnp.cos(x['a'])}
    def f_bwd(res, g):
      cos_x = res['r']
      return ({'a': 2 * cos_x * g['b']},)
    f.defvjp(f_fwd, f_bwd)
    x = {'a': 3.}
    self.assertAllClose(f(x)['b'], jnp.sin(x['a']))
    self.assertAllClose(api.grad(lambda x: f(x)['b'])(x),
                        {'a': 2 * jnp.cos(x['a'])})

  def test_jvp_error(self):
    @jax.custom_vjp
    def f(x):
      return jnp.sin(x)
    def f_fwd(x):
      return f(x), jnp.cos(x)
    def f_rev(cos_x, g):
      return (2 * cos_x * g,)
    f.defvjp(f_fwd, f_rev)

    self.assertRaisesRegex(
        TypeError,
        r"can't apply forward-mode autodiff \(jvp\) to a custom_vjp function.",
        lambda: api.jvp(f, (3.,), (1.,)))
    self.assertRaisesRegex(
        TypeError,
        r"can't apply forward-mode autodiff \(jvp\) to a custom_vjp function.",
        lambda: api.jvp(api.vmap(f), (jnp.arange(3.),), (jnp.ones(3),)))
    self.assertRaisesRegex(
        TypeError,
        r"can't apply forward-mode autodiff \(jvp\) to a custom_vjp function.",
        lambda: api.jvp(jit(f), (3.,), (1.,)))

  def test_kwargs(self):
    # from https://github.com/jax-ml/jax/issues/1938
    @jax.custom_vjp
    def my_fun(x, y, c=1.):
      return c * (x + y)
    my_fun.defvjp(lambda x, y, c=1.: (my_fun(c, y, c), None),
                  lambda _, g: (g, g, g))
    f = lambda x, y: jnp.square(my_fun(x, y, c=2.)).sum()
    f(10., 5.)  # doesn't crash
    api.grad(f)(10., 5.)  # doesn't crash

  def test_initial_style(self):
    @jax.custom_vjp
    def f(x):
      return jnp.sin(x)
    def f_fwd(x):
      return f(x), jnp.cos(x)
    def f_rev(cos_x, g):
      return (2 * cos_x * g,)
    f.defvjp(f_fwd, f_rev)

    def foo(x):
      out, _  = lax.scan(lambda c, _: (f(c), None), x, None, length=1)
      return out

    ans = api.grad(foo)(3.)
    expected = 2. * jnp.cos(3.)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(api.grad(foo))(3.)
    expected = -2. * jnp.sin(3.)
    self.assertAllClose(ans, expected)

  def test_initial_style_vmap(self):
    @jax.custom_vjp
    def f(x):
      assert jnp.ndim(x) == 0
      return 3 * x
    def f_fwd(x):
      return f(x), jnp.cos(x)
    def f_rev(cos_x, g):
      return (2 * cos_x * g,)
    f.defvjp(f_fwd, f_rev)

    def foo(x):
      out, _  = lax.scan(lambda c, _: (f(c), None), x, None, length=1)
      return out

    ans = api.vmap(foo)(jnp.arange(3.))
    expected = 3. * jnp.arange(3.)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(lambda x: api.vmap(foo)(x).sum())(jnp.arange(3.))
    expected = 2. * jnp.cos(jnp.arange(3.))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_nondiff_argnums(self):
    @partial(jax.custom_vjp, nondiff_argnums=(0,))
    def app(f, x):
      return f(x)
    def app_fwd(f, x):
      return app(f, x), jnp.cos(x)
    def app_rev(f, cos_x, g):
      return (cos_x * g,)
    app.defvjp(app_fwd, app_rev)

    ans = app(lambda x: 2 * x, 1)
    expected = 2
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.value_and_grad(lambda x: app(lambda y: 2 * y, x))(1.)
    expected = (2., jnp.cos(1.))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_nondiff_argnames(self):
    @partial(jax.custom_vjp, nondiff_argnames=('f',))
    def app(f, x):
      return f(x)
    def app_fwd(f, x):
      return app(f, x), jnp.cos(x)
    def app_rev(f, cos_x, g):
      return (cos_x * g,)
    app.defvjp(app_fwd, app_rev)

    ans = app(lambda x: 2 * x, 1)
    expected = 2
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.value_and_grad(lambda x: app(lambda y: 2 * y, x))(1.)
    expected = (2., jnp.cos(1.))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_nondiff_argnums_argnames(self):
    @partial(jax.custom_vjp, nondiff_argnums=(0,), nondiff_argnames=('g',))
    def app(f, g, x):
      return f(x) + g(x)
    def app_fwd(f, g, x):
      return app(f, g, x), jnp.cos(x)
    def app_rev(f, g, cos_x, v):
      return (cos_x * v,)
    app.defvjp(app_fwd, app_rev)

    f = lambda x: 2 * x
    g = lambda x: 2 * x
    ans = app(f, g, 1)
    expected = 4
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.value_and_grad(lambda x: app(f, g, x))(1.)
    expected = (4., jnp.cos(1.))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_closed_over_jit_tracer(self):
    # See the comment in CustomJVPTest.test_nondiff_arg_jit_tracer.
    raise unittest.SkipTest("behavior no longer supported")

    # This test is similar to test_nondiff_arg_tracer except it uses lexical
    # closure rather than the nondiff_argnums mechanism. We decided to disallow
    # tracers in nondiff_argnums to greatly simplify bookkeeping while still
    # supporting the cases for which it is necessary.
    def outer(x):
      @jax.custom_vjp
      def f(y):
        return x * y
      def f_fwd(y):
        return f(y), jnp.cos(y)
      def f_rev(cos_y, g):
        return (cos_y * g,)
      f.defvjp(f_fwd, f_rev)
      return f

    @jit
    def g(x, y):
      return outer(x)(y)

    ans = g(2, 3.)
    expected = 6.
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(g, 1)(2., 3.)
    expected = jnp.cos(3.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_closed_over_vmap_tracer(self):
    def outer(x):
      @jax.custom_vjp
      def f(y):
        return x * y
      def f_fwd(y):
        return f(y), jnp.cos(y)
      def f_rev(cos_y, g):
        return (cos_y * g,)
      f.defvjp(f_fwd, f_rev)
      return f

    @api.vmap
    def g(x):
      return outer(x)(3.)

    ans = g(np.arange(3.))
    expected = np.arange(3.) * 3
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_closed_over_tracer3(self):
    def outer(x):
      @jax.custom_vjp
      def f(y):
        return x * y
      def f_fwd(y):
        return f(y), (x, jnp.cos(y))
      def f_rev(res, g):
        x, cos_y = res
        return (cos_y * g * x,)
      f.defvjp(f_fwd, f_rev)
      return api.grad(f)

    @api.vmap
    def g(x):
      return outer(x)(3.)

    ans = g(np.arange(3.))
    expected = np.cos(3.) * np.arange(3.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_nondiff_arg_tracer_error(self):
    # This is similar to the old (now skipped) test_nondiff_arg_tracer, except
    # we're testing for the error message that usage pattern now raises.

    @partial(jax.custom_vjp, nondiff_argnums=(0,))
    def f(x, y):
      return x * y
    def f_fwd(x, y):
      return f(x, y), jnp.cos(y)
    def f_rev(x, cos_y, g):
      return (cos_y * g,)
    f.defvjp(f_fwd, f_rev)

    @jit
    def g(x, y):
      return f(x, y)

    with self.assertRaisesRegex(UnexpectedTracerError, "custom_vjp"):
      _ = g(2, 3.)
    with self.assertRaisesRegex(UnexpectedTracerError, "custom_vjp"):
      _ = api.grad(g, 1)(2., 3.)

  def test_vmap_axes(self):
    raise unittest.SkipTest("TODO")  # TODO(mattjj): write test

  def test_pmap(self):
    raise unittest.SkipTest("TODO")  # TODO(mattjj): write test

  def test_missing_vjp_rule_error(self):
    @jax.custom_vjp
    def foo(x):
      return x ** 2

    self.assertRaisesRegex(
        AttributeError,
        r"No VJP defined for custom_vjp function foo using defvjp.",
        lambda: foo(2))
    self.assertRaisesRegex(
        AttributeError,
        r"No VJP defined for custom_vjp function foo using defvjp.",
        lambda: api.grad(foo)(2.))

  def test_vjp_rule_inconsistent_pytree_structures_error(self):
    @jax.custom_vjp
    def f(x):
      return x

    def foo_fwd(x):
      return x, None

    def foo_bwd(_, g):
      return (g, g)

    f.defvjp(foo_fwd, foo_bwd)

    f(2)  # doesn't crash
    self.assertRaisesRegex(
        TypeError,
        re.escape(
            "Custom VJP bwd rule must produce an output with the same container "
            "(pytree) structure as the args tuple of the primal function, "
            "and in particular must produce a tuple of length equal to the "
            "number of arguments to the primal function, but got bwd output "
            "structure {} for primal input structure {}.".format(
                jax.tree.structure((1, 1)),
                jax.tree.structure((1,)))
        ),
        lambda: api.grad(f)(2.))

  def test_vjp_bwd_returns_non_tuple_error(self):
    @jax.custom_vjp
    def f(x):
      return x

    def foo_fwd(x):
      return x, None

    def foo_bwd(_, g):
      return 2. * g  # Should be a tuple

    f.defvjp(foo_fwd, foo_bwd)
    with self.assertRaisesRegex(TypeError, "Custom VJP bwd rule .* must produce a tuple"):
      api.grad(f)(3.)

  def test_fwd_rule_primal_out_type_doesnt_match_primal_error_message(self):
    # https://github.com/lucidrains/flash-attention-jax/issues/7

    def scan_apply(f, x):
      y, _ = jax.lax.scan(lambda x, _: (f(x), None), x, None, length=1)
      return y

    @jax.custom_vjp
    def f(x):
      return x

    def f_fwd(x):
      return (x, x), None

    def f_bwd(_, y_bar):
      return (y_bar,)

    f.defvjp(f_fwd, f_bwd)

    self.assertRaisesRegex(
        TypeError,
        re.escape(
            "Custom VJP fwd rule f_fwd for function f must produce a pair "
            "(list or tuple of length two) where the first element represents "
            "the primal output (equal to the output of the "
            "custom_vjp-decorated function f) and the second element "
            "represents residuals (i.e. values stored from the forward "
            "pass for use on the backward pass), but instead the fwd rule "
            "output's first element had container/pytree structure:\n"
            "    (float32[], float32[])\n"
            "while the custom_vjp-decorated function f had output "
            "container/pytree structure:\n"
            "    float32[]."
        ),
        lambda: jax.grad(lambda x: scan_apply(f, x))(jnp.float32(1.)))

    def f_fwd2(x):
      return jnp.zeros((3, *x.shape), x.dtype), None

    def f_bwd2(_, y_bar):
      return (y_bar,)

    f.defvjp(f_fwd2, f_bwd2)

    self.assertRaisesRegex(
        TypeError,
        re.escape(
            "Custom VJP fwd rule f_fwd2 for function f must produce a pair "
            "(list or tuple of length two) where the first element represents "
            "the primal output (equal to the output of the "
            "custom_vjp-decorated function f) and the second element "
            "represents residuals (i.e. values stored from the forward "
            "pass for use on the backward pass), but instead the fwd rule "
            "output's first element had shapes/dtypes of:\n"
            "    float32[3]\n"
            "while the custom_vjp-decorated function f had output "
            "shapes/dtypes of:\n"
            "    float32[]"
        ),
        lambda: jax.grad(lambda x: scan_apply(f, x))(jnp.float32(1.)))

  def test_issue2511(self):
    arr = jnp.ones((5, 2, 2))
    foo = lambda x: api.vmap(jnp.linalg.det, (0,))(x)
    api.jit(foo)(arr)  # doesn't crash

  def test_lowering_out_of_traces(self):
    # https://github.com/jax-ml/jax/issues/2578

    class F(collections.namedtuple("F", ["a"])):
      def __call__(self, x):
        return jax.nn.relu(self.a) * x

    @jax.jit
    def g(f, x):
      return f(x)

    jax.grad(g, argnums=(1,))(F(2.0), 0.)  # doesn't crash

  def test_clip_gradient(self):
    # https://github.com/jax-ml/jax/issues/2784
    @jax.custom_vjp
    def _clip_gradient(lo, hi, x):
      return x  # identity function when not differentiating

    def clip_gradient_fwd(lo, hi, x):
      return x, (lo, hi,)

    def clip_gradient_bwd(res, g):
      lo, hi = res
      return (None, None, jnp.clip(g, lo, hi),)

    _clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)

    def clip_gradient(x):
      lo = -0.1
      hi = x + 0.1
      return _clip_gradient(lo, hi, x)

    g = jax.grad(clip_gradient)(0.1)  # doesn't crash
    self.assertAllClose(g, jnp.array(0.2))

  def test_nestable_vjp(self):
    # Verify that https://github.com/jax-ml/jax/issues/3667 is resolved.
    def f(x):
      return x ** 2

    @jax.custom_vjp
    def g(x):
      return f(x)

    def g_fwd(x):
      y, f_vjp = api.vjp(f, x)
      return y, f_vjp

    def g_bwd(f_vjp, y_bar):
      return f_vjp(y_bar)

    g.defvjp(g_fwd, g_bwd)

    # Check that VJP can be nested in simple situations.  For this to pass,
    # vjp has to return a PyTree.
    _, g_vjp = api.vjp(g, 1.0)
    y, = g_vjp(1.0)
    self.assertAllClose(y, jnp.array(2.0))

    # Check that VJP can be nested in complex situations.  For this to pass,
    # vjp can't treat the closed-over tracer x as a static argument.
    @jit
    def z(x):
      _, g_vjp = api.vjp(g, x)
      return g_vjp
    y, = z(1.0)(3.0)
    self.assertAllClose(y, jnp.array(6.0))

  def test_initial_style_vmap_2(self):
    # https://github.com/jax-ml/jax/issues/4173
    x = jnp.ones((10, 3))

    # Create the custom function
    @jax.custom_vjp
    def custom_fun(x):
      return x.sum()

    def forward(x):
      return x.sum(), (jnp.ones_like(x),)

    def backward(res, g):
      return g * res[0],

    custom_fun.defvjp(forward, backward)

    def train_fun(x):

      def summed_fun(x):
        return api.vmap(custom_fun)(x).sum()

      return api.grad(summed_fun)(x)

    def scan_body(carry, inputs):
      x = carry
      return carry, train_fun(x)

    scan_range = jnp.arange(4)
    lax.scan(scan_body, x, scan_range)  # don't crash

  def test_initial_style_vmap_3(self):
    # This is like test_initial_style_vmap except the primal function closes
    # over an array constant.
    y = jnp.arange(1., 4.)

    @jax.custom_vjp
    def f(x):
      assert jnp.ndim(x) == 0
      return 3 * x * jnp.sum(y)
    def f_fwd(x):
      return f(x), jnp.cos(x)
    def f_rev(cos_x, g):
      return (2 * cos_x * g,)
    f.defvjp(f_fwd, f_rev)

    def foo(x):
      out, _  = lax.scan(lambda c, _: (f(c), None), x, None, length=1)
      return out

    ans = api.vmap(foo)(jnp.arange(3.))
    expected = 3. * jnp.arange(3.) * 6
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(lambda x: api.vmap(foo)(x).sum())(jnp.arange(3.))
    expected = 2. * jnp.cos(jnp.arange(3.))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_initial_style_vmap_with_collective(self):

    @jax.custom_vjp
    def f(x):
      return lax.psum(x, 'foo')

    def f_fwd(x):
      return lax.psum(x, 'foo'), None

    def f_bwd(res, dx):
      return dx
    f.defvjp(f_fwd, f_bwd)

    def g(x):
      jaxpr = api.make_jaxpr(f)(x)
      return core.eval_jaxpr(jaxpr.jaxpr, [], x)[0]

    out = api.vmap(lambda _, x: g(x), axis_name='foo', in_axes=(0, None),
        out_axes=None)(jnp.arange(4.), 2.)
    self.assertAllClose(out, 8.)

  def test_bwd_closes_over_tracer(self):
    def f(y):
      @jax.custom_vjp
      def f(x):
        return 2. * jnp.sin(x)

      def fwd(x):
        return f(x), ()

      def bwd(_, g):
        return (2. * jnp.cos(y) * g,)  # capture!

      f.defvjp(fwd, bwd)

      return jax.grad(f)(1.)

    ans = jax.jit(f)(2.)
    self.assertAllClose(ans, 2. * jnp.cos(2.))

    ans = jax.vmap(f)(jnp.arange(3.))
    self.assertAllClose(ans, 2. * jnp.cos(jnp.arange(3.)))

    ans = jax.jit(jax.vmap(f))(jnp.arange(3.))
    self.assertAllClose(ans, 2. * jnp.cos(jnp.arange(3.)))

    ans = jax.vmap(jax.jit(f))(jnp.arange(3.))
    self.assertAllClose(ans, 2. * jnp.cos(jnp.arange(3.)))

    ans = jax.grad(f)(4.)
    self.assertAllClose(ans, -2. * jnp.sin(4.))

  def test_fwd_closes_over_tracer(self):
    def f(y):
      @jax.custom_vjp
      def f(x):
        return 2. * jnp.sin(x)

      def fwd(x):
        return f(x), y

      def bwd(y, g):
        return (2. * jnp.cos(y) * g,)  # capture!

      f.defvjp(fwd, bwd)

      return jax.grad(f)(1.)

    ans = jax.jit(f)(2.)
    self.assertAllClose(ans, 2. * jnp.cos(2.))

    ans = jax.vmap(f)(jnp.arange(3.))
    self.assertAllClose(ans, 2. * jnp.cos(jnp.arange(3.)))

    ans = jax.jit(jax.vmap(f))(jnp.arange(3.))
    self.assertAllClose(ans, 2. * jnp.cos(jnp.arange(3.)))

    ans = jax.vmap(jax.jit(f))(jnp.arange(3.))
    self.assertAllClose(ans, 2. * jnp.cos(jnp.arange(3.)))

    ans = jax.grad(f)(4.)
    self.assertAllClose(ans, -2. * jnp.sin(4.))

  def test_float0(self):
    @jax.custom_vjp
    def f(x, _):
      return x
    def f_fwd(x, _):
      # we need a defined (non-float0) tangent to trigger the rule
      return x, (2., 1)
    def f_rev(*_):
      return (2., 1)
    f.defvjp(f_fwd, f_rev)

    x = 2.
    y = 3
    self.assertEqual(api.grad(f, allow_int=True, argnums=(0, 1))(x, y),
                     (2., np.zeros(shape=(), dtype=float0)))

  def test_float0_initial_style(self):
    @jax.custom_vjp
    def f(x):
      return x
    def f_fwd(x):
      return x, (2., x)
    def f_rev(*_):
      return ((2., jnp.zeros(shape=(), dtype=float0)),)
    f.defvjp(f_fwd, f_rev)

    def foo(x, y):
      out, _ = lax.scan(lambda c, _: (f(c), None), (x, y), None, length=1)
      return out[0]

    x = 2.
    y = 3
    self.assertEqual(api.grad(foo, allow_int=True, argnums=(0, 1))(x, y),
                     (2., np.zeros(shape=(), dtype=float0)))

  def test_remat(self):
    @jax.custom_vjp
    def f(x):
      return jnp.sin(x)
    def f_fwd(x):
      return f(x), jnp.cos(x)
    def f_rev(cos_x, g):
      return (2 * cos_x * g,)
    f.defvjp(f_fwd, f_rev)

    @jax.remat
    def g(x):
      return f(f(x))

    ans = g(2.)
    expected = np.sin(np.sin(2.))
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(g)(2.)
    expected = 4. * api.grad(lambda x: jnp.sin(jnp.sin(x)))(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_remat_higher_order(self):
    @jax.custom_vjp
    def f(x):
      return jnp.sin(x)
    def f_fwd(x):
      return f(x), jnp.cos(x)
    def f_rev(cos_x, g):
      return (2 * cos_x * g,)
    f.defvjp(f_fwd, f_rev)

    def g(x):
      return f(f(x))

    ans = api.grad(api.grad(jax.remat(g)))(2.)
    expected = api.grad(api.grad(g))(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(jax.remat(api.grad(g)))(2.)
    expected = api.grad(api.grad(g))(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = api.grad(api.grad(api.grad(jax.remat(g))))(2.)
    expected = api.grad(api.grad(api.grad(g)))(2.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_bwd_nones(self):
    @jax.custom_vjp
    def f(x, y):
      return x * jnp.sin(y)
    def f_fwd(x, y):
      return f(x, y), jnp.cos(y)
    def f_rev(cos, g):
      return (None, 2 * cos * g)
    f.defvjp(f_fwd, f_rev)

    ans = api.grad(lambda x: f(x, x))(3.)
    expected = 2 * jnp.cos(3.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_bwd_nones_vmap(self):
    @jax.custom_vjp
    def f(x, y):
      return x * jnp.sin(y)
    def f_fwd(x, y):
      return f(x, y), jnp.cos(y)
    def f_rev(cos, g):
      return (None, 2 * cos * g)
    f.defvjp(f_fwd, f_rev)

    ans = api.grad(lambda x: api.vmap(f)(x, x).sum())(jnp.arange(3.))
    expected = 2 * jnp.cos(jnp.arange(3.))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_bwd_nones_pytree(self):
    @jax.custom_vjp
    def f(xs, y):
      x1, x2 = xs
      return x1 * x2 * jnp.sin(y)
    def f_fwd(xs, y):
      return f(xs, y), jnp.cos(y)
    def f_rev(cos, g):
      return (None, 2 * cos * g)
    f.defvjp(f_fwd, f_rev)

    ans = api.grad(lambda x: f((x, x), x))(3.)
    expected = 2 * jnp.cos(3.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_custom_vjp_closure_4521(self):
    # https://github.com/jax-ml/jax/issues/4521
    @jax.custom_vjp
    def g(x, y):
      return None
    def g_fwd(x, y):
      return None, y
    def g_bwd(residuals, z_bar):
      assert False

    g.defvjp(g_fwd, g_bwd)

    def f(xs, y):
      v_g = api.vmap(g, in_axes=(0, None), out_axes=None)
      v_g(xs, y)

    def scan_body(xs, _):
      y = jnp.zeros(1)
      _, vjp_f = api.vjp(f, xs, y)
      vjp_f(None)
      return xs, None

    lax.scan(scan_body, jnp.ones(5), None, 100)  # doesn't crash

  def test_float0_bwd_none(self):
    @jax.custom_vjp
    def f(i, x):
      return jnp.sin(x)
    def f_fwd(i, x):
      return f(i, x), jnp.cos(x)
    def f_rev(cos_x, g):
      return (None, 2 * cos_x * g)
    f.defvjp(f_fwd, f_rev)

    ans = api.grad(f, 1)(jnp.array([1, 2]), 3.)  # doesn't crash
    expected = 2 * jnp.cos(3.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_custom_gradient(self):
    @jax.custom_gradient
    def f(x):
      return x ** 2, lambda g: (g * x,)

    self.assertAllClose(f(3.), 9., check_dtypes=False)
    self.assertAllClose(api.grad(f)(3.), 3., check_dtypes=False)
    self.assertAllClose(api.grad(api.grad(f))(3.), 1., check_dtypes=False)

  def test_custom_gradient_2(self):
    @jax.custom_gradient
    def f(x, y):
      return x * y, lambda g: (y, x)

    self.assertAllClose(f(3., 4.), 12., check_dtypes=False)
    self.assertAllClose(api.grad(f, argnums=(0, 1))(3., 4.), (4., 3.),
                        check_dtypes=False)

  def test_custom_gradient_3(self):
    @jax.custom_gradient
    def f(x):
      vjp = lambda g: (jnp.cos(x) * jnp.arange(3., 6.),)
      return jnp.sum(jnp.sin(x)), vjp

    self.assertAllClose(f(jnp.arange(3)), jnp.sum(jnp.sin(jnp.arange(3.))),
                        check_dtypes=False)
    self.assertAllClose(
        api.grad(f)(jnp.arange(3.)),
        api.grad(lambda x: jnp.sum(jnp.sin(x)))(jnp.arange(3.)) * jnp.arange(3., 6.),
        check_dtypes=False)

  def test_custom_gradient_can_return_singleton_value_in_vjp(self):
    @jax.custom_gradient
    def f(x):
      return x ** 2, lambda g: g * x

    self.assertAllClose(f(3.), 9., check_dtypes=False)
    self.assertAllClose(api.grad(f)(3.), 3., check_dtypes=False)
    self.assertAllClose(api.grad(api.grad(f))(3.), 1., check_dtypes=False)

  def test_closure_convert(self):
    def cos_after(fn, x):
      converted_fn, aux_args = jax.closure_convert(fn, x)
      self.assertLessEqual(len(aux_args), 1)
      return _cos_after(converted_fn, x, *aux_args)

    @partial(jax.custom_vjp, nondiff_argnums=(0,))
    def _cos_after(fn, x, *args):
      return jnp.cos(fn(x, *args))

    def fwd(fn, x, *args):
      y = _cos_after(fn, x, *args)
      return y, (x, args)

    def rev(fn, res, g):
      x, args = res
      x_bar = 17. * x
      args_bars = [42. * a for a in args]
      return (x_bar, *args_bars)

    _cos_after.defvjp(fwd, rev)

    def dist(c, x):
      return jnp.sum((x - c) ** 2.)

    def solve(c, x):
      def closure(x):
        return dist(c, x)
      return cos_after(closure, x)

    c, x = 2. * jnp.ones(2), jnp.ones(2)
    expected = jnp.cos(dist(c, x))
    self.assertAllClose(solve(c, x), expected, check_dtypes=False)
    g_c, g_x = api.grad(solve, argnums=(0, 1))(c, x)
    self.assertAllClose(g_c, 42. * c, check_dtypes=False)
    self.assertAllClose(g_x, 17. * x, check_dtypes=False)

  def test_closure_convert_mixed_consts(self):
    # Like test_closure_convert, but close over values that
    # participate in AD as well as values that do not.
    # See https://github.com/jax-ml/jax/issues/6415

    def cos_after(fn, x):
      converted_fn, aux_args = jax.closure_convert(fn, x)
      self.assertLessEqual(len(aux_args), 1)
      return _cos_after(converted_fn, x, *aux_args)

    @partial(jax.custom_vjp, nondiff_argnums=(0,))
    def _cos_after(fn, x, *args):
      return jnp.cos(fn(x, *args))

    def fwd(fn, x, *args):
      y = _cos_after(fn, x, *args)
      return y, (x, args)

    def rev(fn, res, g):
      x, args = res
      x_bar = 17. * x
      args_bars = [42. * a for a in args]
      return (x_bar, *args_bars)

    _cos_after.defvjp(fwd, rev)

    def dist(c, s, x):
      return jnp.sum(s * (x - c) ** 2.)

    def solve(c, s, x):
      def closure(x):
        return dist(c, s, x)
      return cos_after(closure, x)

    c, s, x = 2. * jnp.ones(2), 3. * jnp.ones(2), jnp.ones(2)
    expected = jnp.cos(dist(c, s, x))
    self.assertAllClose(solve(c, s, x), expected, check_dtypes=False)
    g_c, g_x = api.grad(solve, argnums=(0, 2))(c, s, x)
    self.assertAllClose(g_c, 42. * c, check_dtypes=False)
    self.assertAllClose(g_x, 17. * x, check_dtypes=False)

  def test_closure_convert_pytree_mismatch(self):
    # See https://github.com/jax-ml/jax/issues/23588
    def f(x, z):
      return z * x

    x, z = 2.0, 3.0
    _, vjp = api.vjp(f, x, z)
    vjp_pure, vjp_aux_args = jax.closure_convert(vjp, x)
    vjp_pure(x, *vjp_aux_args)
    with self.assertRaisesRegex(
        TypeError, "The inputs to the closure produced by closure_convert"):
      vjp_pure(x, vjp_aux_args)

  def test_float0_cotangents_automatically_handled(self):
    @jax.custom_vjp
    def f(x, y):
      return x

    def f_fwd(x, y):
      return x, None

    def f_bwd(_, zbar):
      return (0., 1)

    f.defvjp(f_fwd, f_bwd)

    jax.jit(lambda x: jax.vjp(f, 0., x)[1](1.))(1)  # doesn't crash

  def test_custom_vjp_scan_batching_edge_case(self):
    # https://github.com/jax-ml/jax/issues/5832
    @jax.custom_vjp
    def mul(x, coeff): return x * coeff
    def mul_fwd(x, coeff): return mul(x, coeff), (x, coeff)
    def mul_bwd(res, g):
      x, coeff = res
      g_x = g * coeff
      g_coeff = (x * g).sum()
      return g_x, g_coeff
    mul.defvjp(mul_fwd, mul_bwd)

    def scan_over_mul(x, coeff):
      def f_(x, t):
        return mul(x, coeff), None
      y, _ = jax.lax.scan(f_, x, jnp.arange(3))
      return y

    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    x_batch = jax.random.normal(key1, (3, 2))
    covector_batch = jax.random.normal(key2, (3, 2))
    coeff = jnp.array(1., dtype=x_batch.dtype)

    batched_scan_over_mul = jax.vmap(scan_over_mul, in_axes=(0, None), out_axes=0)
    res, vjp_fun = jax.vjp(batched_scan_over_mul, x_batch, coeff)
    vjp_fun(covector_batch)  # doesn't crash

    jtu.check_grads(batched_scan_over_mul, (x_batch, coeff), order=2,
                    modes=['rev'])

  def test_closure_with_vmap2(self):
    # https://github.com/jax-ml/jax/issues/8783
    def h(z):
      def f(x):
        @jax.custom_vjp
        def g(y):
          return x * y

        def g_fwd(y):
          return x * y, (x, x * y, y)
        def g_rev(res, w_bar):
          x, *_ = res
          return (x * w_bar,)
        g.defvjp(g_fwd, g_rev)

        return g(z)

      return jax.vmap(f)(jnp.arange(3., dtype='float32')).sum()

    jtu.check_grads(h, (jnp.float32(3.14),), order=1, modes=['rev'])

  def test_pytrees_not_required_to_contain_nones(self):
    class A(list):
      pass

    def unflatten(_, children):
      assert children[0] is not None
      return A(children)

    tree_util.register_pytree_node(A, lambda x: (x, None), unflatten)

    @jax.custom_vjp
    def f(x):
      return x[0]
    def f_fwd(x):
      return x[0], None
    def f_bwd(_, g):
      return A([g]),
    f.defvjp(f_fwd, f_bwd)

    jax.grad(f)(A([1.]))  # doesn't crash

  def test_vmap_vjp_called_twice(self):
    # https://github.com/jax-ml/jax/pull/14728
    @jax.custom_vjp
    def f(x):
      return x
    f.defvjp(lambda x: (x, None), lambda _, y_bar: (y_bar,))

    _, f_vjp = jax.vjp(jax.vmap(f), jnp.array([3.]))
    f_vjp(jnp.array([3.]))
    f_vjp(jnp.array([3.]))  # doesn't crash

  def test_symbolic_zero_custom_vjp_basic(self):
    ZERO = jax.custom_derivatives.SymbolicZero

    @jax.custom_vjp
    def f(x, y, z):
      return x, x

    def fwd(x, y, z):
      self.assertIsInstance(x, jax.custom_derivatives.CustomVJPPrimal)
      self.assertIsInstance(y, jax.custom_derivatives.CustomVJPPrimal)
      self.assertIsInstance(z, jax.custom_derivatives.CustomVJPPrimal)
      self.assertTrue(x.perturbed)
      self.assertFalse(y.perturbed)
      self.assertFalse(z.perturbed)
      return (x.value, x.value), None

    def fwd_all(x, y, z):
      self.assertIsInstance(x, jax.custom_derivatives.CustomVJPPrimal)
      self.assertIsInstance(y, jax.custom_derivatives.CustomVJPPrimal)
      self.assertIsInstance(z, jax.custom_derivatives.CustomVJPPrimal)
      self.assertTrue(x.perturbed)
      self.assertTrue(y.perturbed)
      self.assertTrue(z.perturbed)
      return (x.value, x.value), None

    def bwd_all(_, g):
      x1, x2 = g
      self.assertFalse(type(x1) is ZERO)
      self.assertFalse(type(x2) is ZERO)
      return x1, x1, x2

    def bwd_fst(_, g):
      x1, x2 = g
      self.assertFalse(type(x1) is ZERO)
      self.assertIs(type(x2), ZERO)
      return x1, x1, x2

    def bwd_snd(_, g):
      x1, x2 = g
      self.assertIs(type(x1), ZERO)
      self.assertFalse(type(x2) is ZERO)
      return x1, x1, x2

    x, y, z = 4., 5., 6.
    i = np.array(7, np.int32)
    zero = np.array(0.)

    f.defvjp(fwd, bwd_all, symbolic_zeros=True)
    h = jax.jit(f)
    jax.jacrev(h)(x, y, z)
    jax.jacrev(lambda x: h(x, y, z))(x)
    jax.jacrev(h, argnums=(0, 1, 2), allow_int=True)(x, i, i)

    f.defvjp(fwd_all, bwd_fst, symbolic_zeros=True)
    fst_f = lambda *xs: f(*xs)[0]
    _, vjp = jax.vjp(fst_f, x, y, z)
    _, _, gz = vjp(x)
    self.assertArraysAllClose(gz, zero)

    f.defvjp(fwd_all, bwd_snd, symbolic_zeros=True)
    snd_f = lambda *xs: f(*xs)[1]
    _, vjp = jax.vjp(snd_f, x, y, z)
    gx, gy, _ = vjp(x)
    self.assertArraysAllClose(gx, zero)
    self.assertArraysAllClose(gy, zero)

    f.defvjp(fwd, bwd_snd, symbolic_zeros=True)
    _, vjp = jax.vjp(lambda x: snd_f(x, y, z), x)
    gx, = vjp(x)
    self.assertArraysAllClose(gx, zero)

  def test_symbolic_zero_custom_vjp_bwd_shape_error(self):
    @jax.custom_vjp
    def f(x, y, z):
      return x, y, z

    def fwd(x, y, z):
      return f(x.value, y.value, z.value), None

    def bwd(_, gs):
      x_bar, y_bar, z_bar = gs
      return y_bar, x_bar, z_bar  # swapped!

    f.defvjp(fwd, bwd, symbolic_zeros=True)

    with self.assertRaisesRegex(
        ValueError,
        r'Consider just returning a None here'):
      jax.grad(lambda x, y, z: f(x, y, z)[2].sum())(
        jnp.ones(1), jnp.ones(2), jnp.ones(3))

  @parameterized.named_parameters(
      ('jit_vmap', True, True),
      ('jit', True, False),
      ('vmap', False, True),
      ('', False, False),
  )
  def test_symbolic_zero_custom_vjp(self, maybe_jit, maybe_vmap):
    # below:
    # * static_scalar will be static in and out
    # * static_array will be static in, but dynamic out
    # * dyn_scalar and dyn_array will be dynamic in and out

    ZERO = jax.custom_derivatives.SymbolicZero

    def f(static_scalar, static_array, dyn_scalar, dyn_array):
      out1 = static_scalar + dyn_scalar
      out2 = static_array + dyn_array
      return static_scalar, static_array, out1, out2

    def _pack(x):
      return lax.broadcast(x, (1,))

    def _unpack(x):
      (x,) = x
      return x

    def _vmap(fun):
      def _fun(*args):
        args = jax.tree.map(_pack, args)
        out = jax.vmap(fun)(*args)
        out = jax.tree.map(_unpack, out)
        return out
      return _fun

    f = jax.custom_vjp(f)

    def fwd(*args):
      xs, pert = [x.value for x in args], [x.perturbed for x in args]
      self.assertFalse(pert[0])
      self.assertFalse(pert[1])
      self.assertTrue(pert[2])
      self.assertTrue(pert[3])
      return f(*xs), xs

    def bwd(res, g):
      static_scalar, *_ = res
      t_static, t_static_arr, t_dyn_scalar, t_dyn_array = g
      self.assertIs(type(t_static), ZERO)
      self.assertFalse(type(t_static_arr) is ZERO)
      self.assertFalse(type(t_dyn_scalar) is ZERO)
      self.assertFalse(type(t_dyn_array)  is ZERO)
      self.assertEqual(t_static.shape, ())
      self.assertEqual(t_static_arr.shape, (2,))
      return (static_scalar + 90,
              t_static_arr  + 91,
              t_dyn_scalar  + 92,
              t_dyn_array   + 93)

    f.defvjp(fwd, bwd, symbolic_zeros=True)

    def g(dyn_scalar, dyn_array):
      if maybe_vmap:
        f_ = _vmap(f)
      else:
        f_ = f
      outs = f_(1., jnp.array([2., 3.]), dyn_scalar, dyn_array)
      return outs[1:]

    def run(primal_ins, cotangent_outs):
      primal_outs, vjp = jax.vjp(g, *primal_ins)
      cotangent_ins = vjp(cotangent_outs)
      return primal_outs, cotangent_ins

    if maybe_jit:
      run = jax.jit(run)

    scalar_type = jax.Array if maybe_jit or maybe_vmap else float
    primal_ins = (4., jnp.array([5., 6.]))
    cotangent_outs = (jnp.array([10., 11.]), 7., jnp.array([8., 9.]))
    primal_outs, cotangent_ins = run(primal_ins, cotangent_outs)

    primal_out1, primal_out2, primal_out3 = primal_outs
    self.assertIsInstance(primal_out1, jax.Array)
    self.assertAllClose(primal_out1, jnp.array([2., 3.]))
    self.assertIsInstance(primal_out2, scalar_type)
    self.assertAllClose(primal_out2, 5.)
    self.assertIsInstance(primal_out3, jax.Array)
    self.assertAllClose(primal_out3, jnp.array([7., 9.]))

    ct_in1, ct_in2 = cotangent_ins
    self.assertIsInstance(ct_in1, scalar_type)
    self.assertAllClose(ct_in1, 99.)
    self.assertIsInstance(ct_in2, jax.Array)
    self.assertArraysAllClose(ct_in2, jnp.array([101., 102.]))

  def test_symbolic_zero_custom_vjp_vmap_output(self):
    @jax.custom_vjp
    def f(x, y):
      return x, y

    def fwd(x, y):
      self.assertTrue(x.perturbed)
      self.assertFalse(y.perturbed)
      return f(x.value, y.value), None

    def bwd(_, g):
      _, ct_y = g
      self.assertIs(type(ct_y), jax.custom_derivatives.SymbolicZero)
      return g

    f.defvjp(fwd, bwd, symbolic_zeros=True)
    jax.grad(lambda x, y: jax.vmap(f)(x, y)[0].sum())(jnp.ones(3), jnp.ones(3))

  def test_symbolic_zero_custom_vjp_custom_pytree(self):
    tree_values = jax.custom_derivatives.custom_vjp_primal_tree_values

    @tree_util.register_pytree_node_class
    class Box:
      def __init__(self_, strict, val):
        if strict:
          # make sure we aren't getting special arguments that should only
          # come up when symbolic_zeros is True
          self.assertFalse(hasattr(val, 'perturbed'))
        self_.strict = strict
        self_.x = val

      def tree_flatten(self_):
        return [self_.x], self_.strict

      @classmethod
      def tree_unflatten(cls, strict, xs):
        x, = xs
        return cls(strict, x)

    x, y = Box(False, jnp.array(72.)), jnp.array(73.)

    @jax.custom_vjp
    def f(box, y):
      return box.x * y

    def fwd0(box, y):
      self.assertTrue(box.x.perturbed)
      self.assertFalse(y.perturbed)
      box, y = map(tree_values, [box, y])
      return f(box, y), (box, y)

    def bwd0(res, g):
      box, y = res
      return y * g, box.x * g

    def fwd1(box, y):
      self.assertFalse(box.x.perturbed)
      self.assertTrue(y.perturbed)
      box, y = map(tree_values, [box, y])
      return f(box, y), (box, y)

    def bwd1(res, g):
      box, y = res
      return y * g, box.x * g

    f.defvjp(fwd0, bwd0, symbolic_zeros=True)
    jax.grad(f, argnums=0)(x, y)
    f.defvjp(fwd1, bwd1, symbolic_zeros=True)
    jax.grad(f, argnums=1)(x, y)

    def fwd_strict(box, y):
      return f(box, y), (box, y)

    def bwd_strict(res, g):
      box, y = res
      return y * g, box.x * g

    f.defvjp(fwd_strict, bwd_strict)
    jax.grad(f)(x, y)

  def test_symbolic_zeros_memoization_caching(self):
    # Tests multiple zero patterns for partial_eval._memoize, and also tests
    # that we're okay with stores being occupied with equal values.
    @jax.custom_vjp
    def f(x, y):
      return x * y

    def f_fwd(x, y):
      return x.value, None

    def f_bwd(_, z_bar):
      return z_bar, None

    f.defvjp(f_fwd, f_bwd, symbolic_zeros=True)

    f_ = core.jaxpr_as_fun(jax.make_jaxpr(f)(2., 3.))
    _ = jax.linearize(f_, 2., 3.)
    _ = jax.linearize(lambda x: f_(x, 3.), 2.)  # don't crash!

  def test_run_rules_more_than_once(self):
    # https://github.com/jax-ml/jax/issues/16614

    @jax.custom_vjp
    def f(x, y):
      return x + y

    def f_fwd(x, y):
      if y.perturbed:
        res = None
      else:
        res = []
      return x.value + y.value, res

    def f_bwd(res, ct):
      return ct, ct

    f.defvjp(f_fwd, f_bwd, symbolic_zeros=True)

    def body(x_y, _):
      x, y = x_y
      return (f(x, y), x), None

    @jax.grad
    def g(x):
      (out, _), _ = lax.scan(body, (x, 1.), xs=None, length=2)
      return out

    g(1.)  # doesn't crash

  def test_nones_representing_zeros_in_subtrees_returned_by_bwd(self):
    # https://github.com/jax-ml/jax/issues/8356
    @jax.custom_vjp
    def f(x):
      return x[0]

    def f_fwd(x):
      return f(x), None

    def f_bwd(_, z_bar):
      return (z_bar, (None, None)),

    f.defvjp(f_fwd, f_bwd)

    jax.grad(f)((1.0, (2.0, 3.0)))  # don't crash

  def test_pytree_nones_returned_by_bwd(self):
    @jax.custom_vjp
    def f(x):
      return x[0]

    def f_fwd(x):
      return f(x), None

    def f_bwd(_, z_bar):
      return (z_bar, (None, None)),

    f.defvjp(f_fwd, f_bwd)

    jax.grad(f)((1.0, (2.0, None)))  # don't crash

  def test_bwd_rule_shape_mismatch(self):
    @jax.custom_vjp
    def foo(x, y):
      return x

    def foo_fwd(x, y):
      return x, None

    def foo_bwd(_, g):
      return jnp.zeros(3), jnp.zeros(3)

    foo.defvjp(foo_fwd, foo_bwd)

    with self.assertRaisesRegex(
        ValueError,
        r'output\[1\] the bwd rule produced an output of shape/dtype float..\[3\]'):
      jax.grad(lambda x, y: foo(x, y * y).sum(), 1)(jnp.ones(3), jnp.ones(4))

  def test_bwd_rule_shape_mismatch_disable(self):
    # TODO(mattjj): remove this test when the config option is removed
    @jax.custom_vjp
    def foo(x, y):
      return x

    def foo_fwd(x, y):
      return x, None

    def foo_bwd(_, g):
      return jnp.zeros(3), jnp.zeros(3)

    foo.defvjp(foo_fwd, foo_bwd)

    with config.custom_vjp_disable_shape_check(True):
      jax.grad(lambda x, y: foo(x, y).sum(), 1)(jnp.ones(3), jnp.ones(4))

  def test_bwd_rule_can_produce_list_or_tuple(self):
    @jax.custom_vjp
    def f(x, y):
      return x * y

    def f_fwd(x, y):
      return f(x, y), (x, y)

    def f_bwd(xy, g):
      x, y = xy
      return [g * y, x * g]  # list, not tuple

    f.defvjp(f_fwd, f_bwd)

    jax.grad(f)(1., 2.)  # don't crash

  def test_optimize_remat(self):
    def fun(x):
      # This array is included to make sure that we handle consts appropriately
      return np.array([1.0])*x

    def fwd(x):
      return np.array([2.0])*x*x/np.array([1.0]), (2 * x,)

    x = jnp.linspace(0, 5.0, 10)
    fwd = custom_derivatives.optimize_remat_of_custom_vjp_fwd(
        fun, api_util.debug_info("custom_vjp fun", fun, (x,), {}),
        fwd, api_util.debug_info("custom_vjp fwd", fwd, (x,), {}))

    self.assertAllClose(jax.jit(fwd)(x)[0], 2*x*x)  # Shouldn't hit custom DCE
    self.assertAllClose(jax.jit(lambda x: fwd(x)[0])(x), x)  # Should be DCEed

  def test_optimize_remat_vmap(self):
    def fun(x):
      return (np.array([1.0])*x)[0]
    def fwd(x):
      return (np.array([2.0])*x*x/np.array([1.0]))[0], (2 * x,)
    x = jnp.linspace(0, 5.0, 10)
    fwd = custom_derivatives.optimize_remat_of_custom_vjp_fwd(
        fun, api_util.debug_info("custom_vjp fun", fun, (x,), {}),
        fwd, api_util.debug_info("custom_vjp fwd", fwd, (x,), {}))
    self.assertAllClose(jax.jit(jax.vmap(fwd))(x)[0], 2*x*x)
    self.assertAllClose(jax.jit(lambda x: jax.vmap(fwd)(x)[0])(x), x)

  def test_optimize_remat_cond(self):
    def fun(x):
      return x
    def fwd(x):
      return x*x, (2 * x,)

    x = jnp.linspace(0, 5.0, 10)
    fwd = custom_derivatives.optimize_remat_of_custom_vjp_fwd(
        fun, api_util.debug_info("custom_vjp fun", fun, (x,), {}),
        fwd, api_util.debug_info("custom_vjp fwd", fwd, (x,), {}))

    def g(x):
      return jax.lax.cond(True, fwd, lambda x: (2.0 * x, (x,)), x)

    self.assertAllClose(jax.jit(g)(x)[0], x*x)
    self.assertAllClose(jax.jit(lambda x: g(x)[0])(x), x)

  def test_optimize_remat_jvp(self):
    def fun(x):
      return x**2
    def fwd_(x):
      return x*x, (2 * x,)

    fwd = custom_derivatives.optimize_remat_of_custom_vjp_fwd(
        fun, api_util.debug_info("custom_vjp fun", fun, (3.2,), {}),
        fwd_, api_util.debug_info("custom_vjp fwd", fwd_, (3.2,), {}))
    calc = jax.jvp(fwd, (3.2,), (1.0,))
    expected = jax.jvp(fwd_, (3.2,), (1.0,))
    self.assertAllClose(calc, expected)

    @jax.jit
    def g(x, t):
      (y, r), (y_dot, r_dot) = jax.jvp(fwd, (x,), (t,))
      return y, y_dot
    calc = g(3.2, 1.0)
    expected = jax.jvp(fun, (3.2,), (1.0,))
    self.assertAllClose(calc, expected)

  def test_optimize_remat_gh21303(self):
    @jax.custom_vjp
    def f(x):
      return jnp.tan(x)

    def f_fwd(x):
      return jnp.sin(x), (x,)

    def f_bwd(res, g):
      x, = res
      cos_x = jnp.cos(x)
      return (cos_x * g,)

    f.defvjp(f_fwd, f_bwd, optimize_remat=True)

    def temp(x):
      out = jax.remat(f)(x)
      out = out ** 2
      return out

    v, g = jax.value_and_grad(temp)(3.2)
    self.assertAllClose(v, jnp.tan(3.2)**2)

  def test_optimize_remat_multiple_args(self):
    def f_(x, y):
      return jnp.sin(x) * y

    @jax.custom_vjp
    def f(x, y):
      return f_(x, y)

    def f_fwd(x, y):
      return f(x, y), (jnp.cos(x), jnp.sin(x), y)

    def f_bwd(res, g):
      cos_x, sin_x, y = res
      return (cos_x * g * y, sin_x * g)

    f.defvjp(f_fwd, f_bwd, optimize_remat=True)
    x, y = 3.2, 1.0
    self.assertAllClose(jax.grad(f)(x, y), jax.grad(f_)(x, y))

  def test_optimize_remat_kwargs(self):
    @jax.custom_vjp
    def f(x, y):
      return jnp.sin(x) * y

    def f_fwd(x, y, *, keyword=False):
      del keyword
      return f(x, y), (jnp.cos(x), jnp.sin(x), y)

    def f_bwd(res, g):
      cos_x, sin_x, y = res
      return (cos_x * g * y, sin_x * g)

    f.defvjp(f_fwd, f_bwd, optimize_remat=True)
    x, y = 3.2, 1.0
    jax.grad(f)(x, y)  # Doesn't error

  def test_optimize_remat_custom_vmap(self):
    # See https://github.com/jax-ml/jax/pull/23000
    @jax.custom_vjp
    def f(x, y):
      return jnp.sin(x) * y

    @jax.custom_batching.custom_vmap
    def f_fwd(x, y):
      return f(x, y), (jnp.cos(x), jnp.sin(x), y)

    @f_fwd.def_vmap
    def f_fwd_vmap(_, in_batched, x, y):
      # Insert a new const here to test the optimize_remat batching rule.
      out = np.array([2.0])*f(x, y)
      out_batched = (True, (True, True, True))
      return (out, (jnp.cos(x), jnp.sin(x), y)), out_batched

    def f_bwd(res, g):
      cos_x, sin_x, y = res
      return (cos_x * g * y, sin_x * g)

    f.defvjp(f_fwd, f_bwd, optimize_remat=True)
    x, y = jnp.linspace(0.0, 1.0, 5), jnp.linspace(2.0, 5.0, 5)
    jax.jit(jax.vmap(jax.grad(f)))(x, y)  # Doesn't error

  def test_dce(self):
    @jax.custom_vjp
    def f(x, y):
      return jnp.sin(x), x + jnp.cos(y)

    def f_fwd(x, y):
      return f(x, y), (jnp.cos(x), jnp.sin(y))

    def f_bwd(res, cts):
      cos_x, sin_y = res
      ct_a, ct_b = cts
      return 2.0 * cos_x * ct_a + 1.5 * ct_b, -0.5 * sin_y * ct_b

    f.defvjp(f_fwd, f_bwd)

    def check_jaxpr(jaxpr, used_outs, includes, excludes):
      dce_jaxpr, _ = pe.dce_jaxpr(jaxpr, used_outs)
      if not dce_jaxpr.eqns:
        assert not includes
        return
      call_jaxpr = dce_jaxpr.eqns[0].params["call_jaxpr"]
      for prim in includes:
        assert any(eqn.primitive == prim for eqn in call_jaxpr.eqns)
      for prim in excludes:
        assert all(eqn.primitive != prim for eqn in call_jaxpr.eqns)

    x, y = 0.1, -1.3
    jaxpr = jax.make_jaxpr(f)(x, y).jaxpr
    check_jaxpr(jaxpr, [True, True], [lax.sin_p, lax.cos_p], [])
    check_jaxpr(jaxpr, [True, False], [lax.sin_p], [lax.cos_p])
    check_jaxpr(jaxpr, [False, True], [lax.cos_p], [lax.sin_p])
    check_jaxpr(jaxpr, [False, False], [], [lax.sin_p, lax.cos_p])

    def dce_jaxpr_as_fun(jaxpr, used_outs):
      jaxpr_, _ = pe.dce_jaxpr(jaxpr, used_outs)
      fun = core.jaxpr_as_fun(pe.close_jaxpr(jaxpr_))
      return lambda *args: fun(*args)[0]

    f0 = dce_jaxpr_as_fun(jaxpr, [True, False])
    f1 = dce_jaxpr_as_fun(jaxpr, [False, True])
    self.assertAllClose(
        api.grad(f0, argnums=(0, 1))(x, y), (2.0 * jnp.cos(x), 0.0))
    self.assertAllClose(
        api.grad(f1, argnums=(0, 1))(x, y), (1.5, -0.5 * jnp.sin(y)))

  def test_resolve_kwargs_error_message(self):
    @jax.custom_vjp
    def f(x, y, *, z=None):
      return jnp.sin(x), x + jnp.cos(y)

    def f_fwd(x, y):
      self.fail("should not be executed")

    def f_bwd(res, cts):
      self.fail("should not be executed")

    f.defvjp(f_fwd, f_bwd)

    with self.assertRaisesRegex(
        TypeError,
        r"The input arguments to the custom_vjp-decorated function f(.*)\n"
        r"missing a required argument: 'y'"
    ):
      f(0.5)

    with self.assertRaisesRegex(
        TypeError,
        r"The input arguments to the custom_vjp-decorated function f(.*)\n"
        "The following keyword arguments could not be resolved to positions: z"
    ):
      f(0.5, 0.1, z=1.0)

  def test_pretty_print(self):
    @jax.custom_vjp
    def f(x):
      return x + 1

    def f_fwd(x):
      return f(x), ()

    def f_bwd(_, g):
      return g
    f.defvjp(f_fwd, f_bwd)

    x = jnp.array([4.2], dtype=jnp.float32)
    jaxpr = jax.make_jaxpr(f)(x)
    actual = jaxpr.pretty_print(use_color=False)
    expected = textwrap.dedent(
        """
        { lambda ; a:f32[1]. let
            b:f32[1] = custom_vjp_call[
              name=f
              bwd=f_bwd
              call_jaxpr={ lambda ; c:f32[1]. let d:f32[1] = add c 1.0:f32[] in (d,) }
              fwd=f_fwd
              symbolic_zeros=False
            ] a
          in (b,) }
        """).strip()
    self.assertEqual(actual, expected)

  def test_custom_lin_pretty_print(self):
    @jax.custom_vjp
    def f(x):
      return x + 1

    def f_fwd(x):
      return f(x), ()

    def f_bwd(_, g):
      return g
    f.defvjp(f_fwd, f_bwd)

    x = jnp.array([4.2], dtype=jnp.float32)
    jaxpr = jax.make_jaxpr(lambda x: jax.jvp(f, (x,), (x,)))(x)
    jaxpr, _ = pe.dce_jaxpr(jaxpr.jaxpr, [False, True])
    actual = jaxpr.pretty_print(use_color=False)
    expected = textwrap.dedent(
        """
        { lambda ; a:f32[1]. let
            b:f32[1] = custom_lin[
              bwd=f_bwd
              in_zeros=[False]
              num_res=0
              symbolic_zeros=False
            ] a
          in (b,) }
        """).strip()
    self.assertEqual(actual, expected)


def transpose_unary(f, x_example):
  def transposed(y):
    x, = api.linear_transpose(f, x_example)(y)
    return x
  return transposed


# This class wraps jax.custom_transpose.custom_transpose in order to pass in a
# particular tree of output type on each call. Otherwise it forwards
# all attribute access.
class _custom_transpose:
  def __init__(self, out_types, fun):
    self.out_types = out_types
    self.fun = jax.custom_transpose.custom_transpose(fun)

  def __getattr__(self, name):
    return getattr(self.fun, name)

  def __call__(self, *args):
    return self.fun(self.out_types, *args)


# This function is meant to be used as a decorator that delegates to
# custom_transpose but makes it easy to specify output argument types
# by example. If used directly a decorator (i.e. not invoked with
# example arguments), assumes a scalar-valued function.
#
# TODO(frostig): remove this (and its uses) once custom_transpose offers
# an option of inferring output types.
def custom_transpose(example_out):
  if isinstance(example_out, Callable):
    out_type = core.get_aval(0.).to_tangent_aval()
    return _custom_transpose(out_type, example_out)
  return partial(
      _custom_transpose,
      jax.tree.map(
          lambda x: core.get_aval(x).to_tangent_aval(), example_out))


class CustomTransposeTest(jtu.JaxTestCase):

  def test_linear_call(self):
    def f(x, y):
      def fn(r, x): return x / r
      def tp(r, t): return t / r
      return x + jax.custom_derivatives.linear_call(fn, tp, y, x)

    def f_ref(x, y):
      return x + x / y

    x = jnp.ones(2) * 6.
    y = jnp.ones(2) * 3.
    self.assertAllClose(f(x, y), f_ref(x, y))

    f1     = lambda x: f(x, y)
    f1_ref = lambda x: f_ref(x, y)
    self.assertAllClose(transpose_unary(f1,     x)(x),
                        transpose_unary(f1_ref, x)(x))

  def test_linear_call_incorrect_transpose(self):
    def f(x, y):
      def fn(r, x): return x / r
      def tp(r, t): return t / (2. * r)  # nb: not the true transpose
      return x + jax.custom_derivatives.linear_call(fn, tp, y, x)

    def f_ref(x, y):
      return x + x / y

    x = jnp.ones(2) * 6.
    y = jnp.ones(2) * 3.
    self.assertAllClose(f(x, y), f_ref(x, y))

    f1     = lambda x: f(x, y)
    f1_ref = lambda x: f_ref(x, 2. * y)  # nb: double the reference divisor
    self.assertAllClose(transpose_unary(f1,     x)(x),
                        transpose_unary(f1_ref, x)(x))

  def test_linear_call_transpose_transpose_transpose(self):
    def fn(r, x): return x / r
    def tp(r, t): return t / (2. * r)  # nb: untrue transpose
    def f_(x, y):
      return x + jax.custom_derivatives.linear_call(fn, tp, y, x)

    x = jnp.ones(2) * 6.
    y = jnp.ones(2) * 3.
    f = lambda x: f_(x, y)
    ft   = transpose_unary(f,   x)
    ftt  = transpose_unary(ft,  x)
    fttt = transpose_unary(ftt, x)
    self.assertAllClose(ft(x), x + tp(y, x))
    self.assertAllClose(f(x),  ftt(x))
    self.assertAllClose(ft(x), fttt(x))

  def test_linear_call_scalar_to_vector(self):
    def f(c, x):
      def fn(_, x):
        return [x, x]

      def tp(_, t):
        t1, t2 = t
        return t1 + t2

      return jax.custom_derivatives.linear_call(fn, tp, (), c * x)

    def f_ref(c, x):
      return [c * x, c * x]

    c, x = 2., 3.
    t = [4., 5.]
    self.assertAllClose(f(c, x), f_ref(c, x))
    self.assertAllClose(transpose_unary(partial(f,     c), x)(t),
                        transpose_unary(partial(f_ref, c), x)(t))

  def test_linear_call_nested(self):
    # identity function with an untrue transpose of 0
    def id_(x):
      def f(_, x): return x
      def t(_, t): return 0.
      return jax.custom_derivatives.linear_call(f, t, (), x)

    # identity function with an untrue transpose of 7, and where both
    # forward and transpose have custom transpositions that should
    # never end up invoked.
    def f(x):
      def f_(_, x): return id_(x)
      def t_(_, t): return id_(7.)
      return jax.custom_derivatives.linear_call(f_, t_, (), x)

    x = 5.
    id_t  = transpose_unary(id_,  x)
    id_tt = transpose_unary(id_t, x)
    ft   = transpose_unary(f,    x)
    ftt  = transpose_unary(ft,   x)
    fttt = transpose_unary(ftt,  x)

    self.assertAllClose(id_(x),   x)
    self.assertAllClose(id_t(x),  0.)
    self.assertAllClose(id_tt(x), x)

    self.assertAllClose(f(x),    x)
    self.assertAllClose(ft(x),   7.)
    self.assertAllClose(ftt(x),  x)
    self.assertAllClose(fttt(x), 7.)

  def test_linear_call_jit(self):
    def f(x, y):
      def fn(r, x): return x / r
      def tp(r, t): return t / r
      return x + jax.custom_derivatives.linear_call(fn, tp, y, x)

    x = jnp.ones(2) * 6.
    y = jnp.ones(2) * 3.
    self.assertAllClose(f(x, y), jax.jit(f)(x, y))

    f1 = lambda x: f(x, y)
    self.assertAllClose(transpose_unary(f1, x)(x),
                        jax.jit(transpose_unary(f1, x))(x))

  def test_linear_call_type_mismatch(self):
    def f(x, y):
      def fn(r, x): return x / r
      def tp(r, t): return None
      return x + jax.custom_derivatives.linear_call(fn, tp, y, x)

    x = jnp.ones(2) * 6.
    y = jnp.ones(2) * 3.
    f1 = lambda x: f(x, y)
    with self.assertRaisesRegex(TypeError, "transpose output pytree"):
      transpose_unary(f1, x)(x)

  def test_linear_call_recursion(self):
    def f(x):
      def fn(_, x): return x
      def tp(_, t): return f(t)
      return jax.custom_derivatives.linear_call(fn, tp, None, x)
    jax.jit(f)(0.1)

  def test_linear_call_grad(self):
    def f(x, y):
      def fn(r, x): return x / r
      def tp(r, t): return t / r
      return x + jax.custom_derivatives.linear_call(fn, tp, y, x)

    def f_ref(x, y):
      return x + x / y

    x = jnp.array(6.)
    y = jnp.array(3.)
    self.assertAllClose(jax.grad(f)(x, y), jax.grad(f_ref)(x, y))

  def test_basic(self):
    def f(x, y):
      @custom_transpose(jnp.ones(2))
      def fn(r, x): return x / r
      @fn.def_transpose
      def tp(r, t): return t / r

      return x + fn(y, x)

    def f_ref(x, y):
      return x + x / y

    x = jnp.ones(2) * 6.
    y = jnp.ones(2) * 3.
    self.assertAllClose(f(x, y), f_ref(x, y))

    f1     = lambda x: f(x, y)
    f1_ref = lambda x: f_ref(x, y)
    self.assertAllClose(transpose_unary(f1,     x)(x),
                        transpose_unary(f1_ref, x)(x))

  def test_incorrect_transpose(self):
    def f(x, y):
      @custom_transpose(jnp.ones(2))
      def fn(r, x): return x / r
      @fn.def_transpose
      def tp(r, t): return t / (2. * r)  # nb: not the true transpose

      return x + fn(y, x)

    def f_ref(x, y):
      return x + x / y

    x = jnp.ones(2) * 6.
    y = jnp.ones(2) * 3.
    self.assertAllClose(f(x, y), f_ref(x, y))

    f1     = lambda x: f(x, y)
    f1_ref = lambda x: f_ref(x, 2. * y)  # nb: double the reference divisor
    self.assertAllClose(transpose_unary(f1,     x)(x),
                        transpose_unary(f1_ref, x)(x))

  def test_transpose_transpose_transpose(self):
    @custom_transpose(jnp.ones(2))
    def fn(r, x): return x / r
    @custom_transpose(jnp.ones(2))
    def tp(r, t): return t / (2. * r)  # nb: untrue transpose

    fn.def_transpose(tp)
    tp.def_transpose(fn)

    def f_(x, y):
      return x + fn(y, x)

    x = jnp.ones(2) * 6.
    y = jnp.ones(2) * 3.
    f = lambda x: f_(x, y)
    ft   = transpose_unary(f,   x)
    ftt  = transpose_unary(ft,  x)
    fttt = transpose_unary(ftt, x)
    self.assertAllClose(ft(x), x + tp(y, x))
    self.assertAllClose(f(x),  ftt(x))
    self.assertAllClose(ft(x), fttt(x))

  def test_scalar_to_vector(self):
    def f(c, x):
      @custom_transpose([0., 0.])
      def fn(_, x):
        return [x, x]

      @fn.def_transpose
      def tp(_, t):
        t1, t2 = t
        return t1 + t2

      return fn((), c * x)

    def f_ref(c, x):
      return [c * x, c * x]

    c, x = 2., 3.
    t = [4., 5.]
    self.assertAllClose(f(c, x), f_ref(c, x))
    self.assertAllClose(transpose_unary(partial(f,     c), x)(t),
                        transpose_unary(partial(f_ref, c), x)(t))

  def test_nested(self):
    # identity function with an untrue transpose of 0
    def id_(x):
      f = custom_transpose(lambda _, x: x)
      t = custom_transpose(lambda _, t: 0.)
      f.def_transpose(t)
      t.def_transpose(f)
      return f((), x)

    # identity function with an untrue transpose of 7, and where both
    # forward and transpose have custom transpositions that should
    # never end up invoked.
    def f(x):
      f_ = custom_transpose(lambda _, x: id_(x))
      t_ = custom_transpose(lambda _, t: id_(7.))
      f_.def_transpose(t_)
      t_.def_transpose(f_)
      return f_((), x)

    x = 5.
    id_t  = transpose_unary(id_,  x)
    id_tt = transpose_unary(id_t, x)
    ft   = transpose_unary(f,    x)
    ftt  = transpose_unary(ft,   x)
    fttt = transpose_unary(ftt,  x)

    self.assertAllClose(id_(x),   x)
    self.assertAllClose(id_t(x),  0.)
    self.assertAllClose(id_tt(x), x)

    self.assertAllClose(f(x),    x)
    self.assertAllClose(ft(x),   7.)
    self.assertAllClose(ftt(x),  x)
    self.assertAllClose(fttt(x), 7.)

  def test_one_degree(self):
    T = lambda f: transpose_unary(f, 0.)

    @custom_transpose
    def f(_, z): return 2. * z
    @f.def_transpose
    def ft(_, z): return 3. * z

    f = partial(f, ())
    self.assertAllClose(2., f(1.))
    self.assertAllClose(3., T(f)(1.))
    self.assertAllClose(3., T(T(f))(1.))
    self.assertAllClose(3., T(T(T(f)))(1.))
    self.assertAllClose(3., T(T(T(T(f))))(1.))  # ...

  def test_two_degrees(self):
    T = lambda f: transpose_unary(f, 0.)

    @custom_transpose
    def f(_, z): return 2. * z

    @f.def_transpose
    @custom_transpose
    def ft(_, z): return 3. * z

    @ft.def_transpose
    def ftt(_, z): return 7. * z

    f = partial(f, ())
    self.assertAllClose(2., f(1.))
    self.assertAllClose(3., T(f)(1.))
    self.assertAllClose(7., T(T(f))(1.))
    self.assertAllClose(7., T(T(T(f)))(1.))
    self.assertAllClose(7., T(T(T(T(f))))(1.))  # ...

  def test_symmetric(self):
    T = lambda f: transpose_unary(f, 0.)

    @custom_transpose
    def f(_, z): return 2. * z
    @custom_transpose
    def g(_, z): return 3. * z

    f.def_transpose(g)
    g.def_transpose(f)

    f = partial(f, ())
    self.assertAllClose(2., f(1.))
    self.assertAllClose(3., T(f)(1.))
    self.assertAllClose(2., T(T(f))(1.))
    self.assertAllClose(3., T(T(T(f)))(1.))
    self.assertAllClose(2., T(T(T(T(f))))(1.))  # ...

  def test_recursive(self):
    T = lambda f: transpose_unary(f, 0.)

    @custom_transpose
    def f(c, z): return c * z

    @f.def_transpose
    def ft(c, z): return f(c + 1., z)

    g = partial(f, 1.)
    self.assertAllClose(1., g(1.))
    self.assertAllClose(2., T(g)(1.))
    self.assertAllClose(3., T(T(g))(1.))
    self.assertAllClose(4., T(T(T(g)))(1.))
    self.assertAllClose(5., T(T(T(T(g))))(1.))  # ...

  def test_jvp_lin(self):
    def f(x, y):
      @custom_transpose(jnp.ones(2))
      def fn(r, x): return x / r
      @fn.def_transpose
      def tp(r, t): return t / r
      return x + fn(y, x)

    def f_ref(x, y): return x + x / y

    x, y, tx = 6., 3., 1.
    g = lambda x: f(x, y)
    g_ref = lambda x: f_ref(x, y)
    self.assertAllClose(api.jvp(g, [x], [tx]), api.jvp(g_ref, [x], [tx]))

  def test_jvp_res(self):
    raise unittest.SkipTest('unimplemented')  # TODO(frostig)

    def f(x, y):
      @custom_transpose(jnp.ones(2))
      def fn(r, x): return x / r
      @fn.def_transpose
      def tp(r, t): return t / r
      return x + fn(y, x)

    def f_ref(x, y): return x + x / y

    x, y, ty = 6., 3., 1.
    g = lambda y: f(x, y)
    g_ref = lambda y: f_ref(x, y)
    self.assertAllClose(api.jvp(g, [y], [ty]), api.jvp(g_ref, [y], [ty]))

  def test_jvp_both(self):
    raise unittest.SkipTest('unimplemented')  # TODO(frostig)

    def f(x, y):
      @custom_transpose(jnp.ones(2))
      def fn(r, x): return x / r
      @fn.def_transpose
      def tp(r, t): return t / r
      return x + fn(y, x)

    def f_ref(x, y): return x + x / y

    x, y, tx, ty = 6., 3., 1., 1.
    self.assertAllClose(api.jvp(f,     [x, y], [tx, ty]),
                        api.jvp(f_ref, [x, y], [tx, ty]))

  def test_make_jaxpr(self):
    def f(x, y):
      @custom_transpose(jnp.ones(2))
      def fn(r, x): return x / r
      @fn.def_transpose
      def tp(r, t): return 2 * t / r

      return x + fn(y, x)

    x = jnp.ones(2) * 6.
    y = jnp.ones(2) * 3.
    f_ = lambda x: f(x, y)
    f_t = transpose_unary(f_, x)

    jaxpr = api.make_jaxpr(f_)(x)
    self.assertIn('custom_transpose_call', str(jaxpr))

    jaxpr_t = api.make_jaxpr(f_t)(x)
    self.assertNotIn('custom_transpose_call', str(jaxpr_t))

  def test_jit(self):
    def f(x, y):
      @custom_transpose(jnp.ones(2))
      def fn(r, x): return x / r
      @fn.def_transpose
      def tp(r, t): return 2 * t / r

      return x + fn(y, x)

    x = jnp.ones(2) * 6.
    y = jnp.ones(2) * 3.
    self.assertAllClose(f(x, y), jax.jit(f)(x, y))

    f_ = lambda x: f(x, y)
    f_t = transpose_unary(f_, x)
    g_ = jax.jit(f_)
    g_t = transpose_unary(g_, x)
    self.assertAllClose(f_(x), jax.jit(f_)(x))
    self.assertAllClose(f_t(x), jax.jit(f_t)(x))
    self.assertAllClose(f_(x), g_(x))
    self.assertAllClose(f_t(x), g_t(x))

  def test_jit_recursive(self):
    def f(x, y):
      @custom_transpose(jnp.ones(2))
      def fn(r, x): return x / r
      @fn.def_transpose
      def tp(r, t): return 2 * fn(r, t)

      return x + fn(y, x)

    x = jnp.ones(2) * 6.
    y = jnp.ones(2) * 3.
    self.assertAllClose(f(x, y), jax.jit(f)(x, y))

    f_ = lambda x: f(x, y)
    f_t = transpose_unary(f_, x)
    g_ = jax.jit(f_)
    g_t = transpose_unary(g_, x)
    self.assertAllClose(f_(x), jax.jit(f_)(x))
    self.assertAllClose(f_t(x), jax.jit(f_t)(x))
    self.assertAllClose(f_(x), g_(x))
    self.assertAllClose(f_t(x), g_t(x))

  def test_jit_signature_deprecation(self):
    fun = lambda x: x
    if deprecations.is_accelerated('jax-jit-positional-args'):
      with self.assertRaisesRegex(TypeError, r'jit\(\) got some positional-only arguments passed as keyword arguments.*'):
        jax.jit(fun=fun)
      with self.assertRaisesRegex(TypeError, r'jit\(\) takes 1 positional argument but 2 were given.*'):
        jax.jit(fun, None)
    else:
      with self.assertWarnsRegex(DeprecationWarning, r'jax\.jit: passing fun by keyword is deprecated.*'):
        jax.jit(fun=fun)
      with self.assertWarnsRegex(DeprecationWarning, r'jax\.jit: passing optional arguments by position is deprecated.*'):
        jax.jit(fun, None)

  def test_cond(self):
    def f(x, y):
      @custom_transpose(jnp.ones(2))
      def fn(r, x): return x / r
      @fn.def_transpose
      def tp(r, t): return 2 * t / r

      return x + fn(y, x)

    def cond_wrap(f):
      return lambda i, x: lax.cond(i > 0, f, lambda x: x, x)

    i = 7.
    x = jnp.ones(2) * 6.
    y = jnp.ones(2) * 3.

    f_ = lambda x: f(x, y)
    f_t = transpose_unary(f_, x)
    g_ = partial(cond_wrap(f_), i)
    g_t = transpose_unary(g_, x)

    self.assertAllClose(f_(x), g_(x))
    self.assertAllClose(f_t(x), g_t(x))

  def test_cond_recursive(self):
    def f(x, y):
      @custom_transpose(jnp.ones(2))
      def fn(r, x): return x / r
      @fn.def_transpose
      def tp(r, t): return 2 * fn(r, t)

      return x + fn(y, x)

    def cond_wrap(f):
      return lambda i, x: lax.cond(i > 0, f, lambda x: x, x)

    i = 7.
    x = jnp.ones(2) * 6.
    y = jnp.ones(2) * 3.

    f_ = lambda x: f(x, y)
    f_t = transpose_unary(f_, x)
    g_ = partial(cond_wrap(f_), i)
    g_t = transpose_unary(g_, x)

    self.assertAllClose(f_(x), g_(x))
    self.assertAllClose(f_t(x), g_t(x))

  def test_compose_custom_jvp(self):
    @jax.custom_jvp
    def f(x):
      return jnp.sin(x)

    @f.defjvp
    def f_jvp(primals, tangents):
      x, = primals
      dx, = tangents
      return f(x), g(x, dx)

    @custom_transpose
    def g(x, dx):
      return jnp.cos(x) * dx

    @g.def_transpose
    def gt(x, t):
      return jnp.cos(x) * t

    with config.use_direct_linearize(True):
      self.assertAllClose(jax.grad(f)(0.5), jnp.cos(0.5))

  def test_input_none(self):
    # ref: https://github.com/jax-ml/jax/issues/29009
    @jax.custom_jvp
    def f(x, y): return y
    @f.defjvp
    def f_jvp(p, t): return f(*p), g(p, t)

    @custom_transpose(jnp.float32(0))
    def g(r, x): return x[1]
    @g.def_transpose
    def gt(r, t): return None, jnp.zeros_like(r[1])

    jax.grad(f, argnums=(1,))(None, jnp.float32(2))  # doesn't crash


class CustomDceTest(jtu.JaxTestCase):

  def test_basic(self):
    @jax.experimental.custom_dce.custom_dce
    def f(x):
      return jnp.sin(x), jnp.cos(x)

    @f.def_dce
    def rule(used_outs, x):
      return (
          jnp.exp(x) if used_outs[0] else None,
          jnp.sqrt(x) if used_outs[1] else None,
      )

    x = jnp.array(1.1234)
    self.assertAllClose(jax.jit(lambda x: f(x)[0])(x), jnp.exp(x))
    self.assertAllClose(jax.jit(lambda x: f(x)[1])(x), jnp.sqrt(x))

  def test_recursive(self):
    @jax.experimental.custom_dce.custom_dce
    def f(x):
      return jnp.exp(x), 10 * jnp.sqrt(x)

    @f.def_dce
    def f_dce(used_outs, x):
      return [2 * v if used else None for used, v in zip(used_outs, f(x))]

    x = 1.1234
    expected = f(x)
    self.assertAllClose(jax.jit(lambda x: f(x)[0])(x), 2 * expected[0])
    self.assertAllClose(jax.jit(lambda x: f(x)[1])(x), 2 * expected[1])

  def test_multiple_rounds(self):
    @jax.experimental.custom_dce.custom_dce
    def f(x, y, z):
      return jnp.sin(x), jnp.sin(y), jnp.sin(z)

    @f.def_dce
    def rule(used_outs, x, y, z):
      patterns.append(used_outs)
      outs = [
          jnp.cos(v) if used else None for used, v in zip(used_outs, (x, y, z))
      ]
      return outs

    patterns = []
    x, y, z = jnp.array(1.), jnp.array(2.), jnp.array(3.)
    jaxpr = jax.make_jaxpr(f)(x, y, z).jaxpr
    new_jaxpr, used_ins = pe.dce_jaxpr(jaxpr, [True, False, True])
    assert used_ins == [True, False, True]
    new_jaxpr, used_ins = pe.dce_jaxpr(new_jaxpr, [True, False])
    assert used_ins == [True, False]
    assert patterns == [(True, False, True), (True, False, False)], patterns

  def test_batching(self):
    @jax.experimental.custom_dce.custom_dce
    def f(x, y):
      return jnp.sin(x), jnp.sin(y)

    @f.def_dce
    def rule(used_outs, x, y):
      return (
          jnp.cos(x) if used_outs[0] else None,
          jnp.cos(y) if used_outs[1] else None,
      )

    x = jnp.linspace(-0.1, 0.2, 5)
    y = jnp.linspace(3.0, 4.0, 5)
    self.assertAllClose(jax.vmap(f)(x, y), f(x, y))
    self.assertAllClose(
        jax.jit(lambda *args: jax.vmap(f)(*args)[0])(x, y), jnp.cos(x)
    )
    self.assertAllClose(
        jax.vmap(jax.jit(lambda *args: f(*args)[0]))(x, y), jnp.cos(x)
    )
    self.assertAllClose(
        jax.jit(lambda *args: jax.vmap(f)(*args)[1])(x, y), jnp.cos(y)
    )
    self.assertAllClose(
        jax.vmap(jax.jit(lambda *args: f(*args)[1]))(x, y), jnp.cos(y)
    )

  def test_composes_with_custom_vjp(self):
    # custom_dce must be the "outer" decorator (for now!) because custom_vjp
    # doesn't pass through DCE.
    @jax.experimental.custom_dce.custom_dce
    @jax.custom_vjp
    def f(x, y):
      return jnp.sin(x) * y, x * jnp.sin(y)

    @f.def_dce
    def f_dce_rule(used_outs, x, y):
      return (
          jnp.cos(x) * y if used_outs[0] else None,
          x * jnp.cos(y) if used_outs[1] else None,
      )

    def f_fwd(x, y):
      return f(x, y), (x, jnp.cos(x), jnp.sin(x), y, jnp.cos(y), jnp.sin(y))

    def f_bwd(res, g):
      ga, gb = g
      x, cos_x, sin_x, y, cos_y, sin_y = res
      return (cos_x * ga * y + sin_y * gb, sin_x * ga + x * cos_y * gb)

    f.defvjp(f_fwd, f_bwd)

    x, y = jnp.array(1.), jnp.array(2.)
    self.assertAllClose(jax.jit(lambda *args: f(*args)[0])(x, y),
                        jnp.cos(x) * y)
    jax.grad(lambda *args: f(*args)[0])(x, y)  # Doesn't crash.

  def test_can_optimize_remat(self):
    @jax.custom_vjp
    def f(x):
      return jnp.tan(x)

    @jax.experimental.custom_dce.custom_dce
    def f_fwd(x):
      return jnp.sin(x), (x,)

    @f_fwd.def_dce
    def f_dce_rule(used_outs, x):
      used_prim, used_res = used_outs
      used_res, = used_res
      if not used_res:
        return f(x), None
      prim, res = f_fwd(x)
      return prim if used_prim else None, res

    def f_bwd(res, g):
      x, = res
      cos_x = jnp.cos(x)
      return (cos_x * g,)

    f.defvjp(f_fwd, f_bwd)

    def temp(x):
      out = jax.remat(f)(x)
      out = out ** 2
      return out

    v, g = jax.value_and_grad(temp)(3.2)
    self.assertAllClose(v, jnp.tan(3.2)**2)

  def test_static_argnums(self):
    @partial(jax.experimental.custom_dce.custom_dce, static_argnums=(0,))
    def g(f, x):
      return f(x), 10 * f(x)

    @g.def_dce
    def g_dce(f, used_outs, x):  # note: static_argnums are always passes first
      self.assertTrue(callable(f))
      return [2 * v if used else None for used, v in zip(used_outs, g(f, x))]

    x = 1.1234
    f = lambda x: jnp.exp(x)
    expected = g(f, x)
    self.assertAllClose(jax.jit(lambda x: g(f, x)[0])(x), 2 * expected[0])
    self.assertAllClose(jax.jit(lambda x: g(f, x)[1])(x), 2 * expected[1])

  def test_shape_mismatch_error(self):
    @jax.experimental.custom_dce.custom_dce
    def f(x):
      return jnp.stack((x, x)), jnp.cos(x)

    @f.def_dce
    def rule(used_outs, x):
      return (
          jnp.exp(x) if used_outs[0] else None,
          x.astype(jnp.int32) if used_outs[1] else None,
      )

    x = jnp.array(1.1234)
    with self.assertRaisesRegex(
        ValueError,
        r'Custom DCE rule .* same shapes/dtypes .* output\[0\]',
    ):
      jax.jit(lambda x: f(x)[0])(x)
    with self.assertRaisesRegex(
        ValueError,
        r'Custom DCE rule .* same shapes/dtypes .* output\[1\]',
    ):
      jax.jit(lambda x: f(x)[1])(x)

  def test_missing_output_error(self):
    @jax.experimental.custom_dce.custom_dce
    def f(x):
      return jnp.sin(x), jnp.cos(x)

    @f.def_dce
    def rule(used_outs, x):
      return None, None

    x = jnp.array(1.1234)
    with self.assertRaisesRegex(
        ValueError,
        r'Custom DCE rule .* produce values for all .* output\[0\]',
    ):
      jax.jit(lambda x: f(x)[0])(x)

  def test_consts(self):
    @jax.experimental.custom_dce.custom_dce
    def f(x):
      return np.eye(1) * jnp.sin(x), jnp.cos(x)

    @f.def_dce
    def rule(used_outs, x):
      return (
          np.full((1, 1), 2.0) * jnp.exp(x) if used_outs[0] else None,
          jnp.sqrt(x) if used_outs[1] else None,
      )

    x = jnp.array(1.1234)
    expected = rule([True, True], x)
    self.assertAllClose(jax.jit(lambda x: f(x)[0])(x), expected[0])
    self.assertAllClose(jax.jit(lambda x: f(x)[1])(x), expected[1])

  def test_resolve_kwargs_error_message(self):
    @jax.experimental.custom_dce.custom_dce
    def f(x, y, *, z=None):
      return jnp.sin(x) * y, x * jnp.sin(y)

    @f.def_dce
    def f_dce_rule(used_outs, x, y):
      self.fail("should not be executed")

    with self.assertRaisesRegex(
        TypeError,
        r"The input arguments to the custom_dce-decorated function f(.*)\n"
        r"missing a required argument: 'y'"
    ):
      f(0.5)

    with self.assertRaisesRegex(
        TypeError,
        r"The input arguments to the custom_dce-decorated function f(.*)\n"
        "The following keyword arguments could not be resolved to positions: z"
    ):
      f(0.5, 0.1, z=1.0)


class CustomVmapTest(jtu.JaxTestCase):

  def test_basic(self):
    @jax.custom_batching.custom_vmap
    def f(x): return jnp.sin(x)

    @f.def_vmap
    def rule(axis_size, in_batched, xs):
      xs_batched, = in_batched
      self.assertEqual(xs_batched, True)
      self.assertEqual(axis_size, xs.shape[0])
      return jnp.cos(xs), xs_batched

    x, xs = jnp.array(1.), jnp.arange(3)
    y = f(x)
    self.assertAllClose(y, jnp.sin(x))
    ys = api.vmap(f)(xs)
    self.assertAllClose(ys, jnp.cos(xs))

  @jax.numpy_dtype_promotion('standard')
  def test_closure(self):
    z = jnp.array([2., 1., 3.])

    @jax.custom_batching.custom_vmap
    def f(x): return z + jnp.sin(x)

    @f.def_vmap
    def rule(axis_size, in_batched, *args):
      self.assertEqual(len(in_batched), 1)
      self.assertEqual(len(args), 1)
      xs, = args
      xs_batched, = in_batched
      self.assertEqual(xs_batched, True)
      self.assertEqual(axis_size, xs.shape[0])
      return z + jnp.cos(xs), xs_batched

    x, xs = jnp.array(1.), jnp.arange(3)
    y = f(x)
    self.assertAllClose(y, z + jnp.sin(x))
    ys = api.vmap(f)(xs)
    self.assertAllClose(ys, z + jnp.cos(xs))

  def test_rule_multi_output(self):
    @jax.custom_batching.custom_vmap
    def f(x): return jnp.sin(x), jnp.cos(x)

    @f.def_vmap
    def rule(axis_size, in_batched, xs):
      return (jnp.cos(xs), jnp.sin(xs)), tuple(in_batched * 2)

    x, xs = jnp.array(1.), jnp.arange(3)
    y1, y2 = f(x)
    self.assertAllClose(y1, jnp.sin(x))
    self.assertAllClose(y2, jnp.cos(x))
    ys1, ys2 = api.vmap(f)(xs)
    self.assertAllClose(ys1, jnp.cos(xs))
    self.assertAllClose(ys2, jnp.sin(xs))

  def test_nary(self):
    @jax.custom_batching.custom_vmap
    def f(x, y): return jnp.sin(x) + y ** 2.

    @f.def_vmap
    def rule(axis_size, in_batched, xs, ys):
      self.assertEqual(in_batched, [True, True])
      self.assertEqual(axis_size, 3)
      self.assertEqual(axis_size, xs.shape[0])
      self.assertEqual(axis_size, ys.shape[0])
      return jnp.cos(xs) + ys ** 2., True

    xs, ys = jnp.arange(3.0), jnp.arange(3.0)
    zs = api.vmap(f)(xs, ys)
    self.assertAllClose(zs, jnp.cos(xs) + ys ** 2.)

  def test_nary_mixed_batching(self):
    @jax.custom_batching.custom_vmap
    def vector_dot(u, v):
      self.assertEqual(u.ndim, 1)
      self.assertEqual(v.ndim, 1)
      return u @ v

    size = 4
    vlen = 3
    in_batched_log = []

    @vector_dot.def_vmap
    def vector_dot_vmap_rule(axis_size, in_batched, u, v):
      in_batched_log.append(in_batched)
      self.assertEqual(axis_size, size)
      u_batched, v_batched = in_batched
      if u_batched:
        self.assertEqual(u.ndim, 2)
        self.assertEqual(u.shape[0], size)
      else:
        self.assertEqual(u.ndim, 1)
        self.assertEqual(u.shape[0], vlen)
      if v_batched:
        self.assertEqual(v.ndim, 2)
        self.assertEqual(v.shape[0], size)
      else:
        self.assertEqual(v.ndim, 1)
        self.assertEqual(v.shape[0], vlen)
      if u_batched and v_batched:
        out = jnp.sum(u * v, axis=1)
      else:
        out = u @ v if u_batched else v @ u
      return out, u_batched or v_batched

    f = vector_dot
    v = lambda *shape: jnp.ones(shape)

    y = api.vmap(f, in_axes=(0, None))(v(4, 3), v(3))
    self.assertAllClose(y, v(4, 3) @ v(3))
    y = api.vmap(f, in_axes=(1, None))(v(3, 4), v(3))
    self.assertAllClose(y, v(3, 4).T @ v(3))
    y = api.vmap(f, in_axes=(None, 0))(v(3), v(4, 3))
    self.assertAllClose(y, v(3) @ v(4, 3).T)
    y = api.vmap(f, in_axes=(0, 0))(v(4, 3), v(4, 3))
    self.assertAllClose(y, jnp.sum(v(4, 3) * v(4, 3), axis=1))
    self.assertEqual(in_batched_log[0], [True, False])
    self.assertEqual(in_batched_log[1], [True, False])
    self.assertEqual(in_batched_log[2], [False, True])
    self.assertEqual(in_batched_log[3], [True, True])

  def test_rule_input_signature(self):
    @jax.custom_batching.custom_vmap
    def f(x): return jnp.sin(x)

    rule_args = []

    @f.def_vmap
    def rule(axis_size, in_batched, xs):
      rule_args.append((axis_size, in_batched))
      return jnp.cos(xs), in_batched[0]

    xs = jnp.arange(3)
    _ = api.vmap(f)(xs)
    (axis_size, in_batched), = rule_args
    self.assertIs(type(axis_size), int)
    self.assertIs(type(in_batched), list)
    self.assertEqual(len(in_batched), 1)

  def test_rule_output_vs_batching_output_mismatch(self):
    @jax.custom_batching.custom_vmap
    def f(x): return jnp.sin(x)

    @f.def_vmap
    def test_rule_abc(axis_size, in_batched, xs):
      return [jnp.sin(xs), jnp.cos(xs)], in_batched

    xs = jnp.arange(3)
    self.assertRaisesRegex(
        ValueError,
        'structure of output value and output batching specification '
        r'returned by custom vmap rule \(test_rule_abc\) do not match.*',
        lambda: api.vmap(f)(xs))

  def test_rule_vs_call_output_mismatch(self):
    @jax.custom_batching.custom_vmap
    def f(x): return jnp.sin(x)

    @f.def_vmap
    def test_rule_abc2(axis_size, in_batched, xs):
      return [jnp.sin(xs)], in_batched

    xs = jnp.arange(3)
    self.assertRaisesRegex(
        ValueError,
        r'structure of output returned by custom vmap rule \(test_rule_abc2\) '
        r'does not match that of original custom-vmapped function.*',
        lambda: api.vmap(f)(xs))

  def test_jvp_basic(self):
    @jax.custom_batching.custom_vmap
    def f(x): return jnp.sin(x)

    @f.def_vmap
    def rule(axis_size, in_batched, xs):
      self.assertEqual(axis_size, 3)
      self.assertEqual(in_batched, [True])
      return jnp.cos(xs), in_batched[0]

    f_jvp = lambda x, tx: api.jvp(f, [x], [tx])

    x, tx = jnp.array(1.), jnp.array(2.)
    xs, txs = jnp.arange(3.), jnp.arange(3.) * 2.

    y, ty = f_jvp(x, tx)
    self.assertAllClose(y, jnp.sin(x))
    self.assertAllClose(ty, jnp.cos(x) * tx)

    ys, tys = api.vmap(f_jvp)(xs, txs)
    self.assertAllClose(ys, jnp.cos(xs))
    self.assertAllClose(tys, -jnp.sin(xs) * txs)

    ys, tys = api.jvp(api.vmap(f), [xs], [txs])
    self.assertAllClose(ys, jnp.cos(xs))
    self.assertAllClose(tys, -jnp.sin(xs) * txs)

  @jax.numpy_dtype_promotion('standard')
  def test_jvp_closure(self):
    z = jnp.array([2., 1., 3.])
    def bcast(x): return z + x - z

    @jax.custom_batching.custom_vmap
    def f(x): return z + jnp.sin(x)

    @f.def_vmap
    def rule(axis_size, in_batched, xs):
      self.assertEqual(axis_size, 3)
      self.assertEqual(in_batched, [True])
      return z + jnp.cos(xs), in_batched[0]

    f_jvp = lambda x, tx: api.jvp(f, [x], [tx])

    x, tx = jnp.array(1.), jnp.array(2.)
    xs, txs = jnp.arange(3.), jnp.arange(3.) * 2.

    y, ty = f_jvp(x, tx)
    self.assertAllClose(y, z + jnp.sin(x))
    self.assertAllClose(ty, bcast(jnp.cos(x)) * tx)

    ys, tys = api.vmap(f_jvp)(xs, txs)
    self.assertAllClose(ys, z + jnp.cos(xs))
    self.assertAllClose(tys, bcast(-jnp.sin(xs)) * txs)

    ys, tys = api.jvp(api.vmap(f), [xs], [txs])
    self.assertAllClose(ys, z + jnp.cos(xs))
    self.assertAllClose(tys, bcast(-jnp.sin(xs)) * txs)

  def test_jvp_nary(self):
    @jax.custom_batching.custom_vmap
    def f(x, y): return jnp.sin(x) + y

    @f.def_vmap
    def rule(axis_size, in_batched, xs, ys):
      self.assertEqual(axis_size, 3)
      self.assertEqual(in_batched, [True, True])
      return jnp.cos(xs) + ys, True

    f_jvp = lambda x, y, tx, ty: api.jvp(f, [x, y], [tx, ty])

    x, y, tx, ty = jnp.arange(4.)
    xs, ys, txs, tys = 4. + jnp.arange(3. * 4).reshape((4, 3))

    zs, tzs = api.vmap(f_jvp)(xs, ys, txs, tys)
    self.assertAllClose(zs, jnp.cos(xs) + ys)
    self.assertAllClose(tzs, -jnp.sin(xs) * txs + tys)

    zs, tzs = api.jvp(api.vmap(f), [xs, ys], [txs, tys])
    self.assertAllClose(zs, jnp.cos(xs) + ys)
    self.assertAllClose(tzs, -jnp.sin(xs) * txs + tys)

  def test_jvp_extra_batched_tangents(self):
    @jax.custom_batching.custom_vmap
    def f(x): return jnp.sin(x)

    @f.def_vmap
    def rule(axis_size, in_batched, xs):
      self.assertEqual(axis_size, 3)
      self.assertEqual(in_batched, [False])
      return jnp.cos(xs), in_batched[0]

    f_jvp = lambda x, tx: api.jvp(f, [x], [tx])

    txs = 2. + jnp.arange(3.)
    x = jnp.array(1, dtype=txs.dtype)
    y, tys = api.vmap(f_jvp, in_axes=(None, 0), out_axes=(None, 0))(x, txs)
    self.assertAllClose(y, jnp.cos(x))
    self.assertAllClose(tys, -jnp.sin(x) * txs)

  def test_jacfwd(self):
    # jacfwd is another way to exercise extra-batched tangents

    @jax.custom_batching.custom_vmap
    def f(x): return jnp.sin(x)

    @f.def_vmap
    def rule(axis_size, in_batched, xs):
      self.assertEqual(axis_size, 3)
      self.assertEqual(in_batched, [False])
      return jnp.cos(xs), in_batched[0]

    x = jnp.arange(3.) + .72
    j = api.jacfwd(f)(x)
    self.assertAllClose(j, -jnp.diag(jnp.sin(x)))

  def test_jvp_extra_batched_primals(self):
    @jax.custom_batching.custom_vmap
    def f(x): return jnp.sin(x)

    @f.def_vmap
    def rule(axis_size, in_batched, xs):
      self.assertEqual(axis_size, 3)
      self.assertEqual(in_batched, [False])
      return jnp.cos(xs), in_batched[0]

    f_jvp = lambda x, tx: api.jvp(f, [x], [tx])

    xs = jnp.arange(3.)
    tx = jnp.array(4, dtype=xs.dtype)
    ys, tys = api.vmap(f_jvp, in_axes=(0, None))(xs, tx)
    self.assertAllClose(ys, jnp.cos(xs))
    self.assertAllClose(tys, -jnp.sin(xs) * tx)

  def test_jvp_extra_batched_primals_with_linear_vmap_rule(self):
    # When a function is linear, its Jacobian is constant. JAX's JVP
    # of linear functions takes advantage of this: when mapping over a
    # batch of primals relative to a fixed (i.e. symbolically
    # replicated) tangent, output tangents remain replicated as well
    # (i.e. JAX will not broadcast them). This is true in general, and
    # this test checks that vmapped JVPs continue to behave this way
    # when custom_vmap is involved and the custom vmap rule is linear.

    @jax.custom_batching.custom_vmap
    def f_linear(x): return 7. * x

    @f_linear.def_vmap
    def linear_rule(axis_size, in_batched, xs):
      return 11. * xs, in_batched[0]

    @jax.custom_batching.custom_vmap
    def f_nonlinear(x): return jnp.sin(x)

    @f_nonlinear.def_vmap
    def nonlinear_rule(axis_size, in_batched, xs):
      return jnp.cos(xs), in_batched[0]

    f_lin_jvp = lambda x, tx: api.jvp(f_linear, [x], [tx])
    f_non_jvp = lambda x, tx: api.jvp(f_nonlinear, [x], [tx])
    xs = jnp.arange(3.)
    tx = jnp.array(4., dtype=xs.dtype)

    # doesn't err
    _ = api.vmap(f_lin_jvp, in_axes=(0, None), out_axes=(0, None))(xs, tx)

    # does err
    self.assertRaisesRegex(
        ValueError, "at vmap out_axes",
        lambda: api.vmap(
            f_non_jvp, in_axes=(0, None), out_axes=(0, None))(xs, tx))

  def test_jvp_dataflow_violation(self):
    # The jvp-of-custom-vmap machinery should not assume the standard
    # dataflow constraint on the JVP of the custom vmap rule (primal
    # outputs independent of tangent inputs). Both jvp and vmap are
    # "forward" transformations under which, at present, we don't
    # enforce the JVP dependence diagram. Because output primals can
    # depend on input tangents, extra-batched input tangents can
    # create batched output primals, as this test checks.

    @jax.custom_jvp
    def cos_with_invalid_dataflow_jvp(x): return jnp.cos(x)

    @cos_with_invalid_dataflow_jvp.defjvp
    def invalid_dataflow_jvp(x, tx):
      [x], [tx] = x, tx
      return jnp.cos(x * tx), tx

    @jax.custom_batching.custom_vmap
    def f(x): return jnp.sin(x)

    @f.def_vmap
    def rule(axis_size, in_batched, xs):
      return cos_with_invalid_dataflow_jvp(xs), in_batched[0]

    f_jvp = lambda x, tx: api.jvp(f, [x], [tx])
    txs = 2. + jnp.arange(3.)
    x = jnp.array(1, dtype=txs.dtype)

    # doesn't err
    ys, tys = api.vmap(f_jvp, in_axes=(None, 0))(x, txs)
    self.assertAllClose(ys, jnp.cos(x * txs))
    self.assertAllClose(tys, txs)

    # does err
    self.assertRaisesRegex(
        ValueError, "at vmap out_axes",
        lambda: api.vmap(
            f_jvp, in_axes=(None, 0), out_axes=(None, 0))(x, txs))

  def test_tree(self):
    tree_sin = partial(jax.tree.map, jnp.sin)
    tree_cos = partial(jax.tree.map, jnp.cos)

    x, xs = jnp.array(1.), jnp.arange(3)
    x  = (x,  [x  + 1, x  + 2], [x  + 3], x  + 4)
    xs = (xs, [xs + 1, xs + 2], [xs + 3], xs + 4)
    in_batched_ref = jax.tree.map(lambda _: True, x)

    @jax.custom_batching.custom_vmap
    def f(xs): return tree_sin(xs)

    @f.def_vmap
    def rule(axis_size, in_batched, xs):
      self.assertEqual(in_batched, [in_batched_ref])
      sz, = {z.shape[0] for z in jax.tree.leaves(xs)}
      self.assertEqual(axis_size, sz)
      return tree_cos(xs), in_batched[0]

    y = f(x)
    self.assertAllClose(y, tree_sin(x))
    ys = api.vmap(f)(xs)
    self.assertAllClose(ys, tree_cos(xs))

  def test_tree_with_nones(self):
    tree_sin = partial(jax.tree.map, jnp.sin)
    tree_cos = partial(jax.tree.map, jnp.cos)

    x, xs = jnp.array(1.), jnp.arange(3)
    x  = (x,  [x  + 1, None], [x  + 3], None)
    xs = (xs, [xs + 1, None], [xs + 3], None)
    in_batched_ref = jax.tree.map(lambda _: True, x)

    @jax.custom_batching.custom_vmap
    def f(xs): return tree_sin(xs)

    @f.def_vmap
    def rule(axis_size, in_batched, xs):
      self.assertEqual(in_batched, [in_batched_ref])
      sz, = {z.shape[0] for z in jax.tree.leaves(xs)}
      self.assertEqual(axis_size, sz)
      return tree_cos(xs), in_batched[0]

    y = f(x)
    self.assertAllClose(y, tree_sin(x))
    ys = api.vmap(f)(xs)
    self.assertAllClose(ys, tree_cos(xs))

  def test_jit(self):
    @jax.custom_batching.custom_vmap
    def f(x): return jnp.sin(x)

    @f.def_vmap
    def rule(axis_size, in_batched, xs):
      self.assertEqual(in_batched, [True])
      self.assertEqual(axis_size, xs.shape[0])
      return jnp.cos(xs), in_batched[0]

    x, xs = jnp.array(1.), jnp.arange(3)
    self.assertAllClose(f(x), jit(f)(x))
    self.assertAllClose(jit(api.vmap(f))(xs), api.vmap(f)(xs))
    self.assertAllClose(api.vmap(jit(f))(xs), api.vmap(f)(xs))

  def test_sequential_vmap_basic(self):
    @jax.custom_batching.sequential_vmap
    def f(x):
      return x + 1.

    def vmap_ref(xs):
      return lax.map(f, xs)

    xs = jnp.arange(3.)
    jaxpr = api.make_jaxpr(api.vmap(f))(xs)
    jaxpr_ref = api.make_jaxpr(vmap_ref)(xs)

    self.assertEqual(str(jaxpr), str(jaxpr_ref))

  def test_sequential_vmap_nary_same_batching(self):
    @jax.custom_batching.sequential_vmap
    def f(x, y):
      return x + y

    def vmap_ref(xs, ys):
      return lax.map(lambda args: f(*args), (xs, ys))

    xs, ys = jnp.arange(3.), 4. + jnp.arange(3.)
    jaxpr = api.make_jaxpr(api.vmap(f))(xs, ys)
    jaxpr_ref = api.make_jaxpr(vmap_ref)(xs, ys)

    self.assertEqual(str(jaxpr), str(jaxpr_ref))

  def test_sequential_vmap_nary_mixed_batching(self):
    @jax.custom_batching.sequential_vmap
    def f(x, y):
      return x + y

    def vmap_ref(xs, y):
      return lax.map(lambda x: f(x, y), xs)

    xs, y = jnp.arange(3.), 4.
    jaxpr = api.make_jaxpr(api.vmap(f, in_axes=(0, None)))(xs, y)
    jaxpr_ref = api.make_jaxpr(vmap_ref)(xs, y)

    self.assertEqual(str(jaxpr), str(jaxpr_ref))

  @parameterized.named_parameters(
    ("1", 1),
    ("8", 4),
    ("12", 8),
    ("16", 16),
  )
  def test_batch_map_basic(self, batch_size: int):
    def f(x):
      self.assertEqual(x.shape, ())
      return x**2

    x = np.arange(16)
    y = jax.lax.map(f, x, batch_size=batch_size)

    np.testing.assert_array_equal(y, x**2)

  @parameterized.named_parameters(
    ("1", 1),
    ("8", 4),
    ("12", 8),
    ("16", 16),
  )
  def test_batch_map_pytrees(self, batch_size: int):
    f = lambda x: {'b': x['a'] ** 2}
    inputs = {'a': np.arange(16)}
    expected = np.arange(16) ** 2

    outputs = jax.lax.map(f, inputs, batch_size=batch_size)
    self.assertAllClose(outputs['b'], expected)

    outputs = jax.lax.map(
      f, inputs, batch_size=batch_size
    )
    self.assertAllClose(outputs['b'], expected)

  def test_batch_divides_axis(self):
    def f(t):
      x, a = t
      self.assertEqual(x.shape, (4,))
      return (x + a)**2

    x = jax.random.randint(jax.random.key(0), (16, 4), -10, 10)
    a = jax.random.randint(jax.random.key(1), (16, 4), -10, 10)

    @jax.jit
    def g(x, a):
      return jax.lax.map(f, (x, a), batch_size=8)

    y = g(x, a)

    self.assertAllClose(y, (x + a)**2)

  def test_undefined_rule(self):
    @jax.custom_batching.custom_vmap
    def f(x): return jnp.sin(x)

    with self.assertRaisesRegex(
        AttributeError, "No batching rule defined for custom_vmap function f"):
      f(0.5)

  def test_kwargs(self):
    @jax.custom_batching.custom_vmap
    def f(x): return jnp.sin(x)

    @f.def_vmap
    def rule(axis_size, in_batched, xs):
      xs_batched, = in_batched
      self.assertEqual(xs_batched, True)
      self.assertEqual(axis_size, xs.shape[0])
      return jnp.cos(xs), xs_batched

    x, xs = jnp.array(1.), jnp.arange(3)
    y = f(x=x)
    self.assertAllClose(y, jnp.sin(x))
    ys = api.vmap(f)(x=xs)
    self.assertAllClose(ys, jnp.cos(xs))

  def test_partial_eval_raises(self):
    @jax.custom_batching.custom_vmap
    def f(x):
      return jnp.sin(x)

    @f.def_vmap
    def rule(axis_size, in_batched, xs):
      del axis_size  # unused
      return jnp.cos(xs), in_batched[0]

    with self.assertRaisesRegex(
        ValueError,
        "Linearization failed to produce known values for all output primals",
    ):
      jax.grad(f)(0.5)

  def test_compose_custom_vjp(self):
    @jax.custom_vjp
    @jax.custom_batching.custom_vmap
    def f(x, y):
      return jnp.sin(x) * y

    @f.def_vmap
    def f_vmap_rule(axis_size, in_batched, xs, ys):
      return jnp.cos(xs) * ys, True

    def f_fwd(x, y):
      return f(x, y), (jnp.cos(x), jnp.sin(x), y)

    def f_bwd(res, g):
      cos_x, sin_x, y = res
      return (cos_x * g * y, sin_x * g)

    f.defvjp(f_fwd, f_bwd)

    xs = jnp.linspace(0, 1, 5)
    ys = jnp.linspace(-0.1, 0.1, 5)
    self.assertAllClose(jax.vmap(f)(xs, ys), jnp.cos(xs) * ys)
    jax.grad(f)(xs[0], ys[0])  # Doesn't crash.

  def test_compose_custom_vjp_bwd_rule(self):
    # This tests the case where both the forward and backward rules are wrapped
    # in custom_vmap.
    @jax.custom_batching.sequential_vmap
    def fun_fwd(x, y):
      return jnp.sin(x) * y, (x, y)

    @jax.custom_batching.sequential_vmap
    def fun_bwd(res, ct):
      x, y = res
      return x * ct, y * ct

    fun = jax.custom_vjp(lambda *args: fun_fwd(*args)[0])
    fun.defvjp(fun_fwd, fun_bwd)

    xs = jnp.linspace(0, 1, 5)
    y = jnp.array(0.5, dtype=xs.dtype)
    f = jax.vmap(jax.jit(fun), in_axes=(0, None))
    out, f_vjp = jax.vjp(f, xs, y)
    f_vjp(out)  # Doesn't crash.

  def test_resolve_kwargs_error_message(self):
    @jax.custom_batching.custom_vmap
    def f(x, y, *, z=None):
      return jnp.sin(x) * y

    @f.def_vmap
    def f_vmap_rule(axis_size, in_batched, xs, ys):
      self.fail("should not be executed")

    with self.assertRaisesRegex(
        TypeError,
        r"The input arguments to the custom_vmap-decorated function f(.*)\n"
        r"missing a required argument: 'y'"
    ):
      f(0.5)

    with self.assertRaisesRegex(
        TypeError,
        r"The input arguments to the custom_vmap-decorated function f(.*)\n"
        "The following keyword arguments could not be resolved to positions: z"
    ):
      f(0.5, 0.1, z=1.0)


class CustomApiTest(jtu.JaxTestCase):
  """Test interactions among the custom_{vmap,jvp,vjp,transpose,*} APIs"""

  def test_method_forwarding(self):
    @jax.custom_batching.custom_vmap
    @jax.custom_jvp
    @jax.custom_transpose.custom_transpose
    def f(x): return 2. * x

    # none of these err:
    @f.def_vmap
    def f_batch(sz, b, xs): return 2. * xs
    @f.defjvp
    def f_jvp(x, tx): return 2. * x, 2. * tx
    @f.def_transpose
    def f_transpose(x): return 2. * x

  def test_def_method_forwarding_all_permutations(self):
    for wraps in it.permutations([
        jax.custom_jvp, jax.custom_transpose.custom_transpose, jax.custom_batching.custom_vmap]):
      f = lambda x: x + 1.
      for wrap in wraps:
        f = wrap(f)
      for methods in it.permutations(['defjvp', 'def_vmap', 'def_transpose']):
        for method in methods:
          self.assertIsInstance(getattr(f, method), Callable)

    for decorators in it.permutations([
        jax.custom_vjp, jax.custom_transpose.custom_transpose, jax.custom_batching.custom_vmap]):
      f = lambda x: x + 1.
      for decorator in decorators:
        f = decorator(f)
      for methods in it.permutations(['defvjp', 'def_vmap', 'def_transpose']):
        for method in methods:
          self.assertIsInstance(getattr(f, method), Callable)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
