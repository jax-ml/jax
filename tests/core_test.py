# Copyright 2018 Google LLC
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


from collections import namedtuple
import gc
import itertools as it
import operator

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from jax import core
from jax import lax
from jax import numpy as jnp
from jax import test_util as jtu
from jax.abstract_arrays import make_shaped_array
from jax.api import jvp, linearize, vjp, jit, make_jaxpr
from jax.core import UnshapedArray, ShapedArray
from jax.tree_util import tree_flatten, tree_unflatten, tree_multimap, tree_reduce, tree_leaves
from jax.util import partial
from jax.interpreters import partial_eval as pe


from jax.config import config
config.parse_flags_with_absl()

_ = pe.PartialVal.unknown(UnshapedArray(np.float32))
__ = pe.PartialVal.unknown(ShapedArray((), np.float32))

def call(f, *args):
  return jit(f)(*args)

def simple_fun(x, y):
  return jnp.sin(x * y)

def simple_fun_fanout(x, y):
  return jnp.sin(x * y) * x

def fun_with_call(x):
  return call(jnp.sin, x)

def fun_with_nested_calls(x):
  def f(y):
    y2 = jnp.sin(y) + 1.0 + (2.0 * x)

    @jit
    def g(z):
      return y2 * z * x + (x * y)

    return call(g, y)

  return call(f, x)

def error(*args):
  def f(*args):
    assert False
  return f

def fun_with_nested_calls_2(x):
  def bar(y):
    def baz(w):
      q = call(lambda x: y, x)
      q = q + call(lambda: y)
      q = q + call(lambda y: w + y, y)
      q = call(lambda w: call(jnp.sin, x) * y, 1.0) + q
      return q
    p, t = jvp(baz, (x + 1.0,), (y,))
    return t + (x * p)
  return call(bar, x)

def fun_call_jitted(x):
  @jit
  def g(z):
    return x * z

  return call(g, x)

def fun_with_two_calls(x):
  return call(jnp.sin, x) + call(jnp.cos, x)

def fun_with_call_closure(x):
  def foo(y, z):
    return (x * x) * jnp.sin(y) * z

  return call(foo, x, jnp.cos(x)) + x

def product_io_fun(x, y):
  xa = x['a']
  xb = x['b']
  y1, (y2, y3) = y
  return jnp.sin(xa + y2), [xb, (y1, y3)]


_rng = np.random.RandomState(42)
R = _rng.randn
CallSpec = namedtuple('CallSpec', ['fun', 'args'])
test_specs_base = [
    CallSpec(simple_fun, (R(3, 2), R(3, 2))),
    CallSpec(simple_fun_fanout, (R(3, 2), R(3, 2))),
    CallSpec(product_io_fun, ({'a': R(2, 2), 'b': R(2, 2)},
                              (R(2, 2), (R(2, 2), R(2, 2))))),
    CallSpec(fun_with_call, (R(3, 2),)),
    CallSpec(fun_with_two_calls, (R(3, 2),)),
    CallSpec(fun_with_call_closure, (R(3, 2),)),
    CallSpec(fun_call_jitted, (R(1,),)),
    CallSpec(fun_with_nested_calls, (R(),)),
    CallSpec(fun_with_nested_calls, (R(3, 2),)),
    CallSpec(fun_with_nested_calls_2, (R(1, 2),)),
]

def jvp_unlinearized(f, primals, tangents):
  out, jvp = linearize(f, *primals)
  return out, jvp(*tangents)

test_specs = []
for ts in test_specs_base:
  test_specs.append(ts)
  test_specs.append(CallSpec(partial(jvp, ts.fun), (ts.args, ts.args)))
  test_specs.append(CallSpec(jit(ts.fun), ts.args))
  test_specs.append(CallSpec(jit(jit(ts.fun)), ts.args))
  test_specs.append(CallSpec(partial(jvp_unlinearized, ts.fun),
                             (ts.args, ts.args)))


def fwd_deriv(f):
  def df(x):
    return jvp(f, (x,), (1.0,))[1]

  return df


class CoreTest(jtu.JaxTestCase):

  def test_tree_multimap(self):
    xs = ({'a': 1}, [2, 3])
    ys = ({'a': 10}, [20, 30])
    ys_bad = ({'a': 10, 'b': 10}, [20, 30])
    zs = ({'a': 11}, [22, 33])

    f = lambda x, y: x + y
    assert tree_multimap(f, xs, ys) == zs
    try:
      tree_multimap(f, xs, ys_bad)
      assert False
    except (TypeError, ValueError):
      pass

  def test_tree_flatten(self):
    flat, _ = tree_flatten(({'a': 1}, [2, 3], 4))
    assert flat == [1, 2, 3, 4]

  def test_tree_unflatten(self):
    tree = [(1, 2), {"roy": (3, [4, 5, ()])}]
    flat, treedef = tree_flatten(tree)
    assert flat == [1, 2, 3, 4, 5]
    tree2 = tree_unflatten(treedef, flat)
    nodes_equal = tree_multimap(operator.eq, tree, tree2)
    assert tree_reduce(operator.and_, nodes_equal)

  @parameterized.parameters(test_specs)
  def test_jit(self, f, args):
    jtu.check_close(jit(f)(*args), f(*args))

  @parameterized.parameters(test_specs)
  def test_jvp(self, f, args):
    jtu.check_jvp(f, partial(jvp, f), args, rtol={np.float32: 3e-2})

  def test_jvp_zeros(self):
    def foo(x):
      def bar(y):
        return jnp.sin(x * y)
      return jvp(bar, (3 * x,), (2 * x,))

    jtu.check_eq(jit(foo)(0.5), foo(0.5))

  @parameterized.parameters(test_specs)
  def test_jvp_linearized(self, f, args):
    jtu.check_jvp(f, partial(jvp_unlinearized, f), args,
                  rtol={np.float32: 3e-2})

  @parameterized.parameters(test_specs)
  def test_vjp(self, f, args):
    jtu.check_vjp(f, partial(vjp, f), args,
                  rtol={np.float32: 3e-1, np.float64: 1e-5},
                  atol={np.float32: 1e-2, np.float64: 1e-5})

  def test_jvp_closure(self):
    def foo(x):
      def bar(y):
        return jnp.multiply(x, y)
      return jvp(bar, (3.0,), (1.0,))[1]
    ans = jvp(foo, (1.0,), (2.0,))
    assert ans == (1.0, 2.0), ans

  def test_jit_closure(self):
    def foo(x):
      @jit
      def bar(y):
        return x + y
      return bar(0.0)
    assert jvp(foo, (1.0,), (2.0,)) == (1.0, 2.0)

  def test_simple_jit(self):
    def foo(x):
      if x.shape == ():
        return x + 1.
      else:
        return x + 2.

    foo2 = jit(foo)
    foo3 = jit(foo2)

    x1, y1 = np.array(1.0), np.array(2.0)
    assert foo(x1) == y1
    assert foo2(x1) == y1
    assert foo3(x1) == y1

    x2, y2 = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    assert np.all(foo(x2) == y2)
    assert np.all(foo2(x2) == y2)
    assert np.all(foo3(x2) == y2)

  def test_product_jit(self):
    def foo(x, tup):
      y, z = tup
      w = x + z
      return (w, {'x': y}), z

    foo2 = jit(foo)
    foo3 = jit(foo2)

    args = (1.0, (2.0, 3.0))
    expected_output = ((4.0, {'x': 2.0}), 3.0)

    assert foo(*args) == expected_output
    assert foo2(*args) == expected_output
    assert foo3(*args) == foo(*args)

  def test_jvp_2(self):
    d_sin = fwd_deriv(jnp.sin)
    d2_sin = fwd_deriv(d_sin)
    d3_sin = fwd_deriv(d2_sin)

    assert d_sin(0.0) == 1.0
    assert d2_sin(0.0) == 0.0
    assert d3_sin(0.0) == -1.0

  def test_reference_cycles(self):
    gc.collect()

    def f(x):
      return x.sum()

    fn = partial(linearize, f)
    params = jnp.zeros([])

    debug = gc.get_debug()
    try:
      fn(params)
      gc.set_debug(gc.DEBUG_SAVEALL)
      self.assertEqual(gc.collect(), 0)
    finally:
      gc.set_debug(debug)

  def test_comparing_var(self):
    newsym = core.gensym()
    a = newsym(core.abstract_unit)
    b = newsym(core.abstract_unit)
    c = newsym(core.abstract_unit)
    assert a < b < c
    assert c > b > a
    assert a != b and b != c and a != c

  def test_var_ordering(self):
    newsym = core.gensym()
    a = newsym(core.abstract_unit)
    b = newsym(core.abstract_unit)
    c = newsym(core.abstract_unit)
    for ordering in it.permutations([a, b, c]):
      assert sorted(list(ordering)) == [a, b, c]

  def test_var_compared_by_identity(self):
    a1 = core.gensym()(core.abstract_unit)
    a2 = core.gensym()(core.abstract_unit)
    assert str(a1) == str(a2)
    assert a1 != a2

  def test_var_tree_flatten(self):
    newsym = core.gensym()
    a, b, c, d = (
        newsym(core.abstract_unit), newsym(core.abstract_unit),
        newsym(core.abstract_unit), newsym(core.abstract_unit))
    syms = {c: d, a: b}
    assert 'bd' == ''.join(map(str, tree_leaves(syms)))

  def test_check_jaxpr_correct(self):
    jaxpr = make_jaxpr(lambda x: jnp.sin(x) + jnp.cos(x))(1.).jaxpr
    core.check_jaxpr(jaxpr)

  def test_check_jaxpr_eqn_mismatch(self):
    def f(x):
      return jnp.sin(x) + jnp.cos(x)

    def new_jaxpr():
      return make_jaxpr(f)(1.).jaxpr

    # jaxpr is:
    #
    # { lambda  ; a.
    #   let b = sin a
    #       c = cos a
    #       d = add b c
    #   in (d,) }
    #
    # NB: eqns[0].outvars[0] and eqns[2].invars[0] are both 'b'

    jaxpr = new_jaxpr()
    jaxpr.eqns[0].outvars[0].aval = make_shaped_array(2)   # int, not float!
    jtu.check_raises_regexp(
        lambda: core.check_jaxpr(jaxpr),
        TypeError, (r"Variable '.' inconsistently typed as ShapedArray(.*), "
                    r"bound as ShapedArray(.*) in '. = sin .'"))

    jaxpr = new_jaxpr()
    jaxpr.eqns[0].outvars[0].aval = make_shaped_array(np.ones((2, 3)))
    jtu.check_raises_regexp(
        lambda: core.check_jaxpr(jaxpr),
        TypeError, (r"Variable '.' inconsistently typed as ShapedArray(.*), "
                    r"bound as ShapedArray(.*) in '. = sin .'"))

  def test_jaxpr_dropvar_from_jit_call(self):
    def inner(x):
      return x + 1, x + 2

    def f(x):
      _, y = jit(inner)(x)
      return y + 3

    jaxpr = make_jaxpr(f)(1).jaxpr
    assert jaxpr.eqns[0].outvars[0] is core.dropvar
    core.check_jaxpr(jaxpr)

  def test_jaxpr_dropvar_from_loop(self):
    def f(x):
      _, y = lax.while_loop(lambda s: s[0] < 0.,
                            lambda s: (jnp.sin(s[0]), jnp.cos(s[1])),
                            (x, x))
      return y + 1.

    jaxpr = make_jaxpr(f)(1.).jaxpr
    assert jaxpr.eqns[0].outvars[0] is core.dropvar
    core.check_jaxpr(jaxpr)

  def test_jaxpr_dropvar_from_cond(self):
    def f(x):
      _, y = lax.cond(x < 0.,
                      lambda x: (jnp.sin(x), x + 1.),
                      lambda x: (jnp.cos(x), x + 2.),
                      x)
      return y

    jaxpr = make_jaxpr(f)(1.).jaxpr
    assert jaxpr.eqns[-1].outvars[0] is core.dropvar
    core.check_jaxpr(jaxpr)

if __name__ == '__main__':
  absltest.main()
