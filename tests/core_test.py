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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator
from collections import namedtuple
from unittest import skip

import numpy as onp
from absl.testing import absltest
from absl.testing import parameterized

from jax import api
from jax import core
from jax import numpy as np
from jax import test_util as jtu
from jax.api import jvp, linearize, vjp, jit
from jax.lax import UnshapedArray, ShapedArray, ConcreteArray
from jax.tree_util import tree_flatten, tree_unflatten, tree_multimap, tree_reduce
from jax.util import partial
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla

from jax.config import config
config.parse_flags_with_absl()

_ = pe.PartialVal((UnshapedArray(onp.float32), core.unit))
__ = pe.PartialVal((ShapedArray((), onp.float32), core.unit))

def call(f, *args):
  return jit(f)(*args)

def simple_fun(x, y):
  return np.sin(x * y)

def simple_fun_fanout(x, y):
  return np.sin(x * y) * x

def fun_with_call(x):
  return call(np.sin, x)

def fun_with_nested_calls(x):
  def f(y):
    y2 = np.sin(y) + 1.0 + (2.0 * x)

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
      return call(lambda w: call(np.sin, x) * y, 1.0) + q
      # return call(lambda w: call(error(np.sin), x) * y, 1.0) + q
    p, t = jvp(baz, (x + 1.0,), (y,))
    return t + (x * p)
  return call(bar, x)

def fun_call_jitted(x):
  @jit
  def g(z):
    return x * z

  return call(g, x)

def fun_with_two_calls(x):
  return call(np.sin, x) + call(np.cos, x)

def fun_with_call_closure(x):
  def foo(y, z):
    return (x * x) * np.sin(y) * z

  return call(foo, x, np.cos(x)) + x

def product_io_fun(x, y):
  xa = x['a']
  xb = x['b']
  y1, (y2, y3) = y
  return np.sin(xa + y2), [xb, (y1, y3)]


R = onp.random.randn
TestSpec = namedtuple('TestSpec', ['fun', 'args'])
test_specs_base = [
    TestSpec(simple_fun, (R(3, 2), R(3, 2))),
    TestSpec(simple_fun_fanout, (R(3, 2), R(3, 2))),
    TestSpec(product_io_fun, ({'a': R(2, 2), 'b': R(2, 2)},
                              (R(2, 2), (R(2, 2), R(2, 2))))),
    TestSpec(fun_with_call, (R(3, 2),)),
    TestSpec(fun_with_two_calls, (R(3, 2),)),
    TestSpec(fun_with_call_closure, (R(3, 2),)),
    TestSpec(fun_call_jitted, (R(1,),)),
    TestSpec(fun_with_nested_calls, (R(),)),
    TestSpec(fun_with_nested_calls, (R(3, 2),)),
    TestSpec(fun_with_nested_calls_2, (R(1, 2),)),
]

def jvp_unlinearized(f, primals, tangents):
  out, jvp = linearize(f, *primals)
  return out, jvp(*tangents)

test_specs = []
for ts in test_specs_base:
  test_specs.append(ts)
  test_specs.append(TestSpec(partial(jvp, ts.fun), (ts.args, ts.args)))
  test_specs.append(TestSpec(jit(ts.fun), ts.args))
  test_specs.append(TestSpec(jit(jit(ts.fun)), ts.args))
  test_specs.append(TestSpec(partial(jvp_unlinearized, ts.fun),
                             (ts.args, ts.args)))


def fwd_deriv(f):
  def df(x):
    return jvp(f, (x,), (1.0,))[1]

  return df

def check_trace_eval(f, pvals, vals, expected_out_pval):
  jaxpr, consts, out_pval, _ = api.trace_to_jaxpr(f, pvals)
  assert expected_out_pval == out_pval, (expected_out_pval, out_pval)
  output_traced = core.eval_jaxpr(jaxpr, consts, (), *vals)
  output_traced = pe.merge_pvals(output_traced, out_pval)
  output_eval = f(*vals)
  assert onp.allclose(output_traced, output_eval), \
      '\neval:         {}\ntrace + eval: {}'.format(output_eval, output_traced)


class CoreTest(jtu.JaxTestCase):

  def test_pack_unpack(self):
    # TODO(dougalm): figure out what jaxpr-tracing api to expose and re-enable
    self.skipTest("disabled")
    y = onp.array(1.0)
    def foo(x):
      x1, y1 = core.pack((x, y))
      assert y1 is y, (y1, y)
      return x1

    pe.trace_to_jaxpr(foo, (_,))

  def test_tup_add(self):
    # TODO(mattjj,dougalm): put tup_add somewhere (was in array_type.py)
    self.skipTest("disabled")
    y = onp.array(1.0)
    def foo(x):
      return np.tup_add(core.pack((x, y)))

    pe.trace_to_jaxpr(foo, (_,))

  def test_tree_multimap(self):
    class MyDict(dict):
      pass
    MyNT = namedtuple('MyNT', ['a', 'b'])
    xs = ({'a': 1}, [2, 3], MyDict({'c': 4}), MyNT(5, 6))
    ys = ({'a': 10}, [20, 30], MyDict({'c': 40}), MyNT(50, 60))
    ys_bad = ({'a': 10, 'b': 10}, [20, 30], MyDict({'c': 40}), MyNT(50, 60))
    zs = ({'a': 11}, [22, 33], MyDict({'c': 44}), MyNT(55, 66))

    f = lambda x, y: x + y
    assert tree_multimap(f, xs, ys) == zs
    try:
      tree_multimap(f, xs, ys_bad)
      assert False
    except TypeError:
      pass

  def test_print_jaxpr_compound(self):
    # TODO(dougalm): figure out what jaxpr-tracing api to expose and re-enable
    self.skipTest("disabled")
    pv = pe.PartialVal((ShapedArray((2, 3), onp.float32), core.unit))
    print(pe.trace_to_jaxpr(fun_with_call_closure, (pv,))[0])

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
    jtu.check_eq(jit(f)(*args), f(*args))

  @parameterized.parameters(test_specs)
  def test_jvp(self, f, args):
    jtu.check_jvp(f, partial(jvp, f), args)

  def test_jvp_zeros(self):
    def foo(x):
      def bar(y):
        x1, y1 = core.pack((x, y))
        return np.sin(x1 * y1)
      return jvp(bar, (3 * x,), (2 * x,))

    jtu.check_eq(jit(foo)(0.5), foo(0.5))

  def test_dynamic_subfun_context(self):
    def foo(x):
      def bar(y):
        return np.multiply(np.sin(x), y)
      return call(bar, x)

    api.trace_to_jaxpr(foo, (__,))

  def test_nested_grad(self):
    self.skipTest("disabled")  # TODO: re-enable this test.
    def foo(x):
      print(type(x), x)
      def bar(y):
        return np.cos(y) * x
      print(x * x)
      return call(bar, x*x)

    print(api.trace_to_jaxpr(api.grad(foo), (__,)))

  def test_nested(self):
    def foo(x):
      def bar(y):
        def baz(w):
          q = call(lambda x: y, x) + call(lambda: y)
          return call(lambda w: call(np.sin, x) * y, 1.0) + q
        p, t = jvp(baz, (x + 1.0,), (y,))
        return t + (x * p)

      return call(bar, x)

    api.trace_to_jaxpr(foo, (__,))

  @parameterized.parameters(test_specs)
  def test_jvp_linearized(self, f, args):
    jtu.check_jvp(f, partial(jvp_unlinearized, f), args)

  @parameterized.parameters(test_specs)
  def test_vjp(self, f, args):
    jtu.check_vjp(f, partial(vjp, f), args)

  def test_jvp_closure(self):
    def foo(x):
      def bar(y):
        return np.multiply(x, y)
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

  def test_simple_trace(self):
    def foo(x):
      return np.sin(x) + np.cos(x)
    pval = pe.PartialVal((ShapedArray((3, 2), onp.float32), core.unit))
    check_trace_eval(foo, (pval,), (onp.random.randn(3, 2),), pval)

  def test_nullary_trace(self):
    def foo():
      return 1.2
    check_trace_eval(foo, (), (), (None, 1.2))

  def test_simple_jit(self):
    def foo(x):
      if x.shape == ():
        return x + 1.
      else:
        return x + 2.

    foo2 = jit(foo)
    foo3 = jit(foo2)

    x1, y1 = onp.array(1.0), onp.array(2.0)
    assert foo(x1) == y1
    assert foo2(x1) == y1
    assert foo3(x1) == y1

    x2, y2 = onp.array([1.0, 2.0]), onp.array([3.0, 4.0])
    assert onp.all(foo(x2) == y2)
    assert onp.all(foo2(x2) == y2)
    assert onp.all(foo3(x2) == y2)

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
    d_sin = fwd_deriv(np.sin)
    d2_sin = fwd_deriv(d_sin)
    d3_sin = fwd_deriv(d2_sin)

    assert d_sin(0.0) == 1.0
    assert d2_sin(0.0) == 0.0
    assert d3_sin(0.0) == -1.0


if __name__ == '__main__':
  absltest.main()
