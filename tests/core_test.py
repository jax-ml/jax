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

from collections import namedtuple
import gc
import operator
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
      q = call(lambda w: call(np.sin, x) * y, 1.0) + q
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
    jtu.check_jvp(f, partial(jvp, f), args, rtol={onp.float32: 3e-2})

  def test_jvp_zeros(self):
    def foo(x):
      def bar(y):
        return np.sin(x * y)
      return jvp(bar, (3 * x,), (2 * x,))

    jtu.check_eq(jit(foo)(0.5), foo(0.5))

  @parameterized.parameters(test_specs)
  def test_jvp_linearized(self, f, args):
    jtu.check_jvp(f, partial(jvp_unlinearized, f), args,
                  rtol={onp.float32: 3e-2})

  @parameterized.parameters(test_specs)
  def test_vjp(self, f, args):
    jtu.check_vjp(f, partial(vjp, f), args,
                  rtol={onp.float32: 7e-2, onp.float64: 1e-5},
                  atol={onp.float32: 1e-2, onp.float64: 1e-5})

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

  def test_reference_cycles(self):
    gc.collect()

    def f(x):
      return x.sum()

    fn = partial(linearize, f)
    params = np.zeros([])

    debug = gc.get_debug()
    try:
      fn(params)
      gc.set_debug(gc.DEBUG_SAVEALL)
      self.assertEqual(gc.collect(), 0)
    finally:
      gc.set_debug(debug)


if __name__ == '__main__':
  absltest.main()
