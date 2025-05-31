# Copyright 2020 The JAX Authors.
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

from functools import partial
import operator

from absl.testing import absltest
import jax
from jax import api_util
from jax._src import linear_util as lu
from jax._src import test_util as jtu
from jax._src import util
from jax._src.util import weakref_lru_cache
jax.config.parse_flags_with_absl()

try:
  from jax._src.lib import utils as jaxlib_utils
except:
  jaxlib_utils = None

class UtilTest(jtu.JaxTestCase):

  def test_wrapped_fun_transforms(self):
    """Test a combination of transforms."""

    def f(*args, **kwargs):
      """The function to be transformed.
      Scales the positional arguments by a factor.
      Takes only one keyword argument, the factor to scale by."""
      factor = kwargs.pop('factor', 2)  # For PY2
      assert not kwargs
      return tuple(a * factor for a in args)

    @lu.transformation_with_aux2
    def kw_to_positional(f, store, factor, *args, **kwargs):
      """A transformation with auxiliary output.
      Turns all keyword parameters into positional ones.

      On entry, append the values of the keyword arguments to the positional
      arguments. On exit, take a list of results and recreate a dictionary
      from the tail of the results. The auxiliary output is the list of
      keyword keys.
      """
      kwargs_keys = kwargs.keys()
      new_args = tuple(kwargs[k] for k in kwargs_keys)
      new_kwargs = dict(factor=factor)
      results = f(*(args + new_args), **new_kwargs)  # Yield transformed (args, kwargs)
      # Assume results correspond 1:1 to the args + new_args
      assert len(results) == len(args) + len(new_args)
      aux_output = len(new_args)
      store.store(aux_output)
      return (results[0:len(args)], dict(zip(kwargs_keys, results[len(args):])))

    # Wraps `f` as a `WrappedFun`.
    wf = lu.wrap_init(
        f,
        debug_info=api_util.debug_info("test", f, (1, 2), dict(three=3, four=4)))
    wf, out_thunk = kw_to_positional(wf, 2)
    # Call the transformed function.
    scaled_positional, scaled_kwargs = wf.call_wrapped(1, 2, three=3, four=4)
    self.assertEqual((2, 4), scaled_positional)
    self.assertEqual(dict(three=6, four=8), scaled_kwargs)
    self.assertEqual(2, out_thunk())

  def test_wrapped_fun_name(self):
    def my_function():
      return

    with self.subTest("function"):
      wrapped = lu.wrap_init(
          my_function,
          debug_info=api_util.debug_info("test", my_function, (), {}),
      )
      self.assertEqual(wrapped.__name__, my_function.__name__)

    with self.subTest("default_partial"):
      my_partial = partial(my_function)
      wrapped = lu.wrap_init(
          my_partial,
          debug_info=api_util.debug_info("test", my_partial, (), {}),
      )
      self.assertEqual(wrapped.__name__, my_function.__name__)

    with self.subTest("nested_default_partial"):
      my_partial = partial(partial(my_function))
      wrapped = lu.wrap_init(
          my_partial,
          debug_info=api_util.debug_info("test", my_partial, (), {}),
      )
      self.assertEqual(wrapped.__name__, my_function.__name__)

    with self.subTest("named_partial"):
      my_partial = partial(my_function)
      my_partial.__name__ = "my_partial"
      wrapped = lu.wrap_init(
          my_partial,
          debug_info=api_util.debug_info("test", my_partial, (), {}),
      )
      self.assertEqual(wrapped.__name__, my_partial.__name__)

    with self.subTest("lambda"):
      l = lambda: my_function()
      wrapped = lu.wrap_init(
          l,
          debug_info=api_util.debug_info("test", l, (), {}),
      )
      self.assertEqual(wrapped.__name__, "<lambda>")

    with self.subTest("unnamed_callable"):

      class MyCallable:

        def __call__(self):
          return

      my_callable = MyCallable()
      wrapped = lu.wrap_init(
          my_callable,
          debug_info=api_util.debug_info("test", my_callable, (), {}),
      )
      self.assertEqual(wrapped.__name__, "<unnamed wrapped function>")

  def test_weakref_lru_cache(self):
    @weakref_lru_cache
    def example_cached_fn(key):
      return object()

    class Key:
      def __init__(self):
        # Make a GC loop.
        self.ref_loop = [self]

    stable_keys = [Key() for _ in range(2049)]
    for i in range(10000):
      example_cached_fn(stable_keys[i % len(stable_keys)])
      example_cached_fn(Key())

  def test_weakref_lru_cache_asan_problem(self):

    @weakref_lru_cache
    def reference_loop_generator(x):
      return x

    for _ in range(4097):
      reference_loop_generator(lambda x: x)


class SafeMapTest(jtu.JaxTestCase):

  def test_safe_map(self):
    def unreachable(*args, **kwargs):
      raise RuntimeError("unreachable")

    self.assertEqual([], util.safe_map(unreachable, []))
    self.assertEqual([], util.safe_map(unreachable, (), []))
    self.assertEqual([], util.safe_map(unreachable, [], [], []))
    self.assertEqual([], util.safe_map(unreachable, [], iter([]), [], []))

    def double(x):
      return x * 2

    self.assertEqual([14], util.safe_map(double, (7,)))
    self.assertEqual([0, 2, 4, 6], util.safe_map(double, range(4)))

    def make_tuple(*args):
      return args

    self.assertEqual(
        [(0, 4), (1, 5), (2, 6), (3, 7)],
        util.safe_map(make_tuple, range(4), range(4, 8)),
    )

  def test_safe_map_errors(self):
    with self.assertRaisesRegex(
        TypeError, "safe_map requires at least 2 arguments"
    ):
      util.safe_map()

    with self.assertRaisesRegex(
        TypeError, "safe_map requires at least 2 arguments"
    ):
      util.safe_map(lambda x: x)

    with self.assertRaisesRegex(TypeError, "'int' object is not callable"):
      util.safe_map(7, range(6))

    def error(*args, **kwargs):
      raise RuntimeError("hello")

    with self.assertRaisesRegex(RuntimeError, "hello"):
      util.safe_map(error, range(6))

    with self.assertRaisesRegex(
        ValueError, r"safe_map\(\) argument 2 is longer than argument 1"
    ):
      util.safe_map(operator.add, range(3), range(4))

    with self.assertRaisesRegex(
        ValueError, r"safe_map\(\) argument 2 is shorter than argument 1"
    ):
      util.safe_map(operator.add, range(7), range(2))

    with self.assertRaisesRegex(
        ValueError, r"safe_map\(\) argument 2 is longer than argument 1"
    ):
      util.safe_map(operator.add, (), range(3))


class SafeZipTest(jtu.JaxTestCase):

  def test_safe_zip(self):
    self.assertEqual([], util.safe_zip([]))
    self.assertEqual([], util.safe_zip((), []))
    self.assertEqual([], util.safe_zip([], [], []))
    self.assertEqual([], util.safe_zip([], iter([]), [], []))
    self.assertEqual([(7,)], util.safe_zip((7,)))
    self.assertEqual([(0,), (1,), (2,), (3,)], util.safe_zip(range(4)))
    self.assertEqual(
        [(0, 4), (1, 5), (2, 6), (3, 7)],
        util.safe_zip(range(4), range(4, 8)),
    )

  def test_safe_zip_errors(self):
    with self.assertRaisesWithLiteralMatch(
        TypeError, "safe_zip requires at least 1 argument."
    ):
      util.safe_zip()

    with self.assertRaisesWithLiteralMatch(
        TypeError, "'function' object is not iterable"
    ):
      util.safe_zip(lambda x: x)

    with self.assertRaisesWithLiteralMatch(
        ValueError, "zip() argument 2 is longer than argument 1"
    ):
      util.safe_zip(range(3), range(4))

    with self.assertRaisesWithLiteralMatch(
        ValueError, "zip() argument 2 is shorter than argument 1"
    ):
      util.safe_zip(range(7), range(2))

    with self.assertRaisesWithLiteralMatch(
        ValueError, "zip() argument 2 is longer than argument 1"
    ):
      util.safe_zip((), range(3))


class Node:
  def __init__(self, parents):
    self.parents = parents


class TopologicalSortTest(jtu.JaxTestCase):

  def _check_topological_sort(self, nodes, order):
    self.assertEqual(sorted(nodes, key=id), sorted(order, key=id))
    visited = set()
    for node in nodes:
      self.assertTrue(all(id(parent) in visited for parent in node.parents))
      visited.add(id(node))

  def test_basic(self):
    a = Node([])
    b = Node([a])
    c = Node([a])
    d = Node([a, c])
    e = Node([b, c])
    out = util.toposort([a, d, e])
    self._check_topological_sort([a, b, c, d, e], out)

  def test_stick(self):
    a = Node([])
    b = Node([a])
    c = Node([b])
    d = Node([c])
    e = Node([d])
    out = util.toposort([e])
    self._check_topological_sort([a, b, c, d, e], out)

  def test_diamonds(self):
    a = Node([])
    b = Node([a])
    c = Node([a])
    d = Node([b, c])
    e = Node([d])
    f = Node([d])
    g = Node([e, f])
    out = util.toposort([g])
    self._check_topological_sort([a, b, c, d, e, f, g], out)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
