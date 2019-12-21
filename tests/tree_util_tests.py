# Copyright 2019 Google LLC
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

import collections

from absl.testing import absltest
from absl.testing import parameterized

from jax import test_util as jtu
from jax import tree_util


def _dummy_func(*args, **kwargs):
  return


ATuple = collections.namedtuple("ATuple", ("foo", "bar"))

class ANamedTupleSubclass(ATuple):
  pass

class AnObject(object):

  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z

  def __eq__(self, other):
    return self.x == other.x and self.y == other.y and self.z == other.z

  def __hash__(self):
    return hash((self.x, self.y, self.z))

  def __repr__(self):
    return "AnObject({},{},{})".format(self.x, self.y, self.z)


tree_util.register_pytree_node(AnObject, lambda o: ((o.x, o.y), o.z),
                               lambda z, xy: AnObject(xy[0], xy[1], z))

PYTREES = [
    ("foo",),
    ((),),
    (([()]),),
    ((1, 2),),
    (((1, "foo"), ["bar", (3, None, 7)]),),
    ([3],),
    ([3, ATuple(foo=(3, ATuple(foo=3, bar=None)), bar={"baz": 34})],),
    ([AnObject(3, None, [4, "foo"])],),
    ({"a": 1, "b": 2},),
    (collections.OrderedDict([("foo", 34), ("baz", 101), ("something", -42)]),),
    (collections.defaultdict(dict,
                             [("foo", 34), ("baz", 101), ("something", -42)]),),
    (ANamedTupleSubclass(foo="hello", bar=3.5),),
]


class TreeTest(jtu.JaxTestCase):

  @parameterized.parameters(*PYTREES)
  def testRoundtrip(self, inputs):
    xs, tree = tree_util.tree_flatten(inputs)
    actual = tree_util.tree_unflatten(tree, xs)
    self.assertEqual(actual, inputs)

  @parameterized.parameters(*PYTREES)
  def testRoundtripWithFlattenUpTo(self, inputs):
    _, tree = tree_util.tree_flatten(inputs)
    if not hasattr(tree, "flatten_up_to"):
      self.skipTest("Test requires Jaxlib >= 0.1.23")
    xs = tree.flatten_up_to(inputs)
    actual = tree_util.tree_unflatten(tree, xs)
    self.assertEqual(actual, inputs)

  @parameterized.parameters(
      (tree_util.Partial(_dummy_func),),
      (tree_util.Partial(_dummy_func, 1, 2),),
      (tree_util.Partial(_dummy_func, x="a"),),
      (tree_util.Partial(_dummy_func, 1, 2, 3, x=4, y=5),),
  )
  def testRoundtripPartial(self, inputs):
    xs, tree = tree_util.tree_flatten(inputs)
    actual = tree_util.tree_unflatten(tree, xs)
    # functools.partial does not support equality comparisons:
    # https://stackoverflow.com/a/32786109/809705
    self.assertEqual(actual.func, inputs.func)
    self.assertEqual(actual.args, inputs.args)
    self.assertEqual(actual.keywords, inputs.keywords)

  @parameterized.parameters(*PYTREES)
  def testRoundtripViaBuild(self, inputs):
    xs, tree = tree_util._process_pytree(tuple, inputs)
    actual = tree_util.build_tree(tree, xs)
    self.assertEqual(actual, inputs)

  def testChildren(self):
    _, tree = tree_util.tree_flatten(((1, 2, 3), (4,)))
    _, c0 = tree_util.tree_flatten((0, 0, 0))
    _, c1 = tree_util.tree_flatten((7,))
    if not callable(tree.children):
      self.skipTest("Test requires Jaxlib >= 0.1.23")
    self.assertEqual([c0, c1], tree.children())

  def testFlattenUpTo(self):
    _, tree = tree_util.tree_flatten([(1, 2), None, ATuple(foo=3, bar=7)])
    if not hasattr(tree, "flatten_up_to"):
      self.skipTest("Test requires Jaxlib >= 0.1.23")
    out = tree.flatten_up_to([({
        "foo": 7
    }, (3, 4)), None, ATuple(foo=(11, 9), bar=None)])
    self.assertEqual(out, [{"foo": 7}, (3, 4), (11, 9), None])

  def testTreeMultimap(self):
    x = ((1, 2), (3, 4, 5))
    y = (([3], None), ({"foo": "bar"}, 7, [5, 6]))
    out = tree_util.tree_multimap(lambda *xs: tuple(xs), x, y)
    self.assertEqual(out, (((1, [3]), (2, None)),
                           ((3, {"foo": "bar"}), (4, 7), (5, [5, 6]))))


if __name__ == "__main__":
  absltest.main()
