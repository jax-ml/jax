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

@tree_util.register_pytree_node_class
class Special:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __repr__(self):
    return "Special(x={}, y={})".format(self.x, self.y)

  def tree_flatten(self):
    return ((self.x, self.y), None)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children)

  def __eq__(self, other):
    return type(self) is type(other) and (self.x, self.y) == (other.x, other.y)

@tree_util.register_pytree_node_class
class FlatCache:
  def __init__(self, structured, *, leaves=None, treedef=None):
    if treedef is None:
      leaves, treedef = tree_util.tree_flatten(structured)
    self._structured = structured
    self.treedef = treedef
    self.leaves = leaves

  def __hash__(self):
    return hash(self.structured)

  def __eq__(self, other):
    return self.structured == other.structured

  def __repr__(self):
    return f"FlatCache({self.structured!r})"

  @property
  def structured(self):
    if self._structured is None:
      self._structured = tree_util.tree_unflatten(self.treedef, self.leaves)
    return self._structured

  def tree_flatten(self):
    return self.leaves, self.treedef

  @classmethod
  def tree_unflatten(cls, meta, data):
    if not tree_util.all_leaves(data):
      data, meta = tree_util.tree_flatten(tree_util.tree_unflatten(meta, data))
    return FlatCache(None, leaves=data, treedef=meta)

TREES = (
    (None,),
    ((),),
    (([()]),),
    ((1, 2),),
    (((1, "foo"), ["bar", (3, None, 7)]),),
    ([3],),
    ([3, ATuple(foo=(3, ATuple(foo=3, bar=None)), bar={"baz": 34})],),
    ([AnObject(3, None, [4, "foo"])],),
    (Special(2, 3.),),
    ({"a": 1, "b": 2},),
    (collections.OrderedDict([("foo", 34), ("baz", 101), ("something", -42)]),),
    (collections.defaultdict(dict,
                             [("foo", 34), ("baz", 101), ("something", -42)]),),
    (ANamedTupleSubclass(foo="hello", bar=3.5),),
    (FlatCache(None),),
    (FlatCache(1),),
    (FlatCache({"a": [1, 2]}),),
)

LEAVES = (
    ("foo",),
    (0.1,),
    (1,),
    (object(),),
)


class TreeTest(jtu.JaxTestCase):

  @parameterized.parameters(*(TREES + LEAVES))
  def testRoundtrip(self, inputs):
    xs, tree = tree_util.tree_flatten(inputs)
    actual = tree_util.tree_unflatten(tree, xs)
    self.assertEqual(actual, inputs)

  @parameterized.parameters(*(TREES + LEAVES))
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

  @parameterized.parameters(*(TREES + LEAVES))
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

  @parameterized.parameters(*TREES)
  def testAllLeavesWithTrees(self, tree):
    leaves = tree_util.tree_leaves(tree)
    self.assertTrue(tree_util.all_leaves(leaves))
    self.assertFalse(tree_util.all_leaves([tree]))

  @parameterized.parameters(*LEAVES)
  def testAllLeavesWithLeaves(self, leaf):
    self.assertTrue(tree_util.all_leaves([leaf]))

  @parameterized.parameters(*TREES)
  def testCompose(self, tree):
    treedef = tree_util.tree_structure(tree)
    inner_treedef = tree_util.tree_structure(["*", "*", "*"])
    composed_treedef = treedef.compose(inner_treedef)
    expected_leaves = treedef.num_leaves * inner_treedef.num_leaves
    self.assertEqual(composed_treedef.num_leaves, expected_leaves)
    expected_nodes = ((treedef.num_nodes - treedef.num_leaves) +
                      (inner_treedef.num_nodes * treedef.num_leaves))
    self.assertEqual(composed_treedef.num_nodes, expected_nodes)
    leaves = [1] * expected_leaves
    composed = tree_util.tree_unflatten(composed_treedef, leaves)
    self.assertEqual(leaves, tree_util.tree_leaves(composed))

  @parameterized.parameters(*TREES)
  def testTranspose(self, tree):
    outer_treedef = tree_util.tree_structure(tree)
    if not outer_treedef.num_leaves:
      self.skipTest("Skipping empty tree")
    inner_treedef = tree_util.tree_structure([1, 1, 1])
    nested = tree_util.tree_map(lambda x: [x, x, x], tree)
    actual = tree_util.tree_transpose(outer_treedef, inner_treedef, nested)
    self.assertEqual(actual, [tree, tree, tree])

  def testTransposeMismatchOuter(self):
    tree = {"a": [1, 2], "b": [3, 4]}
    outer_treedef = tree_util.tree_structure({"a": 1, "b": 2, "c": 3})
    inner_treedef = tree_util.tree_structure([1, 2])
    with self.assertRaisesRegex(TypeError, "Mismatch"):
      tree_util.tree_transpose(outer_treedef, inner_treedef, tree)

  def testTransposeMismatchInner(self):
    tree = {"a": [1, 2], "b": [3, 4]}
    outer_treedef = tree_util.tree_structure({"a": 1, "b": 2})
    inner_treedef = tree_util.tree_structure([1, 2, 3])
    with self.assertRaisesRegex(TypeError, "Mismatch"):
      tree_util.tree_transpose(outer_treedef, inner_treedef, tree)

  def testTransposeWithCustomObject(self):
    outer_treedef = tree_util.tree_structure(FlatCache({"a": 1, "b": 2}))
    inner_treedef = tree_util.tree_structure([1, 2])
    expected = [FlatCache({"a": 3, "b": 5}), FlatCache({"a": 4, "b": 6})]
    actual = tree_util.tree_transpose(outer_treedef, inner_treedef,
                                      FlatCache({"a": [3, 4], "b": [5, 6]}))
    self.assertEqual(expected, actual)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
