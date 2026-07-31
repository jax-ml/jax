# Copyright 2023 The JAX Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import collections
import dataclasses
import gc
import pickle
import warnings

from absl.testing import absltest, parameterized
from jax.jaxlib import _jax

pytree = _jax.pytree


ExampleType = collections.namedtuple("ExampleType", "field0 field1")

registry = pytree.PyTreeRegistry()


class ExampleType2:

  def __init__(self, field0, field1):
    self.field0 = field0
    self.field1 = field1

  def to_iterable(self):
    return [self.field0, self.field1], (None,)


def from_iterable(state, values):
  del state
  return ExampleType2(field0=values[0], field1=values[1])


registry.register_node(ExampleType2, ExampleType2.to_iterable, from_iterable)


@dataclasses.dataclass
class Custom:
  a: int
  b: str


registry.register_dataclass_node(Custom, ["a"], ["b"])


class PyTreeTest(parameterized.TestCase):

  def roundtrip_proto(self, example):
    original = registry.flatten(example)[1]
    self.assertEqual(
        pytree.PyTreeDef.deserialize_using_proto(
            registry, original.serialize_using_proto()
        ),
        original,
    )

  def testSerializeDeserializeNoPickle(self):
    o = object()
    self.roundtrip_proto(({"a": o, "b": o}, [o, (o, o), None]))

  def testSerializeWithFallback(self):
    o = object()
    with self.assertRaises(ValueError):
      self.roundtrip_proto({"a": ExampleType(field0=o, field1=o)})

  def testRegisteredType(self):
    o = object()
    with self.assertRaises(ValueError):
      self.roundtrip_proto({"a": ExampleType2(field0=o, field1=o)})

  def restore_from_traversal(self, traversal):
    # Rebuilds a PyTreeDef straight from pickle state, as pickle.loads does.
    # Entries: (kind, arity, node_data, custom_type, num_leaves, num_nodes).
    treedef = pytree.PyTreeDef.__new__(pytree.PyTreeDef)
    treedef.__setstate__((registry, traversal))
    return treedef

  @parameterized.named_parameters(
      # num_leaves = INT_MIN made flatten_up_to write through a null list
      # (issue #37412); the other cases are caught by existing guards.
      ("num_leaves_int_min", -(2**31), 4),
      ("num_leaves_negative", -1, 4),
      ("num_nodes_int_min", 3, -(2**31)),
      ("num_nodes_negative", 3, -1),
  )
  def testUnpicklingRejectsNegativeCachedCounts(self, num_leaves, num_nodes):
    leaf = (0, 0, None, None, 1, 1)
    # Matches the other malformed-pickle rejections in PyTreeDef::FromPickle,
    # which raise XlaRuntimeError (a RuntimeError subclass in Python).
    with self.assertRaisesRegex(RuntimeError, "negative num_leaves"):
      self.restore_from_traversal(
          [leaf, leaf, leaf, (4, 3, None, None, num_leaves, num_nodes)]
      )

  @parameterized.named_parameters(
      ("no_keys", []),
      ("too_few_keys", ["a"]),
      ("too_many_keys", ["a", "b", "c", "d"]),
  )
  def testUnpicklingRejectsDictKeyArityMismatch(self, keys):
    # MakeNode indexes sorted_dict_keys by [0, arity), so a key list shorter
    # than the arity used to read out of bounds and crash unflatten.
    leaf = (0, 0, None, None, 1, 1)
    with self.assertRaisesRegex(RuntimeError, "dict with .* keys has arity"):
      self.restore_from_traversal(
          [leaf, leaf, leaf, (5, 3, keys, None, 3, 4)]
      )

  def testUnpicklingAcceptsMatchingDictKeys(self):
    o = object()
    original = registry.flatten({"a": o, "b": o})[1]
    restored = self.restore_from_traversal(original.__getstate__()[1])
    self.assertEqual(restored, original)
    self.assertEqual(restored.unflatten([1, 2]), {"a": 1, "b": 2})

  def testFlattenUpToRejectsNegativeNumLeaves(self):
    # treedef_tuple and from_node_data_and_children sum children's counts
    # without overflow checks, so a negative total can still reach here.
    leaf = (0, 0, None, None, 1, 1)
    child = [leaf, leaf, leaf, (4, 3, None, None, 2**30, 4)]
    children = [self.restore_from_traversal(child) for _ in range(2)]

    parent = pytree.treedef_tuple(registry, children)
    self.assertLess(parent.num_leaves, 0)
    with self.assertRaisesRegex(ValueError, "num_leaves is negative"):
      parent.flatten_up_to(([1, 2, 3], [4, 5, 6]))

    parent = pytree.PyTreeDef.from_node_data_and_children(
        registry, (list, None), children
    )
    self.assertLess(parent.num_leaves, 0)
    with self.assertRaisesRegex(ValueError, "num_leaves is negative"):
      parent.flatten_up_to([[1, 2, 3], [4, 5, 6]])

  def testUnpicklingAcceptsValidCachedCounts(self):
    # The counts above are rejected for being negative, not for disagreeing
    # with the traversal, so well-formed state must still restore exactly.
    o = object()
    original = registry.flatten([o, {"a": o, "b": o}, (o, None)])[1]
    restored = self.restore_from_traversal(original.__getstate__()[1])
    self.assertEqual(restored, original)
    self.assertEqual(restored.num_leaves, 4)
    self.assertEqual(
        list(restored.flatten_up_to([1, {"a": 2, "b": 3}, (4, None)])),
        [1, 2, 3, 4],
    )

    # Also exercise a real pickle round trip, which needs a registry that is
    # itself picklable; the module-level one above is not.
    original = pytree.default_registry().flatten([o, {"a": o}, (o, None)])[1]
    restored = pickle.loads(pickle.dumps(original))
    self.assertEqual(restored, original)
    self.assertEqual(restored.num_leaves, 3)
    self.assertEqual(
        list(restored.flatten_up_to([1, {"a": 2}, (3, None)])), [1, 2, 3]
    )

  def roundtrip_node_data(self, example):
    original = registry.flatten(example)[1]
    restored = pytree.PyTreeDef.from_node_data_and_children(
        registry, original.node_data(), original.children()
    )
    self.assertEqual(restored, original)

  def testRoundtripNodeData(self):
    o = object()
    self.roundtrip_node_data([o, o, o])
    self.roundtrip_node_data((o, o, o))
    self.roundtrip_node_data({"a": o, "b": o})
    self.roundtrip_node_data({22: o, 88: o})
    self.roundtrip_node_data(None)
    self.roundtrip_node_data(o)
    self.roundtrip_node_data(ExampleType(field0=o, field1=o))
    self.roundtrip_node_data(ExampleType2(field0=o, field1=o))

  def testCompose(self):
    x = registry.flatten(0)[1]
    y = registry.flatten((0, 0))[1]
    self.assertEqual((x.compose(y)).num_leaves, 2)

  def testDataclassMakeFromNodeData(self):
    c = Custom(1, "a")
    c_leafs, c_tree = registry.flatten(c)
    c_tree2 = pytree.PyTreeDef.from_node_data_and_children(
        registry, c_tree.node_data(), c_tree.children()
    )
    self.assertEqual(c_tree2.unflatten(c_leafs), c)
    self.assertEqual(str(c_tree2), str(c_tree))

  def testTpTraverse(self):
    self.assertContainsSubset(
        [
            pytree.PyTreeRegistry,
            ExampleType2,
            ExampleType2.to_iterable,
            from_iterable,
        ],
        gc.get_referents(registry),
    )
    k1 = "k1"
    k2 = "k2"

    t = ExampleType("a", "b")
    _, treedef = registry.flatten([1, {k1: 2, k2: t}, 5, t])

    self.assertContainsSubset(
        [
            pytree.PyTreeDef,
            registry,
            k1,
            k2,
            ExampleType,
        ],
        gc.get_referents(treedef),
    )

  # TODO(rdyro): Remove this test when iterators throw an error.
  @parameterized.named_parameters(
      ("zip", zip([], [])),
      ("list_iter", iter([])),
      ("set_iter", iter(set())),
      ("generator", (x for x in [])),
      ("map", map(lambda x: x, [])),
      ("filter", filter(lambda x: True, [])),
      ("enumerate", enumerate([])),
      ("reversed", reversed(())),
      ("tuple_iter", iter(())),
      ("dict_key_iter", iter({}.keys())),
      ("dict_value_iter", iter({}.values())),
      ("dict_item_iter", iter({}.items())),
      ("dict_values", {}.values()),
  )
  def testIterableWarning(self, iterator):
    with self.assertWarnsRegex(
        DeprecationWarning,
        "Python iterable type '.*' is treated as a leaf in PyTree.",
    ):
      registry.flatten(iterator)

  def testDictValuesWithLeafPredicate(self):
    d = {"a": 1, "b": 2}
    v = d.values()
    dict_values_type = type(v)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      leaves, _ = registry.flatten(v, lambda x: isinstance(x, dict_values_type))
      self.assertEmpty(w)
    self.assertEqual(leaves, [v])

if __name__ == "__main__":
  absltest.main()
