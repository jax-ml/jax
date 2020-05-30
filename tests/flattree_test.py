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


from functools import partial
import itertools as it
from unittest import SkipTest

import numpy as np
from absl.testing import absltest, parameterized
from jax.interpreters import flattree
from jax.interpreters.flattree import (
    TRIVIAL_TREEDEF, convert_vectorized_tree, convert_leaf_array,
    restore_tree,
)
from jax import undo_tree
from jax.tree_util import tree_flatten, tree_structure
from jax.config import config
import jax.test_util as jtu

config.parse_flags_with_absl()


class FlatTreeTest(jtu.JaxTestCase):

  def assertTreeStateEqual(self, expected, actual, check_dtypes):
    actual_treedefs, actual_leafshapes, actual_leaves = actual
    expected_treedefs, expected_leafshapes, expected_leaves = expected
    self.assertEqual(actual_treedefs, expected_treedefs)
    self.assertEqual(actual_leafshapes, expected_leafshapes)
    self.assertEqual(actual_leaves.keys(), expected_leaves.keys())
    for key in actual_leaves:
      self.assertArraysEqual(actual_leaves[key], expected_leaves[key],
                             check_dtypes=True)

  def assertTreeEqual(self, expected, actual, check_dtypes):
    expected_leaves, expected_treedef = tree_flatten(expected)
    actual_leaves, actual_treedef = tree_flatten(actual)
    self.assertEqual(actual_treedef, expected_treedef)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
      self.assertArraysEqual(actual_leaf, expected_leaf, check_dtypes=True)

  @parameterized.parameters([
      (1.0, ([], [], {(): 1.0})),
      (np.arange(3.0), ([TRIVIAL_TREEDEF], [[(3,)]], {(0,): np.arange(3.0)})),
      (np.array([[1, 2, 3], [4, 5, 6]]),
       ([TRIVIAL_TREEDEF, TRIVIAL_TREEDEF], [[(2,)], [(3,)]],
        {(0, 0): np.array([[1, 2, 3], [4, 5, 6]])})),
  ])
  def test_convert_leaf_array(self, leaf, expected):
    actual = convert_leaf_array(leaf)
    self.assertTreeStateEqual(actual, expected, check_dtypes=True)
    treedefs, _, leaves = actual
    roundtripped = restore_tree(treedefs, leaves)
    self.assertArraysEqual(roundtripped, leaf, check_dtypes=True)

  @parameterized.parameters([
      (np.array(1.0), ([TRIVIAL_TREEDEF], [[()]], {(0,): np.array(1.0)})),
      ({'a': np.array(0.0), 'b': np.array([1.0]), 'c': np.array([2, 3])},
        ([tree_structure({'a': 0, 'b': 0, 'c': 0})],
          [[(), (1,), (2,)]],
          {(0,): np.array(0.0),
           (1,): np.array([1.0]),
           (2,): np.array([2.0, 3.0])})),
  ])
  def test_convert_vectorized_tree(self, tree, expected):
    actual = convert_vectorized_tree(tree)
    self.assertTreeStateEqual(actual, expected, check_dtypes=True)
    treedefs, _, leaves = actual
    roundtripped = restore_tree(treedefs, leaves)
    self.assertTreeEqual(roundtripped, tree, check_dtypes=True)

  @parameterized.parameters([
      ([TRIVIAL_TREEDEF], {(0,): 1.0}, 1.0),
      ([TRIVIAL_TREEDEF, TRIVIAL_TREEDEF], {(0, 0): 2.0}, 2.0),
      ([tree_structure({'a': 0, 'b': 0})], {(0,): 1.0, (1,): 2.0},
        {'a': 1.0, 'b': 2.0}),
      ([tree_structure({'a': 0, 'b': 0}), tree_structure({'c': 0, 'd': 0})],
        {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4},
        {'a': {'c': 1, 'd': 2}, 'b': {'c': 3, 'd': 4}}),
  ])
  def test_restore_tree(self, treedefs, leaves, expected):
    actual = restore_tree(treedefs, leaves)
    self.assertTreeEqual(actual, expected, check_dtypes=True)
