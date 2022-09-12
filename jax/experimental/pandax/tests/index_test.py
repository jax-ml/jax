# Copyright 2022 Google LLC
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

"""Tests for pandax.index."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import tree_util
import numpy as np

import jax.experimental.pandax as jpd


class IndexTest(parameterized.TestCase):

  def test_instantiation(self):
    with self.subTest('from numpy array'):
      arr = np.arange(5, 10)
      ind = jpd.Index(arr)
      self.assertIsInstance(ind, jpd.Index)
      np.testing.assert_array_equal(ind, arr)

    with self.subTest('from JAX array'):
      arr = np.arange(5, 10)
      ind = jpd.Index(arr)
      self.assertIsInstance(ind, jpd.Index)
      np.testing.assert_array_equal(ind, arr)

    with self.subTest('from list of strings'):
      arr = ['a', 'b', 'c']
      ind = jpd.Index(arr)
      self.assertIsInstance(ind, jpd.Index)
      np.testing.assert_array_equal(ind, arr)

    with self.subTest('from range'):
      arr = range(10)
      ind = jpd.Index(arr)
      self.assertIsInstance(ind, jpd.RangeIndex)
      np.testing.assert_array_equal(ind, arr)

    with self.subTest('from Index'):
      arr = jpd.Index(np.arange(10))
      ind = jpd.Index(arr)
      self.assertIsInstance(ind, jpd.Index)
      np.testing.assert_array_equal(ind, arr)

    with self.subTest('from RangeIndex'):
      arr = jpd.RangeIndex(10)
      ind = jpd.Index(arr)
      self.assertIsInstance(ind, jpd.RangeIndex)
      np.testing.assert_array_equal(ind, arr)

  def test_repr(self):
    with self.subTest('Index'):
      ind = jpd.Index(['A', 'B', 'C'])
      self.assertEqual(repr(ind), "Index(['A', 'B', 'C'], dtype=object)")

    with self.subTest('RangeIndex'):
      ind = jpd.Index(range(5))
      self.assertEqual(repr(ind), 'RangeIndex(start=0, stop=5, step=1)')

  def test_flattening(self):
    with self.subTest('numerical Index'):
      arr = jpd.Index(np.arange(5))
      arr_flat, tree = tree_util.tree_flatten(arr)
      arr2 = tree_util.tree_unflatten(tree, arr_flat)
      self.assertIsInstance(arr2, jpd.Index)
      np.testing.assert_array_equal(arr2, arr)

    with self.subTest('string Index'):
      arr = jpd.Index(['A', 'B', 'C'])
      arr_flat, tree = tree_util.tree_flatten(arr)
      arr2 = tree_util.tree_unflatten(tree, arr_flat)
      self.assertIsInstance(arr2, jpd.Index)
      np.testing.assert_array_equal(arr2, arr)

    with self.subTest('RangeIndex'):
      arr = jpd.Index(range(5))
      arr_flat, tree = tree_util.tree_flatten(arr)
      arr2 = tree_util.tree_unflatten(tree, arr_flat)
      self.assertIsInstance(arr2, jpd.RangeIndex)
      np.testing.assert_array_equal(arr2, arr)

  def test_get_indexer(self):
    with self.subTest('numerical Index'):
      ind = jpd.Index([1, 3, 2, 5])
      vals = [1, 2, 3, 4, 5]
      expected = [0, 2, 1, -1, 3]
      np.testing.assert_array_equal(ind.get_indexer(vals), expected)

    with self.subTest('string Index'):
      ind = jpd.Index(['c', 'a', 'b', 'e'])
      vals = ['a', 'b', 'c', 'd', 'e']
      expected = [1, 2, 0, -1, 3]
      np.testing.assert_array_equal(ind.get_indexer(vals), expected)

    with self.subTest('RangeIndex'):
      ind = jpd.RangeIndex(range(1, 10, 2))
      vals = [0, 1, 2, 3, 4, 5]
      expected = [-1, 0, -1, 1, -1, 2]
      np.testing.assert_array_equal(ind.get_indexer(vals), expected)


if __name__ == '__main__':
  absltest.main()
