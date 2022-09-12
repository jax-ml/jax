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

"""Tests for stringarray."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import tree_util
import jax.numpy as jnp
import numpy as np

import jax.experimental.pandax as jpd


class StringarrayTest(parameterized.TestCase):

  def test_representation(self):
    arr = jpd.StringArray(['b', 'a', 'c', 'b', 'a'])
    self.assertEqual(arr._labels, ('a', 'b', 'c'))
    self.assertIsInstance(arr._data, jnp.ndarray)
    np.testing.assert_array_equal(arr._data, jnp.array([1, 0, 2, 1, 0]))

  def test_str(self):
    arr = jpd.StringArray(['b', 'a', 'c', 'b', 'a'])
    with self.assertRaisesRegex(
        ValueError, 'Can only convert scalar StringArray to string.'):
      str(arr)
    val = str(arr[2])
    self.assertIsInstance(val, str)
    self.assertEqual(val, 'c')

  def test_instantiation(self):
    str_list = ['a', 'b', 'c', 'b', 'a']
    str_array = np.array(str_list, dtype=object)

    with self.subTest('from string'):
      arr = jpd.StringArray('hello world')
      np.testing.assert_array_equal(arr, 'hello world')

    with self.subTest('from list'):
      arr = jpd.StringArray(str_list)
      np.testing.assert_array_equal(arr, str_array)

    with self.subTest('from numpy array'):
      arr = jpd.StringArray(str_array)
      np.testing.assert_array_equal(arr, str_array)

    with self.subTest('from StringArray'):
      arr = jpd.StringArray(str_array)
      arr2 = jpd.StringArray(arr)
      np.testing.assert_array_equal(arr, arr2)

  def test_array_attributes(self):
    str_list = ['a', 'b', 'c', 'b', 'a']
    arr = jpd.StringArray(str_list)

    with self.subTest('shape'):
      self.assertEqual(arr.shape, (5,))

    with self.subTest('size'):
      self.assertEqual(arr.size, 5)

    with self.subTest('dtype'):
      self.assertEqual(arr.dtype, np.dtype(object))

    with self.subTest('ndim'):
      self.assertEqual(arr.ndim, 1)

    with self.subTest('len'):
      self.assertLen(arr, 5)

    with self.subTest('array'):
      self.assertIsInstance(np.asarray(arr), np.ndarray)
      np.testing.assert_array_equal(
          np.asarray(arr), np.array(str_list, dtype=object))

    with self.subTest('transpose'):
      self.assertIsInstance(arr.T, jpd.StringArray)
      np.testing.assert_array_equal(arr.T, arr)

  def test_iter(self):
    arr_2d = jpd.StringArray([['a', 'b', 'c'], ['b', 'c', 'd']])
    rows = tuple(arr_2d)
    self.assertIsInstance(rows[0], jpd.StringArray)
    np.testing.assert_array_equal(rows[0], ['a', 'b', 'c'])

    self.assertIsInstance(rows[1], jpd.StringArray)
    np.testing.assert_array_equal(rows[1], ['b', 'c', 'd'])

  def test_repr(self):
    with self.subTest('0d'):
      arr_0d = jpd.StringArray('hello world')
      self.assertEqual(repr(arr_0d), "StringArray('hello world', dtype=object)")

    with self.subTest('1d'):
      arr_1d = jpd.StringArray(['a', 'b', 'c', 'b', 'a'])
      self.assertEqual(
          repr(arr_1d), "StringArray(['a', 'b', 'c', 'b', 'a'], dtype=object)")

  def test_getitem(self):
    arr = jpd.StringArray(['a', 'b', 'c', 'b', 'a'])

    with self.subTest('integer'):
      np.testing.assert_array_equal(arr[0], 'a')
      self.assertIsInstance(arr[-1], jpd.StringArray)

    with self.subTest('slice'):
      np.testing.assert_array_equal(arr[1:3], jpd.StringArray(['b', 'c']))
      self.assertIsInstance(arr[1:3], jpd.StringArray)

    with self.subTest('mask'):
      mask = jnp.arange(5) % 2 == 0
      np.testing.assert_array_equal(arr[mask], jpd.StringArray(['a', 'c', 'a']))
      self.assertIsInstance(arr[mask], jpd.StringArray)

  def test_flattening(self):
    arr = jpd.StringArray(['a', 'b', 'c', 'b', 'a'])
    arr_flat, tree = tree_util.tree_flatten(arr)
    arr2 = tree_util.tree_unflatten(tree, arr_flat)
    self.assertIsInstance(arr2, jpd.StringArray)
    np.testing.assert_array_equal(arr, arr2)

  def test_reshape(self):
    arr = jpd.StringArray(['a', 'b', 'c', 'c', 'b', 'a'])
    actual = arr.reshape(2, 3)
    expected = jpd.StringArray([['a', 'b', 'c'], ['c', 'b', 'a']])
    np.testing.assert_array_equal(actual, expected)

  def test_ravel(self):
    arr = jpd.StringArray([['a', 'b', 'c'], ['c', 'b', 'a']])
    actual = arr.ravel()
    expected = jpd.StringArray(['a', 'b', 'c', 'c', 'b', 'a'])
    np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
  absltest.main()
