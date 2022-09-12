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

import textwrap

from absl.testing import absltest
from absl.testing import parameterized
from jax import tree_util
import numpy as np
import pandas as pd

import jax.experimental.pandax as jpd


class SeriesTest(parameterized.TestCase):

  def assertSeriesEqual(self, ser1, ser2, check_dtypes=True):
    np.testing.assert_array_equal(ser1, ser2)
    np.testing.assert_array_equal(ser1.index, ser2.index)
    if check_dtypes:
      self.assertEqual(ser1.dtype, ser2.dtype)
      self.assertEqual(ser1.index.dtype, ser2.index.dtype)

  def test_instantiation(self):
    with self.subTest("from array"):
      ser = jpd.Series(np.ones(4))
      self.assertIsInstance(ser, jpd.Series)
      self.assertIsInstance(ser.index, jpd.RangeIndex)
      self.assertEqual(ser.index._data, range(4))
      np.testing.assert_array_equal(ser, np.ones(4))

    with self.subTest("from strings"):
      ser = jpd.Series(["one", "two", "three"])
      self.assertIsInstance(ser, jpd.Series)
      self.assertIsInstance(ser.index, jpd.RangeIndex)
      self.assertEqual(ser.index._data, range(3))
      np.testing.assert_array_equal(ser, ["one", "two", "three"])

    with self.subTest("from array with index"):
      ser = jpd.Series([1, 2, 3], index=["A", "B", "C"])
      self.assertIsInstance(ser, jpd.Series)
      self.assertIsInstance(ser.index, jpd.Index)
      np.testing.assert_array_equal(ser.index, ["A", "B", "C"])
      np.testing.assert_array_equal(ser, [1, 2, 3])

    with self.subTest("from series with index"):
      data = jpd.Series([1, 2, 3])
      ser = jpd.Series(data, index=["A", "B", "C"])
      self.assertIsInstance(ser, jpd.Series)
      self.assertIsInstance(ser.index, jpd.Index)
      np.testing.assert_array_equal(ser.index, ["A", "B", "C"])
      np.testing.assert_array_equal(ser, [1, 2, 3])

  def test_attributes(self):
    ser = jpd.Series(
        np.arange(5, dtype="float32"), index=["A", "B", "C", "D", "E"])
    self.assertEqual(ser.dtype, np.dtype("float32"))
    self.assertEqual(ser.shape, (5,))
    self.assertLen(ser, 5)
    self.assertEqual(ser.size, 5)
    self.assertEqual(ser.ndim, 1)
    self.assertIsInstance(ser.index, jpd.Index)

  def test_repr(self):
    ser = jpd.Series([1, 2, 3], index=["A", "B", "C"])
    self.assertEqual(
        repr(ser),
        textwrap.dedent("""\
    A    1
    B    2
    C    3
    dtype: int32"""))

  def test_to_pandas(self):
    data = np.array([1, 2, 3], dtype="int32")
    index = ["A", "B", "C"]
    ser = jpd.Series(data, index=index)
    actual = ser.to_pandas()
    expected = pd.Series(data, index=index)
    self.assertSeriesEqual(actual, expected)

  def test_flattening(self):
    ser = jpd.Series(["A", "B", "C"])
    ser_flat, tree = tree_util.tree_flatten(ser)
    ser2 = tree_util.tree_unflatten(tree, ser_flat)
    self.assertIsInstance(ser2, jpd.Series)
    self.assertSeriesEqual(ser2, ser)

  def test_getitem(self):
    with self.subTest("slice"):
      ser = jpd.Series([1, 2, 3], index=["A", "B", "C"])
      actual = ser[1:]
      expected = jpd.Series([2, 3], index=["B", "C"])
      self.assertSeriesEqual(actual, expected)

    with self.subTest("string index"):
      ser = jpd.Series([1, 2, 3], index=["A", "B", "C"])
      self.assertEqual(ser["A"], 1)
      self.assertEqual(ser["B"], 2)

    with self.subTest("integer index"):
      ser = jpd.Series([1, 2, 3], index=[2, 4, 6])
      self.assertEqual(ser[2], 1)
      self.assertEqual(ser[4], 2)

    with self.subTest("range index"):
      ser = jpd.Series([1, 2, 3])
      self.assertEqual(ser[0], 1)
      self.assertEqual(ser[1], 2)

  def test_groupby(self):
    key = jpd.StringArray(["A", "B", "C", "A", "B", "C"])
    ser = jpd.Series(np.arange(1, 7, dtype="float32"))

    with self.subTest("count"):
      result = ser.groupby(key).count()
      expected = jpd.Series([2, 2, 2], index=["A", "B", "C"], dtype="int32")
      self.assertSeriesEqual(result, expected)

    with self.subTest("max"):
      result = ser.groupby(key).max()
      expected = jpd.Series([4, 5, 6], index=["A", "B", "C"], dtype="float32")
      self.assertSeriesEqual(result, expected)

    with self.subTest("min"):
      result = ser.groupby(key).min()
      expected = jpd.Series([1, 2, 3], index=["A", "B", "C"], dtype="float32")
      self.assertSeriesEqual(result, expected)

    with self.subTest("sum"):
      result = ser.groupby(key).sum()
      expected = jpd.Series([5, 7, 9], index=["A", "B", "C"], dtype="float32")
      self.assertSeriesEqual(result, expected)

    with self.subTest("prod"):
      result = ser.groupby(key).prod()
      expected = jpd.Series([4, 10, 18], index=["A", "B", "C"], dtype="float32")
      self.assertSeriesEqual(result, expected)


if __name__ == "__main__":
  absltest.main()
