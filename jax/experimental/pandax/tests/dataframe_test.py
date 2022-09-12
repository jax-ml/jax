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


class DataFrameTest(parameterized.TestCase):

  def assertDataFrameEqual(self, df1, df2, check_dtypes=True):
    np.testing.assert_array_equal(df1.index, df2.index)
    np.testing.assert_array_equal(df1.columns, df2.columns)
    for key in df1.columns:
      np.testing.assert_array_equal(df1[key], df2[key])
    if check_dtypes:
      self.assertEqual(df1.index.dtype, df2.index.dtype)
      self.assertEqual(df1.columns.dtype, df2.columns.dtype)

  def test_instantiation(self):
    with self.subTest('dict with default index'):
      df = jpd.DataFrame({'key': ['a', 'b', 'c', 'd'], 'val': np.arange(4)})
      np.testing.assert_array_equal(df.index, range(4))
      np.testing.assert_array_equal(df.columns, ['key', 'val'])
      np.testing.assert_array_equal(df['key'], ['a', 'b', 'c', 'd'])
      np.testing.assert_array_equal(df['val'], np.arange(4))

    with self.subTest('dict with specified index'):
      df = jpd.DataFrame({
          'key': ['a', 'b', 'c', 'd'],
          'val': np.arange(4)
      },
                         index=['A', 'B', 'C', 'D'])
      np.testing.assert_array_equal(df.index, ['A', 'B', 'C', 'D'])
      np.testing.assert_array_equal(df.columns, ['key', 'val'])
      np.testing.assert_array_equal(df['key'], ['a', 'b', 'c', 'd'])
      np.testing.assert_array_equal(df['val'], np.arange(4))

  def test_attributes(self):
    dct = {'key': ['a', 'b', 'c', 'd'], 'val': np.arange(4, dtype='int32')}
    df = jpd.DataFrame(dct, index=['A', 'B', 'C', 'D'])
    self.assertEqual(df.shape, (4, 2))
    self.assertLen(df, 4)
    self.assertEqual(df.size, 8)
    self.assertEqual(df.ndim, 2)
    self.assertIsInstance(df.index, jpd.Index)
    self.assertIsInstance(df.columns, jpd.Index)

  def test_repr(self):
    df = jpd.DataFrame({
        'key': ['aa', 'bb', 'cc', 'dd'],
        'val': np.arange(4)
    },
                       index=['A', 'B', 'C', 'D'])
    self.assertEqual(
        repr(df),
        textwrap.dedent("""\
      key  val
    A  aa    0
    B  bb    1
    C  cc    2
    D  dd    3"""))

  def test_to_pandas(self):
    dct = {'key': ['aa', 'bb', 'cc', 'dd'], 'val': np.arange(4, dtype='int32')}
    index = ['A', 'B', 'C', 'D']
    df = jpd.DataFrame(dct, index=index)
    actual = df.to_pandas()
    expected = pd.DataFrame(dct, index=index)
    self.assertDataFrameEqual(actual, expected)

  def test_flattening(self):
    df = jpd.DataFrame({'key': ['a', 'b', 'c', 'd'], 'val': np.arange(4)})
    df_flat, tree = tree_util.tree_flatten(df)
    df2 = tree_util.tree_unflatten(tree, df_flat)
    self.assertIsInstance(df2, jpd.DataFrame)
    self.assertDataFrameEqual(df2, df)

  def test_drop(self):
    df = jpd.DataFrame({
        'key': ['x', 'y', 'z', 'w'],
        'val': np.arange(4)
    },
                       index=['A', 'B', 'C', 'D'])
    with self.subTest('drop key'):
      result = df.drop('key', axis=1)
      expected = jpd.DataFrame({'val': df['val']}, index=df.index)
      self.assertDataFrameEqual(result, expected)

    with self.subTest('drop val'):
      result = df.drop('val', axis=1)
      expected = jpd.DataFrame({'key': df['key']}, index=df.index)
      self.assertDataFrameEqual(result, expected)

    with self.subTest('drop both'):
      # TODO(jakevdp) implement this:
      # result = df.drop(['key', 'val'], axis=1)
      result = df.drop('key', axis=1).drop('val', axis=1)
      expected = jpd.DataFrame({}, index=df.index)
      self.assertDataFrameEqual(result, expected)

  def test_groupby(self):
    # TODO(jakevdp): test with string columns (for count/min/max)
    df = jpd.DataFrame({
        'value1': np.arange(1, 7, dtype='float32'),
        'value2': np.arange(-3, 3, dtype='int32')
    })
    key = np.array([2, 1, 1, 2, 3, 2])

    with self.subTest('count'):
      result = df.groupby(key).count()
      expected = jpd.DataFrame({
          'value1': [2, 3, 1],
          'value2': [2, 3, 1]
      },
                               index=[1, 2, 3])
      self.assertDataFrameEqual(result, expected)

    with self.subTest('min'):
      result = df.groupby(key).min()
      expected = jpd.DataFrame({
          'value1': [2, 1, 5],
          'value2': [-2, -3, 1]
      },
                               index=[1, 2, 3])
      self.assertDataFrameEqual(result, expected)

    with self.subTest('max'):
      result = df.groupby(key).max()
      expected = jpd.DataFrame({
          'value1': [3, 6, 5],
          'value2': [-1, 2, 1]
      },
                               index=[1, 2, 3])
      self.assertDataFrameEqual(result, expected)

    with self.subTest('sum'):
      result = df.groupby(key).sum()
      expected = jpd.DataFrame({
          'value1': [5, 11, 5],
          'value2': [-3, -1, 1]
      },
                               index=[1, 2, 3])
      self.assertDataFrameEqual(result, expected)

    with self.subTest('prod'):
      result = df.groupby(key).prod()
      expected = jpd.DataFrame({
          'value1': [6, 24, 5],
          'value2': [2, 0, 1]
      },
                               index=[1, 2, 3])
      self.assertDataFrameEqual(result, expected)

    df = jpd.DataFrame({
        'key': ['A', 'B', 'C', 'B', 'A', 'B'],
        'val1': np.arange(6),
        'val2': np.array([0, 3, 2, 5, 4, 6])
    })

    with self.subTest('sum by name'):
      result = df.groupby('key').sum()
      expected = jpd.DataFrame({
          'val1': [4, 9, 2],
          'val2': [4, 14, 2]
      },
                               index=['A', 'B', 'C'])
      self.assertDataFrameEqual(result, expected)


if __name__ == '__main__':
  absltest.main()
