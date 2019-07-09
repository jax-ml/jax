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

from absl.testing import absltest
from absl.testing import parameterized

from jax import tree_util
from jax import test_util as jtu


def _dummy_func(*args, **kwargs):
  return


class TreeTest(jtu.JaxTestCase):

  @parameterized.parameters(
      ((1, 2), ),
      ([3], ),
      ({'a': 1, 'b': 2}, )
  )
  def testRoundtrip(self, inputs):
    xs, tree = tree_util.tree_flatten(inputs)
    actual = tree_util.tree_unflatten(tree, xs)
    self.assertEqual(actual, inputs)

  @parameterized.parameters(
      (tree_util.Partial(_dummy_func), ),
      (tree_util.Partial(_dummy_func, 1, 2), ),
      (tree_util.Partial(_dummy_func, x='a'), ),
      (tree_util.Partial(_dummy_func, 1, 2, 3, x=4, y=5), ),
  )
  def testRoundtrip(self, inputs):
    xs, tree = tree_util.tree_flatten(inputs)
    actual = tree_util.tree_unflatten(tree, xs)
    # functools.partial does not support equality comparisons:
    # https://stackoverflow.com/a/32786109/809705
    self.assertEqual(actual.func, inputs.func)
    self.assertEqual(actual.args, inputs.args)
    self.assertEqual(actual.keywords, inputs.keywords)


if __name__ == "__main__":
  absltest.main()
