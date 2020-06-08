# Copyright 2020 Google LLC
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

from absl.testing import absltest

from jax import linear_util as lu
from jax import test_util as jtu
from jax import util

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS

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

    @lu.transformation_with_aux
    def kw_to_positional(factor, *args, **kwargs):
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
      results = yield args + new_args, new_kwargs  # Yield transformed (args, kwargs)
      # Assume results correspond 1:1 to the args + new_args
      assert len(results) == len(args) + len(new_args)
      aux_output = len(new_args)
      yield (results[0:len(args)],
             dict(zip(kwargs_keys, results[len(args):]))), aux_output


    wf = lu.wrap_init(f)  # Wraps `f` as a `WrappedFun`.
    wf, out_thunk = kw_to_positional(wf, 2)
    # Call the transformed function.
    scaled_positional, scaled_kwargs = wf.call_wrapped(1, 2, three=3, four=4)
    self.assertEqual((2, 4), scaled_positional)
    self.assertEqual(dict(three=6, four=8), scaled_kwargs)
    self.assertEqual(2, out_thunk())

  def test_rewrite_future_annotations(self):
    look_for = "from __future__ import annotations"
    text = """
# Some comments
import jax
"""
    text1 = util.rewrite_future_annotations(text, add=True)
    self.assertIn(look_for, text1)
    # Inserting again does not make a difference
    text11 = util.rewrite_future_annotations(text1, add=True)
    self.assertEqual(text1, text11)
    # Removing brings us back
    text2 = util.rewrite_future_annotations(text1, add=False)
    self.assertEqual(text, text2)
    # Removing again makes no difference
    text22 = util.rewrite_future_annotations(text1, add=False)
    self.assertEqual(text2, text22)

    text_no_imports = """
Some text with no imports, and no newline"""
    text1 = util.rewrite_future_annotations(text_no_imports, add=True)
    self.assertIn(look_for, text1)
    # Removing brings us back
    text2 = util.rewrite_future_annotations(text1, add=False)
    self.assertEqual(text_no_imports, text2)


if __name__ == "__main__":
    absltest.main()
