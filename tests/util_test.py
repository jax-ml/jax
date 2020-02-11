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

import inspect
import jax
import jax.numpy as np
from jax import api, lax
from jax import linear_util as lu
from jax import test_util as jtu
from jax import util
from absl.testing import absltest

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

  def test_partial(self):
    def f(x, y: int, z=2) -> int:
      return x + y + z

    def fx(y: int, z=2) -> int:
      return 1 + y + z

    f_partial = util.partial(f, 1)
    self.assertEqual(fx(2, 3), f_partial(2, 3))
    self.assertEqual(inspect.signature(fx), f_partial.__signature__)

    def fy(x, z=2) -> int:
      return 1 + x + z

    f_partial = util.partial(f, y=1)
    self.assertEqual(fy(x=2, z=3), f_partial(x=2, z=3))
    self.assertEqual(inspect.signature(fy), f_partial.__signature__)

    def fz(x, y:int) -> int:
      return 1 + x + y

    f_partial = util.partial(f, z=1)
    self.assertEqual(fz(2, 3), f_partial(2, 3))
    self.assertEqual(inspect.signature(fz), f_partial.__signature__)


if __name__ == '__main__':
  absltest.main()
