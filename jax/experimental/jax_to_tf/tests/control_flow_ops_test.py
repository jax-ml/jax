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
"""Tests for the jax_to_tf conversion for control-flow primitives."""

from absl.testing import absltest
from absl.testing import parameterized
from typing import Any, Callable, Sequence, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax import test_util as jtu
import numpy as np

from jax.experimental.jax_to_tf.tests import tf_test_util

from jax.config import config
config.parse_flags_with_absl()


class ControlFlowOpsTest(tf_test_util.JaxToTfTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_function={with_function}",
         with_function=with_function)
    for with_function in [False, True]))
  def test_cond(self, with_function=False):
    def f_jax(pred, x):
      return lax.cond(pred, lambda t: t + 1., lambda f: f, x)

    self.ConvertAndCompare(f_jax, True, 1., with_function=with_function)
    self.ConvertAndCompare(f_jax, False, 1., with_function=with_function)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_function={with_function}",
         with_function=with_function)
    for with_function in [False, True]))
  def test_cond_multiple_results(self, with_function=False):
    def f_jax(pred, x):
      return lax.cond(pred, lambda t: (t + 1., 1.), lambda f: (f + 2., 2.), x)

    self.ConvertAndCompare(f_jax, True, 1., with_function=with_function)
    self.ConvertAndCompare(f_jax, False, 1., with_function=with_function)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_function={with_function}",
         with_function=with_function)
    for with_function in [False, True]))
  def test_while_single_carry(self, with_function=False):
    """A while with a single carry"""
    def func(x):
      # Equivalent to:
      #      for(i=x; i < 4; i++);
      return lax.while_loop(lambda c: c < 4, lambda c: c + 1, x)

    self.ConvertAndCompare(func, 0, with_function=with_function)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_function={with_function}",
         with_function=with_function)
    for with_function in [False, True]))
  def test_while(self, with_function=False):
    # Some constants to capture in the conditional branches
    cond_const = np.ones(3, dtype=np.float32)
    body_const1 = np.full_like(cond_const, 1.)
    body_const2 = np.full_like(cond_const, 2.)

    def func(x):
      # Equivalent to:
      #      c = [1, 1, 1]
      #      for(i=0; i < 3; i++)
      #        c += [1, 1, 1] + [2, 2, 2]
      #
      # The function is set-up so that it captures constants in the
      # body of the functionals. This covers some cases in the representation
      # of the lax.while primitive.
      def cond(idx_carry):
        i, c = idx_carry
        return i < jnp.sum(lax.tie_in(i, cond_const))  # Capture cond_const

      def body(idx_carry):
        i, c = idx_carry
        return (i + 1, c + body_const1 + body_const2)

      return lax.while_loop(cond, body, (0, x))

    self.ConvertAndCompare(func, cond_const, with_function=with_function)


  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_function={with_function}",
         with_function=with_function)
    for with_function in [False, True]))
  def test_while_batched(self, with_function=True):
    """A while with a single carry"""
    def product(x, y):
      # Equivalent to "x * y" implemented as:
      #      res = 0.
      #      for(i=0; i < y; i++)
      #         res += x
      return lax.while_loop(lambda idx_carry: idx_carry[0] < y,
                            lambda idx_carry: (idx_carry[0] + 1,
                                               idx_carry[1] + x),
                            (0, 0.))

    # We use vmap to compute result[i, j] = i * j
    xs = np.arange(4, dtype=np.int32)
    ys = np.arange(5, dtype=np.int32)

    def product_xs_y(xs, y):
      return jax.vmap(product, in_axes=(0, None))(xs, y)
    def product_xs_ys(xs, ys):
      return jax.vmap(product_xs_y, in_axes=(None, 0))(xs, ys)

    self.ConvertAndCompare(product_xs_ys, xs, ys, with_function=with_function)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_function={with_function}",
         with_function=with_function)
    for with_function in [False, True]))
  def test_scan(self, with_function=False):
    def f_jax(xs, ys):
      body_const = np.ones((2, ), dtype=np.float32)  # Test constant capture
      def body(res0, inputs):
        x, y = inputs
        return res0 + x * y, body_const
      return lax.scan(body, 0., (xs, ys))

    arg = np.arange(10, dtype=np.float32)
    self.ConvertAndCompare(f_jax, arg, arg, with_function=with_function)


if __name__ == "__main__":
  absltest.main()
