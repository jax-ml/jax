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
"""Tests for jax.api_util."""

import functools
import itertools as it
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import api_util
from jax import numpy as jnp
from jax import test_util as jtu

from jax.config import config
config.parse_flags_with_absl()


class ApiUtilTest(jtu.JaxTestCase):

  def test_donation_vector(self):
    params = {"a": jnp.ones([]), "b": jnp.ones([])}
    state = {"c": jnp.ones([]), "d": jnp.ones([])}
    x = jnp.ones([])
    args = params, state, x

    for size in range(4):
      for donate_argnums in it.permutations((0, 1, 2), size):
        for kwargs in ({}, {"a": x}):
          expected = ()
          expected += (True, True) if 0 in donate_argnums else (False, False)
          expected += (True, True) if 1 in donate_argnums else (False, False)
          expected += (True,) if 2 in donate_argnums else (False,)
          if kwargs:
            expected += (False,)
          self.assertEqual(
              expected, api_util.donation_vector(donate_argnums, args, kwargs))

  @parameterized.parameters(
      ((0,), (0,)),
      ((0, 1), (1, 2)),
      ((0, 1, 2), (0, 1, 2)),
  )
  def test_rebase_donate_argnums_rejects_overlapping(self, donate, static):
    with self.assertRaisesRegex(ValueError, "cannot intersect"):
      api_util.rebase_donate_argnums(donate, static)

  @parameterized.parameters(
      ((), (), ()),
      ((), (1, 2, 3), ()),
      ((0,), (2, 3), (0,)),
      ((0, 1), (2, 3), (0, 1)),
      ((2, 3), (0, 1), (0, 1)),
      ((3, 2), (1, 0), (0, 1)),
      ((3,), (0, 1), (1,)),
      ((3, 3, 3,), (0, 1), (1,)),
  )
  def test_rebase_donate_argnums(self, donate, static, expected):
    self.assertEqual(expected,
                     api_util.rebase_donate_argnums(donate, static))

  @parameterized.parameters(
      (1., 2., 3., (2., 12.)),
      (2., 4., 6., (4., 48.)),
  )
  def test_kwargable_grad_shoud_pass(self, x, y, z, expected):
    @api_util.kwargable
    @functools.partial(jax.grad, argnums=(0, 2))
    def f(x, y, z=z):
      return x**2 + (y * z**2)

    self.assertSequenceAlmostEqual(f(x, y, z), expected)
    self.assertSequenceAlmostEqual(f(x, y), expected)
    self.assertSequenceAlmostEqual(f(x, y=y), expected)
    self.assertSequenceAlmostEqual(f(y=y, x=x), expected)
    self.assertSequenceAlmostEqual(f(z=z, y=y, x=x), expected)

  @parameterized.parameters(
      (1., 2., 3., (2., 12.)),
      (2., 4., 6., (4., 48.)),
  )
  def test_kwargable_grad_should_fail(self, x, y, z, expected):
    @api_util.kwargable
    @functools.partial(jax.grad, argnums=(0, 2))
    def f(x, y, z=z):
      return x**2 + (y * z**2)

    name = f.__name__
    expected = fr"^{name}\(\) missing positional arguments: 'x' and 'y'$"
    with self.assertRaisesRegex(TypeError, expected):
      f()          # pylint: disable=no-value-for-parameter

    expected = fr"^{name}\(\) got unexpected keyword argument: 'q'$"
    with self.assertRaisesRegex(TypeError, expected):
      f(x, y, q=z)  # pylint: disable=unexpected-keyword-arg

    expected = fr"^{name}\(\) got multiple values for argument: 'y'$"
    with self.assertRaisesRegex(TypeError, expected):
      f(x, y, y=z)  # pylint: disable=redundant-keyword-arg

  def test_kwargable_vmap_shoud_pass(self):
    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])
    z = jnp.array([7, 8, 9])
    expected = jnp.array([35, 56, 81])

    @api_util.kwargable
    @jax.vmap
    def f(x, y, z=z):
      return (x + y) * z

    self.assertSequenceAlmostEqual(f(x, y, z), expected)
    self.assertSequenceAlmostEqual(f(x, y), expected)
    self.assertSequenceAlmostEqual(f(x, y=y), expected)
    self.assertSequenceAlmostEqual(f(y=y, x=x), expected)
    self.assertSequenceAlmostEqual(f(z=z, y=y, x=x), expected)

  def test_kwargable_vmap_should_fail(self):
    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])
    z = jnp.array([7, 8, 9])
    expected = jnp.array([35, 56, 81])

    @api_util.kwargable
    @jax.vmap
    def f(x, y, z=z):
      return (x + y) * z

    name = f.__name__
    expected = fr"^{name}\(\) missing positional arguments: 'x' and 'y'$"
    with self.assertRaisesRegex(TypeError, expected):
      f()          # pylint: disable=no-value-for-parameter

    expected = fr"^{name}\(\) got unexpected keyword argument: 'q'$"
    with self.assertRaisesRegex(TypeError, expected):
      f(x, y, q=z)  # pylint: disable=unexpected-keyword-arg

    expected = fr"^{name}\(\) got multiple values for argument: 'y'$"
    with self.assertRaisesRegex(TypeError, expected):
      f(x, y, y=z)  # pylint: disable=redundant-keyword-arg

if __name__ == "__main__":
  absltest.main()
