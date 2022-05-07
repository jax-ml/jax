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

import itertools as it
from typing import Tuple
from absl.testing import absltest
from absl.testing import parameterized
from jax._src import api_util
from jax import numpy as jnp
from jax._src import test_util as jtu
from jax.config import config
import math
config.parse_flags_with_absl()


class ApiUtilTest(jtu.JaxTestCase):

  def assertArgTuplesEqual(self,
                           tuple1: Tuple[Tuple[int, ...], Tuple[str, ...]],
                           tuple2:  Tuple[Tuple[int, ...], Tuple[str, ...]]):
    """
    Asserts that elements and count of argnums and argnames for each of the
    two tuples are the same.
    """
    self.assertLen(tuple1, 2)
    self.assertLen(tuple2, 2)
    self.assertCountEqual(tuple1[0], tuple2[0])
    self.assertCountEqual(tuple1[1], tuple2[1])

  @parameterized.parameters(
    (None, None, (), ()),  # Empty
    (0, None, (0,), ("a",)),  # Integer input
    (None, "a", (0,), ("a",)),  # String input
    ((0, 1, 2), None, (0, 1, 2), ("a", "b", "c")),  # argnums -> argnames,
    (None, ("a", "b", "c"), (0, 1, 2), ("a", "b", "c")),  # argnames -> argnums,
    ((0, 2), None, (0, 2), ("a", "c")),  # Partial argnums -> argnames
    (None, ("a", "c"), (0, 2), ("a", "c")),  # Partial argnames -> argnums
    ((2, 1, 0), None, (2, 1, 0), ("a", "b", "c")),  # Unordered argnums
    (None, ("c", "b", "a"), (0, 1, 2), ("c", "b", "a")),  # Unordered argnames
    ((0,), ("b",), (0, 1), ("a", "b")),  # Mixed
  )
  def test_infer_argnums_and_argnames(self, argnums, argnames, expected_argnums, expected_argnames):
    def f(a, /, b, *, c):
      ...

    self.assertArgTuplesEqual(
      (expected_argnums, expected_argnames),
      api_util.infer_argnums_and_argnames(f, argnums, argnames)
    )

  def test_infer_argnums_and_argnames_invalid_sig(self):
    f = math.log  # See: https://github.com/python/cpython/issues/73485

    self.assertArgTuplesEqual(
      (
        (),
        ("base",)
      ),
      api_util.infer_argnums_and_argnames(f, None, ("base"))
    )

    self.assertArgTuplesEqual(
      (
        (0,),
        ()
      ),
      api_util.infer_argnums_and_argnames(f, (0,), None)
    )

  def test_infer_argnums_and_argnames_var_args(self):
    def g(x, y, *args):
      ...

    argnums, argnames = api_util.infer_argnums_and_argnames(
        g, argnums=(1, 2), argnames=None)

    self.assertArgTuplesEqual(
      (argnums, argnames),
      ((1, 2), ("y",)),
    )

    def h(x, y, **kwargs):
      ...

    argnums, argnames = api_util.infer_argnums_and_argnames(
        h, argnums=None, argnames=('foo', 'bar'))
    self.assertArgTuplesEqual(
      (argnums, argnames),
      ((), ("foo", "bar"))
    )

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

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
