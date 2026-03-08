# Copyright 2020 The JAX Authors.
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

import itertools as it
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import api_util
from jax import numpy as jnp
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()


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
              expected,
              api_util.donation_vector(donate_argnums, (),
                                       jax.tree.structure((args, kwargs))))

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

  def test_resolve_kwargs(self):
    def fun(x, y, z=3):
      return x, y, z
    assert api_util.resolve_kwargs(fun, (1,), {"y": 2}) == (1, 2, 3)
    assert api_util.resolve_kwargs(fun, (1, 2), {"z": 3}) == (1, 2, 3)
    assert api_util.resolve_kwargs(
        fun, (), {"x": 1, "y": 2, "z": 3}) == (1, 2, 3)

  def test_resolve_kwargs_with_keyword(self):
    def fun(x, y, z, *, kw=True):
      del kw
      return x, y, z
    assert api_util.resolve_kwargs(fun, (1, 2), {"z": 3}) == (1, 2, 3)
    with self.assertRaisesRegex(TypeError, "keyword arguments"):
      api_util.resolve_kwargs(fun, (1, 2), {"z": 3, "kw": False})

  def test_flatten_axes_valid_prefix(self):
    treedef = jax.tree.structure({"a": [1, 2], "b": 3})
    spec = {"a": 0, "b": 1}
    result = api_util.flatten_axes("test", treedef, spec)
    self.assertEqual(result, [0, 0, 1])

  def test_flatten_axes_error_shows_mismatch_detail(self):
    treedef = jax.tree.structure({"a": 1, "b": 2})
    spec = {"a": 0, "b": (1, 2)}
    with self.assertRaisesRegex(
        ValueError, r"(?s)Mismatch details.*different types"
    ):
      api_util.flatten_axes("test_spec", treedef, spec)

  def test_flatten_axes_error_shows_all_mismatches(self):
    treedef = jax.tree.structure({"a": 1, "b": 2})
    spec = {"a": [0, 1], "b": (2, 3)}
    with self.assertRaisesRegex(ValueError, r"Mismatch details \(2 found\)"):
      api_util.flatten_axes("test_spec", treedef, spec)

  def test_flatten_axis_resources_valid_prefix(self):
    tree = jax.tree.structure({"a": [1, 2], "b": 3})
    shardings = {"a": None, "b": None}
    result = api_util.flatten_axis_resources("test", tree, shardings, False)
    self.assertEqual(result, (None, None, None))

  def test_flatten_axis_resources_error_shows_mismatch_detail(self):
    tree = jax.tree.structure({"a": 1, "b": 2})
    shardings = {"a": None, "b": (None, None)}
    with self.assertRaisesRegex(
        ValueError, r"(?s)Mismatch details.*different types"
    ):
      api_util.flatten_axis_resources("test_spec", tree, shardings, False)

  def test_flatten_axis_resources_error_shows_all_mismatches(self):
    tree = jax.tree.structure({"a": 1, "b": 2})
    shardings = {"a": [None, None], "b": (None, None)}
    with self.assertRaisesRegex(ValueError, r"Mismatch details \(2 found\)"):
      api_util.flatten_axis_resources("test_spec", tree, shardings, False)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
