# Copyright 2026 The JAX Authors.
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

import string
from absl.testing import absltest
from absl.testing import parameterized
import hypothesis
from hypothesis import strategies as st
import jax
from jax._src import test_util as jtu
from jax._src.pallas import einshape
import numpy as np


jax.config.parse_flags_with_absl()
jtu.setup_hypothesis()


class EinshapeParseTest(parameterized.TestCase):

  @parameterized.parameters(
      ("ab->ba", (2, 3), [einshape.Transpose((1, 0))]),
      (
          "ab(cd)->cabd",
          (2, 3, 20),
          [einshape.SplitDims(2, (4, 5)), einshape.Transpose((2, 0, 1, 3))],
          {"c": 4},
      ),
      ("abcd->ab(cd)", (2, 3, 4, 5), [einshape.MergeDims(2, 2)]),
      ("a(bc)->abc", (10, 12), [einshape.SplitDims(1, (3, 4))], {"b": 3}),
      (
          "(ab)c->(ba)c",
          (6, 5),
          [
              einshape.SplitDims(0, (2, 3)),
              einshape.Transpose((1, 0, 2)),
              einshape.MergeDims(0, 2),
          ],
          {"a": 2},
      ),
      (
          "ab(cde)->cadeb",
          (2, 3, 4 * 5 * 6),
          [
              einshape.SplitDims(2, (4, 5, 6)),
              einshape.Transpose((2, 0, 3, 4, 1)),
          ],
          {"c": 4, "d": 5},
      ),
  )
  def test_get_einshape_transforms(
      self, equation, input_shape, expected_ops, sizes=None
  ):
    sizes = sizes or {}
    ops = einshape.get_einshape_transforms(equation, input_shape, **sizes)
    self.assertEqual(ops, expected_ops)

  def test_identity(self):
    ops = einshape.get_einshape_transforms("abc->abc", (2, 3, 4))
    self.assertEqual(ops, [])

  @hypothesis.given(
      st.lists(st.integers(1, 4), min_size=1, max_size=5).map(tuple), st.data()
  )
  @hypothesis.settings(max_examples=50, deadline=None)
  def test_hypothesis_get_einshape_transforms(self, atomic_shape, data):
    names = list(string.ascii_lowercase)[: len(atomic_shape)]
    dim_sizes = dict(zip(names, atomic_shape))

    # Randomly group names for LHS
    lhs_groups = []
    remaining_names = list(names)
    while remaining_names:
      group_size = data.draw(st.integers(1, len(remaining_names)))
      lhs_groups.append(remaining_names[:group_size])
      remaining_names = remaining_names[group_size:]

    # Randomly group names for RHS (after permutation)
    rhs_names = data.draw(st.permutations(names))
    rhs_groups = []
    remaining_rhs_names = list(rhs_names)
    while remaining_rhs_names:
      group_size = data.draw(st.integers(1, len(remaining_rhs_names)))
      rhs_groups.append(remaining_rhs_names[:group_size])
      remaining_rhs_names = remaining_rhs_names[group_size:]

    def format_side(groups):
      res = ""
      for g in groups:
        if len(g) == 1:
          res += g[0]
        else:
          res += "(" + "".join(g) + ")"
      return res

    equation = f"{format_side(lhs_groups)}->{format_side(rhs_groups)}"

    # Construct input shape
    lhs_shape = []
    for g in lhs_groups:
      prod = 1
      for n in g:
        prod *= dim_sizes[n]
      lhs_shape.append(prod)

    # We might need to provide some sizes for LHS splits
    kwargs = {}
    for g in lhs_groups:
      if len(g) > 1:
        for n in g[:-1]:
          kwargs[n] = dim_sizes[n]

    ops = einshape.get_einshape_transforms(equation, tuple(lhs_shape), **kwargs)

    # Verify the ops by applying them to a symbolic or dummy shape
    current_shape = list(lhs_shape)
    for op in ops:
      if isinstance(op, einshape.SplitDims):
        self.assertEqual(current_shape[op.index], np.prod(op.sizes))
        current_shape[op.index : op.index + 1] = list(op.sizes)
      elif isinstance(op, einshape.MergeDims):
        merged_size = np.prod(current_shape[op.index : op.index + op.count])
        current_shape[op.index : op.index + op.count] = [merged_size]
      elif isinstance(op, einshape.Transpose):
        current_shape = [current_shape[i] for i in op.permutation]

    # Verify final shape matches RHS
    expected_rhs_shape = []
    for g in rhs_groups:
      prod = 1
      for n in g:
        prod *= dim_sizes[n]
      expected_rhs_shape.append(prod)
    self.assertEqual(tuple(current_shape), tuple(expected_rhs_shape))


if __name__ == "__main__":
  absltest.main()
