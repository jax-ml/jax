# Copyright 2025 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Mosaic GPU's `equations` module."""

from absl.testing import parameterized
from jax._src import config
from jax._src import test_util as jtu
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.mosaic.gpu import equations

config.parse_flags_with_absl()

C = equations.Constant
Eq = equations.Equation
V = equations.Variable


class EquationSystemTest(parameterized.TestCase):

  def test_equation_system_is_unsatisfiable_if_assignments_are_incompatible(self):
    v0 = V(0)
    layout0, layout1 = [C(mgpu.WGSplatFragLayout((1, i))) for i in (1, 2)]
    system = equations.EquationSystem(
        equations=[Eq(v0, layout0), Eq(v0, layout1)],
    )
    self.assertIsInstance(equations.reduce(system), equations.Unsatisfiable)

  def test_equation_system_is_unsatisfiable_if_constraints_are_unsatisfiable(
      self,
  ):
    v0 = V(0)
    layout0, layout1 = [C(mgpu.WGSplatFragLayout((1, i))) for i in (1, 2)]
    system = equations.EquationSystem(
        assignments={v0: layout0},
        constraints=[equations.Relayout(v0, layout1)],
    )
    self.assertIsInstance(equations.reduce(system), equations.Unsatisfiable)

  @parameterized.parameters(*equations._SUPPORTED_TILED_RELAYOUTS)
  def test_reduce_equation_system_removes_satisfed_relayouts(self, src, tgt):
    system = equations.EquationSystem(
        constraints=[equations.Relayout(C(src), C(tgt))],
    )
    self.assertEqual(equations.reduce(system), equations.EquationSystem())

  def test_relayout_constraint_does_not_hold_for_incompatible_layouts(self):
    self.assertFalse(
        equations.Relayout(
            C(mgpu.WGMMA_ROW_LAYOUT), C(mgpu.WGMMA_COL_LAYOUT)
        ).holds()
    )

  def test_distinct_constraint_does_not_hold_for_identical_expressions(self):
    self.assertFalse(equations.Distinct(V(1), V(1)).holds())

  def test_distinct_constraint_holds_for_unequal_constants(self):
    layout0, layout1 = mgpu.WGMMA_LAYOUT, mgpu.WGMMA_ROW_LAYOUT
    self.assertTrue(equations.Distinct(C(layout0), C(layout1)).holds())

  def test_distinct_constraint_is_unknown_for_unreduced_unequal_expressions(self):
    self.assertIsNone(equations.Distinct(C(1), V(0)).holds())

  def test_reduce_equation_system_removes_tautological_equations_and_constraints(
      self,
  ):
    v0, v1 = V(0), V(1)
    layout0, layout1 = mgpu.WGMMA_LAYOUT, mgpu.WGMMA_ROW_LAYOUT
    system = equations.EquationSystem(
        equations=[Eq(v0, v1), Eq(v0, v0)],
        constraints=[equations.Relayout(v0, v0),
                     equations.Distinct(C(layout0), C(layout1)),
                     equations.Distinct(v0, v1)],
    )
    self.assertLen(equations.reduce(system).equations, 1)
    self.assertLen(equations.reduce(system).constraints, 1)

  def test_reduce_equation_system_of_simplified_system_is_noop(self):
    v0, v1 = V(0), V(1)
    system = equations.EquationSystem(
        equations=[Eq(v0, v1)],
    )
    self.assertEqual(equations.reduce(system), system)

  def test_reduce_equation_system_assigns_variables_with_known_equations(self):
    v0, v1 = V(0), V(1)
    layout = C(mgpu.WGSplatFragLayout((1, 1)))

    with self.subTest("left-to-right-assignment"):
      system = equations.EquationSystem(
          equations=[Eq(v0, layout), Eq(v0, v1)],
      )
      self.assertEqual(
          equations.reduce(system),
          equations.EquationSystem(assignments={v0: layout, v1: layout})
      )

    with self.subTest("right-to-left-assignment"):
      system = equations.EquationSystem(
          equations=[Eq(v1, layout), Eq(v0, v1)],
      )
      self.assertEqual(
          equations.reduce(system),
          equations.EquationSystem(assignments={v0: layout, v1: layout})
      )

  def test_equation_system_unknowns_are_all_the_variables_without_assignment(self):
    v0, v1, v2, v3 = V(0), V(1), V(2), V(3)
    layout = C(mgpu.WGSplatFragLayout((1, 1)))
    least_replicated = equations.LeastReplicated((v2, v3))
    most_replicated = equations.MostReplicated((least_replicated,))
    system = equations.EquationSystem(assignments={v0: layout},
                                      equations=[Eq(v1, most_replicated)])
    self.assertSequenceEqual(system.unknowns(), [v1, v2, v3])

  def test_intersection_of_conflicting_systems_is_unsatisfiable(self):
    v0 = V(0)
    layout0, layout1 = [C(mgpu.WGSplatFragLayout((1, i))) for i in (1, 2)]
    system0 = equations.EquationSystem(assignments={v0: layout0})
    system1 = equations.EquationSystem(assignments={v0: layout1})
    self.assertIsInstance(system0 & system1, equations.Unsatisfiable)

  def test_intersection_of_compatible_systems_is_union_of_fields(self):
    v0, v1, v2 = V(0), V(1), V(2)
    layout0, layout1, layout2 = [C(mgpu.WGSplatFragLayout((1, i))) for i in (1, 2, 3)]
    system0 = equations.EquationSystem(equations=[Eq(v0, layout0)])
    system1 = equations.EquationSystem(
        assignments={v2: layout2},
        equations=[Eq(v1, layout1)],
    )
    system_intersection = system0 & system1
    self.assertEqual(
        system_intersection,
        equations.EquationSystem(
            assignments={v2: layout2},
            equations=[Eq(v0, layout0), Eq(v1, layout1)],
        ),
    )
    self.assertSequenceEqual(system0.unknowns(), [v0])
    self.assertSequenceEqual(system1.unknowns(), [v1])
    self.assertSequenceEqual(system_intersection.unknowns(), [v0, v1])

  def test_reduce_extracts_most_replicated_expression_correctly(self):
    v0 = V(0)
    shape = (1, 128)
    layout0 = C(mgpu.WGSplatFragLayout(shape))
    layout1 = C(mgpu.WGStridedFragLayout(shape, vec_size=1))
    with self.subTest("most-replicated-expression-exists"):
      system = equations.EquationSystem(
          equations=[Eq(v0, equations.MostReplicated((layout0, layout1)))],
      )
      self.assertEqual(
          equations.reduce(system),
          equations.EquationSystem(assignments={v0: layout0})
      )

    with self.subTest("most-replicated-expression-is-unique-expression"):
      system = equations.EquationSystem(
          equations=[Eq(v0, equations.MostReplicated((layout0,)))],
      )
      self.assertEqual(
          equations.reduce(system),
          equations.EquationSystem(assignments={v0: layout0})
      )

    with self.subTest("most-replicated-expression-does-not-exist"):
      system = equations.EquationSystem(
          equations=[Eq(v0, equations.MostReplicated((layout1, v0)))],
      )
      self.assertEqual(equations.reduce(system), system)

  def test_reduce_extracts_least_replicated_expression_correctly(self):
    v0 = V(0)
    shape = (1, 128)
    layout0 = C(mgpu.WGSplatFragLayout(shape))
    layout1 = C(mgpu.WGStridedFragLayout(shape, vec_size=1))
    with self.subTest("least-replicated-expression-exists"):
      system = equations.EquationSystem(
          equations=[Eq(v0, equations.LeastReplicated([layout0, layout1]))],
      )
      self.assertEqual(
          equations.reduce(system),
          equations.EquationSystem(assignments={v0: layout1})
      )

    with self.subTest("least-replicated-expression-is-unique-expression"):
      system = equations.EquationSystem(
          equations=[Eq(v0, equations.LeastReplicated((layout0,)))],
      )
      self.assertEqual(
          equations.reduce(system),
          equations.EquationSystem(assignments={v0: layout0})
      )

    with self.subTest("least-replicated-expression-does-not-exist"):
      system = equations.EquationSystem(
          equations=[Eq(v0, equations.LeastReplicated((layout0, v0)))],
      )
      self.assertEqual(equations.reduce(system), system)

  def test_reduce_most_replicated_expression_reduces_compatible_layouts(self):
    splat_layout = C(mgpu.WGSplatFragLayout((128, 64)))
    tiled_layout = C(mgpu.WGMMA_LAYOUT)
    self.assertEqual(
        equations.reduce_expression(
            equations.MostReplicated((splat_layout, tiled_layout)),
            {},
        ),
        splat_layout,
    )

  def test_reduce_most_replicated_expression_is_unsatisfiable_for_incompatible_layouts(
      self,
  ):
    splat_layout = C(mgpu.WGSplatFragLayout((1, 2)))
    tiled_layout = C(mgpu.WGMMA_LAYOUT)
    self.assertIsInstance(
        equations.reduce_expression(
            equations.MostReplicated((splat_layout, tiled_layout)),
            {},
        ),
        equations.Unsatisfiable,
    )

  def test_reduce_least_replicated_expression_reduces_compatible_layouts(self):
    splat_layout = C(mgpu.WGSplatFragLayout((128, 64)))
    tiled_layout = C(mgpu.WGMMA_LAYOUT)
    self.assertEqual(
        equations.reduce_expression(
            equations.LeastReplicated((splat_layout, tiled_layout)),
            {},
        ),
        tiled_layout,
    )

  def test_reduce_least_replicated_expression_is_unsatisfiable_for_incompatible_layouts(
      self,
  ):
    splat_layout = C(mgpu.WGSplatFragLayout((1, 2)))
    tiled_layout = C(mgpu.WGMMA_LAYOUT)
    self.assertIsInstance(
        equations.reduce_expression(
            equations.LeastReplicated((splat_layout, tiled_layout)),
            {},
        ),
        equations.Unsatisfiable,
    )

  @parameterized.named_parameters(
      ("reduce_to_row_layout", (1,), mgpu.WGMMA_ROW_LAYOUT),
      ("reduce_to_col_layout", (0,), mgpu.WGMMA_COL_LAYOUT),
  )
  def test_reduce_reduce_expression_reduces_layout(self, axes, expected_layout):
    tiled_layout = C(mgpu.WGMMA_LAYOUT)
    self.assertEqual(
        equations.reduce_expression(
            equations.Reduce(tiled_layout, axes=axes), {}
        ),
        C(expected_layout),
    )

  def test_reduce_reduce_expression_with_unsupported_layout_raises_error(self):
    layout = C(mgpu.WGStridedFragLayout((128, 8), vec_size=8))
    with self.assertRaises(NotImplementedError):
      equations.reduce_expression(equations.Reduce(layout, axes=(0,)), {})


if __name__ == "__main__":
  parameterized.absltest.main(testLoader=jtu.JaxTestLoader())
