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
from jax._src.interpreters import mlir as mlir_interpreter
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import llvm
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.mosaic.gpu import equations
from jax.experimental.mosaic.gpu import launch_context as lc

config.parse_flags_with_absl()

RL = equations.RegisterLayout
Eq = equations.Equation
V = equations.Variable


def _make_ir_context():
  context = ir.Context()
  context.append_dialect_registry(mlir_interpreter.upstream_dialects)
  context.load_all_available_dialects()
  mgpu.dialect.register_dialect(context)
  return context


class EquationSystemTest(parameterized.TestCase):

  def test_equation_system_is_unsatisfiable_if_assignments_are_incompatible(self):
    v0 = V(0)
    layout0, layout1 = [RL(mgpu.WGSplatFragLayout((1, i))) for i in (1, 2)]
    system = equations.EquationSystem(
        equations=[Eq(v0, layout0), Eq(v0, layout1)],
    )
    self.assertIsInstance(equations.reduce(system), equations.Unsatisfiable)

  def test_equation_system_is_unsatisfiable_if_constraints_are_unsatisfiable(
      self,
  ):
    v0 = V(0)
    layout0, layout1 = [RL(mgpu.WGSplatFragLayout((1, i))) for i in (1, 2)]
    system = equations.EquationSystem(
        assignments={v0: layout0},
        constraints=[equations.Relayout(v0, layout1)],
    )
    self.assertIsInstance(equations.reduce(system), equations.Unsatisfiable)

  @parameterized.parameters(*equations._SUPPORTED_TILED_RELAYOUTS)
  def test_reduce_equation_system_removes_satisfed_relayouts(self, src, tgt):
    system = equations.EquationSystem(
        constraints=[equations.Relayout(RL(src), RL(tgt))],
    )
    self.assertEqual(equations.reduce(system), equations.EquationSystem())

  def test_relayout_constraint_does_not_hold_for_incompatible_layouts(self):
    self.assertFalse(
        equations.Relayout(
            RL(mgpu.WGMMA_ROW_LAYOUT), RL(mgpu.WGMMA_COL_LAYOUT)
        ).holds()
    )

  def test_distinct_constraint_does_not_hold_for_identical_expressions(self):
    self.assertFalse(equations.Distinct(V(1), V(1)).holds())

  def test_distinct_constraint_holds_for_unequal_constants(self):
    layout0, layout1 = mgpu.WGMMA_LAYOUT, mgpu.WGMMA_ROW_LAYOUT
    self.assertTrue(equations.Distinct(RL(layout0), RL(layout1)).holds())

  def test_distinct_constraint_is_unknown_for_unreduced_unequal_expressions(self):
    self.assertIsNone(equations.Distinct(RL(1), V(0)).holds())

  def test_reduce_equation_system_removes_tautological_equations_and_constraints(
      self,
  ):
    v0, v1 = V(0), V(1)
    layout0, layout1 = mgpu.WGMMA_LAYOUT, mgpu.WGMMA_ROW_LAYOUT
    system = equations.EquationSystem(
        equations=[Eq(v0, v1), Eq(v0, v0)],
        constraints=[
            equations.Relayout(v0, v0),
            equations.Distinct(RL(layout0), RL(layout1)),
            equations.Distinct(v0, v1),
        ],
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
    layout = RL(mgpu.WGSplatFragLayout((1, 1)))

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
    layout = RL(mgpu.WGSplatFragLayout((1, 1)))
    least_replicated = equations.LeastReplicated((v2, v3))
    most_replicated = equations.MostReplicated((least_replicated,))
    system = equations.EquationSystem(assignments={v0: layout},
                                      equations=[Eq(v1, most_replicated)])
    self.assertSequenceEqual(system.unknowns(), [v1, v2, v3])

  def test_intersection_of_conflicting_systems_is_unsatisfiable(self):
    v0 = V(0)
    layout0, layout1 = [RL(mgpu.WGSplatFragLayout((1, i))) for i in (1, 2)]
    system0 = equations.EquationSystem(assignments={v0: layout0})
    system1 = equations.EquationSystem(assignments={v0: layout1})
    self.assertIsInstance(system0 & system1, equations.Unsatisfiable)

  def test_intersection_of_compatible_systems_is_union_of_fields(self):
    v0, v1, v2 = V(0), V(1), V(2)
    layout0, layout1, layout2 = [
        RL(mgpu.WGSplatFragLayout((1, i))) for i in (1, 2, 3)
    ]
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
    layout0 = RL(mgpu.WGSplatFragLayout(shape))
    layout1 = RL(mgpu.WGStridedFragLayout(shape, vec_size=1))
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
    layout0 = RL(mgpu.WGSplatFragLayout(shape))
    layout1 = RL(mgpu.WGStridedFragLayout(shape, vec_size=1))
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
    splat_layout = RL(mgpu.WGSplatFragLayout((128, 64)))
    tiled_layout = RL(mgpu.WGMMA_LAYOUT)
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
    splat_layout = RL(mgpu.WGSplatFragLayout((1, 2)))
    tiled_layout = RL(mgpu.WGMMA_LAYOUT)
    self.assertIsInstance(
        equations.reduce_expression(
            equations.MostReplicated((splat_layout, tiled_layout)),
            {},
        ),
        equations.Unsatisfiable,
    )

  def test_reduce_least_replicated_expression_reduces_compatible_layouts(self):
    splat_layout = RL(mgpu.WGSplatFragLayout((128, 64)))
    tiled_layout = RL(mgpu.WGMMA_LAYOUT)
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
    splat_layout = RL(mgpu.WGSplatFragLayout((1, 2)))
    tiled_layout = RL(mgpu.WGMMA_LAYOUT)
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
    tiled_layout = RL(mgpu.WGMMA_LAYOUT)
    self.assertEqual(
        equations.reduce_expression(
            equations.Reduce(tiled_layout, axes=axes), {}
        ),
        RL(expected_layout),
    )

  def test_reduce_reduce_expression_with_unsupported_layout_is_irreducible(self):
    layout = RL(mgpu.WGStridedFragLayout((128, 8), vec_size=8))
    expr = equations.Reduce(layout, axes=(0,))
    self.assertEqual(equations.reduce_expression(expr, {}), expr)

  def test_reduce_broadcast_of_splat_layout_is_reduced_to_splat_layout(self):
    layout = RL(mgpu.WGSplatFragLayout((128,)))
    valid_shape = (128, 8)
    self.assertEqual(
        equations.reduce_expression(
            equations.BroadcastInDim(layout, axes=(0,), shape=valid_shape), {}
        ),
        RL(mgpu.WGSplatFragLayout((128, 8))),
    )

  def test_reduce_broadcast_of_splat_layout_is_unsatisfiable_for_incompatible_shape(self):
    layout = RL(mgpu.WGSplatFragLayout((128,)))
    invalid_shape = (129, 8)
    self.assertIsInstance(
        equations.reduce_expression(
            equations.BroadcastInDim(layout, axes=(0,), shape=invalid_shape), {}
        ),
        equations.Unsatisfiable,
    )

  def test_reduce_broadcast_of_strided_layout_is_irreducible(self):
    layout = RL(mgpu.WGStridedFragLayout((128,), vec_size=1))
    expr = equations.BroadcastInDim(layout, axes=(0,), shape=(128, 8))
    self.assertEqual(equations.reduce_expression(expr, {}), expr)

  def test_reduce_broadcast_of_tiled_layout_is_irreducible(self):
    layout = RL(mgpu.WGMMA_LAYOUT)
    expr = equations.BroadcastInDim(layout, axes=(1, 2), shape=(8, 128, 8))
    self.assertEqual(equations.reduce_expression(expr, {}), expr)

  def test_reduce_reshape_of_splat_layout_is_reduced_to_splat_layout(self):
    layout = RL(mgpu.WGSplatFragLayout((1024,)))
    source_shape, target_shape = (1024,), (128, 8)
    self.assertEqual(
        equations.reduce_expression(
            equations.Reshape(layout, source_shape, target_shape), {}
        ),
        RL(mgpu.WGSplatFragLayout(target_shape)),
    )

  def test_reduce_reshape_of_strided_layout_is_reduced_to_strided_layout(self):
    layout = RL(mgpu.WGStridedFragLayout((1024,), vec_size=8))
    source_shape, target_shape = (1024,), (128, 8)
    self.assertEqual(
        equations.reduce_expression(
            equations.Reshape(layout, source_shape, target_shape), {}
        ),
        RL(mgpu.WGStridedFragLayout(target_shape, vec_size=8)),
    )

  def test_reduce_reshape_of_tiled_layout_with_indivisible_shape_is_irreducible(self):
    layout = RL(mgpu.WGMMA_LAYOUT)
    source_shape, target_shape = (128, 8), (129, 8)
    eq = equations.Reshape(layout, source_shape, target_shape)
    self.assertEqual(equations.reduce_expression(eq, {}), eq)

  def test_reduce_reshape_of_tiled_layout_with_modified_minor_tiled_dimensions_is_irreducible(
      self,
  ):
    layout = RL(mgpu.WGMMA_LAYOUT)
    source_shape, target_shape = (2, 128, 8), (2, 64, 16)
    eq = equations.Reshape(layout, source_shape, target_shape)
    self.assertEqual(equations.reduce_expression(eq, {}), eq)

  def test_reduce_reshape_of_tiled_layout_with_compatible_shape_is_identity(
      self,
  ):
    layout = RL(mgpu.WGMMA_LAYOUT)
    source_shape, target_shape = (2, 128, 8), (256, 8)
    eq = equations.Reshape(layout, source_shape, target_shape)
    self.assertEqual(equations.reduce_expression(eq, {}), layout)

  def test_relayout_of_non_splat_to_splat_is_unsatisfiable_shortcut(
      self,
  ):
    splat_layout = RL(mgpu.WGSplatFragLayout((128,)))
    v0, v1 = V(0), V(1)
    system = equations.EquationSystem(
        assignments={v1: splat_layout},
        constraints=[
            equations.Distinct(v0, splat_layout),
            equations.Relayout(v0, v1),
        ],
    )
    self.assertIsInstance(equations.reduce(system), equations.Unsatisfiable)

  def test_saturate_distinct_from_splat_does_not_create_duplicate_constraints(
      self,
  ):
    splat_layout = RL(mgpu.WGSplatFragLayout((128,)))
    v0, v1, v2 = V(0), V(1), V(2)
    system = equations.EquationSystem(constraints = [
          equations.Distinct(v0, splat_layout),
          equations.Distinct(v1, splat_layout),
          equations.Relayout(v0, v2),
          equations.Relayout(v1, v2),
      ],
    )

    self.assertEqual(
        equations.saturate_distinct_from_splat(system),
        equations.EquationSystem(
            constraints=[
                equations.Distinct(v0, splat_layout),
                equations.Distinct(v1, splat_layout),
                equations.Relayout(v0, v2),
                equations.Relayout(v1, v2),
                equations.Distinct(v2, splat_layout),
            ],
        ),
    )

  def test_saturate_distinct_from_splat_does_not_affect_non_splat(
      self,
  ):
    splat_layout = RL(mgpu.WGSplatFragLayout((128,)))
    wgmma_layout = RL(mgpu.WGMMA_LAYOUT)
    v0, v1, v2, v3, v4 = V(0), V(1), V(2), V(3), V(4)
    system = equations.EquationSystem(constraints = [
          equations.Distinct(v0, splat_layout),
          equations.Distinct(v1, wgmma_layout),
          equations.Relayout(v0, v2),
          equations.Relayout(v1, v3),
          equations.Relayout(v4, v0),
      ],
    )

    self.assertEqual(
        equations.saturate_distinct_from_splat(system),
        equations.EquationSystem(
            constraints=[
                equations.Distinct(v0, splat_layout),
                equations.Distinct(v1, wgmma_layout),
                equations.Relayout(v0, v2),
                equations.Relayout(v1, v3),
                equations.Relayout(v4, v0),
                equations.Distinct(v2, splat_layout),
            ],
        ),
    )

  @parameterized.parameters(
      (mgpu.WGMMA_LAYOUT, (64, 64), True),
      (mgpu.WGMMA_LAYOUT, (64,), False),
      (mgpu.WGMMA_LAYOUT, None, False),
      (mgpu.WGMMA_ROW_LAYOUT, None, True),
      (mgpu.WGMMA_ROW_LAYOUT, (64,), False),
      (mgpu.WGMMA_COL_LAYOUT, None, True),
      (mgpu.WGMMA_COL_LAYOUT, (64,), False),
      (mgpu.WGSplatFragLayout((16, 16)), None, True),
      (mgpu.WGSplatFragLayout((16, 16)), (16,), False),
      (mgpu.WGStridedFragLayout((16, 128), vec_size=4), None, True),
      (mgpu.WGStridedFragLayout((16, 128), vec_size=4), (1,), False),
  )
  def test_smem_is_transferable(self, layout, tiling, expected):
    eq_layout = equations.RegisterLayout(layout)
    eq_tiling = equations.SMEMTiling(lc.TileTransform(tiling) if tiling else None)

    is_transferable = equations.IsTransferable(eq_layout, eq_tiling, ())
    self.assertEqual(is_transferable.holds(), expected)
    is_transferable = equations.IsTransferable(eq_tiling, eq_layout, ())
    self.assertEqual(is_transferable.holds(), expected)

  def test_transposed_constraint(self):
    def transposed(lhs, rhs):
      lhs = equations.SMEMTiling(None if lhs is None else lc.TileTransform(lhs))
      rhs = equations.SMEMTiling(None if rhs is None else lc.TileTransform(rhs))
      return equations.Transposed(lhs, rhs)

    self.assertTrue(transposed(None, None).holds())
    self.assertTrue(transposed((1,), (1,)).holds())
    self.assertFalse(transposed((1,), (2,)).holds())
    self.assertTrue(transposed((1, 2), (2, 1)).holds())
    self.assertTrue(transposed((2, 2), (2, 2)).holds())
    self.assertFalse(transposed((2, 3), (2, 2)).holds())

  def test_divides_constraint(self):
    def divides(tiling, dims):
      tiling = None if tiling is None else lc.TileTransform(tiling)
      return equations.Divides(equations.SMEMTiling(tiling), dims)

    self.assertTrue(divides(None, []).holds())
    self.assertTrue(divides(None, [[5, 15], [16]]).holds())

    self.assertTrue(divides((5, 8), [[3], [5], [8]]).holds())
    self.assertTrue(divides((5, 8), [[3], [5, 10], [8, 0, 16]]).holds())
    self.assertFalse(divides((1, 3, 5, 8), [[3], [5, 10], [8, 0, 16]]).holds())

    self.enter_context(_make_ir_context())
    self.enter_context(ir.Location.unknown())
    ir.Module.create()

    c = lambda x: arith.constant(ir.IntegerType.get_signless(32), x)
    self.assertTrue(divides((5, 8), [[c(3)], [c(5)], [c(8)]]).holds())
    self.assertTrue(divides((5, 8), [[c(3)], [c(5), 10], [c(8), 16]]).holds())
    self.assertTrue(divides((5, 8), [[c(3)], [5, c(10)], [8]]).holds())
    self.assertFalse(divides((5, 8), [[c(3)], [5, c(4)], [8]]).holds())

    u = llvm.mlir_undef(ir.IntegerType.get_signless(32))
    self.assertTrue(divides((5, 8), [[u], [c(5)], [c(8)]]).holds())
    self.assertFalse(divides((5, 8), [[10], [8, u]]).holds())

  def test_merge_divides_constraints(self):
    def divides(var, dims):
      return equations.Divides(equations.Variable(var), dims)

    self.assertEqual(equations.merge_divides_constraints([
        divides(0, [[16]]),
        divides(1, [[8]]),
    ]), [
        divides(0, [[16]]),
        divides(1, [[8]]),
    ])

    self.assertEqual(equations.merge_divides_constraints([
        divides(0, [[16]]),
        divides(0, [[8]]),
    ]), [divides(0, [[16, 8]])])

    self.assertEqual(equations.merge_divides_constraints([
        divides(0, [[16, 10]]),
        divides(0, [[8]]),
    ]), [divides(0, [[16, 10, 8]])])

    self.assertEqual(equations.merge_divides_constraints([
        divides(0, [[16, 10]]),
        divides(0, [[5],[8]]),
    ]), [divides(0, [[5], [8, 16, 10]])])

    self.assertEqual(equations.merge_divides_constraints([
        divides(0, [[16, 10]]),
        divides(0, [[5],[8]]),
        divides(1, [[1], [2, 4], [5, 10]]),
        divides(1, [[9], [20]]),
    ]), [
        divides(0, [[5], [8, 16, 10]]),
        divides(1, [[1], [2, 4, 9], [5, 10, 20]]),
    ])

  def test_tiled_constraint(self,):
    def tiled(tiling, dims):
      tiling = None if tiling is None else lc.TileTransform(tiling)
      return equations.Tiled( equations.SMEMTiling(tiling), dims)

    self.assertTrue(tiled(None, 0).holds())
    self.assertFalse(tiled((1,), 0).holds())
    self.assertFalse(tiled(None, 1).holds())
    self.assertTrue(tiled((1,), 1).holds())
    self.assertTrue(tiled((1,2), 2).holds())
    self.assertFalse(tiled((1,2), 1).holds())


if __name__ == "__main__":
  parameterized.absltest.main(testLoader=jtu.JaxTestLoader())
