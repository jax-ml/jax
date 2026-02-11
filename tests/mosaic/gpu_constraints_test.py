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
"""Tests for Mosaic GPU's `constraints` module."""

from absl.testing import parameterized
from jax._src import config
from jax._src import test_util as jtu
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.mosaic.gpu import constraints as cs
from jax.experimental.mosaic.gpu import fragmented_array as fa
from jax.experimental.mosaic.gpu import launch_context as lc
from jax.experimental.mosaic.gpu import tcgen05

config.parse_flags_with_absl()

RL = cs.RegisterLayout
Eq = cs.Equals
V = cs.Variable


class ConstraintSystemTest(parameterized.TestCase):

  def test_constraint_system_is_unsatisfiable_if_assignments_are_incompatible(
      self,
  ):
    v0 = V(0)
    layout0, layout1 = [RL(mgpu.WGSplatFragLayout((1, i))) for i in (1, 2)]
    system = cs.ConstraintSystem(
        constraints=[Eq(v0, layout0), Eq(v0, layout1)],
    )
    self.assertIsInstance(cs.reduce(system), cs.Unsatisfiable)

  def test_constraint_system_is_unsatisfiable_if_constraints_are_unsatisfiable(
      self,
  ):
    v0 = V(0)
    layout0, layout1 = [RL(mgpu.WGSplatFragLayout((1, i))) for i in (1, 2)]
    system = cs.ConstraintSystem(
        assignments={v0: layout0},
        constraints=[cs.Relayout(v0, layout1, 32)],
    )
    self.assertIsInstance(cs.reduce(system), cs.Unsatisfiable)

  @parameterized.parameters(*cs._SUPPORTED_TILED_RELAYOUTS)
  def test_reduce_constraint_system_removes_satisfed_relayouts(self, src, tgt):
    system = cs.ConstraintSystem(
        constraints=[cs.Relayout(RL(src), RL(tgt), 4)],
    )
    self.assertEqual(cs.reduce(system), cs.ConstraintSystem())

  def test_relayout_constraint_does_not_hold_for_incompatible_layouts(self):
    self.assertFalse(
        cs.Relayout(
            RL(mgpu.WGMMA_ROW_LAYOUT), RL(mgpu.WGMMA_COL_LAYOUT), 32
        ).holds()
    )

  def test_not_of_type_constraint_holds_for_different_types(self):
    layout = RL(mgpu.WGMMA_LAYOUT)
    self.assertTrue(cs.NotOfType(layout, mgpu.WGSplatFragLayout).holds())

  def test_not_of_type_constraint_does_not_holds_for_same_types(self):
    layout = RL(mgpu.WGSplatFragLayout((1, 128)))
    self.assertFalse(cs.NotOfType(layout, mgpu.WGSplatFragLayout).holds())

  def test_not_of_type_constraint_is_unknown_for_unreduced_expression(self):
    self.assertIsNone(cs.NotOfType(V(0), mgpu.WGSplatFragLayout).holds())

  def test_reduce_constraint_system_removes_tautological_constraints_and_constraints(
      self,
  ):
    v0, v1 = V(0), V(1)
    system = cs.ConstraintSystem(
        constraints=[
            Eq(v0, v1),
            Eq(v0, v0),
            cs.Relayout(v0, v0, 32),
            cs.NotOfType(RL(mgpu.WGMMA_LAYOUT), mgpu.WGSplatFragLayout),
            cs.NotOfType(v1, mgpu.WGSplatFragLayout),
        ],
    )
    self.assertLen(cs.reduce(system).constraints, 2)

  def test_reduce_constraint_system_of_simplified_system_is_noop(self):
    v0, v1 = V(0), V(1)
    system = cs.ConstraintSystem(constraints=[Eq(v0, v1)])
    self.assertEqual(cs.reduce(system), system)

  def test_reduce_constraint_system_assigns_variables_with_known_constraints(
      self,
  ):
    v0, v1 = V(0), V(1)
    layout = RL(mgpu.WGSplatFragLayout((1, 1)))

    with self.subTest("left-to-right-assignment"):
      system = cs.ConstraintSystem(
          constraints=[Eq(v0, layout), Eq(v0, v1)],
      )
      self.assertEqual(
          cs.reduce(system),
          cs.ConstraintSystem(assignments={v0: layout, v1: layout}),
      )

    with self.subTest("right-to-left-assignment"):
      system = cs.ConstraintSystem(
          constraints=[Eq(v1, layout), Eq(v0, v1)],
      )
      self.assertEqual(
          cs.reduce(system),
          cs.ConstraintSystem(assignments={v0: layout, v1: layout}),
      )

  def test_constraint_system_unknowns_are_all_the_variables_without_assignment(
      self,
  ):
    v0, v1, v2, v3 = V(0), V(1), V(2), V(3)
    layout = RL(mgpu.WGSplatFragLayout((1, 1)))
    system = cs.ConstraintSystem(
        assignments={v0: layout},
        constraints=[Eq(v1, v2), cs.Relayout(v2, v3, 32)],
    )
    self.assertSequenceEqual(system.unknowns(), [v1, v2, v3])

  def test_intersection_of_conflicting_systems_is_unsatisfiable(self):
    v0 = V(0)
    layout0, layout1 = [RL(mgpu.WGSplatFragLayout((1, i))) for i in (1, 2)]
    system0 = cs.ConstraintSystem(assignments={v0: layout0})
    system1 = cs.ConstraintSystem(assignments={v0: layout1})
    self.assertIsInstance(system0 & system1, cs.Unsatisfiable)

  def test_intersection_of_compatible_systems_is_union_of_fields(self):
    v0, v1, v2 = V(0), V(1), V(2)
    layout0, layout1, layout2 = [
        RL(mgpu.WGSplatFragLayout((1, i))) for i in (1, 2, 3)
    ]
    system0 = cs.ConstraintSystem(constraints=[Eq(v0, layout0)])
    system1 = cs.ConstraintSystem(
        assignments={v2: layout2},
        constraints=[Eq(v1, layout1)],
    )
    system_intersection = system0 & system1
    self.assertEqual(
        system_intersection,
        cs.ConstraintSystem(
            assignments={v2: layout2},
            constraints=[Eq(v0, layout0), Eq(v1, layout1)],
        ),
    )
    self.assertSequenceEqual(system0.unknowns(), [v0])
    self.assertSequenceEqual(system1.unknowns(), [v1])
    self.assertSequenceEqual(system_intersection.unknowns(), [v0, v1])

  @parameterized.named_parameters(
      ("reduce_to_row_layout", (1,), mgpu.WGMMA_ROW_LAYOUT),
      ("reduce_to_col_layout", (0,), mgpu.WGMMA_COL_LAYOUT),
  )
  def test_reduce_reduce_expression_reduces_layout(self, axes, expected_layout):
    tiled_layout = RL(mgpu.WGMMA_LAYOUT)
    self.assertEqual(
        cs.reduce_expression(cs.Reduce(tiled_layout, axes=axes), {}),
        RL(expected_layout),
    )

  def test_reduce_reduce_expression_with_unsupported_layout_is_irreducible(self):
    layout = RL(mgpu.WGStridedFragLayout((128, 8), vec_size=8))
    expr = cs.Reduce(layout, axes=(0,))
    self.assertEqual(cs.reduce_expression(expr, {}), expr)

  def test_reduce_broadcast_of_splat_layout_is_reduced_to_splat_layout(self):
    layout = RL(mgpu.WGSplatFragLayout((128,)))
    valid_shape = (128, 8)
    self.assertEqual(
        cs.reduce_expression(
            cs.BroadcastInDim(layout, axes=(0,), shape=valid_shape), {}
        ),
        RL(mgpu.WGSplatFragLayout((128, 8))),
    )

  def test_reduce_broadcast_of_splat_layout_is_unsatisfiable_for_incompatible_shape(self):
    layout = RL(mgpu.WGSplatFragLayout((128,)))
    invalid_shape = (129, 8)
    self.assertIsInstance(
        cs.reduce_expression(
            cs.BroadcastInDim(layout, axes=(0,), shape=invalid_shape),
            {},
        ),
        cs.Unsatisfiable,
    )

  def test_reduce_broadcast_of_strided_layout_is_irreducible(self):
    layout = RL(mgpu.WGStridedFragLayout((128,), vec_size=1))
    expr = cs.BroadcastInDim(layout, axes=(0,), shape=(128, 8))
    self.assertEqual(cs.reduce_expression(expr, {}), expr)

  def test_reduce_broadcast_of_tiled_layout_is_irreducible(self):
    layout = RL(mgpu.WGMMA_LAYOUT)
    expr = cs.BroadcastInDim(layout, axes=(1, 2), shape=(8, 128, 8))
    self.assertEqual(cs.reduce_expression(expr, {}), expr)

  def test_reduce_reshape_of_splat_layout_is_reduced_to_splat_layout(self):
    layout = RL(mgpu.WGSplatFragLayout((1024,)))
    source_shape, target_shape = (1024,), (128, 8)
    self.assertEqual(
        cs.reduce_expression(
            cs.Reshape(layout, source_shape, target_shape), {}
        ),
        RL(mgpu.WGSplatFragLayout(target_shape)),
    )

  def test_reduce_reshape_of_strided_layout_is_reduced_to_strided_layout(self):
    layout = RL(mgpu.WGStridedFragLayout((1024,), vec_size=8))
    source_shape, target_shape = (1024,), (128, 8)
    self.assertEqual(
        cs.reduce_expression(
            cs.Reshape(layout, source_shape, target_shape), {}
        ),
        RL(mgpu.WGStridedFragLayout(target_shape, vec_size=8)),
    )

  def test_reduce_reshape_of_tiled_layout_with_indivisible_shape_is_irreducible(self):
    layout = RL(mgpu.WGMMA_LAYOUT)
    source_shape, target_shape = (128, 8), (129, 8)
    eq = cs.Reshape(layout, source_shape, target_shape)
    self.assertEqual(cs.reduce_expression(eq, {}), eq)

  def test_reduce_reshape_of_tiled_layout_with_modified_minor_tiled_dimensions_is_irreducible(
      self,
  ):
    layout = RL(mgpu.WGMMA_LAYOUT)
    source_shape, target_shape = (2, 128, 8), (2, 64, 16)
    eq = cs.Reshape(layout, source_shape, target_shape)
    self.assertEqual(cs.reduce_expression(eq, {}), eq)

  def test_reduce_reshape_of_tiled_layout_with_compatible_shape_is_identity(
      self,
  ):
    layout = RL(mgpu.WGMMA_LAYOUT)
    source_shape, target_shape = (2, 128, 8), (256, 8)
    eq = cs.Reshape(layout, source_shape, target_shape)
    self.assertEqual(cs.reduce_expression(eq, {}), layout)

  def test_relayout_of_non_splat_to_splat_is_unsatisfiable_shortcut(
      self,
  ):
    splat_layout = RL(mgpu.WGSplatFragLayout((128,)))
    v0, v1 = V(0), V(1)
    system = cs.ConstraintSystem(
        assignments={v1: splat_layout},
        constraints=[
            cs.NotOfType(v0, mgpu.WGSplatFragLayout),
            cs.Relayout(v0, v1, 32),
        ],
    )
    self.assertIsInstance(cs.reduce(system), cs.Unsatisfiable)

  def test_saturate_distinct_from_splat_does_not_create_duplicate_constraints(
      self,
  ):
    bw = 32
    v0, v1, v2 = V(0), V(1), V(2)
    system = cs.ConstraintSystem(
        constraints=[
            cs.NotOfType(v0, mgpu.WGSplatFragLayout),
            cs.NotOfType(v1, mgpu.WGSplatFragLayout),
            cs.Relayout(v0, v2, bw),
            cs.Relayout(v1, v2, bw),
        ],
    )

    self.assertEqual(
        cs.saturate_distinct_from_splat(system),
        cs.ConstraintSystem(
            constraints=[
                cs.NotOfType(v0, mgpu.WGSplatFragLayout),
                cs.NotOfType(v1, mgpu.WGSplatFragLayout),
                cs.Relayout(v0, v2, bw),
                cs.Relayout(v1, v2, bw),
                cs.NotOfType(v2, mgpu.WGSplatFragLayout),
            ],
        ),
    )

  def test_saturate_distinct_from_splat_does_not_affect_non_splat(
      self,
  ):
    bw = 32
    v0, v1, v2, v3, v4 = V(0), V(1), V(2), V(3), V(4)
    system = cs.ConstraintSystem(
        constraints=[
            cs.NotOfType(v0, mgpu.WGSplatFragLayout),
            cs.NotOfType(v1, mgpu.WGStridedFragLayout),
            cs.Relayout(v0, v2, bw),
            cs.Relayout(v1, v3, bw),
            cs.Relayout(v4, v0, bw),
        ],
    )

    self.assertEqual(
        cs.saturate_distinct_from_splat(system),
        cs.ConstraintSystem(
            constraints=[
                cs.NotOfType(v0, mgpu.WGSplatFragLayout),
                cs.NotOfType(v1, mgpu.WGStridedFragLayout),
                cs.Relayout(v0, v2, bw),
                cs.Relayout(v1, v3, bw),
                cs.Relayout(v4, v0, bw),
                cs.NotOfType(v2, mgpu.WGSplatFragLayout),
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
    eq_layout = cs.RegisterLayout(layout)
    eq_tiling = cs.SMEMTiling(lc.TileTransform(tiling) if tiling else None)

    reg_to_smem = cs.IsTransferable(eq_layout, eq_tiling, ())
    self.assertEqual(reg_to_smem.holds(), expected)
    smem_to_reg = cs.IsTransferable(eq_tiling, eq_layout, ())
    self.assertEqual(smem_to_reg.holds(), expected)

  def test_transpose_expression(self):
    def transpose(tiling):
      transform = None if tiling is None else lc.TileTransform(tiling)
      return cs.Transpose(cs.SMEMTiling(transform))

    self.assertEqual(
        cs.reduce_expression(transpose(None), {}),
        cs.SMEMTiling(None),
    )
    self.assertEqual(
        cs.reduce_expression(transpose((2, 3)), {}),
        cs.SMEMTiling(lc.TileTransform((3, 2))),
    )

  def test_divides_constraint_are_satisfied_by_empty_tiling(self):
    self.assertTrue(cs.Divides(cs.SMEMTiling(None), (1, 2)).holds())

  def test_divides_constraints_are_satisfied_by_divisor_tiling(self):
    with self.subTest("SMEMTiling"):
      tiling = cs.SMEMTiling(lc.TileTransform((2, 2)))
      self.assertTrue(cs.Divides(tiling, (4, 6)).holds())
    with self.subTest("RegisterLayout"):
      tiling = cs.RegisterLayout(fa.WGMMA_LAYOUT)
      self.assertTrue(cs.Divides(tiling, (0, 64)).holds())
    with self.subTest("TMEMLayout"):
      layout = tcgen05.tmem_default_layout(packing=1)
      tiling = cs.TMEMLayout(layout)
      self.assertTrue(cs.Divides(tiling, (0, 64)).holds())

  def test_divides_constraints_are_not_satisfied_by_non_divisor_tiling(self):
    with self.subTest("SMEMTiling"):
      tiling = cs.SMEMTiling(lc.TileTransform((2, 2)))
      self.assertFalse(cs.Divides(tiling, (4, 3)).holds())
    with self.subTest("RegisterLayout"):
      tiling = cs.RegisterLayout(fa.WGMMA_LAYOUT)
      self.assertFalse(cs.Divides(tiling, (3, 64)).holds())
    with self.subTest("TMEMLayout"):
      layout = tcgen05.tmem_default_layout(packing=1)
      tiling = cs.TMEMLayout(layout)
      self.assertFalse(cs.Divides(tiling, (3, 64)).holds())

  def test_reduce_merges_divides_constraints_on_same_variable(self):
    v0, v1 = cs.Variable(0), cs.Variable(1)
    constraints = [
        cs.Divides(v0, (18, 17)),
        cs.Divides(v0, (3, 19)),
        cs.Divides(v1, (6, 1, 3)),
    ]
    self.assertEqual(
        cs.reduce(cs.ConstraintSystem(constraints=constraints)).constraints,
        [
            cs.Divides(v0, (3, 1)),
            cs.Divides(v1, (6, 1, 3)),
        ],
    )

    # Check that merging constraints with different lenghts yields a constraint
    # whose length matches the one of the shorter tiling_multiple.
    constraints = [
        cs.Divides(v0, (16, 10)),
        cs.Divides(v0, (8,)),
    ]
    self.assertEqual(
        cs.reduce(cs.ConstraintSystem(constraints=constraints)).constraints,
        [
            cs.Divides(v0, (2,)),
        ],
    )

  def test_saturate_divides_constraints_for_equal_vars(self):
    def equals(a, b):
      return cs.Equals(cs.Variable(a), cs.Variable(b))
    def divides(var, dims):
      return cs.Divides(cs.Variable(var), dims)

    # One equality
    s = cs.ConstraintSystem(
        constraints=[
            equals(0, 1),
            divides(0, (1,)),
        ],
    )
    got = cs.saturate_divides_constraints_for_equal_vars(s)
    want = [equals(0, 1), divides(0, (1,)), divides(1, (1,))]
    self.assertEqual(got.constraints, want)

    # Five transitively equal variables and one disconnected one.
    s = cs.ConstraintSystem(
        constraints=[
            equals(0, 1),
            equals(2, 3),
            equals(2, 4),
            equals(1, 4),
            divides(0, (1,)),
            divides(5, (1,)),
        ],
    )
    got = cs.saturate_divides_constraints_for_equal_vars(s)
    want = [
        equals(0, 1),
        equals(2, 3),
        equals(2, 4),
        equals(1, 4),
        divides(0, (1,)),
        divides(1, (1,)),
        divides(2, (1,)),
        divides(3, (1,)),
        divides(4, (1,)),
        divides(5, (1,)),
    ]
    self.assertEqual(got.constraints, want)

  @parameterized.parameters(
    (fa.WGMMA_LAYOUT_UPCAST_4X, fa.WGMMA_LAYOUT_UPCAST_2X, 8),
    (fa.WGMMA_LAYOUT_UPCAST_4X, fa.WGMMA_LAYOUT, 8),
    (fa.WGMMA_LAYOUT_UPCAST_2X, fa.WGMMA_LAYOUT, 32),
  )
  def test_forcing_relayout_on_unsupported_bitwidth_raises(self, src, dst, bitwidth):
    self.assertFalse(cs.Relayout(cs.RegisterLayout(src), cs.RegisterLayout(dst), bitwidth).holds())

  @parameterized.parameters(
      (fa.WGMMA_LAYOUT_UPCAST_4X, fa.WGMMA_LAYOUT_UPCAST_2X, 4),
      (fa.WGMMA_LAYOUT_UPCAST_4X, fa.WGMMA_LAYOUT, 4),
      (fa.WGMMA_LAYOUT_UPCAST_2X, fa.WGMMA_LAYOUT, 16),
  )
  def test_forcing_relayout_on_supported_bitwidth_succeeds(self, src, dst, bitwidth):
    self.assertTrue(cs.Relayout(cs.RegisterLayout(src), cs.RegisterLayout(dst), bitwidth).holds())

  @parameterized.product(
      bitwidth=(16, 32),
      swizzle=(32, 64, 128)
  )
  def test_tiling_is_valid_mma_tiling_holds_for_valid_tiling(self, swizzle, bitwidth):
    swizzle_elems = swizzle * 8 // bitwidth
    layout = cs.SMEMTiling(lc.TileTransform((8, swizzle_elems)))
    self.assertTrue(cs.IsValidMmaTiling(layout, bitwidth).holds())

  def test_tiling_is_valid_mma_tiling_does_not_hold_for_invalid_tiling(self):
    layout = cs.SMEMTiling(lc.TileTransform((8, 8)))
    self.assertFalse(cs.IsValidMmaTiling(layout, 16).holds())

if __name__ == "__main__":
  parameterized.absltest.main(testLoader=jtu.JaxTestLoader())
