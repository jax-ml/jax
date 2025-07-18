# Copyright 2024 The JAX Authors. All Rights Reserved.
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
"""Layout inference tests for the Mosaic GPU MLIR dialect."""

# pylint: disable=g-complex-comprehension

import enum
from typing import ClassVar

from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.interpreters import mlir as mlir_interpreter
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import llvm
from jax._src.lib.mlir.dialects import scf
from jax._src.lib.mlir.dialects import vector
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.mosaic.gpu import equations as eqns
from jax.experimental.mosaic.gpu import layout_inference2
from jax.experimental.mosaic.gpu import layouts

config.parse_flags_with_absl()


def _make_ir_context():
  context = ir.Context()
  context.append_dialect_registry(mlir_interpreter.upstream_dialects)
  context.load_all_available_dialects()
  mgpu.dialect.register_dialect(context)
  return context


def layout_cast(x: ir.Value, layout: mgpu.FragmentedLayout | ir.Attribute) -> ir.Value:
  """Convenience wrapper around `mgpu.dialect.layout_cast`."""
  if isinstance(layout, mgpu.FragmentedLayout):
    layout = layouts.to_layout_attr(layout)
  return mgpu.dialect.layout_cast(x, layout)


class InferenceImplementation(enum.Enum):
  LEGACY = 1
  EQUATIONS = 2


def undefs(*tys: ir.Type) -> list[ir.Value]:
  """Returns a list of `llvm.mlir_undef` values of the given types."""
  return [llvm.mlir_undef(ty) for ty in tys]


class LayoutInferenceTestMetaclass(parameterized.TestGeneratorMetaclass):
  def __new__(mcs, *args, inference_impl=InferenceImplementation.LEGACY):
    cls = super().__new__(mcs, *args)
    cls.INFERENCE_IMPL = inference_impl
    return cls


class LayoutInferenceTest(parameterized.TestCase, metaclass=LayoutInferenceTestMetaclass):
  INFERENCE_IMPL: ClassVar[InferenceImplementation]

  def setUp(self):
    if jax.version._version != jax.lib.__version__:
      raise self.skipTest("Test requires matching jax and jaxlib versions")
    super().setUp()
    self.enter_context(_make_ir_context())
    self.enter_context(ir.Location.unknown())
    self.module = ir.Module.create()

  def infer_layout(self, module):
    if self.INFERENCE_IMPL == InferenceImplementation.LEGACY:
      mgpu.infer_layout(module)
    else:
      mgpu.infer_layout2(module)

  def skip_if_equations(self):
    # TODO(bchetioui): delete once equations work everywhere.
    if self.INFERENCE_IMPL == InferenceImplementation.EQUATIONS:
      self.skipTest("Equations-based layout inference is not supported yet")

  def checkInLayouts(self, op, in_layouts):
    self.assertSequenceEqual(op.attributes["in_layouts"], in_layouts)

  def checkOutLayouts(self, op, out_layouts):
    self.assertSequenceEqual(op.attributes["out_layouts"], out_layouts)

  def test_infer_strided_layout_default(self):
    with ir.InsertionPoint(self.module.body):
      ty = ir.VectorType.get((128,), ir.BF16Type.get())
      x = llvm.UndefOp(ty)

    # Not setting any layouts on the module should default in ops having a
    # strided fragmented layout.
    self.infer_layout(self.module)

    layout = layouts.to_layout_attr(
        mgpu.WGStridedFragLayout.from_shaped_type(ty)
    )

    self.checkInLayouts(x, [])
    self.checkOutLayouts(x, [layout])

  def test_infer_strided_layout_from_shape_cast(self):
    self.skip_if_equations()
    shape = (16, 8)
    elt_type = ir.BF16Type.get()
    src_type = ir.VectorType.get(shape, elt_type)
    dst_type = ir.VectorType.get([*reversed(shape)], elt_type)

    with ir.InsertionPoint(self.module.body):
      [x] = undefs(src_type)
      op = vector.ShapeCastOp(dst_type, x)

    self.infer_layout(self.module)

    in_layout = layouts.to_layout_attr(
        mgpu.WGStridedFragLayout.from_shaped_type(src_type)
    )
    out_layout = layouts.to_layout_attr(
        mgpu.WGStridedFragLayout.from_shaped_type(dst_type)
    )

    self.checkInLayouts(op, [in_layout])
    self.checkOutLayouts(op, [out_layout])

    # Ensure that we can recover the original layout.
    del op.attributes["in_layouts"]
    self.infer_layout(self.module)
    self.checkInLayouts(op, [in_layout])

  def test_infer_splat_layout_for_splat_constants(self):
    shape = (16, 8)
    elt_type = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      ty = ir.VectorType.get(shape, elt_type)
      c0 = ir.FloatAttr.get(elt_type, 0)
      splat = arith.ConstantOp(ty, ir.DenseElementsAttr.get_splat(ty, c0))

    # Not setting any layouts on the module should default in all ops having a
    # splat fragmented layout.
    self.infer_layout(self.module)

    layout = layouts.to_layout_attr(mgpu.WGSplatFragLayout(shape=shape))

    self.assertEmpty(splat.attributes["in_layouts"])
    self.checkOutLayouts(splat, [layout])

  def test_infer_layout_from_consumer_for_non_splat_constant(self):
    shape = (16, 8)
    elt_type = ir.BF16Type.get()
    layout = layouts.to_layout_attr(
        mgpu.WGStridedFragLayout(shape=shape, vec_size=1)
    )

    with ir.InsertionPoint(self.module.body):
      ty = ir.VectorType.get(shape, elt_type)
      attr_list = [
          ir.FloatAttr.get(elt_type, i) for i in range(shape[0] * shape[1])
      ]
      c = arith.ConstantOp(ty, ir.DenseElementsAttr.get(attr_list, ty))
      layout_cast(c, layout)

    self.infer_layout(self.module)

    self.assertEmpty(c.attributes["in_layouts"])
    self.checkOutLayouts(c, [layout])

  def test_infer_splat_layout_for_vector_splat(self):
    shape = (16, 8)
    splat_layout = layouts.to_layout_attr(mgpu.WGSplatFragLayout(shape=shape))
    with ir.InsertionPoint(self.module.body):
      bf16 = ir.BF16Type.get()
      ty = ir.VectorType.get(shape, bf16)
      lhs, rhs = undefs(bf16, ty)
      rhs = layout_cast(rhs, splat_layout)
      splat = vector.SplatOp(rhs.type, lhs)
      add = arith.AddFOp(splat.result, rhs)

    self.infer_layout(self.module)

    self.assertEmpty(splat.attributes["in_layouts"])
    self.checkOutLayouts(splat, [splat_layout])

    self.checkInLayouts(add, [splat_layout, splat_layout])
    self.checkOutLayouts(add, [splat_layout])

  @parameterized.parameters(
      mgpu.WGSplatFragLayout(shape=(32, 4)),
      mgpu.WGStridedFragLayout(shape=(32, 4), vec_size=1),
  )
  def test_pointwise_op_propagates_argument_layouts(self, layout):

    with ir.InsertionPoint(self.module.body):
      ty = ir.VectorType.get(layout.shape, ir.BF16Type.get())
      lhs, rhs = undefs(ty, ty)
      lhs = layout_cast(lhs, layout)
      rhs = layout_cast(rhs, layout)
      add = arith.AddFOp(lhs, rhs)

    self.infer_layout(self.module)

    layout_attr = layouts.to_layout_attr(layout)
    self.checkInLayouts(add, [layout_attr, layout_attr])
    self.checkOutLayouts(add, [layout_attr])

  def test_vector_load_does_not_allow_splat_result(self):
    shape = (32, 4)
    splat_layout_attr = layouts.to_layout_attr(
        mgpu.WGSplatFragLayout(shape=shape)
    )
    strided_layout_attr = layouts.to_layout_attr(
        mgpu.WGStridedFragLayout(shape=shape, vec_size=1)
    )

    with ir.InsertionPoint(self.module.body):
      vec_ty = ir.VectorType.get(shape, ir.BF16Type.get())
      ref_ty = ir.MemRefType.get(shape, ir.BF16Type.get())
      vec, ref = undefs(vec_ty, ref_ty)
      zero = mgpu.utils.c(0, ir.IntegerType.get_signless(32))
      load_op = vector.LoadOp(vec_ty, ref, [zero])
      lhs = layout_cast(vec, splat_layout_attr)
      arith.AddFOp(lhs, load_op.result)

    self.infer_layout(self.module)

    self.checkInLayouts(load_op, [])
    self.checkOutLayouts(load_op, [strided_layout_attr])

  def test_infer_layout_cast_layout(self):
    shape = (128, 64)
    splat_layout = layouts.to_layout_attr(mgpu.WGSplatFragLayout(shape=shape))
    wgmma_layout = layouts.to_layout_attr(mgpu.WGMMA_LAYOUT)

    with ir.InsertionPoint(self.module.body):
      [x] = undefs(ir.VectorType.get(shape, ir.BF16Type.get()))
      x = mgpu.dialect.layout_cast(x, splat_layout)
      add = arith.AddFOp(x, x)
      cast = mgpu.dialect.LayoutCastOp(add.result, wgmma_layout)

    self.infer_layout(self.module)
    self.checkOutLayouts(add, [splat_layout])
    self.checkInLayouts(cast, [wgmma_layout])
    self.checkOutLayouts(cast, [wgmma_layout])

  @parameterized.parameters(
      (0, mgpu.WGMMA_ROW_LAYOUT, None, mgpu.WGMMA_ROW_LAYOUT, mgpu.WGMMA_LAYOUT),
      (1, mgpu.WGMMA_COL_LAYOUT, None, mgpu.WGMMA_COL_LAYOUT, mgpu.WGMMA_LAYOUT),
      (0, None, mgpu.WGMMA_LAYOUT, mgpu.WGMMA_ROW_LAYOUT, mgpu.WGMMA_LAYOUT),
      (1, None, mgpu.WGMMA_LAYOUT, mgpu.WGMMA_COL_LAYOUT, mgpu.WGMMA_LAYOUT),
  )
  def test_infer_broadcast_in_dim_layout(
      self, broadcast_dim, in_cast, out_cast, in_layout, out_layout
  ):
    self.skip_if_equations()
    in_shape = (64,)
    out_shape = (64, 64)

    with ir.InsertionPoint(self.module.body):
      [x] = undefs(ir.VectorType.get(in_shape, ir.F32Type.get()))
      x = layout_cast(x, in_cast) if in_cast is not None else x
      out_type = ir.VectorType.get(out_shape, ir.F32Type.get())
      bcast = mgpu.dialect.BroadcastInDimOp(out_type, x, [broadcast_dim])
      if out_cast is not None:
        layout_cast(bcast.result, out_cast)

    self.infer_layout(self.module)
    self.checkInLayouts(bcast, [layouts.to_layout_attr(in_layout)])
    self.checkOutLayouts(bcast, [layouts.to_layout_attr(out_layout)])

  @parameterized.parameters(
      (1, mgpu.WGMMA_LAYOUT, None, None, mgpu.WGMMA_LAYOUT, mgpu.WGMMA_ROW_LAYOUT),
      (0, mgpu.WGMMA_LAYOUT, None, None, mgpu.WGMMA_LAYOUT, mgpu.WGMMA_COL_LAYOUT),
      (1, None, None, mgpu.WGMMA_ROW_LAYOUT, mgpu.WGMMA_LAYOUT, mgpu.WGMMA_ROW_LAYOUT),
      (0, None, None, mgpu.WGMMA_COL_LAYOUT, mgpu.WGMMA_LAYOUT, mgpu.WGMMA_COL_LAYOUT),
      (1, None, mgpu.WGMMA_ROW_LAYOUT, None, mgpu.WGMMA_LAYOUT, mgpu.WGMMA_ROW_LAYOUT),
      (0, None, mgpu.WGMMA_COL_LAYOUT, None, mgpu.WGMMA_LAYOUT, mgpu.WGMMA_COL_LAYOUT),
      (1, None, mgpu.WGMMA_ROW_LAYOUT, mgpu.WGMMA_ROW_LAYOUT, mgpu.WGMMA_LAYOUT, mgpu.WGMMA_ROW_LAYOUT),
      (0, None, mgpu.WGMMA_COL_LAYOUT, mgpu.WGMMA_COL_LAYOUT, mgpu.WGMMA_LAYOUT, mgpu.WGMMA_COL_LAYOUT),
  )
  def test_infer_multi_reduce_layout(
      self, reduce_dim, in_cast, acc_cast, out_cast, in_layout, out_layout
  ):
    self.skip_if_equations()
    with ir.InsertionPoint(self.module.body):
      in_ty = ir.VectorType.get((64, 64), ir.F32Type.get())
      acc_ty = ir.VectorType.get((64,), ir.F32Type.get())
      x, acc = undefs(in_ty, acc_ty)
      x = layout_cast(x, in_cast) if in_cast is not None else x
      acc = layout_cast(acc, acc_cast) if acc_cast is not None else acc
      kind = vector.CombiningKind.MAXIMUMF
      red = vector.MultiDimReductionOp(kind, x, acc, [reduce_dim])
      if out_cast is not None:
        layout_cast(red.result, out_cast)

    self.infer_layout(self.module)
    in_layout_attr = layouts.to_layout_attr(in_layout)
    out_layout_attr = layouts.to_layout_attr(out_layout)
    self.checkInLayouts(red, [in_layout_attr, out_layout_attr])
    self.checkOutLayouts(red, [out_layout_attr])

  def test_infer_layout_traverses_ops_correctly(self):
    shape = (16, 8)
    elt_type = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      ab_type = ir.VectorType.get(shape, elt_type)
      a, b = undefs(ab_type, ab_type)
      bool_type = ir.IntegerType.get_signless(1)
      cst_true = arith.constant(bool_type, ir.IntegerAttr.get(bool_type, 1))
      if_op = scf.IfOp(cst_true)
      with ir.InsertionPoint(if_op.then_block):
        add = arith.AddFOp(a, b)
        scf.yield_([])

    self.infer_layout(self.module)
    self.assertIn("in_layouts", add.attributes)
    self.assertIn("out_layouts", add.attributes)

  @parameterized.parameters(
      (shape, layout)
      for shape in [(64, 32)]
      for layout in [
          mgpu.WGSplatFragLayout(shape),
          mgpu.WGStridedFragLayout(shape, vec_size=4),
          mgpu.WGMMA_LAYOUT,
      ]
  )
  def test_infer_layout_from_yield_op_in_layouts_for_for_op(
      self, shape, layout
  ):
    if self.INFERENCE_IMPL == InferenceImplementation.LEGACY:
      self.skipTest(
          "The legacy implementation does not return the same results, and "
          "will be removed soon. The new implementation returns better results."
      )

    with ir.InsertionPoint(self.module.body):
      ab_type = ir.VectorType.get(shape, ir.BF16Type.get())
      i32 = ir.IntegerType.get_signless(32)
      lower_bound, upper_bound, step, a, b = undefs(
          i32, i32, i32, ab_type, ab_type
      )
      for_op = scf.ForOp(lower_bound, upper_bound, step, [a, b])
      [loop_a, loop_b] = list(for_op.inner_iter_args)
      with ir.InsertionPoint(for_op.body):
        add = layout_cast(arith.addf(loop_a, loop_b), layout)
        yield_op = scf.YieldOp([add, add])

    self.infer_layout(self.module)

    carry_layouts = [layouts.to_layout_attr(layout)] * 2
    self.checkOutLayouts(yield_op, [])
    self.checkInLayouts(for_op, carry_layouts)
    self.checkOutLayouts(for_op, carry_layouts)

  def test_infer_layout_from_body_op_to_yield_op_to_for_op(self):
    shape = (64, 64)
    with ir.InsertionPoint(self.module.body):
      c_ty = ir.VectorType.get(shape, ir.BF16Type.get())
      ab_type = ir.MemRefType.get(shape, ir.BF16Type.get())
      i32 = ir.IntegerType.get_signless(32)
      lower_bound, upper_bound, step, a, b, c = undefs(
          i32, i32, i32, ab_type, ab_type, c_ty
      )
      for_op = scf.ForOp(lower_bound, upper_bound, step, [a, b, c])
      with ir.InsertionPoint(for_op.body):
        [loop_a, loop_b, loop_c] = list(for_op.inner_iter_args)
        new_loop_c = mgpu.dialect.wgmma(loop_c, loop_a, loop_b)
        yield_op = scf.YieldOp([loop_a, loop_b, new_loop_c])

    self.infer_layout(self.module)

    wgmma_layout = layouts.to_layout_attr(mgpu.WGMMA_LAYOUT)
    self.checkInLayouts(yield_op, [wgmma_layout])
    self.checkOutLayouts(yield_op, [])
    self.checkInLayouts(for_op, [wgmma_layout])
    self.checkOutLayouts(for_op, [wgmma_layout])

  @parameterized.parameters(
      ((), None, (), None),
      ((64, 32), mgpu.WGMMA_LAYOUT, (), None),
      ((), None, (64, 32), mgpu.WGMMA_LAYOUT),
      ((64,), mgpu.WGMMA_ROW_LAYOUT, (64, 32), mgpu.WGMMA_LAYOUT),
  )
  def test_infer_while_op_layouts(
      self, init_shape, init_layout, result_shape, result_layout
  ):
    f32 = ir.F32Type.get()
    in_type = ir.VectorType.get(init_shape, f32) if init_shape else f32
    out_type = ir.VectorType.get(result_shape, f32) if result_shape else f32
    with ir.InsertionPoint(self.module.body):
      i1 = ir.IntegerType.get_signless(1)
      condition, init, result = undefs(i1, in_type, out_type)
      init = layout_cast(init, init_layout) if init_layout else init
      result = layout_cast(result, result_layout) if result_layout else result
      while_op = scf.WhileOp([out_type], [init])
      before_block = while_op.before.blocks.append(init.type)
      with ir.InsertionPoint(before_block):
        scf.condition(condition, [result])
      after_block = while_op.after.blocks.append(out_type)
      with ir.InsertionPoint(after_block):
        scf.yield_([init])

    self.infer_layout(self.module)

    if init_layout is not None or result_layout is not None:
      init_layouts = [layouts.to_layout_attr(init_layout)] if init_layout else []
      result_layouts = [layouts.to_layout_attr(result_layout)] if result_layout else []
      self.checkInLayouts(while_op, init_layouts)
      self.checkOutLayouts(while_op, result_layouts)

  def test_infer_layout_has_no_layout_for_non_vector_types(self):
    shape = (32, 4)
    elt_ty = ir.BF16Type.get()
    with ir.InsertionPoint(self.module.body):
      ref_ty = ir.MemRefType.get(shape, elt_ty)
      array_ty = ir.VectorType.get(shape, elt_ty)
      ref, array = undefs(ref_ty, array_ty)
      zero_index = arith.constant(ir.IndexType.get(), 0)
      vector_store = vector.store(array, ref, [zero_index, zero_index])

    self.infer_layout(self.module)

    self.assertIn("in_layouts", vector_store.attributes)
    self.assertIn("out_layouts", vector_store.attributes)

    # The vector store should have a layout for the input array, but not for the
    # memref.
    self.assertLen(vector_store.attributes["in_layouts"], 1)
    self.assertEmpty(vector_store.attributes["out_layouts"])

  @parameterized.parameters(
      mgpu.WGStridedFragLayout((64, 16), vec_size=1),
      mgpu.WGMMA_LAYOUT,
  )
  def test_infer_layout_picks_non_splat_layout_over_splat_layout(self, layout):
    shape = (64, 16)
    splat_layout = layouts.to_layout_attr(mgpu.WGSplatFragLayout(shape))
    non_splat_layout = layouts.to_layout_attr(layout)
    with ir.InsertionPoint(self.module.body):
      elt_type = ir.BF16Type.get()
      ty = ir.VectorType.get(shape, elt_type)
      lhs, rhs = undefs(ty, ty)
      lhs = layout_cast(lhs, non_splat_layout)
      rhs = layout_cast(rhs, splat_layout)
      add = arith.AddFOp(lhs, rhs)

    self.infer_layout(self.module)

    self.checkInLayouts(add, [non_splat_layout, non_splat_layout])
    self.checkOutLayouts(add, [non_splat_layout])

  def test_infer_layout_preserves_splat_layouts_in_producers(self):
    shape = (32, 4)
    splat_layout = layouts.to_layout_attr(mgpu.WGSplatFragLayout(shape))
    strided_layout = layouts.to_layout_attr(
        mgpu.WGStridedFragLayout(shape, vec_size=1)
    )
    with ir.InsertionPoint(self.module.body):
      elt_type = ir.BF16Type.get()
      ty = ir.VectorType.get(shape, elt_type)
      lhs, rhs = undefs(ty, ty)
      lhs = layout_cast(lhs, splat_layout)
      rhs = layout_cast(rhs, splat_layout)
      add0 = arith.AddFOp(lhs, rhs)
      cast = layout_cast(add0, strided_layout)
      add1 = arith.AddFOp(cast, cast)

    self.infer_layout(self.module)

    self.checkInLayouts(add0, [splat_layout, splat_layout])
    self.checkOutLayouts(add0, [splat_layout])
    self.checkInLayouts(add1, [strided_layout, strided_layout])
    self.checkOutLayouts(add1, [strided_layout])

  def test_optimization_barrier_op_propagates_user_layouts(self):
    wgmma_layout = layouts.to_layout_attr(mgpu.WGMMA_LAYOUT)

    with ir.InsertionPoint(self.module.body):
      ty = ir.VectorType.get((32, 4), ir.BF16Type.get())
      lhs, rhs = undefs(ty, ty)
      optimization_barrier = mgpu.dialect.OptimizationBarrierOp([lhs, rhs])
      lhs, rhs = optimization_barrier.results
      layout_cast(arith.addf(lhs, rhs), wgmma_layout)

    self.infer_layout(self.module)

    self.checkInLayouts(optimization_barrier, [wgmma_layout, wgmma_layout])
    self.checkOutLayouts(optimization_barrier, [wgmma_layout, wgmma_layout])

  def test_optimization_barrier_op_propagates_producer_layouts(self):
    shape = (32, 4)
    splat_layout = layouts.to_layout_attr(mgpu.WGSplatFragLayout(shape))
    with ir.InsertionPoint(self.module.body):
      ty = ir.VectorType.get(shape, ir.BF16Type.get())
      lhs, rhs = undefs(ty, ty)
      lhs = layout_cast(lhs, splat_layout)
      rhs = layout_cast(rhs, splat_layout)
      add = arith.addf(lhs, rhs)
      optimization_barrier = mgpu.dialect.OptimizationBarrierOp([add])

    self.infer_layout(self.module)

    self.checkInLayouts(optimization_barrier, [splat_layout])
    self.checkOutLayouts(optimization_barrier, [splat_layout])


V = eqns.Variable
H = layout_inference2.Hint
E = eqns.Equation
C = eqns.Constant


def _undef_equation_system(
    op: llvm.UndefOp,
) -> tuple[eqns.EquationSystem, layout_inference2.OperandOrResultsForVariable]:
  # This rule is only called if the single output of the undef op is a vector,
  # so we can just return a trivial mapping.
  result = layout_inference2.OperandOrResult(
      op, layout_inference2.VariableType.RESULT, 0
  )
  return eqns.EquationSystem(), {eqns.Variable(result): [result]}


class LayoutInferenceTestEquations(LayoutInferenceTest, inference_impl=InferenceImplementation.EQUATIONS):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    mgpu.layout_inference2._add_equation_system_derivation_rule(llvm.UndefOp)(
        _undef_equation_system
    )

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    del mgpu.layout_inference2._equation_system_derivation_rules[
        llvm.UndefOp.OPERATION_NAME
    ]

  def test_hint_and_constraint_extraction_works_correctly(self):
    layout = mgpu.WGMMA_ROW_LAYOUT
    with ir.InsertionPoint(self.module.body):
      x = llvm.UndefOp(ir.VectorType.get((64,), ir.BF16Type.get()))
      lc = layout_cast(x, layouts.to_layout_attr(layout)).owner.opview

    x_system, x_mapping = _undef_equation_system(x)
    lc_system, lc_mapping = layout_inference2._layout_cast_equation_system(lc)
    assignments = x_system.assignments | lc_system.assignments
    hints, [constraint] = layout_inference2.derive_hints_and_constraints(
        x_mapping | lc_mapping
    )
    [hint_cst] = layout_inference2.reduce_hints(hints, assignments)

    [x_variable] = x_mapping.keys()
    [lc_variable] = lc_mapping.keys()
    self.assertEqual(hint_cst.variable, x_variable)
    self.assertEqual(hint_cst.expression, C(layout))
    self.assertEqual(constraint, eqns.Relayout(x_variable, lc_variable))

  def test_unambiguous_hints_are_used_to_assign_variables_correctly(self):
    v0 = V(0)
    assignments = layout_inference2.find_assignments_for(
        {v0},
        eqns.EquationSystem(),
        # Voluntarily use conflicting hints to check that we use one of them
        # deterministically. This may require updating if we decide to change
        # the traversal order in the future.
        [H(v0, C(mgpu.WGMMA_ROW_LAYOUT)), H(v0, C(mgpu.WGMMA_COL_LAYOUT))],
    )
    self.assertEqual(assignments, {v0: C(mgpu.WGMMA_COL_LAYOUT)})

  def test_cannot_find_assignments_for_unsatisfiable_equation_system(self):
    with ir.InsertionPoint(self.module.body):
      x = llvm.UndefOp(ir.VectorType.get((64,), ir.BF16Type.get()))

    [key] = layout_inference2.operands_and_results(x)
    variable = eqns.Variable(key)
    assignments = layout_inference2.find_assignments_for(
        {variable},
        eqns.EquationSystem(
            equations=[
                E(variable, C(mgpu.WGMMA_ROW_LAYOUT)),
                E(variable, C(mgpu.WGMMA_COL_LAYOUT)),
            ]
        ),
        hints=[],
    )
    self.assertIsInstance(assignments, eqns.Unsatisfiable)

  def test_hint_that_would_make_system_unsatisfiable_is_not_used_in_solution(self):
    with ir.InsertionPoint(self.module.body):
      ty = ir.VectorType.get((32, 4), ir.BF16Type.get())
      op0, op1 = [llvm.mlir_undef(ty).owner.opview for _ in range(2)]
    [kv0] = layout_inference2.operands_and_results(op0)
    [kv1] = layout_inference2.operands_and_results(op1)
    v0, v1 = eqns.Variable(kv0), eqns.Variable(kv1)
    splat_layout = C(mgpu.WGSplatFragLayout((3, 128)))
    assignments = layout_inference2.find_assignments_for(
        {v0},
        eqns.EquationSystem(
            equations=[
                E(
                    v0,
                    eqns.MostReplicated(
                        [v1, C(mgpu.WGStridedFragLayout((3, 128), vec_size=1))]
                    ),
                )
            ]
        ),
        # The first hint would make the system unsatisfiable, but the second
        # hint should be used to find a solution.
        hints=[H(v1, C(mgpu.WGMMA_LAYOUT)), H(v1, splat_layout)],
    )
    self.assertEqual(assignments, {v0: splat_layout})

  def test_hint_can_be_chosen_when_constant_exists_in_least_replicated_expression(self):
    v0, v1 = V(0), V(1)
    layout = C(mgpu.WGMMA_LAYOUT)
    assignment = layout_inference2.extract_variable_assignment_from_hint(
        H(v0, eqns.LeastReplicated([layout, v1])),
    )
    self.assertEqual(assignment, (v0, layout))

  def test_hint_cannot_be_chosen_when_constant_exists_in_most_replicated_expression(self):
    v0, v1 = V(0), V(1)
    layout = C(mgpu.WGSplatFragLayout((1, 128)))
    assignment = layout_inference2.extract_variable_assignment_from_hint(
        H(v0, eqns.MostReplicated([layout, v1])),
    )
    self.assertEqual(assignment, (v0, layout))

  def test_hint_is_still_extracted_when_underlying_expression_is_unsatisfiable(self):
    v0, v1 = V(0), V(1)
    layout0 = C(mgpu.WGSplatFragLayout((1, 128)))
    layout1 = C(mgpu.WGStridedFragLayout((1, 256), vec_size=2))
    hint_expr = eqns.LeastReplicated(
        [layout0, eqns.MostReplicated([layout1, v1])]
    )
    self.assertIsInstance(
        eqns.reduce_expression(hint_expr, {v1: layout1}), eqns.Unsatisfiable
    )
    _, expr = layout_inference2.extract_variable_assignment_from_hint(
        H(v0, hint_expr))
    self.assertIsNotNone(expr)

  def test_least_replicated_hint_is_still_resolved_when_all_known_choices_are_replicated(
      self,
  ):
    v0, v1 = V(0), V(1)
    layout0 = C(mgpu.WGSplatFragLayout((1, 128)))
    layout1 = C(mgpu.WGSplatFragLayout((1, 129)))
    assignment = layout_inference2.extract_variable_assignment_from_hint(
        H(v0, eqns.LeastReplicated([v1, layout0, layout1])),
    )
    self.assertIsNotNone(assignment)

  @parameterized.parameters("registers", "shared")
  def test_infer_wgmma_layout_correctly(self, lhs_memory_space):
    f32 = ir.F32Type.get()
    shape = (64, 64)

    with ir.InsertionPoint(self.module.body):
      vec_ty = ir.VectorType.get(shape, f32)
      ref_ty = ir.MemRefType.get(shape, f32)
      lhs_ty = ref_ty if lhs_memory_space == "shared" else vec_ty
      acc, lhs, rhs = undefs(vec_ty, lhs_ty, ref_ty)
      wgmma_op = mgpu.dialect.WGMMAOp(acc, lhs, rhs)

    self.infer_layout(self.module)

    wgmma_layout = layouts.to_layout_attr(mgpu.WGMMA_LAYOUT)
    in_layouts = [wgmma_layout]
    out_layouts = [wgmma_layout]
    if lhs_memory_space == "registers":
      in_layouts.append(wgmma_layout)

    self.checkInLayouts(wgmma_op, in_layouts)
    self.checkOutLayouts(wgmma_op, out_layouts)

  def test_layout_cast_of_vector_load_to_splat_raises(self):
    shape = (32, 4)
    splat_layout = mgpu.WGSplatFragLayout(shape=shape)
    with ir.InsertionPoint(self.module.body):
      vec_ty = ir.VectorType.get(shape, ir.BF16Type.get())
      ref_ty = ir.MemRefType.get(shape, ir.BF16Type.get())
      [ref] = undefs(ref_ty)
      zero = mgpu.utils.c(0, ir.IntegerType.get_signless(32))
      loaded = vector.load(vec_ty, ref, [zero])
      layout_cast(loaded, splat_layout)

    with self.assertRaisesRegex(
        ValueError, "user-provided layout casts are unsatisfiable"
    ):
      self.infer_layout(self.module)

  def test_layout_cast_of_non_splat_constant_to_splat_raises(self):
    shape = (128,)
    splat_layout = mgpu.WGSplatFragLayout(shape=shape)
    with ir.InsertionPoint(self.module.body):
      bf16 = ir.BF16Type.get()
      ty = ir.VectorType.get(shape, bf16)
      values = [ir.FloatAttr.get(bf16, float(i)) for i in range(shape[0])]
      constant = arith.constant(ty, ir.DenseElementsAttr.get(values, ty))
      layout_cast(constant, splat_layout)

    with self.assertRaisesRegex(
        ValueError, "user-provided layout casts are unsatisfiable"
    ):
      self.infer_layout(self.module)


if __name__ == "__main__":
  parameterized.absltest.main(testLoader=jtu.JaxTestLoader())
