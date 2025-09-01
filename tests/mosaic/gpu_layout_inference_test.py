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

from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.interpreters import mlir as mlir_interpreter
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import llvm
from jax._src.lib.mlir.dialects import math
from jax._src.lib.mlir.dialects import scf
from jax._src.lib.mlir.dialects import vector
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.mosaic.gpu import equations as eqns
from jax.experimental.mosaic.gpu import fragmented_array as fa
from jax.experimental.mosaic.gpu import layout_inference
from jax.experimental.mosaic.gpu import layouts
from jax.experimental.mosaic.gpu import tcgen05

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


def undefs(*tys: ir.Type) -> list[ir.Value]:
  """Returns a list of `llvm.mlir_undef` values of the given types."""
  return [llvm.mlir_undef(ty) for ty in tys]


V = eqns.Variable
H = layout_inference.Hint
E = eqns.Equation
RL = eqns.RegisterLayout


def _undef_equation_system(
    ctx: layout_inference.DerivationContext,
    op: llvm.UndefOp,
) -> tuple[
    eqns.EquationSystem,
    layout_inference.OperandOrResultsForVariable,
    list[layout_inference.Hint],
]:
  del ctx
  # This rule is only called if the single output of the undef op is a vector or
  # TMEM reference, so we can just return a trivial mapping.
  result = layout_inference.OperandOrResult(
      op, layout_inference.VariableType.RESULT, 0
  )
  return eqns.EquationSystem(), {eqns.Variable(result): [result]}, []


class LayoutInferenceTest(parameterized.TestCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    layout_inference._add_equation_system_derivation_rule(llvm.UndefOp)(
        _undef_equation_system
    )

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    del layout_inference._equation_system_derivation_rules[
        llvm.UndefOp.OPERATION_NAME
    ]

  def setUp(self):
    if jax.version._version != jax.lib.__version__:
      raise self.skipTest("Test requires matching jax and jaxlib versions")
    super().setUp()
    self.enter_context(_make_ir_context())
    self.enter_context(ir.Location.unknown())
    self.module = ir.Module.create()

  def checkInLayouts(self, op, in_layouts):
    in_layouts = [
        layouts.to_layout_attr(l) if isinstance(l, mgpu.FragmentedLayout) else l
        for l in in_layouts
    ]
    self.assertSequenceEqual(op.attributes["in_layouts"], in_layouts)

  def checkOutLayouts(self, op, out_layouts):
    out_layouts = [
        layouts.to_layout_attr(l) if isinstance(l, mgpu.FragmentedLayout) else l
        for l in out_layouts
    ]
    self.assertSequenceEqual(op.attributes["out_layouts"], out_layouts)

  def checkInTmemLayouts(self, op, in_layouts):
    in_layouts = [
        layouts.to_layout_attr(l) if isinstance(l, tcgen05.TMEMLayout) else l
        for l in in_layouts
    ]
    self.assertSequenceEqual(op.attributes["in_tmem_layouts"], in_layouts)

  def checkOutTmemLayouts(self, op, out_layouts):
    out_layouts = [
        layouts.to_layout_attr(l) if isinstance(l, tcgen05.TMEMLayout) else l
        for l in out_layouts
    ]
    self.assertSequenceEqual(op.attributes["out_tmem_layouts"], out_layouts)

  def test_infer_strided_layout_default(self):
    with ir.InsertionPoint(self.module.body):
      ty = ir.VectorType.get((128,), ir.BF16Type.get())
      x = llvm.UndefOp(ty)

    # Not setting any layouts on the module should default in ops having a
    # strided fragmented layout.
    mgpu.infer_layout(self.module)

    strided_layout = mgpu.WGStridedFragLayout.from_shaped_type(ty)
    assert strided_layout is not None
    layout = layouts.to_layout_attr(strided_layout)

    self.assertNotIn("in_layouts", x.attributes)
    self.checkOutLayouts(x, [layout])

  @parameterized.parameters(
      (mgpu.WGMMA_LAYOUT, None), (None, mgpu.WGMMA_LAYOUT)
  )
  def test_infer_layout_bidirectionally_through_shape_cast(
      self, in_layout, out_layout
  ):
    assert in_layout is not None or out_layout is not None
    elt_type = ir.BF16Type.get()
    src_type = ir.VectorType.get((2, 128, 8), elt_type)
    dst_type = ir.VectorType.get((256, 8), elt_type)

    with ir.InsertionPoint(self.module.body):
      [x] = undefs(src_type)
      if in_layout is not None:
        x = layout_cast(x, in_layout)
      op = vector.ShapeCastOp(dst_type, x)
      if out_layout is not None:
        layout_cast(op.result, out_layout)

    mgpu.infer_layout(self.module)

    expected_layout = in_layout or out_layout
    self.checkInLayouts(op, [expected_layout])
    self.checkOutLayouts(op, [expected_layout])

  def test_infer_splat_layout_for_splat_constants(self):
    shape = (16, 8)
    elt_type = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      ty = ir.VectorType.get(shape, elt_type)
      c0 = ir.FloatAttr.get(elt_type, 0)
      splat = arith.ConstantOp(ty, ir.DenseElementsAttr.get_splat(ty, c0))

    # Not setting any layouts on the module should default in all ops having a
    # splat fragmented layout.
    mgpu.infer_layout(self.module)

    layout = layouts.to_layout_attr(mgpu.WGSplatFragLayout(shape=shape))

    self.assertNotIn("in_layouts", splat.attributes)
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

    mgpu.infer_layout(self.module)

    self.assertNotIn("in_layouts", c.attributes)
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

    mgpu.infer_layout(self.module)

    self.assertNotIn("in_layouts", splat.attributes)
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

    mgpu.infer_layout(self.module)

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

    mgpu.infer_layout(self.module)

    self.assertNotIn("in_layouts", load_op.attributes)
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

    mgpu.infer_layout(self.module)
    self.checkOutLayouts(add, [splat_layout])
    self.checkInLayouts(cast, [wgmma_layout])
    self.checkOutLayouts(cast, [wgmma_layout])

  @parameterized.parameters(
      (0, mgpu.WGMMA_ROW_LAYOUT, None),
      (1, mgpu.WGMMA_COL_LAYOUT, None),
      (0, None, mgpu.WGMMA_LAYOUT),
      (1, None, mgpu.WGMMA_LAYOUT),
      (0, mgpu.TCGEN05_ROW_LAYOUT, None),
      (0, None, mgpu.TCGEN05_LAYOUT),
      (1, None, mgpu.TCGEN05_LAYOUT),
  )
  def test_infer_broadcast_in_dim_layout(self, broadcast_dim, in_cast, out_cast):
    in_shape = (128,)
    out_shape = (128, 128)

    with ir.InsertionPoint(self.module.body):
      [x] = undefs(ir.VectorType.get(in_shape, ir.F32Type.get()))
      x = layout_cast(x, in_cast) if in_cast is not None else x
      out_type = ir.VectorType.get(out_shape, ir.F32Type.get())
      bcast = mgpu.dialect.BroadcastInDimOp(out_type, x, [broadcast_dim])
      if out_cast is not None:
        layout_cast(bcast.result, out_cast)

    # The tests always expect WGMMA or TCGEN05 as the out layout.
    if out_cast == mgpu.TCGEN05_LAYOUT or in_cast == mgpu.TCGEN05_ROW_LAYOUT:
      out_layout = mgpu.TCGEN05_LAYOUT
    else:
      out_layout = mgpu.WGMMA_LAYOUT

    in_layout = out_layout.reduce((1 - broadcast_dim,))

    mgpu.infer_layout(self.module)
    self.checkInLayouts(bcast, [layouts.to_layout_attr(in_layout)])
    self.checkOutLayouts(bcast, [layouts.to_layout_attr(out_layout)])

  @parameterized.parameters(
      (1, mgpu.WGMMA_LAYOUT, None, None),
      (0, mgpu.WGMMA_LAYOUT, None, None),
      (1, None, None, mgpu.WGMMA_ROW_LAYOUT),
      (0, None, None, mgpu.WGMMA_COL_LAYOUT),
      (1, None, mgpu.WGMMA_ROW_LAYOUT, None),
      (0, None, mgpu.WGMMA_COL_LAYOUT, None),
      (1, None, mgpu.WGMMA_ROW_LAYOUT, mgpu.WGMMA_ROW_LAYOUT),
      (0, None, mgpu.WGMMA_COL_LAYOUT, mgpu.WGMMA_COL_LAYOUT),
      (1, mgpu.TCGEN05_LAYOUT, None, None),
      (1, None, None, mgpu.TCGEN05_ROW_LAYOUT),
      (1, None, mgpu.TCGEN05_ROW_LAYOUT, None),
      (1, None, mgpu.TCGEN05_ROW_LAYOUT, mgpu.TCGEN05_ROW_LAYOUT)
  )
  def test_infer_multi_reduce_layout(
      self, reduce_dim, in_cast, acc_cast, out_cast
  ):
    with ir.InsertionPoint(self.module.body):
      in_ty = ir.VectorType.get((128, 128), ir.F32Type.get())
      acc_ty = ir.VectorType.get((128,), ir.F32Type.get())
      x, acc = undefs(in_ty, acc_ty)
      x = layout_cast(x, in_cast) if in_cast is not None else x
      acc = layout_cast(acc, acc_cast) if acc_cast is not None else acc
      kind = vector.CombiningKind.MAXIMUMF
      red = vector.MultiDimReductionOp(kind, x, acc, [reduce_dim])
      if out_cast is not None:
        layout_cast(red.result, out_cast)

    mgpu.infer_layout(self.module)
    targets_tcgen05 = any(layout in {mgpu.TCGEN05_LAYOUT, mgpu.TCGEN05_ROW_LAYOUT} for layout in [in_cast, acc_cast, out_cast])
    # The tests always expect WGMMA or TCGEN05 as the source layout.
    in_layout = mgpu.TCGEN05_LAYOUT if targets_tcgen05 else mgpu.WGMMA_LAYOUT
    out_layout = in_layout.reduce((reduce_dim,))
    self.checkInLayouts(red, [in_layout, out_layout])
    self.checkOutLayouts(red, [out_layout])

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

    mgpu.infer_layout(self.module)
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

    mgpu.infer_layout(self.module)

    carry_layouts = [layouts.to_layout_attr(layout)] * 2
    self.assertNotIn("out_layouts", yield_op.attributes)
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

    mgpu.infer_layout(self.module)

    wgmma_layout = layouts.to_layout_attr(mgpu.WGMMA_LAYOUT)
    self.checkInLayouts(yield_op, [wgmma_layout])
    self.assertNotIn("out_layouts", yield_op.attributes)
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

    mgpu.infer_layout(self.module)

    if init_layout:
      self.checkInLayouts(while_op, [layouts.to_layout_attr(init_layout)])
    if result_layout:
      self.checkOutLayouts(while_op, [layouts.to_layout_attr(result_layout)])

  @parameterized.parameters(
      (None, mgpu.WGMMA_ROW_LAYOUT, mgpu.WGMMA_LAYOUT),
      (mgpu.WGMMA_LAYOUT, mgpu.WGMMA_COL_LAYOUT, None),
  )
  def test_infer_index_switch_op_layouts(
      self,
      out0_layout: mgpu.FragmentedLayout | None,
      out3_layout: mgpu.FragmentedLayout,
      out4_layout: mgpu.FragmentedLayout | None
  ):
    out_layouts = [out0_layout or out4_layout, out3_layout]
    assert None not in out_layouts
    f32 = ir.F32Type.get()
    out_type = ir.VectorType.get((128, 128), f32)
    with ir.InsertionPoint(self.module.body):
      i1 = ir.IntegerType.get_signless(1)
      [condition] = undefs(i1)
      index_switch = scf.IndexSwitchOp(
          [out_type, out_type, f32],
          condition,
          ir.DenseI64ArrayAttr.get(range(3)),
          num_caseRegions=2,
      )
      with ir.InsertionPoint(index_switch.caseRegions[0].blocks.append()):
        out0, out1, dummy0 = undefs(out_type, out_type, f32)
        if out0_layout is not None:
          out0 = layout_cast(out0, out0_layout)
        yield0 = scf.YieldOp([out0, out1, dummy0])
      with ir.InsertionPoint(index_switch.caseRegions[1].blocks.append()):
        out2, out3, dummy1 = undefs(out_type, out_type, f32)
        if out3_layout is not None:
          out3 = layout_cast(out3, out3_layout)
        yield1 = scf.YieldOp([out2, out3, dummy1])
      with ir.InsertionPoint(index_switch.defaultRegion.blocks.append()):
        out4, out5, dummy2 = undefs(out_type, out_type, f32)
        if out4_layout is not None:
          out4 = layout_cast(out4, out4_layout)
        yield2 = scf.YieldOp([out4, out5, dummy2])

    mgpu.infer_layout(self.module)

    self.assertNotIn("in_layouts", index_switch.attributes)
    self.assertNotIn("out_layouts", yield0.attributes)
    self.assertNotIn("out_layouts", yield1.attributes)
    self.assertNotIn("out_layouts", yield2.attributes)

    self.checkOutLayouts(index_switch, out_layouts)
    self.checkInLayouts(yield0, out_layouts)
    self.checkInLayouts(yield1, out_layouts)
    self.checkInLayouts(yield2, out_layouts)

  def test_infer_layout_has_no_layout_for_non_vector_types(self):
    shape = (32, 4)
    elt_ty = ir.BF16Type.get()
    with ir.InsertionPoint(self.module.body):
      ref_ty = ir.MemRefType.get(shape, elt_ty)
      array_ty = ir.VectorType.get(shape, elt_ty)
      ref, array = undefs(ref_ty, array_ty)
      zero_index = arith.constant(ir.IndexType.get(), 0)
      vector_store = vector.store(array, ref, [zero_index, zero_index])

    mgpu.infer_layout(self.module)

    # The vector store should have a layout for the input array, but not for the
    # memref.
    self.assertIn("in_layouts", vector_store.attributes)
    self.assertLen(vector_store.attributes["in_layouts"], 1)
    self.assertNotIn("out_layouts", vector_store.attributes)

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

    mgpu.infer_layout(self.module)

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

    mgpu.infer_layout(self.module)

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

    mgpu.infer_layout(self.module)

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

    mgpu.infer_layout(self.module)

    self.checkInLayouts(optimization_barrier, [splat_layout])
    self.checkOutLayouts(optimization_barrier, [splat_layout])

  def test_custom_primitive_op_retains_layouts(self):
    with ir.InsertionPoint(self.module.body):
      wgmma_layout = layouts.to_layout_attr(mgpu.WGMMA_LAYOUT)
      wgmma_row_layout = layouts.to_layout_attr(mgpu.WGMMA_ROW_LAYOUT)
      op = mgpu.dialect.custom_primitive(
          result=[],
          operands_=[],
          in_layouts=[wgmma_layout],
          in_transforms=[],
          out_layouts=[wgmma_row_layout],
      )

    mgpu.infer_layout(self.module)
    self.checkInLayouts(op, [wgmma_layout])
    self.checkOutLayouts(op, [wgmma_row_layout])

  def test_hint_and_constraint_extraction_works_correctly(self):
    layout = mgpu.WGMMA_ROW_LAYOUT
    with ir.InsertionPoint(self.module.body):
      x = llvm.UndefOp(ir.VectorType.get((64,), ir.BF16Type.get()))
      lc = layout_cast(x, layouts.to_layout_attr(layout)).owner.opview

    ctx = layout_inference.DerivationContext()
    x_system, x_mapping, _ = _undef_equation_system(ctx, x)
    lc_system, lc_mapping, _ = layout_inference._layout_cast_equation_system(
        ctx, lc
    )
    assignments = x_system.assignments | lc_system.assignments
    hints, [constraint] = layout_inference.derive_hints_and_constraints(
        x_mapping | lc_mapping
    )
    [hint_cst] = layout_inference.reduce_hints(hints, assignments)

    [x_variable] = x_mapping.keys()
    [lc_variable] = lc_mapping.keys()
    self.assertEqual(hint_cst.variable, x_variable)
    self.assertEqual(hint_cst.expression, RL(layout))
    self.assertEqual(constraint, eqns.Relayout(x_variable, lc_variable))

  def test_unambiguous_hints_are_used_to_assign_variables_correctly(self):
    v0 = V(0)
    assignments = layout_inference.find_assignments_for(
        {v0},
        eqns.EquationSystem(),
        # Voluntarily use conflicting hints to check that we use one of them
        # deterministically. This may require updating if we decide to change
        # the traversal order in the future.
        [H(v0, RL(mgpu.WGMMA_ROW_LAYOUT)), H(v0, RL(mgpu.WGMMA_COL_LAYOUT))],
    )
    self.assertEqual(assignments, {v0: RL(mgpu.WGMMA_ROW_LAYOUT)})

  def test_cannot_find_assignments_for_unsatisfiable_equation_system(self):
    with ir.InsertionPoint(self.module.body):
      x = llvm.UndefOp(ir.VectorType.get((64,), ir.BF16Type.get()))

    [key] = layout_inference.operands_and_results(x)
    variable = eqns.Variable(key)
    assignments = layout_inference.find_assignments_for(
        {variable},
        eqns.EquationSystem(
            equations=[
                E(variable, RL(mgpu.WGMMA_ROW_LAYOUT)),
                E(variable, RL(mgpu.WGMMA_COL_LAYOUT)),
            ]
        ),
        hints=[],
    )
    self.assertIsInstance(assignments, eqns.Unsatisfiable)

  def test_hint_that_would_make_system_unsatisfiable_is_not_used_in_solution(self):
    with ir.InsertionPoint(self.module.body):
      ty = ir.VectorType.get((32, 4), ir.BF16Type.get())
      op0, op1 = [llvm.mlir_undef(ty).owner.opview for _ in range(2)]
    [kv0] = layout_inference.operands_and_results(op0)
    [kv1] = layout_inference.operands_and_results(op1)
    v0, v1 = eqns.Variable(kv0), eqns.Variable(kv1)
    splat_layout = RL(mgpu.WGSplatFragLayout((3, 128)))
    assignments = layout_inference.find_assignments_for(
        {v0},
        eqns.EquationSystem(
            equations=[
                E(
                    v0,
                    eqns.MostReplicated(
                        [v1, RL(mgpu.WGStridedFragLayout((3, 128), vec_size=1))]
                    ),
                )
            ]
        ),
        # The first hint would make the system unsatisfiable, but the second
        # hint should be used to find a solution.
        hints=[H(v1, RL(mgpu.WGMMA_LAYOUT)), H(v1, splat_layout)],
    )
    self.assertEqual(assignments, {v0: splat_layout})

  def test_hint_can_be_chosen_when_constant_exists_in_least_replicated_expression(self):
    v0, v1 = V(0), V(1)
    layout = RL(mgpu.WGMMA_LAYOUT)
    assignment = layout_inference.extract_variable_assignment_from_hint(
        H(v0, eqns.LeastReplicated([layout, v1])),
    )
    self.assertEqual(assignment, (v0, layout))

  def test_hint_cannot_be_chosen_when_constant_exists_in_most_replicated_expression(self):
    v0, v1 = V(0), V(1)
    layout = RL(mgpu.WGSplatFragLayout((1, 128)))
    assignment = layout_inference.extract_variable_assignment_from_hint(
        H(v0, eqns.MostReplicated([layout, v1])),
    )
    self.assertEqual(assignment, (v0, layout))

  def test_hint_is_still_extracted_when_underlying_expression_is_unsatisfiable(self):
    v0, v1 = V(0), V(1)
    layout0 = RL(mgpu.WGSplatFragLayout((1, 128)))
    layout1 = RL(mgpu.WGStridedFragLayout((1, 256), vec_size=2))
    hint_expr = eqns.LeastReplicated(
        [layout0, eqns.MostReplicated([layout1, v1])]
    )
    self.assertIsInstance(
        eqns.reduce_expression(hint_expr, {v1: layout1}), eqns.Unsatisfiable
    )
    _, expr = layout_inference.extract_variable_assignment_from_hint(
        H(v0, hint_expr))
    self.assertIsNotNone(expr)

  def test_least_replicated_hint_is_still_resolved_when_all_known_choices_are_replicated(
      self,
  ):
    v0, v1 = V(0), V(1)
    layout0 = RL(mgpu.WGSplatFragLayout((1, 128)))
    layout1 = RL(mgpu.WGSplatFragLayout((1, 129)))
    assignment = layout_inference.extract_variable_assignment_from_hint(
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

    mgpu.infer_layout(self.module)

    wgmma_layout = layouts.to_layout_attr(mgpu.WGMMA_LAYOUT)
    in_layouts = [wgmma_layout]
    out_layouts = [wgmma_layout]
    if lhs_memory_space == "registers":
      in_layouts.append(wgmma_layout)

    self.checkInLayouts(wgmma_op, in_layouts)
    self.checkOutLayouts(wgmma_op, out_layouts)

  def test_vector_broadcast_from_scalar_infers_splat_layout(self):
    shape = (128,)
    f32 = ir.F32Type.get()
    layout = mgpu.WGSplatFragLayout(shape)
    with ir.InsertionPoint(self.module.body):
      source, = undefs(f32)
      bcast = vector.BroadcastOp(ir.VectorType.get(shape, f32), source)

    mgpu.infer_layout(self.module)
    self.assertNotIn("in_layouts", bcast.attributes)
    self.checkOutLayouts(bcast, [layout])

  def test_vector_reduction_infers_reducible_producer_layout(self):
    shape = (128,)
    f32 = ir.F32Type.get()
    layout = mgpu.WGMMA_ROW_LAYOUT
    with ir.InsertionPoint(self.module.body):
      source, = undefs(ir.VectorType.get(shape, f32))
      source = layout_cast(source, layout)
      reduction = vector.ReductionOp(f32, vector.CombiningKind.ADD, source)

    mgpu.infer_layout(self.module)
    self.checkInLayouts(reduction, [layout])
    self.assertNotIn("out_layouts", reduction.attributes)

  def test_infer_layout_of_custom_primitive_op_uses_argument_layouts(self):
    in_layouts = [mgpu.WGMMA_LAYOUT, mgpu.WGMMA_ROW_LAYOUT]
    out_layouts = [mgpu.WGMMA_COL_LAYOUT]
    with ir.InsertionPoint(self.module.body):
      f32 = ir.F32Type.get()
      vec_ty = ir.VectorType.get((128, 128), f32)
      op = mgpu.dialect.CustomPrimitiveOp(
          result=[vec_ty],
          operands_=undefs(f32, vec_ty, vec_ty, f32),
          in_layouts=[layouts.to_layout_attr(l) for l in in_layouts],
          in_transforms=[],
          out_layouts=[layouts.to_layout_attr(l) for l in out_layouts],
      )

    mgpu.infer_layout(self.module)
    self.checkInLayouts(op, in_layouts)
    self.checkOutLayouts(op, out_layouts)

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
      mgpu.infer_layout(self.module)

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
      mgpu.infer_layout(self.module)

  def test_cant_infer_tmem_layout_when_no_hint_is_provided(self):
    f32 = ir.F32Type.get()
    i32 = ir.IntegerType.get_signless(32)
    ptr_type = ir.MemRefType.get((1,), i32, memory_space=mgpu.utils.smem())
    ref_ty = ir.MemRefType.get((128, 128), f32, memory_space=mgpu.utils.tmem())

    with ir.InsertionPoint(self.module.body):
      ptr = llvm.mlir_undef(ptr_type)
      ref = mgpu.dialect.tmem_alloc(result=ref_ty, smem_ptr=ptr)
      mgpu.dialect.tmem_dealloc(ref)

    # TODO(allanrenucci): Should we infer a default layout instead?
    with self.assertRaisesRegex(
        ValueError, "Failed to infer a possible set of layouts"
    ):
      mgpu.infer_layout(self.module)

  def test_infer_tmem_layout_cast_correctly(self):
    f32 = ir.F32Type.get()
    ref_ty = ir.MemRefType.get((128, 128), f32, memory_space=mgpu.utils.tmem())
    layout = layouts.to_layout_attr(mgpu.TMEM_NATIVE_LAYOUT)

    with ir.InsertionPoint(self.module.body):
      ref = llvm.mlir_undef(ref_ty)
      op = mgpu.dialect.TmemLayoutCastOp(ref, layout)

    mgpu.infer_layout(self.module)
    self.checkInTmemLayouts(op, [layout])
    self.checkOutTmemLayouts(op, [layout])

  def test_cant_relayout_tmem(self):
    f32 = ir.F32Type.get()
    ref_ty = ir.MemRefType.get((128, 128), f32, memory_space=mgpu.utils.tmem())

    with ir.InsertionPoint(self.module.body):
      ref = llvm.mlir_undef(ref_ty)
      layout = tcgen05.tmem_default_layout(packing=1)
      ref = mgpu.dialect.tmem_layout_cast(ref, layouts.to_layout_attr(layout))
      layout = tcgen05.tmem_half_lane_layout(columns=128, packing=1)
      mgpu.dialect.tmem_layout_cast(ref, layouts.to_layout_attr(layout))

    with self.assertRaisesRegex(
        ValueError, "Failed to infer a possible set of layouts."
    ):
      mgpu.infer_layout(self.module)

  def test_infer_tmem_alloc_layout_correctly(self):
    f32 = ir.F32Type.get()
    i32 = ir.IntegerType.get_signless(32)
    ptr_type = ir.MemRefType.get((1,), i32, memory_space=mgpu.utils.smem())
    ref_ty = ir.MemRefType.get((128, 128), f32, memory_space=mgpu.utils.tmem())
    layout = layouts.to_layout_attr(mgpu.TMEM_NATIVE_LAYOUT)

    with ir.InsertionPoint(self.module.body):
      ptr = llvm.mlir_undef(ptr_type)
      op = mgpu.dialect.TmemAllocOp(ref_ty, ptr)
      mgpu.dialect.tmem_layout_cast(op.result, layout)

    mgpu.infer_layout(self.module)
    self.assertNotIn("in_tmem_layouts", op.attributes)
    self.checkOutTmemLayouts(op, [layout])

  def test_tmem_dealloc_propagates_producer_layout(self):
    f32 = ir.F32Type.get()
    ref_ty = ir.MemRefType.get((128, 128), f32, memory_space=mgpu.utils.tmem())
    layout = layouts.to_layout_attr(mgpu.TMEM_NATIVE_LAYOUT)

    with ir.InsertionPoint(self.module.body):
      ref = llvm.mlir_undef(ref_ty)
      ref = mgpu.dialect.tmem_layout_cast(ref, layout)
      op = mgpu.dialect.TmemDeallocOp(ref)

    mgpu.infer_layout(self.module)
    self.checkInTmemLayouts(op, [layout])
    self.assertNotIn("out_tmem_layouts", op.attributes)

  @parameterized.parameters(False, True)
  def test_infer_tcgen05_mma_layouts_correctly(self, a_in_tmem):
    f16, f32, i1 = ir.F16Type.get(), ir.F32Type.get(), ir.IntegerType.get_signless(1)
    shape = (128, 128)
    acc_type = ir.MemRefType.get(shape, f32, memory_space=mgpu.utils.tmem())
    a_mem_space = mgpu.utils.tmem() if a_in_tmem else mgpu.utils.smem()
    a_type = ir.MemRefType.get(shape, f16, memory_space=a_mem_space)
    b_type = ir.MemRefType.get(shape, f16, memory_space=mgpu.utils.smem())

    with ir.InsertionPoint(self.module.body):
      [acc, a, b, accumulate] = undefs(acc_type, a_type, b_type, i1)
      op = mgpu.dialect.TcGen05MMAOp(acc, a, b, accumulate)

    mgpu.infer_layout(self.module)
    self.assertNotIn("out_tmem_layouts", op.attributes)
    acc_layout = tcgen05._infer_tmem_layout(shape, collective=False, packing=1)
    a_layout = tcgen05._infer_tmem_layout(shape, collective=False, packing=2)
    expected_layouts = [acc_layout, a_layout] if a_in_tmem else [acc_layout]
    self.checkInTmemLayouts(op, expected_layouts)

  def test_async_load_tmem_accepts_compatible_in_out_layouts(self):
    f32 = ir.F32Type.get()
    i32 = ir.IntegerType.get_signless(32)
    shape = (128, 128)
    ptr_type = ir.MemRefType.get((1,), i32, memory_space=mgpu.utils.smem())
    ref_type = ir.MemRefType.get(shape, f32, memory_space=mgpu.utils.tmem())
    in_layout = tcgen05.tmem_default_layout(packing=1)
    in_layout = layouts.to_layout_attr(in_layout)
    out_layout = layouts.to_layout_attr(fa.TCGEN05_LAYOUT)

    with ir.InsertionPoint(self.module.body):
      ptr = llvm.mlir_undef(ptr_type)
      ref = mgpu.dialect.tmem_alloc(ref_type, ptr)
      ref = mgpu.dialect.tmem_layout_cast(ref, in_layout)
      op = mgpu.dialect.AsyncLoadTmemOp(ref)
      mgpu.dialect.layout_cast(op.result, out_layout)

    mgpu.infer_layout(self.module)
    self.checkInTmemLayouts(op, [in_layout])
    self.checkOutLayouts(op, [out_layout])

  def test_async_load_tmem_rejects_incompatible_in_out_layouts(self):
    f32 = ir.F32Type.get()
    i32 = ir.IntegerType.get_signless(32)
    shape = (128, 128)
    ptr_type = ir.MemRefType.get((1,), i32, memory_space=mgpu.utils.smem())
    ref_type = ir.MemRefType.get(shape, f32, memory_space=mgpu.utils.tmem())
    in_layout = tcgen05.tmem_half_lane_layout(columns=shape[1], packing=1)
    in_layout = layouts.to_layout_attr(in_layout)
    out_layout = layouts.to_layout_attr(fa.TCGEN05_LAYOUT)

    with ir.InsertionPoint(self.module.body):
      ptr = llvm.mlir_undef(ptr_type)
      ref = mgpu.dialect.tmem_alloc(ref_type, ptr)
      ref = mgpu.dialect.tmem_layout_cast(ref, in_layout)
      op = mgpu.dialect.AsyncLoadTmemOp(ref)
      mgpu.dialect.layout_cast(op.result, out_layout)

    with self.assertRaisesRegex(
        ValueError, "Failed to infer a possible set of layouts."
    ):
      mgpu.infer_layout(self.module)

  def test_async_store_tmem_accepts_compatible_src_dest_layouts(self):
    f32 = ir.F32Type.get()
    i32 = ir.IntegerType.get_signless(32)
    shape = (128, 128)
    ptr_type = ir.MemRefType.get((1,), i32, memory_space=mgpu.utils.smem())
    dest_type = ir.MemRefType.get(shape, f32, memory_space=mgpu.utils.tmem())
    src_type = ir.VectorType.get(shape, f32)
    src_layout = layouts.to_layout_attr(fa.TCGEN05_LAYOUT)
    dest_layout = tcgen05.tmem_default_layout(packing=1)
    dest_layout = layouts.to_layout_attr(dest_layout)

    with ir.InsertionPoint(self.module.body):
      [ptr, src] = undefs(ptr_type, src_type)
      src = mgpu.dialect.layout_cast(src, src_layout)
      dest = mgpu.dialect.tmem_alloc(dest_type, ptr)
      dest = mgpu.dialect.tmem_layout_cast(dest, dest_layout)
      op = mgpu.dialect.AsyncStoreTmemOp(src, dest)

    mgpu.infer_layout(self.module)
    self.checkInLayouts(op, [src_layout])
    self.checkInTmemLayouts(op, [dest_layout])

  def test_layout_inference_gelu_does_not_timeout(self):
    # This test is intended to make sure that the constraint-based layout
    # inference does not timeout on a Gelu kernel. This was previously the case,
    # and we want to make sure that regressions don't happen.
    with ir.InsertionPoint(self.module.body):
      shape = (128,)
      f32 = ir.F32Type.get()
      vector_ty = ir.VectorType.get(shape, f32)
      memref_ty = ir.MemRefType.get(shape, f32)

      # The code below is essentially jax.nn.gelu().
      c_05 = arith.constant(vector_ty, ir.DenseElementsAttr.get_splat(vector_ty, ir.FloatAttr.get(f32, 0.5)))
      c_1 = arith.constant(vector_ty, ir.DenseElementsAttr.get_splat(vector_ty,  ir.FloatAttr.get(f32, 1.0)))
      c_079 = arith.constant(vector_ty, ir.DenseElementsAttr.get_splat(vector_ty,  ir.FloatAttr.get(f32, 0.797884583)))
      c_044 = arith.constant(vector_ty, ir.DenseElementsAttr.get_splat(vector_ty,  ir.FloatAttr.get(f32, 0.044715)))

      zero = mgpu.utils.c(0, ir.IntegerType.get_signless(32))
      memref = llvm.UndefOp(memref_ty)
      load = vector.LoadOp(vector_ty, memref, [zero])
      x = load.result
      x2 = arith.mulf(x, x)
      x3 = arith.mulf(x2, x)
      y = arith.mulf(x3, c_044)
      x_y = arith.addf(x, y)
      z = arith.mulf(x_y, c_079)
      t = math.tanh(z)
      u = arith.addf(t, c_1)
      v = arith.mulf(u, c_05)
      r = arith.mulf(x, v)
      store = vector.StoreOp(r, memref, [zero])

    mgpu.infer_layout(self.module)

    strided_layout = layouts.to_layout_attr(mgpu.WGStridedFragLayout(shape, 1))
    self.checkOutLayouts(load, [strided_layout])
    self.checkInLayouts(store, [strided_layout])


if __name__ == "__main__":
  parameterized.absltest.main(testLoader=jtu.JaxTestLoader())
