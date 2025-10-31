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

import math

from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
from jax._src.interpreters import mlir as mlir_interpreter
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import builtin
from jax._src.lib.mlir.dialects import llvm
from jax._src.lib.mlir.dialects import math as math_dialect
from jax._src.lib.mlir.dialects import memref
from jax._src.lib.mlir.dialects import scf
from jax._src.lib.mlir.dialects import vector
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.mosaic.gpu import equations as eqns
from jax.experimental.mosaic.gpu import fragmented_array as fa
from jax.experimental.mosaic.gpu import inference_utils
from jax.experimental.mosaic.gpu import launch_context as lc
from jax.experimental.mosaic.gpu import layout_inference
from jax.experimental.mosaic.gpu import layouts
from jax.experimental.mosaic.gpu import tcgen05
from jax.experimental.mosaic.gpu import test_util as mtu
import numpy as np

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
    layout_inference.ValueSitesForVariable,
    list[layout_inference.Hint],
]:
  del ctx
  # This rule is only called if the single output of the undef op is a vector or
  # TMEM reference, so we can just return a trivial mapping.
  result = layout_inference.ValueSite(
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
      splat = vector.BroadcastOp(rhs.type, lhs)
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
      elt_ty = ir.BF16Type.get()
      ab_type = ir.VectorType.get(shape, elt_ty)
      ref_ty = ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem())
      i32 = ir.IntegerType.get_signless(32)
      lower_bound, upper_bound, step, a, b, ref = undefs(
          i32, i32, i32, ab_type, ab_type, ref_ty
      )
      for_op = scf.ForOp(lower_bound, upper_bound, step, [a, b, ref])
      [loop_a, loop_b, loop_ref] = list(for_op.inner_iter_args)
      with ir.InsertionPoint(for_op.body):
        add = layout_cast(arith.addf(loop_a, loop_b), layout)

        transforms = ir.ArrayAttr.get([
          mgpu.dialect.TileTransformAttr.get((8, 64)),
          mgpu.dialect.SwizzleTransformAttr.get(128),
        ])
        loop_ref = mgpu.dialect.with_transforms(loop_ref, transforms)

        yield_op = scf.YieldOp([add, add, loop_ref])

    mgpu.infer_layout(self.module)

    carry_layouts = [layouts.to_layout_attr(layout)] * 2
    self.assertNotIn("out_layouts", yield_op.attributes)
    self.checkInLayouts(for_op, carry_layouts)
    self.checkOutLayouts(for_op, carry_layouts)
    [in_transform] = inference_utils.in_transforms(for_op)
    self.assertSequenceEqual(in_transform, transforms)
    [out_transform] = inference_utils.out_transforms(for_op)
    self.assertSequenceEqual(out_transform, transforms)

  def test_infer_layout_from_body_op_to_yield_op_to_for_op(self):
    shape = (64, 64)
    with ir.InsertionPoint(self.module.body):
      elt_ty = ir.BF16Type.get()
      c_ty = ir.VectorType.get(shape, elt_ty)
      ab_type = ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem())
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

    transforms = ir.ArrayAttr.get([
      mgpu.dialect.TileTransformAttr.get((8, 64)),
      mgpu.dialect.SwizzleTransformAttr.get(128),
    ])
    in_transforms = inference_utils.in_transforms(for_op)
    self.assertSequenceEqual(in_transforms, [transforms, transforms])
    out_transforms = inference_utils.out_transforms(for_op)
    self.assertSequenceEqual(out_transforms, [transforms, transforms])

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
      cast = layout_cast(add0.result, strided_layout)
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
      lc = layout_cast(x.result, layouts.to_layout_attr(layout)).owner.opview

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

  @parameterized.parameters(*layout_inference.MemorySpace)
  def test_relayout_only_derived_for_registers(self, memory_space):
    with ir.InsertionPoint(self.module.body):
      shape = (128,)
      f32 = ir.F32Type.get()
      match memory_space:
        case layout_inference.MemorySpace.REG:
          ty = ir.VectorType.get(shape, f32)
        case layout_inference.MemorySpace.TMEM:
          ty = ir.MemRefType.get(shape, f32, memory_space=mgpu.utils.tmem())
        case layout_inference.MemorySpace.SMEM:
          ty = ir.MemRefType.get(shape, f32, memory_space=mgpu.utils.smem())
        case _:
          raise ValueError(f"Unsupported memory space: {memory_space}")

      [producer] = undefs(ty)
      consumer = builtin.unrealized_conversion_cast([ty], [producer])

      r = layout_inference.ValueSite(
          producer.owner, layout_inference.VariableType.RESULT, 0
      )
      r_var = eqns.Variable(r)
      o = layout_inference.ValueSite(
          consumer.owner, layout_inference.VariableType.OPERAND, 0
      )
      o_var = eqns.Variable(o)

      hints, relayouts = layout_inference.derive_hints_and_constraints(
          layout_inference.ValueSitesForVariable({r_var: [r], o_var: [o]})
      )

      if memory_space == layout_inference.MemorySpace.REG:
        hint0 = layout_inference.Hint(r_var, eqns.MostReplicated((o_var,)))
        hint1 = layout_inference.Hint(
            o_var, eqns.MostReplicated((eqns.LeastReplicated((r_var,)),))
        )

        self.assertEqual(hints, [hint0, hint1])
        self.assertEqual(relayouts, [eqns.Relayout(r_var, o_var)])
      else:
        self.assertEmpty(hints)
        self.assertEmpty(relayouts)

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

    [key] = layout_inference.vector_value_sites(x)
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
    [kv0] = layout_inference.vector_value_sites(op0)
    [kv1] = layout_inference.vector_value_sites(op1)
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

  def test_infer_layout_for_tmem_alloc_by_default(self):
    f32 = ir.F32Type.get()
    i32 = ir.IntegerType.get_signless(32)
    shape = (128, 128)
    ptr_type = ir.MemRefType.get((1,), i32, memory_space=mgpu.utils.smem())
    ref_ty = ir.MemRefType.get(shape, f32, memory_space=mgpu.utils.tmem())

    with ir.InsertionPoint(self.module.body):
      ptr = llvm.mlir_undef(ptr_type)
      op = mgpu.dialect.TmemAllocOp(result=ref_ty, smem_ptr=ptr)

    mgpu.infer_layout(self.module)
    self.assertNotIn("in_tmem_layouts", op.attributes)
    layout = tcgen05._infer_tmem_layout(shape, collective=False, packing=1)
    self.checkOutTmemLayouts(op, [layout])

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

  def test_infer_async_load_chooses_in_tmem_layouts_compatible_with_register_layout(self):
    f32 = ir.F32Type.get()
    i32 = ir.IntegerType.get_signless(32)
    shape = (128, 128)
    ptr_type = ir.MemRefType.get((1,), i32, memory_space=mgpu.utils.smem())
    ref_type = ir.MemRefType.get(shape, f32, memory_space=mgpu.utils.tmem())
    out_layout = layouts.to_layout_attr(fa.TCGEN05_LAYOUT)

    with ir.InsertionPoint(self.module.body):
      ptr = llvm.mlir_undef(ptr_type)
      ref = mgpu.dialect.tmem_alloc(ref_type, ptr)
      op = mgpu.dialect.AsyncLoadTmemOp(ref)
      mgpu.dialect.layout_cast(op.result, out_layout)

    mgpu.infer_layout(self.module)
    in_layout = tcgen05.tmem_default_layout(packing=1)
    in_layout = layouts.to_layout_attr(in_layout)
    self.checkInTmemLayouts(op, [in_layout])
    self.checkOutLayouts(op, [out_layout])

  def test_infer_async_load_chooses_out_layouts_compatible_with_tmem_layout(self):
    f32 = ir.F32Type.get()
    i32 = ir.IntegerType.get_signless(32)
    shape = (128, 128)
    ptr_type = ir.MemRefType.get((1,), i32, memory_space=mgpu.utils.smem())
    ref_type = ir.MemRefType.get(shape, f32, memory_space=mgpu.utils.tmem())
    in_layout = tcgen05.tmem_default_layout(packing=1)
    in_layout = layouts.to_layout_attr(in_layout)

    with ir.InsertionPoint(self.module.body):
      ptr = llvm.mlir_undef(ptr_type)
      ref = mgpu.dialect.tmem_alloc(ref_type, ptr)
      ref = mgpu.dialect.tmem_layout_cast(ref, in_layout)
      op = mgpu.dialect.AsyncLoadTmemOp(ref)

    mgpu.infer_layout(self.module)
    self.checkInTmemLayouts(op, [in_layout])
    out_layout = layouts.to_layout_attr(fa.TCGEN05_LAYOUT)
    self.checkOutLayouts(op, [out_layout])

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
      memref = llvm.mlir_undef(memref_ty)
      load = vector.LoadOp(vector_ty, memref, [zero])
      x = load.result
      x2 = arith.mulf(x, x)
      x3 = arith.mulf(x2, x)
      y = arith.mulf(x3, c_044)
      x_y = arith.addf(x, y)
      z = arith.mulf(x_y, c_079)
      t = math_dialect.tanh(z)
      u = arith.addf(t, c_1)
      v = arith.mulf(u, c_05)
      r = arith.mulf(x, v)
      store = vector.StoreOp(r, memref, [zero])

    mgpu.infer_layout(self.module)

    strided_layout = layouts.to_layout_attr(mgpu.WGStridedFragLayout(shape, 1))
    self.checkOutLayouts(load, [strided_layout])
    self.checkInLayouts(store, [strided_layout])

  @parameterized.parameters(
      ((32, 256), ir.BF16Type, False, None, 16),
      ((32, 256), ir.BF16Type, False, (2, 64), 128),
      ((32, 256), ir.BF16Type, False, (2, 32), 64),
      ((32, 256), ir.BF16Type, False, (2, 16), 32),
      ((32, 256), ir.BF16Type, False, (2, 8), 16),
      ((5, 32, 256), ir.BF16Type, False, (2, 64), 128),
      ((5, 32, 256), ir.BF16Type, False, (2, 16), 32),
      ((3, 32, 256), ir.Float8E4M3FNType, False, (2, 128), 128),
      ((3, 32, 256), ir.Float8E4M3FNType, False, (2, 64), 64),
      ((3, 32, 256), ir.Float8E4M3FNType, False, (2, 32), 32),
      ((3, 32, 256), ir.Float8E4M3FNType, False, (2, 16), 16),
      ((3, 32, 256), ir.BF16Type, True, (16, 32), 32),
      ((3, 32, 256), ir.BF16Type, False, (64,), 128),
      ((256,), ir.BF16Type, False, (2, 2), None),
  )
  def test_compute_swizzle(self, shape, type, transposed, tiling, want_swizzle):
    with ir.InsertionPoint(self.module.body):
      ref_ty = ir.MemRefType.get(shape, type.get())
      if transposed:
        strides, offset = ref_ty.get_strides_and_offset()
        strides[-1], strides[-2] = strides[-2], strides[-1]
        layout = ir.StridedLayoutAttr.get(offset, strides)
        ref_ty = ir.MemRefType.get(shape, type.get(), layout)

      tile_transform = None if tiling is None else lc.TileTransform(tiling)

      if want_swizzle is None:
        with self.assertRaises(ValueError):
          layout_inference._compute_swizzle(ref_ty, tile_transform)
      else:
        swizzle = layout_inference._compute_swizzle(ref_ty, tile_transform)
        self.assertEqual(swizzle, mgpu.dialect.SwizzlingMode(want_swizzle))

  @parameterized.parameters([False, True])
  def test_conjure_smem_assignment_from_is_transferrable(self, transposed):
    # Create a var to use in the equation system.
    shape = (128, 128)
    f32 = ir.F32Type.get()
    layout = ir.StridedLayoutAttr.get(0, [1, 128]) if transposed else None
    ref_ty = ir.MemRefType.get(shape, f32, layout=layout, memory_space=mgpu.utils.smem())
    [ref] = undefs(ref_ty)
    value_site = layout_inference.ValueSite(
        operation=ref.owner,
        type=layout_inference.VariableType.RESULT,
        index=0,
    )
    var = eqns.Variable(value_site)

    def conjure(constraints) -> list[tuple[eqns.Variable, eqns.Constant]]:
      system = eqns.EquationSystem(constraints=constraints)
      return list(layout_inference.conjure_assignment({var}, system, []))

    # Yield only empty tiling with no constraints.
    with self.subTest("no_constraints_yield_empty_tiling"):
      self.assertEqual(conjure([]), [(var, eqns.SMEMTiling(None))])

    # Yield empty if not an mma layout.
    with self.subTest("not_mma_layout_yield_empty_tiling"):
      layout = eqns.RegisterLayout(fa.WGSplatFragLayout(shape))
      constraints = [eqns.IsTransferable(layout, var, (128, 128))]
      conjured = conjure(constraints)
      self.assertEqual(conjured, [(var, eqns.SMEMTiling(None))])

    wgmma_layout = eqns.RegisterLayout(fa.WGMMA_LAYOUT)

    # Yield also maximal tiling with no Divides constraints.
    with self.subTest("no_divides_constraints_yield_maximal_tiling_with_mma"):
      constraints = [eqns.IsTransferable(wgmma_layout, var, (128, 128))]
      conjured = conjure(constraints)
      if transposed:
        expected_tiling = (32, 8)
      else:
        expected_tiling = (8, 32)
      self.assertEqual(conjured, [
              (var, eqns.SMEMTiling(lc.TileTransform(expected_tiling))),
              (var, eqns.SMEMTiling(None)),
          ]
      )

    # Yield also valid tiling with Divides constraints.
    with self.subTest("divides_constraints_yield_valid_tiling"):
      constraints = [
          eqns.IsTransferable(wgmma_layout, var, (128, 128)),
          eqns.Divides(var, (32, 16)),
      ]
      conjured = conjure(constraints)
      if transposed:
        expected_tiling = (32, 8)
      else:
        expected_tiling = (8, 16)
      self.assertEqual(conjured, [
              (var, eqns.SMEMTiling(lc.TileTransform(expected_tiling))),
              (var, eqns.SMEMTiling(None)),
          ]
      )

  def test_conjure_orders_hints_correctly(self):
    # Create a var to use in the equation system.
    shape = (128, 128)
    f32 = ir.F32Type.get()
    [val] = undefs(ir.VectorType.get(shape, f32))
    value_site = layout_inference.ValueSite(
        operation=val.owner,
        type=layout_inference.VariableType.RESULT,
        index=0,
    )
    var = eqns.Variable(value_site)

    hints = [
        layout_inference.Hint(
            var,
            eqns.RegisterLayout(fa.WGSplatFragLayout((128, 128))),
        ),
        layout_inference.Hint(
            var,
            eqns.RegisterLayout(fa.WGMMA_LAYOUT),
        ),
        layout_inference.Hint(
            var,
            eqns.RegisterLayout(fa.WGStridedFragLayout(shape, vec_size=4)),
        ),
    ]

    system = eqns.EquationSystem()
    ordered = list(layout_inference.conjure_assignment({var}, system, hints))
    expected = [
        (var, eqns.RegisterLayout(fa.WGMMA_LAYOUT)),
        (var, eqns.RegisterLayout(fa.WGSplatFragLayout((128, 128)))),
        (var, eqns.RegisterLayout(fa.WGStridedFragLayout(shape, vec_size=4))),
        (var, eqns.RegisterLayout(fa.WGStridedFragLayout(shape, vec_size=2))),
    ]
    self.assertEqual(ordered, expected)

  def test_memref_load_store_op_transforms_are_empty(self):
    with ir.InsertionPoint(self.module.body):
      i32 = ir.IntegerType.get_signless(32)
      ref_ty = ir.MemRefType.get((), i32, memory_space=mgpu.utils.smem())

      [val, load_ref, store_ref] = undefs(i32, ref_ty, ref_ty)
      load_op = memref.LoadOp(load_ref, [])
      store_op = memref.StoreOp(val, store_ref, [])

      mgpu.infer_layout(self.module)

      want = ir.ArrayAttr.get([ir.ArrayAttr.get([])])
      self.assertEqual(inference_utils.in_transforms(load_op), want)
      self.assertEqual(inference_utils.in_transforms(store_op), want)

  @parameterized.product(
      swizzle=tuple(mgpu.dialect.SwizzlingMode),
      dtype=(jnp.bfloat16, jnp.float32),
      lhs_in_registers=(False, True),
  )
  def test_infer_transforms_for_wgmma_op(self, swizzle, dtype, lhs_in_registers):
    swizzle_elems = swizzle // np.dtype(dtype).itemsize
    m = 64
    # Note: `group_m` and `group_k` should be coprime with 2 for the test to be
    # correct. Otherwise, we may infer larger swizzles than the test intends to
    # check.
    group_m, group_k = 3, 3
    lhs_shape = (group_m * m, group_k * swizzle_elems)
    rhs_shape = (group_k * swizzle_elems, group_k * swizzle_elems)
    out_shape = (group_m * m, group_k * swizzle_elems)

    with ir.InsertionPoint(self.module.body):
      elt_ty = mgpu.utils.dtype_to_ir_type(dtype)
      lhs_ref_ty = ir.MemRefType.get(lhs_shape, elt_ty, memory_space=mgpu.utils.smem())
      lhs_vec_ty = ir.VectorType.get(lhs_shape, elt_ty)
      lhs_ty = lhs_vec_ty if lhs_in_registers else lhs_ref_ty
      rhs_ty = ir.MemRefType.get(rhs_shape, elt_ty, memory_space=mgpu.utils.smem())
      acc_ty = ir.VectorType.get(out_shape, elt_ty)
      [acc, lhs, rhs] = undefs(acc_ty, lhs_ty, rhs_ty)
      wgmma_op = mgpu.dialect.WGMMAOp(acc, lhs, rhs)

    mgpu.infer_layout(self.module)

    wgmma_layout = layouts.to_layout_attr(mgpu.WGMMA_LAYOUT)
    arg_transforms = ir.ArrayAttr.get([
        mgpu.dialect.TileTransformAttr.get((8, swizzle_elems)),
        mgpu.dialect.SwizzleTransformAttr.get(int(swizzle)),
    ])

    in_layouts = [wgmma_layout]
    out_layouts = [wgmma_layout]
    in_transforms = [arg_transforms]
    if lhs_in_registers:
      in_layouts.append(wgmma_layout)
    else:
      in_transforms.append(arg_transforms)

    self.checkInLayouts(wgmma_op, in_layouts)
    self.checkOutLayouts(wgmma_op, out_layouts)
    self.assertSequenceEqual(
        inference_utils.in_transforms(wgmma_op), in_transforms
    )

  @parameterized.product(
      swizzle_lhs=tuple(mgpu.dialect.SwizzlingMode),
      swizzle_rhs=tuple(mgpu.dialect.SwizzlingMode),
      dtype=(jnp.bfloat16, jnp.float32),
      lhs_in_tmem=(False, True),
  )
  def test_infer_transforms_for_tcgen05_mma_op(
      self, swizzle_lhs, swizzle_rhs, dtype, lhs_in_tmem
  ):
    swizzle_elems_lhs = swizzle_lhs // np.dtype(dtype).itemsize
    swizzle_elems_rhs = swizzle_rhs // np.dtype(dtype).itemsize
    m = 128
    # Note: `group_m` and `group_k` should be coprime with 2 for the test to be
    # correct. Otherwise, we may infer larger swizzles than the test intends to
    # check.
    group_k, group_n = 3, 5
    lhs_shape = (m, group_k * swizzle_elems_lhs)
    rhs_shape = (group_k * swizzle_elems_lhs, group_n * swizzle_elems_rhs)
    out_shape = (m, group_n * swizzle_elems_rhs)

    with ir.InsertionPoint(self.module.body):
      elt_ty = mgpu.utils.dtype_to_ir_type(dtype)
      lhs_mem_space = mgpu.utils.tmem() if lhs_in_tmem else mgpu.utils.smem()
      lhs_ty = ir.MemRefType.get(lhs_shape, elt_ty, memory_space=lhs_mem_space)
      rhs_ty = ir.MemRefType.get(rhs_shape, elt_ty, memory_space=mgpu.utils.smem())
      acc_ty = ir.MemRefType.get(out_shape, elt_ty, memory_space=mgpu.utils.tmem())
      [acc, lhs, rhs] = undefs(acc_ty, lhs_ty, rhs_ty)
      accumulate = arith.constant(ir.IntegerType.get_signless(1), 1)
      tcgen05_mma_op = mgpu.dialect.TcGen05MMAOp(acc, lhs, rhs, accumulate)

    mgpu.infer_layout(self.module)

    self.assertNotIn("out_tmem_layouts", tcgen05_mma_op.attributes)
    acc_layout = tcgen05._infer_tmem_layout(out_shape, collective=False, packing=1)
    a_packing = 2 if dtype == jnp.bfloat16 else 1
    a_layout = tcgen05._infer_tmem_layout(lhs_shape, collective=False, packing=a_packing)
    expected_layouts = [acc_layout, a_layout] if lhs_in_tmem else [acc_layout]
    self.checkInTmemLayouts(tcgen05_mma_op, expected_layouts)

    arg_transforms_lhs = ir.ArrayAttr.get([
        mgpu.dialect.TileTransformAttr.get((8, swizzle_elems_lhs)),
        mgpu.dialect.SwizzleTransformAttr.get(int(swizzle_lhs)),
    ])
    arg_transforms_rhs = ir.ArrayAttr.get([
        mgpu.dialect.TileTransformAttr.get((8, swizzle_elems_rhs)),
        mgpu.dialect.SwizzleTransformAttr.get(int(swizzle_rhs)),
    ])
    if lhs_in_tmem:
      transforms = [arg_transforms_rhs]
    else:
      transforms = [arg_transforms_lhs, arg_transforms_rhs]

    self.assertSequenceEqual(
        inference_utils.in_transforms(tcgen05_mma_op), transforms
    )

  @parameterized.parameters(mgpu.dialect.AsyncLoadOp, mgpu.dialect.AsyncStoreOp)
  def test_infer_transforms_for_async_load_store(self, op_type):
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      gmem_ty = ir.MemRefType.get(shape, elt_ty)
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem())
      barrier_ty = ir.Type.parse("!mosaic_gpu.barrier")
      gmem_ref, smem_ref, barrier = undefs(gmem_ty, smem_ty, barrier_ty)

      transforms = ir.ArrayAttr.get([
          mgpu.dialect.TileTransformAttr.get((8, 32)),
          mgpu.dialect.SwizzleTransformAttr.get(64),
      ])
      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      smem_ref = mgpu.dialect.with_transforms(smem_ref, transforms)
      if op_type == mgpu.dialect.AsyncLoadOp:
        op = mgpu.dialect.AsyncLoadOp(
            source=gmem_ref,
            destination=smem_ref,
            barrier=barrier,
            indices=[zero, zero],
            slice_lengths=shape,
            collective=ir.ArrayAttr.get([]),
        )
      else:
        op = mgpu.dialect.AsyncStoreOp(
            source=smem_ref,
            destination=gmem_ref,
            indices=[zero, zero],
            slice_lengths=shape,
        )

    mgpu.infer_layout(self.module)

    self.assertSequenceEqual(
        inference_utils.in_transforms(op), [transforms]
    )

  @parameterized.product(
      layout=tuple(mtu.RegisterLayout),
      major_dim_index=(0, 3, 4),
  )
  def test_infer_transforms_for_vector_load_op(
      self, layout, major_dim_index
  ):
    if layout == mtu.RegisterLayout.WG_SPLAT:
      self.skipTest("WG_SPLAT is not supported for `vector.load`.")

    big_shape = (128, 128)
    small_shape = (64, 64)
    elt_ty = ir.BF16Type.get()
    layout = layout.to_mgpu(small_shape, elt_ty)

    with ir.InsertionPoint(self.module.body):
      smem_ty = ir.MemRefType.get(big_shape, elt_ty, memory_space=mgpu.utils.smem())
      [smem_ref] = undefs(smem_ty)

      c = lambda x: arith.constant(ir.IntegerType.get_signless(32), x)
      zero = c(0)
      major_index = c(major_dim_index)

      vector_op = vector.LoadOp(
          ir.VectorType.get(small_shape, elt_ty), smem_ref, [major_index, zero]
      )

      layout_cast(vector_op.result, layout)

    if inference_utils.is_mma_layout(layout):
      expected_major_dim = math.gcd(8, major_dim_index)
      expected_transforms = ir.ArrayAttr.get([
          mgpu.dialect.TileTransformAttr.get((expected_major_dim, 64)),
          mgpu.dialect.SwizzleTransformAttr.get(128),
      ])
    else:
      expected_transforms = ir.ArrayAttr.get([])

    mgpu.infer_layout(self.module)
    self.assertSequenceEqual(
        inference_utils.in_transforms(vector_op), [expected_transforms]
    )

  @parameterized.product(
      layout=tuple(mtu.RegisterLayout),
      major_dim_index=(0, 3, 4),
  )
  def test_infer_transforms_for_vector_store_op(
      self, layout, major_dim_index
  ):
    big_shape = (128, 128)
    small_shape = (64, 64)
    elt_ty = ir.BF16Type.get()
    layout = layout.to_mgpu(small_shape, elt_ty)

    with ir.InsertionPoint(self.module.body):
      smem_ty = ir.MemRefType.get(big_shape, elt_ty, memory_space=mgpu.utils.smem())
      [smem_ref] = undefs(smem_ty)

      c = lambda x: arith.constant(ir.IntegerType.get_signless(32), x)
      zero = c(0)
      major_index = c(major_dim_index)

      [value_to_store] = undefs(ir.VectorType.get(small_shape, elt_ty))
      vector_op = vector.StoreOp(value_to_store, smem_ref, [major_index, zero])
      layout_cast(value_to_store, layout)

    if inference_utils.is_mma_layout(layout):
      expected_major_dim = math.gcd(8, major_dim_index)
      expected_transforms = ir.ArrayAttr.get([
          mgpu.dialect.TileTransformAttr.get((expected_major_dim, 64)),
          mgpu.dialect.SwizzleTransformAttr.get(128),
      ])
    else:
      expected_transforms = ir.ArrayAttr.get([])

    mgpu.infer_layout(self.module)
    self.assertSequenceEqual(
        inference_utils.in_transforms(vector_op), [expected_transforms]
    )

  def test_slice_smem_gets_empty_by_default(self):
    with ir.InsertionPoint(self.module.body):
      shape = (64, 64)
      elt_ty = ir.BF16Type.get()
      i32 = ir.IntegerType.get_signless(32)
      [offset] = undefs(i32)
      ref_ty = ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem())
      slice_smem_op = mgpu.dialect.SliceSMEMOp(ref_ty, offset)

      transforms = ir.ArrayAttr.get([])
      mgpu.infer_layout(self.module)
      self.assertSequenceEqual(
          inference_utils.out_transforms(slice_smem_op), [transforms]
      )

  def test_infer_transforms_preserves_with_transforms_requirements(self):
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      ref_ty = ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem())
      [ref] = undefs(ref_ty)

      transforms = ir.ArrayAttr.get([
        mgpu.dialect.TileTransformAttr.get((8, 64)),
        mgpu.dialect.SwizzleTransformAttr.get(128),
      ])
      mgpu.dialect.with_transforms(ref, transforms)

    mgpu.infer_layout(self.module)
    self.assertSequenceEqual(
        inference_utils.out_transforms(ref.owner), [transforms]
    )

  def test_infer_transforms_fails_on_conflicting_with_transforms_requirements(self):
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      ref_ty = ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem())
      [ref] = undefs(ref_ty)

      transforms1 = ir.ArrayAttr.get([
        mgpu.dialect.TileTransformAttr.get((8, 64)),
        mgpu.dialect.SwizzleTransformAttr.get(128),
      ])
      transforms2 = ir.ArrayAttr.get([
        mgpu.dialect.TileTransformAttr.get((16, 64)),
        mgpu.dialect.SwizzleTransformAttr.get(128),
      ])
      mgpu.dialect.with_transforms(ref, transforms1)
      mgpu.dialect.with_transforms(ref, transforms2)

    with self.assertRaisesRegex(ValueError, "Failed to infer"):
      mgpu.infer_layout(self.module)

  def test_infer_transforms_sets_default_empty_transforms_on_async_load(self):
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      gmem_ty = ir.MemRefType.get(shape, elt_ty)
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem())
      barrier_ty = ir.Type.parse("!mosaic_gpu.barrier")
      [gmem_ref, smem_ref, barrier] = undefs(gmem_ty, smem_ty, barrier_ty)

      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      async_load_op = mgpu.dialect.AsyncLoadOp(
          source=gmem_ref,
          destination=smem_ref,
          barrier=barrier,
          indices=[zero, zero],
          slice_lengths=shape,
          collective=ir.ArrayAttr.get([]),
      )

    mgpu.infer_layout(self.module)
    [in_transform] = inference_utils.in_transforms(async_load_op)
    self.assertSequenceEqual(in_transform, ir.ArrayAttr.get([]))

  @parameterized.parameters([False, True])
  def test_infer_transforms_for_memref_cast_op(self, annotate_producer):
    with ir.InsertionPoint(self.module.body):
      shape = (64, 64)
      elt_ty = ir.BF16Type.get()
      ref_ty = ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem())
      [ref] = undefs(ref_ty)

      transforms = ir.ArrayAttr.get([
        mgpu.dialect.TileTransformAttr.get((8, 64)),
        mgpu.dialect.SwizzleTransformAttr.get(128),
      ])

      if annotate_producer:
        ref = mgpu.dialect.with_transforms(ref, transforms)
      cast = memref.cast(ref_ty, ref)
      if not annotate_producer:
        mgpu.dialect.with_transforms(cast, transforms)

      mgpu.infer_layout(self.module)
      self.assertSequenceEqual(
          inference_utils.in_transforms(cast.owner), [transforms]
      )
      self.assertSequenceEqual(
          inference_utils.out_transforms(cast.owner), [transforms]
      )

  @parameterized.parameters([False, True])
  def test_infer_transforms_for_subview_raises_on_slice_incompatible_with_tile(
      self, annotate_input
  ):
    shape = (2, 64, 64)
    elt_ty = ir.BF16Type.get()

    in_ref_ty = ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem())
    out_ref_ty = ir.MemRefType.get((2, 64, 32), elt_ty, memory_space=mgpu.utils.smem())

    with ir.InsertionPoint(self.module.body):
      [in_ref] = undefs(in_ref_ty)

      transforms = ir.ArrayAttr.get([
        mgpu.dialect.TileTransformAttr.get((32, 16)),
        mgpu.dialect.SwizzleTransformAttr.get(32),
      ])

      if annotate_input:
        in_ref = mgpu.dialect.with_transforms(in_ref, transforms)

      subview_op = memref.SubViewOp(
          out_ref_ty,
          in_ref,
          [],
          [],
          [],
          static_offsets = [1, 0, 0],
          static_sizes = [2, 64, 8],
          static_strides = [1, 1, 1]
      )

      if not annotate_input:
        mgpu.dialect.with_transforms(subview_op.result, transforms)

    with self.assertRaisesRegex(ValueError, "Failed to infer"):
      mgpu.infer_layout(self.module)

  @parameterized.parameters([False, True])
  def test_infer_transforms_for_sibling_subviews_and_distant_op(
      self, even_offsets
  ):
    # This test uses the following op tree extracted from this ragged dot
    # kernel:
    # https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py
    #
    #   subview_op0   (slice = 64, 64)
    #   - subview_op1 (slice = 2, 64)
    #   - subview_op2 (slice = 4, 64, either at an even or odd offset)
    #   - subview_op3 (slice = 8, 64)
    #   - user_op0    (in_transforms = [tile(64, 64), swizzle(32)])
    #
    # First the in_transforms of user_op0 have to be propagated up to
    # subview_op0. Then they have to be propagated down and resolved. Finally
    # all subview ops need to have the same transforms.

    source_shape = (64, 64)
    elt_ty = ir.BF16Type.get()
    source_ref_ty = ir.MemRefType.get(source_shape, elt_ty, memory_space=mgpu.utils.smem())

    slice1_shape = (2, 64)
    slice2_shape = (4, 64)
    slice3_shape = (8, 64)

    slice0_ref_ty = ir.MemRefType.get(source_shape, elt_ty, memory_space=mgpu.utils.smem())
    slice1_ref_ty = ir.MemRefType.get(slice1_shape, elt_ty, memory_space=mgpu.utils.smem())
    slice2_ref_ty = ir.MemRefType.get(slice2_shape, elt_ty, memory_space=mgpu.utils.smem())
    slice3_ref_ty = ir.MemRefType.get(slice3_shape, elt_ty, memory_space=mgpu.utils.smem())

    want_tt = mgpu.dialect.TileTransformAttr.get((2 if even_offsets else 1, 64))

    with ir.InsertionPoint(self.module.body):
      [source_ref] = undefs(source_ref_ty)
      subview_op0 = memref.SubViewOp(
          slice0_ref_ty,
          source_ref,
          [],  # dynamic offsets
          [],  # dynamic sizes
          [],  # dynamic strides
          static_offsets=[0, 0],
          static_sizes=source_shape,
          static_strides=[1, 1],
      )

      transforms_0 = ir.ArrayAttr.get([want_tt])
      mgpu.dialect.WithTransformsOp(subview_op0.result, transforms_0)

      subview_op1 = memref.SubViewOp(
          slice1_ref_ty,
          subview_op0,
          [],  # dynamic offsets
          [],  # dynamic sizes
          [],  # dynamic strides
          static_offsets=[0, 0],
          static_sizes=slice1_shape,
          static_strides=[1, 1],
      )

      subview_op2 = memref.SubViewOp(
          slice2_ref_ty,
          subview_op0,
          [],  # dynamic offsets
          [],  # dynamic sizes
          [],  # dynamic strides
          static_offsets=[16 if even_offsets else 15, 0],
          static_sizes=slice2_shape,
          static_strides=[1, 1],
      )

      # The following ops are just to test the dynamic offsets support.
      c = lambda x: arith.constant(ir.IntegerType.get_signless(32), x)
      c64 = c(64)
      c32 = c(32)
      c16 = c(16)
      subi = arith.subi(c64, c32)
      maxsi = arith.maxsi(c16, subi)
      addi = arith.addi(maxsi, subi)
      andi = arith.andi(addi, maxsi)
      idx = arith.index_cast(ir.IndexType.get(), andi)
      subview_op3 = memref.SubViewOp(
          slice3_ref_ty,
          subview_op0,
          [idx],  # dynamic offsets
          [],  # dynamic sizes
          [],  # dynamic strides
          static_offsets=[ir.ShapedType.get_dynamic_size(), 0],
          static_sizes=slice3_shape,
          static_strides=[1, 1],
      )

    mgpu.infer_layout(self.module)

    want = ir.ArrayAttr.get([
        want_tt,
        mgpu.dialect.SwizzleTransformAttr.get(128),
    ])

    self.assertSequenceEqual(inference_utils.out_transforms(source_ref.owner), [want])
    self.assertSequenceEqual(inference_utils.in_transforms(subview_op0), [want])
    self.assertSequenceEqual(inference_utils.out_transforms(subview_op0), [want])
    self.assertSequenceEqual(inference_utils.in_transforms(subview_op1), [want])
    self.assertSequenceEqual(inference_utils.out_transforms(subview_op1), [want])
    self.assertSequenceEqual(inference_utils.in_transforms(subview_op2), [want])
    self.assertSequenceEqual(inference_utils.out_transforms(subview_op2), [want])
    self.assertSequenceEqual(inference_utils.in_transforms(subview_op3), [want])
    self.assertSequenceEqual(inference_utils.out_transforms(subview_op3), [want])

  @parameterized.parameters([False, True])
  def test_infer_transforms_for_subview_handles_dynamic_offsets(
      self, annotate_input
  ):
    shape = (32, 32, 32, 32)
    elt_ty = ir.BF16Type.get()

    in_ref_ty = ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem())
    out_ref_ty = ir.MemRefType.get((16, 16, 32, 32), elt_ty, memory_space=mgpu.utils.smem())

    with ir.InsertionPoint(self.module.body):
      [in_ref] = undefs(in_ref_ty)

      transforms = ir.ArrayAttr.get([
          mgpu.dialect.TileTransformAttr.get((4, 8, 16)),
          mgpu.dialect.SwizzleTransformAttr.get(32),
      ])

      if annotate_input:
        in_ref = mgpu.dialect.with_transforms(in_ref, transforms)

      c = lambda x: arith.constant(ir.IntegerType.get_signless(32), x)
      subview_op = memref.SubViewOp(
          out_ref_ty,
          in_ref,
          [c(16), c(4), arith.muli(c(8), c(3))],
          [],
          [],
          static_offsets=[
              ir.ShapedType.get_dynamic_size(),
              ir.ShapedType.get_dynamic_size(),
              ir.ShapedType.get_dynamic_size(),
              0,
          ],
          static_sizes=[16, 16, 32, 32],
          static_strides=[1, 1, 1, 1],
      )

      if not annotate_input:
        mgpu.dialect.with_transforms(subview_op.result, transforms)

    mgpu.infer_layout(self.module)

    self.assertSequenceEqual(
        inference_utils.in_transforms(subview_op), [transforms]
    )
    self.assertSequenceEqual(
        inference_utils.out_transforms(subview_op), [transforms]
    )

  def test_custom_primitive_op_retains_transforms(self):
    with ir.InsertionPoint(self.module.body):
      transforms = ir.ArrayAttr.get([
          mgpu.dialect.TileTransformAttr.get((64, 64)),
          mgpu.dialect.SwizzleTransformAttr.get(128),
      ])
      ref_ty = ir.MemRefType.get(
          (128, 128), ir.BF16Type.get(), memory_space=mgpu.utils.smem()
      )
      [ref] = undefs(ref_ty)
      op = mgpu.dialect.custom_primitive(
          result=[],
          operands_=[ref],
          in_layouts=[],
          in_transforms=[transforms],
          out_layouts=[],
      )

    mgpu.infer_layout(self.module)
    self.assertSequenceEqual(inference_utils.in_transforms(op), [transforms])

  def test_custom_primitive_op_with_conflicting_transforms_is_unsat(self):
    with ir.InsertionPoint(self.module.body):
      transforms_a = ir.ArrayAttr.get([
          mgpu.dialect.TileTransformAttr.get((64, 64)),
      ])
      transforms_b = ir.ArrayAttr.get([
          mgpu.dialect.TileTransformAttr.get((32, 32)),
      ])
      ref_ty = ir.MemRefType.get(
          (128, 128), ir.BF16Type.get(), memory_space=mgpu.utils.smem()
      )
      [ref] = undefs(ref_ty)
      mgpu.dialect.custom_primitive(
          result=[],
          operands_=[ref, ref],
          in_layouts=[],
          in_transforms=[transforms_a, transforms_b],
          out_layouts=[],
      )

    with self.assertRaisesRegex(ValueError, "Failed to infer"):
      mgpu.infer_layout(self.module)

  @parameterized.parameters([False, True])
  def test_infer_transforms_for_memref_transpose(self, annotate_input):
    in_shape = (32, 64)
    out_shape = (64, 32)
    elt_ty = ir.BF16Type.get()

    in_ref_ty = ir.MemRefType.get(
        in_shape, elt_ty, memory_space=mgpu.utils.smem()
    )
    layout = ir.StridedLayoutAttr.get(0, strides=[1, 64])
    out_ref_ty = ir.MemRefType.get(
        out_shape, elt_ty, layout=layout, memory_space=mgpu.utils.smem()
    )

    with ir.InsertionPoint(self.module.body):
      [in_ref] = undefs(in_ref_ty)

      in_transforms = ir.ArrayAttr.get([
          mgpu.dialect.TileTransformAttr.get((8, 16)),
          mgpu.dialect.SwizzleTransformAttr.get(32),
      ])

      if annotate_input:
        in_ref = mgpu.dialect.with_transforms(in_ref, in_transforms)

      permutation = ir.AffineMap.get_permutation((1, 0))
      transpose_op = memref.TransposeOp(out_ref_ty, in_ref, permutation)

      out_transforms = ir.ArrayAttr.get([
          mgpu.dialect.TileTransformAttr.get((16, 8)),
          mgpu.dialect.SwizzleTransformAttr.get(32),
      ])

      if not annotate_input:
        mgpu.dialect.with_transforms(transpose_op.result, out_transforms)

    mgpu.infer_layout(self.module)

    self.assertSequenceEqual(
        inference_utils.in_transforms(transpose_op), [in_transforms]
    )
    self.assertSequenceEqual(
        inference_utils.out_transforms(transpose_op), [out_transforms]
    )

  def test_default_strided_layout_assignment_is_deterministic(self):
    with ir.InsertionPoint(self.module.body):
      shape = (8, 128)
      src_elt_ty = ir.IntegerType.get_signless(32)
      dst_elt_ty = ir.IntegerType.get_signless(16)
      src_ref_ty = ir.MemRefType.get(shape, src_elt_ty)
      dst_ref_ty = ir.MemRefType.get(shape, dst_elt_ty)
      zero = mgpu.utils.c(0, ir.IntegerType.get_signless(32))
      src_ref, dst_ref = undefs(src_ref_ty, dst_ref_ty)

      # Make sure to have at least three ops such that the default assignment
      # can pick a vector size from data types of various lengths.
      src = vector.load(ir.VectorType.get(shape, src_elt_ty), src_ref, [zero])
      conversion = arith.TruncIOp(ir.VectorType.get(shape, dst_elt_ty), src)
      vector.store(conversion.result, dst_ref, [zero])

    mgpu.infer_layout(self.module)

    # The default assignment should yield a strided layout here. The specific
    # vector size does not matter to the test, but it is important that it is
    # consistent between several runs of the test. If the logic changes such
    # that another vector size is deterministically chosen, it is likely fine to
    # edit this.
    layout = fa.WGStridedFragLayout(shape, vec_size=2)
    self.checkInLayouts(conversion, [layout])
    self.checkOutLayouts(conversion, [layout])

  def test_infer_layout_for_vector_extract_strided_slice(self):
    layout = layouts.to_layout_attr(fa.WGMMA_LAYOUT)
    with ir.InsertionPoint(self.module.body):
      i16 = ir.IntegerType.get_signless(16)
      src_ty = ir.VectorType.get([128, 128], i16)
      [src] = undefs(src_ty)
      src = mgpu.dialect.layout_cast(src, layout)
      dest_ty = ir.VectorType.get([64, 64], i16)
      op = vector.ExtractStridedSliceOp(dest_ty, src, [0, 64], [64, 64], [1, 1])
    mgpu.infer_layout(self.module)
    self.checkInLayouts(op, [layout])
    self.checkOutLayouts(op, [layout])

  @parameterized.named_parameters(
      (
          "tiled_layout_non_divisible_by_offset",
          mtu.RegisterLayout.WGMMA,
          [3, 64],
      ),
      ("strided_layout", mtu.RegisterLayout.WG_STRIDED, [0, 64]),
      ("splat_layout", mtu.RegisterLayout.WG_SPLAT, [0, 64]),
  )
  def test_infer_layout_for_vector_extract_strided_slice_fails(
      self, layout, offsets
  ):
    with ir.InsertionPoint(self.module.body):
      i16 = ir.IntegerType.get_signless(16)
      src_ty = ir.VectorType.get([128, 128], i16)
      [src] = undefs(src_ty)
      layout_attr = layout.to_layout_attr(src_ty.shape, src_ty.element_type)
      src = mgpu.dialect.layout_cast(src, layout_attr)
      dest_ty = ir.VectorType.get([64, 64], i16)
      vector.extract_strided_slice(dest_ty, src, offsets, [64, 64], [1, 1])
    with self.assertRaisesRegex(
        ValueError, "Failed to infer a possible set of layouts."
    ):
      mgpu.infer_layout(self.module)


if __name__ == "__main__":
  parameterized.absltest.main(testLoader=jtu.JaxTestLoader())
