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
from jax._src.lib.mlir.dialects import func
from jax._src.lib.mlir.dialects import scf
from jax._src.lib.mlir.dialects import vector
import jax.experimental.mosaic.gpu as mgpu
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


class LayoutInferenceTest(parameterized.TestCase):

  def setUp(self):
    if jax.version._version != jax.lib.__version__:
      raise self.skipTest("Test requires matching jax and jaxlib versions")
    super().setUp()
    self.enter_context(_make_ir_context())
    self.enter_context(ir.Location.unknown())
    self.module = ir.Module.create()

  def checkInLayouts(self, op, in_layouts):
    self.assertSequenceEqual(op.attributes["in_layouts"], in_layouts)

  def checkOutLayouts(self, op, out_layouts):
    self.assertSequenceEqual(op.attributes["out_layouts"], out_layouts)

  def test_infer_strided_layout_default(self):
    shape = (16, 8)
    elt_type = ir.BF16Type.get()
    add = None

    def body(a, b):
      nonlocal add
      add = arith.AddFOp(a, b)

    with ir.InsertionPoint(self.module.body):
      ty = ir.VectorType.get(shape, elt_type)
      func.FuncOp.from_py_func(ty, ty)(body)

    # Not setting any layouts on the module should default in ops having a
    # strided fragmented layout.
    mgpu.infer_layout(self.module)

    layout = layouts.to_layout_attr(
        mgpu.WGStridedFragLayout.from_shaped_type(ty)
    )

    self.checkInLayouts(add, [layout, layout])
    self.checkOutLayouts(add, [layout])

  def test_infer_strided_layout_from_shape_cast(self):
    shape = (16, 8)
    elt_type = ir.BF16Type.get()
    src_type = ir.VectorType.get(shape, elt_type)
    dst_type = ir.VectorType.get([*reversed(shape)], elt_type)
    op = None

    def body(x):
      nonlocal op
      op = vector.ShapeCastOp(dst_type, x)

    with ir.InsertionPoint(self.module.body):
      func.FuncOp.from_py_func(src_type)(body)

    mgpu.infer_layout(self.module)

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
    mgpu.infer_layout(self.module)
    self.checkInLayouts(op, [in_layout])

  def test_infer_splat_layout_for_splat_constants(self):
    shape = (16, 8)
    elt_type = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      ty = ir.VectorType.get(shape, elt_type)
      c0 = ir.FloatAttr.get(elt_type, 0)
      c1 = ir.FloatAttr.get(elt_type, 1)
      splat0 = arith.ConstantOp(ty, ir.DenseElementsAttr.get_splat(ty, c0))
      splat1 = arith.ConstantOp(ty, ir.DenseElementsAttr.get_splat(ty, c1))
      add = arith.AddFOp(splat0, splat1)

    # Not setting any layouts on the module should default in all ops having a
    # splat fragmented layout.
    mgpu.infer_layout(self.module)

    layout = layouts.to_layout_attr(mgpu.WGSplatFragLayout(shape=shape))

    self.assertEmpty(splat0.attributes["in_layouts"])
    self.checkOutLayouts(splat0, [layout])

    self.assertEmpty(splat1.attributes["in_layouts"])
    self.checkOutLayouts(splat1, [layout])

    self.checkInLayouts(add, [layout, layout])
    self.checkOutLayouts(add, [layout])

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

    self.assertEmpty(c.attributes["in_layouts"])
    self.checkOutLayouts(c, [layout])

  @parameterized.parameters(True, False)
  def test_infer_splat_layout_for_vector_splat(self, rhs_splat):
    add = splat = None
    shape = (16, 8)
    layout = layouts.to_layout_attr(mgpu.WGSplatFragLayout(shape=shape))

    def body(lhs, rhs):
      nonlocal add, splat
      rhs = layout_cast(rhs, layout) if rhs_splat else rhs
      splat = vector.SplatOp(rhs.type, lhs)
      add = arith.AddFOp(splat.result, rhs)

    with ir.InsertionPoint(self.module.body):
      elt_type = ir.BF16Type.get()
      ty = ir.VectorType.get(shape, elt_type)
      func.FuncOp.from_py_func(elt_type, ty)(body)

    mgpu.infer_layout(self.module)

    self.assertEmpty(splat.attributes["in_layouts"])
    self.checkOutLayouts(splat, [layout])

    add_layout = layout if rhs_splat else layouts.to_layout_attr(
        mgpu.WGStridedFragLayout.from_shaped_type(ty)
    )

    self.checkInLayouts(add, [add_layout, add_layout])
    self.checkOutLayouts(add, [add_layout])

  @parameterized.parameters(
      mgpu.WGSplatFragLayout(shape=(32, 4)),
      mgpu.WGStridedFragLayout(shape=(32, 4), vec_size=1),
  )
  def test_pointwise_op_propagates_argument_layouts(self, layout):
    add = None

    def body(lhs, rhs):
      nonlocal add
      lhs = layout_cast(lhs, layout)
      rhs = layout_cast(rhs, layout)
      add = arith.AddFOp(lhs, rhs)

    with ir.InsertionPoint(self.module.body):
      ty = ir.VectorType.get(layout.shape, ir.BF16Type.get())
      func.FuncOp.from_py_func(ty, ty)(body)

    mgpu.infer_layout(self.module)

    layout_attr = layouts.to_layout_attr(layout)
    self.checkInLayouts(add, [layout_attr, layout_attr])
    self.checkOutLayouts(add, [layout_attr])

  def test_infer_layout_cast_layout(self):
    add = cast = None

    shape = (128, 64)
    splat_layout = layouts.to_layout_attr(mgpu.WGSplatFragLayout(shape=shape))
    wgmma_layout = layouts.to_layout_attr(mgpu.WGMMA_LAYOUT)

    def body(x):
      nonlocal add, cast
      x = mgpu.dialect.layout_cast(x, splat_layout)
      add = arith.AddFOp(x, x)
      cast = mgpu.dialect.LayoutCastOp(add.result, wgmma_layout)

    with ir.InsertionPoint(self.module.body):
      elt_type = ir.BF16Type.get()
      ty = ir.VectorType.get(shape, elt_type)
      func.FuncOp.from_py_func(ty)(body)

    mgpu.infer_layout(self.module)
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
    bcast = None
    in_shape = (64,)
    out_shape = (64, 64)

    def body(x):
      nonlocal bcast
      if in_cast is not None:
        x = mgpu.dialect.LayoutCastOp(x, layouts.to_layout_attr(in_cast))

      out_type = ir.VectorType.get(out_shape, ir.F32Type.get())
      bcast = mgpu.dialect.BroadcastInDimOp(out_type, x, [broadcast_dim])

      if out_cast is not None:
        mgpu.dialect.LayoutCastOp(
            bcast.result, layouts.to_layout_attr(out_cast)
        )

    with ir.InsertionPoint(self.module.body):
      ty = ir.VectorType.get(in_shape, ir.F32Type.get())
      func.FuncOp.from_py_func(ty)(body)

    mgpu.infer_layout(self.module)
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
    red = None

    in_shape = (64, 64)
    out_shape = (64,)

    def body(x, acc):
      nonlocal red
      if in_cast is not None:
        x = mgpu.dialect.LayoutCastOp(x, layouts.to_layout_attr(in_cast))
      if acc_cast is not None:
        acc = mgpu.dialect.LayoutCastOp(acc, layouts.to_layout_attr(acc_cast))

      kind = vector.CombiningKind.MAXIMUMF
      red = vector.MultiDimReductionOp(kind, x, acc, [reduce_dim])

      if out_cast is not None:
        mgpu.dialect.LayoutCastOp(
            red.result, layouts.to_layout_attr(out_cast)
        )

    with ir.InsertionPoint(self.module.body):
      in_ty = ir.VectorType.get(in_shape, ir.F32Type.get())
      acc_ty = ir.VectorType.get(out_shape, ir.F32Type.get())
      func.FuncOp.from_py_func(in_ty, acc_ty)(body)

    mgpu.infer_layout(self.module)
    in_layout_attr = layouts.to_layout_attr(in_layout)
    out_layout_attr = layouts.to_layout_attr(out_layout)
    self.checkInLayouts(red, [in_layout_attr, out_layout_attr])
    self.checkOutLayouts(red, [out_layout_attr])

  def test_infer_layout_traverses_ops_correctly(self):
    shape = (16, 8)
    elt_type = ir.BF16Type.get()
    add = None

    def body(a, b):
      bool_type = ir.IntegerType.get_signless(1)
      cst_true = arith.constant(bool_type, ir.IntegerAttr.get(bool_type, 1))
      if_op = scf.IfOp(cst_true)
      with ir.InsertionPoint(if_op.then_block):
        nonlocal add
        add = arith.AddFOp(a, b)
        scf.yield_([])

    with ir.InsertionPoint(self.module.body):
      ab_type = ir.VectorType.get(shape, elt_type)
      func.FuncOp.from_py_func(ab_type, ab_type)(body)

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
    for_op = yield_op = None

    def body(lower_bound, upper_bound, step, a, b):
      nonlocal for_op
      for_op = scf.ForOp(lower_bound, upper_bound, step, [a, b])
      [loop_a, loop_b] = list(for_op.inner_iter_args)
      with ir.InsertionPoint(for_op.body):
        nonlocal yield_op
        add = arith.addf(loop_a, loop_b)
        add = layout_cast(add, layout)
        yield_op = scf.YieldOp([add, add])

    with ir.InsertionPoint(self.module.body):
      ab_type = ir.VectorType.get(shape, ir.BF16Type.get())
      i32 = ir.IntegerType.get_signless(32)
      func.FuncOp.from_py_func(i32, i32, i32, ab_type, ab_type)(body)

    mgpu.infer_layout(self.module)

    if isinstance(layout, mgpu.WGSplatFragLayout):
      # In the splat case, we should not propagate the splat layout from the
      # yield op. That is because we can not convert other layouts to a splat
      # layout---which could cause trouble if the initial carries have a
      # different layout. Instead, we should get the default annotation, i.e.
      # strided layouts.
      strided_layout = layouts.to_layout_attr(
          mgpu.WGStridedFragLayout.from_shaped_type(ab_type)
      )
      carry_layouts = [strided_layout, strided_layout]
      self.checkOutLayouts(yield_op, [])
      self.checkInLayouts(for_op, carry_layouts)
      self.checkOutLayouts(for_op, carry_layouts)
    else:
      carry_layouts = [layouts.to_layout_attr(layout)] * 2
      self.checkOutLayouts(yield_op, [])
      self.checkInLayouts(for_op, carry_layouts)
      self.checkOutLayouts(for_op, carry_layouts)

  def test_infer_layout_from_body_op_to_yield_op_to_for_op(self):
    for_op = yield_op = None
    shape = (64, 64)

    def body(lower_bound, upper_bound, step, a, b, c):
      nonlocal for_op
      for_op = scf.ForOp(lower_bound, upper_bound, step, [a, b, c])
      with ir.InsertionPoint(for_op.body):
        nonlocal yield_op
        [loop_a, loop_b, loop_c] = list(for_op.inner_iter_args)
        new_loop_c = mgpu.dialect.wgmma(loop_c, loop_a, loop_b)
        yield_op = scf.YieldOp([loop_a, loop_b, new_loop_c])

    with ir.InsertionPoint(self.module.body):
      c_ty = ir.VectorType.get(shape, ir.BF16Type.get())
      ab_ty = ir.MemRefType.get(shape, ir.BF16Type.get())
      i32 = ir.IntegerType.get_signless(32)
      func.FuncOp.from_py_func(i32, i32, i32, ab_ty, ab_ty, c_ty)(body)

    mgpu.infer_layout(self.module)

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
    while_op = condition_op = yield_op = None

    def body(condition, init, result):
      nonlocal while_op, condition_op, yield_op
      init = layout_cast(init, init_layout) if init_layout else init
      result = layout_cast(result, result_layout) if result_layout else result
      while_op = scf.WhileOp([out_type], [init])
      before_block = while_op.before.blocks.append(init.type)
      with ir.InsertionPoint(before_block):
        condition_op = scf.ConditionOp(condition, [result])

      after_block = while_op.after.blocks.append(out_type)
      with ir.InsertionPoint(after_block):
        yield_op = scf.YieldOp([init])

    with ir.InsertionPoint(self.module.body):
      i1 = ir.IntegerType.get_signless(1)
      func.FuncOp.from_py_func(i1, in_type, out_type)(body)

    mgpu.infer_layout(self.module)

    if init_layout is not None or result_layout is not None:
      init_layouts = [layouts.to_layout_attr(init_layout)] if init_layout else []
      result_layouts = [layouts.to_layout_attr(result_layout)] if result_layout else []
      self.checkInLayouts(while_op, init_layouts)
      self.checkOutLayouts(while_op, result_layouts)

  def test_infer_layout_has_no_layout_for_non_vector_types(self):
    shape = (32, 4)
    elt_ty = ir.BF16Type.get()

    vector_store = None

    def body(ref, array):
      nonlocal vector_store
      zero_index = arith.constant(ir.IndexType.get(), 0)
      vector_store = vector.store(array, ref, [zero_index, zero_index])

    with ir.InsertionPoint(self.module.body):
      ref_ty = ir.MemRefType.get(shape, elt_ty)
      array_ty = ir.VectorType.get(shape, elt_ty)
      func.FuncOp.from_py_func(ref_ty, array_ty)(body)

    mgpu.infer_layout(self.module)

    self.assertIn("in_layouts", vector_store.attributes)
    self.assertIn("out_layouts", vector_store.attributes)

    # The vector store should have a layout for the input array, but not for the
    # memref.
    self.assertLen(vector_store.attributes["in_layouts"], 1)
    self.assertEmpty(vector_store.attributes["out_layouts"])

  @parameterized.parameters(
      mgpu.WGStridedFragLayout((32, 4), vec_size=1),
      mgpu.WGMMA_LAYOUT,
  )
  def test_infer_layout_picks_non_splat_layout_over_splat_layout(
      self, layout
  ):
    add = None
    shape = (32, 4)
    splat_layout = layouts.to_layout_attr(mgpu.WGSplatFragLayout(shape))
    non_splat_layout = layouts.to_layout_attr(layout)

    def body(lhs, rhs):
      nonlocal add
      lhs = layout_cast(lhs, non_splat_layout)
      rhs = layout_cast(rhs, splat_layout)
      add = arith.AddFOp(lhs, rhs)

    with ir.InsertionPoint(self.module.body):
      elt_type = ir.BF16Type.get()
      ty = ir.VectorType.get(shape, elt_type)
      func.FuncOp.from_py_func(ty, ty)(body)

    mgpu.infer_layout(self.module)

    self.checkInLayouts(add, [non_splat_layout, non_splat_layout])
    self.checkOutLayouts(add, [non_splat_layout])

  def test_infer_layout_preserves_splat_layouts_in_producers(self):
    add0 = add1 = None
    shape = (32, 4)
    splat_layout = layouts.to_layout_attr(mgpu.WGSplatFragLayout(shape))
    strided_layout = layouts.to_layout_attr(
        mgpu.WGStridedFragLayout(shape, vec_size=1)
    )

    def body(lhs, rhs):
      nonlocal add0, add1
      lhs = layout_cast(lhs, splat_layout)
      rhs = layout_cast(rhs, splat_layout)
      add0 = arith.AddFOp(lhs, rhs)
      cast = layout_cast(add0, strided_layout)
      add1 = arith.AddFOp(cast, cast)

    with ir.InsertionPoint(self.module.body):
      elt_type = ir.BF16Type.get()
      ty = ir.VectorType.get(shape, elt_type)
      func.FuncOp.from_py_func(ty, ty)(body)

    mgpu.infer_layout(self.module)

    self.checkInLayouts(add0, [splat_layout, splat_layout])
    self.checkOutLayouts(add0, [splat_layout])
    self.checkInLayouts(add1, [strided_layout, strided_layout])
    self.checkOutLayouts(add1, [strided_layout])

  def test_infer_layout_does_not_assign_default_layouts_to_func(self):

    def body(lhs, rhs):
      arith.AddFOp(lhs, rhs)

    with ir.InsertionPoint(self.module.body):
      shape = (32, 4)
      ty = ir.VectorType.get(shape, ir.BF16Type.get())
      f = func.FuncOp.from_py_func(ty, ty)(body).func_op

    mgpu.infer_layout(self.module)
    self.assertNotIn("in_layouts", f.attributes)
    self.assertNotIn("out_layouts", f.attributes)

  def test_optimization_barrier_op_propagates_user_layouts(self):
    add = optimization_barrier = None
    wgmma_layout = layouts.to_layout_attr(mgpu.WGMMA_LAYOUT)

    def body(lhs, rhs):
      nonlocal add, optimization_barrier
      optimization_barrier = mgpu.dialect.OptimizationBarrierOp([lhs, rhs])
      lhs, rhs = optimization_barrier.results
      add = arith.AddFOp(lhs, rhs)
      add = layout_cast(add, wgmma_layout)

    with ir.InsertionPoint(self.module.body):
      ty = ir.VectorType.get((32, 4), ir.BF16Type.get())
      func.FuncOp.from_py_func(ty, ty)(body)

    mgpu.infer_layout(self.module)

    self.checkInLayouts(optimization_barrier, [wgmma_layout, wgmma_layout])
    self.checkOutLayouts(optimization_barrier, [wgmma_layout, wgmma_layout])

  def test_optimization_barrier_op_propagates_producer_layouts(self):
    add = optimization_barrier = None
    shape = (32, 4)
    splat_layout = layouts.to_layout_attr(mgpu.WGSplatFragLayout(shape))

    def body(lhs, rhs):
      nonlocal add, optimization_barrier
      lhs = layout_cast(lhs, splat_layout)
      rhs = layout_cast(rhs, splat_layout)
      add = arith.AddFOp(lhs, rhs)
      optimization_barrier = mgpu.dialect.OptimizationBarrierOp([add])

    with ir.InsertionPoint(self.module.body):
      ty = ir.VectorType.get(shape, ir.BF16Type.get())
      func.FuncOp.from_py_func(ty, ty)(body)

    mgpu.infer_layout(self.module)

    self.checkInLayouts(optimization_barrier, [splat_layout])
    self.checkOutLayouts(optimization_barrier, [splat_layout])


if __name__ == "__main__":
  parameterized.absltest.main(testLoader=jtu.JaxTestLoader())
