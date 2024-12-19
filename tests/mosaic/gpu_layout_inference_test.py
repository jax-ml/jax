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

from absl.testing import parameterized
from jax._src import config
from jax._src import test_util as jtu
from jax._src.interpreters import mlir as mlir_interpreter
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import func
from jax._src.lib.mlir.dialects import scf
import jax.experimental.mosaic.gpu as mgpu

config.parse_flags_with_absl()


def _make_ir_context():
  context = ir.Context()
  context.append_dialect_registry(mlir_interpreter.upstream_dialects)
  context.load_all_available_dialects()
  mgpu.dialect.register_dialect(context)
  return context


class LayoutInferenceTest(parameterized.TestCase):

  def setUp(self):
    if mgpu.dialect is None:
      raise self.skipTest("Test requires Mosaic GPU dialect")
    super().setUp()
    self.enter_context(_make_ir_context())
    self.enter_context(ir.Location.unknown())
    self.module = ir.Module.create()

  def test_infer_layout_default(self):
    shape = (16, 8)
    elt_type = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      ab_type = ir.VectorType.get(shape, elt_type)
      const_zero = ir.FloatAttr.get(elt_type, 0)
      const_one = ir.FloatAttr.get(elt_type, 1)
      a = arith.ConstantOp(
          ab_type, ir.DenseElementsAttr.get_splat(ab_type, const_zero)
      )
      b = arith.ConstantOp(
          ab_type, ir.DenseElementsAttr.get_splat(ab_type, const_one)
      )
      arith.addf(arith.addf(a, b), b)

    # Not setting any layouts on the module should default in ops having a
    # strided fragmented layout.
    mgpu.infer_layout(self.module)

    layout = mgpu.to_strided_fragmented_layout_attr(
        mgpu.WGStridedFragLayout.from_shaped_type(ab_type)
    )
    for op in self.module.body.operations:
      self.assertIn("in_layouts", op.attributes)
      self.assertIn("out_layouts", op.attributes)

      self.assertSequenceEqual(
          op.attributes["in_layouts"], [layout] * len(op.operands)
      )
      self.assertSequenceEqual(
          op.attributes["out_layouts"], [layout] * len(op.results)
      )

  def test_infer_layout_for_pointwise_op(self):
    shape = (4, 8)
    elt_type = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      ab_type = ir.VectorType.get(shape, elt_type)
      const_zero = ir.FloatAttr.get(elt_type, 0)
      const_one = ir.FloatAttr.get(elt_type, 1)
      a = arith.ConstantOp(
          ab_type, ir.DenseElementsAttr.get_splat(ab_type, const_zero)
      )
      b = arith.ConstantOp(
          ab_type, ir.DenseElementsAttr.get_splat(ab_type, const_one)
      )
      add = arith.addf(arith.addf(a, b), b)

    layout = mgpu.to_splat_fragmented_layout_attr(mgpu.WGSplatFragLayout(shape))
    add.owner.attributes["out_layouts"] = ir.ArrayAttr.get([layout])
    mgpu.infer_layout(self.module)

    for op in self.module.body.operations:
      self.assertIn("in_layouts", op.attributes)
      self.assertIn("out_layouts", op.attributes)

      self.assertSequenceEqual(
          op.attributes["in_layouts"], [layout] * len(op.operands)
      )
      self.assertSequenceEqual(
          op.attributes["out_layouts"], [layout] * len(op.results)
      )

  def test_infer_layout_traverses_ops_correctly(self):
    shape = (16, 8)
    elt_type = ir.BF16Type.get()
    add_op = None

    def body(a, b):
      bool_type = ir.IntegerType.get_signless(1)
      cst_true = arith.constant(bool_type, ir.IntegerAttr.get(bool_type, 1))
      if_op = scf.IfOp(cst_true)
      with ir.InsertionPoint(if_op.then_block):
        nonlocal add_op
        add_op = arith.addf(a, b)
        scf.yield_([])

    with ir.InsertionPoint(self.module.body):
      ab_type = ir.VectorType.get(shape, elt_type)
      func.FuncOp.from_py_func(ab_type, ab_type)(body)

    mgpu.infer_layout(self.module)

    self.assertIn("in_layouts", add_op.owner.attributes)
    self.assertIn("out_layouts", add_op.owner.attributes)


if __name__ == "__main__":
  parameterized.absltest.main(testLoader=jtu.JaxTestLoader())
