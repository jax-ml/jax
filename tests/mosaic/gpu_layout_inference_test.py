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
"""(Deviceless) layout inference tests for the Mosaic GPU MLIR dialect."""

from absl.testing import parameterized
from jax._src import config
from jax._src import test_util as jtu
from jax._src.interpreters import mlir as mlir_interpreter
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import llvm
from jax._src.lib.mlir.dialects import scf
from jax.experimental.mosaic.gpu import dialect as mgpu  # pylint: disable=g-importing-member
from jax.experimental.mosaic.gpu.dialect_lowering import infer_layout  # pylint: disable=g-importing-member
from jax.experimental.mosaic.gpu.dialect_lowering import splat_fragmented_layout  # pylint: disable=g-importing-member
from jax.experimental.mosaic.gpu.dialect_lowering import strided_fragmented_layout  # pylint: disable=g-importing-member


config.parse_flags_with_absl()


def _make_ir_context():
  context = ir.Context()
  context.append_dialect_registry(mlir_interpreter.upstream_dialects)
  context.load_all_available_dialects()
  mgpu.register_dialect(context)
  return context


class LayoutInferenceTest(parameterized.TestCase):

  def setUp(self):
    if mgpu is None:
      raise self.skipTest("Test requires Mosaic GPU dialect")
    super().setUp()
    self.enter_context(_make_ir_context())
    self.enter_context(ir.Location.unknown())
    self.module = ir.Module.create()

  def test_infer_layout_for_pointwise_op(self):
    shape = (4, 8)
    elt_type = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      ab_type = ir.VectorType.get(shape, elt_type)
      a = llvm.mlir_undef(ab_type)
      b = llvm.mlir_undef(ab_type)
      add = arith.addf(arith.addf(a, b), b)

    with self.subTest("infer_layout_with_no_layout_set"):
      infer_layout(self.module)

      # infer_layout should not have added any layout attributes, since there
      # is nothing to infer layouts from.
      for op in self.module.body.operations:
        self.assertNotIn("in_layouts", op.attributes)
        self.assertNotIn("out_layouts", op.attributes)

    add.owner.attributes["out_layouts"] = ir.ArrayAttr.get(
        [strided_fragmented_layout()]
    )

    with self.subTest("infer_layout_with_consistent_layout_set"):
      infer_layout(self.module)

      layout = strided_fragmented_layout()
      for op in self.module.body.operations:
        self.assertIn("in_layouts", op.attributes)
        self.assertIn("out_layouts", op.attributes)

        self.assertSequenceEqual(
            op.attributes["in_layouts"], [layout] * len(op.operands)
        )
        self.assertSequenceEqual(
            op.attributes["out_layouts"], [layout] * len(op.results)
        )

    with self.subTest("infer_layout_with_inconsistent_layout_set"):
      add.owner.attributes["out_layouts"] = ir.ArrayAttr.get(
          [splat_fragmented_layout()]
      )
      with self.assertRaisesRegex(
          ValueError,
          "Inferred out_layout .* but result has layout",
      ):
        infer_layout(self.module)


if __name__ == "__main__":
  parameterized.absltest.main(testLoader=jtu.JaxTestLoader())
