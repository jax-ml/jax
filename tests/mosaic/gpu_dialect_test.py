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
"""(Deviceless) tests for the Mosaic GPU MLIR dialect."""

from absl.testing import parameterized
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lib.mlir import ir
from jax.experimental.mosaic.gpu import dialect as mgpu  # pylint: disable=g-importing-member


_cext = mgpu._cext if mgpu is not None else None


config.parse_flags_with_absl()


def _make_ir_context():
  context = ir.Context()
  mgpu.register_dialect(context)
  return context


class DialectTest(parameterized.TestCase):

  def setUp(self):
    if mgpu is None:
      raise self.skipTest("Test requires Mosaic GPU dialect")
    super().setUp()
    self.enter_context(_make_ir_context())
    self.enter_context(ir.Location.unknown())
    self.module = ir.Module.create()

  def test_dialect_module_is_loaded(self):
    self.assertTrue(_cext.globals._check_dialect_module_loaded("mosaic_gpu"))

  def test_initialize_barrier_op_result_memref_must_wrap_barriers(self):
    with ir.InsertionPoint(self.module.body):
      mgpu.initialize_barrier(
          ir.MemRefType.get((1, 2), ir.F32Type.get()), arrival_count=1)
    with self.assertRaisesRegex(
        ir.MLIRError, "must be memref of barrier values"):
      self.module.operation.verify()

  def test_initialize_barrier_op_arrival_count_must_be_strictly_positive(self):
    with ir.InsertionPoint(self.module.body):
      mgpu.initialize_barrier(
          ir.MemRefType.get((1, 2), ir.Type.parse("!mosaic_gpu.barrier")),
          arrival_count=0)
    with self.assertRaisesRegex(ir.MLIRError, "value is positive"):
      self.module.operation.verify()

  def test_initialize_barrier_op_with_a_positive_arrival_count_passes(self):
    with ir.InsertionPoint(self.module.body):
      mgpu.initialize_barrier(
          ir.MemRefType.get((1, 2), ir.Type.parse("!mosaic_gpu.barrier")),
          arrival_count=1)
    self.assertTrue(self.module.operation.verify())
    self.assertIsInstance(self.module.body.operations[0],
                          mgpu.InitializeBarrierOp)


if __name__ == "__main__":
  parameterized.absltest.main(testLoader=jtu.JaxTestLoader())
