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

from typing import Callable

from absl.testing import parameterized
from jax._src import config
from jax._src import test_util as jtu
from jax._src.interpreters import mlir as mlir_interpreter
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import nvvm
from jax._src.lib.mlir.dialects import scf

from jax.experimental.mosaic.gpu import dialect as mgpu  # pylint: disable=g-importing-member
from jax.experimental.mosaic.gpu import lower_mgpu_dialect  # pylint: disable=g-importing-member,g-multiple-import

_cext = mgpu._cext if mgpu is not None else None


config.parse_flags_with_absl()


def _make_ir_context():
  context = ir.Context()
  context.append_dialect_registry(mlir_interpreter.upstream_dialects)
  context.load_all_available_dialects()
  mgpu.register_dialect(context)
  return context


def walk_operations(op: ir.OpView, callback):
  for region in op.operation.regions:
    for block in region:
      for block_op in block:
        walk_operations(block_op, callback)
  callback(op)


def find_if(module: ir.Module,
            predicate: Callable[[ir.OpView], bool]) -> list[ir.OpView]:
  result = []
  def callback(op: ir.OpView):
    if predicate(op):
      result.append(op)
  for op in module.body.operations:
    walk_operations(op, callback)
  return result


def is_mosaic_gpu_op(op: ir.OpView) -> bool:
  return op.name.startswith("mosaic_gpu.")


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


class DialectLoweringTest(DialectTest):

  def test_lowering_removes_mosaic_gpu_ops(self):
    with ir.InsertionPoint(self.module.body):
      mgpu.initialize_barrier(
          ir.MemRefType.get((1, 2), ir.Type.parse("!mosaic_gpu.barrier")),
          arrival_count=1)
    lower_mgpu_dialect(self.module)

    self.assertEmpty(
        list(filter(is_mosaic_gpu_op, self.module.body.operations)))

  def test_lowering_traverses_regions_correctly(self):
    with ir.InsertionPoint(self.module.body):
      bool_type = ir.IntegerType.get_signless(1)
      cst_true = arith.constant(bool_type, ir.IntegerAttr.get(bool_type, 1))
      if_op = scf.IfOp(cst_true)
      with ir.InsertionPoint(if_op.then_block):
        mgpu.initialize_barrier(
            ir.MemRefType.get((1, 2), ir.Type.parse("!mosaic_gpu.barrier")),
            arrival_count=1)
        scf.yield_([])
    lower_mgpu_dialect(self.module)

    self.assertEmpty(
        list(filter(is_mosaic_gpu_op, if_op.then_block.operations)))

  def test_initialize_barrier_op_lowering_rule(self):
    shape = (3, 4)
    num_shape_elements = shape[0] * shape[1]
    arrival_count = 1337

    with ir.InsertionPoint(self.module.body):
      mgpu.initialize_barrier(
          ir.MemRefType.get(shape, ir.Type.parse("!mosaic_gpu.barrier")),
          arrival_count=arrival_count)
    lower_mgpu_dialect(self.module)

    all_mbarrier_init_shared_ops = find_if(
        self.module,
        lambda op: op.name == nvvm.MBarrierInitSharedOp.OPERATION_NAME)

    # One nvvm.mbarrier_init_shared is issued per barrier.
    self.assertLen(all_mbarrier_init_shared_ops, num_shape_elements)

    # Each barrier has its count equal to the arrival count.
    for op in all_mbarrier_init_shared_ops:
      count = op.count.owner.opview
      self.assertIsInstance(count, arith.ConstantOp)
      self.assertEqual(count.literal_value, arrival_count)


if __name__ == "__main__":
  parameterized.absltest.main(testLoader=jtu.JaxTestLoader())
