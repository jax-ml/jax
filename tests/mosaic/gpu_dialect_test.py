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
from jax._src.lib.mlir.dialects import func
from jax._src.lib.mlir.dialects import gpu
from jax._src.lib.mlir.dialects import llvm
from jax._src.lib.mlir.dialects import memref
from jax._src.lib.mlir.dialects import nvvm
from jax._src.lib.mlir.dialects import scf
from jax.experimental.mosaic.gpu import dialect as mgpu  # pylint: disable=g-importing-member
from jax.experimental.mosaic.gpu import gpu_address_space_to_nvptx  # pylint: disable=g-importing-member,g-multiple-import
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


def find_if(
    module: ir.Module, predicate: Callable[[ir.OpView], bool]
) -> list[ir.OpView]:
  result = []

  def callback(op: ir.OpView):
    if predicate(op):
      result.append(op)

  for op in module.body.operations:
    walk_operations(op, callback)
  return result


def is_mosaic_gpu_op(op: ir.OpView) -> bool:
  return op.name.startswith("mosaic_gpu.")


def workgroup_ptr_ty() -> ir.Type:
  workgroup_nvptx_address_space = gpu_address_space_to_nvptx(
      gpu.AddressSpace.Workgroup
  )
  return ir.Type.parse(f"!llvm.ptr<{workgroup_nvptx_address_space}>")


class MosaicGpuTest(parameterized.TestCase):

  def setUp(self):
    if mgpu is None:
      raise self.skipTest("Test requires Mosaic GPU dialect")
    super().setUp()
    self.enter_context(_make_ir_context())
    self.enter_context(ir.Location.unknown())
    self.module = ir.Module.create()


class DialectTest(MosaicGpuTest):

  def test_dialect_module_is_loaded(self):
    self.assertTrue(_cext.globals._check_dialect_module_loaded("mosaic_gpu"))

  def test_initialize_barrier_op_result_memref_must_wrap_barriers(self):
    with ir.InsertionPoint(self.module.body):
      mgpu.initialize_barrier(
          ir.MemRefType.get((1, 2), ir.F32Type.get()),
          llvm.UndefOp(workgroup_ptr_ty()),
          arrival_count=1,
      )
    with self.assertRaisesRegex(
        ir.MLIRError, "must be memref of barrier values"
    ):
      self.module.operation.verify()

  def test_initialize_barrier_op_arrival_count_must_be_strictly_positive(self):
    with ir.InsertionPoint(self.module.body):
      mgpu.initialize_barrier(
          ir.MemRefType.get((1, 2), ir.Type.parse("!mosaic_gpu.barrier")),
          llvm.UndefOp(workgroup_ptr_ty()),
          arrival_count=0,
      )
    with self.assertRaisesRegex(ir.MLIRError, "value is positive"):
      self.module.operation.verify()

  def test_initialize_barrier_op_with_a_non_shared_base_pointer_fails(self):
    with ir.InsertionPoint(self.module.body):
      mgpu.initialize_barrier(
          ir.MemRefType.get((1, 2), ir.Type.parse("!mosaic_gpu.barrier")),
          llvm.UndefOp(ir.Type.parse(f"!llvm.ptr<{0}>")),
          arrival_count=1,
      )
    with self.assertRaisesRegex(ir.MLIRError, "pointer in address space 3"):
      self.module.operation.verify()

  def test_initialize_barrier_op_with_a_positive_arrival_count_passes(self):
    with ir.InsertionPoint(self.module.body):
      mgpu.initialize_barrier(
          ir.MemRefType.get((1, 2), ir.Type.parse("!mosaic_gpu.barrier")),
          llvm.UndefOp(workgroup_ptr_ty()),
          arrival_count=1,
      )
    self.assertTrue(self.module.operation.verify())
    self.assertIsInstance(
        self.module.body.operations[1], mgpu.InitializeBarrierOp
    )

  def test_async_load_op_dest_must_be_contiguous(self):
    with ir.InsertionPoint(self.module.body):
      func.FuncOp.from_py_func(
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.MemRefType.get(
              [4, 8],
              ir.F32Type.get(),
              layout=ir.Attribute.parse("strided<[16, 1]>"),
          ),
          ir.MemRefType.get([], ir.Type.parse("!mosaic_gpu.barrier")),
          ir.IntegerType.get_signless(32),
          ir.IntegerType.get_signless(32),
          name="async_load",
      )(
          lambda source, destination, barrier, *indices: mgpu.async_load(
              source,
              destination,
              barrier,
              indices,
              slice_lengths=[4, 8],
              transforms=ir.ArrayAttr.get([]),
              collective=ir.ArrayAttr.get([]),
          )
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The `destination` memref must be contiguous",
    ):
      self.module.operation.verify()

  def test_async_load_op_source_and_dest_must_have_same_element_type(self):
    with ir.InsertionPoint(self.module.body):
      func.FuncOp.from_py_func(
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.MemRefType.get([4, 8], ir.F64Type.get()),
          ir.MemRefType.get([], ir.Type.parse("!mosaic_gpu.barrier")),
          ir.IntegerType.get_signless(32),
          ir.IntegerType.get_signless(32),
          name="async_load",
      )(
          lambda source, destination, barrier, *indices: mgpu.async_load(
              source,
              destination,
              barrier,
              indices,
              slice_lengths=[4, 8],
              transforms=ir.ArrayAttr.get([]),
              collective=ir.ArrayAttr.get([]),
          )
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "`source` and `destination` memrefs must have the same element",
    ):
      self.module.operation.verify()

  def test_async_load_op_slice_lengths_must_be_larger_than_minus_two(self):
    with ir.InsertionPoint(self.module.body):
      func.FuncOp.from_py_func(
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.MemRefType.get([], ir.Type.parse("!mosaic_gpu.barrier")),
          ir.IntegerType.get_signless(32),
          ir.IntegerType.get_signless(32),
          name="async_load",
      )(
          lambda source, destination, barrier, *indices: mgpu.async_load(
              source,
              destination,
              barrier,
              indices,
              slice_lengths=[-2, 8],
              transforms=ir.ArrayAttr.get([]),
              collective=ir.ArrayAttr.get([]),
          )
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The `slice_lengths` attribute must not contain values less than -1",
    ):
      self.module.operation.verify()

  def test_async_load_op_source_and_dest_ranks_must_match_with_collapse(self):
    with ir.InsertionPoint(self.module.body):
      func.FuncOp.from_py_func(
          ir.MemRefType.get([1, 4, 8], ir.F32Type.get()),
          ir.MemRefType.get([4], ir.F32Type.get()),
          ir.MemRefType.get([], ir.Type.parse("!mosaic_gpu.barrier")),
          ir.IntegerType.get_signless(32),
          ir.IntegerType.get_signless(32),
          ir.IntegerType.get_signless(32),
          name="async_load",
      )(
          lambda source, destination, barrier, *indices: mgpu.async_load(
              source,
              destination,
              barrier,
              indices,
              slice_lengths=[-1, 4, 8],
              transforms=ir.ArrayAttr.get([]),
              collective=ir.ArrayAttr.get([]),
          )
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "`destination` plus the number of collapsed dimensions as indicated",
    ):
      self.module.operation.verify()

  def test_async_load_op_indices_size_must_match_source_rank(self):
    with ir.InsertionPoint(self.module.body):
      func.FuncOp.from_py_func(
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.MemRefType.get([], ir.Type.parse("!mosaic_gpu.barrier")),
          ir.IntegerType.get_signless(32),
          name="async_load",
      )(
          lambda source, destination, barrier, *indices: mgpu.async_load(
              source,
              destination,
              barrier,
              indices,
              slice_lengths=[4, 8],
              transforms=ir.ArrayAttr.get([]),
              collective=ir.ArrayAttr.get([]),
          )
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The size of `indices` must be equal to the rank of `source`",
    ):
      self.module.operation.verify()

  def test_async_load_op_slice_lengths_size_must_match_source_rank(self):
    with ir.InsertionPoint(self.module.body):
      func.FuncOp.from_py_func(
          ir.MemRefType.get([4], ir.F32Type.get()),
          ir.MemRefType.get([4], ir.F32Type.get()),
          ir.MemRefType.get([], ir.Type.parse("!mosaic_gpu.barrier")),
          ir.IntegerType.get_signless(32),
          name="async_load",
      )(
          lambda source, destination, barrier, *indices: mgpu.async_load(
              source,
              destination,
              barrier,
              indices,
              slice_lengths=[4, 8],
              transforms=ir.ArrayAttr.get([]),
              collective=ir.ArrayAttr.get([]),
          )
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The size of `slice_lengths` must be equal to the rank of `source`",
    ):
      self.module.operation.verify()

  def test_async_load_op_slice_collective_must_be_unique(self):
    with ir.InsertionPoint(self.module.body):
      func.FuncOp.from_py_func(
          ir.MemRefType.get([4], ir.F32Type.get()),
          ir.MemRefType.get([4], ir.F32Type.get()),
          ir.MemRefType.get([], ir.Type.parse("!mosaic_gpu.barrier")),
          ir.IntegerType.get_signless(32),
          name="async_load",
      )(
          lambda source, destination, barrier, *indices: mgpu.async_load(
              source,
              destination,
              barrier,
              indices,
              slice_lengths=[4],
              transforms=ir.ArrayAttr.get([]),
              collective=ir.ArrayAttr.get([
                  ir.Attribute.parse(
                      f"#mosaic_gpu.dim<{mgpu.Dimension.x.name}>"
                  ),
                  ir.Attribute.parse(
                      f"#mosaic_gpu.dim<{mgpu.Dimension.x.name}>"
                  ),
              ]),
          )
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The `collective` attribute must not contain duplicate dimensions",
    ):
      self.module.operation.verify()

  def test_async_store_op_source_must_be_contiguous(self):
    with ir.InsertionPoint(self.module.body):
      func.FuncOp.from_py_func(
          ir.MemRefType.get(
              [4, 8],
              ir.F32Type.get(),
              layout=ir.Attribute.parse("strided<[16, 1]>"),
          ),
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.IntegerType.get_signless(32),
          ir.IntegerType.get_signless(32),
          name="async_store",
      )(
          lambda source, destination, *indices: mgpu.async_store(
              source,
              destination,
              indices,
              slice_lengths=[4, 8],
              transforms=ir.ArrayAttr.get([]),
          )
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The `source` memref must be contiguous",
    ):
      self.module.operation.verify()

  def test_async_store_op_source_and_dest_must_have_same_element_type(self):
    with ir.InsertionPoint(self.module.body):
      func.FuncOp.from_py_func(
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.MemRefType.get([4, 8], ir.F64Type.get()),
          ir.IntegerType.get_signless(32),
          ir.IntegerType.get_signless(32),
          name="async_store",
      )(
          lambda source, destination, *indices: mgpu.async_store(
              source,
              destination,
              indices,
              slice_lengths=[4, 8],
              transforms=ir.ArrayAttr.get([]),
          )
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "`source` and `destination` memrefs must have the same element",
    ):
      self.module.operation.verify()

  def test_async_store_op_slice_lengths_must_be_larger_than_minus_two(self):
    with ir.InsertionPoint(self.module.body):
      func.FuncOp.from_py_func(
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.IntegerType.get_signless(32),
          ir.IntegerType.get_signless(32),
          name="async_store",
      )(
          lambda source, destination, *indices: mgpu.async_store(
              source,
              destination,
              indices,
              slice_lengths=[-2, 8],
              transforms=ir.ArrayAttr.get([]),
          )
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The `slice_lengths` attribute must not contain values less than -1",
    ):
      self.module.operation.verify()

  def test_async_store_op_source_and_dest_ranks_must_match_with_collapse(self):
    with ir.InsertionPoint(self.module.body):
      func.FuncOp.from_py_func(
          ir.MemRefType.get([4], ir.F32Type.get()),
          ir.MemRefType.get([1, 4, 8], ir.F32Type.get()),
          ir.IntegerType.get_signless(32),
          ir.IntegerType.get_signless(32),
          ir.IntegerType.get_signless(32),
          name="async_store",
      )(
          lambda source, destination, *indices: mgpu.async_store(
              source,
              destination,
              indices,
              slice_lengths=[-1, 4, 8],
              transforms=ir.ArrayAttr.get([]),
          )
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "`source` plus the number of collapsed dimensions as indicated",
    ):
      self.module.operation.verify()

  def test_async_store_op_indices_size_must_match_destination_rank(self):
    with ir.InsertionPoint(self.module.body):
      func.FuncOp.from_py_func(
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.IntegerType.get_signless(32),
          name="async_store",
      )(
          lambda source, destination, *indices: mgpu.async_store(
              source,
              destination,
              indices,
              slice_lengths=[4, 8],
              transforms=ir.ArrayAttr.get([]),
          )
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The size of `indices` must be equal to the rank of `destination`",
    ):
      self.module.operation.verify()

  def test_async_store_op_slice_lengths_size_must_match_source_rank(self):
    with ir.InsertionPoint(self.module.body):
      func.FuncOp.from_py_func(
          ir.MemRefType.get([4], ir.F32Type.get()),
          ir.MemRefType.get([4], ir.F32Type.get()),
          ir.IntegerType.get_signless(32),
          name="async_store",
      )(
          lambda source, destination, *indices: mgpu.async_store(
              source,
              destination,
              indices,
              slice_lengths=[4, 8],
              transforms=ir.ArrayAttr.get([]),
          )
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The size of `slice_lengths` must be equal to the rank of"
        " `destination`",
    ):
      self.module.operation.verify()

  def test_wgmma_types_match(self):
    with ir.InsertionPoint(self.module.body):
      func.FuncOp.from_py_func(
          ir.RankedTensorType.get([128, 160], ir.BF16Type.get()),
          ir.MemRefType.get([2, 4, 64, 32], ir.F16Type.get()),
          ir.MemRefType.get([4, 5, 32, 32], ir.BF16Type.get()),
          name="wgmma",
      )(
          lambda accumulator, a, b: mgpu.wgmma(
              accumulator,
              a,
              b,
              swizzle=ir.Attribute.parse("#mosaic_gpu.swizzle<swizzle_64>"),
          )
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The `a` and `b` inputs must have the same element type.",
    ):
      self.module.operation.verify()

  def test_wgmma_b_rank_is_4(self):
    with ir.InsertionPoint(self.module.body):
      func.FuncOp.from_py_func(
          ir.RankedTensorType.get([128, 160], ir.BF16Type.get()),
          ir.MemRefType.get([2, 4, 64, 32], ir.BF16Type.get()),
          ir.MemRefType.get([5, 32, 32], ir.BF16Type.get()),
          name="wgmma",
      )(
          lambda accumulator, a, b: mgpu.wgmma(
              accumulator,
              a,
              b,
              swizzle=ir.Attribute.parse("#mosaic_gpu.swizzle<swizzle_64>"),
          )
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The `b` input must have rank 4.",
    ):
      self.module.operation.verify()

  def test_wgmma_b_shape_dim_3(self):
    with ir.InsertionPoint(self.module.body):
      func.FuncOp.from_py_func(
          ir.RankedTensorType.get([128, 160], ir.BF16Type.get()),
          ir.MemRefType.get([2, 4, 64, 32], ir.BF16Type.get()),
          ir.MemRefType.get([4, 5, 32, 16], ir.BF16Type.get()),
          name="wgmma",
      )(
          lambda accumulator, a, b: mgpu.wgmma(
              accumulator,
              a,
              b,
              swizzle=ir.Attribute.parse("#mosaic_gpu.swizzle<swizzle_64>"),
          )
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"The n group size \(16\) must be equal to swizzle/element_bytewidth "
        r"\(32\)",
    ):
      self.module.operation.verify()

  def test_wgmma_b_shape_dim_2(self):
    with ir.InsertionPoint(self.module.body):
      func.FuncOp.from_py_func(
          ir.RankedTensorType.get([128, 160], ir.BF16Type.get()),
          ir.MemRefType.get([2, 4, 64, 32], ir.BF16Type.get()),
          ir.MemRefType.get([4, 5, 64, 32], ir.BF16Type.get()),
          name="wgmma",
      )(
          lambda accumulator, a, b: mgpu.wgmma(
              accumulator,
              a,
              b,
              swizzle=ir.Attribute.parse("#mosaic_gpu.swizzle<swizzle_64>"),
          )
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"The k group size \(64\) must be equal to swizzle/element_bytewidth "
        r"\(32\)",
    ):
      self.module.operation.verify()

  # TODO(b/381371456): Add tests for the other WGMMA inputs.


class DialectLoweringTest(MosaicGpuTest):

  def test_lowering_removes_mosaic_gpu_ops(self):
    with ir.InsertionPoint(self.module.body):
      mgpu.initialize_barrier(
          ir.MemRefType.get((1, 2), ir.Type.parse("!mosaic_gpu.barrier")),
          llvm.UndefOp(workgroup_ptr_ty()),
          arrival_count=1,
      )
    lower_mgpu_dialect(self.module)

    self.assertEmpty(
        list(filter(is_mosaic_gpu_op, self.module.body.operations))
    )

  def test_lowering_traverses_regions_correctly(self):
    with ir.InsertionPoint(self.module.body):
      bool_type = ir.IntegerType.get_signless(1)
      cst_true = arith.constant(bool_type, ir.IntegerAttr.get(bool_type, 1))
      if_op = scf.IfOp(cst_true)
      with ir.InsertionPoint(if_op.then_block):
        mgpu.initialize_barrier(
            ir.MemRefType.get((1, 2), ir.Type.parse("!mosaic_gpu.barrier")),
            llvm.UndefOp(workgroup_ptr_ty()),
            arrival_count=1,
        )
        scf.yield_([])
    lower_mgpu_dialect(self.module)

    self.assertEmpty(
        list(filter(is_mosaic_gpu_op, if_op.then_block.operations))
    )

  def test_initialize_barrier_op_lowering_rule(self):
    shape = (3, 4)
    num_shape_elements = shape[0] * shape[1]
    arrival_count = 1337

    with ir.InsertionPoint(self.module.body):
      barriers_ref = mgpu.initialize_barrier(
          ir.MemRefType.get(shape, ir.Type.parse("!mosaic_gpu.barrier")),
          llvm.UndefOp(workgroup_ptr_ty()),
          arrival_count=arrival_count,
      )
      # Add a user for barriers_ref to make sure that the lowering keeps types
      # consistent.
      memref.copy(barriers_ref, barriers_ref)

    self.assertTrue(self.module.operation.verify())
    lower_mgpu_dialect(self.module)
    self.assertTrue(self.module.operation.verify())

    all_mbarrier_init_shared_ops = find_if(
        self.module,
        lambda op: op.name == nvvm.MBarrierInitSharedOp.OPERATION_NAME,
    )

    # One nvvm.mbarrier_init_shared is issued per barrier.
    self.assertLen(all_mbarrier_init_shared_ops, num_shape_elements)

    # Each barrier has its count equal to the arrival count.
    for op in all_mbarrier_init_shared_ops:
      count = op.count.owner.opview
      self.assertIsInstance(count, arith.ConstantOp)
      self.assertEqual(count.literal_value, arrival_count)


if __name__ == "__main__":
  parameterized.absltest.main(testLoader=jtu.JaxTestLoader())
