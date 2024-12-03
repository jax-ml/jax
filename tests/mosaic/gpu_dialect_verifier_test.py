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
"""(Deviceless) tests for the verifies of the Mosaic GPU MLIR dialect."""

from absl.testing import parameterized
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func
from jax._src.lib.mlir.dialects import llvm
from jax.experimental.mosaic.gpu import dialect as mgpu  # pylint: disable=g-importing-member
from .gpu_dialect_test_util import DialectTest
from .gpu_dialect_test_util import workgroup_ptr_ty

config.parse_flags_with_absl()


class DialectVerifierTest(DialectTest):

  def test_initialize_barrier_op_result_memref_must_wrap_barriers(self):
    with ir.InsertionPoint(self.module.body):
      mgpu.initialize_barrier(
          ir.MemRefType.get((1, 2), ir.F32Type.get()),
          llvm.UndefOp(workgroup_ptr_ty()), arrival_count=1)
    with self.assertRaisesRegex(
        ir.MLIRError, "must be memref of barrier values"
    ):
      self.module.operation.verify()

  def test_initialize_barrier_op_arrival_count_must_be_strictly_positive(self):
    with ir.InsertionPoint(self.module.body):
      mgpu.initialize_barrier(
          ir.MemRefType.get((1, 2), ir.Type.parse("!mosaic_gpu.barrier")),
          llvm.UndefOp(workgroup_ptr_ty()),
          arrival_count=0)
    with self.assertRaisesRegex(ir.MLIRError, "value is positive"):
      self.module.operation.verify()

  def test_initialize_barrier_op_with_a_non_shared_base_pointer_fails(self):
    with ir.InsertionPoint(self.module.body):
      mgpu.initialize_barrier(
          ir.MemRefType.get((1, 2), ir.Type.parse("!mosaic_gpu.barrier")),
          llvm.UndefOp(ir.Type.parse(f"!llvm.ptr<{0}>")),
          arrival_count=1)
    with self.assertRaisesRegex(ir.MLIRError, "pointer in address space 3"):
      self.module.operation.verify()

  def test_initialize_barrier_op_with_a_positive_arrival_count_passes(self):
    with ir.InsertionPoint(self.module.body):
      mgpu.initialize_barrier(
          ir.MemRefType.get((1, 2), ir.Type.parse("!mosaic_gpu.barrier")),
          llvm.UndefOp(workgroup_ptr_ty()),
          arrival_count=1)
    self.assertTrue(self.module.operation.verify())
    self.assertIsInstance(self.module.body.operations[1],
                          mgpu.InitializeBarrierOp)

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


if __name__ == "__main__":
  parameterized.absltest.main(testLoader=jtu.JaxTestLoader())
