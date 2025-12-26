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

from collections.abc import Callable

from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
from jax._src.interpreters import mlir as mlir_interpreter
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import builtin
from jax._src.lib.mlir.dialects import gpu
from jax._src.lib.mlir.dialects import llvm
from jax._src.lib.mlir.dialects import memref
from jax._src.lib.mlir.dialects import nvvm
from jax._src.lib.mlir.dialects import scf
from jax._src.lib.mlir.dialects import vector
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.mosaic.gpu import dialect_lowering as lowering
from jax.experimental.mosaic.gpu import layouts
from jax.experimental.mosaic.gpu import tcgen05
from jax.experimental.mosaic.gpu import utils as mgpu_utils

_cext = mgpu.dialect._cext if mgpu.dialect is not None else None


config.parse_flags_with_absl()


def _make_ir_context():
  context = ir.Context()
  context.append_dialect_registry(mlir_interpreter.upstream_dialects)
  context.load_all_available_dialects()
  mgpu.dialect.register_dialect(context)
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
  workgroup_nvptx_address_space = mgpu_utils.gpu_address_space_to_nvptx(
      gpu.AddressSpace.Workgroup
  )
  return ir.Type.parse(f"!llvm.ptr<{workgroup_nvptx_address_space}>")


def undefs(*tys: ir.Type) -> list[ir.Value]:
  """Returns a list of undefined values of the given types."""
  # TODO(allanrenucci): Use `ub.poison` once Python bindings are available.
  return [builtin.unrealized_conversion_cast([ty], []) for ty in tys]


class MosaicGpuTest(parameterized.TestCase):

  def setUp(self):
    if jax.version._version != jax.lib.__version__:
      raise self.skipTest("Test requires matching jax and jaxlib versions")
    super().setUp()
    self.enter_context(_make_ir_context())
    self.enter_context(ir.Location.unknown())
    self.module = ir.Module.create()


class DialectTest(MosaicGpuTest):

  def test_dialect_module_is_loaded(self):
    self.assertTrue(_cext.globals._check_dialect_module_loaded("mosaic_gpu"))

  def test_initialize_barrier_op_arrival_count_must_be_strictly_positive(self):
    with ir.InsertionPoint(self.module.body):
      mgpu.dialect.initialize_barrier(
          llvm.UndefOp(workgroup_ptr_ty()),
          arrival_count=0,
          num_barriers=2,
      )
    with self.assertRaisesRegex(ir.MLIRError, "value is positive"):
      self.module.operation.verify()

  def test_initialize_barrier_op_with_a_non_shared_base_pointer_fails(self):
    with ir.InsertionPoint(self.module.body):
      mgpu.dialect.initialize_barrier(
          llvm.UndefOp(ir.Type.parse(f"!llvm.ptr<{0}>")),
          arrival_count=1,
          num_barriers=2,
      )
    with self.assertRaisesRegex(ir.MLIRError, "pointer in address space 3"):
      self.module.operation.verify()

  def test_initialize_barrier_op_with_a_positive_arrival_count_passes(self):
    with ir.InsertionPoint(self.module.body):
      mgpu.dialect.initialize_barrier(
          llvm.UndefOp(workgroup_ptr_ty()),
          arrival_count=1,
          num_barriers=2,
      )
    self.assertTrue(self.module.operation.verify())

  def test_async_load_op_dest_must_be_contiguous(self):
    with ir.InsertionPoint(self.module.body):
      source, destination, barrier, *indices = undefs(
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.MemRefType.get(
              [4, 8],
              ir.F32Type.get(),
              layout=ir.Attribute.parse("strided<[16, 1]>"),
          ),
          ir.MemRefType.get([], ir.Type.parse("!mosaic_gpu.barrier")),
          ir.IntegerType.get_signless(32),
          ir.IntegerType.get_signless(32),
      )
      mgpu.dialect.async_load(
          source,
          destination,
          barrier,
          indices,
          slice_lengths=[4, 8],
          collective=ir.ArrayAttr.get([]),
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The `destination` memref must be contiguous",
    ):
      self.module.operation.verify()

  def test_async_load_op_source_and_dest_must_have_same_element_type(self):
    with ir.InsertionPoint(self.module.body):
      source, destination, barrier, *indices = undefs(
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.MemRefType.get([4, 8], ir.F64Type.get()),
          ir.MemRefType.get([], ir.Type.parse("!mosaic_gpu.barrier")),
          ir.IntegerType.get_signless(32),
          ir.IntegerType.get_signless(32),
      )
      mgpu.dialect.async_load(
          source,
          destination,
          barrier,
          indices,
          slice_lengths=[4, 8],
          collective=ir.ArrayAttr.get([]),
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "`source` and `destination` memrefs must have the same element",
    ):
      self.module.operation.verify()

  def test_async_load_op_slice_lengths_must_be_larger_than_minus_two(self):
    with ir.InsertionPoint(self.module.body):
      source, destination, barrier, *indices = undefs(
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.MemRefType.get([], ir.Type.parse("!mosaic_gpu.barrier")),
          ir.IntegerType.get_signless(32),
          ir.IntegerType.get_signless(32),
      )
      mgpu.dialect.async_load(
          source,
          destination,
          barrier,
          indices,
          slice_lengths=[-2, 8],
          collective=ir.ArrayAttr.get([]),
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The `slice_lengths` attribute must not contain values less than -1",
    ):
      self.module.operation.verify()

  def test_async_load_op_source_and_dest_ranks_must_match_with_collapse(self):
    with ir.InsertionPoint(self.module.body):
      source, destination, barrier, *indices = undefs(
          ir.MemRefType.get([1, 4, 8], ir.F32Type.get()),
          ir.MemRefType.get([4], ir.F32Type.get()),
          ir.MemRefType.get([], ir.Type.parse("!mosaic_gpu.barrier")),
          ir.IntegerType.get_signless(32),
          ir.IntegerType.get_signless(32),
          ir.IntegerType.get_signless(32),
      )
      mgpu.dialect.async_load(
          source,
          destination,
          barrier,
          indices,
          slice_lengths=[-1, 4, 8],
          collective=ir.ArrayAttr.get([]),
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "`destination` plus the number of collapsed dimensions as indicated",
    ):
      self.module.operation.verify()

  def test_async_load_op_indices_size_must_match_source_rank(self):
    with ir.InsertionPoint(self.module.body):
      source, destination, barrier, *indices = undefs(
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.MemRefType.get([], ir.Type.parse("!mosaic_gpu.barrier")),
          ir.IntegerType.get_signless(32),
      )
      mgpu.dialect.async_load(
          source,
          destination,
          barrier,
          indices,
          slice_lengths=[4, 8],
          collective=ir.ArrayAttr.get([]),
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The size of `indices` must be equal to the rank of `source`",
    ):
      self.module.operation.verify()

  def test_async_load_op_slice_lengths_size_must_match_source_rank(self):
    with ir.InsertionPoint(self.module.body):
      source, destination, barrier, *indices = undefs(
          ir.MemRefType.get([4], ir.F32Type.get()),
          ir.MemRefType.get([4], ir.F32Type.get()),
          ir.MemRefType.get([], ir.Type.parse("!mosaic_gpu.barrier")),
          ir.IntegerType.get_signless(32),
      )
      mgpu.dialect.async_load(
          source,
          destination,
          barrier,
          indices,
          slice_lengths=[4, 8],
          collective=ir.ArrayAttr.get([]),
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The size of `slice_lengths` must be equal to the rank of `source`",
    ):
      self.module.operation.verify()

  def test_async_load_op_slice_collective_must_be_unique(self):
    with ir.InsertionPoint(self.module.body):
      i32 = ir.IntegerType.get_signless(32)
      source, destination, barrier, *indices = undefs(
          ir.MemRefType.get([4], ir.F32Type.get()),
          ir.MemRefType.get([4], ir.F32Type.get()),
          ir.MemRefType.get([], ir.Type.parse("!mosaic_gpu.barrier")),
          i32,
      )
      mgpu.dialect.async_load(
          source,
          destination,
          barrier,
          indices,
          slice_lengths=[4],
          collective=ir.ArrayAttr.get([
              ir.IntegerAttr.get(i32, mgpu.dialect.Dimension.x),
              ir.IntegerAttr.get(i32, mgpu.dialect.Dimension.x),
          ]),
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The `collective` attribute must not contain duplicate dimensions",
    ):
      self.module.operation.verify()

  def test_async_store_op_source_must_be_contiguous(self):
    with ir.InsertionPoint(self.module.body):
      source, destination, *indices = undefs(
          ir.MemRefType.get(
              [4, 8],
              ir.F32Type.get(),
              layout=ir.Attribute.parse("strided<[16, 1]>"),
          ),
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.IntegerType.get_signless(32),
          ir.IntegerType.get_signless(32),
      )
      mgpu.dialect.async_store(
          source,
          destination,
          indices,
          slice_lengths=[4, 8],
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The `source` memref must be contiguous",
    ):
      self.module.operation.verify()

  def test_async_store_op_source_and_dest_must_have_same_element_type(self):
    with ir.InsertionPoint(self.module.body):
      source, destination, *indices = undefs(
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.MemRefType.get([4, 8], ir.F64Type.get()),
          ir.IntegerType.get_signless(32),
          ir.IntegerType.get_signless(32),
      )
      mgpu.dialect.async_store(
          source,
          destination,
          indices,
          slice_lengths=[4, 8],
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "`source` and `destination` memrefs must have the same element",
    ):
      self.module.operation.verify()

  def test_async_store_op_slice_lengths_must_be_larger_than_minus_two(self):
    with ir.InsertionPoint(self.module.body):
      source, destination, *indices = undefs(
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.IntegerType.get_signless(32),
          ir.IntegerType.get_signless(32),
      )
      mgpu.dialect.async_store(
          source,
          destination,
          indices,
          slice_lengths=[-2, 8],
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The `slice_lengths` attribute must not contain values less than -1",
    ):
      self.module.operation.verify()

  def test_async_store_op_source_and_dest_ranks_must_match_with_collapse(self):
    with ir.InsertionPoint(self.module.body):
      source, destination, *indices = undefs(
          ir.MemRefType.get([4], ir.F32Type.get()),
          ir.MemRefType.get([1, 4, 8], ir.F32Type.get()),
          ir.IntegerType.get_signless(32),
          ir.IntegerType.get_signless(32),
          ir.IntegerType.get_signless(32),
      )
      mgpu.dialect.async_store(
          source,
          destination,
          indices,
          slice_lengths=[-1, 4, 8],
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "`source` plus the number of collapsed dimensions as indicated",
    ):
      self.module.operation.verify()

  def test_async_store_op_indices_size_must_match_destination_rank(self):
    with ir.InsertionPoint(self.module.body):
      source, destination, *indices = undefs(
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.MemRefType.get([4, 8], ir.F32Type.get()),
          ir.IntegerType.get_signless(32),
      )
      mgpu.dialect.async_store(
          source,
          destination,
          indices,
          slice_lengths=[4, 8],
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The size of `indices` must be equal to the rank of `destination`",
    ):
      self.module.operation.verify()

  def test_async_store_op_slice_lengths_size_must_match_source_rank(self):
    with ir.InsertionPoint(self.module.body):
      source, destination, *indices = undefs(
          ir.MemRefType.get([4], ir.F32Type.get()),
          ir.MemRefType.get([4], ir.F32Type.get()),
          ir.IntegerType.get_signless(32),
      )
      mgpu.dialect.async_store(
          source,
          destination,
          indices,
          slice_lengths=[4, 8],
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The size of `slice_lengths` must be equal to the rank of"
        " `destination`",
    ):
      self.module.operation.verify()

  def test_wgmma_types_match(self):
    with ir.InsertionPoint(self.module.body):
      acc, a, b = undefs(
          ir.VectorType.get([128, 160], ir.F32Type.get()),
          ir.MemRefType.get([128, 128], ir.F16Type.get()),
          ir.MemRefType.get([128, 160], ir.BF16Type.get()),
      )
      mgpu.dialect.wgmma(acc, a, b)

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The `a` and `b` inputs must have the same element type.",
    ):
      self.module.operation.verify()

  def test_wgmma_acc_m_dim_not_multiple_of_64(self):
    with ir.InsertionPoint(self.module.body):
      acc, a, b = undefs(
          ir.VectorType.get([127, 160], ir.F32Type.get()),
          ir.MemRefType.get([127, 128], ir.BF16Type.get()),
          ir.MemRefType.get([128, 160], ir.BF16Type.get()),
      )
      mgpu.dialect.wgmma(acc, a, b)

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"accumulator.*must be a multiple of 64",
    ):
      self.module.operation.verify()

  def test_wgmma_acc_m_not_equal_to_a_m_dim(self):
    with ir.InsertionPoint(self.module.body):
      acc, a, b = undefs(
          ir.VectorType.get([256, 160], ir.F32Type.get()),
          ir.MemRefType.get([512, 128], ir.BF16Type.get()),
          ir.MemRefType.get([128, 160], ir.BF16Type.get()),
      )
      mgpu.dialect.wgmma(acc, a, b)

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"accumulator's first dimension 256 must be equal to.*`a`",
    ):
      self.module.operation.verify()

  def test_wgmma_a_k_dim_not_equal_to_b_k_dim(self):
    with ir.InsertionPoint(self.module.body):
      acc, a, b = undefs(
          ir.VectorType.get([128, 160], ir.F32Type.get()),
          ir.MemRefType.get([128, 128], ir.BF16Type.get()),
          ir.MemRefType.get([160, 160], ir.BF16Type.get()),
      )
      mgpu.dialect.wgmma(acc, a, b)

    with self.assertRaisesRegex(
        ir.MLIRError,
        "`a`'s contracting dimension 128 must be equal to the first dimension"
        " of `b`",
    ):
      self.module.operation.verify()

  def test_wgmma_b_n_dim_not_equal_to_acc_n_dim(self):
    with ir.InsertionPoint(self.module.body):
      acc, a, b = undefs(
          ir.VectorType.get([128, 160], ir.F32Type.get()),
          ir.MemRefType.get([128, 128], ir.BF16Type.get()),
          ir.MemRefType.get([128, 192], ir.BF16Type.get()),
      )
      mgpu.dialect.wgmma(acc, a, b)

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"`b`'s non-contracting dimension 192 must be equal to the",
    ):
      self.module.operation.verify()

  def test_tcgen05_mma_types_match(self):
    with ir.InsertionPoint(self.module.body):
      acc, a, b, accumulate = undefs(
          ir.MemRefType.get([128, 160], ir.F16Type.get()),
          ir.MemRefType.get([128, 128], ir.F16Type.get()),
          ir.MemRefType.get([128, 160], ir.BF16Type.get()),
          ir.IntegerType.get_signless(1),
      )
      mgpu.dialect.tcgen05_mma(acc, a, b, accumulate)

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The `a` and `b` inputs must have the same element type.",
    ):
      self.module.operation.verify()

  def test_tcgen05_mma_acc_m_dim_not_multiple_of_128(self):
    with ir.InsertionPoint(self.module.body):
      acc, a, b, accumulate = undefs(
          ir.MemRefType.get([127, 160], ir.F16Type.get()),
          ir.MemRefType.get([127, 128], ir.F16Type.get()),
          ir.MemRefType.get([128, 160], ir.F16Type.get()),
          ir.IntegerType.get_signless(1),
      )
      mgpu.dialect.tcgen05_mma(acc, a, b, accumulate)

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"accumulator.*must be a multiple of 32",
    ):
      self.module.operation.verify()

  def test_tcgen05_mma_acc_m_not_equal_to_a_m_dim(self):
    with ir.InsertionPoint(self.module.body):
      acc, a, b, accumulate = undefs(
          ir.MemRefType.get([256, 160], ir.F16Type.get()),
          ir.MemRefType.get([512, 128], ir.F16Type.get()),
          ir.MemRefType.get([128, 160], ir.F16Type.get()),
          ir.IntegerType.get_signless(1),
      )
      mgpu.dialect.tcgen05_mma(acc, a, b, accumulate)

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"accumulator's first dimension 256 must be equal to.*`a`",
    ):
      self.module.operation.verify()

  def test_tcgen05_mma_a_k_dim_not_equal_to_b_k_dim(self):
    with ir.InsertionPoint(self.module.body):
      acc, a, b, accumulate = undefs(
          ir.MemRefType.get([128, 160], ir.F16Type.get()),
          ir.MemRefType.get([128, 128], ir.F16Type.get()),
          ir.MemRefType.get([160, 160], ir.F16Type.get()),
          ir.IntegerType.get_signless(1),
      )
      mgpu.dialect.tcgen05_mma(acc, a, b, accumulate)

    with self.assertRaisesRegex(
        ir.MLIRError,
        "`a`'s contracting dimension 128 must be equal to the first dimension"
        " of `b`",
    ):
      self.module.operation.verify()

  def test_tcgen05_mma_b_n_dim_not_equal_to_acc_n_dim(self):
    with ir.InsertionPoint(self.module.body):
      acc, a, b, accumulate = undefs(
          ir.MemRefType.get([128, 160], ir.F16Type.get()),
          ir.MemRefType.get([128, 128], ir.F16Type.get()),
          ir.MemRefType.get([128, 192], ir.F16Type.get()),
          ir.IntegerType.get_signless(1),
      )
      mgpu.dialect.tcgen05_mma(acc, a, b, accumulate)

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"`b`'s non-contracting dimension 192 must be equal to the",
    ):
      self.module.operation.verify()

  def test_tcgen05_mma_b_n_dim_not_equal_to_half_acc_n_dim(self):
    with ir.InsertionPoint(self.module.body):
      acc, a, b, accumulate = undefs(
          ir.MemRefType.get([128, 160], ir.F16Type.get()),
          ir.MemRefType.get([128, 128], ir.F16Type.get()),
          ir.MemRefType.get([128, 160], ir.F16Type.get()),
          ir.IntegerType.get_signless(1),
      )
      mgpu.dialect.tcgen05_mma(acc, a, b, accumulate, collective=True)

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"`b`'s non-contracting dimension 160 must be half",
    ):
      self.module.operation.verify()

  def test_tcgen05_mma_acc_mem_space_is_tmem(self):
    smem = mgpu_utils.smem()
    with ir.InsertionPoint(self.module.body):
      acc, a, b, accumulate = undefs(
          ir.MemRefType.get([128, 160], ir.F16Type.get(), memory_space=smem),
          ir.MemRefType.get([128, 128], ir.F16Type.get()),
          ir.MemRefType.get([128, 160], ir.F16Type.get()),
          ir.IntegerType.get_signless(1),
      )
      mgpu.dialect.tcgen05_mma(acc, a, b, accumulate)

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"The accumulator must be in TMEM",
    ):
      self.module.operation.verify()

  def test_tcgen05_mma_a_mem_space_is_smem_or_tmem(self):
    tmem = mgpu_utils.tmem()
    with ir.InsertionPoint(self.module.body):
      acc, a, b, accumulate = undefs(
          ir.MemRefType.get([128, 160], ir.F16Type.get(), memory_space=tmem),
          ir.MemRefType.get([128, 128], ir.F16Type.get()),
          ir.MemRefType.get([128, 160], ir.F16Type.get()),
          ir.IntegerType.get_signless(1),
      )
      mgpu.dialect.tcgen05_mma(acc, a, b, accumulate)

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"The `a` input must be in TMEM or SMEM",
    ):
      self.module.operation.verify()

  def test_tcgen05_mma_b_mem_space_is_smem(self):
    smem, tmem = mgpu_utils.smem(), mgpu_utils.tmem()
    with ir.InsertionPoint(self.module.body):
      acc, a, b, accumulate = undefs(
          ir.MemRefType.get([128, 160], ir.F16Type.get(), memory_space=tmem),
          ir.MemRefType.get([128, 128], ir.F16Type.get(), memory_space=smem),
          ir.MemRefType.get([128, 160], ir.F16Type.get(), memory_space=tmem),
          ir.IntegerType.get_signless(1),
      )
      mgpu.dialect.tcgen05_mma(acc, a, b, accumulate)

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"The `b` input must be in SMEM",
    ):
      self.module.operation.verify()

  def test_tcgen05_mma_scale_arg_missing(self):
    smem, tmem = mgpu_utils.smem(), mgpu_utils.tmem()
    f8e0m0 = ir.Float8E8M0FNUType.get()
    with ir.InsertionPoint(self.module.body):
      acc, a, b, accumulate, a_scale = undefs(
          ir.MemRefType.get([128, 160], ir.F16Type.get(), memory_space=tmem),
          ir.MemRefType.get([128, 128], ir.F16Type.get(), memory_space=smem),
          ir.MemRefType.get([128, 160], ir.F16Type.get(), memory_space=smem),
          ir.IntegerType.get_signless(1),
          ir.MemRefType.get([128, 4], f8e0m0, memory_space=tmem),
      )
      mgpu.dialect.tcgen05_mma(acc, a, b, accumulate, a_scale=a_scale)

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"Either none or both scales should be provided.",
    ):
      self.module.operation.verify()

  def test_tcgen05_mma_a_scale_mem_space_is_tmem(self):
    smem, tmem = mgpu_utils.smem(), mgpu_utils.tmem()
    f8e0m0 = ir.Float8E8M0FNUType.get()
    with ir.InsertionPoint(self.module.body):
      acc, a, b, accumulate, a_scale, b_scale = undefs(
          ir.MemRefType.get([128, 160], ir.F16Type.get(), memory_space=tmem),
          ir.MemRefType.get([128, 128], ir.F16Type.get(), memory_space=smem),
          ir.MemRefType.get([128, 160], ir.F16Type.get(), memory_space=smem),
          ir.IntegerType.get_signless(1),
          ir.MemRefType.get([128, 4], f8e0m0, memory_space=smem),
          ir.MemRefType.get([160, 4], f8e0m0, memory_space=tmem),
      )
      mgpu.dialect.tcgen05_mma(
          acc, a, b, accumulate, a_scale=a_scale, b_scale=b_scale
      )

    with self.assertRaisesRegex(
ir.MLIRError,
        r"The `a_scale` input must be in TMEM",
    ):
      self.module.operation.verify()

  def test_tcgen05_mma_b_scale_mem_space_is_tmem(self):
    smem, tmem = mgpu_utils.smem(), mgpu_utils.tmem()
    f8e0m0 = ir.Float8E8M0FNUType.get()
    with ir.InsertionPoint(self.module.body):
      acc, a, b, accumulate, a_scale, b_scale = undefs(
          ir.MemRefType.get([128, 160], ir.F16Type.get(), memory_space=tmem),
          ir.MemRefType.get([128, 128], ir.F16Type.get(), memory_space=smem),
          ir.MemRefType.get([128, 160], ir.F16Type.get(), memory_space=smem),
          ir.IntegerType.get_signless(1),
          ir.MemRefType.get([128, 4], f8e0m0, memory_space=tmem),
          ir.MemRefType.get([160, 4], f8e0m0, memory_space=smem),
      )
      mgpu.dialect.tcgen05_mma(
          acc, a, b, accumulate, a_scale=a_scale, b_scale=b_scale
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"The `b_scale` input must be in TMEM",
    ):
      self.module.operation.verify()

  def test_tiled_layout_attr_parsing(self):
    with ir.InsertionPoint(self.module.body):
      for layout in (
          mgpu.WGMMA_LAYOUT,
          mgpu.WGMMA_ROW_LAYOUT,
          mgpu.WGMMA_COL_LAYOUT,
          mgpu.WGMMA_TRANSPOSED_LAYOUT,
      ):
        attr = layouts.to_tiled_layout_attr(layout)
        parsed_layout = layouts.from_tiled_layout_attr(attr)
        self.assertEqual(layout, parsed_layout)

  def test_broadcast_in_dim_ok(self):
    with ir.InsertionPoint(self.module.body):
      (operand,) = undefs(ir.VectorType.get([64], ir.F32Type.get()))
      mgpu.dialect.broadcast_in_dim(
          ir.VectorType.get([64, 64], ir.F32Type.get()),
          operand,
          broadcast_dimensions=[0],
      )

    self.assertTrue(self.module.operation.verify())

  def test_broadcast_in_dim_no_0d(self):
    with ir.InsertionPoint(self.module.body):
      (operand,) = undefs(ir.VectorType.get([], ir.F32Type.get()))
      mgpu.dialect.broadcast_in_dim(
          ir.VectorType.get([64], ir.F32Type.get()),
          operand,
          broadcast_dimensions=[],
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"The input vector must have rank > 0",
    ):
      self.module.operation.verify()

  def test_broadcast_in_dim_no_input_larger_than_output(self):
    with ir.InsertionPoint(self.module.body):
      (operand,) = undefs(ir.VectorType.get([64, 64], ir.F32Type.get()))
      mgpu.dialect.broadcast_in_dim(
          ir.VectorType.get([64], ir.F32Type.get()),
          operand,
          broadcast_dimensions=[],
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"rank of the input vector must be smaller",
    ):
      self.module.operation.verify()

  def test_broadcast_in_dim_too_many_dims(self):
    with ir.InsertionPoint(self.module.body):
      (operand,) = undefs(ir.VectorType.get([64], ir.F32Type.get()))
      mgpu.dialect.broadcast_in_dim(
          ir.VectorType.get([64, 64], ir.F32Type.get()),
          operand,
          broadcast_dimensions=[0, 1],
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"size of the `broadcast_dimensions` attribute must be",
    ):
      self.module.operation.verify()

  def test_broadcast_in_dim_dim_oob(self):
    with ir.InsertionPoint(self.module.body):
      (operand,) = undefs(ir.VectorType.get([64], ir.F32Type.get()))
      mgpu.dialect.broadcast_in_dim(
          ir.VectorType.get([64, 64], ir.F32Type.get()),
          operand,
          broadcast_dimensions=[2],
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"must be in the range \[0, result.shape.rank",
    ):
      self.module.operation.verify()

  def test_broadcast_in_dim_dim_transpose(self):
    with ir.InsertionPoint(self.module.body):
      (operand,) = undefs(ir.VectorType.get([64, 64, 64, 64], ir.F32Type.get()))
      mgpu.dialect.broadcast_in_dim(
          ir.VectorType.get([64, 64, 64, 64], ir.F32Type.get()),
          operand,
          broadcast_dimensions=[0, 1, 3, 2],
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"`broadcast_dimensions` attribute must be strictly increasing",
    ):
      self.module.operation.verify()

  def test_custom_primitive_op_args_must_match_args_of_terminator(self):
    with ir.InsertionPoint(self.module.body):
      shape = (128,)
      elt_ty = ir.F32Type.get()
      ty = ir.VectorType.get(shape, elt_ty)
      strided_layout = mgpu.WGStridedFragLayout.from_shaped_type(ty)
      assert strided_layout is not None
      out_layouts = ir.ArrayAttr.get([layouts.to_layout_attr(strided_layout)])

      op = mgpu.dialect.CustomPrimitiveOp(
          result=[ty],
          operands_=[],
          in_layouts=[],
          in_transforms=[],
          out_layouts=out_layouts,
      )
      block = op.body.blocks.append()
      with ir.InsertionPoint(block):
        v = llvm.mlir_undef(ir.VectorType.get([256], ir.F32Type.get()))
        mgpu.dialect.ReturnOp(operands_=[v])

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"type of return operand 0 \('vector<256xf32>'\) doesn't match the"
        r" result type \('vector<128xf32>'\) in custom_primitive",
    ):
      self.module.operation.verify()

  def test_custom_primitive_op_results_must_be_scalar_or_vector(self):
    with ir.InsertionPoint(self.module.body):
      ref_ty = ir.MemRefType.get((128, 128), ir.F32Type.get())
      op = mgpu.dialect.CustomPrimitiveOp(
          result=[ref_ty],
          operands_=[],
          in_layouts=[],
          in_transforms=[],
          out_layouts=[],
      )
      block = op.body.blocks.append()
      with ir.InsertionPoint(block):
        [ref] = undefs(ref_ty)
        mgpu.dialect.ReturnOp(operands_=[ref])

    with self.assertRaisesRegex(
        ir.MLIRError,
        r"Custom primitive can only return scalars or vectors.",
    ):
      self.module.operation.verify()

  def test_tmem_alloc_op_must_have_smem_ref_input(self):
    with ir.InsertionPoint(self.module.body):
      (smem_ptr,) = undefs(
          ir.MemRefType.get([], ir.IntegerType.get_signless(32))
      )
      mgpu.dialect.tmem_alloc(
          result=ir.MemRefType.get(
              [128, 32],
              ir.BF16Type.get(),
              memory_space=mgpu_utils.tmem(),
          ),
          smem_ptr=smem_ptr,
          collective=False,
          packing=1,
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The `smem_ptr` memref must have the Workgroup address space",
    ):
      self.module.operation.verify()

  def test_tmem_alloc_op_result_must_have_tmem_memory_space(self):
    with ir.InsertionPoint(self.module.body):
      (smem_ptr,) = undefs(
          ir.MemRefType.get(
              [],
              ir.IntegerType.get_signless(32),
              memory_space=mgpu_utils.smem(),
          )
      )
      mgpu.dialect.tmem_alloc(
          result=ir.MemRefType.get(
              [128, 32],
              ir.BF16Type.get(),
          ),
          smem_ptr=smem_ptr,
          collective=False,
          packing=1,
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The tmem memref must have a mosaic_gpu.tmem memory space",
    ):
      self.module.operation.verify()

  def test_tmem_alloc_op_exact_column_count_must_be_at_most_512(self):
    with ir.InsertionPoint(self.module.body):
      (smem_ptr,) = undefs(
          ir.MemRefType.get(
              [],
              ir.IntegerType.get_signless(32),
              memory_space=mgpu_utils.smem(),
          )
      )
      mgpu.dialect.tmem_alloc(
          result=ir.MemRefType.get(
              [128, 1024],
              ir.BF16Type.get(),
              memory_space=mgpu_utils.tmem(),
          ),
          smem_ptr=smem_ptr,
          collective=False,
          packing=1,
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The number of allocated columns must be less than or equal to 512 but"
        " got: 1024",
    ):
      self.module.operation.verify()

  def test_tmem_alloc_op_bad_packing(self):
    with ir.InsertionPoint(self.module.body):
      (smem_ptr,) = undefs(
          ir.MemRefType.get(
              [],
              ir.IntegerType.get_signless(32),
              memory_space=mgpu_utils.smem(),
          )
      )
      mgpu.dialect.tmem_alloc(
          result=ir.MemRefType.get(
              [128, 128],
              ir.BF16Type.get(),
              memory_space=mgpu_utils.tmem(),
          ),
          smem_ptr=smem_ptr,
          collective=False,
          packing=4,
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "Only unpacked, or fully packed allocations are supported.",
    ):
      self.module.operation.verify()

  def test_tmem_alloc_op_exact_false_column_count_15_ok(self):
    with ir.InsertionPoint(self.module.body):
      (smem_ptr,) = undefs(
          ir.MemRefType.get(
              [],
              ir.IntegerType.get_signless(32),
              memory_space=mgpu_utils.smem(),
          )
      )
      mgpu.dialect.tmem_alloc(
          result=ir.MemRefType.get(
              [128, 15],
              ir.BF16Type.get(),
              memory_space=mgpu_utils.tmem(),
          ),
          smem_ptr=smem_ptr,
          collective=False,
          packing=1,
      )

    self.assertTrue(self.module.operation.verify())

  def test_tmem_alloc_op_exact_false_column_count_100_ok(self):
    with ir.InsertionPoint(self.module.body):
      (smem_ptr,) = undefs(
          ir.MemRefType.get(
              [],
              ir.IntegerType.get_signless(32),
              memory_space=mgpu_utils.smem(),
          )
      )
      mgpu.dialect.tmem_alloc(
          result=ir.MemRefType.get(
              [128, 100],
              ir.BF16Type.get(),
              memory_space=mgpu_utils.tmem(),
          ),
          smem_ptr=smem_ptr,
          collective=False,
          packing=1,
      )

    self.assertTrue(self.module.operation.verify())

  def test_tmem_alloc_op_exact_false_column_count_777_packed_not_ok(self):
    with ir.InsertionPoint(self.module.body):
      (smem_ptr,) = undefs(
          ir.MemRefType.get(
              [],
              ir.IntegerType.get_signless(32),
              memory_space=mgpu_utils.smem(),
          )
      )
      mgpu.dialect.tmem_alloc(
          result=ir.MemRefType.get(
              [128, 777],
              ir.BF16Type.get(),
              memory_space=mgpu_utils.tmem(),
          ),
          smem_ptr=smem_ptr,
          collective=False,
          packing=2,
      )

    with self.assertRaisesRegex(
        ir.MLIRError,
        "The number of unpacked columns must be divisible by the packing",
    ):
      self.module.operation.verify()

  def test_tmem_alloc_op_exact_false_column_count_778_packed_ok(self):
    with ir.InsertionPoint(self.module.body):
      (smem_ptr,) = undefs(
          ir.MemRefType.get(
              [],
              ir.IntegerType.get_signless(32),
              memory_space=mgpu_utils.smem(),
          )
      )
      mgpu.dialect.tmem_alloc(
          result=ir.MemRefType.get(
              [128, 778],
              ir.BF16Type.get(),
              memory_space=mgpu_utils.tmem(),
          ),
          smem_ptr=smem_ptr,
          collective=False,
          packing=2,
      )

    self.assertTrue(self.module.operation.verify())

  def test_tmem_alloc_dealloc_packed_large_shape_ok(self):
    with ir.InsertionPoint(self.module.body):
      ref_ty = ir.MemRefType.get(
          [128, 1024],
          ir.BF16Type.get(),
          memory_space=mgpu_utils.tmem(),
      )
      (smem_ptr,) = undefs(
          ir.MemRefType.get(
              [],
              ir.IntegerType.get_signless(32),
              memory_space=mgpu_utils.smem(),
          )
      )
      # This allocation would exceed the 512 columns limit if it were not packed.
      ref = mgpu.dialect.tmem_alloc(
          result=ref_ty,
          smem_ptr=smem_ptr,
          collective=False,
          packing=2,
      )
      mgpu.dialect.tmem_dealloc(ref)
    self.assertTrue(self.module.operation.verify())

  def test_tmem_layout_cast_invalid_tmem_ref(self):
    with ir.InsertionPoint(self.module.body):
      (tmem_ref,) = undefs(
          ir.MemRefType.get(
              [128, 128],
              ir.BF16Type.get(),
              memory_space=mgpu_utils.smem(),
          )
      )
      mgpu.dialect.tmem_layout_cast(
          tmem_ref, layouts.to_layout_attr(tcgen05.TMEM_NATIVE_LAYOUT)
      )
    with self.assertRaisesRegex(
        ir.MLIRError,
        "The tmem memref must have a mosaic_gpu.tmem memory space",
    ):
      self.module.operation.verify()

  def test_vector_store_op_src_dst_shape_mismatch(self):
    with ir.InsertionPoint(self.module.body):
      src_ty = ir.VectorType.get((8,), ir.BF16Type.get())
      dst_ty = ir.MemRefType.get((4,), ir.BF16Type.get())
      (src, dst) = undefs(src_ty, dst_ty)
      mgpu.dialect.vector_store(src, dst)
    with self.assertRaisesRegex(
        ir.MLIRError,
        "The source and destination must have the same shape",
    ):
      self.module.operation.verify()

  def test_vector_store_op_src_dst_dtype_mismatch(self):
    with ir.InsertionPoint(self.module.body):
      src_ty = ir.VectorType.get((8,), ir.BF16Type.get())
      dst_ty = ir.MemRefType.get((8,), ir.F32Type.get())
      (src, dst) = undefs(src_ty, dst_ty)
      mgpu.dialect.vector_store(src, dst)
    with self.assertRaisesRegex(
        ir.MLIRError,
        "The source and destination must have the same element type",
    ):
      self.module.operation.verify()

  def test_broadcasted_iota_op_invalid_dimension(self):
    with ir.InsertionPoint(self.module.body):
      ty = ir.VectorType.get((2,), ir.F32Type.get())
      mgpu.dialect.broadcasted_iota(ty, dimension=2)
    with self.assertRaisesRegex(
        ir.MLIRError,
        "dimension=2 must be smaller than the rank=1 of the result.",
    ):
      self.module.operation.verify()

  def test_print_layout_op_invalid_ref(self):
    with ir.InsertionPoint(self.module.body):
      ref_ty = ir.MemRefType.get(
          (2,), ir.F32Type.get(), memory_space=mgpu_utils.smem()
      )
      (ref,) = undefs(ref_ty)
      mgpu.dialect.print_layout("tmem: {}", ref)
    with self.assertRaisesRegex(
        ir.MLIRError,
        "The tmem memref must have a mosaic_gpu.tmem memory space",
    ):
      self.module.operation.verify()


class DialectLoweringTest(MosaicGpuTest):

  def test_lowering_removes_mosaic_gpu_ops(self):
    with ir.InsertionPoint(self.module.body):
      mgpu.dialect.initialize_barrier(
          llvm.UndefOp(workgroup_ptr_ty()),
          arrival_count=1,
          num_barriers=2,
      )
    mgpu.lower_mgpu_dialect(self.module, None)

    self.assertEmpty(
        list(filter(is_mosaic_gpu_op, self.module.body.operations))
    )

  def test_lowering_traverses_regions_correctly(self):
    with ir.InsertionPoint(self.module.body):
      bool_type = ir.IntegerType.get_signless(1)
      cst_true = arith.constant(bool_type, ir.IntegerAttr.get(bool_type, 1))
      if_op = scf.IfOp(cst_true)
      with ir.InsertionPoint(if_op.then_block):
        mgpu.dialect.initialize_barrier(
            llvm.UndefOp(workgroup_ptr_ty()),
            arrival_count=1,
            num_barriers=2,
        )
        scf.yield_([])
    mgpu.lower_mgpu_dialect(self.module, None)

    self.assertEmpty(
        list(filter(is_mosaic_gpu_op, if_op.then_block.operations))
    )

  def test_initialize_barrier_op_lowering_rule(self):
    num_barriers = 4
    arrival_count = 1337

    with ir.InsertionPoint(self.module.body):
      mgpu.dialect.initialize_barrier(
          llvm.UndefOp(workgroup_ptr_ty()),
          arrival_count=arrival_count,
          num_barriers=num_barriers,
      )

    self.assertTrue(self.module.operation.verify())
    mgpu.lower_mgpu_dialect(self.module, None)
    self.assertTrue(self.module.operation.verify())

    all_mbarrier_init_ops = find_if(
        self.module,
        lambda op: op.name == nvvm.MBarrierInitOp.OPERATION_NAME,
    )

    # One nvvm.mbarrier_init_shared is issued per barrier.
    self.assertLen(all_mbarrier_init_ops, num_barriers)

    # Each barrier has its count equal to the arrival count times the
    # warpgroup size.
    for op in all_mbarrier_init_ops:
      count = op.count.owner
      self.assertIsInstance(count, arith.ConstantOp)
      self.assertEqual(
          count.literal_value, arrival_count * mgpu_utils.WARPGROUP_SIZE
      )

  def test_lowering_vector_op_without_layout_fails(self):
    shape = (3, 4)
    elt_ty = ir.BF16Type.get()
    with ir.InsertionPoint(self.module.body):
      ref = llvm.mlir_undef(ir.MemRefType.get(shape, elt_ty))
      mgpu.dialect.vector_load(ref)
    with self.assertRaisesRegex(
        ValueError, "missing a layout and can not be lowered"
    ):
      mgpu.lower_mgpu_dialect(self.module, None)

  def test_lowering_eliminates_layouts(self):
    shape = (4, 128)
    elt_ty = ir.BF16Type.get()
    with ir.InsertionPoint(self.module.body):
      ref = llvm.mlir_undef(ir.MemRefType.get(shape, elt_ty))
      load = mgpu.dialect.vector_load(ref)
      strided_layout = mgpu.WGStridedFragLayout.from_shaped_type(load.type)
      assert strided_layout is not None
      load.owner.attributes["out_layouts"] = ir.ArrayAttr.get([
          layouts.to_layout_attr(strided_layout)
      ])

    mgpu.lower_mgpu_dialect(self.module, None)

    all_ops_with_layouts = find_if(
        self.module,
        lambda op: (
            "out_layouts" in op.attributes or "in_layouts" in op.attributes
        ),
    )
    self.assertEmpty(all_ops_with_layouts)

  def test_lowering_splat_constant(self):
    elt_ty = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      vec_ty = ir.VectorType.get((16, 8), elt_ty)
      zero = ir.FloatAttr.get(elt_ty, 0)
      cst = arith.ConstantOp(
          vec_ty, ir.DenseElementsAttr.get_splat(vec_ty, zero)
      )
      cst.attributes["out_layouts"] = ir.ArrayAttr.get([
          layouts.to_layout_attr(
              mgpu.WGStridedFragLayout.from_shaped_type(vec_ty)
          )
      ])

    mgpu.lower_mgpu_dialect(self.module, None)

    cst_ops = find_if(
        self.module,
        lambda op: isinstance(op, arith.ConstantOp),
    )
    self.assertLen(cst_ops, 1)
    self.assertEqual(cst_ops[0].result.type, elt_ty)

  def test_lowering_vector_load_and_store_ops(self):
    shape = (8, 128)
    elt_ty = ir.BF16Type.get()
    with ir.InsertionPoint(self.module.body):
      ref = llvm.mlir_undef(ir.MemRefType.get(shape, elt_ty))
      reg = mgpu.dialect.vector_load(ref)
      mgpu.dialect.vector_store(reg, ref)

    mgpu.infer_layout(self.module)
    mgpu.lower_mgpu_dialect(self.module, None)

    all_loads = find_if(
        self.module,
        lambda op: isinstance(op, vector.LoadOp),
    )
    all_stores = find_if(
        self.module,
        lambda op: isinstance(op, vector.StoreOp),
    )

    # The shape is (8, 128). Assuming a single warpgroup (128 threads), we
    # expect each thread to load 8 elements---with two vectorized loads of size
    # 8 bytes.
    self.assertLen(all_loads, 2)
    self.assertLen(all_stores, 2)

    def check_type(ty: ir.Type):
      self.assertTrue(ir.VectorType.get((4,), elt_ty).isinstance(ty))

    load1, load2, *_ = all_loads  # Variadic unpacking to silence linter.
    check_type(load1.result.type)
    check_type(load2.result.type)

    store1, store2, *_ = all_stores  # Variadic unpacking to silence linter.
    check_type(store1.valueToStore.type)
    check_type(store2.valueToStore.type)

  def test_lowering_for(self):
    shape = (4, 128)
    i32 = ir.IntegerType.get_signless(32)
    splat_layout_attr = layouts.to_layout_attr(mgpu.WGSplatFragLayout(shape))
    with ir.InsertionPoint(self.module.body):
      i1 = arith.constant(ir.IndexType.get(), 1)
      c1 = arith.constant(i32, 1)
      splat = vector.BroadcastOp(
          ir.VectorType.get(shape, i32),
          arith.constant(i32, 1234),
      )
      splat.attributes["out_layouts"] = ir.ArrayAttr.get([
          splat_layout_attr
      ])
      ptr = llvm.mlir_undef(ir.Type.parse("!llvm.ptr"))
      ref = mgpu_utils.ptr_as_memref(ptr, ir.MemRefType.get(shape, i32))
      other_vec = mgpu.dialect.VectorLoadOp(ref)
      strided_layout_attr = layouts.to_layout_attr(
          mgpu.WGStridedFragLayout.from_shaped_type(other_vec.result.type)
      )
      other_vec.attributes["out_layouts"] = ir.ArrayAttr.get([strided_layout_attr])
      for_op = scf.ForOp(i1, i1, i1, [c1, splat.result])
      for_op.attributes["in_layouts"] = ir.ArrayAttr.get([strided_layout_attr])
      for_op.attributes["out_layouts"] = ir.ArrayAttr.get([strided_layout_attr])
      with ir.InsertionPoint(for_op.body):
        i, int_carry, vec_carry = for_op.body.arguments
        new_int_carry = arith.addi(int_carry, arith.index_castui(i32, i))
        new_vec_carry = arith.AddIOp(vec_carry, other_vec)
        new_vec_carry.attributes["in_layouts"] = ir.ArrayAttr.get([strided_layout_attr] * 2)
        new_vec_carry.attributes["out_layouts"] = ir.ArrayAttr.get([strided_layout_attr])
        yield_op = scf.YieldOp([new_int_carry, new_vec_carry])
        yield_op.attributes["in_layouts"] = ir.ArrayAttr.get([strided_layout_attr])

    mgpu.lower_mgpu_dialect(self.module, None)
    self.module.operation.verify()
    [for_op] = find_if(self.module, lambda op: isinstance(op, scf.ForOp))
    result_types = [r.type for r in for_op.results]
    reg_vec_ty = ir.VectorType.get((2,), i32)
    self.assertSequenceEqual(result_types, [i32, reg_vec_ty, reg_vec_ty])

  def test_lowering_slice_smem_op(self):
    with ir.InsertionPoint(self.module.body):
      shift = 1234
      i32 = ir.IntegerType.get_signless(32)
      memref_ty = ir.MemRefType.get((4, 32), i32, memory_space=mgpu_utils.smem())
      offset = arith.constant(i32, shift)
      op = mgpu.dialect.SliceSMEMOp(memref_ty, offset)
      op.attributes["out_transforms"] = ir.ArrayAttr.get([ir.ArrayAttr.get([])])

    mgpu.lower_mgpu_dialect(self.module, None)
    # Avoid making a change detector, only validate that lowering runs as
    # expected.
    self.assertEmpty(
        find_if(
            self.module, lambda op: isinstance(op, mgpu.dialect.SliceSMEMOp)
        )
    )

  @parameterized.parameters(
      (arith.ExtFOp, jnp.bfloat16, jnp.float32),
      (arith.ExtSIOp, jnp.int16, jnp.int32),
      (arith.ExtUIOp, jnp.int16, jnp.uint32),
      (arith.FPToSIOp, jnp.float32, jnp.int32),
      (arith.FPToUIOp, jnp.float32, jnp.uint32),
      (arith.SIToFPOp, jnp.int16, jnp.float32),
      (arith.TruncFOp, jnp.float32, jnp.float16),
      (arith.TruncIOp, jnp.int32, jnp.int16),
      (arith.UIToFPOp, jnp.uint32, jnp.float32),
  )
  def test_lower_conversion_op_lowers_to_same_op(self, op, in_dtype, out_dtype):
    shape = (4, 32)

    with ir.InsertionPoint(self.module.body):
      scalar_in_ty = mgpu_utils.dtype_to_ir_type(in_dtype)
      scalar_out_ty = mgpu_utils.dtype_to_ir_type(out_dtype)
      in_ty = ir.VectorType.get(shape, scalar_in_ty)
      out_ty = ir.VectorType.get(shape, scalar_out_ty)
      if ir.IntegerType.isinstance(scalar_in_ty):
        zero = ir.IntegerAttr.get(scalar_in_ty, 0)
      else:
        zero = ir.FloatAttr.get(scalar_in_ty, 0)
      splat_zero = arith.ConstantOp(
          in_ty, ir.DenseElementsAttr.get_splat(in_ty, zero)
      )
      op(out_ty, splat_zero)

    mgpu.infer_layout(self.module)
    mgpu.lower_mgpu_dialect(self.module, None)

    conversion_ops = find_if(self.module, lambda o: isinstance(o, op))
    # This is a splat, so we expect a single conversion op involving a scalar
    # after lowering.
    self.assertLen(conversion_ops, 1)
    self.assertEqual(conversion_ops[0].result.type, scalar_out_ty)

  @parameterized.parameters(
      (True, False, False),
      (False, True, False),
      (False, False, True),
  )
  def test_custom_primitive_op_must_have_number_of_annotations_matching_operands_and_results(
      self, omit_in_layouts, omit_in_transforms, omit_out_layouts
  ):
    vec_ty = ir.VectorType.get((4, 32), ir.BF16Type.get())
    out_layouts = [
        layouts.to_layout_attr(
            mgpu.WGStridedFragLayout.from_shaped_type(vec_ty)
        )
    ]
    in_layouts = out_layouts * 2
    in_transforms = [
        ir.ArrayAttr.get([mgpu.dialect.SwizzleTransformAttr.get(128)])
    ]

    in_layouts = [] if omit_in_layouts else in_layouts
    in_transforms = [] if omit_in_transforms else in_transforms
    out_layouts = [] if omit_out_layouts else out_layouts

    with ir.InsertionPoint(self.module.body):
      ref_ty = ir.MemRefType.get(
          (4, 32), ir.BF16Type.get(), memory_space=mgpu_utils.smem()
      )
      vec1, vec2, ref = undefs(vec_ty, vec_ty, ref_ty)
      op = mgpu.dialect.CustomPrimitiveOp(
          [vec_ty], [vec1, vec2, ref], in_layouts, in_transforms, out_layouts
      )
      args_ty = [arg.type for arg in op.operands_]
      block = op.body.blocks.append(*args_ty)
      with ir.InsertionPoint(block):
        out = undefs(vec_ty)
        mgpu.dialect.ReturnOp(out)

    if omit_in_layouts:
      error = "layout for each vector operand"
    elif omit_in_transforms:
      error = "transforms for each memref operand in smem"
    else:
      assert omit_out_layouts
      error = "layout for each vector result"

    with self.assertRaisesRegex(ir.MLIRError, error):
      self.module.operation.verify()

  def test_memref_transforms_with_transpose(self):
    with ir.InsertionPoint(self.module.body):
      ty_in = ir.MemRefType.get(
          (64, 128),
          ir.BF16Type.get(),
          memory_space=mgpu_utils.smem(),
      )
      ref = memref.alloc(ty_in, [], [])

      ref = mgpu_utils.memref_transpose(ref, (1, 0))
      # This tiling is applied to the transposed memref.
      transforms = [mgpu.TileTransform(tiling=(16, 32))]

      ref_transformed = lowering.reinterpret_smem_ref(ref, transforms)
      ty_transformed = ir.MemRefType(ref_transformed.type)
      self.assertEqual(ty_transformed.shape, [8, 2, 16, 32])
      strides, _ = ty_transformed.get_strides_and_offset()
      self.assertEqual(strides, [512, 4096, 1, 16])

  def test_optimized_gmem_transfers_are_not_supported(self):
    def body(ctx, input, output, scratch):
      del ctx, output, scratch
      reg = mgpu.dialect.vector_load(input, optimized=True)
      layout = layouts.to_layout_attr(mgpu.WGMMA_LAYOUT)
      mgpu.dialect.layout_cast(reg, layout)

    shape = (128, 128)
    dtype = jnp.bfloat16
    with self.assertRaisesRegex(
        NotImplementedError, "Only optimized transfers to SMEM supported"
    ):
      mgpu.as_gpu_kernel(
          body,
          grid=(1, 1, 1),
          block=(128, 1, 1),
          in_shape=jax.ShapeDtypeStruct(shape, dtype),
          out_shape=jax.ShapeDtypeStruct(shape, dtype),
          smem_scratch_shape=(),
          thread_semantics=mgpu.LoweringSemantics.Warpgroup,
      )

  def test_inconsistent_collective_attributes_in_kernel_raise(self):
    def body(ctx, out, smem_ptr):
      del ctx, out
      ref_ty = ir.MemRefType.get(
          (128, 128),
          ir.BF16Type.get(),
          memory_space=mgpu_utils.tmem(),
      )
      mgpu.dialect.tmem_alloc(ref_ty, smem_ptr, collective=False)
      mgpu.dialect.tmem_alloc(ref_ty, smem_ptr, collective=True)

    with self.assertRaisesRegex(
        ValueError,
        "Collective attributes are inconsistent across operations in the"
        " kernel",
    ):
      mgpu.as_gpu_kernel(
          body,
          grid=(1, 1, 1),
          block=(128, 1, 1),
          in_shape=(),
          out_shape=(jax.ShapeDtypeStruct((), jnp.int32),),
          smem_scratch_shape=jax.ShapeDtypeStruct((), jnp.int32),
          thread_semantics=mgpu.LoweringSemantics.Warpgroup,
      )


if __name__ == "__main__":
  parameterized.absltest.main(testLoader=jtu.JaxTestLoader())
