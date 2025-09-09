# Copyright 2025 The JAX Authors. All Rights Reserved.
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
"""Transform inference tests for the Mosaic GPU MLIR dialect."""

# pylint: disable=g-complex-comprehension

from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
from jax._src.interpreters import mlir as mlir_interpreter
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import llvm
from jax._src.lib.mlir.dialects import memref
from jax._src.lib.mlir.dialects import vector
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.mosaic.gpu import fragmented_array as fa
from jax.experimental.mosaic.gpu import inference_utils
from jax.experimental.mosaic.gpu import layouts as layouts_lib
import numpy as np


config.parse_flags_with_absl()


def _make_ir_context():
  context = ir.Context()
  context.append_dialect_registry(mlir_interpreter.upstream_dialects)
  context.load_all_available_dialects()
  mgpu.dialect.register_dialect(context)
  return context


def undefs(*tys: ir.Type) -> list[ir.Value]:
  """Returns a list of `llvm.mlir_undef` values of the given types."""
  return [llvm.mlir_undef(ty) for ty in tys]


class TransformInferenceTest(parameterized.TestCase):

  def setUp(self):
    if jax.version._version != jax.lib.__version__:
      raise self.skipTest("Test requires matching jax and jaxlib versions")
    super().setUp()
    self.enter_context(_make_ir_context())
    self.enter_context(ir.Location.unknown())
    self.module = ir.Module.create()

  @parameterized.parameters(
      (swizzle, dtype)
      for swizzle in mgpu.dialect.SwizzlingMode
      for dtype in [jnp.bfloat16, jnp.float32]
  )
  def test_infer_transforms_for_wgmma_op(self, swizzle, dtype):
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
      lhs_ty = ir.MemRefType.get(lhs_shape, elt_ty, memory_space=mgpu.utils.smem())
      rhs_ty = ir.MemRefType.get(rhs_shape, elt_ty, memory_space=mgpu.utils.smem())
      acc_ty = ir.VectorType.get(out_shape, elt_ty)
      [acc, lhs, rhs] = undefs(acc_ty, lhs_ty, rhs_ty)
      wgmma_op = mgpu.dialect.WGMMAOp(acc, lhs, rhs)

    mgpu.infer_transforms(self.module)

    arg_transforms = ir.ArrayAttr.get([
        mgpu.dialect.TileTransformAttr.get((8, swizzle_elems)),
        mgpu.dialect.SwizzleTransformAttr.get(int(swizzle)),
    ])

    self.assertSequenceEqual(
        inference_utils.in_transforms(wgmma_op),
        [arg_transforms, arg_transforms],
    )
    self.assertEmpty(inference_utils.out_transforms(wgmma_op))

  def test_infer_transforms_for_async_load_derives_from_destination(self):
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      gmem_ty = ir.MemRefType.get(shape, elt_ty)
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem())
      barrier_ty = ir.Type.parse("!mosaic_gpu.barrier")
      gmem_ref, smem_ref, barrier = undefs(gmem_ty, smem_ty, barrier_ty)

      transforms = ir.ArrayAttr.get(
          [mgpu.dialect.TileTransformAttr.get((8, 32))]
      )
      transformed_smem_ref = mgpu.dialect.with_transforms(smem_ref, transforms)

      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      async_load_op = mgpu.dialect.AsyncLoadOp(
          source=gmem_ref,
          destination=transformed_smem_ref,
          barrier=barrier,
          indices=[zero, zero],
          slice_lengths=shape,
          collective=ir.ArrayAttr.get([]),
      )

    mgpu.infer_transforms(self.module)

    self.assertSequenceEqual(
        inference_utils.in_transforms(async_load_op), [transforms]
    )
    self.assertEmpty(inference_utils.out_transforms(async_load_op))

  def test_infer_transforms_for_async_store_op_derives_from_source(self):
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      gmem_ty = ir.MemRefType.get(shape, elt_ty)
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem())
      gmem_ref, smem_ref = undefs(gmem_ty, smem_ty)

      transforms = ir.ArrayAttr.get(
          [mgpu.dialect.TileTransformAttr.get((8, 32))]
      )
      smem_ref = mgpu.dialect.with_transforms(smem_ref, transforms)

      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      async_store_op = mgpu.dialect.AsyncStoreOp(
          source=smem_ref,
          destination=gmem_ref,
          indices=[zero, zero],
          slice_lengths=shape,
      )

    mgpu.infer_transforms(self.module)

    self.assertSequenceEqual(
        inference_utils.in_transforms(async_store_op), [transforms]
    )
    self.assertEmpty(inference_utils.out_transforms(async_store_op))

  def test_infer_transforms_for_vector_load_op_derives_from_destination(self):
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem())
      [smem_ref] = undefs(smem_ty)

      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      vector_load_op = vector.LoadOp(
          ir.VectorType.get(shape, elt_ty), smem_ref, [zero] * len(shape)
      )

    vector_load_op.attributes["out_layouts"] = ir.ArrayAttr.get(
        [layouts_lib.to_layout_attr(fa.WGMMA_LAYOUT)]
    )

    mgpu.infer_transforms(self.module)

    expected_transforms = ir.ArrayAttr.get([
        mgpu.dialect.TileTransformAttr.get((8, 64)),
        mgpu.dialect.SwizzleTransformAttr.get(128),
    ])

    self.assertSequenceEqual(
        inference_utils.in_transforms(vector_load_op), [expected_transforms]
    )
    self.assertEmpty(inference_utils.out_transforms(vector_load_op))

  def test_infer_transforms_for_vector_load_op_derives_from_source(self):
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem())
      [smem_ref] = undefs(smem_ty)

      transforms = ir.ArrayAttr.get(
          [mgpu.dialect.TileTransformAttr.get((8, 64))]
      )
      smem_ref = mgpu.dialect.with_transforms(smem_ref, transforms)

      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      vector_load_op = vector.LoadOp(
          ir.VectorType.get(shape, elt_ty), smem_ref, [zero] * len(shape)
      )

    vector_load_op.attributes["out_layouts"] = ir.ArrayAttr.get(
        [layouts_lib.to_layout_attr(fa.WGStridedFragLayout(shape, vec_size=4))]
    )

    mgpu.infer_transforms(self.module)

    self.assertSequenceEqual(
        inference_utils.in_transforms(vector_load_op), [transforms]
    )
    self.assertEmpty(inference_utils.out_transforms(vector_load_op))

  def test_infer_transforms_for_vector_load_op_raises_on_mismatches(self):
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem())
      [smem_ref] = undefs(smem_ty)

      transforms = ir.ArrayAttr.get(
          [mgpu.dialect.TileTransformAttr.get((8, 64))]
      )
      smem_ref = mgpu.dialect.with_transforms(smem_ref, transforms)

      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      vector_load_op = vector.LoadOp(
          ir.VectorType.get(shape, elt_ty), smem_ref, [zero] * len(shape)
      )

    vector_load_op.attributes["out_layouts"] = ir.ArrayAttr.get(
        [layouts_lib.to_layout_attr(fa.WGMMA_LAYOUT)]
    )

    with self.assertRaisesRegex(NotImplementedError, "Conflicting transforms"):
      mgpu.infer_transforms(self.module)

  def test_infer_transforms_for_vector_store_op_derives_from_destination(self):
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem())
      value_ty = ir.VectorType.get(shape, elt_ty)
      [smem_ref, value_to_store] = undefs(smem_ty, value_ty)

      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      vector_store_op = vector.StoreOp(
          value_to_store, smem_ref, [zero] * len(shape)
      )

    vector_store_op.attributes["in_layouts"] = ir.ArrayAttr.get(
        [layouts_lib.to_layout_attr(fa.WGMMA_LAYOUT)]
    )

    mgpu.infer_transforms(self.module)

    expected_transforms = ir.ArrayAttr.get([
        mgpu.dialect.TileTransformAttr.get((8, 64)),
        mgpu.dialect.SwizzleTransformAttr.get(128),
    ])

    self.assertSequenceEqual(
        inference_utils.in_transforms(vector_store_op), [expected_transforms]
    )
    self.assertEmpty(inference_utils.out_transforms(vector_store_op))

  def test_infer_transforms_for_vector_store_op_derives_from_source(self):
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem())
      value_ty = ir.VectorType.get(shape, elt_ty)
      [smem_ref, value_to_store] = undefs(smem_ty, value_ty)

      transforms = ir.ArrayAttr.get(
          [mgpu.dialect.TileTransformAttr.get((8, 64))]
      )
      smem_ref = mgpu.dialect.with_transforms(smem_ref, transforms)

      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      vector_store_op = vector.StoreOp(
          value_to_store, smem_ref, [zero] * len(shape)
      )

    vector_store_op.attributes["in_layouts"] = ir.ArrayAttr.get(
        [layouts_lib.to_layout_attr(fa.WGStridedFragLayout(shape, vec_size=4))]
    )

    mgpu.infer_transforms(self.module)

    self.assertSequenceEqual(
        inference_utils.in_transforms(vector_store_op), [transforms]
    )
    self.assertEmpty(inference_utils.out_transforms(vector_store_op))

  def test_infer_transforms_for_vector_store_op_raises_on_mismatches(self):
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem())
      value_ty = ir.VectorType.get(shape, elt_ty)
      [smem_ref, value_to_store] = undefs(smem_ty, value_ty)

      transforms = ir.ArrayAttr.get(
          [mgpu.dialect.TileTransformAttr.get((8, 64))]
      )
      smem_ref = mgpu.dialect.with_transforms(smem_ref, transforms)

      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      vector_store_op = vector.StoreOp(
          value_to_store, smem_ref, [zero] * len(shape)
      )

    vector_store_op.attributes["in_layouts"] = ir.ArrayAttr.get(
        [layouts_lib.to_layout_attr(fa.WGMMA_LAYOUT)]
    )

    with self.assertRaisesRegex(NotImplementedError, "Conflicting transforms"):
      mgpu.infer_transforms(self.module)

  def test_infer_transforms_for_slice_smem_op_derives_from_user(self):
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      i32 = ir.IntegerType.get_signless(32)
      [offset] = undefs(i32)
      slice_smem_op = mgpu.dialect.SliceSMEMOp(
          ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem()), offset
      )
      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      load_offsets = [zero] * len(shape)
      vector_load_op = vector.LoadOp(
          ir.VectorType.get(shape, elt_ty), slice_smem_op.result, load_offsets
      )

    vector_load_op.attributes["out_layouts"] = ir.ArrayAttr.get(
        [layouts_lib.to_layout_attr(fa.WGMMA_LAYOUT)]
    )

    mgpu.infer_transforms(self.module)

    expected_transforms = ir.ArrayAttr.get([
        mgpu.dialect.TileTransformAttr.get((8, 64)),
        mgpu.dialect.SwizzleTransformAttr.get(128),
    ])

    self.assertEmpty(inference_utils.in_transforms(slice_smem_op))
    self.assertSequenceEqual(
        inference_utils.out_transforms(slice_smem_op), [expected_transforms]
    )

  def test_infer_transforms_for_slice_smem_op_raises_on_mismatches(self):
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    with ir.InsertionPoint(self.module.body):
      i32 = ir.IntegerType.get_signless(32)
      [offset] = undefs(i32)
      slice_smem_op = mgpu.dialect.SliceSMEMOp(
          ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem()), offset
      )
      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      load_offsets = [zero] * len(shape)
      vector_load_op1 = vector.LoadOp(
          ir.VectorType.get(shape, elt_ty), slice_smem_op.result, load_offsets
      )

      transforms = ir.ArrayAttr.get(
          [ir.ArrayAttr.get([mgpu.dialect.TileTransformAttr.get((8, 32))])]
      )
      smem_ref = mgpu.dialect.with_transforms(slice_smem_op.result, transforms)

      vector_load_op2 = vector.LoadOp(
          ir.VectorType.get(shape, elt_ty), smem_ref, load_offsets
      )

    vector_load_op1.attributes["out_layouts"] = ir.ArrayAttr.get(
        [layouts_lib.to_layout_attr(fa.WGMMA_LAYOUT)]
    )
    vector_load_op2.attributes["out_layouts"] = ir.ArrayAttr.get(
        [layouts_lib.to_layout_attr(fa.WGStridedFragLayout(shape, vec_size=4))]
    )

    with self.assertRaisesRegex(NotImplementedError, "Conflicting transforms"):
      mgpu.infer_transforms(self.module)

  @parameterized.parameters([False, True])
  def test_infer_transforms_for_subview_op_propagates_undisturbed_tile_and_swizzle_transforms(
      self, annotate_input
  ):
    shape = (2, 64, 64)
    elt_ty = ir.BF16Type.get()

    in_ref_ty = ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem())
    out_ref_ty = ir.MemRefType.get(shape[2:], elt_ty, memory_space=mgpu.utils.smem())

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
          static_offsets=[1, 0, 0],
          static_sizes=[1, 64, 64],
          static_strides=[1, 1, 1],
      )

      if not annotate_input:
        mgpu.dialect.with_transforms(subview_op.result, transforms)

    mgpu.infer_transforms(self.module)

    self.assertSequenceEqual(
        inference_utils.in_transforms(subview_op), [transforms]
    )
    self.assertSequenceEqual(
        inference_utils.out_transforms(subview_op), [transforms]
    )

  def test_infer_transforms_sets_default_emptry_transforms(self):
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

    mgpu.infer_transforms(self.module)
    [in_transform] = inference_utils.in_transforms(async_load_op)
    self.assertSequenceEqual(in_transform, ir.ArrayAttr.get([]))
    self.assertEmpty(inference_utils.out_transforms(async_load_op))

  @parameterized.parameters([False, True])
  def test_infer_transforms_for_subview_op_raises_on_disturbed_transforms(
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
          static_sizes = [2, 64, 32],
          static_strides = [1, 1, 1]
      )

      if not annotate_input:
        mgpu.dialect.with_transforms(subview_op.result, transforms)

    with self.assertRaises(NotImplementedError):
      mgpu.infer_transforms(self.module)

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

      transforms_0 = ir.ArrayAttr.get([
          mgpu.dialect.TileTransformAttr.get((64, 64)),
          mgpu.dialect.SwizzleTransformAttr.get(32),
      ])
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

    mgpu.infer_transforms(self.module)

    want = ir.ArrayAttr.get([
          mgpu.dialect.TileTransformAttr.get((2 if even_offsets else 1, 64)),
          mgpu.dialect.SwizzleTransformAttr.get(32),
      ])

    self.assertSequenceEqual(inference_utils.in_transforms(subview_op0), [want])
    self.assertSequenceEqual(inference_utils.out_transforms(subview_op0), [want])
    self.assertSequenceEqual(inference_utils.in_transforms(subview_op1), [want])
    self.assertSequenceEqual(inference_utils.out_transforms(subview_op1), [want])
    self.assertSequenceEqual(inference_utils.in_transforms(subview_op2), [want])
    self.assertSequenceEqual(inference_utils.out_transforms(subview_op2), [want])
    self.assertSequenceEqual(inference_utils.in_transforms(subview_op3), [want])
    self.assertSequenceEqual(inference_utils.out_transforms(subview_op3), [want])

  def test_custom_primitive_op_retains_transforms(self):
    with ir.InsertionPoint(self.module.body):
      transforms = ir.ArrayAttr.get([
          mgpu.dialect.TileTransformAttr.get((64, 64)),
          mgpu.dialect.SwizzleTransformAttr.get(32),
      ])
      op = mgpu.dialect.custom_primitive(
          result=[],
          operands_=[],
          in_layouts=[],
          in_transforms=[transforms],
          out_layouts=[],
      )

    mgpu.infer_transforms(self.module)
    self.assertSequenceEqual(inference_utils.in_transforms(op), [transforms])

  @parameterized.parameters([False, True])
  def test_infer_transforms_for_subview_handles_dynamic_offsets(
      self, annotate_input
  ):
    shape = (32, 32, 32)
    elt_ty = ir.BF16Type.get()

    in_ref_ty = ir.MemRefType.get(shape, elt_ty, memory_space=mgpu.utils.smem())
    out_ref_ty = ir.MemRefType.get((16, 16, 32), elt_ty, memory_space=mgpu.utils.smem())

    with ir.InsertionPoint(self.module.body):
      [in_ref] = undefs(in_ref_ty)

      transforms = ir.ArrayAttr.get([
          mgpu.dialect.TileTransformAttr.get((16, 16)),
          mgpu.dialect.SwizzleTransformAttr.get(32),
      ])

      if annotate_input:
        in_ref = mgpu.dialect.with_transforms(in_ref, transforms)

      c = lambda x: arith.constant(ir.IntegerType.get_signless(32), x)
      subview_op = memref.SubViewOp(
          out_ref_ty,
          in_ref,
          [c(16), c(4)],
          [],
          [],
          static_offsets=[
              ir.ShapedType.get_dynamic_size(),
              ir.ShapedType.get_dynamic_size(),
              0,
          ],
          static_sizes=[16, 16, 32],
          static_strides=[1, 1, 1],
      )

      if not annotate_input:
        mgpu.dialect.with_transforms(subview_op.result, transforms)

    mgpu.infer_transforms(self.module)

    expected_transforms = ir.ArrayAttr.get([
        mgpu.dialect.TileTransformAttr.get((1, 16)),
        mgpu.dialect.SwizzleTransformAttr.get(32),
    ])

    self.assertSequenceEqual(
        inference_utils.in_transforms(subview_op), [expected_transforms]
    )
    self.assertSequenceEqual(
        inference_utils.out_transforms(subview_op), [expected_transforms]
    )

if __name__ == "__main__":
  parameterized.absltest.main(testLoader=jtu.JaxTestLoader())
