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
from jax._src.lib.mlir.dialects import builtin
from jax._src.lib.mlir.dialects import func
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
    wgmma_op = None

    def body(accumulator, lhs, rhs):
      nonlocal wgmma_op
      wgmma_op = mgpu.dialect.WGMMAOp(accumulator, lhs, rhs)

    with ir.InsertionPoint(self.module.body):
      smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
      elt_ty = mgpu.utils.dtype_to_ir_type(dtype)
      lhs_ty = ir.MemRefType.get(lhs_shape, elt_ty, memory_space=smem)
      rhs_ty = ir.MemRefType.get(rhs_shape, elt_ty, memory_space=smem)
      acc_ty = ir.VectorType.get(out_shape, elt_ty)
      func.FuncOp.from_py_func(acc_ty, lhs_ty, rhs_ty)(body)

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
    async_load_op = None
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    def body(gmem_ref, smem_ref, barrier):
      nonlocal async_load_op
      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      async_load_op = mgpu.dialect.AsyncLoadOp(
          source=gmem_ref,
          destination=smem_ref,
          barrier=barrier,
          indices=[zero, zero],
          slice_lengths=shape,
          collective=ir.ArrayAttr.get([]),
      )

    with ir.InsertionPoint(self.module.body):
      smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
      gmem_ty = ir.MemRefType.get(shape, elt_ty)
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=smem)
      barrier_ty = ir.Type.parse("!mosaic_gpu.barrier")
      f = func.FuncOp.from_py_func(gmem_ty, smem_ty, barrier_ty)(body).func_op

    transforms = ir.ArrayAttr.get(
        [mgpu.dialect.TransposeTransformAttr.get((1, 0))]
    )
    f.attributes["in_transforms"] = ir.ArrayAttr.get([transforms])

    mgpu.infer_transforms(self.module)

    self.assertSequenceEqual(
        inference_utils.in_transforms(async_load_op), [transforms]
    )
    self.assertEmpty(inference_utils.out_transforms(async_load_op))

  def test_infer_transforms_for_async_store_op_derives_from_source(self):
    async_store_op = None
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    def body(gmem_ref, smem_ref):
      nonlocal async_store_op
      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      async_store_op = mgpu.dialect.AsyncStoreOp(
          source=smem_ref,
          destination=gmem_ref,
          indices=[zero, zero],
          slice_lengths=shape,
      )

    with ir.InsertionPoint(self.module.body):
      smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
      gmem_ty = ir.MemRefType.get(shape, elt_ty)
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=smem)
      f = func.FuncOp.from_py_func(gmem_ty, smem_ty)(body).func_op

    transforms = ir.ArrayAttr.get(
        [mgpu.dialect.TransposeTransformAttr.get((1, 0))]
    )
    f.attributes["in_transforms"] = ir.ArrayAttr.get([transforms])

    mgpu.infer_transforms(self.module)

    self.assertSequenceEqual(
        inference_utils.in_transforms(async_store_op), [transforms]
    )
    self.assertEmpty(inference_utils.out_transforms(async_store_op))

  def test_infer_transforms_for_vector_load_op_derives_from_destination(self):
    vector_load_op = None
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    def body(smem_ref):
      nonlocal vector_load_op
      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      vector_load_op = vector.LoadOp(
          ir.VectorType.get(shape, elt_ty), smem_ref, [zero] * len(shape)
      )

    with ir.InsertionPoint(self.module.body):
      smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=smem)
      func.FuncOp.from_py_func(smem_ty)(body)

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
    vector_load_op = None
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    def body(smem_ref):
      nonlocal vector_load_op
      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      vector_load_op = vector.LoadOp(
          ir.VectorType.get(shape, elt_ty), smem_ref, [zero] * len(shape)
      )

    with ir.InsertionPoint(self.module.body):
      smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=smem)
      f = func.FuncOp.from_py_func(smem_ty)(body).func_op

    vector_load_op.attributes["out_layouts"] = ir.ArrayAttr.get(
        [layouts_lib.to_layout_attr(fa.WGStridedFragLayout(shape, vec_size=4))]
    )
    transforms = ir.ArrayAttr.get([mgpu.dialect.TileTransformAttr.get((8, 64))])
    f.attributes["in_transforms"] = ir.ArrayAttr.get([transforms])

    mgpu.infer_transforms(self.module)

    self.assertSequenceEqual(
        inference_utils.in_transforms(vector_load_op), [transforms]
    )
    self.assertEmpty(inference_utils.out_transforms(vector_load_op))

  def test_infer_transforms_for_vector_load_op_raises_on_mismatches(self):
    vector_load_op = None
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    def body(smem_ref):
      nonlocal vector_load_op
      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      vector_load_op = vector.LoadOp(
          ir.VectorType.get(shape, elt_ty), smem_ref, [zero] * len(shape)
      )

    with ir.InsertionPoint(self.module.body):
      smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=smem)
      f = func.FuncOp.from_py_func(smem_ty)(body).func_op

    vector_load_op.attributes["out_layouts"] = ir.ArrayAttr.get(
        [layouts_lib.to_layout_attr(fa.WGMMA_LAYOUT)]
    )
    transforms = ir.ArrayAttr.get([mgpu.dialect.TileTransformAttr.get((8, 64))])
    f.attributes["in_transforms"] = ir.ArrayAttr.get([transforms])

    with self.assertRaisesRegex(NotImplementedError, "Conflicting transforms"):
      mgpu.infer_transforms(self.module)

  def test_infer_transforms_for_vector_store_op_derives_from_destination(self):
    vector_store_op = None
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    def body(smem_ref, value_to_store):
      nonlocal vector_store_op
      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      vector_store_op = vector.StoreOp(
          value_to_store, smem_ref, [zero] * len(shape)
      )

    with ir.InsertionPoint(self.module.body):
      smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=smem)
      value_ty = ir.VectorType.get(shape, elt_ty)
      func.FuncOp.from_py_func(smem_ty, value_ty)(body)

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
    vector_store_op = None
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    def body(smem_ref, value_to_store):
      nonlocal vector_store_op
      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      vector_store_op = vector.StoreOp(
          value_to_store, smem_ref, [zero] * len(shape)
      )

    with ir.InsertionPoint(self.module.body):
      smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=smem)
      value_ty = ir.VectorType.get(shape, elt_ty)
      f = func.FuncOp.from_py_func(smem_ty, value_ty)(body).func_op

    vector_store_op.attributes["in_layouts"] = ir.ArrayAttr.get(
        [layouts_lib.to_layout_attr(fa.WGStridedFragLayout(shape, vec_size=4))]
    )
    transforms = ir.ArrayAttr.get([mgpu.dialect.TileTransformAttr.get((8, 64))])
    f.attributes["in_transforms"] = ir.ArrayAttr.get([transforms])

    mgpu.infer_transforms(self.module)

    self.assertSequenceEqual(
        inference_utils.in_transforms(vector_store_op), [transforms]
    )
    self.assertEmpty(inference_utils.out_transforms(vector_store_op))

  def test_infer_transforms_for_vector_store_op_raises_on_mismatches(self):
    vector_store_op = None
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    def body(smem_ref, value_to_store):
      nonlocal vector_store_op
      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      vector_store_op = vector.StoreOp(
          value_to_store, smem_ref, [zero] * len(shape)
      )

    with ir.InsertionPoint(self.module.body):
      smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=smem)
      value_ty = ir.VectorType.get(shape, elt_ty)
      f = func.FuncOp.from_py_func(smem_ty, value_ty)(body).func_op

    vector_store_op.attributes["in_layouts"] = ir.ArrayAttr.get(
        [layouts_lib.to_layout_attr(fa.WGMMA_LAYOUT)]
    )
    transforms = ir.ArrayAttr.get([mgpu.dialect.TileTransformAttr.get((8, 64))])
    f.attributes["in_transforms"] = ir.ArrayAttr.get([transforms])

    with self.assertRaisesRegex(NotImplementedError, "Conflicting transforms"):
      mgpu.infer_transforms(self.module)

  def test_infer_transforms_for_slice_smem_op_derives_from_user(self):
    slice_smem_op = vector_load_op = None
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()
    smem = ir.Attribute.parse("#gpu.address_space<workgroup>")

    def body(offset):
      nonlocal slice_smem_op, vector_load_op
      slice_smem_op = mgpu.dialect.SliceSMEMOp(
          ir.MemRefType.get(shape, elt_ty, memory_space=smem), offset
      )
      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      load_offsets = [zero] * len(shape)
      vector_load_op = vector.LoadOp(
          ir.VectorType.get(shape, elt_ty), slice_smem_op.result, load_offsets
      )

    with ir.InsertionPoint(self.module.body):
      func.FuncOp.from_py_func(ir.IntegerType.get_signless(32))(body)

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
    slice_smem_op = vector_load_op1 = vector_load_op2 = None
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()
    smem = ir.Attribute.parse("#gpu.address_space<workgroup>")

    def body(offset):
      nonlocal slice_smem_op, vector_load_op1, vector_load_op2
      slice_smem_op = mgpu.dialect.SliceSMEMOp(
          ir.MemRefType.get(shape, elt_ty, memory_space=smem), offset
      )
      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      load_offsets = [zero] * len(shape)
      vector_load_op1 = vector.LoadOp(
          ir.VectorType.get(shape, elt_ty), slice_smem_op.result, load_offsets
      )
      vector_load_op2 = vector.LoadOp(
          ir.VectorType.get(shape, elt_ty), slice_smem_op.result, load_offsets
      )

    with ir.InsertionPoint(self.module.body):
      func.FuncOp.from_py_func(ir.IntegerType.get_signless(32))(body)

    vector_load_op1.attributes["out_layouts"] = ir.ArrayAttr.get(
        [layouts_lib.to_layout_attr(fa.WGMMA_LAYOUT)]
    )
    vector_load_op2.attributes["out_layouts"] = ir.ArrayAttr.get(
        [layouts_lib.to_layout_attr(fa.WGStridedFragLayout(shape, vec_size=4))]
    )
    vector_load_op2.attributes["in_transforms"] = ir.ArrayAttr.get(
        [ir.ArrayAttr.get([mgpu.dialect.TransposeTransformAttr.get((1, 0))])]
    )

    with self.assertRaisesRegex(NotImplementedError, "Conflicting transforms"):
      mgpu.infer_transforms(self.module)

  @parameterized.parameters([False, True])
  def test_infer_transforms_for_subview_op_propagates_undisturbed_tile_and_swizzle_transforms(
      self, annotate_input
  ):
    subview_op = user_op = None
    shape = (2, 64, 64)
    elt_ty = ir.BF16Type.get()
    smem = ir.Attribute.parse("#gpu.address_space<workgroup>")

    in_ref_ty = ir.MemRefType.get(shape, elt_ty, memory_space=smem)
    out_ref_ty = ir.MemRefType.get(shape[2:], elt_ty, memory_space=smem)

    def body(in_ref):
      nonlocal subview_op, user_op
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
      user_op = builtin.UnrealizedConversionCastOp(
          [out_ref_ty], [subview_op.result]
      )

    with ir.InsertionPoint(self.module.body):
      f = func.FuncOp.from_py_func(in_ref_ty)(body).func_op

    transforms = ir.ArrayAttr.get([
        mgpu.dialect.TileTransformAttr.get((32, 16)),
        mgpu.dialect.SwizzleTransformAttr.get(32),
    ])

    if annotate_input:
      f.attributes["in_transforms"] = ir.ArrayAttr.get([transforms])
    else:
      user_op.attributes["in_transforms"] = ir.ArrayAttr.get([transforms])

    mgpu.infer_transforms(self.module)

    self.assertSequenceEqual(
        inference_utils.in_transforms(subview_op), [transforms]
    )
    self.assertSequenceEqual(
        inference_utils.out_transforms(subview_op), [transforms]
    )

  def test_infer_transforms_sets_default_emptry_transforms(self):
    async_load_op = None
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    def body(gmem_ref, smem_ref, barrier):
      nonlocal async_load_op
      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      async_load_op = mgpu.dialect.AsyncLoadOp(
          source=gmem_ref,
          destination=smem_ref,
          barrier=barrier,
          indices=[zero, zero],
          slice_lengths=shape,
          collective=ir.ArrayAttr.get([]),
      )

    with ir.InsertionPoint(self.module.body):
      smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
      gmem_ty = ir.MemRefType.get(shape, elt_ty)
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=smem)
      barrier_ty = ir.Type.parse("!mosaic_gpu.barrier")
      func.FuncOp.from_py_func(gmem_ty, smem_ty, barrier_ty)(body).func_op

    mgpu.infer_transforms(self.module)
    [in_transform] = inference_utils.in_transforms(async_load_op)
    self.assertSequenceEqual(in_transform, ir.ArrayAttr.get([]))
    self.assertEmpty(inference_utils.out_transforms(async_load_op))

  @parameterized.parameters([False, True])
  def test_infer_transforms_for_subview_op_raises_on_disturbed_transforms(
      self, annotate_input
  ):
    subview_op = user_op = None
    shape = (2, 64, 64)
    elt_ty = ir.BF16Type.get()
    smem = ir.Attribute.parse("#gpu.address_space<workgroup>")

    in_ref_ty = ir.MemRefType.get(shape, elt_ty, memory_space=smem)
    out_ref_ty = ir.MemRefType.get((2, 64, 32), elt_ty, memory_space=smem)

    def body(in_ref):
      nonlocal subview_op, user_op
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
      user_op = builtin.UnrealizedConversionCastOp(
          [out_ref_ty], [subview_op.result]
      )

    with ir.InsertionPoint(self.module.body):
      f = func.FuncOp.from_py_func(in_ref_ty)(body).func_op

    transforms = ir.ArrayAttr.get([
        mgpu.dialect.TileTransformAttr.get((32, 16)),
        mgpu.dialect.SwizzleTransformAttr.get(32),
    ])

    if annotate_input:
      f.attributes["in_transforms"] = ir.ArrayAttr.get([transforms])
    else:
      user_op.attributes["in_transforms"] = ir.ArrayAttr.get([transforms])

    if annotate_input:
      f.attributes["in_transforms"] = ir.ArrayAttr.get([transforms])
    else:
      user_op.attributes["in_transforms"] = ir.ArrayAttr.get([transforms])

    with self.assertRaises(NotImplementedError):
      mgpu.infer_transforms(self.module)


if __name__ == "__main__":
  parameterized.absltest.main(testLoader=jtu.JaxTestLoader())
