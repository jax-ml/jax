# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transform inference pass for the MLIR Mosaic GPU dialect.

The transform inference pass is meant to run on IR that has already been
annotated with layouts (see `layout_inference.py` for the relevant pass).
"""

from collections.abc import Callable
from functools import partial
import itertools
from typing import cast

from jax._src.lib import mosaic_gpu_dialect as mgpu
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import builtin
from jax._src.lib.mlir.dialects import gpu
from jax._src.lib.mlir.dialects import memref
from jax._src.lib.mlir.dialects import vector

from . import fragmented_array as fa
from . import inference_utils
from . import layouts as layouts_lib
from . import utils

# mypy: ignore-errors

OptionalTransforms = tuple[list[ir.Attribute], list[ir.Attribute]] | None
TransformInferenceRule = Callable[[ir.OpView], OptionalTransforms]
_transform_inference_rules: dict[str, TransformInferenceRule] = {}


def _add_transform_inference_rule(
    op: type[ir.OpView], rule: TransformInferenceRule
):
  if op is not None:
    _transform_inference_rules[op.OPERATION_NAME] = rule  # pytype: disable=attribute-error
  return rule


def _set_transform_attributes(
    op: ir.OpView,
    in_transforms: list[ir.Attribute],
    out_transforms: list[ir.Attribute],
):
  op.attributes["in_transforms"] = ir.ArrayAttr.get(in_transforms)
  op.attributes["out_transforms"] = ir.ArrayAttr.get(out_transforms)


def infer_transforms_for_wgmma_ref(ref_ty: ir.MemRefType) -> ir.ArrayAttr:
  if len(ref_ty.shape) != 2:
    raise ValueError(f"Expected a 2D memref, got {ref_ty}")

  element_bytewidth = utils.bytewidth(ref_ty.element_type)
  strides, _ = ref_ty.get_strides_and_offset()

  if strides[0] < strides[1]:
    raise NotImplementedError("Transpositions aren't handled yet.")

  minor_dim = ref_ty.shape[1]
  major_tiling = 8

  # Try tiling with all swizzling modes starting from the largest one.
  for swizzle in [
      mgpu.SwizzlingMode.k128ByteSwizzle,
      mgpu.SwizzlingMode.k64ByteSwizzle,
      mgpu.SwizzlingMode.k32ByteSwizzle,
      mgpu.SwizzlingMode.kNoSwizzle,
  ]:
    swizzle_elems = swizzle // element_bytewidth
    if minor_dim % swizzle_elems == 0:
      minor_tiling = swizzle_elems
      break
  else:
    # No valid tile transform can be inferred.
    raise ValueError(
        f"{ref_ty.shape} is not a valid WGMMA shape"
    )

  return ir.ArrayAttr.get([
      mgpu.TileTransformAttr.get((major_tiling, minor_tiling)),
      mgpu.SwizzleTransformAttr.get(minor_tiling * element_bytewidth),
  ])


@partial(_add_transform_inference_rule, mgpu.WGMMAOp)
def infer_wgmma_transforms(op: mgpu.WGMMAOp) -> OptionalTransforms:
  b_transforms = infer_transforms_for_wgmma_ref(ir.MemRefType(op.b.type))
  if ir.MemRefType.isinstance(op.a.type):
    a_transforms = infer_transforms_for_wgmma_ref(
        cast(ir.MemRefType, op.a.type)
    )
    return [a_transforms, b_transforms], []
  return [b_transforms], []


@partial(_add_transform_inference_rule, mgpu.AsyncStoreOp)
def _infer_async_store_transforms(op: mgpu.AsyncStoreOp) -> OptionalTransforms:
  in_transforms = inference_utils.value_transforms(op.source)
  return None if in_transforms is None else ([in_transforms], [])


@partial(_add_transform_inference_rule, mgpu.AsyncLoadOp)
def _infer_async_load_transforms(op: mgpu.AsyncLoadOp) -> OptionalTransforms:
  in_transforms = inference_utils.value_transforms(op.destination)
  return None if in_transforms is None else ([in_transforms], [])


@partial(_add_transform_inference_rule, vector.LoadOp)
@partial(_add_transform_inference_rule, vector.StoreOp)
def _infer_vector_load_store_transforms(
    op: vector.LoadOp | vector.StoreOp,
) -> OptionalTransforms:
  for i in op.indices:
    index_defining_op = i.owner.opview
    if (
        not isinstance(index_defining_op, arith.ConstantOp)
        or index_defining_op.literal_value != 0
    ):
      # TODO(bchetioui): handle slicing.
      raise NotImplementedError(
          f"Only constants with value 0 are supported as indices for {op}"
      )

  if isinstance(op, vector.LoadOp):
    [layout_attr] = inference_utils.out_layouts(op)
  else:
    assert isinstance(op, vector.StoreOp)
    [layout_attr] = inference_utils.in_layouts(op)

  layout = layouts_lib.from_layout_attr(layout_attr)
  transforms = inference_utils.value_transforms(op.base)

  if layout == fa.WGMMA_LAYOUT:
    layout_transforms = infer_transforms_for_wgmma_ref(
        ir.MemRefType(op.base.type)
    )
  elif (isinstance(layout, fa.WGStridedFragLayout) or
        isinstance(layout, fa.WGSplatFragLayout)):
    layout_transforms = None
  else:
    raise NotImplementedError(
        f"Got layout {layout} which is not yet supported"
    )

  if transforms is not None and layout_transforms is not None:
    if transforms != layout_transforms:
      raise NotImplementedError(
          f"Conflicting transforms for {op.base} in {op}: "
          f"{transforms} != {layout_transforms}."
      )
    return [transforms], []

  if transforms is not None:
    return [transforms], []

  if layout_transforms is not None:
    return [layout_transforms], []

  return None

# TODO(bchetioui): remove this once jaxlib minimum version >= 0.5.2.
SliceSMEMOp = getattr(mgpu, "SliceSMEMOp", None)

@partial(_add_transform_inference_rule, SliceSMEMOp)
def _infer_slice_smem_transforms(op: SliceSMEMOp) -> OptionalTransforms:
  transforms = None
  uses = cast(ir.OpResult, op.result).uses

  for op_operand_use in uses:
    consumer = op_operand_use.owner
    op_user = consumer.operands[op_operand_use.operand_number]
    out_transforms = inference_utils.in_transforms_for_operand(
        consumer, op_user
    )
    if transforms is not None and out_transforms is not None:
      if transforms != out_transforms:
        raise NotImplementedError(
            f"Conflicting transforms for {op_user} in {op}: "
            f"{transforms} != {out_transforms}."
        )
    elif out_transforms is not None:
      transforms = out_transforms

  return None if transforms is None else ([], [transforms])


# TODO(bchetioui,apaszke): this empty rule is necessary while Mosaic doesn't use
# the dialect in all cases.
# The rule is necessary in order to handle the lowering of `utils.memref_ptr`
# which is used in `_construct_smem_reftree`.
@partial(_add_transform_inference_rule, builtin.UnrealizedConversionCastOp)
def _infer_unrealized_conversion_cast_transforms(
    _: builtin.UnrealizedConversionCastOp,
) -> OptionalTransforms:
  return None


@partial(_add_transform_inference_rule, memref.ViewOp)
def _infer_memref_view_transforms(op: memref.ViewOp) -> OptionalTransforms:
  if not isinstance(op.source.owner.opview, gpu.DynamicSharedMemoryOp):
    raise NotImplementedError(
        "Memref view transforms are only inferred when the op is a direct user "
        f"of a DynamicSharedMemoryOp but got {op}."
    )
  transforms = inference_utils.value_transforms(op.source)
  if transforms is not None:
    raise NotImplementedError(
        "memref view with in_transforms aren't yet supported"
    )
  uses = cast(ir.OpResult, op.result).uses

  for op_operand_use in uses:
    consumer = op_operand_use.owner
    op_user = consumer.operands[op_operand_use.operand_number]
    out_transforms = inference_utils.in_transforms_for_operand(
        consumer, op_user
    )
    if transforms is not None and out_transforms is not None:
      if transforms != out_transforms:
        raise ValueError(
            f"Conflicting transforms for {op_user} in {op}: "
            f"{transforms} != {out_transforms}."
        )
    elif out_transforms is not None:
      transforms = out_transforms

  # TODO(bchetioui): do we actually need to assign a transform to the input of
  # the view op? Presumably, it'll only be used to access scratch memory.
  return None if transforms is None else ([], [transforms])


# TODO(bchetioui,apaszke): this empty rule is necessary while Mosaic doesn't use
# the dialect in all cases.
@partial(_add_transform_inference_rule, gpu.DynamicSharedMemoryOp)
def _infer_dynamic_smem_transforms(
    _: gpu.DynamicSharedMemoryOp,
) -> OptionalTransforms:
  return None


def _should_have_transforms(op: ir.OpView) -> bool:
  """Returns 'True' if the operation should be assigned in/out transforms."""
  return any(
      map(
          inference_utils.is_transformable_smem_memref,
          itertools.chain(op.operands, op.results),
      )
  )


def infer_transforms(module: ir.Module):
  """Infers transforms for the given module.

  Transforms are to memrefs what layouts are to vectors. More specifically,
  transforms describe mappings between SMEM refs and GMEM refs, and are
  determined based on how SMEM refs are used. For that reason, we always
  annotate and apply memrefs on SMEM refs.

  The pass is meant to be called on a module where layouts have been fully
  specified. We error out if two distinct sets of transforms are competing to
  annotate the same memref.
  """
  def inference_step(op: ir.Operation):
    if not _should_have_transforms(op):
      return
    elif inference_rule := _transform_inference_rules.get(op.OPERATION_NAME, None):  # pytype: disable=attribute-error
      pass
    else:
      raise NotImplementedError(f"Can not infer transforms for {op}")

    maybe_transforms = inference_rule(op)
    if maybe_transforms is None:
      return

    _set_transform_attributes(op, *maybe_transforms)

  # It's enough to do a single backwards propagation (starting from vector
  # users), and then a single forward propagation (to feed into the async loads
  # and stores).
  for op in module.body:
    inference_utils.traverse_op(
        op, inference_step, inference_utils.TraversalOrder.BACKWARDS
    )
  for op in module.body:
    inference_utils.traverse_op(
        op, inference_step, inference_utils.TraversalOrder.FORWARD
    )
