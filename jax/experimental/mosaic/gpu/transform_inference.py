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

from . import inference_utils
from . import utils

# mypy: ignore-errors

OptionalTransforms = tuple[list[ir.Attribute], list[ir.Attribute]] | None
TransformInferenceRule = Callable[[ir.OpView], OptionalTransforms]
_transform_inference_rules: dict[str, TransformInferenceRule] = {}


def _add_transform_inference_rule(
    op: type[ir.OpView], rule: TransformInferenceRule
):
  _transform_inference_rules[op.OPERATION_NAME] = rule  # pytype: disable=attribute-error


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

  Transforms are to smem refs the what layouts are to vectors.

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
