# Copyright 2024 The JAX Authors.
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

"""Layout inference pass for the MLIR Mosaic GPU dialect."""

from collections.abc import Callable
import enum
import itertools
from typing import List, Tuple, Type, cast

from jax._src.lib import mosaic_gpu_dialect as mgpu
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith

# mypy: ignore-errors


def strided_fragmented_layout():
  layout = mgpu.FragmentedLayout.WGStridedFragLayout
  return ir.Attribute.parse(f"#mosaic_gpu.fragmented_layout<{layout}>")


def splat_fragmented_layout():
  layout = mgpu.FragmentedLayout.WGSplatFragLayout
  return ir.Attribute.parse(f"#mosaic_gpu.fragmented_layout<{layout}>")


_layout_inference_rules: dict[
    str,
    Callable[[ir.OpView], Tuple[List[ir.Attribute], List[ir.Attribute]] | None],
] = {}


def _add_layout_inference_rule(
    op: Type[ir.OpView],
    rule: Callable[
        [ir.OpView], Tuple[List[ir.Attribute], List[ir.Attribute]] | None
    ],
):
  _layout_inference_rules[op.OPERATION_NAME] = rule  # pytype: disable=attribute-error


def _set_layout_attributes(
    op: ir.OpView,
    in_layouts: List[ir.Attribute],
    out_layouts: List[ir.Attribute],
):
  op.attributes["in_layouts"] = ir.ArrayAttr.get(in_layouts)
  op.attributes["out_layouts"] = ir.ArrayAttr.get(out_layouts)


def _extract_any_layout_from_op(op: ir.OpView) -> ir.Attribute | None:
  if "in_layouts" in op.attributes and len(op.operands) > 0:
    return cast(ir.ArrayAttr, op.attributes["in_layouts"])[0]
  elif "out_layouts" in op.attributes and len(op.results) > 0:
    return cast(ir.ArrayAttr, op.attributes["out_layouts"])[0]

  return None


def _infer_pointwise_op_layouts(
    op: ir.OpView,
) -> Tuple[List[ir.Attribute], List[ir.Attribute]] | None:
  layout = _extract_any_layout_from_op(op)
  # The op had no layout set. Since we're annotating ops, we may need to
  # derive layout information from user or producer ops.
  if layout is None:
    # First, we iterate on users.
    for op_result in op.results:
      for op_user in cast(ir.OpResult, op_result).uses:
        layout = _extract_any_layout_from_op(op_user.owner)
        if layout:
          break
      else:
        continue
      break

  if layout is None:
    # Still no layout set. We iterate on producers.
    for operand in op.operands:
      if isinstance(operand.owner, ir.Operation) or isinstance(
          operand.owner, ir.OpView
      ):
        layout = _extract_any_layout_from_op(operand.owner)
        if layout:
          break

  if layout is None:
    return None

  return ([layout for _ in op.operands], [layout for _ in op.results])


for op in (
    arith.AddFOp,
    arith.ConstantOp,
    arith.MulFOp,
):
  _add_layout_inference_rule(op, _infer_pointwise_op_layouts)


def _layout_inference_should_process_op(op: ir.OpView) -> bool:
  """Returns 'true' if the layout inference pass can skip the operation."""

  def is_array(v: ir.Value):
    ty = v.type
    return ir.RankedTensorType.isinstance(ty) or ir.VectorType.isinstance(ty)

  return any(map(is_array, itertools.chain(op.operands, op.results)))


def _has_any_layout_set(op: ir.OpView) -> bool:
  return "in_layouts" in op.attributes or "out_layouts" in op.attributes


class TraversalOrder(enum.Enum):
  """Traversal orders with respect to the data flow for IR."""

  FORWARD = 1
  BACKWARDS = 2


def traverse_op(
    op: ir.OpView,
    callback: Callable[[ir.OpView], None],
    traversal_order: TraversalOrder = TraversalOrder.FORWARD,
):
  """Traverses the operation and applies the callback in the given order."""
  for region in op.operation.regions:
    for block in region:
      if traversal_order == TraversalOrder.FORWARD:
        ops_to_traverse = block
      else:
        ops_to_traverse = reversed(list(block))
      for block_op in ops_to_traverse:
        traverse_op(block_op, callback, traversal_order)
  callback(op)


def infer_layout(module: ir.Module):
  def inference_step(op: ir.Operation):
    if not _layout_inference_should_process_op(op):
      return
    elif inference_rule := _layout_inference_rules.get(op.OPERATION_NAME, None):  # pytype: disable=attribute-error
      pass
    else:
      raise NotImplementedError(f"Can not infer layout for {op}")

    maybe_layouts = inference_rule(op)
    if maybe_layouts is None:
      return

    _set_layout_attributes(op, *maybe_layouts)

  # We run two passes over the module, in order to make sure that layouts
  # defined in the middle of the computation are propagated wherever they need
  # to be propagated. We start with a backwards (root-to-parameters) pass to
  # propagate the information as far up as possible, and then a forward pass
  # (parameters-to-root).
  #
  # Backwards pass
  for op in module.body:
    traverse_op(op, inference_step, TraversalOrder.BACKWARDS)

  # Forward pass
  for op in module.body:
    traverse_op(op, inference_step, TraversalOrder.FORWARD)

  # At this point, layouts have been propagated as far as they could be
  # propagated. However, it is possible for some operations to remain
  # unannotated---for example, if there were no annotations on any operation in
  # the module at the start of this function. We annotate all the remaining ops
  # that should be annotated with a strided fragmented layout.
  def set_default_layout(op: ir.OpView):
    layout = strided_fragmented_layout()
    if _layout_inference_should_process_op(op) and not _has_any_layout_set(op):
      _set_layout_attributes(
          op, [layout] * len(op.operands), [layout] * len(op.results))

  for op in module.body:
    traverse_op(op, set_default_layout)
