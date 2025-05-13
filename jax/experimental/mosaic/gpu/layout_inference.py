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

from collections.abc import Callable, Sequence
import dataclasses
import enum
from functools import partial
import math
from typing import cast

from jax._src.lib import mosaic_gpu_dialect as mgpu
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import math as mlir_math
from jax._src.lib.mlir.dialects import memref
from jax._src.lib.mlir.dialects import scf
from jax._src.lib.mlir.dialects import vector
import numpy as np

from . import fragmented_array as fa
from . import inference_utils
from . import layouts as layouts_lib
from . import utils


# mypy: ignore-errors

OptionalLayouts = tuple[list[ir.Attribute], list[ir.Attribute]] | None
LayoutInferenceRule = Callable[[ir.OpView], OptionalLayouts]
_layout_inference_rules: dict[str, LayoutInferenceRule] = {}


def _add_layout_inference_rule(op: type[ir.OpView], rule: LayoutInferenceRule):
  if op is not None:
    _layout_inference_rules[op.OPERATION_NAME] = rule  # pytype: disable=attribute-error
  return rule


def _set_layout_attributes(
    op: ir.OpView,
    in_layouts: list[ir.Attribute],
    out_layouts: list[ir.Attribute],
):
    op.attributes["in_layouts"] = ir.ArrayAttr.get(in_layouts)
    op.attributes["out_layouts"] = ir.ArrayAttr.get(out_layouts)


def _choose_representative_layout(
    layouts: set[ir.Attribute],
) -> ir.Attribute | None:
  """Chooses an appropriate layout from a given set of possible layouts.

  Given the input set of possible layouts, this function extracts a single
  representative layout. Currently, this function only works with strided,
  splat, and tiled layouts.

  Returns:
    A single layout that can be used to annotate the operation, or None if the
    input set is empty.
  """

  if not layouts:
    return None

  strided_layouts: list[fa.WGStridedFragLayout] = [
      layouts_lib.from_layout_attr(layout)
      for layout in layouts
      if layouts_lib.is_strided_fragmented_layout(layout)
  ]

  splat_layouts: list[fa.WGSplatFragLayout] = list(
      map(
          layouts_lib.from_layout_attr,
          filter(layouts_lib.is_splat_fragmented_layout, layouts),
      )
  )

  tiled_layouts: list[fa.TiledLayout] = list(
      map(
          layouts_lib.from_layout_attr,
          filter(layouts_lib.is_tiled_layout, layouts),
      )
  )

  if len(splat_layouts) + len(strided_layouts) + len(tiled_layouts) != len(
      layouts
  ):
    raise ValueError(
        f"Expected only strided, splat, and tiled layouts, got {layouts}"
    )

  if len(splat_layouts) > 1:
    raise NotImplementedError(
        "Finding a representative layout for several distinct splat layouts "
        "is not supported."
    )

  if len(strided_layouts) > 1:
    raise NotImplementedError(
        "Finding a representative layout for several distinct strided layouts "
        "is not supported."
    )

  if len(tiled_layouts) > 1:
    raise NotImplementedError(
        "Finding a representative layout for several distinct tiled layouts "
        "is not supported."
    )

  if tiled_layouts and strided_layouts:
    raise NotImplementedError(
        "Mixing strided and tiled layouts is not supported."
    )

  if tiled_layouts:
    return layouts_lib.to_layout_attr(tiled_layouts[0])

  if strided_layouts:
    [strided_layout] = strided_layouts
    return layouts_lib.to_layout_attr(strided_layout)

  [splat_layout] = splat_layouts
  return layouts_lib.to_layout_attr(splat_layout)


def _infer_pointwise_op_layouts(op: ir.OpView) -> OptionalLayouts:

  def is_array(v: ir.Value) -> bool:
    return ir.VectorType.isinstance(v.type)

  num_vector_operands = len([o for o in op.operands if is_array(o)])
  num_vector_results = len([r for r in op.results if is_array(r)])

  if inference_utils.has_in_layouts_set(op):
    op_in_layouts = inference_utils.in_layouts(op)
    if op_in_layouts:
      layout = op_in_layouts[0]
      return (num_vector_operands * [layout], num_vector_results * [layout])

  if inference_utils.has_out_layouts_set(op):
    op_out_layouts = inference_utils.out_layouts(op)
    if op_out_layouts:
      layout = op_out_layouts[0]
      return (num_vector_operands * [layout], num_vector_results * [layout])

  layouts = set()

  # We can also try to infer layouts from the layout of producer and
  # consumer operations.
  #
  # We first look at producers; this enables e.g. propagating splat layouts as
  # far down as possible, until since we may be able to propagate splat layouts
  # further down before requiring a relayout in that way.
  all_inputs_have_layout = True
  for operand in op.operands:
    if not ir.VectorType.isinstance(operand.type):
      continue
    if (layout := inference_utils.value_layout(operand)) is not None:
      layouts.add(layout)
    else:
      all_inputs_have_layout = False

  # We only look at consumers if we haven't found a possible layout yet. This is
  # to avoid propagating more complicated layouts up, to e.g. preserve splat
  # layouts as far down as possible.
  if not layouts:
    for op_result in op.results:
      if not ir.VectorType.isinstance(op_result.type):
        continue
      for op_operand_use in cast(ir.OpResult, op_result).uses:
        consumer = op_operand_use.owner
        op_user = consumer.operands[op_operand_use.operand_number]
        layout = inference_utils.in_layout_for_operand(consumer, op_user)
        if layout is not None:
          layouts.add(layout)

  # TODO(bchetioui): when propagating up, the representative layout should be
  # chosen in the opposite way as when propagating down. E.g., when propagating
  # down, we should pick a strided layout over a splat layout; when propagating
  # up, we should pick a splat layout over a strided layout.
  # This is left for a future change, and currently we only do "down
  # propagation".
  layout = _choose_representative_layout(layouts)
  # It is unsafe to conclude that this op produces a splat if not all inputs
  # have been inferred: some of them might turn out not to be splats!
  if layouts_lib.is_splat_fragmented_layout(layout) and not all_inputs_have_layout:
    return None
  if layout is None:
    return None

  return (num_vector_operands * [layout], num_vector_results * [layout])


for op in [
    arith.AddIOp,
    arith.AddFOp,
    arith.AndIOp,
    arith.BitcastOp,
    arith.CmpFOp,
    arith.CmpIOp,
    arith.ExtFOp,
    arith.ExtSIOp,
    arith.ExtUIOp,
    arith.FPToSIOp,
    arith.FPToUIOp,
    arith.MaximumFOp,
    arith.MaxUIOp,
    arith.MaxSIOp,
    arith.MinimumFOp,
    arith.MinUIOp,
    arith.MinSIOp,
    arith.MulIOp,
    arith.MulFOp,
    arith.OrIOp,
    arith.FloorDivSIOp,
    arith.DivUIOp,
    arith.DivFOp,
    arith.RemUIOp,
    arith.RemSIOp,
    arith.RemFOp,
    arith.SIToFPOp,
    arith.UIToFPOp,
    arith.SubIOp,
    arith.SubFOp,
    arith.TruncFOp,
    arith.TruncIOp,
    arith.XOrIOp,
    mlir_math.ExpOp,
    mlir_math.Exp2Op,
    mlir_math.LogOp,
    mlir_math.RsqrtOp,
    mlir_math.TanhOp,
    vector.LoadOp,
    vector.StoreOp,
]:
  _add_layout_inference_rule(op, _infer_pointwise_op_layouts)


# TODO(bchetioui): remove once minimum jaxlib >= 0.5.3.
OptimizationBarrierOp = getattr(mgpu, "OptimizationBarrierOp", None)


@partial(_add_layout_inference_rule, OptimizationBarrierOp)
def _infer_optimization_barrier_op_layout(
    op: OptimizationBarrierOp,
) -> OptionalLayouts:
  def is_array(v: ir.Value) -> bool:
    return ir.VectorType.isinstance(v.type)

  if inference_utils.has_in_layouts_set(op):
    op_in_layouts = list(inference_utils.in_layouts(op))
    return op_in_layouts, op_in_layouts

  if inference_utils.has_out_layouts_set(op):
    op_out_layouts = list(inference_utils.out_layouts(op))
    return op_out_layouts, op_out_layouts

  layouts = [None] * len(op.operands)
  for i, operand in enumerate(filter(is_array, op.operands)):
    layouts[i] = inference_utils.value_layout(operand)

  for i, result in enumerate(filter(is_array, op.results)):
    possible_layouts = set()
    for op_operand_use in cast(ir.OpResult, result).uses:
      consumer = op_operand_use.owner
      op_user = consumer.operands[op_operand_use.operand_number]
      layout = inference_utils.in_layout_for_operand(consumer, op_user)
      if layout is not None:
        possible_layouts.add(layout)
      if possible_layouts and layouts[i] is None:
        # TODO(bchetioui): we could actually just pick any user layout here,
        # and optimize later. This is fine for now.
        layouts[i] = _choose_representative_layout(possible_layouts)

  # TODO(bchetioui): handle annotating layout for only certain operands.
  # Otherwise, layouts may not get propagated through optimization barriers, if
  # a single branch does not carry any forcing layout, which is pretty bad.
  if any(layout is None for layout in layouts):
    return None

  return layouts, layouts


@partial(_add_layout_inference_rule, arith.ConstantOp)
def _infer_constant_op_layout(constant_op: arith.ConstantOp) -> OptionalLayouts:
  if not ir.VectorType.isinstance(constant_op.result.type):
    return None

  shaped_ty = cast(ir.ShapedType, constant_op.result.type)
  value = constant_op.value
  layout = None
  if (
      ir.DenseElementsAttr.isinstance(value)
      and ir.DenseElementsAttr(value).is_splat
  ):
    layout = layouts_lib.to_splat_fragmented_layout_attr(
        fa.WGSplatFragLayout(shape=shaped_ty.shape)
    )
  # If the constant is not a splat, there is no obvious good choice of layout.
  # We need to look at the consumers of the constant to find a layout that works
  # for them. If there are several users with N different layouts, we can
  # arbitrarily choose any one of them for the constant, since we expect
  # whichever choice we make to lead to N-1 relayouts, which all have the same
  # cost.
  #
  # We assign a strided layout if the constant has no user, for completeness.
  elif constant_op.result.uses:
    for use in cast(ir.OpResult, constant_op.result).uses:
      consumer = use.owner
      operand = consumer.operands[use.operand_number]
      layout = inference_utils.in_layout_for_operand(consumer, operand)
      if layout is not None:
        break

  # If the constant is not a splat, has no user, or a layout could not be
  # determined from looking at the users, we assign a strided layout for
  # completeness.
  if layout is None:
    layout = layouts_lib.to_strided_fragmented_layout_attr(
        fa.WGStridedFragLayout.from_shaped_type(shaped_ty)
    )

  return [], [layout]


def _layouts_from_values(values: Sequence[ir.Value]) -> list[ir.Attribute] | None:
  layouts = []
  for value in values:
    if not ir.VectorType.isinstance(value.type):
      continue
    if (layout := inference_utils.value_layout(value)) is not None:
      if layouts_lib.is_splat_fragmented_layout(layout):
        return None
      layouts.append(layout)
    else:
      # Not all layouts could be inferred for vector ops. Return for now.
      return None
  return layouts

@partial(_add_layout_inference_rule, scf.YieldOp)
def _infer_yield_op_layout(op: scf.YieldOp) -> OptionalLayouts:
  layouts = _layouts_from_values(op.results_)
  if layouts is None:
    return None
  return (layouts, [])


@partial(_add_layout_inference_rule, scf.ConditionOp)
def _infer_condition_op_layout(op: scf.ConditionOp) -> OptionalLayouts:
  layouts = _layouts_from_values(op.args)
  if layouts is None:
    return None
  return (layouts, [])


def _last_op(region: ir.Region, expected_op_type: type[ir.OpView]):
  [block] = region.blocks
  last_op = block.operations[len(block.operations) - 1]
  assert isinstance(last_op, expected_op_type)
  return last_op


def _infer_from_op(op: ir.OpView) -> list[ir.Attribute] | None:
  if not inference_utils.has_in_layouts_set(op):
    return None
  in_layouts = list(inference_utils.in_layouts(op))
  if any(
      layouts_lib.is_splat_fragmented_layout(layout)
      for layout in in_layouts
  ):
    return None
  return in_layouts


def _infer_from_yield_ops(op: ir.Operation) -> list[ir.Attribute] | None:
  candidates = []
  for region in op.regions:
    yield_layouts = _infer_from_op(_last_op(region, scf.YieldOp))
    if yield_layouts is not None:
      candidates.append(yield_layouts)
  if not candidates:
    return None
  return [_choose_representative_layout(set(c)) for c in zip(*candidates)]


@partial(_add_layout_inference_rule, scf.ForOp)
def _infer_for_op_layout(op: scf.ForOp) -> OptionalLayouts:
  # TODO(bchetioui): we don't attempt to propagate from outside for the moment.
  # For the existing kernels, propagating from the YieldOp should be enough.
  if layouts := _infer_from_yield_ops(op):
    return layouts, layouts
  return None


@partial(_add_layout_inference_rule, scf.WhileOp)
def _infer_while_op_layout(op: scf.WhileOp) -> OptionalLayouts:
  # TODO(dasenov): we don't attempt to propagate from outside for the moment.

  # Note that the inputs or results do not necessarily contain vector types. If
  # there is no vector type, the corresponding layouts (in_layouts or
  # out_layouts) should be an empty list.

  yield_op = _last_op(op.after, scf.YieldOp)
  needs_in_layouts = inference_utils.should_have_layout(yield_op)
  in_layouts = _infer_from_op(yield_op) if needs_in_layouts else []

  condition_op = _last_op(op.before, scf.ConditionOp)
  needs_out_layouts = inference_utils.should_have_layout(condition_op)
  out_layouts = _infer_from_op(condition_op) if needs_out_layouts else []

  if in_layouts is None or out_layouts is None:
    return None
  return in_layouts, out_layouts


@partial(_add_layout_inference_rule, scf.IfOp)
def _infer_if_op_layout(op: scf.IfOp) -> OptionalLayouts:
  if layouts := _infer_from_yield_ops(op):
    return [], layouts
  return None


@partial(_add_layout_inference_rule, scf.IndexSwitchOp)
def _infer_index_switch_op_layout(op: scf.IndexSwitchOp) -> OptionalLayouts:
  if layouts := _infer_from_yield_ops(op):
    return [], layouts
  return None


@partial(_add_layout_inference_rule, vector.SplatOp)
def _infer_splat_op_layout(splat_op: vector.SplatOp) -> OptionalLayouts:
  layout = layouts_lib.to_splat_fragmented_layout_attr(
      fa.WGSplatFragLayout(
          shape=cast(ir.ShapedType, splat_op.result.type).shape
      )
  )
  return [], [layout]


def _update_layout_shape(
    layout: ir.Attribute, shape: Sequence[int], origin: str
) -> ir.Attribute:
  if layouts_lib.is_splat_fragmented_layout(
      layout
  ) or layouts_lib.is_strided_fragmented_layout(layout):
    return layouts_lib.to_layout_attr(
        dataclasses.replace(layouts_lib.from_layout_attr(layout), shape=shape)
    )
  raise NotImplementedError(f"Unsupported {origin} layout: {layout}.")


@partial(_add_layout_inference_rule, vector.ShapeCastOp)
def _infer_shape_cast_op_layout(op: vector.ShapeCastOp) -> OptionalLayouts:
  in_layout = inference_utils.value_layout(op.source)
  if in_layout is None:
    out_layout = inference_utils.value_layout(op.result)
    if out_layout is None:
      return None
    in_layout = _update_layout_shape(
        out_layout, ir.VectorType(op.source.type).shape, "source"
    )
    return [in_layout], [out_layout]

  out_layout = _update_layout_shape(
      in_layout, ir.VectorType(op.result.type).shape, "result"
  )
  return [in_layout], [out_layout]


@partial(_add_layout_inference_rule, vector.ReductionOp)
def _infer_reduction_op_layout(op: vector.ReductionOp) -> OptionalLayouts:
  if layout := inference_utils.value_layout(op.vector):
    return [layout], []
  return None


@partial(_add_layout_inference_rule, vector.MultiDimReductionOp)
def _infer_multi_dim_reduction_op_layout(
    op: vector.MultiDimReductionOp,
) -> OptionalLayouts:
  if inference_utils.has_any_layout_set(op):
    # At the moment we either have all layouts or none. So if we found some
    # layouts, set just return the same ones.
    op_in_layouts = list(inference_utils.in_layouts(op))
    op_out_layouts = list(inference_utils.out_layouts(op))
    return op_in_layouts, op_out_layouts

  in_ty = ir.VectorType(op.source.type)
  out_ty = ir.VectorType(op.result.type)
  if len(in_ty.shape) != 2 or len(out_ty.shape) != 1:
    raise NotImplementedError(
        f"Only 2D -> 1D reductions are supported: {op}"
    )

  wgmma_layout = layouts_lib.to_layout_attr(fa.WGMMA_LAYOUT)
  wgmma_row_layout = layouts_lib.to_layout_attr(fa.WGMMA_ROW_LAYOUT)
  wgmma_col_layout = layouts_lib.to_layout_attr(fa.WGMMA_COL_LAYOUT)
  reduction_dims = list(op.reduction_dims)

  # Find out the layout of the source.
  in_layout = inference_utils.value_layout(op.source)
  if in_layout is not None and in_layout == wgmma_layout:
    if reduction_dims == [0]:
      out_layout = wgmma_col_layout
    elif reduction_dims == [1]:
      out_layout = wgmma_row_layout
    else:
      raise NotImplementedError(
          f"Invalid reduction dimensions: {reduction_dims}"
      )
    return [in_layout, out_layout], [out_layout]

  # The source either has no layout or its layout is not WGMMA so we don't know
  # yet how to handle it. Find out the layout of the result and see if that is
  # WGMMA_ROW or WGMMA_COL which would imply the input is WGMMA. We can look at
  # either the consumers or the acc input (they should have the same layout).
  out_layouts = set()

  # Get acc layout.
  acc_layout = inference_utils.value_layout(op.acc)
  if acc_layout is not None:
    out_layouts.add(acc_layout)

  # Get user layouts.
  for use in cast(ir.OpResult, op.result).uses:
    consumer = use.owner
    operand = consumer.operands[use.operand_number]
    layout = inference_utils.in_layout_for_operand(consumer, operand)
    if layout:
      out_layouts.add(layout)

  if not out_layouts:
    # We couldn't find any definitive layouts, so we can't infer anything.
    return None

  out_layout = _choose_representative_layout(out_layouts)
  if out_layout is None:
    raise NotImplementedError(
        f"Could not choose a best layout from {out_layouts}"
    )
  if out_layout != wgmma_row_layout and out_layout != wgmma_col_layout:
    # We don't have a layout we can handle in the output, so we can't infer
    # anything.
    return None

  if (out_layout == wgmma_row_layout and reduction_dims == [1]) or (
      out_layout == wgmma_col_layout and reduction_dims == [0]
  ):
    in_layout = wgmma_layout
  else:
    raise NotImplementedError(
        f"Unsupported output layout: {out_layout} for reduction dimensions"
        f" {reduction_dims}"
    )

  return [in_layout, out_layout], [out_layout]


@partial(_add_layout_inference_rule, mgpu.LayoutCastOp)
def _infer_layout_cast_op_layout(
    layout_cast_op: mgpu.LayoutCastOp,
) -> OptionalLayouts:
  return [layout_cast_op.new_layout], [layout_cast_op.new_layout]


# TODO(dasenov): Remove this after the minimal jaxlib version is 0.6.1.
if hasattr(mgpu, "BroadcastInDimOp"):
  @partial(_add_layout_inference_rule, mgpu.BroadcastInDimOp)
  def _infer_broadcast_in_dim_op_layout(
      op: mgpu.BroadcastInDimOp,
  ) -> OptionalLayouts:
    if inference_utils.has_any_layout_set(op):
      op_in_layouts = list(inference_utils.in_layouts(op))
      op_out_layouts = list(inference_utils.out_layouts(op))
      return op_in_layouts, op_out_layouts

    in_ty = ir.VectorType(op.operand.type)
    out_ty = ir.VectorType(op.result.type)
    if len(in_ty.shape) != 1 or len(out_ty.shape) != 2:
      raise NotImplementedError(
          "Broadcast in dim with non-trivial broadcast dimensions is not"
          f" supported: {op}"
      )

    # Find out the layout of the output from the consumers.
    user_layouts = set()
    for use in cast(ir.OpResult, op.result).uses:
      consumer = use.owner
      operand = consumer.operands[use.operand_number]
      layout = inference_utils.in_layout_for_operand(consumer, operand)
      if layout is not None:
        user_layouts.add(layout)
    if user_layouts:
      out_layout = _choose_representative_layout(user_layouts)

      if out_layout is None:
        raise ValueError(f"Could not choose a best layout from {user_layouts}")

      if out_layout != layouts_lib.to_layout_attr(fa.WGMMA_LAYOUT):
        raise NotImplementedError(f"Unsupported layout: {out_layout}")

      broadcast_dims = list(op.broadcast_dimensions)
      if broadcast_dims == [0]:
        in_layout = layouts_lib.to_layout_attr(fa.WGMMA_ROW_LAYOUT)
      elif broadcast_dims == [1]:
        in_layout = layouts_lib.to_layout_attr(fa.WGMMA_COL_LAYOUT)
      else:
        raise ValueError(f"Invalid broadcast dimensions: {broadcast_dims}")

      return [in_layout], [out_layout]

    # The consumers did not have any layouts set. Find out the layout of the
    # input and infer the output layout from it.
    in_layout = inference_utils.value_layout(op.operand)
    if in_layout is None:
      return None

    broadcast_dims = list(op.broadcast_dimensions)
    if (
        broadcast_dims == [0]
        and in_layout == layouts_lib.to_layout_attr(fa.WGMMA_ROW_LAYOUT)
    ) or (
        broadcast_dims == [1]
        and in_layout == layouts_lib.to_layout_attr(fa.WGMMA_COL_LAYOUT)
    ):
      out_layout = layouts_lib.to_layout_attr(fa.WGMMA_LAYOUT)
      return [in_layout], [out_layout]
    else:
      raise NotImplementedError(
          f"Unsupported layout: {in_layout} for broadcast dimensions"
          f" {broadcast_dims}"
      )


@partial(_add_layout_inference_rule, mgpu.WGMMAOp)
def _infer_wgmma_op_layout(wgmma_op: mgpu.WGMMAOp) -> OptionalLayouts:
  layout = layouts_lib.to_layout_attr(fa.WGMMA_LAYOUT)

  if ir.VectorType.isinstance(wgmma_op.a.type):
    return [layout, layout], [layout]

  return [layout], [layout]


def _earliest_use(regions: list[ir.Region], uses: Sequence[ir.OpOperand]) -> ir.OpView:
  owners = [use.owner for use in uses]
  for region in regions:
    for block in region:
      for op in block:
        if op in owners:
          return op
  raise ValueError("None of uses are in the given block")


def _insert_memref_layout_cast(layout: ir.Attribute, view_op: memref.ViewOp):
  mem_ref_type = ir.MemRefType(view_op.result.type)
  memref_new_type = ir.MemRefType.get(
    mem_ref_type.shape,
    mem_ref_type.element_type,
    layout,
    mem_ref_type.memory_space,
  )
  uses = list(view_op.result.uses)
  with ir.InsertionPoint(_earliest_use(view_op.parent.regions, uses)):
    cast_op = memref.cast(memref_new_type, view_op.result)
  for use in uses:
    use.owner.operands[use.operand_number] = cast_op


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
    if not inference_utils.should_have_layout(op):
      return
    elif inference_rule := _layout_inference_rules.get(op.OPERATION_NAME, None):  # pytype: disable=attribute-error
      pass
    else:
      raise NotImplementedError(f"Can not infer layout for {op}")

    maybe_layouts = inference_rule(op)
    if maybe_layouts is None:
      return

    _set_layout_attributes(op, *maybe_layouts)

  # TODO(bchetioui): consider switching the order of the passes. This would
  # allow propagating "simpler" layouts further down in the computation, which
  # is more efficient when possible.
  #
  # We run two passes over the module, in order to make sure that layouts
  # defined in the middle of the computation are propagated wherever they need
  # to be propagated. We start with a backwards (root-to-parameters) pass to
  # propagate the information as far up as possible, and then a forward pass
  # (parameters-to-root).
  #
  # Backwards pass
  for op in module.body:
    inference_utils.traverse_op(
        op, inference_step, inference_utils.TraversalOrder.BACKWARDS
    )

  # Forward pass
  for op in module.body:
    inference_utils.traverse_op(
        op, inference_step, inference_utils.TraversalOrder.FORWARD
    )

  # At this point, layouts have been propagated as far as they could be
  # propagated. However, it is possible for some operations to remain
  # unannotated---for example, if there were no annotations on any operation in
  # the module at the start of this function. We annotate all the remaining ops
  # that should be annotated with a strided fragmented layout, whose vector size
  # is derived from the narrowest type and vector size used in the program. We
  # make sure to derive a single vector size in order to avoid relayouts at
  # lowering time.
  default_vector_size = math.inf
  def update_default_vector_size_from_vector(v: ir.Value):
    nonlocal default_vector_size
    max_vec_size_for_v = (
          np.prod(cast(ir.ShapedType, v.type).shape) // fa.WARPGROUP_SIZE
      )
    desired_vec_size = 64 // utils.bitwidth(v.type.element_type)
    default_vector_size = min(
        default_vector_size, max_vec_size_for_v, desired_vec_size
    )

  def update_default_vector_size_from_op(op: ir.OpView):
    for i, v in enumerate(
        filter(lambda v: ir.VectorType.isinstance(v.type), op.operands)
    ):
      if inference_utils.attr_element("in_layouts", op, i) is None:
        update_default_vector_size_from_vector(v)

    for i, v in enumerate(
        filter(lambda v: ir.VectorType.isinstance(v.type), op.results)
    ):
      if inference_utils.attr_element("out_layouts", op, i) is None:
        update_default_vector_size_from_vector(v)

  for op in module.body:
    traverse_op(op, update_default_vector_size_from_op)

  if default_vector_size == math.inf:  # Nothing to annotate.
    return

  def to_default_layout(ty: ir.Type) -> ir.Attribute | None:
    if not ir.VectorType.isinstance(ty):
      return None
    layout = fa.WGStridedFragLayout(
        shape=cast(ir.ShapedType, ty).shape, vec_size=default_vector_size
    )
    return layouts_lib.to_strided_fragmented_layout_attr(layout)

  def set_default_layout(op: ir.OpView):
    if inference_utils.should_have_layout(
        op
    ) and not inference_utils.has_any_layout_set(op):
      in_layouts = []
      for operand in op.operands:
        if (layout := to_default_layout(operand.type)) is not None:
          in_layouts.append(layout)

      out_layouts = []
      for result in op.results:
        if (layout := to_default_layout(result.type)) is not None:
          out_layouts.append(layout)

      _set_layout_attributes(op, in_layouts, out_layouts)

  for op in module.body:
    traverse_op(op, set_default_layout)
