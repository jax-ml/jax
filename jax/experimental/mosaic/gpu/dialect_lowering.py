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

"""Lowering rules and pass for the MLIR Mosaic GPU dialect."""

from collections.abc import Callable
import enum
import functools
import itertools
import operator
from typing import List, Sequence, Tuple, Type, cast

from jax._src.interpreters import mlir as mlir_interpreter
from jax._src.lib import mosaic_gpu_dialect as mgpu
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import gpu
from jax._src.lib.mlir.dialects import llvm
from jax._src.lib.mlir.dialects import nvvm

from .utils import c, ptr_as_memref, single_thread_predicate

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
        callback(block_op)
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


MlirLoweringRule = Callable[[ir.Operation | ir.OpView], Sequence[ir.Value]]


_lowerings: dict[str, MlirLoweringRule] = {}


# TODO(bchetioui): Remove this when minimum jaxlib version >= 0.4.36.
# Jaxlib doesn't contain Mosaic GPU dialect bindings.
InitializeBarrierOp = mgpu.InitializeBarrierOp if mgpu is not None else None

def _register_lowering(
    op: str | Type[ir.OpView]
) -> Callable[[MlirLoweringRule], MlirLoweringRule]:
  def wrapper(f):
    op_name = op if isinstance(op, str) else op.OPERATION_NAME  # pytype: disable=attribute-error
    _lowerings[op_name] = f
    return f

  return wrapper


def _lowered_barrier_type() -> ir.Type:
  return ir.IntegerType.get_signless(64)


def gpu_address_space_to_nvptx(address_space: gpu.AddressSpace) -> int:
  match address_space:
    case gpu.AddressSpace.Global:
      return 1
    case gpu.AddressSpace.Workgroup:
      return 3
    case _:
      raise NotImplementedError(f"address_space not supported: {address_space}")


@_register_lowering(InitializeBarrierOp)
def _initialize_barrier_op_lowering_rule(
    initialize_barrier_op: InitializeBarrierOp) -> Sequence[ir.Value]:

  shape = initialize_barrier_op.barriers_ref.type.shape
  num_barriers = functools.reduce(operator.mul, shape, 1)

  i32 = ir.IntegerType.get_signless(32)
  workgroup_nvptx_address_space = gpu_address_space_to_nvptx(
      gpu.AddressSpace.Workgroup)
  ptr_ty = ir.Type.parse(f"!llvm.ptr<{workgroup_nvptx_address_space}>")

  lowered_barrier_type = _lowered_barrier_type()

  predicate = single_thread_predicate(per_block=True)
  for i in range(num_barriers):
    nvvm.mbarrier_init_shared(
        llvm.getelementptr(ptr_ty, initialize_barrier_op.base_pointer, [], [i],
                           lowered_barrier_type),
        c(initialize_barrier_op.arrival_count.value, i32),
        predicate=predicate
    )

  barrier_base_ptr = llvm.getelementptr(
      ir.Type.parse("!llvm.ptr"),
      initialize_barrier_op.base_pointer, [], [0], lowered_barrier_type)

  return ptr_as_memref(
      barrier_base_ptr, initialize_barrier_op.barriers_ref.type),


def lower_mgpu_dialect(module: ir.Module):
  module.context.append_dialect_registry(mlir_interpreter.upstream_dialects)
  module.context.load_all_available_dialects()

  lowered_operations: set[ir.Operation | ir.OpView] = set()

  def _lower_op(op: ir.OpView):
    if op.name not in _lowerings:
      return
    lowering_rule = _lowerings[op.name]
    new_results = lowering_rule(op)
    for old, new in zip(op.results, new_results):
      old.replace_all_uses_with(new)
    lowered_operations.add(op)

  def _traverse_and_lower_op(op: ir.OpView):
    for region in op.operation.regions:
      for block in region:
        for block_op in list(block):
          with ir.InsertionPoint(block_op):
            _traverse_and_lower_op(block_op)
    _lower_op(op)

  with ir.InsertionPoint(module.body):
    for op in module.body:
      _traverse_and_lower_op(op)

  for lowered_op in lowered_operations:
    lowered_op.erase()
