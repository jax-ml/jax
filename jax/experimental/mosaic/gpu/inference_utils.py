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

"""Layout & transform inference convenience utils."""

from collections.abc import Callable, Sequence
import enum
import itertools
from typing import cast

from jax._src.lib.mlir import ir

MlirOperation = ir.Operation | ir.OpView

def in_layouts(op: MlirOperation) -> Sequence[ir.Attribute]:
  """Returns the in_layouts attribute of the given operation.

  Raises:
    ValueError: If the operation does not have an in_layouts attribute.
  """
  if "in_layouts" not in op.attributes:
    raise ValueError(f"{op} does not have an in_layouts attribute.")
  return op.attributes["in_layouts"]  # type: ignore


def out_layouts(op: MlirOperation) -> Sequence[ir.Attribute]:
  """Returns the out_layouts attribute of the given operation.

  Raises:
    ValueError: If the operation does not have an out_layouts attribute.
  """
  if "out_layouts" not in op.attributes:
    raise ValueError(f"{op} does not have an out_layouts attribute.")
  return op.attributes["out_layouts"]  # type: ignore


def should_have_layout(op: MlirOperation) -> bool:
  """Returns 'true' if the operation should be assigned a layout."""

  is_array = lambda v: ir.VectorType.isinstance(v.type)
  return any(map(is_array, itertools.chain(op.operands, op.results)))  # type: ignore


def has_in_layouts_set(op: MlirOperation) -> bool:
  return "in_layouts" in op.attributes


def has_out_layouts_set(op: MlirOperation) -> bool:
  return "out_layouts" in op.attributes


def has_any_layout_set(op: MlirOperation) -> bool:
  return has_in_layouts_set(op) or has_out_layouts_set(op)


def in_layout_for_operand(
    op: MlirOperation,
    operand: ir.Value,
) -> ir.Attribute | None:
  """Returns the layout of the operand in the given operation if it is set.

  Raises:
    ValueError: If `operand` is not an operand of `op`, or if `operand` is not a
      Vector.
  """
  if not ir.VectorType.isinstance(operand.type):
    raise ValueError(f"{operand} is not a vector.")

  operand_number = [
      o for o in op.operands if ir.VectorType.isinstance(o.type)
  ].index(operand)

  if not has_in_layouts_set(op):
    return None

  return in_layouts(op)[operand_number]


def value_layout(value: ir.Value) -> ir.Attribute | None:
  """Returns the layout for a given value as defined by its owner.

  Raises:
    ValueError: If `result` is not a Vector.
  """
  if not ir.VectorType.isinstance(value.type):
    raise ValueError(f"{value} is not a vector.")

  owner = value.owner
  if isinstance(owner, ir.Operation):
    if not has_out_layouts_set(owner):
      return None
    value_result_number = [
        r for r in owner.results if ir.VectorType.isinstance(r.type)
    ].index(value)
    return out_layouts(owner)[value_result_number]

  # Block case, useful when attempting to derive layouts for ops
  # depending on function parameters, or loop block arguments.
  if isinstance(owner, ir.Block):
    owner_op = owner.owner
    block = cast(ir.Block, owner)
    if not has_in_layouts_set(owner_op):
      return None
    value_arg_number = [
        r for r in block.arguments if ir.VectorType.isinstance(r.type)
    ].index(value)
    return in_layouts(owner_op)[value_arg_number]

  raise NotImplementedError(
      f"{owner} is not a function block nor an operation."
  )


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
        ops_to_traverse = list(block)
      else:
        ops_to_traverse = reversed(list(block))  # type: ignore
      for block_op in ops_to_traverse:
        traverse_op(block_op, callback, traversal_order)
  callback(op)
