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
from functools import partial
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


def in_transforms(op: MlirOperation) -> Sequence[ir.Attribute]:
  """Returns the in_transforms attribute of the given operation.

  Raises:
    ValueError: If the operation does not have an in_transforms attribute.
  """
  if "in_transforms" not in op.attributes:
    raise ValueError(f"{op} does not have an in_transforms attribute.")
  return op.attributes["in_transforms"]  # type: ignore


def out_transforms(op: MlirOperation) -> Sequence[ir.Attribute]:
  """Returns the out_transforms attribute of the given operation.

  Raises:
    ValueError: If the operation does not have an out_transforms attribute.
  """
  if "out_transforms" not in op.attributes:
    raise ValueError(f"{op} does not have an out_transforms attribute.")
  return op.attributes["out_transforms"]  # type: ignore


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


def has_in_transforms_set(op: MlirOperation) -> bool:
  return "in_transforms" in op.attributes


def has_out_transforms_set(op: MlirOperation) -> bool:
  return "out_transforms" in op.attributes


def attr_element(
    attr_name: str, op: MlirOperation, index: int
) -> ir.Attribute | None:
  """Returns `op.attributes[attr_name][index]` if it exists, otherwise None.

  If `op.attributes[attr_name]` exists, then `index` must be a valid index into
  the attribute array.
  """
  if attr_name not in op.attributes:
    return None
  attr = op.attributes[attr_name]
  if not attr:
    return None
  return op.attributes[attr_name][index]  # type: ignore


def _in_attr_for_operand(
    op: MlirOperation,
    operand: ir.Value,
    attr_name: str,
) -> ir.Attribute | None:
  if attr_name == "in_layouts":
    predicate = lambda v: ir.VectorType.isinstance(v.type)
  elif attr_name == "in_transforms":
    predicate = is_transformable_smem_memref
  else:
    raise ValueError(f"Unknown attribute: {attr_name}")

  operand_number = [o for o in op.operands if predicate(o)].index(operand)

  return attr_element(attr_name, op, operand_number)


in_layout_for_operand = partial(
    _in_attr_for_operand, attr_name="in_layouts"
)
in_transforms_for_operand = partial(
    _in_attr_for_operand, attr_name="in_transforms"
)

def is_transformable_smem_memref(v: ir.Value) -> bool:
  """Whether the value is a memref in SMEM on which transforms should be applied."""
  barrier_ty = ir.Type.parse("!mosaic_gpu.barrier")
  smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
  return (
      ir.MemRefType.isinstance(v.type)
      # barriers have no business being transformed
      and v.type.element_type != barrier_ty  # pylint: disable=attribute-error
      and v.type.memory_space is not None  # pylint: disable=attribute-error
      and v.type.memory_space == smem  # pylint: disable=attribute-error
  )


def _value_attr(value: ir.Value, attr_type: str) -> ir.Attribute | None:
  if attr_type == "layouts":
    predicate = lambda v: ir.VectorType.isinstance(v.type)
  elif attr_type == "transforms":
    predicate = is_transformable_smem_memref
  else:
    raise ValueError(f"Unknown attribute: {attr_type}")

  in_attr_type = "in_" + attr_type
  out_attr_type = "out_" + attr_type

  owner = value.owner
  if isinstance(owner, ir.Operation):
    if out_attr_type not in owner.attributes:
      return None
    value_result_number = [r for r in owner.results if predicate(r)].index(
        value
    )
    return owner.attributes[out_attr_type][value_result_number]  # type: ignore

  # Block case, useful when attempting to derive layouts for ops
  # depending on function parameters, or loop block arguments.
  if isinstance(owner, ir.Block):
    owner_op = owner.owner
    block = cast(ir.Block, owner)
    if in_attr_type not in owner_op.attributes:
      return None
    value_arg_number = [r for r in block.arguments if predicate(r)].index(value)
    return owner_op.attributes[in_attr_type][value_arg_number]  # type: ignore

  raise NotImplementedError(
      f"{owner} is not a function block nor an operation."
  )


def value_layout(value: ir.Value) -> ir.Attribute | None:
  """Returns the layout for a given value as defined by its owner.

  Raises:
    ValueError: If `result` is not a Vector.
  """
  if not ir.VectorType.isinstance(value.type):
    raise ValueError(f"{value} is not a vector.")

  return _value_attr(value, "layouts")


def value_transforms(value: ir.Value) -> ir.Attribute | None:
  """Returns the transforms for a given value as defined by its owner.

  Raises:
    ValueError: If `result` is not a memref.
  """
  if not ir.MemRefType.isinstance(value.type):
    raise ValueError(f"{value} is not a memref.")

  return _value_attr(value, "transforms")


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
