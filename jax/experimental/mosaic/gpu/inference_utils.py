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

from jax._src.lib.mlir import ir


def in_layouts(op: ir.OpView) -> Sequence[ir.Attribute]:
  """Returns the in_layouts attribute of the given operation.

  Raises:
    ValueError: If the operation does not have an in_layouts attribute.
  """
  if "in_layouts" not in op.attributes:
    raise ValueError(f"{op} does not have an in_layouts attribute.")
  return op.attributes["in_layouts"]  # type: ignore


def out_layouts(op: ir.OpView) -> Sequence[ir.Attribute]:
  """Returns the out_layouts attribute of the given operation.

  Raises:
    ValueError: If the operation does not have an out_layouts attribute.
  """
  if "out_layouts" not in op.attributes:
    raise ValueError(f"{op} does not have an out_layouts attribute.")
  return op.attributes["out_layouts"]  # type: ignore


def should_have_layout(op: ir.OpView) -> bool:
  """Returns 'true' if the operation should be assigned a layout."""

  is_array = lambda v: ir.VectorType.isinstance(v.type)
  return any(map(is_array, itertools.chain(op.operands, op.results)))  # type: ignore


def has_in_layouts_set(op: ir.OpView) -> bool:
  return "in_layouts" in op.attributes


def has_out_layouts_set(op: ir.OpView) -> bool:
  return "out_layouts" in op.attributes


def has_any_layout_set(op: ir.OpView) -> bool:
  return has_in_layouts_set(op) or has_out_layouts_set(op)


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
