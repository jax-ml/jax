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

from collections.abc import Sequence
from functools import partial
from typing import cast, Union

from jax._src.lib.mlir import ir

from . import fragmented_array as fa
from . import tcgen05
from . import utils

MlirOperation = Union[ir.Operation, ir.OpView]

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


def in_tmem_layouts(op: MlirOperation) -> Sequence[ir.Attribute]:
  """Returns the in_tmem_layouts attribute of the given operation.

  Raises:
    ValueError: If the operation does not have an in_tmem_layouts attribute.
  """
  if "in_tmem_layouts" not in op.attributes:
    raise ValueError(f"{op} does not have an in_tmem_layouts attribute.")
  return op.attributes["in_tmem_layouts"]  # type: ignore


def out_tmem_layouts(op: MlirOperation) -> Sequence[ir.Attribute]:
  """Returns the out_tmem_layouts attribute of the given operation.

  Raises:
    ValueError: If the operation does not have an out_tmem_layouts attribute.
  """
  if "out_tmem_layouts" not in op.attributes:
    raise ValueError(f"{op} does not have an out_tmem_layouts attribute.")
  return op.attributes["out_tmem_layouts"]  # type: ignore


def should_have_in_tmem_layout(op: MlirOperation) -> bool:
  """Returns 'true' if the operation operands should be assigned a TMEM layout."""
  return any(
      isinstance(v.type, ir.MemRefType) and utils.is_tmem_ref(v)
      for v in op.operands
  )


def should_have_out_tmem_layout(op: MlirOperation) -> bool:
  """Returns 'true' if the operation results should be assigned a TMEM layout."""
  return any(
      isinstance(v.type, ir.MemRefType) and utils.is_tmem_ref(v)
      for v in op.results
  )


def should_have_tmem_layout(op: MlirOperation) -> bool:
  """Returns 'true' if the operation should be assigned a TMEM layout."""
  return should_have_in_tmem_layout(op) or should_have_out_tmem_layout(op)


def has_in_tmem_layouts_set(op: MlirOperation) -> bool:
  return "in_tmem_layouts" in op.attributes


def has_out_tmem_layouts_set(op: MlirOperation) -> bool:
  return "out_tmem_layouts" in op.attributes


def should_have_in_layout(op: MlirOperation) -> bool:
  """Returns 'true' if the operation operands should be assigned a layout."""
  return any(isinstance(v.type, ir.VectorType) for v in op.operands)


def should_have_out_layout(op: MlirOperation) -> bool:
  """Returns 'true' if the operation results should be assigned a layout."""
  return any(isinstance(v.type, ir.VectorType) for v in op.results)


def should_have_layout(op: MlirOperation) -> bool:
  """Returns 'true' if the operation should be assigned a layout."""
  return should_have_in_layout(op) or should_have_out_layout(op)


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
    predicate = lambda v: isinstance(v.type, ir.VectorType)
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


def should_have_in_transforms(op: ir.OpView) -> bool:
  """Returns 'True' if the operation should be assigned in transforms."""
  return any(map(is_transformable_smem_memref, op.operands))


def should_have_out_transforms(op: ir.OpView) -> bool:
  """Returns 'True' if the operation should be assigned out transforms."""
  return any(map(is_transformable_smem_memref, op.results))


def should_have_transforms(op: ir.OpView) -> bool:
  """Returns 'True' if the operation should be assigned in/out transforms."""
  return should_have_in_transforms(op) or should_have_out_transforms(op)


def is_transformable_smem_memref(v: ir.Value) -> bool:
  """Whether the value is a memref in SMEM on which transforms should be applied."""
  barrier_ty = ir.Type.parse("!mosaic_gpu.barrier")
  return (
      isinstance(v.type, ir.MemRefType)
      # barriers have no business being transformed
      and v.type.element_type != barrier_ty  # pylint: disable=attribute-error
      and utils.is_smem_ref(v)
  )


def _value_attr(value: ir.Value, attr_type: str) -> ir.Attribute | None:
  if attr_type == "layouts":
    predicate = lambda v: isinstance(v.type, ir.VectorType)
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


def is_mma_layout(layout: fa.FragmentedLayout) -> bool:
  if not isinstance(layout, fa.TiledLayout):
    return False
  if layout in {
      fa.WGMMA_LAYOUT,
      fa.WGMMA_LAYOUT_ACC_32BIT,
      fa.WGMMA_LAYOUT_UPCAST_2X,
      fa.WGMMA_LAYOUT_UPCAST_4X,
      fa.WGMMA_TRANSPOSED_LAYOUT,
      fa.WGMMA_LAYOUT_8BIT,
      fa.TCGEN05_LAYOUT,
      fa.TCGEN05_TRANSPOSED_LAYOUT,
  }:
    return True
  if len(layout.tiling.tiles[0]) != 2:
    return False
  columns = layout.tiling.tiles[0][1]
  return columns % 16 == 0 and (
      layout == tcgen05.fa_m64_collective_layout(columns)
  )
