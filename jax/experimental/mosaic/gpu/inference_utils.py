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
from typing import Union

from jax._src.lib import mosaic_gpu_dialect as mgpu
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
  return _array_attr(op, "in_layouts")


def out_layouts(op: MlirOperation) -> Sequence[ir.Attribute]:
  """Returns the out_layouts attribute of the given operation.

  Raises:
    ValueError: If the operation does not have an out_layouts attribute.
  """
  return _array_attr(op, "out_layouts")


def in_transforms(op: MlirOperation) -> Sequence[ir.ArrayAttr]:
  """Returns the in_transforms attribute of the given operation.

  Raises:
    ValueError: If the operation does not have an in_transforms attribute.
  """
  return _array_attr(op, "in_transforms")  # pyrefly: ignore[bad-return]


def out_transforms(op: MlirOperation) -> Sequence[ir.ArrayAttr]:
  """Returns the out_transforms attribute of the given operation.

  Raises:
    ValueError: If the operation does not have an out_transforms attribute.
  """
  return _array_attr(op, "out_transforms")  # pyrefly: ignore[bad-return]


def in_tmem_layouts(op: MlirOperation) -> Sequence[ir.Attribute]:
  """Returns the in_tmem_layouts attribute of the given operation.

  Raises:
    ValueError: If the operation does not have an in_tmem_layouts attribute.
  """
  return _array_attr(op, "in_tmem_layouts")


def out_tmem_layouts(op: MlirOperation) -> Sequence[ir.Attribute]:
  """Returns the out_tmem_layouts attribute of the given operation.

  Raises:
    ValueError: If the operation does not have an out_tmem_layouts attribute.
  """
  return _array_attr(op, "out_tmem_layouts")


def _array_attr(op: MlirOperation, name: str) -> Sequence[ir.Attribute]:
  try:
    result = op.attributes[name]
  except KeyError:
    raise ValueError(f"{op} does not have an {name} attribute") from None
  if not isinstance(result, ir.ArrayAttr):
    raise TypeError(f"{op} has {name} of an unexpected type: {result}")
  return result  # pyrefly: ignore[bad-return]


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
  assert isinstance(attr, ir.ArrayAttr)
  return attr[index]


def _in_attr_for_operand(
    op: MlirOperation,
    operand: ir.Value,
    attr_name: str,
) -> ir.Attribute | None:
  if attr_name == "in_layouts":
    predicate = lambda v: isinstance(v.type, ir.VectorType)
  elif attr_name == "in_transforms":
    predicate = is_transformable_smem_memref
  elif attr_name == "in_tmem_layouts":
    predicate = (
        lambda v: isinstance(v.type, ir.MemRefType)
        and ir.MemRefType(v.type).memory_space == utils.tmem()
    )
  else:
    raise ValueError(f"Unknown attribute: {attr_name}")

  operand_number = [o for o in op.operands if predicate(o)].index(operand)

  return attr_element(attr_name, op, operand_number)


in_layout_for_operand = partial(
    _in_attr_for_operand, attr_name="in_layouts"
)
in_tmem_layout_for_operand = partial(
    _in_attr_for_operand, attr_name="in_tmem_layouts"
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
  return (
      isinstance(v.type, ir.MemRefType)
      # barriers have no business being transformed
      and not isinstance(v.type.element_type, mgpu.BarrierType)
      and utils.is_smem_ref(v)
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


def compute_swizzle(minor_tiling: int, bitwidth: int) -> int:
  """Computes the swizzle for the given minor tiled dimension and bitwidth."""
  tiling_bitwidth = minor_tiling * bitwidth
  if tiling_bitwidth % 8:
    raise ValueError("Minor tiling dimension is not byte aligned. "
                     f"Got {minor_tiling} elements of {bitwidth} bits.")
  tiling_bytewidth = tiling_bitwidth // 8
  # Do not swizzle if the bytewidth of the minor tiling dimension does not
  # exactly match a swizzle width.
  if tiling_bytewidth in [128, 64, 32]:
    return tiling_bytewidth
  return 16  # no swizzle
