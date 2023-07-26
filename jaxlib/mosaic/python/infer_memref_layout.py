# Copyright 2023 The JAX Authors.
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

"""Inference for memref layout and memory space."""

# mypy: ignore-errors
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import func
import numpy as np

from . import tpu


def _round_up(x: int, to: int):
  return ((x + to - 1) // to) * to


def _tiling_factor(
    num_128s: int, hardware_generation: int, bitwidth: int
) -> int:
  """Returns the number of 128-element groups in a tile.

  Arguments:
    num_128s: A number of 128-element groups in the full operand.
    hardware_generation: An integer indicating the target TPU generation.
    bitwidth: The bitwidth of the element type of the operand.
  """
  if num_128s == 1 and hardware_generation >= 4 and bitwidth == 32:
    return 1
  elif num_128s <= 2 and (
      bitwidth == 32 or (hardware_generation >= 4 and bitwidth == 16)
  ):
    return 2
  elif num_128s <= 4:
    return 4
  else:
    return 8


def infer_memref(
    memref: ir.MemRefType, hardware_generation: int
) -> ir.MemRefType:
  """Infers the layout and memory space attributes of the given memref.

  Arguments:
    memref: The memref type with potentially missing layout and memory space.
    hardware_generation: The TPU hardware generation to target.

  Returns:
    A memref type with layout and memory space filled in.
  """
  vmem = ir.Attribute.parse("#tpu.memory_space<vmem>")
  if tpu.private_has_no_memory_space(memref):
    memory_space = vmem
  else:
    memory_space = memref.memory_space
  if ir.AffineMapAttr.isinstance(memref.layout):
    if not tpu.private_is_identity(memref.layout):
      raise NotImplementedError("Non-identity affine layout")
    bitwidth: int
    if ir.BF16Type.isinstance(memref.element_type):
      bitwidth = 16
    elif ir.F32Type.isinstance(memref.element_type):
      bitwidth = 32
    elif ir.IntegerType.isinstance(memref.element_type):
      bitwidth = ir.IntegerType(memref.element_type).width
    else:
      raise NotImplementedError(
          f"Unrecognized element type: {memref.element_type}"
      )
    # Infer the layout.
    if memref.rank == 1:
      tile = _tiling_factor(
          ((memref.shape[-1] + 127) // 128), hardware_generation, bitwidth
      ) * 128
      if bitwidth == 32:
        trailing_tiles = ""
      else:
        if bitwidth.bit_count() != 1 or bitwidth > 32:
          raise NotImplementedError(f"Unsupported bitwidth: {bitwidth}")
        trailing_tiles = f"(128)({32 // bitwidth},1)"
      layout = ir.Attribute.parse(
          f"#tpu.tiled<{memref.rank},({tile}){trailing_tiles}>"
      )
    else:
      leading_tile = _tiling_factor(
          memref.shape[-2], hardware_generation, bitwidth
      )
      if bitwidth == 32:
        trailing_tiles = ""
      else:
        if bitwidth.bit_count() != 1 or bitwidth > 32:
          raise NotImplementedError(f"Unsupported bitwidth: {bitwidth}")
        trailing_tiles = f"({32 // bitwidth},1)"
      layout = ir.Attribute.parse(
          f"#tpu.tiled<{memref.rank},({leading_tile},128){trailing_tiles}>"
      )
  elif tpu.private_is_tiled_layout(memref.layout):
    layout = memref.layout
  else:
    raise NotImplementedError("Unrecognized layout annotation")
  new_shape = list(memref.shape)
  # Make sure only the first tile might introduce padding.
  first_tile, *_ = tiles = tpu.private_get_tiles(layout)
  tiled_dims = np.asarray(first_tile, dtype=np.int32)
  for t in tiles:
    t = np.asarray(t, dtype=np.int32)
    if len(t) > len(tiled_dims):
      raise NotImplementedError("Layout too complicated")
    untiled_prefix, tiled_suffix = tiled_dims[:-len(t)], tiled_dims[-len(t):]
    if np.any(tiled_suffix % t != 0):
      raise NotImplementedError("Layout too complicated")
    tiled_dims = np.concatenate([untiled_prefix, tiled_suffix // t, t])
  untiled_dims = len(new_shape) - len(first_tile)
  if untiled_dims < 0:
    raise ValueError("Invalid tiling")
  for i, size in enumerate(first_tile):
    new_shape[untiled_dims + i] = _round_up(
        new_shape[untiled_dims + i], to=size
    )
  return ir.MemRefType.get(
      new_shape, memref.element_type, layout, memory_space
  )


def infer_module(module: ir.Module, hardware_generation: int):
  """Infers the layout and memory space attributes of function memref arguments.

  In the future we should require those annotations from Mosaic users, but it's
  best to keep them internal for as long as they are under development.

  Arguments:
    module: The MLIR module on which to perform the inference.
    hardware_generation: The TPU hardware generation to target.
  """
  # TODO(apaszke): Do layout assignment for scoped allocations too.
  for f in module.body:
    assert isinstance(f, func.FuncOp)
    if len(f.body.blocks) != 1:
      raise ValueError("Functions should only have a single block")
    (entry,) = f.body.blocks
    new_arg_types = []
    with ir.InsertionPoint.at_block_begin(entry):
      for arg in entry.arguments:
        try:
          memref = ir.MemRefType(arg.type)
        except ValueError:
          new_arg_types.append(arg.type)
          continue
        new_memref = infer_memref(memref, hardware_generation)
        arg.set_type(new_memref)
        new_arg_types.append(new_memref)
        if memref != new_memref:
          # Some standard MLIR ops have static checks that seems unreasonable,
          # and we know they hold in the way they are used in Mosaic. Still,
          # verification with layouts likes to fail, because it can't statically
          # prove the properties.
          erase_op = tpu.EraseLayoutOp(
              ir.MemRefType.get(
                  new_memref.shape, memref.element_type,
                  None, new_memref.memory_space
              ),
              arg,
          )
          tpu.private_replace_all_uses_except(arg, erase_op.result, erase_op)
      f.attributes["function_type"] = ir.TypeAttr.get(
          ir.FunctionType.get(new_arg_types, f.type.results)
      )
