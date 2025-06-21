# Copyright 2025 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import enum
import math

from jax._src.lib import mosaic_gpu_dialect as mgpu_dialect
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import llvm

from . import utils

# mypy: ignore-errors

def tiled_memref_shape(ref: ir.Value):
  """Returns the 2D untiled shape and element type of a tiled 4D memref."""
  ref_ty = ir.MemRefType(ref.type)
  if ref_ty.rank != 4:
    raise ValueError(f"Expected a 4D memref, got: {ref_ty}")
  logical_shape = (
      ref_ty.shape[0] * ref_ty.shape[2], ref_ty.shape[1] * ref_ty.shape[3]
  )
  return logical_shape, ref_ty.element_type


class Dim(enum.Enum):
  K = enum.auto()
  MN = enum.auto()


def create_descriptor(
    ref: ir.Value,
    swizzle: int,
    group_size: tuple[int, int],  # Instruction group size on each operand dim.
    logical_k_major: bool,  # False for LHS, True for RHS.
    # Soft deprecated. Use small tiling instead.
    large_tile: tuple[int, int] | None = None,
):
  ref_ty = ir.MemRefType(ref.type)
  element_bytewidth = utils.bytewidth(ref_ty.element_type)
  swizzle_elems = swizzle // element_bytewidth
  ref_strides, _ = ref_ty.get_strides_and_offset()
  ref_byte_strides = [s * element_bytewidth for s in ref_strides]
  mn_large_tile = k_large_tile = None
  if logical_k_major:
    _, mn_tiles, k_tiling, mn_tiling = ref_ty.shape
    k_tile_stride, mn_tile_stride, k_tiling_stride, mn_tiling_stride = (
        ref_byte_strides
    )
    k_group_size, mn_group_size = group_size
    if large_tile is not None:
      k_large_tile, mn_large_tile = large_tile
  else:
    mn_tiles, _, mn_tiling, k_tiling = ref_ty.shape
    mn_tile_stride, k_tile_stride, mn_tiling_stride, k_tiling_stride = (
        ref_byte_strides
    )
    mn_group_size, k_group_size = group_size
    if large_tile is not None:
      mn_large_tile, k_large_tile = large_tile

  IGNORED = 0
  MMA_ATOM_ROWS = 8
  MMA_BYTEWIDTH_K = 32
  mma_width_k = MMA_BYTEWIDTH_K // element_bytewidth
  # As far as I can tell (which does not seem to fully align with the way MMA is
  # documented in PTX docs), MMA expects the data to be tiled into matrices
  # of shape 8 x swizzle_elems, with swizzle_elems dim being the fastest
  # changing. I call this submatrix an MMA atom.
  #
  # The role of the SMEM descriptor is to specify the striding pattern between
  # those atoms. The fastest changing dimension is called the "leading"
  # dimension and it specifies the stride between consecutive atoms that share
  # the same coordinate along that dim. The slower dimension is called a
  # "stride" dimension.
  if (
      large_tile is not None
      and k_large_tile == k_tiling
      and (mn_large_tile == mn_tiling or mn_tiles == 1 and mn_tiling < mn_large_tile)
      # There are configurations where large tiles are same size as small ones.
      # We use the small path since it has fewer restrictions.
      and set(large_tile) != {MMA_ATOM_ROWS, swizzle_elems}
  ):  # Large tiles.
    if (
        k_tiling_stride == element_bytewidth
        and mn_tiling_stride == k_tiling * element_bytewidth
    ):
      fastest_dim = Dim.K
      leading_byte_offset = IGNORED  # TC assumes K to be contiguous here.
      # MMA atoms in a group are contiguous, so we increment by the MMA atom
      # size. However, we only have one level of striding, and so if the group
      # size exceeds a single large tile (and there is more than one tile) then
      # that tiled dimension must be contiguous after tiles or else we would
      # need another striding level.
      if (
          mn_tiles > 1
          and mn_group_size > mn_tiling
          and mn_tile_stride != math.prod(large_tile) * element_bytewidth
      ):
        raise ValueError(
            "MMA layout with large tiles that is K-fastest only supports"
            " multiple MN tiles when the tiled MN dimension is a contiguous"
            " stack of tiles "
            f"({mn_tiles}, {mn_tile_stride} != {math.prod(large_tile)} * {element_bytewidth})"
        )
      stride_byte_offset = MMA_ATOM_ROWS * swizzle
      desc_k_stride = MMA_BYTEWIDTH_K  # K is contiguous.
    elif (
        k_tiling_stride == k_tiling * element_bytewidth
        and mn_tiling_stride == element_bytewidth
    ):
      if k_large_tile != mn_large_tile:
        raise ValueError(
            "MMA layout with large tiles that is MN-fastest is only supported"
            " when the tiling is square"
        )
      fastest_dim = Dim.MN
      # Next swizzle atom with the same K coordinate is in the next MN tile.
      leading_byte_offset = mn_tile_stride
      # MMA atoms in a group are contiguous and a group does not exceed a tile.
      assert k_large_tile == k_group_size
      stride_byte_offset = MMA_ATOM_ROWS * swizzle
      # Each row is swizzle bytes wide, and we read mma_width_k rows at a time.
      assert mn_large_tile == swizzle // element_bytewidth
      desc_k_stride = mma_width_k * swizzle
    else:
      raise ValueError("MMA tiles must be contiguous")
  else:  # Small tiles.
    if k_tiling_stride > mn_tiling_stride:
      slower_tiling, faster_tiling = k_tiling, mn_tiling
    else:
      faster_tiling, slower_tiling = k_tiling, mn_tiling
    if slower_tiling != MMA_ATOM_ROWS or faster_tiling != swizzle_elems:
      raise ValueError(
          f"Tiling should be ({MMA_ATOM_ROWS}, swizzle_elems) where"
          f" swizzle_elems = swizzle // bytewidth(dtype) (= {swizzle} //"
          f" {element_bytewidth} = {swizzle_elems}), but got ({slower_tiling},"
          f" {faster_tiling})"
      )
    if k_tiling_stride == element_bytewidth and mn_tiling_stride == swizzle:
      fastest_dim = Dim.K
      leading_byte_offset = IGNORED  # TC assumes K to be contiguous here.
      stride_byte_offset = mn_tile_stride
      desc_k_stride = MMA_BYTEWIDTH_K  # K is contiguous.
    elif k_tiling_stride == swizzle and mn_tiling_stride == element_bytewidth:
      fastest_dim = Dim.MN
      leading_byte_offset = mn_tile_stride
      stride_byte_offset = k_tile_stride
      k_tiles_per_mma = mma_width_k // MMA_ATOM_ROWS
      desc_k_stride = k_tile_stride * k_tiles_per_mma
    else:
      raise ValueError("MMA tiles must be contiguous")
  desc_base = encode_descriptor(
      ref,
      leading_byte_offset=leading_byte_offset,
      stride_byte_offset=stride_byte_offset,
      swizzle=swizzle,
  )

  mn_tiles_per_group, rem = divmod(mn_group_size, mn_tiling)
  assert not rem
  mn_group_stride = mn_tile_stride * mn_tiles_per_group
  k_tiles_per_group, rem = divmod(k_group_size, k_tiling)
  assert not rem
  k_group_stride = k_tile_stride * k_tiles_per_group

  return (
      (desc_base, desc_k_stride),
      (mn_group_stride, k_group_stride),
      fastest_dim,
  )


def encode_addr(x: int):
  result = (x & 0x3FFFF) >> 4
  if result << 4 != x:
    raise ValueError(f"Cannot encode value in an MMA descriptor: {x}")
  return result


def encode_descriptor(
    memref_arg,
    leading_byte_offset: int,
    stride_byte_offset: int,
    swizzle: int | mgpu_dialect.SwizzlingMode | None,
    const_init: int = 0,
):
  i64 = ir.IntegerType.get_signless(64)
  ptr_val = llvm.ptrtoint(i64, utils.memref_ptr(memref_arg, 3))
  c = lambda x: arith.constant(i64, x)
  if swizzle is None or swizzle == mgpu_dialect.SwizzlingMode.kNoSwizzle:
    swizzle_encoding = 0
  elif swizzle == mgpu_dialect.SwizzlingMode.k128ByteSwizzle:
    swizzle_encoding = 1
  elif swizzle == mgpu_dialect.SwizzlingMode.k64ByteSwizzle:
    swizzle_encoding = 2
  elif swizzle == mgpu_dialect.SwizzlingMode.k32ByteSwizzle:
    swizzle_encoding = 3
  else:
    raise NotImplementedError(swizzle)
  encoded_base_addr = llvm.lshr(llvm.and_(ptr_val, c(0x3FFFF)), c(4))
  # We ignore the offset
  desc_const = (
      const_init
      | (encode_addr(leading_byte_offset) << 16)
      | (encode_addr(stride_byte_offset) << 32)
  )
  desc = llvm.or_(arith.shli(c(swizzle_encoding), c(62)), c(desc_const))
  desc = llvm.or_(encoded_base_addr, desc)
  return desc
