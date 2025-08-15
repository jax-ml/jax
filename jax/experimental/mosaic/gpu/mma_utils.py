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
    mma_bytewidth_k: int = 32,
):
  ref_ty = ir.MemRefType(ref.type)
  element_bitwidth = utils.bitwidth(ref_ty.element_type)
  swizzle_elems = 8 * swizzle // element_bitwidth
  ref_strides, _ = ref_ty.get_strides_and_offset()
  def to_byte_stride(stride: int):
    if element_bitwidth >= 8:
      assert element_bitwidth % 8 == 0
      return stride * element_bitwidth // 8
    else:
      packing = 8 // element_bitwidth
      assert stride % packing == 0
      return stride // packing
  mn_large_tile = k_large_tile = None
  if logical_k_major:
    _, mn_tiles, k_tiling, mn_tiling = ref_ty.shape
    k_tile_stride, mn_tile_stride, k_tiling_stride, mn_tiling_stride = (
        ref_strides
    )
    k_group_size, mn_group_size = group_size
    if large_tile is not None:
      k_large_tile, mn_large_tile = large_tile
  else:
    mn_tiles, _, mn_tiling, k_tiling = ref_ty.shape
    mn_tile_stride, k_tile_stride, mn_tiling_stride, k_tiling_stride = (
        ref_strides
    )
    mn_group_size, k_group_size = group_size
    if large_tile is not None:
      mn_large_tile, k_large_tile = large_tile

  IGNORED = 0
  MMA_ATOM_ROWS = 8
  mma_width_k = 8 * mma_bytewidth_k // element_bitwidth
  desc_k_tiling: tuple[int, ...] = ()
  desc_k_strides: tuple[int, ...]
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
      and mma_bytewidth_k == 32
  ):  # Large tiles.
    if k_tiling_stride == 1 and mn_tiling_stride == k_tiling:
      fastest_dim = Dim.K
      leading_byte_offset = IGNORED  # TC assumes K to be contiguous here.
      assert k_tiling == k_group_size  # Else we need multi-level striding.
      # MMA atoms in a group are contiguous, so we increment by the MMA atom
      # size. However, we only have one level of striding, and so if the group
      # size exceeds a single large tile (and there is more than one tile) then
      # that tiled dimension must be contiguous after tiles or else we would
      # need another striding level.
      if (
          mn_tiles > 1
          and mn_group_size > mn_tiling
          and mn_tile_stride != math.prod(large_tile)
      ):
        raise ValueError(
            "MMA layout with large tiles that is K-fastest only supports"
            " multiple MN tiles when the tiled MN dimension is a contiguous"
            " stack of tiles "
            f"({mn_tiles}, {mn_tile_stride} != {math.prod(large_tile)})"
        )
      stride_byte_offset = MMA_ATOM_ROWS * swizzle
      desc_k_strides = (mma_bytewidth_k,)  # K is contiguous.
    elif k_tiling_stride == k_tiling and mn_tiling_stride == 1:
      if k_large_tile != mn_large_tile:
        raise ValueError(
            "MMA layout with large tiles that is MN-fastest is only supported"
            " when the tiling is square"
        )
      fastest_dim = Dim.MN
      # Next swizzle atom with the same K coordinate is in the next MN tile.
      leading_byte_offset = to_byte_stride(mn_tile_stride)
      # MMA atoms in a group are contiguous and a group does not exceed a tile.
      assert k_large_tile == k_group_size
      stride_byte_offset = MMA_ATOM_ROWS * swizzle
      # Each row is swizzle bytes wide, and we read mma_width_k rows at a time.
      assert mn_large_tile == 8 * swizzle // element_bitwidth
      desc_k_strides = (mma_width_k * swizzle,)
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
          f" swizzle_elems = 8 * swizzle // bitwidth(dtype) (= 8 * {swizzle} //"
          f" {element_bitwidth} = {swizzle_elems}), but got ({slower_tiling},"
          f" {faster_tiling})"
      )
    if k_tiling_stride == 1 and mn_tiling_stride * element_bitwidth == MMA_ATOM_ROWS * swizzle:
      fastest_dim = Dim.K
      leading_byte_offset = IGNORED  # TC assumes K to be contiguous here.
      stride_byte_offset = to_byte_stride(mn_tile_stride)
      if k_tiling == k_group_size:
        desc_k_strides = (mma_bytewidth_k,)  # K is contiguous.
      elif k_group_size % k_tiling == 0:
        desc_k_tiling = (k_tiling // mma_width_k,)
        desc_k_strides = (MMA_ATOM_ROWS * swizzle, mma_bytewidth_k)
      else:
        if k_tiling < mma_width_k:
          raise ValueError(
              "K dimension tiling is smaller than the width of a single MMA"
              " instruction. Increase swizzle."
          )
        raise NotImplementedError(f"{k_group_size=} must be larger than {k_tiling=}")
    elif k_tiling_stride * element_bitwidth == MMA_ATOM_ROWS * swizzle and mn_tiling_stride == 1:
      fastest_dim = Dim.MN
      leading_byte_offset = to_byte_stride(mn_tile_stride)
      stride_byte_offset = to_byte_stride(k_tile_stride)
      k_tiles_per_mma = mma_width_k // MMA_ATOM_ROWS
      desc_k_strides = (to_byte_stride(k_tile_stride) * k_tiles_per_mma,)
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
  mn_group_stride = to_byte_stride(mn_tile_stride) * mn_tiles_per_group
  k_tiles_per_group, rem = divmod(k_group_size, k_tiling)
  assert not rem
  k_group_stride = to_byte_stride(k_tile_stride) * k_tiles_per_group

  return (
      (desc_base, (desc_k_tiling, desc_k_strides)),
      (mn_group_stride, k_group_stride),
      fastest_dim,
  )


def encode_addr(x: int):
  result = (x & 0x3FFFF) >> 4
  if result << 4 != x:
    raise ValueError(f"Cannot encode value in an MMA descriptor: {x}")
  return result


def encode_descriptor(
    ref_arg,
    leading_byte_offset: int,
    stride_byte_offset: int,
    swizzle: int | mgpu_dialect.SwizzlingMode | None,
    const_init: int = 0,
):
  i64 = ir.IntegerType.get_signless(64)
  if isinstance(ref_arg.type, ir.MemRefType):
    ptr = utils.memref_ptr(ref_arg, 3)
  else:
    ptr = ref_arg
  assert ptr.type == ir.Type.parse("!llvm.ptr<3>"), ptr.type
  ptr_val = llvm.ptrtoint(i64, ptr)
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
