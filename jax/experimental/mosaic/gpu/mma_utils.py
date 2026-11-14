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

import dataclasses
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


_MMA_ATOM_ROWS = 8


@dataclasses.dataclass(frozen=True, kw_only=True)
class SmemDescriptor:
  """MMA shared memory operand descriptor and slicing metadata.

  Attributes:
    addr_base: Dynamic `i32` `ir.Value` holding the 16-byte encoded shared
      memory base address `((ptr & 0x3FFFF) >> 4)`.
    fastest_dim: The contiguous dimension in shared memory.
  """

  addr_base: ir.Value
  fastest_dim: Dim
  _desc_const: int
  _mn_tiles: int
  _k_tiles: int
  _mn_tiling: int
  _k_tiling: int
  _mn_tile_stride: int
  _k_tile_stride: int
  _k_tiling_stride: int
  _element_bitwidth: int

  def encoded_offset(self, mn_idx: int, k_idx: int) -> int:
    """Returns the 64-bit constant to add to `addr_base` for `(mn_idx, k_idx)`.

    Packs the static descriptor metadata (LBO, SBO, swizzle) in bits [16:63]
    with the 16-byte encoded slice byte offset in bits [0:13].
    """
    mn_tile, mn_rem = divmod(mn_idx, self._mn_tiling)
    if mn_rem:
      raise ValueError(
          f"MN offset {mn_idx} must be a multiple of tile size"
          f" {self._mn_tiling}"
      )
    if self.fastest_dim == Dim.MN and k_idx % _MMA_ATOM_ROWS:
      raise ValueError(
          f"K offset {k_idx} must be a multiple of {_MMA_ATOM_ROWS} when MN is"
          " fastest"
      )
    k_tile, k_in_tile = divmod(k_idx, self._k_tiling)
    if not (0 <= mn_tile < self._mn_tiles):
      raise ValueError(
          f"MN offset {mn_idx} out of bounds for"
          f" {self._mn_tiles * self._mn_tiling} elements"
      )
    if not (0 <= k_tile < self._k_tiles):
      raise ValueError(
          f"K offset {k_idx} out of bounds for"
          f" {self._k_tiles * self._k_tiling} elements"
      )
    bit_offset = (
        mn_tile * self._mn_tile_stride
        + k_tile * self._k_tile_stride
        + k_in_tile * self._k_tiling_stride
    ) * self._element_bitwidth
    if bit_offset % 8:
      raise ValueError(
          f"Slice ({mn_idx}, {k_idx}) offset ({bit_offset} bits) must be"
          " byte-aligned"
      )
    return self._desc_const | encode_addr(bit_offset // 8)

  def slice_i64(self, mn_idx: int, k_idx: int) -> ir.Value:
    """Materializes the `i64` SMEM descriptor for slice `(mn_idx, k_idx)`."""
    i64 = ir.IntegerType.get_signless(64)
    offset = self.encoded_offset(mn_idx, k_idx)
    return arith.addi(arith.extui(i64, self.addr_base), _i64_const(offset))


def create_descriptor(
    ref: ir.Value,
    swizzle: int,
    mn_size: int,  # Single-instruction extent along the M or N dimension.
    logical_k_major: bool,  # False for LHS (M, K), True for RHS (K, N).
    # Soft deprecated. Use small tiling instead.
    large_tile: tuple[int, int] | None = None,
) -> SmemDescriptor:
  ref_ty = ir.MemRefType(ref.type)
  element_bitwidth = utils.bitwidth(ref_ty.element_type)
  swizzle_elems = 8 * swizzle // element_bitwidth
  ref_strides, _ = ref_ty.get_strides_and_offset()
  def to_byte_stride(stride: int) -> int:
    bit_stride = stride * element_bitwidth
    assert bit_stride % 8 == 0
    return bit_stride // 8
  mn_large_tile = k_large_tile = None
  if logical_k_major:
    k_tiles, mn_tiles, k_tiling, mn_tiling = ref_ty.shape
    k_tile_stride, mn_tile_stride, k_tiling_stride, mn_tiling_stride = (
        ref_strides
    )
    if large_tile is not None:
      k_large_tile, mn_large_tile = large_tile
  else:
    mn_tiles, k_tiles, mn_tiling, k_tiling = ref_ty.shape
    mn_tile_stride, k_tile_stride, mn_tiling_stride, k_tiling_stride = (
        ref_strides
    )
    if large_tile is not None:
      mn_large_tile, k_large_tile = large_tile

  IGNORED = 0
  if k_tiling_stride > mn_tiling_stride:
    slower_tiling, faster_tiling = k_tiling, mn_tiling
  else:
    faster_tiling, slower_tiling = k_tiling, mn_tiling
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
      and mn_large_tile is not None
      and (
          mn_large_tile == mn_tiling
          or mn_tiles == 1
          and mn_tiling < mn_large_tile
      )
      # There are configurations where large tiles are same size as small ones.
      # We use the small path since it has fewer restrictions.
      and (slower_tiling, faster_tiling) != (_MMA_ATOM_ROWS, swizzle_elems)
  ):  # Large tiles.
    if k_tiling_stride == 1 and mn_tiling_stride == k_tiling:
      fastest_dim = Dim.K
      leading_byte_offset = IGNORED  # TC assumes K to be contiguous here.
      assert k_tiling == swizzle_elems
      # MMA atoms in a group are contiguous, so we increment by the MMA atom
      # size. However, we only have one level of striding, and so if the group
      # size exceeds a single large tile (and there is more than one tile) then
      # that tiled dimension must be contiguous after tiles or else we would
      # need another striding level.
      if (
          mn_tiles > 1
          and mn_size > mn_tiling
          and mn_tile_stride != math.prod(large_tile)
      ):
        raise ValueError(
            "MMA layout with large tiles that is K-fastest only supports"
            " multiple MN tiles when the tiled MN dimension is a contiguous"
            " stack of tiles "
            f"({mn_tiles}, {mn_tile_stride} != {math.prod(large_tile)})"
        )
      stride_byte_offset = _MMA_ATOM_ROWS * swizzle
    elif k_tiling_stride == mn_large_tile and mn_tiling_stride == 1:
      if k_large_tile != mn_large_tile:
        raise ValueError(
            "MMA layout with large tiles that is MN-fastest is only supported"
            " when the tiling is square"
        )
      assert mn_large_tile == swizzle_elems
      fastest_dim = Dim.MN
      # Next swizzle atom with the same K coordinate is in the next MN tile.
      leading_byte_offset = to_byte_stride(mn_tile_stride)
      stride_byte_offset = _MMA_ATOM_ROWS * swizzle
    else:
      raise ValueError("MMA tiles must be contiguous")
  else:  # Small tiles.
    if slower_tiling != _MMA_ATOM_ROWS or faster_tiling != swizzle_elems:
      raise ValueError(
          f"Tiling should be ({_MMA_ATOM_ROWS}, swizzle_elems) where"
          f" swizzle_elems = 8 * swizzle // bitwidth(dtype) (= 8 * {swizzle} //"
          f" {element_bitwidth} = {swizzle_elems}), but got ({slower_tiling},"
          f" {faster_tiling})"
      )
    if k_tiling_stride == 1 and mn_tiling_stride == swizzle_elems:
      fastest_dim = Dim.K
      leading_byte_offset = IGNORED  # TC assumes K to be contiguous here.
      stride_byte_offset = to_byte_stride(mn_tile_stride)
    elif mn_tiling_stride == 1 and k_tiling_stride == swizzle_elems:
      fastest_dim = Dim.MN
      leading_byte_offset = to_byte_stride(mn_tile_stride)
      stride_byte_offset = to_byte_stride(k_tile_stride)
    else:
      raise ValueError("MMA tiles must be contiguous")

  if mn_size % mn_tiling:
    raise ValueError(
        f"The M or N MMA instruction size was chosen to be {mn_size},"
        " which is not a multiple of the tiling of the non-contracting"
        f" dimension {mn_tiling}"
    )

  addr_base, desc_const = _split_descriptor(
      ref,
      leading_byte_offset=leading_byte_offset,
      stride_byte_offset=stride_byte_offset,
      swizzle=swizzle,
  )
  assert (desc_const & 0xFFFF) == 0

  return SmemDescriptor(
      addr_base=addr_base,
      fastest_dim=fastest_dim,
      _desc_const=desc_const,
      _mn_tiles=mn_tiles,
      _k_tiles=k_tiles,
      _mn_tiling=mn_tiling,
      _k_tiling=k_tiling,
      _mn_tile_stride=mn_tile_stride,
      _k_tile_stride=k_tile_stride,
      _k_tiling_stride=k_tiling_stride,
      _element_bitwidth=element_bitwidth,
  )


def encode_addr(x: int) -> int:
  result = (x & 0x3FFFF) >> 4
  if result << 4 != x:
    raise ValueError(f"Cannot encode value in an MMA descriptor: {x}")
  return result


def _i64_const(val: int) -> ir.Value:
  i64 = ir.IntegerType.get_signless(64)
  # Descriptor constants frequently have the MSB set, leading to errors when
  # trying to create ir.IntegerAttr through the MLIR Python bindings. This
  # should be easy enough for LLVM to constant fold away.
  if val >> 63:
    low = arith.constant(i64, val & 0xFFFFFFFF)
    high = arith.constant(i64, val >> 32)
    return llvm.or_(low, arith.shli(high, arith.constant(i64, 32)))
  return arith.constant(i64, val)


def _split_descriptor(
    ref_arg: ir.Value,
    leading_byte_offset: int,
    stride_byte_offset: int,
    swizzle: int | mgpu_dialect.SwizzlingMode | None,
    const_init: int = 0,
) -> tuple[ir.Value, int]:
  i32 = ir.IntegerType.get_signless(32)
  i64 = ir.IntegerType.get_signless(64)
  if isinstance(ref_arg.type, ir.MemRefType):
    ptr = utils.memref_ptr(ref_arg)
  else:
    ptr = ref_arg
  assert ptr.type == llvm.PointerType.get(address_space=3), ptr.type
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
      | (swizzle_encoding << 62)
  )
  # The encoded base addr fits within a single 32-bit register.
  return arith.trunci(i32, encoded_base_addr), desc_const


def encode_descriptor(
    ref_arg: ir.Value,
    leading_byte_offset: int,
    stride_byte_offset: int,
    swizzle: int | mgpu_dialect.SwizzlingMode | None,
    const_init: int = 0,
) -> ir.Value:
  i64 = ir.IntegerType.get_signless(64)
  addr_base, desc_const = _split_descriptor(
      ref_arg, leading_byte_offset, stride_byte_offset, swizzle, const_init
  )
  return llvm.or_(arith.extui(i64, addr_base), _i64_const(desc_const))
