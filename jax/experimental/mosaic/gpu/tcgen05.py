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

from __future__ import annotations

import dataclasses
import math

from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import memref
import numpy as np

from . import utils
from . import fragmented_array as fa
from . import mma_utils
from .launch_context import LaunchContext

# MyPy does a terrible job with the MLIR API.
# mypy: ignore-errors


TMEM_ROWS = 128
TCGEN05_SMEM_DESCRIPTOR_BIT = 1 << 46

def create_instr_descriptor(
    m: int,
    n: int,
    acc_dtype,
    input_dtype,
    transpose_a: bool = False,
    transpose_b: bool = False,
):
  f32 = ir.F32Type.get()
  bf16 = ir.BF16Type.get()
  f16 = ir.F16Type.get()
  if input_dtype not in {f16, bf16}:
    raise NotImplementedError("Only float16 and bfloat16 inputs supported")
  if acc_dtype not in {f32, f16}:
    raise NotImplementedError("Only float32 and float16 accumulators supported")

  desc = 0
  # We ignore sparsity in bits 0-3
  desc |= (acc_dtype == f32) << 4  # D dtype, bits 4-5
  # Bit 6 is reserved
  desc |= (input_dtype == bf16) << 7  # A dtype, bits 7-9
  desc |= (input_dtype == bf16) << 10  # B dtype, bits 10-12
  # We ignore negate bits 13-14
  desc |= transpose_a << 15  # Transpose A
  desc |= transpose_b << 16  # Transpose B
  if n % 8 or n > 256:
    raise ValueError(f"N must be a multiple of 8 and <= 256, got: {n}")
  desc |= (n >> 3) << 17  # N, bits 17-22
  # Bit 23 is reserved
  if m % 16 or m > 256:
    raise ValueError(f"M must be a multiple of 16 and <= 256, got: {m}")
  desc |= (m >> 4) << 24  # M >> 4, bits 24-28
  # Bit 29 is reserved
  # We ignore max shift under .ws, bits 30-31
  return arith.constant(ir.IntegerType.get_signless(32), desc)


def mma(
    d: TMEMRef,
    a: ir.Value,
    b: ir.Value,
    *,
    a_swizzle: int = 128,
    b_swizzle: int = 128,
    accumulate: ir.Value | bool = True,
    collective: bool = False,
):
  if a_swizzle == 16 or b_swizzle == 16:
    raise NotImplementedError("No swizzle is not supported")
  i32 = ir.IntegerType.get_signless(32)
  i64 = ir.IntegerType.get_signless(64)
  if isinstance(accumulate, bool):
    accumulate = arith.constant(ir.IntegerType.get_signless(1), accumulate)
  if a_swizzle != b_swizzle:
    raise NotImplementedError(f"{a_swizzle=} != {b_swizzle=}")
  swizzle = a_swizzle
  num_cta = 2 if collective else 1

  # Step 1. Establish the shape and element type of the operation.
  if not ir.MemRefType.isinstance(a.type):
    raise ValueError(f"A must be a memref, got {a.type}")
  if not ir.MemRefType.isinstance(b.type):
    raise ValueError(f"B must be a memref, got: {b.type}")
  (k, n), element_type = mma_utils.tiled_memref_shape(b)
  (m, k2), element_type2 = mma_utils.tiled_memref_shape(a)
  if k != k2:
    raise ValueError(
        "MMA requires A and B to have the same contraction dimension (K),"
        f" got: {k2} and {k}"
    )
  if element_type != element_type2:
    raise ValueError(
        "MMA requires A and B to have the same element type, got:"
        f" {element_type2} and {element_type}"
    )
  if d.shape != (m, n * num_cta):
    raise ValueError(
        f"Accumulator shape mismatch: expected {(m, n * num_cta)}, got {d.shape}"
    )
  if d.layout != (expected_layout := _infer_tmem_layout(d.shape, collective)):
    raise ValueError(
        f"Accumulator layout mismatch: expected {expected_layout}, got {d.layout}"
    )
  f32 = ir.F32Type.get()
  if element_type == f32 or element_type == ir.BF16Type.get():
    if d.dtype != f32:
      raise ValueError(
          f"MMA with element type {element_type} only supports accumulators"
          f" of type f32, but got: {d.dtype}"
      )
  elif element_type == ir.F16Type.get():
    if d.dtype != element_type and d.dtype != f32:
      raise ValueError(
          "MMA with element type f16 only supports accumulators of type f32"
          f" or f16, but got: {d.dtype}"
      )

  # Step 2. Decide on the instruction shapes we'll use. Note that with swizzles,
  # instructions must be issued in groups of the same width as the swizzle.
  m_group_elems = d.layout.elements_in_tile[0]
  if m_group_elems != 128:
    raise NotImplementedError("Only 128-row accumulators supported for now")
  k_group_elems = swizzle // utils.bytewidth(element_type)
  if n % 8:
    raise ValueError(f"N must be a multiple of 8, got: {n}")
  elif n > 256 and n != 512:
    raise ValueError("Only N below 256 or N=512 are supported")
  n_group_elems = min(n, 256 // num_cta)
  if m % m_group_elems:
    raise ValueError(f"M must be a multiple of {m_group_elems}, got: {m}")
  if k % k_group_elems:
    raise ValueError(f"K must be a multiple of {k_group_elems}, got: {k}")
  if n % n_group_elems:
    raise ValueError(f"N must be a multiple of {n_group_elems}, got: {n}")
  m_groups = m // m_group_elems
  k_groups = k // k_group_elems
  n_groups = n // n_group_elems
  # TODO(apaszke): Require users to bitcast input refs to tf32 before WGMMA.
  wgmma_element_type = (
      ir.FloatTF32Type.get() if element_type == ir.F32Type.get() else element_type
  )

  # Step 3. Compute the operand descriptors.
  (
      (a_desc_base, a_k_instr_stride),
      (a_m_group_stride, a_k_group_stride),
      a_fastest,
  ) = mma_utils.create_descriptor(
      a,
      swizzle=swizzle,
      group_size=(m_group_elems, k_group_elems),
      logical_k_major=False,
  )
  (
      (b_desc_base, b_k_instr_stride),
      (b_n_group_stride, b_k_group_stride),
      b_fastest,
  ) = mma_utils.create_descriptor(
      b,
      swizzle=swizzle,
      group_size=(k_group_elems, n_group_elems),
      logical_k_major=True,
  )

  # Step 4. Issue the instructions.
  true = arith.constant(ir.IntegerType.get_signless(1), 1)
  n_collective_group_elems = n_group_elems * num_cta
  for mi, ni, ki in np.ndindex(m_groups, n_groups, k_groups):
    a_offset = mi * a_m_group_stride + ki * a_k_group_stride
    a_mk = arith.addi(a_desc_base, utils.c(mma_utils.encode_addr(a_offset), i64))
    b_offset = ni * b_n_group_stride + ki * b_k_group_stride
    b_nk = arith.addi(b_desc_base, utils.c(mma_utils.encode_addr(b_offset), i64))
    if m_groups != 1:
      raise NotImplementedError("D needs to be sliced")
    acc = accumulate if ki == 0 else true
    _do_mma(
        arith.addi(
            d.address, arith.constant(i32, ni * n_collective_group_elems)
        ),
        a_mk,
        b_nk,
        d_type=ir.F32Type.get(),
        m=m_group_elems,
        n=n_group_elems,
        collective=collective,
        a_transpose=a_fastest != mma_utils.Dim.K,
        b_transpose=b_fastest != mma_utils.Dim.K,
        a_k_stride=a_k_instr_stride,
        b_k_stride=b_k_instr_stride,
        accumulate=acc,
        swizzle=swizzle,
        element_type=wgmma_element_type,
    )


def _do_mma(
    d_addr: ir.Value,
    a_desc: ir.Value,
    b_desc: ir.Value,
    a_transpose: bool,
    b_transpose: bool,
    a_k_stride: int,
    b_k_stride: int,
    m: int,
    n: int,
    swizzle: int,
    element_type: ir.Type,
    d_type: ir.Type,
    accumulate: ir.Value,
    collective: bool,
):
  i1 = ir.IntegerType.get_signless(1)
  i64 = ir.IntegerType.get_signless(64)
  kn_tiling = swizzle // utils.bytewidth(element_type)
  instr_k = 32 // utils.bytewidth(element_type)
  if a_k_stride % 16 or b_k_stride % 16:
    raise ValueError

  if ir.F16Type.isinstance(element_type) or ir.BF16Type.isinstance(element_type):
    kind = "f16"
  else:
    raise NotImplementedError(f"Unsupported input element type: {element_type}")

  num_cta = 2 if collective else 1
  i_desc = create_instr_descriptor(
      m * num_cta, n * num_cta, d_type, element_type, a_transpose, b_transpose
  )
  for _ in range(kn_tiling // instr_k):
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [d_addr, a_desc, b_desc, i_desc, accumulate],
        f"tcgen05.mma.cta_group::{num_cta}.kind::{kind} [$0], $1, $2, $3, $4;",
        "r,l,l,r,b",
        has_side_effects=True,
    )
    accumulate = arith.constant(i1, 1)
    a_desc = arith.addi(a_desc, arith.constant(i64, a_k_stride >> 4))
    b_desc = arith.addi(b_desc, arith.constant(i64, b_k_stride >> 4))


def commit_arrive(
    barrier: utils.BarrierRef | ir.Value,
    collective: bool = False,
    ctx: LaunchContext | None = None,
):
  if isinstance(barrier, utils.BarrierRef):
    barrier = barrier.get_ptr()
  elif barrier.type != ir.Type.parse("!llvm.ptr<3>"):
    raise ValueError(
        "barrier must be a Mosaic barrier or a SMEM pointer, got:"
        f" {barrier.type}"
    )
  if collective:
    if ctx is None:
      raise ValueError("ctx must be provided for collective barriers")
    # TODO(apaszke): This is just 0b11 shifted by the even CTA index.
    if ctx.cluster_size != (2, 1, 1):
      raise NotImplementedError("Collective arrivals only support (2, 1, 1)-shaped clusters")
    ptx = """
    {
        .reg .b16 msk;
        mov.b16 msk, 3;
        tcgen05.commit.cta_group::2.mbarrier::arrive::one.multicast::cluster.b64 [$0], msk;
    }
    """
  else:
    ptx = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [$0];"
  return llvm.inline_asm(
      ir.Type.parse("!llvm.void"), [barrier], ptx, "l", has_side_effects=True
  )


def tmem_alloc(tmem_addr: ir.Value, ncols: int, collective: bool = False, exact: bool = True):
  if ir.MemRefType.isinstance(tmem_addr.type):
    ref_ty = ir.MemRefType(tmem_addr.type)
    if ref_ty.element_type != ir.IntegerType.get_signless(32):
      raise ValueError(f"tmem_addr must be an i32 memref, got: {ref_ty}")
    if ref_ty.memory_space != ir.Attribute.parse("#gpu.address_space<workgroup>"):
      raise ValueError(f"tmem_addr must be in shared memory, got: {ref_ty}")
    if math.prod(ref_ty.shape) != 1:
      raise ValueError(f"tmem_addr must contain a single element, got: {ref_ty}")
    tmem_addr = utils.memref_ptr(tmem_addr, memory_space=3)
  elif tmem_addr.type != ir.Type.parse("!llvm.ptr<3>"):
    raise ValueError(f"tmem_addr must be an SMEM pointer or a memref, got: {tmem_addr.type}")
  if exact:
    if ncols.bit_count() != 1 or not 32 <= ncols <= 512:
      raise ValueError(f"ncols must be a power of 2 and within [32, 512], got: {ncols}")
  else:
    ncols = max(32, 1 << (ncols - 1).bit_length())
    if ncols > 512:
      raise ValueError(
          f"After rounding up, got {ncols} columns, exceeding the limit of 512"
      )
  num_cta = 2 if collective else 1
  return llvm.inline_asm(
      ir.Type.parse("!llvm.void"),
      [tmem_addr],
      f"tcgen05.alloc.cta_group::{num_cta}.sync.aligned.shared::cta.b32  [$0], {ncols};",
      "r",
      has_side_effects=True,
  )

def tmem_relinquish_alloc_permit():
  return llvm.inline_asm(
      ir.Type.parse("!llvm.void"),
      [],
      "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;",
      "",
      has_side_effects=True,
  )

def tmem_load(tmem_addr, shape, num):
  if num.bit_count() != 1 or num > 128:
    raise ValueError(f"num must be a power of 2 and <= 128, got: {num}")
  match shape:
    case "16x128b":
      num_out_regs = 2
    case "16x256b":
      num_out_regs = 4
    case _:
      raise NotImplementedError(f"{shape=} is unsupported")
  if num * num_out_regs >= 256:
    raise ValueError(
        f"Loading too much TMEM at once: {num=} and each load requires"
        f" {num_out_regs} registers, which exceeds the limit of 256"
    )
  num_out_regs *= num
  i32 = ir.IntegerType.get_signless(32)
  out_regs = ",".join("$" + str(i) for i in range(num_out_regs))
  regs = llvm.inline_asm(
      ir.Type.parse(
          "!llvm.struct<(" + ",".join("i32" for _ in range(num_out_regs)) + ")>"
      ),
      [tmem_addr],
      f"tcgen05.ld.sync.aligned.{shape}.x{num}.b32    {{{out_regs}}}, [${num_out_regs}];",
      "=r," * num_out_regs + "r",
      has_side_effects=True,
  )
  return [llvm.extractvalue(i32, regs, [i]) for i in range(num_out_regs)]


@dataclasses.dataclass(frozen=True)
class TMEMLayout:
  """Represents the way a shape is laid out in TMEM.

  Only 2D shapes are supported. Row tiling must be between 32 and 128, and be
  a power of 2. If the row tiling is smaller than 128 (the row count in TMEM),
  the tiles are linearized in row-major order, but laid out in TMEM in a
  column-major order.

  Consider an array that is (128, 128) and we apply tiling of (64, 64):

    +------------------+------------------+
    | [0:64, 0:64]     | [0:64, 64:128]   |
    +------------------+------------------+
    | [64:128, 0:64]   | [64:128, 64:128] |
    +------------------+------------------+

  In TMEM it will be laid out as follows:

    +------------------+------------------+
    | [0:64, 0:64]     | [64:128, 0:64]   |
    +------------------+------------------+
    | [0:64, 64:128]   | [64:128, 64:128] |
    +------------------+------------------+

  The above is further complicated by column_tile_stride, which is used to
  swizzle the ordering of column tiles. That is, if column_tile_stride is 2,
  we will first lay out all tiles that have the column index 0, 2, 4, and so on
  until we run out of tiles. Only then we lay out the tiles with column index
  1, 3, etc.
  """
  elements_in_tile: tuple[int, int]
  column_tile_stride: int = 1

  def __post_init__(self):
    row_tiling = self.elements_in_tile[0]
    if not 32 <= row_tiling <= 128:
      raise ValueError(
          f"Row tiling must be between 32 and 128, got: {row_tiling}"
      )
    if row_tiling.bit_count() != 1:
      raise ValueError(f"Row tiling must be a power of 2, got: {row_tiling}")

  def check_shape(self, shape: tuple[int, ...]):
    if len(shape) != 2:
      raise ValueError(f"TMEM can only represent 2D shapes, got {shape}")
    if any(s % t for s, t in zip(shape, self.elements_in_tile)):
      raise ValueError(
          f"{shape} is divisible into tiles of shape {self.elements_in_tile}"
      )

  def cols_in_shape(self, shape: tuple[int, int]):
    cols_in_tile = self.elements_in_tile[1]
    tiles_in_row = TMEM_ROWS // self.elements_in_tile[0]
    num_tiles = math.prod(utils.tile_shape(shape, self.elements_in_tile)[:-2])
    assert num_tiles % tiles_in_row == 0
    return num_tiles // tiles_in_row * cols_in_tile


def _infer_tmem_layout(shape: tuple[int, int], collective: bool) -> TMEMLayout:
  if shape[0] > TMEM_ROWS:
    raise ValueError(
        "Can only infer TMEM layout for shapes with at most 128 rows, got:"
        f" {shape[0]}"
    )
  if shape[0] < 32:
    raise ValueError(
        "Can only infer TMEM layout for shapes with at least 32 rows, got:"
        f" {shape[0]}"
    )
  if shape[0].bit_count() != 1:
    raise ValueError(
        "Can only infer TMEM layout for shapes with row count that's a power of"
        f" 2, got: {shape[0]}"
    )
  if shape[1] % 8:
    raise ValueError(
        "Can only infer TMEM layout for shapes with column count that's a"
        f" multiple of 8, got: {shape[1]}"
    )
  if collective and shape[1] == 512:
    return TMEMLayout(elements_in_tile=(shape[0], 128), column_tile_stride=2)
  else:
    return TMEMLayout(elements_in_tile=(shape[0], 8))


@dataclasses.dataclass(frozen=True)
class TMEMRef:
  address: ir.Value
  shape: tuple[int, int]
  dtype: ir.Type
  layout: TMEMLayout

  @classmethod
  def from_alloc(
      cls,
      tmem_addr_ref: ir.Value,
      shape: tuple[int, int],
      dtype,
      collective: bool | None = None,
      layout: TMEMLayout | None = None,
  ):
    i32 = ir.IntegerType.get_signless(32)
    if not ir.MemRefType.isinstance(tmem_addr_ref.type):
      raise ValueError(f"tmem_addr_ref must be a memref or a pointer, got: {tmem_addr_ref.type}")
    addr_ref_ty = ir.MemRefType(tmem_addr_ref.type)
    smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
    if addr_ref_ty.memory_space != smem:
      raise ValueError(f"tmem_addr_ref must be in workgroup memory, got: {addr_ref_ty}")
    if addr_ref_ty.element_type != i32:
      raise ValueError(f"tmem_addr_ref must be an i32 memref, got: {addr_ref_ty}")
    if math.prod(addr_ref_ty.shape) != 1:
      raise ValueError(f"tmem_addr_ref must contain a single element, got: {addr_ref_ty}")
    i0 = arith.ConstantOp.create_index(0)
    tmem_addr = memref.load(tmem_addr_ref, [i0] * addr_ref_ty.rank)
    if shape[0] < 32:
      raise ValueError(f"TMEM refs must have at least 32 rows, got: {shape[0]}")
    if layout is None:
      if collective is None:
        raise ValueError(
            "collective argument must be provided when TMEM layout is inferred"
        )
      layout = _infer_tmem_layout(shape, collective)
    else:
      layout.check_shape(shape)
    # TODO: Do we have to do this??
    # warp_idx = utils.warp_idx(sync=False)
    # tmem_addr = arith.ori(tmem_addr, arith.shli(warp_idx, utils.c(21, i32)))
    return cls(tmem_addr, shape, dtype, layout)

  def slice(self, *idxs):
    base_idx, slice_shape, is_squeezed = utils.parse_indices(idxs, self.shape)
    if any(is_squeezed):
      raise ValueError("TMEM can only be sliced, not indexed")
    if self.layout != TMEMLayout(elements_in_tile=(TMEM_ROWS, 8)):
      raise NotImplementedError(
          "Slicing only implemented for refs with standard layout, got:"
          f" {self.layout}"
      )
    if base_idx[0] != 0 or slice_shape[0] != TMEM_ROWS:
      raise NotImplementedError("TMEM cannot be sliced along rows")
    if slice_shape[1] % 8:
      raise NotImplementedError(
          "TMEM column slice length must be a multiple of 8"
      )
    col_idx = base_idx[1]
    if not isinstance(col_idx, ir.Value):
      col_idx = arith.constant(ir.IntegerType.get_signless(32), col_idx)
    return TMEMRef(
        address=arith.addi(self.address, col_idx),
        shape=tuple(slice_shape),
        layout=self.layout,
        dtype=self.dtype,
    )

  def __getitem__(self, *idxs):
    i32 = ir.IntegerType.get_signless(32)
    base_idxs, slice_shape, is_squeezed = utils.parse_indices(idxs, self.shape)
    if any(is_squeezed):
      raise ValueError("TMEM loads only support slicing")
    if any(idx != 0 for idx in base_idxs) or tuple(slice_shape) != self.shape:
      raise NotImplementedError("Slicing of TMEM not impelmented yet")
    if self.shape[1] % 8:
      raise NotImplementedError
    if self.dtype != ir.F32Type.get():
      raise NotImplementedError(self.dtype)
    layout = _m128_256bit_32bit_layout(self.shape)
    regs_shape = layout.registers_shape(self.shape)
    if self.layout == TMEMLayout(elements_in_tile=(TMEM_ROWS, 8)):
      # load_32xcols returns a 4xN array, but the FA tiling we use here tiles
      # columns before rows, and so it is Nx4 (after ignoring all 1 dims).
      registers = _load_32xcols(
          self.address, self.shape[1], self.dtype
      ).T.reshape(regs_shape)
    elif self.layout == TMEMLayout(elements_in_tile=(TMEM_ROWS, 128), column_tile_stride=2):
      if self.shape[1] % 128 != 0:
        raise ValueError(
            f"TMEM layout {self.layout} is not compatible with shape {self.shape}"
        )
      num_column_tiles = self.shape[1] // 128
      column_tile_stride = self.layout.column_tile_stride
      num_strided_col_groups = utils.ceil_div(num_column_tiles, column_tile_stride)
      tiles = []
      for col_tile_base in range(num_strided_col_groups):
        for col_tile in range(col_tile_base, num_column_tiles, column_tile_stride):
          tiles.append(
              _load_32xcols(
                  arith.addi(self.address, arith.constant(i32, col_tile * 128)),
                  cols=128,
                  dtype=self.dtype,
              )
          )
      registers = np.concatenate(tiles, axis=1).T.reshape(regs_shape)
    else:
      raise NotImplementedError(
          f"Loads only implemented for refs with standard layout, got: {self.layout}"
      )
    return fa.FragmentedArray(_registers=registers, _layout=layout, _is_signed=None)

def _load_32xcols(base_addr, cols, dtype):
  # See https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-16256b
  i32 = ir.IntegerType.get_signless(32)
  assert cols % 8 == 0
  cols_per_num_tile = 8
  load_shape = "16x256b"
  num = cols // 8
  if num <= 32:
    num_tiling = num
  elif num == 64:
    num_tiling = 32
  else:
    raise NotImplementedError(num)
  vector_regs = np.ndarray((4, num), dtype=object)
  # We load 16 lanes at a time, but need 32 in total.
  for row_group in range(2):
    addr_row = arith.addi(base_addr, arith.constant(i32, (row_group * 16) << 16))
    regs = []
    for num_group in range(num // num_tiling):
      addr_row_col = arith.addi(
          addr_row,
          arith.constant(i32, num_tiling * num_group * cols_per_num_tile),
      )
      regs += tmem_load(addr_row_col, load_shape, num_tiling)
    regs = [llvm.bitcast(dtype, r) for r in regs]
    undef = llvm.mlir_undef(ir.VectorType.get((2,), dtype))
    for r_low, r_high, idx in zip(regs[::2], regs[1::2], np.ndindex(num, 2)):
      high_undef = llvm.insertelement(undef, r_low, utils.c(0, i32))
      vreg = llvm.insertelement(high_undef, r_high, utils.c(1, i32))
      vector_regs[idx[1] + 2 * row_group, idx[0]] = vreg
  return vector_regs


def _m128_256bit_32bit_layout(shape: tuple[int, ...]):
  if len(shape) != 2:
    raise ValueError(f"Shape {shape} is not 2D")
  if shape[0] % 128 != 0 or shape[1] % 8 != 0:
    raise ValueError(f"Shape {shape} is not a multiple of 64x8")
  return fa.TiledLayout(
      fa.Tiling(((128, 8), (32, 8), (8, 8), (1, 2))),
      warp_dim=-8,
      lane_dims=(-4, -3),
      vector_dim=-1,
  )
