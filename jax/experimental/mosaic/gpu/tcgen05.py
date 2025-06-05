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
LAYOUT = fa.TCGEN05_LAYOUT
ROW_LAYOUT = fa.TCGEN05_ROW_LAYOUT
COL_LAYOUT = fa.TCGEN05_COL_LAYOUT

# A layout resembling the logical organization of TMEM. The 128 rows in a tile
# are assigned to 128 lanes in the warpgroup. Useful when the result needs to be
# processed in registers and then stored back into TMEM. Should not be used if
# the result is to be written back to SMEM, as there is no good way to store it
# without bank conflicts.
#
# We use a vector_dim of 2, to be able to make sure that the vectors are always
# a multiple of 32-bits, even when the data is 16-bits.
TMEM_NATIVE_LAYOUT = fa.TiledLayout(
    fa.Tiling(((128, 2), (32, 2))),
    warp_dim=-4,
    lane_dims=(-2,),
    vector_dim=-1,
)


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
  if acc_dtype not in {f32, f16}:
    raise NotImplementedError("Only float32 and float16 accumulators supported")
  if utils.bitwidth(input_dtype) == 16:
    if input_dtype not in {f16, bf16}:
      raise NotImplementedError(
          "The only supported 16-bit input types are float16 and bfloat16, got"
          f" {input_dtype}"
      )
    desc = 0
    desc |= (acc_dtype == f32) << 4  # D dtype, bits 4-5
    # Bit 6 is reserved
    desc |= (input_dtype == bf16) << 7  # A dtype, bits 7-9
    desc |= (input_dtype == bf16) << 10  # B dtype, bits 10-12
    return _finish_instr_descriptor(desc, m, n, transpose_a, transpose_b)
  elif utils.bitwidth(input_dtype) == 8:
    desc = 0
    desc |= (acc_dtype == f32) << 4  # D dtype, bits 4-5
    # Bit 6 is reserved
    if input_dtype == ir.Float8E4M3FNType.get():
      input_dtype_enum = 0
    elif input_dtype == ir.Float8E5M2Type.get():
      input_dtype_enum = 1
    else:
      raise NotImplementedError(f"Unsupported input dtype: {input_dtype}")
    desc |= input_dtype_enum << 7  # A dtype, bits 7-9
    desc |= input_dtype_enum << 10  # B dtype, bits 10-12
    return _finish_instr_descriptor(desc, m, n, transpose_a, transpose_b)
  else:
    raise NotImplementedError(f"Unsupported input dtype: {input_dtype}")


def _finish_instr_descriptor(
    desc: int, m: int, n: int, transpose_a: bool, transpose_b: bool,
):
  # We ignore sparsity in bits 0-3
  # A, B and D types are set by the caller
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
    a: ir.Value | TMEMRef,
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
  if not ir.MemRefType.isinstance(b.type):
    raise ValueError(f"B must be a memref, got: {b.type}")
  (k, n), element_type = mma_utils.tiled_memref_shape(b)
  if isinstance(a, TMEMRef):
    m, k2 = a.shape
    element_type2 = a.dtype
    if collective:
      raise NotImplementedError("Collective not supported for TMEMRef")
    if a.layout != (expected_layout := _infer_tmem_layout(a.shape, packing=2)):
      raise ValueError(
          f"A layout mismatch: expected {expected_layout}, got {a.layout}"
      )
  else:
    if not ir.MemRefType.isinstance(a.type):
      raise ValueError(f"A must be a memref, got {a.type}")
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
  expected_d_layout = (
      TMEM_COLLECTIVE_N512_LAYOUT
      if collective and n * num_cta == 512
      else TMEM_DEFAULT_LAYOUT
  )
  if d.layout != expected_d_layout:
    raise ValueError(
        f"Accumulator layout mismatch: expected {expected_d_layout}, got {d.layout}"
    )
  f32 = ir.F32Type.get()
  f16 = ir.F16Type.get()
  if element_type == f32 or element_type == ir.BF16Type.get():
    if d.dtype != f32:
      raise ValueError(
          f"MMA with element type {element_type} only supports accumulators"
          f" of type f32, but got: {d.dtype}"
      )
  elif any(
      t.isinstance(element_type)
      for t in {ir.F16Type, ir.Float8E5M2Type, ir.Float8E4M3FNType}
  ):
    if d.dtype != f16 and d.dtype != f32:
      raise ValueError(
          f"MMA with element type {element_type} only supports accumulators of"
          f" type f32 or f16, but got: {d.dtype}"
      )
  else:
    raise NotImplementedError(f"Unsupported element type: {element_type}")

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
  # TODO(apaszke): Require users to bitcast input refs to tf32 before MMA.
  mma_element_type = (
      ir.FloatTF32Type.get() if element_type == ir.F32Type.get() else element_type
  )

  # Step 3. Compute the operand descriptors.
  if not isinstance(a, TMEMRef):
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
  else:
    a_fastest = mma_utils.Dim.K
    a_k_instr_stride = None
    a_m_group_stride = a_k_group_stride = a_desc_base = None
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
    if isinstance(a, TMEMRef):
      a_mk = a.slice(slice(None), utils.ds(ki * k_group_elems, k_group_elems)).address
    else:
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
        d_type=d.dtype,
        m=m_group_elems,
        n=n_group_elems,
        collective=collective,
        a_transpose=a_fastest != mma_utils.Dim.K,
        b_transpose=b_fastest != mma_utils.Dim.K,
        a_k_stride=a_k_instr_stride,
        b_k_stride=b_k_instr_stride,
        accumulate=acc,
        swizzle=swizzle,
        element_type=mma_element_type,
    )


def _do_mma(
    d_addr: ir.Value,
    a_desc_or_addr: ir.Value,  # TMEM address if a_k_stride is None
    b_desc: ir.Value,
    a_transpose: bool,
    b_transpose: bool,
    a_k_stride: int | None,
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
  i32 = ir.IntegerType.get_signless(32)
  i64 = ir.IntegerType.get_signless(64)
  elem_bytewidth = utils.bytewidth(element_type)
  kn_tiling = swizzle // elem_bytewidth
  instr_k = 32 // elem_bytewidth
  packing = 4 // elem_bytewidth
  if (a_k_stride is not None and a_k_stride % 16) or b_k_stride % 16:
    raise ValueError

  if ir.F16Type.isinstance(element_type) or ir.BF16Type.isinstance(element_type):
    kind = "f16"
  elif ir.Float8E5M2Type.isinstance(element_type):
    kind = "f8f6f4"
  elif ir.Float8E4M3FNType.isinstance(element_type):
    kind = "f8f6f4"
  else:
    raise NotImplementedError(f"Unsupported input element type: {element_type}")

  num_cta = 2 if collective else 1
  i_desc = create_instr_descriptor(
      m * num_cta, n * num_cta, d_type, element_type, a_transpose, b_transpose
  )
  a_in_tmem = a_k_stride is None
  a_ptx = "[$1]" if a_in_tmem else "$1"
  a_ptx_constraint = "r" if a_in_tmem else "l"
  assert a_desc_or_addr.type == ir.IntegerType.get_signless(32 if a_in_tmem else 64)
  for _ in range(kn_tiling // instr_k):
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [d_addr, a_desc_or_addr, b_desc, i_desc, accumulate],
        f"tcgen05.mma.cta_group::{num_cta}.kind::{kind} [$0], {a_ptx}, $2, $3, $4;",
        f"r,{a_ptx_constraint},l,r,b",
        has_side_effects=True,
    )
    accumulate = arith.constant(i1, 1)
    if not a_in_tmem:
      a_desc_or_addr = arith.addi(
          a_desc_or_addr, arith.constant(i64, a_k_stride >> 4)
      )
    else:
      a_desc_or_addr = arith.addi(
          a_desc_or_addr, arith.constant(i32, instr_k // packing)
      )
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


def _alloc_ncols(ncols: int, exact: bool):
  if exact:
    if ncols.bit_count() != 1 or not 32 <= ncols <= 512:
      raise ValueError(f"ncols must be a power of 2 and within [32, 512], got: {ncols}")
  else:
    ncols = max(32, 1 << (ncols - 1).bit_length())
    if ncols > 512:
      raise ValueError(
          f"After rounding up, got {ncols} columns, exceeding the limit of 512"
      )
  return ncols


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
  ncols = _alloc_ncols(ncols, exact)
  num_cta = 2 if collective else 1
  return llvm.inline_asm(
      ir.Type.parse("!llvm.void"),
      [tmem_addr],
      f"tcgen05.alloc.cta_group::{num_cta}.sync.aligned.shared::cta.b32  [$0], {ncols};",
      "r",
      has_side_effects=True,
  )


def tmem_dealloc(tmem_addr: ir.Value, ncols: int, collective: bool = False, exact: bool = True):
  if tmem_addr.type != ir.IntegerType.get_signless(32):
    raise ValueError(f"tmem_addr must be an i32, got: {tmem_addr.type}")
  ncols = _alloc_ncols(ncols, exact)
  num_cta = 2 if collective else 1
  return llvm.inline_asm(
      ir.Type.parse("!llvm.void"),
      [tmem_addr],
      f"tcgen05.dealloc.cta_group::{num_cta}.sync.aligned.b32  $0, {ncols};",
      "r",
      has_side_effects=True,
  )


def tmem_relinquish_alloc_permit(collective: bool):
  num_cta = 2 if collective else 1
  return llvm.inline_asm(
      ir.Type.parse("!llvm.void"),
      [],
      f"tcgen05.relinquish_alloc_permit.cta_group::{num_cta}.sync.aligned;",
      "",
      has_side_effects=True,
  )

def _tmem_access_helper(shape, num):
  if num.bit_count() != 1 or num > 128:
    raise ValueError(f"num must be a power of 2 and <= 128, got: {num}")
  match shape:
    case "32x32b":
      num_regs = 1
    case "16x128b":
      num_regs = 2
    case "16x256b":
      num_regs = 4
    case _:
      raise NotImplementedError(f"{shape=} is unsupported")
  num_regs *= num
  if num_regs > 255:
    raise ValueError(
        f"TMEM translation too big : {shape=} and {num=} involve"
        f" {num_regs} registers per-thread, which exceeds the limit of 255"
    )
  regs_vector = ",".join(f"${i}" for i in range(num_regs))
  regs_vector = "{" + regs_vector + "}"
  return num_regs, regs_vector


def tmem_load(tmem_addr, shape, num, pack: bool):
  i32 = ir.IntegerType.get_signless(32)
  num_out_regs, regs_vector = _tmem_access_helper(shape, num)
  pack_mod = ".pack::16b" if pack else ""
  regs = llvm.inline_asm(
      ir.Type.parse(
          "!llvm.struct<(" + ",".join("i32" for _ in range(num_out_regs)) + ")>"
      ),
      [tmem_addr],
      f"tcgen05.ld.sync.aligned.{shape}.x{num}{pack_mod}.b32 {regs_vector}, [${num_out_regs}];",
      "=r," * num_out_regs + "r",
      has_side_effects=True,
  )
  return [llvm.extractvalue(i32, regs, [i]) for i in range(num_out_regs)]


def wait_tmem_load():
  llvm.inline_asm(
      ir.Type.parse("!llvm.void"),
      [],
      "tcgen05.wait::ld.sync.aligned;",
      "",
      has_side_effects=True,
  )
  utils.warpgroup_barrier()


def tmem_store(tmem_addr, shape, num, regs, unpack: bool):
  num_out_regs, regs_vector = _tmem_access_helper(shape, num)
  pack_mod = ".unpack::16b" if unpack else ""
  llvm.inline_asm(
      ir.Type.parse("!llvm.void"),
      [*regs, tmem_addr],
      f"tcgen05.st.sync.aligned.{shape}.x{num}{pack_mod}.b32 [${num_out_regs}], {regs_vector};",
      "r," * num_out_regs + "r",
      has_side_effects=True,
  )


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
  packing: int = 1

  def __post_init__(self):
    row_tiling = self.elements_in_tile[0]
    if not 32 <= row_tiling <= 128:
      raise ValueError(
          f"Row tiling must be between 32 and 128, got: {row_tiling}"
      )
    if row_tiling.bit_count() != 1:
      raise ValueError(f"Row tiling must be a power of 2, got: {row_tiling}")
    if self.elements_in_tile[1] % self.packing:
      raise ValueError(
          f"Column tiling must be a multiple of packing={self.packing}, got:"
          f" {self.elements_in_tile[1]}"
      )

  def check_type(self, shape: tuple[int, ...], dtype: ir.Type):
    if len(shape) != 2:
      raise ValueError(f"TMEM can only represent 2D shapes, got {shape}")
    if any(s % t for s, t in zip(shape, self.elements_in_tile)):
      raise ValueError(
          f"{shape} is divisible into tiles of shape {self.elements_in_tile}"
      )
    if self.packing not in {1, fully_packed := 32 // utils.bitwidth(dtype)}:
      raise ValueError(
          f"For {utils.bitwidth(dtype)}-bit types, only packing=1 and"
          f" packing={fully_packed} are supported, but got: {self.packing}"
      )

  def cols_in_shape(self, shape: tuple[int, int]):
    cols_in_tile = self.elements_in_tile[1] // self.packing
    tiles_in_row = TMEM_ROWS // self.elements_in_tile[0]
    num_tiles = math.prod(utils.tile_shape(shape, self.elements_in_tile)[:-2])
    assert num_tiles % tiles_in_row == 0
    return num_tiles // tiles_in_row * cols_in_tile


def _infer_tmem_layout(shape: tuple[int, int], packing: int = 1) -> TMEMLayout:
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
  return TMEMLayout(elements_in_tile=(shape[0], 8), packing=packing)


TMEM_DEFAULT_LAYOUT = TMEMLayout(elements_in_tile=(TMEM_ROWS, 8), packing=1)
TMEM_COLLECTIVE_N512_LAYOUT = TMEMLayout(
    elements_in_tile=(TMEM_ROWS, 128), column_tile_stride=2, packing=1
)

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
      layout.check_type(shape, dtype)
    # TODO: Do we have to do this??
    # warp_idx = utils.warp_idx(sync=False)
    # tmem_addr = arith.ori(tmem_addr, arith.shli(warp_idx, utils.c(21, i32)))
    return cls(tmem_addr, shape, dtype, layout)

  def slice(self, *idxs):
    i32 = ir.IntegerType.get_signless(32)
    base_idx, slice_shape, is_squeezed = utils.parse_indices(idxs, self.shape)
    if any(is_squeezed):
      raise ValueError("TMEM can only be sliced, not indexed")
    match self.layout:
      case TMEMLayout(elements_in_tile=(r, 8), packing=packing) if (
          r == TMEM_ROWS
      ):
        pass
      case _:
        raise NotImplementedError(
            "Slicing only implemented for refs with standard layout, got:"
            f" {self.layout}"
        )
    if base_idx[0] != 0 or slice_shape[0] != TMEM_ROWS:
      raise NotImplementedError("TMEM cannot be sliced along rows")
    if slice_shape[1] % 8:
      raise NotImplementedError(
          "TMEM column slice length must be a multiple of 8. "
          f"Got {slice_shape[1]}."
      )
    col_idx = base_idx[1]
    if not isinstance(col_idx, ir.Value):
      col_idx = arith.constant(i32, col_idx)
    if col_idx.type == ir.IndexType.get():
      col_idx = arith.index_cast(i32, col_idx)
    if packing != 1:
      col_idx = arith.divui(col_idx, arith.constant(i32, packing))
    return TMEMRef(
        address=arith.addi(self.address, col_idx),
        shape=tuple(slice_shape),
        layout=self.layout,
        dtype=self.dtype,
    )

  def load(self, layout: fa.TiledLayout = LAYOUT):
    i32 = ir.IntegerType.get_signless(32)
    if self.shape[1] % 8:
      raise NotImplementedError
    if utils.bitwidth(self.dtype) not in {16, 32}:
      raise NotImplementedError(f"Unsupported dtype: {self.dtype}")
    if layout == LAYOUT:
      regs_shape = layout.registers_shape(self.shape)
      match self.layout:
        case TMEMLayout(elements_in_tile=(r, 8), packing=packing) if (
            r == TMEM_ROWS
        ):
          # load_32xcols returns a 4xN array, but the FA tiling we use here tiles
          # columns before rows, and so it is Nx4 (after ignoring all 1 dims).
          registers = _load_32xcols(
              self.address, self.shape[1], self.dtype, packing
          ).T.reshape(regs_shape)
        case TMEMLayout(elements_in_tile=(r, 128), column_tile_stride=2) if r == TMEM_ROWS:
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
                      tmem_packing=1,
                  )
              )
          registers = np.concatenate(tiles, axis=1).T.reshape(regs_shape)
        case _:
          raise NotImplementedError(
              f"Loads only implemented for refs with standard layout, got: {self.layout}"
          )
    elif layout == TMEM_NATIVE_LAYOUT:
      regs_shape = layout.registers_shape(self.shape)
      match self.layout:
        case TMEMLayout(elements_in_tile=(r, c), packing=packing) if (
            r == TMEM_ROWS and c % 2 == 0
        ):
          registers = _load_32xcols_native(
              self.address, self.shape[1], self.dtype, packing
          ).reshape(regs_shape)
        case _:
          raise NotImplementedError(
              "Loads only implemented for refs with standard layout, got:"
              f" {self.layout}"
          )
    else:
      raise ValueError(
          "TMEM loads can only produce results in the tcgen05 layouts"
          f" ({LAYOUT} and {TMEM_NATIVE_LAYOUT}), but got: {layout}"
      )
    return fa.FragmentedArray(_registers=registers, _layout=layout, _is_signed=None)

  def store(self, value):
    if self.shape[1] % 8:
      raise NotImplementedError
    if utils.bitwidth(self.dtype) not in {16, 32}:
      raise NotImplementedError(f"Unsupported dtype: {self.dtype}")
    if not isinstance(value, fa.FragmentedArray):
      raise ValueError(f"TMEM stores expect a FragmentedArray, got: {value}")
    if value.shape != self.shape:
      raise ValueError(
          f"Stored array has shape {value.shape}, but TMEM has shape"
          f" {self.shape}"
      )
    if value.mlir_dtype != self.dtype:
      raise ValueError(
          f"Stored array has dtype {value.mlir_dtype}, but TMEM has dtype"
          f" {self.dtype}"
      )
    if value.layout == LAYOUT:
      # TODO(apaszke): Collective MMA layout
      match self.layout:
        case TMEMLayout(elements_in_tile=(r, 8), packing=packing) if (
            r == TMEM_ROWS
        ):
          # store_32xcols needs a 4xN array, but the FA tiling we use here tiles
          # columns before rows, and so it is Nx4 (after ignoring all 1 dims).
          _store_32xcols(
              self.address, value.registers.T.reshape((4, -1)), packing
          )
        case _:
          raise NotImplementedError(
              f"Stores only implemented for refs with standard layout, got: {self.layout}"
          )
    elif value.layout == TMEM_NATIVE_LAYOUT:
      # TODO(apaszke): Collective MMA layout
      match self.layout:
        case TMEMLayout(elements_in_tile=(r, c), packing=packing) if (
            r == TMEM_ROWS and c % 2 == 0
        ):
          _store_32xcols_native(
              self.address, value.registers.reshape(-1), packing
          )
        case _:
          raise NotImplementedError(
              f"Stores only implemented for refs with standard layout, got: {self.layout}"
          )
    else:
      raise ValueError(
          f"Stored array has layout {value.layout}, but only tcgen05.LAYOUT and"
          " tcgen05.TMEM_NATIVE_LAYOUT are supported"
      )

  def _debug_print(self):
    i32 = ir.IntegerType.get_signless(32)
    num_cols = self.layout.cols_in_shape(self.shape)
    lane = arith.remui(utils.thread_idx(), arith.constant(i32, utils.WARPGROUP_SIZE))
    for c in range(num_cols):
      val = llvm.inline_asm(
          i32,
          [arith.addi(self.address, arith.constant(i32, c))],
          "tcgen05.ld.sync.aligned.32x32b.x1.b32 {$0}, [$1];",
          "=r,r",
      )
      dtype_bitwidth = utils.bitwidth(self.dtype)
      full_packing = 32 // dtype_bitwidth
      if self.layout.packing == 1:
        if dtype_bitwidth < 32:
          val = arith.trunci(ir.IntegerType.get_signless(dtype_bitwidth), val)
        val = utils.bitcast(val, self.dtype)
      elif self.layout.packing == full_packing:
        val = utils.bitcast(val, ir.VectorType.get((full_packing,), self.dtype))
      else:
        raise NotImplementedError(f"Unsupported packing: {self.layout.packing}")
      # TODO(apaszke): Make this print logical, not physical location.
      utils.debug_print(f"[{{}}, {c}]: {{}}", lane, val, uniform=False)


def _transfer_32xcols(
    base_addr: ir.Value,
    cols: int,
    atom_shape: tuple[int, int],
    tmem_packing: int,
    reg_packing: int,
):
  """Generates a sequence of parameters for a given TMEM read or write.

  Arguments:
    base_addr: The base address of the TMEM region.
    cols: The number of logical columns to transfer.
    atom_shape: The logical shape of the tile written by the warp in a single
      TMEM transfer.
    tmem_packing: Packing degree in TMEM. When packing is 1, but the data is
      16-bit, we expect that each transfer actually involves double the number
      of physical columns.
    reg_packing: The number of elements that fit in a single 32-bit register.
  """
  i32 = ir.IntegerType.get_signless(32)
  atom_rows, atom_cols = atom_shape
  assert cols % atom_cols == 0
  total_num = cols // atom_cols
  assert total_num.bit_count() == 1
  regs_per_instr = atom_shape[0] * atom_shape[1] // (utils.WARP_SIZE * reg_packing)
  # We artificially lower the instr_num compared to its limits, because higher
  # values can lead to register spills..
  instr_num = min(total_num, 32 // regs_per_instr)
  assert 32 % atom_rows == 0
  num_row_steps = 32 // atom_rows
  for lane_step in range(num_row_steps):
    addr_row = arith.addi(base_addr, utils.c((lane_step * atom_rows) << 16, i32))
    cols_per_instr = instr_num * atom_cols
    for num_step in range(total_num // instr_num):
      num_slice = slice(num_step * instr_num, (num_step + 1) * instr_num)
      addr_row_col = arith.addi(
          addr_row, utils.c(num_step * cols_per_instr // tmem_packing, i32)
      )
      yield addr_row_col, instr_num, lane_step, num_slice


def _store_32xcols(base_addr, vector_regs, tmem_packing):
  i32 = ir.IntegerType.get_signless(32)
  assert vector_regs.ndim == 2 and vector_regs.shape[0] == 4
  cols = vector_regs.shape[1] * 8

  reg_packing = 64 // utils.bitwidth(vector_regs.flat[0].type)
  if reg_packing == 1:
    store_shape = "16x256b"  # 4 threads * 64 bits per vreg = 256 bits
    regs = np.empty((4, vector_regs.shape[1], 2), dtype=object)
    c0 = arith.constant(i32, 0)
    c1 = arith.constant(i32, 1)
    for idx, vreg in np.ndenumerate(vector_regs):
      regs[(*idx, 0)] = llvm.extractelement(vreg, c0)
      regs[(*idx, 1)] = llvm.extractelement(vreg, c1)
    regs = regs.reshape(2, 2, vector_regs.shape[1], 2).swapaxes(1, 2)
    # From a single lane perspective a num tile consists of a 2x2, with the
    # minor dim traversing columns and major being 8 rows apart.
    # See https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-16256b
    assert regs.shape[-2:] == (2, 2)
    assert tmem_packing == 1
    unpack = False
  elif reg_packing == 2:
    store_shape = "16x128b"  # 4 threads * 32 bits per vreg = 128 bits
    # From a single lane perspective a num tile has 2 registers, 8 rows apart.
    # See https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-16128b
    regs = vector_regs.reshape(2, 2, vector_regs.shape[1]).swapaxes(1, 2)
    assert 1 <= tmem_packing <= 2
    unpack = tmem_packing == 1
  else:
    raise NotImplementedError(reg_packing)

  it = _transfer_32xcols(base_addr, cols, (16, 8), tmem_packing, reg_packing)
  for addr_row_col, instr_num, lane_step, num_slice in it:
    regs_slice = regs[lane_step, num_slice].flat
    tmem_store(addr_row_col, store_shape, instr_num, regs_slice, unpack)


def _store_32xcols_native(base_addr, vector_regs, tmem_packing):
  i32 = ir.IntegerType.get_signless(32)
  assert vector_regs.ndim == 1
  cols = len(vector_regs) * TMEM_NATIVE_LAYOUT.vector_length

  reg_packing = 64 // utils.bitwidth(vector_regs.flat[0].type)
  store_shape = "32x32b"
  if reg_packing == 1:
    store_atom_shape = (32, 1)
    regs = [None] * (len(vector_regs) * 2)
    c0 = arith.constant(i32, 0)
    c1 = arith.constant(i32, 1)
    for idx, vreg in enumerate(vector_regs):
      regs[2 * idx] = llvm.extractelement(vreg, c0)
      regs[2 * idx + 1] = llvm.extractelement(vreg, c1)
    assert tmem_packing == 1
    unpack = False
  elif reg_packing == 2:
    store_atom_shape = (32, 2)
    regs = vector_regs
    assert 1 <= tmem_packing <= 2
    unpack = tmem_packing == 1
  else:
    raise NotImplementedError(reg_packing)

  it = _transfer_32xcols(base_addr, cols, store_atom_shape, tmem_packing, reg_packing)
  for addr_row_col, instr_num, lane_step, num_slice in it:
    assert lane_step == 0
    regs_slice = regs[num_slice]
    tmem_store(addr_row_col, store_shape, instr_num, regs_slice, unpack)


def _load_32xcols(base_addr, cols, dtype, tmem_packing):
  i32 = ir.IntegerType.get_signless(32)
  vec_ty = ir.VectorType.get((2,), dtype)
  reg_packing = 32 // utils.bitwidth(dtype)
  if reg_packing == 1:
    load_shape = "16x256b"  # 4 threads * 64 bits per vreg = 256 bits
    assert tmem_packing == 1
    pack = False
  elif reg_packing == 2:
    load_shape = "16x128b"  # 4 threads * 32 bits per vreg = 128 bits
    assert 1 <= tmem_packing <= 2
    pack = tmem_packing == 1
  else:
    raise NotImplementedError(reg_packing)

  vector_regs = np.ndarray((4, cols // 8), dtype=object)

  it = _transfer_32xcols(base_addr, cols, (16, 8), tmem_packing, reg_packing)
  c0 = arith.constant(i32, 0)
  c1 = arith.constant(i32, 1)
  for addr_row_col, instr_num, lane_step, num_slice in it:
    regs = tmem_load(addr_row_col, load_shape, instr_num, pack)
    row_slice = slice(lane_step * 2, (lane_step + 1) * 2)
    # This aliases the original array, so updates will be reflected there.
    vector_regs_update = vector_regs[row_slice, num_slice]
    assert vector_regs_update.shape == (2, instr_num), (vector_regs_update.shape, instr_num)
    if reg_packing == 1:
      regs = [llvm.bitcast(dtype, r) for r in regs]
      # From a single lane perspective a num tile consists of a 2x2, with the
      # minor dim traversing columns and major being 8 rows apart.
      # See https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-16256b
      regs = np.asarray(regs, dtype=object).reshape(instr_num, 2, 2).swapaxes(0, 1)
      undef = llvm.mlir_undef(vec_ty)
      assert regs.shape == (*vector_regs_update.shape, 2)
      for idx in np.ndindex(vector_regs_update.shape):
        high_undef = llvm.insertelement(undef, regs[(*idx, 0)], c0)
        vreg = llvm.insertelement(high_undef, regs[(*idx, 1)], c1)
        vector_regs_update[idx] = vreg
    else:
      assert reg_packing == 2
      regs = [llvm.bitcast(vec_ty, r) for r in regs]
      # From a single lane perspective a num tile has 2 registers, 8 rows apart.
      # See https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-16128b
      regs = np.asarray(regs, dtype=object).reshape(instr_num, 2).swapaxes(0, 1)
      vector_regs_update[...] = regs

  return vector_regs


def _load_32xcols_native(base_addr, cols, dtype, tmem_packing):
  i32 = ir.IntegerType.get_signless(32)
  vec_ty = ir.VectorType.get((2,), dtype)
  reg_packing = 32 // utils.bitwidth(dtype)
  load_shape = "32x32b"
  if reg_packing == 1:
    load_atom_shape = (32, 1)
    assert tmem_packing == 1
    pack = False
  elif reg_packing == 2:
    load_atom_shape = (32, 2)
    assert 1 <= tmem_packing <= 2
    pack = tmem_packing == 1
  else:
    raise NotImplementedError(reg_packing)

  it = _transfer_32xcols(base_addr, cols, load_atom_shape, tmem_packing, reg_packing)
  c0 = arith.constant(i32, 0)
  c1 = arith.constant(i32, 1)
  regs = [None] * (cols // reg_packing)
  for addr_row_col, instr_num, lane_step, num_slice in it:
    assert lane_step == 0, lane_step
    instr_regs = tmem_load(addr_row_col, load_shape, instr_num, pack)
    if reg_packing == 1:
      regs[num_slice] = [llvm.bitcast(dtype, r) for r in instr_regs]
    else:
      assert reg_packing == 2
      regs[num_slice] = [llvm.bitcast(vec_ty, r) for r in instr_regs]

  if reg_packing == 1:
    vector_regs = np.ndarray((cols // 2,), dtype=object)
    undef = llvm.mlir_undef(vec_ty)
    for idx in range(vector_regs.size):
      high_undef = llvm.insertelement(undef, regs[2 * idx], c0)
      vreg = llvm.insertelement(high_undef, regs[2 * idx + 1], c1)
      vector_regs[idx] = vreg
  else:
    assert reg_packing == 2
    vector_regs = np.asarray(regs, dtype=object)

  assert vector_regs.shape == (cols // TMEM_NATIVE_LAYOUT.vector_length,)
  return vector_regs


def _m128_layout(shape: tuple[int, ...]):
  if len(shape) != 2:
    raise ValueError(f"Shape {shape} is not 2D")
  if shape[0] % 128 != 0 or shape[1] % 8 != 0:
    raise ValueError(f"Shape {shape} is not a multiple of 64x8")
  return LAYOUT


def commit_tmem():
  void = ir.Type.parse("!llvm.void")
  llvm.inline_asm(
      void, [], "tcgen05.wait::st.sync.aligned;", "", has_side_effects=True,
  )
  utils.warpgroup_barrier()
