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

from jax._src.lib import mosaic_gpu_dialect as mgpu_dialect
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import memref
import numpy as np

from . import utils
from . import fragmented_array as fa
from . import _wgmma
from .launch_context import LaunchContext

# MyPy does a terrible job with the MLIR API.
# mypy: ignore-errors


TMEM_ROWS = 128
TCGEN05_SMEM_DESCRIPTOR_BIT = 1 << 46

def create_smem_descriptor(
    memref_arg,
    leading_byte_offset: int,
    stride_byte_offset: int,
    swizzle: int | mgpu_dialect.SwizzlingMode | None,
):
  return _wgmma.create_descriptor(
      memref_arg,
      leading_byte_offset,
      stride_byte_offset,
      swizzle,
      memory_space=3,
      const_init=TCGEN05_SMEM_DESCRIPTOR_BIT,
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
  i64 = ir.IntegerType.get_signless(64)
  if not ir.MemRefType.isinstance(a.type):
    raise ValueError(f"A must be a memref, got {a.type}")
  if not ir.MemRefType.isinstance(b.type):
    raise ValueError(f"B must be a memref, got: {b.type}")
  if a_swizzle != b_swizzle:
    raise NotImplementedError(f"{a_swizzle=} != {b_swizzle=}")
  if isinstance(accumulate, bool):
    accumulate = arith.constant(ir.IntegerType.get_signless(1), accumulate)
  num_cta = 2 if collective else 1

  (
      a_desc_base,
      b_desc_base,
      (m, k, n),
      (m_mem_tiling, kn_mem_tiling),
      element_type,
      mma_params,
      a_k_byte_stride,
      b_k_byte_stride,
  ) = _wgmma._validate_mma(
      a,
      b,
      a_swizzle,
      _wgmma.WGMMALayout.ROW_MAJOR,
      _wgmma.WGMMALayout.COL_MAJOR,
      descriptor_const_init=TCGEN05_SMEM_DESCRIPTOR_BIT,
  )

  # The sizes of instruction we'll be using
  k_instr_tiling = kn_mem_tiling
  if (m_instr_tiling := d.layout.elements_in_tile[0]) != m_mem_tiling:
    raise ValueError(
        f"A's row tiling must be equal to {m_instr_tiling} (inferred from"
        f" accumulator's TMEM layout), got: {m_mem_tiling}"
    )
  if n * num_cta <= 256:
    n_instr_tiling = n
  elif n * num_cta == 512:
    if collective:
      raise NotImplementedError("Collective MMA with effective N=512 is unsupported")
    n_instr_tiling = 256
  else:
    raise NotImplementedError("The only supported N larger than 256 is 512")

  a_strides, _ = ir.MemRefType(a.type).get_strides_and_offset()
  a_m_byte_stride = a_strides[0] * utils.bytewidth(element_type)
  b_strides, _ = ir.MemRefType(b.type).get_strides_and_offset()
  b_n_byte_stride = b_strides[1] * utils.bytewidth(element_type)

  groups_k = k // k_instr_tiling
  groups_m = m // m_instr_tiling
  groups_n = n // n_instr_tiling
  n_mem_tiles_per_instr = n_instr_tiling // kn_mem_tiling

  # TODO(apaszke): Verify that the cluster shape matches the expectation of
  # collective MMA.
  expected_acc_shape = (m, n * (2 if collective else 1))
  if d.shape != expected_acc_shape:
    raise ValueError(
        f"Accumulator shape mismatch: expected {expected_acc_shape}, got {d.shape}"
    )

  true = arith.constant(ir.IntegerType.get_signless(1), 1)
  for mi, ni, ki in np.ndindex(groups_m, groups_n, groups_k):
    a_offset = mi * a_m_byte_stride + ki * a_k_byte_stride
    a_mk = arith.addi(a_desc_base, utils.c(_wgmma.wgmma_encode(a_offset), i64))
    b_offset = ni * n_mem_tiles_per_instr * b_n_byte_stride + ki * b_k_byte_stride
    b_nk = arith.addi(b_desc_base, utils.c(_wgmma.wgmma_encode(b_offset), i64))
    if groups_m != 1:
      raise NotImplementedError("D needs to be sliced")
    acc = accumulate if ki == 0 else true
    _do_mma(
        d.slice(
            slice(None), utils.ds(ni * n_instr_tiling, n_instr_tiling)
        ).address,
        a_mk,
        b_nk,
        d_type=ir.F32Type.get(),
        m=m_instr_tiling,
        n=n_instr_tiling,
        collective=collective,
        **mma_params,
        accumulate=acc,
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
  """
  elements_in_tile: tuple[int, int]

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


def _infer_tmem_layout(shape: tuple[int, int]) -> TMEMLayout:
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
  return TMEMLayout(elements_in_tile=(shape[0], 1))


@dataclasses.dataclass(frozen=True)
class TMEMRef:
  address: ir.Value
  shape: tuple[int, int]
  dtype: ir.Type
  layout: TMEMLayout

  @classmethod
  def from_alloc(cls, tmem_addr_ref: ir.Value, shape: tuple[int, int], dtype, layout: TMEMLayout | None = None):
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
      layout = _infer_tmem_layout(shape)
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
    if self.layout.elements_in_tile[0] != TMEM_ROWS:
      raise NotImplementedError(
          f"Slicing only implemented for refs with tiling of {TMEM_ROWS} rows"
      )
    if base_idx[0] != 0 or slice_shape[0] != TMEM_ROWS:
      raise NotImplementedError("TMEM cannot be sliced along rows")
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
    if self.layout.elements_in_tile[0] != TMEM_ROWS:
      raise NotImplementedError(
          f"Loads only implemented for refs with tiling of {TMEM_ROWS} rows"
      )
    if self.shape[1] % 8:
      raise NotImplementedError
    if self.dtype != ir.F32Type.get():
      raise NotImplementedError(self.dtype)
    layout = _m128_256bit_32bit_layout(self.shape)
    regs_shape = layout.registers_shape(self.shape)
    num = self.shape[1] // 8
    # TODO(apaszke): Make the tiling configurable through the args too.
    if num <= 32:
      num_tiling = num
    elif num == 64:
      num_tiling = 32
    else:
      raise NotImplementedError(num)
    registers = np.empty(regs_shape, dtype=object)
    # We load 16 lanes at a time, but need 32 in total.
    for row_group in range(2):
      addr_row = arith.addi(self.address, arith.constant(i32, (row_group * 16) << 16))
      regs = []
      cols_per_num_tile = 8  # This depends on the 16x256b below.
      for num_group in range(num // num_tiling):
        addr_row_col = arith.addi(
            addr_row,
            arith.constant(i32, num_tiling * num_group * cols_per_num_tile),
        )
        regs += tmem_load(addr_row_col, "16x256b", num_tiling)
      regs = [llvm.bitcast(self.dtype, r) for r in regs]
      vector_regs = []
      undef = llvm.mlir_undef(ir.VectorType.get((2,), self.dtype))
      for r_low, r_high in zip(regs[::2], regs[1::2]):
        high_undef = llvm.insertelement(undef, r_low, utils.c(0, i32))
        vreg = llvm.insertelement(high_undef, r_high, utils.c(1, i32))
        vector_regs.append(vreg)
      # Dimension 4 is the one where we split 32 rows into tiles of 8.
      regs_slice = (slice(None),) * 4 + (slice(row_group * 2, (row_group + 1) * 2),)
      registers[regs_slice] = np.asarray(vector_regs, dtype=object).reshape(registers[regs_slice].shape)
    return fa.FragmentedArray(_registers=registers, _layout=layout, _is_signed=None)


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
