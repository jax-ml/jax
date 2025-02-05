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

from jax._src import dtypes
from jax._src.lib import mosaic_gpu_dialect as mgpu_dialect
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import memref
import numpy as np

from . import utils
from . import fragmented_array as fa
from . import _wgmma

def create_smem_descriptor(
    memref_arg,
    leading_byte_offset: int,
    stride_byte_offset: int,
    swizzle: int | mgpu_dialect.SwizzlingMode | None,
):
  blackwell_bit = 1 << 46
  return _wgmma.create_descriptor(
      memref_arg,
      leading_byte_offset,
      stride_byte_offset,
      swizzle,
      memory_space=3,
      const_init=blackwell_bit,
  )

def create_instr_descriptor(
    m: int,
    n: int,
    acc_dtype,
    input_dtype,
    transpose_a: bool = False,
    transpose_b: bool = False,
):
  if input_dtype not in {np.float16, dtypes.bfloat16}:
    raise NotImplementedError("Only float16 and bfloat16 inputs supported")
  if acc_dtype not in {np.float32, np.float16}:
    raise NotImplementedError("Only float32 and float16 accumulators supported")

  desc = 0
  # We ignore sparsity in bits 0-3
  desc |= (acc_dtype == np.float32) << 4  # D dtype, bits 4-5
  # Bit 6 is reserved
  desc |= (input_dtype == dtypes.bfloat16) << 7  # A dtype, bits 7-9
  desc |= (input_dtype == dtypes.bfloat16) << 10  # B dtype, bits 10-12
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
  return arith.constant(ir.IntegerType.get_signless(32), desc)  # type: ignore


def mma(dtype, num_cta, d_tmem, adesc, bdesc, idesc, enable_input_d):
  if not (1 <= num_cta <= 2):
    raise ValueError(f"num_cta must be 1 or 2, got: {num_cta}")
  return llvm.inline_asm(
      ir.Type.parse("!llvm.void"),
      [d_tmem, adesc, bdesc, idesc, enable_input_d],
      f"tcgen05.mma.cta_group::1.kind::{dtype} [$0], $1, $2, $3, $4;",
      "r,l,l,r,b",
      has_side_effects=True,
  )

def commit_arrive(barrier):
  return llvm.inline_asm(
      ir.Type.parse("!llvm.void"),
      [barrier],
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [$0];",
      "l",
      has_side_effects=True
  )

def tmem_alloc(tmem_addr, ncols: int):
  if ncols.bit_count() != 1 or not 32 <= ncols <= 512:
    raise ValueError(f"ncols must be a power of 2 and within [32, 512], got: {ncols}")
  return llvm.inline_asm(
      ir.Type.parse("!llvm.void"),
      [tmem_addr],
      f"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32  [$0], {ncols};",
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


class TMEMLayout(enum.Enum):
  """Layout of the array in TMEM.

  See https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-organization
  """
  D = "D"


@dataclasses.dataclass(frozen=True)
class TMEMRef:
  address: ir.Value
  layout: TMEMLayout
  num_cols: int
  dtype: ir.Type

  @classmethod
  def from_alloc(cls, tmem_addr_ref: ir.Value, layout: TMEMLayout, num_cols: int, dtype: ir.Type):
    i32 = ir.IntegerType.get_signless(32)
    if not ir.MemRefType.isinstance(tmem_addr_ref.type):
      raise ValueError(f"tmem_addr_ref must be a memref or a pointer, got: {tmem_addr_ref.type}")
    addr_ref_ty = ir.MemRefType(tmem_addr_ref.type)
    smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
    if addr_ref_ty.memory_space != smem:
      raise ValueError(f"tmem_addr_ref must be in workgroup memory, got: {addr_ref_ty}")
    if addr_ref_ty.element_type != i32:
      raise ValueError(f"tmem_addr_ref must be an i32 memref, got: {addr_ref_ty}")
    tmem_addr = memref.load(tmem_addr_ref, [arith.ConstantOp.create_index(0)])
    # TODO: Do we have to do this??
    # warp_idx = utils.warp_idx(sync=False)
    # tmem_addr = arith.ori(tmem_addr, arith.shli(warp_idx, utils.c(21, i32)))
    return cls(tmem_addr, layout, num_cols, dtype)

  @property
  def num_rows(self):
    match self.layout:
      case TMEMLayout.D:
        return 128
      case _:
        raise NotImplementedError(self.layout)

  @property
  def shape(self):
    return (self.num_rows, self.num_cols)

  def __getitem__(self, *idxs):
    i32 = ir.IntegerType.get_signless(32)
    base_idxs, slice_shape, is_squeezed = utils.parse_indices(idxs, self.shape)
    if any(is_squeezed):
      raise ValueError("TMEM loads only support slicing")
    if any(idx != 0 for idx in base_idxs) or tuple(slice_shape) != self.shape:
      raise NotImplementedError("Slicing of TMEM not impelmented yet")
    if self.layout != TMEMLayout.D:
      raise NotImplementedError(self.layout)
    if self.num_cols % 8:
      raise NotImplementedError
    if self.dtype != ir.F32Type.get():
      raise NotImplementedError(self.dtype)
    layout = _m128_256bit_32bit_layout(self.shape)
    regs_shape = layout.registers_shape(self.shape)
    num = self.num_cols // 8
    registers = np.empty(regs_shape, dtype=object)
    # We load 16 lanes at a time, but need 32 in total.
    for row_group in range(2):
      addr = arith.addi(self.address, arith.constant(i32, (row_group * 16) << 16))
      regs = tmem_load(addr, "16x256b", num)
      regs = [llvm.bitcast(self.dtype, r) for r in regs]
      vector_regs = []
      undef = llvm.mlir_undef(ir.VectorType.get((2,), self.dtype))
      for r_low, r_high in zip(regs[::2], regs[1::2]):
        high_undef = llvm.insertelement(undef, r_low, utils.c(0, i32))
        vreg = llvm.insertelement(high_undef, r_high, utils.c(1, i32))
        vector_regs.append(vreg)
      # Dimension 4 is the one where we split 32 rows into tiles of 8.
      regs_slice = [slice(None)] * 4 + [slice(row_group * 2, (row_group + 1) * 2)]
      registers[*regs_slice] = np.asarray(vector_regs, dtype=object).reshape(registers[*regs_slice].shape)
    return fa.FragmentedArray(_registers=registers, _layout=layout, _is_signed=None)


def _m128_256bit_32bit_layout(shape: tuple[int, ...]):
  """Returns a tiled layout that is easy to relayout to WGMMA layout after doubling the bitwidth."""
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
