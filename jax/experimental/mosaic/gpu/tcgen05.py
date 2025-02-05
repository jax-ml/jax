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

from jax._src import dtypes
from jax._src.lib import mosaic_gpu_dialect as mgpu_dialect
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import llvm
import numpy as np

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

def tmem_alloc(tmem_addr, ncols):
  return llvm.inline_asm(
      ir.Type.parse("!llvm.void"),
      [tmem_addr, ncols],
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32  [$0], $1;",
      "r,r",
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

def tmem_load(tmem_addr, num):
  if num.bit_count() != 1 or num > 128:
    raise ValueError(f"num must be a power of 2 and <= 128, got: {num}")
  i32 = ir.IntegerType.get_signless(32)
  out_regs = ",".join("$" + str(i) for i in range(num))
  regs = llvm.inline_asm(
      ir.Type.parse(
          "!llvm.struct<(" + ",".join("i32" for _ in range(num)) + ")>"
      ),
      [tmem_addr],
      f"tcgen05.ld.sync.aligned.32x32b.x{num}.b32    {{{out_regs}}}, [${num}];",
      "=r," * num + "r",
      has_side_effects=True,
  )
  out_ty = ir.VectorType.get([num], i32)
  out_vec = llvm.mlir_undef(out_ty)
  for i in range(num):
    out_vec = llvm.insertelement(
        out_vec, llvm.extractvalue(i32, regs, [i]), arith.constant(i32, i)
    )
  return out_vec
