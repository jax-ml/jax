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

import math
from jax.experimental.mosaic.gpu import fragmented_array as fa
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import vector
from . import mma_utils
from . import utils


def mma_m16(
    acc: fa.FragmentedArray, a: fa.FragmentedArray, b: fa.FragmentedArray
) -> fa.FragmentedArray:
  """Performs a 16x8x16 matrix multiplication using MMA instructions."""
  index = ir.IndexType.get()
  if math.prod(a.shape) != 16 * 16 * 4:
    raise ValueError(f"Unexpected shape for a: {a.shape}")
  if math.prod(b.shape) != 16 * 8 * 4:
    raise ValueError(f"Unexpected shape for b: {b.shape}")
  if math.prod(acc.shape) != 16 * 8 * 4:
    raise ValueError(f"Unexpected shape for acc: {acc.shape}")

  if a.mlir_dtype != b.mlir_dtype:
    raise ValueError(
        f"Operands must have the same dtype, but found {a.mlir_dtype} and"
        f" {b.mlir_dtype}"
    )

  if a.mlir_dtype not in (ir.F16Type.get(), ir.BF16Type.get()):
    raise ValueError(f"Unsupported operand dtype: {a.mlir_dtype}")
  if acc.mlir_dtype != ir.F32Type.get():
    raise ValueError(f"Unsupported accumulator dtype: {acc.mlir_dtype}")

  num_acc_regs = 4
  num_a_regs = 4
  num_b_regs = 2

  acc_regs = [  # pylint: disable=g-complex-comprehension
      vector.extractelement(reg, position=utils.c(pos, index))
      for reg in acc.registers.flat
      for pos in range(2)
  ]
  a_regs = [mma_utils.as_i32_reg(r) for r in a.registers.flatten()]
  b_regs = [mma_utils.as_i32_reg(r) for r in b.registers.flatten()]

  if a.mlir_dtype == ir.F16Type.get():
    instr = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
  elif a.mlir_dtype == ir.BF16Type.get():
    instr = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
  else:
    raise ValueError(f"Unsupported dtype {a.mlir_dtype}")
  out_regs_str = "{" + ",".join([f"$0", f"$1", f"$2", f"$3"]) + "}"
  a_regs_str = "{" + ",".join([f"${num_acc_regs+i}" for i in range(num_a_regs)]) + "}"
  b_regs_str = (
      "{" + ",".join([f"${num_acc_regs+num_a_regs+i}" for i in range(num_b_regs)]) + "}"
  )
  c_regs_str = (
      "{"
      + ",".join([f"${num_acc_regs+ num_a_regs + num_b_regs + i}" for i in range(num_acc_regs)])
      + "}"
  )
  ptx = f"{instr} {out_regs_str}, {a_regs_str}, {b_regs_str}, {c_regs_str};"
  # See: https://llvm.org/docs/LangRef.html#inline-assembler-expressions
  constraints = (
      f"{','.join(['=f']*num_acc_regs)},"  # Output accumulator regs
      f"{','.join(['r']*num_a_regs)},"  # Input A regs
      f"{','.join(['r']*num_b_regs)},"
      f"{','.join(['f']*num_acc_regs)}"  # Input accumulator regs
  )

  in_operands = [*a_regs, *b_regs, *acc_regs]
  acc_struct_type = ir.Type.parse(
      f"!llvm.struct<({','.join(str(acc.mlir_dtype) for _ in acc_regs)})>"
  )
  out_regs_struct = llvm.inline_asm(
      acc_struct_type,
      in_operands,
      ptx,
      constraints,
      asm_dialect=0,
      has_side_effects=False,
  )
  out_regs = [
      llvm.extractvalue(acc.mlir_dtype, out_regs_struct, [i])
      for i in range(len(acc_regs))
  ]
  out_regs = mma_utils.as_fragmented_reg_ndarray(
      out_regs, dtype=acc.mlir_dtype, shape=acc.registers.shape
  )
  return fa.FragmentedArray(
      _registers=out_regs, _layout=acc.layout, _is_signed=None
  )


# TODO(cperivol): More datatypes other than (b)f16.
def mma_map_m16n8k16(
    acc: fa.FragmentedArray, a: fa.FragmentedArray, b: fa.FragmentedArray
) -> fa.FragmentedArray:
  """Computes `acc + a @ b` using m16n8k16 MMA instructions.

  This function computes the matrix multiplication of `a` and `b` and adds it
  to `acc`. The operation is tiled, meaning that the inputs are divided into
  16x16 (for `a`), 16x8 (for `b`), and 16x8 (for `acc`) tiles, and the MMA
  instruction is applied to each tile.

  All operands must have `TiledLayout`s. The layouts are defined by the
  `make_mma_layout` function, which ensures that the tiles are mapped to the
  warps correctly. The warp mapping is defined by the `warp_shape` argument
  of `make_mma_layout`, which can be (1, 4), (2, 2), or (4, 1).

  Args:
    acc: A `FragmentedArray` representing the accumulator.
    a: A `FragmentedArray` representing the left-hand side operand.
    b: A `FragmentedArray` representing the right-hand side operand.

  Returns:
    A new `FragmentedArray` with the result of the computation.
  """

  # Do not modify the accumualtor itself.
  acc = acc.copy()

  (m, k) = a.shape
  (n, k1) = b.shape
  (m1, n1) = acc.shape


  if a.mlir_dtype != ir.BF16Type.get() or b.mlir_dtype != ir.BF16Type.get():
    raise NotImplementedError("Only bf16 supported")

  if not isinstance(a.layout, fa.TiledLayout):
    raise ValueError("Only tiled layouts are supported for mma")
  if not isinstance(b.layout, fa.TiledLayout):
    raise ValueError("Only tiled layouts are supported for mma")
  if not isinstance(acc.layout, fa.TiledLayout):
    raise ValueError("Only tiled layouts are supported for mma")

  m_tile, k_tile = a.layout.base_tile_shape
  n_tile, k_tile2 = b.layout.base_tile_shape
  warp_layout = (m_tile // 16, n_tile // 8)

  # We need to be able to tile by m16n8k16 and map
  ns, ms = m // m_tile, m // m_tile
  if m != m1:
    raise ValueError(f"M mismatch: {m} != {m1}")
  if n // 8 != m // 16:
    raise ValueError(f"M/N mismatch: {n // 8} != {m // 16}")
  if k != k1:
    raise ValueError(f"K mismatch: {k} != {k1}")
  if k // 16 != n1 // 8:
    raise ValueError(f"K/N mismatch: {k // 16} != {n1 // 8}")

  if fa.make_mma_m16n8k16_layout(warp_layout, fa.MMAOperand.LHS) != a.layout:
    raise ValueError("Bad layout for A")
  if fa.make_mma_m16n8k16_layout(warp_layout, fa.MMAOperand.RHS) != b.layout:
    raise ValueError("Bad layout for B")
  if fa.make_mma_m16n8k16_layout(warp_layout, fa.MMAOperand.ACC) != acc.layout:
    raise ValueError("Bad layout for acc")

  if (
      m_tile % 16 != 0
      or n_tile % 8 != 0
      or k_tile % 16 != 0
  ):
    raise ValueError(
        "Invalid tile shapes for MMA"
        f" {a.registers.shape=}, {b.registers.shape=}, {acc.registers.shape=}"
    )


  for i in range(ms):
    for j in range(ns):
      def arr_slice(ti, tj):
        return (slice(i * ti, (i + 1) * ti), slice(j * tj, (j + 1) * tj))
      acc_slice = arr_slice(m_tile, n_tile)
      a_slice = a[arr_slice(m_tile, k_tile)]
      b_slice = b[arr_slice(n_tile, k_tile)]
      acc[acc_slice] = mma_m16(acc[acc_slice], a_slice, b_slice)

  return acc
