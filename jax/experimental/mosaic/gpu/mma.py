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

import itertools
from jax.experimental.mosaic.gpu import fragmented_array as fa
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import vector
import numpy as np
from . import utils


class MMALayouts:
  """Container for MMA layouts, providing a convenient way to create
  layouts for MMA operands based on warp configuration.
  """

  lhs = fa.TiledLayout(
      fa.Tiling(((64, 16), (16, 8), (8, 8), (2,))),
      warp_dims=(-7,),
      lane_dims=(-3, -2),
      vector_dim=-1,
  )
  rhs = fa.TiledLayout(
      fa.Tiling(((8, 16), (8, 8), (2,))),
      warp_dims=(fa.Replicated(4),),
      lane_dims=(-3, -2),
      vector_dim=-1,
  )
  acc = fa.TiledLayout(
      fa.Tiling(((64, 8), (16, 8), (8, 8), (2,))),
      warp_dims=(-7,),
      lane_dims=(-3, -2),
      vector_dim=-1,
  )


def _mma_single_tile(
    acc: fa.FragmentedArray, a: fa.FragmentedArray, b: fa.FragmentedArray
) -> fa.FragmentedArray:
  """Performs `acc + a @ b.T` using warp level MMA instructions."""

  # Muliply by 4 because the fragmtned array has a tile per warp.
  assert a.shape == (64, 16)
  assert b.shape == (8, 16)
  assert acc.shape == (64, 8)
  assert a.mlir_dtype == b.mlir_dtype
  assert a.mlir_dtype in (ir.F16Type.get(), ir.BF16Type.get())
  assert acc.mlir_dtype == ir.F32Type.get()
  assert (
      isinstance(acc.layout, fa.TiledLayout)
      and isinstance(a.layout, fa.TiledLayout)
      and isinstance(b.layout, fa.TiledLayout)
  )
  num_acc_regs, num_a_regs, num_b_regs = 4, 4, 2

  acc_regs = [  # pylint: disable=g-complex-comprehension
      vector.extract(
          reg,
          dynamic_position=[],
          static_position=ir.DenseI64ArrayAttr.get([pos]),
      )
      for reg in acc.registers.flatten()
      for pos in range(acc.layout.vector_length)
  ]
  i32 = ir.IntegerType.get_signless(32)
  a_regs = [utils.bitcast(r, i32) for r in a.registers.flatten()]
  b_regs = [utils.bitcast(r, i32) for r in b.registers.flatten()]

  # Make sure we have the right number of registers for the instruction.
  assert len(a_regs) == 4
  assert len(acc_regs) == 4
  assert len(b_regs) == 2

  instr = f"mma.sync.aligned.m16n8k16.row.col.f32.{a.mlir_dtype}.{b.mlir_dtype}.f32"
  counter = itertools.count()
  n_regs_str = lambda n: (
      "{" + ",".join([f"${next(counter)}" for _ in range(n)]) + "}"
  )
  out_regs_str = n_regs_str(num_acc_regs)
  a_regs_str = n_regs_str(num_a_regs)
  b_regs_str = n_regs_str(num_b_regs)
  c_regs_str = n_regs_str(num_acc_regs)
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
      has_side_effects=False,
  )
  out_regs = [
      llvm.extractvalue(acc.mlir_dtype, out_regs_struct, [i])
      for i in range(len(acc_regs))
  ]
  vec_regs = []
  vec_undef = llvm.mlir_undef(ir.VectorType.get((2,), acc.mlir_dtype))
  for first, second in zip(out_regs[::2], out_regs[1::2]):
    vec = llvm.insertelement(vec_undef, first, position=utils.c(0, i32))
    vec = llvm.insertelement(vec, second, position=utils.c(1, i32))
    vec_regs.append(vec)
  out_regs = np.asarray(vec_regs, dtype=object).reshape(acc.registers.shape)
  return fa.FragmentedArray(
      _registers=out_regs, _layout=acc.layout, _is_signed=None
  )


# TODO(cperivol): More datatypes other than (b)f16.
def mma(
    acc: fa.FragmentedArray,
    a: fa.FragmentedArray,
    b: fa.FragmentedArray,
) -> fa.FragmentedArray:
  """Computes `acc + a @ b.T` using synchronouse MMA instructions.

  All operands must have `TiledLayout`s. The layouts must be generated
  by the `MMALayouts` class, which ensures that the tiles are mapped
  to the warps correctly.

  Args:
    acc: A `FragmentedArray` with a `TiledLayout` generated from
      `MMALayouts.acc`.
    a: A `FragmentedArray` with a `TiledLayout`  generated from
      `MMALayouts.lhs`.
    b: A `FragmentedArray` with a `TiledLayout` generated from `MMALayouts.rhs`.

  Returns:
    A new `FragmentedArray` with the result of the computation with
      the same type as `acc`.
  """

  (m, k) = a.shape
  (n, k2) = b.shape
  (m2, n2) = acc.shape

  if m != m2:
    raise ValueError(f"M mismatch: {m} != {m2}")
  if n != n2:
    raise ValueError(f"N mismatch: {n} != {n2}")
  if k != k2:
    raise ValueError(f"K mismatch: {k} != {k2}")

  # todo(cperivol): A tile shape can have dimensions that are higher
  # multiples of the mma op size as long as those dimensions are not
  # sharded across warps.
  bf16 = ir.BF16Type.get()
  f16 = ir.F16Type.get()
  if a.mlir_dtype != b.mlir_dtype:
    raise ValueError(f"Dtype mismatch: {a.mlir_dtype} != {b.mlir_dtype}")
  if a.mlir_dtype not in (bf16, f16):
    raise NotImplementedError("Only bf16 and f16 supported for the operands.")
  if acc.mlir_dtype != ir.F32Type.get():
    raise NotImplementedError("Only f32 accumulator supported.")

  if MMALayouts.lhs != a.layout:
    raise ValueError("Expected MMALayouts.lhs layout for A")
  if MMALayouts.rhs != b.layout:
    raise ValueError("Expected MMALayouts.rhs layout for B")
  if MMALayouts.acc != acc.layout:
    raise ValueError("Expected MMALayouts.acc layout for acc")

  assert isinstance(a.layout, fa.TiledLayout)
  assert isinstance(b.layout, fa.TiledLayout)
  assert isinstance(acc.layout, fa.TiledLayout)
  m_tile, k_tile = a.layout.base_tile_shape
  n_tile, k_tile2 = b.layout.base_tile_shape
  m_tile2, n_tile2 = acc.layout.base_tile_shape

  assert k_tile == k_tile2
  assert m_tile2 == m_tile
  assert n_tile2 == n_tile

  num_m_tiles, num_n_tiles, num_k_tiles = m // m_tile, n // n_tile, k // k_tile
  if m != m2:
    raise ValueError(f"M mismatch: {m} != {m2}")
  if n != n2:
    raise ValueError(f"N mismatch: {n} != {n2}")
  if k != k2:
    raise ValueError(f"K mismatch: {k} != {k2}")

  assert m_tile == 64 and n_tile == 8 and k_tile == 16, (
      f"Tile shape {m_tile}, {n_tile}, {k_tile} not supported."
  )

  # Do not modify the accumualtor itself.
  acc = acc.copy()
  s = lambda idx, length: slice(idx * length, (idx + 1) * length)
  for k_idx in range(num_k_tiles):
    for m_idx in range(num_m_tiles):
      for n_idx in range(num_n_tiles):
        ms = s(m_idx, m_tile)
        ns = s(n_idx, n_tile)
        ks = s(k_idx, k_tile)
        acc[ms, ns] = _mma_single_tile(acc[ms, ns], a[ms, ks], b[ns, ks])

  return acc
