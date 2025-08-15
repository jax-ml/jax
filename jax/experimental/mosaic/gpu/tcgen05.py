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

from collections.abc import Sequence
import dataclasses
import math
from typing import Any, cast, Callable

import itertools
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import memref
import numpy as np

from . import utils
from . import fragmented_array as fa
from . import mma_utils
from .launch_context import LaunchContext


TMEM_ROWS = 128
TMEM_MAX_COLS = 512
TCGEN05_SMEM_DESCRIPTOR_BIT = 1 << 46
LAYOUT = fa.TCGEN05_LAYOUT
TRANSPOSED_LAYOUT = fa.TCGEN05_TRANSPOSED_LAYOUT
ROW_LAYOUT = fa.TCGEN05_ROW_LAYOUT
COL_LAYOUT = fa.TCGEN05_COL_LAYOUT
TMEM_NATIVE_LAYOUT = fa.TMEM_NATIVE_LAYOUT


def create_instr_descriptor(
    m: int,
    n: int,
    acc_dtype,
    input_dtype,
    transpose_a: bool = False,
    transpose_b: bool = False,
    sparsity_selector: int | None = None,
):
  f16 = ir.F16Type.get()
  f32 = ir.F32Type.get()
  i32 = ir.IntegerType.get_signless(32)

  desc = 0
  if sparsity_selector is not None:
    assert 0 <= sparsity_selector < 3
    desc |= sparsity_selector
    desc |= 1 << 2  # Enable sparsity
  if acc_dtype == f16:
    d_type_val = 0
  elif acc_dtype == f32:
    d_type_val = 1
  elif acc_dtype == i32:
    d_type_val = 2
  else:
    raise NotImplementedError(f"Unsupported accumulator dtype: {acc_dtype}")
  desc |= (d_type_val << 4)  # D type, bits 4-5
  # Bit 6 is reserved
  if input_dtype == f16:
    assert acc_dtype in {f16, f32}
    ab_type_val = 0
  elif input_dtype == ir.BF16Type.get():
    assert acc_dtype == f32
    ab_type_val = 1
  elif input_dtype == ir.Float8E4M3FNType.get():
    assert acc_dtype in {f16, f32}
    ab_type_val = 0
  elif input_dtype == ir.Float8E5M2Type.get():
    assert acc_dtype in {f16, f32}
    ab_type_val = 1
  elif input_dtype == ir.IntegerType.get_signless(8):  # Only s8 for now.
    assert acc_dtype == i32
    ab_type_val = 1
  else:
    raise NotImplementedError(f"Unsupported input dtype: {input_dtype}")
  desc |= (ab_type_val << 7)   # A dtype, bits 7-9
  desc |= (ab_type_val << 10)  # B dtype, bits 10-12
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


def _create_scaled_instr_descriptor(
    get_input_encoding: Callable[[ir.Type], int],
    m: int,
    n: int,
    a_type: ir.Type,
    b_type: ir.Type,
    a_scale_idx: int,
    b_scale_idx: int,
    transpose_a: bool = False,
    transpose_b: bool = False,
):
  desc = 0
  # Bits 0, 1 are reserved
  # We ignore sparsity (bit 2)
  # Bit 3 is reserved
  assert 0 <= b_scale_idx < 4
  desc |= b_scale_idx << 4  # B scale factor data ID, bits 4-5
  # Bit 6 is reserved
  desc |= get_input_encoding(a_type) << 7  # A dtype, bits 7-9
  desc |= get_input_encoding(b_type) << 10  # B dtype, bits 10-12
  # We ignore negate bits 13-14
  desc |= transpose_a << 15  # Transpose A
  desc |= transpose_b << 16  # Transpose B
  if n % 8 or n > 256:
    raise ValueError(f"N must be a multiple of 8 and <= 256, got: {n}")
  desc |= (n >> 3) << 17  # N, bits 17-22
  desc |= 1 << 23  # Scale matrix type
  # Bits 24-26 are reserved
  if m % 128 or m > 256:
    raise ValueError(f"M must be a multiple of 16 and <= 256, got: {m}")
  desc |= (m >> 7) << 27  # M >> 7, bits 27-28
  desc |= a_scale_idx << 29  # A scale factor data ID, bits 29-30
  # Bit 31 is reserved
  return arith.constant(ir.IntegerType.get_signless(32), desc)


def create_scaled_f8f6f4_instr_descriptor(*args, **kwargs):
  def get_input_encoding(ty):
    if ty == ir.Float8E4M3FNType.get():
      return 0
    elif ty == ir.Float8E5M2Type.get():
      return 1
    else:
      raise NotImplementedError(f"Unsupported input dtype: {ty}")
  return _create_scaled_instr_descriptor(get_input_encoding, *args, **kwargs)


def create_scaled_f4_instr_descriptor(*args, **kwargs):
  def get_input_encoding(ty):
    if ty == ir.Float4E2M1FNType.get():
      return 1
    else:
      raise NotImplementedError(f"Unsupported input dtype: {ty}")
  return _create_scaled_instr_descriptor(get_input_encoding, *args, **kwargs)


def mma(
    d: TMEMRef,
    a: ir.Value | TMEMRef,
    b: ir.Value,
    *,
    a_swizzle: int = 128,
    b_swizzle: int = 128,
    a_scale: TMEMRef | None = None,
    b_scale: TMEMRef | None = None,
    a_sparse_metadata: TMEMRef | None = None,
    accumulate: ir.Value | bool = True,
    collective: bool = False,
) -> None:
  if a_swizzle == 16 or b_swizzle == 16:
    raise NotImplementedError("No swizzle is not supported")
  i32 = ir.IntegerType.get_signless(32)
  i64 = ir.IntegerType.get_signless(64)
  if isinstance(accumulate, bool):
    accumulate = arith.constant(ir.IntegerType.get_signless(1), accumulate)
  num_cta = 2 if collective else 1
  if (is_scaled := a_scale is not None) != (b_scale is not None):
    raise ValueError("Either none or both scales should be provided")
  is_sparse = a_sparse_metadata is not None
  if is_scaled and is_sparse:
    raise NotImplementedError("Block-scaled sparse matmuls unsupported")

  # Step 1. Establish the shape and element type of the operation.
  if not ir.MemRefType.isinstance(b.type):
    raise ValueError(f"B must be a memref, got: {b.type}")
  (k, n), element_type = mma_utils.tiled_memref_shape(b)
  if isinstance(a, TMEMRef):
    m, k2 = a.shape
    element_type2 = a.dtype
    if is_scaled or is_sparse:
      raise NotImplementedError("A in TMEM unsupported for block-scaled and sparse matmuls")
    if m != 128:
      raise NotImplementedError(f"Only M=128 is supported for MMA with A in TMEM, but got M={m}")
    # Watch out: this layout must be consistent with D's layout (up to packing).
    expected_packing = 32 // utils.bitwidth(element_type)
    expected_layout = _infer_tmem_layout(
        a.shape, collective, packing=expected_packing
    )
    if a.layout != expected_layout:
      raise ValueError(
          f"A layout mismatch: expected {expected_layout}, got {a.layout}"
      )
  else:
    if not ir.MemRefType.isinstance(a.type):
      raise ValueError(f"A must be a memref, got {a.type}")
    (m, k2), element_type2 = mma_utils.tiled_memref_shape(a)
  if is_sparse:
    k2 *= 2
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
  if m == 128:
    if d.layout != (expected_d_layout := tmem_default_layout(packing=1)):
      raise ValueError(
          f"Accumulator layout mismatch: expected {expected_d_layout}, got {d.layout}"
      )
    n_lane_groups = 1
  elif m == 64:
    if is_scaled:
      raise NotImplementedError("MMA with block scaling is not supported for M=64")
    if is_sparse:
      raise NotImplementedError("Sparse MMA not supported for M=64")
    # Watch out: this layout must be consistent with A's layout (up to packing).
    # 2CTA M=128 instruction uses a different TMEM layout than 1CTA M=64.
    expected_d_layout = _infer_tmem_layout(d.shape, collective, packing=1)
    if d.layout != expected_d_layout:
      raise ValueError(
          f"Accumulator layout mismatch: expected {expected_d_layout}, got {d.layout}"
      )
    if collective:
      n_lane_groups = 1
    else:
      n_lane_groups = 2
      # We can't split N into groups if we would partition it below the tile size.
      # TODO: We only need to check this if N is the minormost dim in B.
      if 8 * b_swizzle // utils.bitwidth(element_type) > n // n_lane_groups:
        raise ValueError("Swizzle is too big for MMA with M=64. Try lowering it.")
  else:
    raise ValueError(f"Only M=128 and M=64 are supported for MMA, but got M={m}")
  f32 = ir.F32Type.get()
  f16 = ir.F16Type.get()
  s32 = ir.IntegerType.get_signless(32)
  if element_type == f32 or element_type == ir.BF16Type.get():
    if element_type == f32 and is_sparse:
      raise NotImplementedError("Only 16-bit types supported for sparse MMA")
    if is_scaled:
      raise ValueError(
          f"MMA with element type {element_type} does not support block scaling"
      )
    if d.dtype != f32:
      raise ValueError(
          f"MMA with element type {element_type} only supports accumulators"
          f" of type f32, but got: {d.dtype}"
      )
  elif element_type == f16:
    if is_scaled:
      raise ValueError(
          f"MMA with element type {element_type} does not support block scaling"
      )
    if d.dtype != f16 and d.dtype != f32:
      raise ValueError(
          f"MMA with element type {element_type} only supports accumulators of"
          f" type f32 or f16, but got: {d.dtype}"
      )
  elif any(
      t.isinstance(element_type)
      for t in {ir.Float8E5M2Type, ir.Float8E4M3FNType}
  ):
    if is_sparse:
      raise NotImplementedError("Only 16-bit types supported for sparse MMA")
    if d.dtype != f16 and d.dtype != f32:
      raise ValueError(
          f"MMA with element type {element_type} only supports accumulators of"
          f" type f32 or f16, but got: {d.dtype}"
      )
    if is_scaled and d.dtype != f32:
      raise ValueError(
          f"Block-scaled MMA with element type {element_type} only supports f32"
          f" accumulators, but got: {d.dtype}"
      )
  elif any(
      t.isinstance(element_type) for t in {ir.Float4E2M1FNType}
  ):
    if is_sparse:
      raise NotImplementedError("Only 16-bit types supported for sparse MMA")
    if not is_scaled:
      raise ValueError(
          f"MMA with element type {element_type} only supports block scaling"
      )
    if d.dtype != f32:
      raise ValueError(
          f"Block-scaled MMA with element type {element_type} only supports f32"
          f" accumulators, but got: {d.dtype}"
      )
  elif element_type == ir.IntegerType.get_signless(8):
    if is_sparse:
      raise NotImplementedError("Only 16-bit types supported for sparse MMA")
    if is_scaled:
      raise ValueError(
          f"MMA with element type {element_type} does not support block scaling"
      )
    if d.dtype != s32:
      raise ValueError(
          "MMA with element type s8 only supports s32 accumulators, but got:"
          f" {d.dtype}"
      )
  else:
    raise NotImplementedError(f"Unsupported element type: {element_type}")

  # Step 2. Decide on the instruction shapes we'll use. Note that with swizzles,
  # instructions must be issued in groups of the same width as the swizzle.
  m_group_elems = m  # We have already verified M is supported above.
  k_group_elems = 8 * max(a_swizzle * (1 + is_sparse), b_swizzle) // utils.bitwidth(element_type)
  if is_sparse and k_group_elems < 64:
    # This is a limitation of the implementation below. We could relax it if we
    # ever need to support k=32.
    k_group_elems = 64
  if n % 8:
    raise ValueError(f"N must be a multiple of 8, got: {n}")
  if n.bit_count() != 1:
    raise ValueError(f"N must be a power of 2, got: {n}")
  # TODO: We could relax those constraints if we have multiple n_lane_groups,
  # since we will be unrolling the instructions anyway.
  if collective and n > 128:
    raise ValueError("Only N <= 128 are supported for collective MMA")
  elif n > 512:
    raise ValueError("Only N <= 512 are supported for MMA")
  n_group_elems = min(n // n_lane_groups, 256 // num_cta)
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

  # Check that the shapes and element types are correct for block scaling.
  if is_scaled:
    if collective:
      raise NotImplementedError("MMA with block scaling does not support collective")
    assert m == 128  # Checked above.
    if n % 32:
      raise ValueError(
          f"MMA with block scaling requires N to be divisible by 32, got: {n}"
      )
    if k_group_elems != 128 or a_swizzle != b_swizzle:
      assert utils.bitwidth(element_type) <= 8
      expected_swizzle = 128 // (8 // utils.bitwidth(element_type))
      raise NotImplementedError(
          "MMA with block scaling requires swizzle to be"
          f" {expected_swizzle} for dtype {element_type}, got:"
          f" {a_swizzle=} and {b_swizzle=}"
      )
    assert a_scale is not None and b_scale is not None
    if a_scale.shape != (m, 4):
      raise ValueError(
          f"A scale shape mismatch: expected ({m}, 4), got {a_scale.shape}"
      )
    if a_scale.dtype != ir.Float8E8M0FNUType.get():
      raise ValueError(
          f"A scale dtype mismatch: expected f8e8m0fnu, got {a_scale.dtype}"
      )
    if b_scale.shape != (n, 4):
      raise ValueError(
          f"B scale shape mismatch: expected ({n}, 4), got {b_scale.shape}"
      )
    if b_scale.dtype != ir.Float8E8M0FNUType.get():
      raise ValueError(
          f"B scale dtype mismatch: expected f8e8m0fnu, got {b_scale.dtype}"
      )
  if is_sparse:
    a_sparse_metadata = cast(TMEMRef, a_sparse_metadata)
    if collective:
      raise NotImplementedError("Collective sparse MMA unsupported")
    if n % 32:
      raise ValueError(f"Sparse MMA requires N to be divisible by 32, got: {n}")
    if a_sparse_metadata.shape != (m, k // 2):
      raise ValueError(
          f"A sparse metadata shape mismatch: expected {(m, k // 2)}, got"
          f" {a_sparse_metadata.shape}"
      )
    if a_sparse_metadata.dtype != ir.IntegerType.get_signless(2):
      raise ValueError(
          "A sparse metadata dtype mismatch: expected i2, got"
          f" {a_sparse_metadata.dtype}"
      )

  # Step 3. Compute the operand descriptors.
  if not isinstance(a, TMEMRef):
    # Both dense and sparse matmul consume A with a K bytewidth of 32, only
    # the group size is halved when it's sparse.
    (
        (a_desc_base, a_k_instr_strides),
        (a_m_group_stride, a_k_group_stride),
        a_fastest,
    ) = mma_utils.create_descriptor(
        a,
        swizzle=a_swizzle,
        group_size=(m_group_elems, k_group_elems // (1 + is_sparse)),
        logical_k_major=False,
        mma_bytewidth_k=32,
    )
  else:
    a_fastest = mma_utils.Dim.K
    a_k_instr_strides = None
    a_m_group_stride = a_k_group_stride = a_desc_base = None
  (
      (b_desc_base, b_k_instr_strides),
      (b_n_group_stride, b_k_group_stride),
      b_fastest,
  ) = mma_utils.create_descriptor(
      b,
      swizzle=b_swizzle,
      group_size=(k_group_elems, n_group_elems),
      logical_k_major=True,
      mma_bytewidth_k=64 if is_sparse else 32,
  )

  if is_scaled and utils.bitwidth(mma_element_type) == 4:
    if a_fastest != mma_utils.Dim.K:
      raise ValueError(
          "4-bit block scaled MMA only supports K-fastest operands, but A is M-fastest"
      )
    if b_fastest != mma_utils.Dim.K:
      raise ValueError(
          "4-bit block scaled MMA only supports K-fastest operands, but B is N-fastest"
      )
  if is_sparse:
    if b_swizzle == 32 and b_fastest == mma_utils.Dim.K:
      raise NotImplementedError(
          "B tiling too small. Increase swizzle or transpose the input."
      )

  # Step 4. Issue the instructions.
  true = arith.constant(ir.IntegerType.get_signless(1), 1)
  n_collective_group_elems = n_group_elems * num_cta
  n_col_groups = n_groups // n_lane_groups
  assert d.layout.base_tile_shape[0] % 4 == 0
  lanes_per_n_group = d.layout.base_tile_shape[0] // 4
  a_sparse_addr_base = a_sparse_metadata.address if is_sparse else None  # type: ignore
  for mi, ni, ki in np.ndindex(m_groups, n_groups, k_groups):
    if isinstance(a, TMEMRef):
      if m_groups != 1:
        raise NotImplementedError("A address calculation for multiple M tiles")
      a_mk = a.slice(slice(None), utils.ds(ki * k_group_elems, k_group_elems)).address
    else:
      a_offset = mi * a_m_group_stride + ki * a_k_group_stride
      a_mk = arith.addi(a_desc_base, utils.c(mma_utils.encode_addr(a_offset), i64))
    b_offset = ni * b_n_group_stride + ki * b_k_group_stride
    b_nk = arith.addi(b_desc_base, utils.c(mma_utils.encode_addr(b_offset), i64))
    if a_sparse_addr_base is not None:
      if n_groups != 1 or m_groups != 1:
        raise NotImplementedError("A sparse metadata address calculation for multiple tiles")
      assert k_group_elems % 32 == 0
      cols_per_k_group = k_group_elems // 32
      a_sparse_addr = arith.addi(a_sparse_addr_base, utils.c(ki * cols_per_k_group, i32))
    else:
      a_sparse_addr = None
    if is_scaled and (m_groups != 1 or n_groups != 1 or k_groups != 1):
      raise NotImplementedError("Block-scaled metadata address calculation for multiple tiles")
    acc = accumulate if ki == 0 else true
    ni_lane_group, ni_col = ni // n_col_groups, ni % n_col_groups
    d_offset = (
        ((ni_lane_group * lanes_per_n_group) << 16)
        + ni_col * n_collective_group_elems
    )
    if m_groups != 1:
      raise NotImplementedError("D address calculation for multiple M tiles")
    _do_mma(
        arith.addi(d.address, arith.constant(i32, d_offset)),
        a_mk,
        b_nk,
        d_type=d.dtype,
        m=m_group_elems,
        n=n_group_elems,
        k=k_group_elems,
        collective=collective,
        a_transpose=a_fastest != mma_utils.Dim.K,
        b_transpose=b_fastest != mma_utils.Dim.K,
        a_k_strides=a_k_instr_strides,
        b_k_strides=b_k_instr_strides,
        a_scale_addr=a_scale.address if a_scale is not None else None,
        b_scale_addr=b_scale.address if b_scale is not None else None,
        a_sparse_addr=a_sparse_addr,
        accumulate=acc,
        element_type=mma_element_type,
    )


def _do_mma(
    d_addr: ir.Value,
    a_desc_or_addr: ir.Value,  # TMEM address if a_k_stride is None
    b_desc: ir.Value,
    a_transpose: bool,
    b_transpose: bool,
    a_k_strides: tuple[tuple[int, ...], tuple[int, ...]] | None,
    b_k_strides: tuple[tuple[int, ...], tuple[int, ...]],
    a_scale_addr: ir.Value | None,
    b_scale_addr: ir.Value | None,
    a_sparse_addr: ir.Value | None,
    m: int,
    n: int,
    k: int,
    element_type: ir.Type,
    d_type: ir.Type,
    accumulate: ir.Value,
    collective: bool,
):
  i1 = ir.IntegerType.get_signless(1)
  i32 = ir.IntegerType.get_signless(32)
  i64 = ir.IntegerType.get_signless(64)
  a_k_idx_tiling, a_k_strides = a_k_strides or (None, None)
  b_k_idx_tiling, b_k_strides = b_k_strides
  assert all(s % 16 == 0 for s in itertools.chain(a_k_strides or (), b_k_strides))
  assert (a_scale_addr is None) == (b_scale_addr is None)
  is_scaled = a_scale_addr is not None
  is_sparse = a_sparse_addr is not None
  elem_bitwidth = utils.bitwidth(element_type)
  instr_k = (1 + is_sparse) * 8 * 32 // elem_bitwidth
  packing = 8 * 4 // elem_bitwidth
  assert not is_sparse or elem_bitwidth == 16  # Only 16-bit supported for now.

  extra_args: Sequence[object]
  scale_steps = None
  if is_scaled:
    if (ir.Float8E5M2Type.isinstance(element_type) or
        ir.Float8E4M3FNType.isinstance(element_type)):
      kind = "mxf8f6f4.block_scale.scale_vec::1X"
      scale_steps = 4
      create_scaled_instr_descriptor = create_scaled_f8f6f4_instr_descriptor
    elif ir.Float4E2M1FNType.isinstance(element_type):
      assert not a_transpose and not b_transpose
      kind = "mxf4.block_scale.scale_vec::2X"
      scale_steps = 2
      create_scaled_instr_descriptor = create_scaled_f4_instr_descriptor
    else:
      raise NotImplementedError(f"Unsupported element type for block scaling: {element_type}")
    extra_args = (a_scale_addr, b_scale_addr)
    extra_ptx = "[$5], [$6], "
    extra_constraints = ",r,r"
  else:
    if ir.F16Type.isinstance(element_type) or ir.BF16Type.isinstance(element_type):
      kind = "f16"
    elif ir.Float8E5M2Type.isinstance(element_type):
      kind = "f8f6f4"
    elif ir.Float8E4M3FNType.isinstance(element_type):
      kind = "f8f6f4"
    elif ir.IntegerType.get_signless(8).isinstance(element_type):
      kind = "i8"
    else:
      raise NotImplementedError(f"Unsupported input element type: {element_type}")
    extra_args = ()
    extra_constraints = extra_ptx = ""

    def create_scaled_instr_descriptor(*args):
      raise NotImplementedError

  num_cta = 2 if collective else 1
  a_in_tmem = a_k_strides is None
  a_ptx = "[$1]" if a_in_tmem else "$1"
  a_ptx_constraint = "r" if a_in_tmem else "l"
  sparse_mod = ".sp" if is_sparse else ""
  sparse_meta_ptx = "[$5], " if is_sparse else ""
  extra_constraints += ",r" if is_sparse else ""
  sparse_addr: tuple[Any, ...] = ()
  assert a_desc_or_addr.type == ir.IntegerType.get_signless(32 if a_in_tmem else 64)
  assert scale_steps is None or scale_steps == k // instr_k
  def _get_offset(idx: int, idx_tiling: tuple[int, ...], strides: tuple[int, ...]):
    assert len(idx_tiling) + 1 == len(strides)
    idxs = []
    for t in idx_tiling:
      idxs.append(idx // t)
      idx = idx % t
    idxs.append(idx)
    offset = sum(i * s for i, s in zip(idxs, strides, strict=True))
    return arith.constant(i64, offset >> 4)
  for k_step in range(k // instr_k):
    if is_scaled:
      assert scale_steps is not None
      scale_vec_width = 4 // scale_steps
      scale_id = k_step * scale_vec_width
      i_desc = create_scaled_instr_descriptor(
          m, n, element_type, element_type, scale_id, scale_id, a_transpose, b_transpose
      )
    else:
      sp_selector = None
      if is_sparse:
        assert (k // instr_k) % 2 == 0
        sp_selector = k_step % 2
        selector_width = 64
        k_steps_for_col_inc = selector_width // instr_k
        # If the K group is large, we need to increment the sparse metadata.
        # TODO(apaszke): At this point the purpose of this function is becoming
        # less clear, since we end up replicating address arithmetic that's
        # already there in the caller. We should unify them into a single loop.
        sparse_addr = (
            arith.addi(
                a_sparse_addr, utils.c(k_step // k_steps_for_col_inc * 2, i32)
            ),
        )
      i_desc = create_instr_descriptor(
          m * num_cta, n * num_cta, d_type, element_type, a_transpose, b_transpose, sparsity_selector=sp_selector
      )
    if a_in_tmem:
      a_desc_or_addr_instr = arith.addi(
          a_desc_or_addr, arith.constant(i32, k_step * instr_k // packing)
      )
    else:
      assert a_k_idx_tiling is not None and a_k_strides is not None
      a_desc_or_addr_instr = arith.addi(
          a_desc_or_addr, _get_offset(k_step, a_k_idx_tiling, a_k_strides)
      )
    b_desc_instr = arith.addi(b_desc, _get_offset(k_step, b_k_idx_tiling, b_k_strides))
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [d_addr, a_desc_or_addr_instr, b_desc_instr, i_desc, accumulate, *extra_args, *sparse_addr],
        f"tcgen05.mma{sparse_mod}.cta_group::{num_cta}.kind::{kind} [$0], {a_ptx}, $2, {sparse_meta_ptx}$3, {extra_ptx}$4;",
        f"r,{a_ptx_constraint},l,r,b" + extra_constraints,
        has_side_effects=True,
    )
    accumulate = arith.constant(i1, 1)


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
      ir.Type.parse("!llvm.void"), [barrier], ptx, "r", has_side_effects=True
  )


def tmem_alloc_exact_ncols(ncols: int, exact: bool) -> int:
  """Returns the exact number of columns to allocate in TMEM.

  The number of columns is rounded up to the nearest power of 2.

  Args:
    ncols: The number of columns to allocate.
    exact: If true, throws an error if the number of columns is not a power of 2
      and within [32, 512].
  """
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


def tmem_alloc(tmem_addr: ir.Value, ncols: int, collective: bool = False, exact: bool = True) -> tuple[ir.Value, int]:
  if ir.MemRefType.isinstance(tmem_addr.type):
    ref_ty = ir.MemRefType(tmem_addr.type)
    if ref_ty.element_type != ir.IntegerType.get_signless(32):
      raise ValueError(f"tmem_addr must be an i32 memref, got: {ref_ty}")
    if not utils.is_smem_ref(ref_ty):
      raise ValueError(f"tmem_addr must be in shared memory, got: {ref_ty}")
    if math.prod(ref_ty.shape) != 1:
      raise ValueError(f"tmem_addr must contain a single element, got: {ref_ty}")
    tmem_addr = utils.memref_ptr(tmem_addr, memory_space=3)
  elif tmem_addr.type != ir.Type.parse("!llvm.ptr<3>"):
    raise ValueError(f"tmem_addr must be an SMEM pointer or a memref, got: {tmem_addr.type}")
  ncols = tmem_alloc_exact_ncols(ncols, exact)
  num_cta = 2 if collective else 1
  return llvm.inline_asm(
      ir.Type.parse("!llvm.void"),
      [tmem_addr],
      f"tcgen05.alloc.cta_group::{num_cta}.sync.aligned.shared::cta.b32  [$0], {ncols};",
      "r",
      has_side_effects=True,
  ), ncols


def tmem_dealloc(tmem_addr: ir.Value, ncols: int, collective: bool = False, exact: bool = True):
  if tmem_addr.type != ir.IntegerType.get_signless(32):
    raise ValueError(f"tmem_addr must be an i32, got: {tmem_addr.type}")
  ncols = tmem_alloc_exact_ncols(ncols, exact)
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
class TMEMLayout(fa.TiledLayout):
  """Represents the way a shape is laid out in TMEM.

  The layout describes how the shape is split across the 128 rows (lanes) of
  TMEM. We reinterpret warp_dims as the partitioning of TMEM into 4 banks, each
  accessible from a single warp. The 32 lanes inside each bank are assigned
  consecutive elements from lane_dims. The data within each lane is linearized
  in row-major order, with each vector padded up to 32 bits (wider vectors are
  unsupported).
  """

  def check_type(self, shape: tuple[int, ...], bitwidth: int):
    if len(shape) != 2:
      raise ValueError(f"TMEM can only represent 2D shapes, got {shape}")
    if any(s % t for s, t in zip(shape, self.base_tile_shape)):
      raise ValueError(
          f"{shape} is not divisible into tiles of shape {self.base_tile_shape}"
      )
    if self.vector_length not in {1, fully_packed := 32 // bitwidth}:
      raise ValueError(
          f"For {bitwidth}-bit types, the vector length must be 1 or"
          f" {fully_packed} , but got: {self.vector_length}"
      )

  def cols_in_shape(self, shape: tuple[int, int], bitwidth: int):
    self.check_type(shape, bitwidth)
    return math.prod(shape) // TMEM_ROWS // self.vector_length

  def canonicalize(self) -> "TMEMLayout":
    layout = super().canonicalize()
    return TMEMLayout(
        layout.tiling,
        layout.warp_dims,
        layout.lane_dims,
        layout.vector_dim,
        _check_canonical=False,
    )


def _infer_tmem_load_registers_layout(
    tmem_layout: TMEMLayout, columns: int, packing: int
) -> fa.TiledLayout:
  if tmem_layout == tmem_default_layout(packing=packing):
    return LAYOUT
  if tmem_layout == tmem_half_lane_layout(columns, packing=packing):
    return fa.WGMMA_LAYOUT
  if tmem_layout == tmem_m64_collective_layout(columns, packing=packing):
    return fa_m64_collective_layout(columns)
  raise ValueError(f"TMEM layout {tmem_layout} is not supported")


def _infer_tmem_layout(shape: tuple[int, int], collective: bool, packing: int) -> TMEMLayout:
  if len(shape) != 2:
    raise ValueError(f"TMEM can only represent 2D shapes, got {shape}")
  if packing > 8 or packing.bit_count() != 1:
    raise ValueError(f"Packing must be <= 8 and a power of 2, got: {packing}")
  if shape[1] % packing:
    raise ValueError(f"Minor dimension of shape must be divisible by packing, got: {shape}")
  if shape[0] == TMEM_ROWS:
    return tmem_default_layout(packing)
  elif shape[0] == TMEM_ROWS // 2:
    if collective:
      return tmem_m64_collective_layout(shape[1], packing)
    else:
      return tmem_half_lane_layout(shape[1], packing)
  else:
    raise ValueError(f"Unsupported shape: {shape}")


def tmem_default_layout(packing: int = 1):
  """A TMEM layout used for 1CTA MMA with M=128 and 2CTA MMA with M=256."""
  if packing.bit_count() != 1:
    raise ValueError(f"Packing must be a power of 2, got: {packing}")
  return TMEMLayout(
      fa.Tiling(((TMEM_ROWS, packing), (fa.WARP_SIZE, packing))),
      warp_dims=(-4,),
      lane_dims=(-2,),
      vector_dim=-1,
  )


def tmem_half_lane_layout(columns, packing: int = 1):
  """A TMEM layout used for 1CTA MMA with M=64."""
  if packing > columns or packing.bit_count() != 1:
    raise ValueError(f"Packing must be <= 8 and a power of 2, got: {packing}")
  if columns % 16:
    raise ValueError(f"Columns must be a multiple of 16, got: {columns}")
  return TMEMLayout(
      fa.Tiling((
          (TMEM_ROWS // 2, columns),
          (fa.WARP_SIZE // 2, columns // 2),
          (packing,),
      )),
      warp_dims=(-5,),
      lane_dims=(-4, -3),
      vector_dim=-1,
  )


def tmem_m64_collective_layout(columns, packing: int = 1):
  """A TMEM layout used for 2CTA MMA with M=128."""
  if packing > 8 or packing.bit_count() != 1:
    raise ValueError(f"Packing must be <= 8 and a power of 2, got: {packing}")
  if columns % 16:
    raise ValueError(f"Columns must be a multiple of 16, got: {columns}")
  return TMEMLayout(
      fa.Tiling((
          (TMEM_ROWS // 2, columns),
          (fa.WARP_SIZE, columns // 2),
          (packing,),
      )),
      warp_dims=(-4, -5,),
      lane_dims=(-3,),
      vector_dim=-1,
  )


def fa_m64_collective_layout(columns):
  """The register layout for transfers to/from tmem_m64_collective_layout."""
  if columns % 8:
    raise ValueError(f"Columns must be a multiple of 8, got: {columns}")
  return fa.TiledLayout(
      fa.Tiling((
          (TMEM_ROWS // 2, columns), (fa.WARP_SIZE, columns // 2), (8, 8), (2,)
      )),
      warp_dims=(-6, -7),
      lane_dims=(-3, -2),
      vector_dim=-1,
  )


def scales_layout():
  """A TMEM layout for A and B scales in .scale_vec::1X configuration.

  See https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
  """
  return TMEMLayout(
      fa.Tiling(((TMEM_ROWS, 4), (TMEM_ROWS // 4, 1))),
      warp_dims=(fa.Replicated(times=4),),
      lane_dims=(-2,),
      vector_dim=-3,
  )


def sparse_meta_layout():
  """A TMEM layout for A sparsity metadata.

  See https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m128-256
  """
  # TODO(apaszke): This does not really describe this layout and we can't do it
  # until we add support for multiple vector dims. Still, it's ok to do for now,
  # because we don't use TMEM layouts for any automatic transformations at the
  # moment and only ever compare it for equality.
  return TMEMLayout(
      fa.Tiling(((TMEM_ROWS, 16), (TMEM_ROWS // 4, 1), (16, 1), (8, 1))),
      warp_dims=(-8,),
      lane_dims=(-2, -4, -6),
      vector_dim=-7,
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
    if not utils.is_smem_ref(addr_ref_ty):
      raise ValueError(f"tmem_addr_ref must be in shared memory, got: {addr_ref_ty}")
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
      layout = _infer_tmem_layout(shape, collective, packing=1)
    else:
      layout.check_type(shape, utils.bitwidth(dtype))
    # TODO: Do we have to do this??
    # warp_idx = utils.warp_idx(sync=False)
    # tmem_addr = arith.ori(tmem_addr, arith.shli(warp_idx, utils.c(21, i32)))
    return cls(tmem_addr, shape, dtype, layout)

  def slice(self, *idxs):
    i32 = ir.IntegerType.get_signless(32)
    base_idx, slice_shape, is_squeezed = utils.parse_indices(idxs, self.shape)
    if any(is_squeezed):
      raise ValueError("TMEM can only be sliced, not indexed")
    if base_idx == [0] * len(base_idx) and slice_shape == list(self.shape):
      return self  # Trival slice
    if self.layout == tmem_default_layout(packing=1):
      packing = 1
    elif self.layout == tmem_default_layout(packing=2):
      packing = 2
    else:
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

  def load(self, layout: fa.TiledLayout | None = None, is_signed: bool | None = None):
    if utils.bitwidth(self.dtype) not in {16, 32}:
      raise NotImplementedError(f"Unsupported dtype: {self.dtype}")
    packing = self.layout.vector_length
    if layout is None:
      layout = _infer_tmem_load_registers_layout(
          self.layout, self.shape[1], packing
      )
    regs_shape = layout.registers_shape(self.shape)
    if regs_shape[0] != 1:  # We'll need to issue multiple loads below.
      raise NotImplementedError("Loading multiple row tiles")
    if layout == LAYOUT and self.layout == tmem_default_layout(packing=packing):
      registers = _load_32xcols(
          self.address, self.shape[1], self.dtype, packing
      ).T.reshape(regs_shape)
    elif layout == TMEM_NATIVE_LAYOUT and self.layout == tmem_default_layout(packing=packing):
      registers = _load_32xcols_native(
          self.address, self.shape[1], self.dtype, packing
      ).reshape(regs_shape)
    elif layout == fa.WGMMA_LAYOUT and self.layout == tmem_half_lane_layout(self.shape[1], packing=packing):
      # Load half the columns, since they are folded over lanes.
      raw_registers = _load_32xcols(
          self.address, self.shape[1] // 2, self.dtype, packing
      )
      assert raw_registers.shape[0] == 4
      registers = np.concatenate([raw_registers[:2], raw_registers[2:]], axis=1)
      registers = registers.T.reshape(regs_shape)
    elif layout == fa_m64_collective_layout(self.shape[1]) and self.layout == tmem_m64_collective_layout(self.shape[1], packing=packing):
      regs_shape = layout.registers_shape(self.shape)
      # We take half the columns, because they are split over halves of TMEM.
      registers = _load_32xcols(
          self.address, self.shape[1] // 2, self.dtype, packing
      ).reshape(regs_shape)
    else:
      raise ValueError(
          f"Loads from TMEM layout {self.layout} to register layout"
          f" {layout} are not supported"
      )
    return fa.FragmentedArray(
        _registers=registers, _layout=layout, _is_signed=is_signed
    )

  def store(self, value):
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
    packing = self.layout.vector_length
    if value.layout == LAYOUT and self.layout == tmem_default_layout(packing=packing):
      _store_32xcols(
          self.address, value.registers.T.reshape((4, -1)), packing
      )
    elif value.layout == TMEM_NATIVE_LAYOUT and self.layout == tmem_default_layout(packing=packing):
      _store_32xcols_native(
          self.address, value.registers.reshape(-1), packing
      )
    elif value.layout == fa.WGMMA_LAYOUT and self.layout == tmem_half_lane_layout(self.shape[1], packing=packing):
      registers = value.registers.T.reshape(2, -1)
      registers = np.concatenate(np.split(registers, 2, axis=1), axis=0)
      _store_32xcols(self.address, registers, packing)
    elif value.layout == fa_m64_collective_layout(self.shape[1]) and self.layout == tmem_m64_collective_layout(self.shape[1], packing=packing):
      _store_32xcols(self.address, value.registers.reshape(4, -1), packing)
    else:
      raise ValueError(
          f"Storing from register layout {value.layout} to TMEM layout"
          f" {self.layout} is not supported"
      )

  def _debug_print(self):
    i32 = ir.IntegerType.get_signless(32)
    num_cols = self.layout.cols_in_shape(self.shape, utils.bitwidth(self.dtype))
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
      if self.layout.vector_length == 1:
        if dtype_bitwidth < 32:
          val = arith.trunci(ir.IntegerType.get_signless(dtype_bitwidth), val)
        val = utils.bitcast(val, self.dtype)
      elif self.layout.vector_length == full_packing:
        val = utils.bitcast(val, ir.VectorType.get((full_packing,), self.dtype))
      else:
        raise NotImplementedError(
            f"Unsupported packing: {self.layout.vector_length}"
        )
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


def commit_tmem():
  void = ir.Type.parse("!llvm.void")
  llvm.inline_asm(
      void, [], "tcgen05.wait::st.sync.aligned;", "", has_side_effects=True,
  )
  utils.warpgroup_barrier()


def wait_load_tmem():
  void = ir.Type.parse("!llvm.void")
  llvm.inline_asm(
      void, [], "tcgen05.wait::ld.sync.aligned;", "", has_side_effects=True,
  )
  utils.warpgroup_barrier()


def async_copy_scales_smem_to_tmem(smem_ref: ir.Value, tmem_ref: TMEMRef):
  """Asynchronously copies the scale data from SMEM to TMEM.

  The result of the copy can be awaited by calling ``commit_arrive`` and waiting
  on the chosen ``Barrier``. However, if TMEM reference is to be consumed by a
  MMA issued in the same thread, no additional synchronization is needed.

  At the moment the function requires ``smem_ref`` to be contiguous and have a
  shape of (MN // 128, 32, 16) for 8-bit scales (here MN stands for the size of
  the non-contracting dimension which is M or N), matching the scale layout for
  .scale_vec::1X. See https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
  for more details. Note that we always put the non-contracting dimension first.
  If you have a (MN, 4) array of scales in JAX (where MN is divisible by 128),
  you can prepare it for use in the kernel this way::

      scales.reshape(-1, 4, 32, 4).swapaxes(1, 2).reshape(-1, 32, 16)

  The TMEM ref is expected to have the logical shape of the scales (MN, 4), and
  the layout created by ``scales_layout()``.
  """
  i32 = ir.IntegerType.get_signless(32)
  smem_ty = ir.MemRefType(smem_ref.type)
  if (dtype := smem_ty.element_type) != tmem_ref.dtype:
    raise ValueError(f"Incompatible dtypes: SMEM has {dtype}, TMEM has {tmem_ref.dtype}")
  if dtype != ir.Float8E8M0FNUType.get():
    raise NotImplementedError(f"Unsupported dtype: {dtype}, only f8e8m0fnu supported")
  if tmem_ref.shape[0] % TMEM_ROWS:
    raise ValueError(f"TMEM reference must have a multiple of {TMEM_ROWS} rows, but got {tmem_ref.shape[0]}")
  if tmem_ref.shape[1] != 4:
    raise ValueError(f"TMEM reference must have 4 colums, but got {tmem_ref.shape[1]}")
  if tmem_ref.layout != scales_layout():
    raise ValueError(f"TMEM layout {tmem_ref.layout} is not supported")
  smem_shape = tuple(smem_ty.shape)
  expected_smem_shape = (tmem_ref.shape[0] // TMEM_ROWS, 32, 16)
  if smem_shape != expected_smem_shape:
    raise NotImplementedError(
        f"SMEM has {smem_shape}, but expected {expected_smem_shape} for TMEM"
        f" ref shape {tmem_ref.shape}"
    )
  strides, _ = smem_ty.get_strides_and_offset()
  if strides != utils.get_contiguous_strides(smem_shape):
    raise ValueError("Only copies from contiguous SMEM references are supported")
  row_tile_stride = strides[0]
  if row_tile_stride % 4:
    raise ValueError("Column tile stride must be a multiple of 4")
  row_tile_stride_i32 = row_tile_stride // 4
  smem_base_ptr = utils.memref_ptr(smem_ref, 3)
  for row_tile in range(expected_smem_shape[0]):
    load_ptr = utils.getelementptr(
        smem_base_ptr, [row_tile * row_tile_stride_i32], i32
    )
    store_ptr = arith.addi(tmem_ref.address, arith.constant(i32, 4 * row_tile))
    # The "core matrix" here is the same as in MMA: 8x(16 bytes).
    desc = mma_utils.encode_descriptor(load_ptr, 0, 8 * 16, swizzle=None)
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [store_ptr, desc],
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [$0], $1;",
        "r,l",
        has_side_effects=True,
    )


def async_copy_sparse_metadata_smem_to_tmem(smem_ref: ir.Value, tmem_ref: TMEMRef):
  i8 = ir.IntegerType.get_signless(8)
  i32 = ir.IntegerType.get_signless(32)
  smem_ty = ir.MemRefType(smem_ref.type)
  if (dtype := smem_ty.element_type) != tmem_ref.dtype:
    raise ValueError(f"Incompatible dtypes: SMEM has {dtype}, TMEM has {tmem_ref.dtype}")
  if dtype != ir.IntegerType.get_signless(2):
    raise NotImplementedError(f"Unsupported dtype: {dtype}, only i2 supported")
  if tmem_ref.shape[0] % 128:
    raise ValueError(f"TMEM reference must have a multiple of 128 rows, but got {tmem_ref.shape[0]}")
  if tmem_ref.shape[1] % 64:
    raise ValueError(f"TMEM reference must have a multiple of 64 colums, but got {tmem_ref.shape[1]}")
  if tmem_ref.layout != sparse_meta_layout():
    raise ValueError(f"TMEM layout {tmem_ref.layout} is not supported")
  smem_shape = tuple(smem_ty.shape)
  expected_smem_shape = (tmem_ref.shape[0] // 128, tmem_ref.shape[1] // 64, 128, 64)
  if smem_shape != expected_smem_shape:
    raise NotImplementedError(
        f"SMEM has {smem_shape}, but expected {expected_smem_shape} for TMEM"
        f" ref shape {tmem_ref.shape}"
    )
  strides, _ = smem_ty.get_strides_and_offset()
  if strides != utils.get_contiguous_strides(smem_shape):
    raise ValueError("Only copies from contiguous SMEM references are supported")
  if expected_smem_shape[0] != 1:
    raise NotImplementedError("Only M=128 supported")
  k_tile_stride = strides[1]
  if k_tile_stride % 16:
    raise ValueError("K tile stride must be a multiple of 16")
  k_tile_byte_stride = k_tile_stride // 4
  smem_base_ptr = utils.memref_ptr(smem_ref, 3)
  for k_tile in range(expected_smem_shape[1]):
    load_ptr = utils.getelementptr(
        smem_base_ptr, [k_tile * k_tile_byte_stride], i8
    )
    store_ptr = arith.addi(tmem_ref.address, arith.constant(i32, 4 * k_tile))
    # The "core matrix" here is the same as in MMA: 8x(16 bytes).
    desc = mma_utils.encode_descriptor(load_ptr, 0, 8 * 16, swizzle=None)
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [store_ptr, desc],
        "tcgen05.cp.cta_group::1.128x128b [$0], $1;",
        "r,l",
        has_side_effects=True,
    )
