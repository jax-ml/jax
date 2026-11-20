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
from jax._src import dtypes
from jax._src import typing
from jax.experimental.mosaic.gpu import fragmented_array as fa
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import vector
import numpy as np
from . import utils


SUPPORTED_F8_TYPES = (ir.Float8E4M3FNType, ir.Float8E5M2Type)


class MMALayouts:
  """Container for MMA layouts, providing a convenient way to create
  layouts for MMA operands based on warp configuration.
  """

  def __init__(
      self,
      element_type: ir.Type | typing.DTypeLike,
      *,
      m_warps: int = 4,
      block_size: int | None = None,
  ):
    if m_warps not in (1, 2, 4):
      raise ValueError(f"m_warps must be 1, 2, or 4, but got {m_warps=}")
    if block_size not in (None, 16, 32):
      raise ValueError(f"block_size must be 16, 32 or None, but got {block_size}")
    n_warps = 4 // m_warps
    if isinstance(element_type, ir.Type):
      bitwidth = utils.bitwidth(element_type)
    else:
      bitwidth = dtypes.itemsize_bits(element_type)
    # 64-bit operands use the same layouts as 32-bit operands on Hopper+: one
    # element per (wide) register.
    elems_per_reg = max(32 // bitwidth, 1)
    k = 8 * elems_per_reg
    sub_k = 4 * elems_per_reg
    self.lhs = fa.TiledLayout(
        fa.Tiling(((m_warps * 16, k), (16, sub_k), (8, sub_k), (elems_per_reg,))),
        warp_dims=(-7, fa.Replicated(4 // m_warps)),
        lane_dims=(-3, -2),
        vector_dim=-1,
        _check_canonical=False,
    ).canonicalize()
    self.rhs = fa.TiledLayout(
        fa.Tiling(((k, n_warps * 8), (sub_k, 8), (elems_per_reg, 1))),
        warp_dims=(fa.Replicated(4 // n_warps), -5,),
        lane_dims=(-3, -4),
        vector_dim=-2,
        _check_canonical=False,
    ).canonicalize()
    self.acc = fa.TiledLayout(
        fa.Tiling(((m_warps * 16, n_warps * 8), (16, 8), (8, 8), (2,))),
        warp_dims=(-7, -6),
        lane_dims=(-3, -2),
        vector_dim=-1,
        _check_canonical=False,
    ).canonicalize()

    self.lhs_scales = None
    self.rhs_scales = None
    if block_size is not None:
      k_sf = 64 // block_size
      self.lhs_scales = fa.TiledLayout(
          fa.Tiling(tiles=(
              (m_warps * 16, k_sf),
              (16, k_sf),
              (8, k_sf),
          )),
          warp_dims=(-6,),
          lane_dims=(-2, fa.Replicated(2), -4),
          vector_dim=-1,
          _check_canonical=False,
      ).canonicalize()

      self.rhs_scales = fa.TiledLayout(
          fa.Tiling(tiles=(
              (k_sf, n_warps * 8),
              (k_sf, 8),
          )),
          warp_dims=(
              fa.Replicated(m_warps),
              -3 if n_warps > 1 else fa.Replicated(times=1),
          ),
          lane_dims=(-1, fa.Replicated(times=4)),
          vector_dim=-2,
          _check_canonical=False,
      ).canonicalize()

  @classmethod
  def for_shape(
      cls,
      element_type: ir.Type | typing.DTypeLike,
      m: int,
      n: int,
      *,
      block_size: int | None = None
  ) -> "MMALayouts":
    """Returns `MMALayouts` with `m_warps` selected for the given MMA shape.

    The four warps of the warpgroup are distributed as `m_warps` along the M
    dimension and `n_warps = 4 // m_warps` along the N dimension. We pick the
    largest valid `m_warps` so that the accumulator tile `(m_warps * 16,
    n_warps * 8)` evenly divides the `(m, n)` shape.
    """
    for m_warps in (4, 2, 1):
      n_warps = 4 // m_warps
      if m % (m_warps * 16) == 0 and n % (n_warps * 8) == 0:
        return cls(element_type, m_warps=m_warps, block_size=block_size)
    raise ValueError(
        f"No valid m_warps in (1, 2, 4) for MMA shape {(m, n)=}: the "
        "accumulator tile (m_warps * 16, (4 // m_warps) * 8) must evenly "
        "divide the shape."
    )


def _ptx_dtype_str(dtype: ir.Type, *, is_signed: bool | None = None) -> str:
  if isinstance(dtype, ir.Float8E4M3FNType):
    # F8E4M3FNU (ue4m3) does not exist in MLIR. We use F8E4M3FN (e4m3) instead.
    return "e4m3" if is_signed in (None, True) else "ue4m3"
  elif isinstance(dtype, ir.Float8E5M2Type):
    return "e5m2"
  elif isinstance(dtype, ir.Float4E2M1FNType):
    return "e2m1"
  elif isinstance(dtype, ir.Float8E8M0FNUType):
    return "ue8m0"
  elif isinstance(dtype, ir.IntegerType):
    if is_signed is None:
      raise ValueError("is_signed must be specified for integer types")
    prefix = "s" if is_signed else "u"
    return f"{prefix}{dtype.width}"
  return str(dtype)


def _mma_single_tile(
    acc: fa.FragmentedArray,
    a: fa.FragmentedArray,
    b: fa.FragmentedArray,
    *,
    a_scale: fa.FragmentedArray | None = None,
    b_scale: fa.FragmentedArray | None = None,
) -> fa.FragmentedArray:
  """Performs `acc + a @ b` using warp level MMA instructions."""
  i32 = ir.IntegerType.get_signless(32)
  f64 = ir.F64Type.get()

  assert a.mlir_dtype == b.mlir_dtype
  is_integer = isinstance(a.mlir_dtype, ir.IntegerType)
  is_f64 = a.mlir_dtype == f64
  if is_f64 and utils.get_arch() < utils.HOPPER:
    # TODO(bchetioui): Support sm_80 via the `m8n8k4` MMA shape.
    raise NotImplementedError("f64 MMA is only supported on Hopper+ hardware.")
  k_tile = 8 if is_f64 else 256 // utils.bitwidth(a.mlir_dtype)
  if is_integer:
    assert acc.mlir_dtype == i32
  elif is_f64:
    assert acc.mlir_dtype == f64
  else:
    assert acc.mlir_dtype == ir.F32Type.get()
  assert acc.is_signed in {None, True}
  assert (
      isinstance(acc.layout, fa.TiledLayout)
      and isinstance(a.layout, fa.TiledLayout)
      and isinstance(b.layout, fa.TiledLayout)
  )

  if (is_scaled := a_scale is not None) != (b_scale is not None):
    raise ValueError("a_scale and b_scale must both be present or absent.")

  if is_scaled and is_f64:
    raise ValueError("f64 is not supported for block scaled MMA.")

  num_acc_regs, num_a_regs, num_b_regs = 4, 4, 2

  acc_regs = [
      vector.extract(
          reg,
          dynamic_position=[],
          static_position=ir.DenseI64ArrayAttr.get([pos]),
      )
      for reg in acc.registers.flatten()
      for pos in range(acc.layout.vector_length)
  ]
  if is_f64:
    ab_reg_type = ir.IntegerType.get_signless(64)
  else:
    ab_reg_type = i32
  a_regs = [utils.bitcast(r, ab_reg_type) for r in a.registers.flatten()]
  b_regs = [utils.bitcast(r, ab_reg_type) for r in b.registers.flatten()]

  # Make sure we have the right number of registers for the instruction.
  assert len(a_regs) == num_a_regs
  assert len(acc_regs) == num_acc_regs
  assert len(b_regs) == num_b_regs

  a_ptx_dtype = _ptx_dtype_str(a.mlir_dtype, is_signed=a.is_signed)
  b_ptx_dtype = _ptx_dtype_str(b.mlir_dtype, is_signed=b.is_signed)
  if is_integer:
    acc_ptx_dtype = "s32"
  elif is_f64:
    acc_ptx_dtype = "f64"
  else:
    acc_ptx_dtype = "f32"

  counter = itertools.count()
  n_regs_str = lambda n: (
      "{" + ",".join([f"${next(counter)}" for _ in range(n)]) + "}"
  )
  out_regs_str = n_regs_str(num_acc_regs)
  a_regs_str = n_regs_str(num_a_regs)
  b_regs_str = n_regs_str(num_b_regs)
  c_regs_str = n_regs_str(num_acc_regs)

  if is_scaled:
    assert a_scale is not None and b_scale is not None
    a_scale_regs = [utils.bitcast(r, i32) for r in a_scale.registers.flatten()]
    b_scale_regs = [utils.bitcast(r, i32) for r in b_scale.registers.flatten()]
    assert len(a_scale_regs) == 1
    assert len(b_scale_regs) == 1
    # scale type can be: F8E8M0FNU or F8E4M3FNU
    # F8E4M3FNU (ue4m3) does not exist in MLIR. We use F8E4M3FN (e4m3) instead.
    # but for PTX scale_type should be unsigned: ue4m3 or ue8m0
    is_signed = False
    scale_ptx_dtype = _ptx_dtype_str(a_scale.mlir_dtype, is_signed=is_signed)
    b_scale_ptx_dtype = _ptx_dtype_str(b_scale.mlir_dtype, is_signed=is_signed)
    assert scale_ptx_dtype == b_scale_ptx_dtype

    a_scale_str = n_regs_str(1)
    a_scale_byte_thread_ids_str = n_regs_str(2)
    b_scale_str = n_regs_str(1)
    b_scale_byte_thread_ids_str = n_regs_str(2)

    i16 = ir.IntegerType.get_signless(16)
    # TODO: Adapt for .scale_vec::2X case
    a_scale_byte_thread_ids = [utils.c(0, i16), utils.c(0, i16)]
    b_scale_byte_thread_ids = [utils.c(0, i16), utils.c(0, i16)]

    # mma.sync.aligned.m16n8k64.row.col.kind.block_scale.scale_vec_size.f32.e2m1.e2m1.f32.stype d, a, b, c, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data, {byte-id-b, thread-id-b};
    # .kind           = {.kind::mxf4nvf4};
    # .scale_vec_size = {.scale_vec::2X, .scale_vec::4X};
    # .stype          = {.ue8m0, .ue4m3};

    scale_vec = f"scale_vec::{a_scale.shape[1]}X"
    kind = f".kind::mxf4nvf4.block_scale.{scale_vec}"
    instr_dtypes = f".{acc_ptx_dtype}.{a_ptx_dtype}.{b_ptx_dtype}.{acc_ptx_dtype}.{scale_ptx_dtype}"
    instr = f"mma.sync.aligned.m16n8k{k_tile}.row.col{kind}{instr_dtypes}"
    out_regs = utils.inline_ptx(
        f"{instr} {out_regs_str}, {a_regs_str}, {b_regs_str}, {c_regs_str}, "
        f"{a_scale_str}, {a_scale_byte_thread_ids_str}, {b_scale_str}, {b_scale_byte_thread_ids_str};",
        *a_regs,
        *b_regs,
        *acc_regs,
        *a_scale_regs,
        *a_scale_byte_thread_ids,
        *b_scale_regs,
        *b_scale_byte_thread_ids,
        result_types=[acc.mlir_dtype] * len(acc_regs),
        convergent=True,
    )
  else:
    instr = f"mma.sync.aligned.m16n8k{k_tile}.row.col.{acc_ptx_dtype}.{a_ptx_dtype}.{b_ptx_dtype}.{acc_ptx_dtype}"
    out_regs = utils.inline_ptx(
        f"{instr} {out_regs_str}, {a_regs_str}, {b_regs_str}, {c_regs_str};",
        *a_regs,
        *b_regs,
        *acc_regs,
        result_types=[acc.mlir_dtype] * len(acc_regs),
        convergent=True,
    )

  vec_regs = []
  vec_undef = llvm.mlir_undef(ir.VectorType.get((2,), acc.mlir_dtype))
  for first, second in zip(out_regs[::2], out_regs[1::2]):
    vec = llvm.insertelement(vec_undef, first, position=utils.c(0, i32))
    vec = llvm.insertelement(vec, second, position=utils.c(1, i32))
    vec_regs.append(vec)
  out_regs = np.asarray(vec_regs, dtype=object).reshape(acc.registers.shape)
  return fa.FragmentedArray(
      _registers=out_regs, _layout=acc.layout, _is_signed=acc.is_signed
  )


def mma(
    acc: fa.FragmentedArray,
    a: fa.FragmentedArray,
    b: fa.FragmentedArray,
    *,
    a_scale: fa.FragmentedArray | None = None,
    b_scale: fa.FragmentedArray | None = None,
) -> fa.FragmentedArray:
  """Computes `acc + a @ b` using synchronous warp-level MMA instructions.

  All operands must have `TiledLayout`s. The layouts must be generated
  by the `MMALayouts` class, which ensures that the tiles are mapped
  to the warps correctly.

  Args:
    acc: A `FragmentedArray` with a `TiledLayout` generated from
      `MMALayouts.acc`.
    a: A `FragmentedArray` with a `TiledLayout`  generated from
      `MMALayouts.lhs`.
    b: A `FragmentedArray` with a `TiledLayout` generated from `MMALayouts.rhs`.
    a_scale: Optional A scale `FragmentedArray` with a `TiledLayout`  generated from
      `MMALayouts.???`.
    b_scale: Optional B scale `FragmentedArray` with a `TiledLayout`  generated from
      `MMALayouts.???`.

  Returns:
    A new `FragmentedArray` with the result of the computation with
      the same type as `acc`.
  """

  (m, k) = a.shape
  (k2, n) = b.shape
  (m2, n2) = acc.shape

  if m != m2:
    raise ValueError(f"M mismatch: {m} != {m2}")
  if n != n2:
    raise ValueError(f"N mismatch: {n} != {n2}")
  if k != k2:
    raise ValueError(f"K mismatch: {k} != {k2}")

  if (is_scaled := a_scale is not None) != (b_scale is not None):
    raise ValueError("a_scale and b_scale must both be present or absent.")

  if is_scaled:
    assert a_scale is not None and b_scale is not None
    m3, sfk = a_scale.shape
    sfk2, n3 = b_scale.shape
    if m != m3 or n != n3 or sfk != sfk2:
      raise ValueError(
          f"Incompatible shapes for matrix multiplication: lhs={a.shape},"
          f" rhs={b.shape}, lhs_scale={a_scale.shape},"
          f" rhs_scale={b_scale.shape}"
      )
    if k % sfk != 0:
      raise ValueError(
          f"K={k} is not divisible by A scale second dimension"
          f" {sfk}"
      )

  # todo(cperivol): A tile shape can have dimensions that are higher
  # multiples of the mma op size as long as those dimensions are not
  # sharded across warps.
  i4 = ir.IntegerType.get_signless(4)
  i8 = ir.IntegerType.get_signless(8)
  i32 = ir.IntegerType.get_signless(32)
  bf16 = ir.BF16Type.get()
  f16 = ir.F16Type.get()
  f32 = ir.F32Type.get()
  f64 = ir.F64Type.get()
  f8e4m3fn = ir.Float8E4M3FNType.get()
  f8e8m0fnu = ir.Float8E8M0FNUType.get()
  f8e5m2 = ir.Float8E5M2Type.get()
  f4e2m1fn = ir.Float4E2M1FNType.get()
  if (element_type := a.mlir_dtype) != b.mlir_dtype:
    raise ValueError(f"Dtype mismatch: {a.mlir_dtype} != {b.mlir_dtype}")
  if element_type not in (bf16, f16, f64, f8e4m3fn, f8e5m2, f4e2m1fn, i8, i4):
    raise NotImplementedError(f"Unsupported operand type: {element_type}")
  if isinstance(element_type, ir.IntegerType):
    if acc.mlir_dtype != i32:
      raise NotImplementedError("Only s32 accumulator supported for integer operands.")
    if not acc.is_signed:
      raise ValueError("Only signed accumulator supported for integer operands.")
  elif element_type == f64:
    if acc.mlir_dtype != f64:
      raise NotImplementedError(
          "Only f64 accumulator supported for f64 operands."
      )
  elif acc.mlir_dtype != f32:
    raise NotImplementedError("Only f32 accumulator supported for floating operands.")

  can_infer_from_acc_layout = (
      isinstance(acc.layout, fa.TiledLayout)
      and len(acc.layout.base_tile_shape) == 2
      and acc.layout.base_tile_shape[0] % 16 == 0
  )
  if not can_infer_from_acc_layout:
    raise ValueError("Expected MMALayouts.acc for acc")
  m_warps = acc.layout.base_tile_shape[0] // 16  # type: ignore
  layouts = MMALayouts(element_type, m_warps=m_warps)
  if layouts.lhs != a.layout:
    raise ValueError("Expected MMALayouts.lhs layout for A")
  if layouts.rhs != b.layout:
    raise ValueError("Expected MMALayouts.rhs layout for B")
  if layouts.acc != acc.layout:
    raise ValueError("Expected MMALayouts.acc layout for acc")

  if is_scaled:
    assert a_scale is not None and b_scale is not None
    if (scale_element_type := a_scale.mlir_dtype) != b_scale.mlir_dtype:
      raise ValueError(f"Dtype mismatch: {a_scale.mlir_dtype} != {b_scale.mlir_dtype}")
    if element_type not in (f4e2m1fn, ):
      raise NotImplementedError(f"Unsupported A and B type: {element_type} for scaled MMA")
    if scale_element_type not in (f8e4m3fn, f8e8m0fnu):
      raise NotImplementedError(f"Unsupported A and B type: {element_type} for scaled MMA")

  assert isinstance(a.layout, fa.TiledLayout)
  assert isinstance(b.layout, fa.TiledLayout)
  assert isinstance(acc.layout, fa.TiledLayout)
  m_tile, k_tile = a.layout.base_tile_shape
  k_tile2, n_tile = b.layout.base_tile_shape
  m_tile2, n_tile2 = acc.layout.base_tile_shape

  assert k_tile == k_tile2
  assert m_tile2 == m_tile
  assert n_tile2 == n_tile

  sfk_tile = None
  if is_scaled:
    assert a_scale is not None and b_scale is not None
    assert isinstance(a_scale.layout, fa.TiledLayout)
    assert isinstance(b_scale.layout, fa.TiledLayout)

    m_tile2, sfk_tile = a_scale.layout.base_tile_shape
    sfk_tile2, n_tile2 = b_scale.layout.base_tile_shape

    assert m_tile == m_tile2
    assert sfk_tile == sfk_tile2
    assert n_tile2 == n_tile

  num_m_tiles, num_n_tiles, num_k_tiles = m // m_tile, n // n_tile, k // k_tile

  # Do not modify the accumualtor itself.
  acc = acc.copy()
  s = lambda idx, length: slice(idx * length, (idx + 1) * length)
  scales = {"a_scale": None, "b_scale": None}
  for k_idx in range(num_k_tiles):
    for m_idx in range(num_m_tiles):
      for n_idx in range(num_n_tiles):
        ms = s(m_idx, m_tile)
        ns = s(n_idx, n_tile)
        ks = s(k_idx, k_tile)
        if is_scaled:
          assert sfk_tile is not None
          assert a_scale is not None and b_scale is not None
          sfks = s(k_idx, sfk_tile)
          scales["a_scale"] = a_scale[ms, sfks]
          scales["b_scale"] = b_scale[sfks, ns]
        acc[ms, ns] = _mma_single_tile(
            acc[ms, ns], a[ms, ks], b[ks, ns], **scales
        )

  return acc
