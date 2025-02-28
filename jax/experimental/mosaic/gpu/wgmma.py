# Copyright 2024 The JAX Authors. All Rights Reserved.
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
import functools
import itertools
from typing import Any

import jax
from jax._src.lib import mosaic_gpu_dialect as mgpu_dialect
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import nvvm
from jaxlib.mlir.dialects import vector
import numpy as np

from . import fragmented_array as fa
from . import utils

# mypy: ignore-errors

c = utils.c
bytewidth = utils.bytewidth


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class WGMMAAccumulator:
  """A FragmentedArray that has is synchronized with the async proxy.

  This implies that it requires no additional synchronization when passed in
  as a WGMMA accumulator. In particular, when created from a
  FragmentedArray, the necessary synchronization is inserted at construction.
  """
  value: fa.FragmentedArray

  def __init__(self, *, _value: fa.FragmentedArray, _sync: bool = True):
    if _value.layout not in (fa.WGMMA_LAYOUT, fa.TILED_LAYOUT_WGMMA):
      raise ValueError("Only WGMMA layouts supported in WGMMAAccumulator")
    self.value = _value
    if _sync:
      self.value = wgmma_fence(_value)

  @classmethod
  def zero(cls, m, n, dtype=None, *, is_signed: bool | None = None):
    if m % 64 or n % 8:
      raise ValueError
    if is_signed is False:
      raise TypeError("PTX does not support unsigned WGMMA accumulators")
    f32 = ir.F32Type.get()
    if dtype is None:
      dtype = f32
    zero = arith.constant(dtype, ir.FloatAttr.get(dtype, 0.0))
    return cls(
        _value=fa.FragmentedArray.splat(
            zero, (m, n), fa.WGMMA_LAYOUT, is_signed=is_signed
        )
    )

  @classmethod
  def from_registers(cls, registers):
    return cls(_value=registers)

  def tree_flatten(self):
    return (self.value,), ()

  @classmethod
  def tree_unflatten(cls, aux, value):
    del aux
    return cls(_value=value[0], _sync=False)


def wgmma_encode(x: int):
  result = (x & 0x3FFFF) >> 4
  if result << 4 != x:
    raise ValueError(f"Cannot encode value in a WGMMA descriptor: {x}")
  return result


def llvm_add(x, y):
  return llvm.add(x, y, overflow_flags=llvm.IntegerOverflowFlags.none)


def create_descriptor(
    memref_arg,
    leading_byte_offset: int,
    stride_byte_offset: int,
    swizzle: int | mgpu_dialect.SwizzlingMode | None,
    memory_space: int | None = None,
    const_init: int = 0,
):
  i64 = ir.IntegerType.get_signless(64)
  ptr_val = llvm.ptrtoint(i64, utils.memref_ptr(memref_arg, memory_space))
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
  encoded_base_addr = llvm.LShrOp(
      llvm.AndOp(ptr_val, c(0x3FFFF, i64)).result, c(4, i64)
  )
  # We ignore the offset
  desc_const = (
      const_init
      | (wgmma_encode(leading_byte_offset) << 16)
      | (wgmma_encode(stride_byte_offset) << 32)
  )
  desc = llvm.or_(
      arith.shli(c(swizzle_encoding, i64), c(62, i64)), c(desc_const, i64)
  )
  desc = llvm.or_(encoded_base_addr.result, desc)
  return desc


def _unpack_i32(vec_ty, r):
  i32 = ir.IntegerType.get_signless(32)
  return vector.bitcast(
      vec_ty, vector.splat(ir.VectorType.get((1,), i32), r)
  )


def _supported_wgmma_types(dtype, abtype) -> bool:
  input_types_are = lambda ty: ty.isinstance(abtype)
  if ir.F32Type.isinstance(dtype):
    return any(input_types_are(ty) for ty in (ir.FloatTF32Type, ir.BF16Type, ir.F16Type))
  elif ir.F16Type.isinstance(dtype):
    return input_types_are(ir.F16Type)
  else:
    return False


def wgmma_m64(
    acc: np.ndarray,  # of register Values
    a,
    b_descriptor: ir.Value,
    a_transpose: bool | None,
    b_transpose: bool,
    a_k_stride: int | None,
    b_k_stride: int,
    n: int,
    swizzle: int,
    element_type: ir.Type,
):
  out_ty = ir.VectorType(acc.flat[0].type).element_type
  if not _supported_wgmma_types(out_ty, element_type):
    raise ValueError(f"Usupported wgmma types {(out_ty, element_type)=}")
  if n % 8:
    raise ValueError

  i32 = ir.IntegerType.get_signless(32)
  i64 = ir.IntegerType.get_signless(64)
  index = ir.IndexType.get()
  if b_k_stride % 16:
    raise ValueError
  # Only 16-bit types support transposes
  supports_transpose = bytewidth(element_type) == 2
  if not supports_transpose and (a_transpose or b_transpose):
    raise ValueError("Only f16 WGMMA supports transposes")
  if a_in_regs := isinstance(a, fa.FragmentedArray):
    if a.mlir_dtype != ir.F16Type.get() and a.mlir_dtype != ir.BF16Type.get():
      raise ValueError(f"Unsupported A register array dtype: {a.mlir_dtype}")
    # Column count must be equal to swizzle // bytewidth.
    if a.layout not in (fa.TILED_LAYOUT_WGMMA, fa.WGMMA_LAYOUT) or a.shape != (64, swizzle // 2):
      raise ValueError("Unsupported A register array layout")
    if a_k_stride is not None or a_transpose is not None:
      raise ValueError("Unsupported WGMMA features with A in registers")
  else:
    if a_k_stride is None or a_k_stride % 16:
      raise ValueError
    if a_transpose is None:
      raise ValueError

  if ir.F32Type.isinstance(out_ty):
    num_acc_regs = n // 2
    out_ty_field = out_ty
    acc_regs = [  # pylint: disable=g-complex-comprehension
        vector.extractelement(reg, position=c(pos, index))
        for reg in acc.flat
        for pos in range(2)
    ]
    to_acc_vec_regs = functools.partial(_as_fragmented_reg_ndarray, dtype=out_ty, shape=acc.shape)
    acc_constraint = "f"
  elif ir.F16Type.isinstance(out_ty):
    num_acc_regs = n // 4
    out_ty_field = i32
    acc_regs = [_as_i32_reg(reg) for reg in acc.flat]
    vec_ty = ir.VectorType(acc.flat[0].type)
    to_acc_vec_regs = lambda regs : np.array([_unpack_i32(vec_ty, reg) for reg in regs]).reshape(acc.shape)
    acc_constraint = "r"
  else:
    raise ValueError(f"WGMMA instruciton only supports f32 and f16 out (got {out_ty})")

  num_imm_regs = 4 if supports_transpose else 2

  if a_in_regs:
    a_reg_constraints = ["r"] * 4  # 4x f16x2 registers
    num_imm_regs -= 1  # transpose not supported for a in registers
  else:
    a_reg_constraints = ["l"]  # descriptor
  # Reference for i/o aliasing: https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html
  # Seems like it's not actually documented in LLVM IR docs.
  reg_constraints_list = (
      [f"={acc_constraint}"] * num_acc_regs  # accumulator registers
      + [str(i) for i in range(num_acc_regs)]  # we alias outputs as inputs, too.
      + a_reg_constraints  # a descriptor / registers
      + ["l"] * 1  # b descriptor
      + ["n"] * (1 + num_imm_regs)  # literal constants
  )
  reg_constraints = ",".join(reg_constraints_list)

  reg_count = itertools.count()

  def take_regs(n):
    return (f"${i}" for i in itertools.islice(reg_count, n))

  acc_reg_vector = "{" + ",".join(take_regs(num_acc_regs)) + "}"
  for _ in take_regs(num_acc_regs):  # Ignore next entries: aliasing.
    pass
  if a_in_regs:
    a_regs = "{" + ",".join(take_regs(len(a_reg_constraints))) + "}"
  else:
    a_regs, = take_regs(1)
  b_desc_reg, use_out_reg = take_regs(2)
  imm_regs = ", ".join(take_regs(num_imm_regs))  # Immediate regs (scale, ...).
  assert next(reg_count) == len(reg_constraints_list)
  el_ty = element_type
  k_instr = 32 // bytewidth(element_type)
  wgmma_instr = (
      f"wgmma.mma_async.sync.aligned.m64n{n}k{k_instr}.{out_ty}.{el_ty}.{el_ty} "
      f"{acc_reg_vector}, {a_regs}, {b_desc_reg}, p, {imm_regs};"
  )
  ptx = f"{{ .reg .pred p; setp.ne.b32 p, {use_out_reg}, 0; {wgmma_instr} }}\n"

  def lc(x):
    return llvm.ConstantOp(i32, ir.IntegerAttr.get(i32, x)).result

  use_out = scale_a = scale_b = lc(1)
  imms = [use_out, scale_a, scale_b]
  if supports_transpose and a_transpose is not None:
    imms += [lc(int(a_transpose)), lc(int(b_transpose))]
  elif supports_transpose:
    imms += [lc(int(b_transpose))]
  if acc.ndim != 4 or acc.shape[0] != 1 or acc.shape[2:] != (2, 1):
    raise ValueError(acc.shape)
  acc_struct_type = ir.Type.parse(
      f"!llvm.struct<({','.join(str(out_ty_field) for _ in acc_regs)})>"
  )
  for i in range((swizzle // bytewidth(element_type)) // k_instr):
    # Slice out the relevant part of A or advance the A descriptor.
    if a_in_regs:
      a_slice = a[:, (i * 16) : ((i + 1) * 16)]
      a_args = [_as_i32_reg(v) for v in a_slice.registers.flat]
    else:
      if i > 0:
        a = llvm_add(
            a,
            llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, a_k_stride >> 4)),
        )
      a_args = [a]
    # Advance the B descriptor.
    if i > 0:
      b_descriptor = llvm_add(
          b_descriptor,
          llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, b_k_stride >> 4)),
      )
    assert len(a_args) == len(a_reg_constraints)
    acc_struct = llvm.inline_asm(
        acc_struct_type,
        [*acc_regs, *a_args, b_descriptor, *imms],
        ptx,
        reg_constraints,
        asm_dialect=0,
        has_side_effects=True,
    )
    acc_regs = [
        llvm.extractvalue(out_ty_field, acc_struct, [i]) for i in range(len(acc_regs))
    ]
  return to_acc_vec_regs(acc_regs)


class WGMMALayout(enum.Enum):
  ROW_MAJOR = enum.auto()
  COL_MAJOR = enum.auto()


def _validate_mma(
    a: Any,
    b: ir.Value,
    swizzle: int,
    descriptor_const_init: int = 0,
):
  # We need swizzle >= 32 to ensure that our K tiling is larger than the MMA
  # instruction's K width.
  if swizzle < 32:
    raise ValueError(f"Unsupported swizzle: {swizzle}")

  # Get A type.
  if a_in_smem := isinstance(a, ir.Value):
    if not ir.MemRefType.isinstance(a.type):
      raise ValueError(f"When A is an ir.Value, it must be a memref, got: {a.type}")
    a_ty = ir.MemRefType(a.type)
    a_element_type = a_ty.element_type
    a_shape = tuple(a_ty.shape)
    if a_ty.memory_space != ir.Attribute.parse("#gpu.address_space<workgroup>"):
      raise ValueError("A must be in workgroup memory when it's a reference")
    if len(a_shape) != 4:
      raise ValueError(f"A must be 4D when it's a reference, got rank {len(a_shape)}")
  elif hasattr(a, "shape") and hasattr(a, "mlir_dtype"):
    a_element_type = a.mlir_dtype
    a_shape = a.shape
  else:
    raise NotImplementedError(f"Unsupported A type: {type(a)}")

  # Get B type (always a reference).
  b_ty = ir.MemRefType(b.type)
  if b_ty.rank != 4:
    raise ValueError(f"B must be 4D, got rank {b_ty.rank}")

  # Veirfy element types and compute the tiling.
  if (element_type := a_element_type) != b_ty.element_type:
    raise ValueError(
        f"A and B must have the same element type, got: {a_element_type} and"
        f" {b_ty.element_type}"
    )
  supported_types = {ir.F16Type.get(), ir.BF16Type.get(), ir.F32Type.get()}
  if element_type not in supported_types:
    raise ValueError(a_element_type)
  element_bytewidth = bytewidth(element_type)
  swizzle_elems = swizzle // element_bytewidth

  # Verify the shape and strides of B are as expected.
  b_k_tiles, n_tiles, b_k_tiling, n_tiling = b_ty.shape
  k = b_k_tiles * b_k_tiling
  n = n_tiles * n_tiling

  b_strides, _ = b_ty.get_strides_and_offset()
  b_byte_strides = [s * element_bytewidth for s in b_strides]
  b_k_byte_stride, b_n_byte_stride, *b_tile_byte_strides = b_byte_strides
  if (
      b_byte_strides[1] != n_tiling * b_k_tiling * element_bytewidth
      and n_tiles != 1  # When there's only one tile, we never jump between them
  ):
    raise ValueError("B tiles must be contiguous along the N dimension")
  if b_tile_byte_strides == [swizzle, element_bytewidth]:  # N-fastest
    b_order = WGMMALayout.ROW_MAJOR
    # This first case (n_tiles == 1) is to allow the somewhat weird case of
    # loading a small amount of N-fastest data, that needs to be padded to a
    # larger tile due to swizzle. In this case we allow slicing the big tile
    # before WGMMA to avoid unnecessary compute on padding.
    if n_tiles == 1:
      if n_tiling % 8:
        raise ValueError("N tile size must be a multiple of 8")
    elif n_tiling != swizzle_elems:
      raise ValueError(
          "Row major RHS (N-fastest) requires the N tile size to be equal to"
          f" the swizzle tile size ({swizzle_elems}), but got {n_tiling}"
      )
    if b_k_tiling not in {32 // element_bytewidth, swizzle_elems}:
      raise ValueError(
          "Row major RHS (N-fastest) requires the K tile size to be either"
          f" the swizzle tile size ({swizzle_elems}) or 32 bytes"
          f" ({32 // element_bytewidth}), but got {b_k_tiling}"
      )
  elif b_tile_byte_strides == [element_bytewidth, swizzle]:  # K-fastest
    b_order = WGMMALayout.COL_MAJOR
    if b_k_tiling != swizzle_elems:
      raise ValueError(
          "Column major RHS (K-fastest) requires the K tile size to be equal"
          f" to the swizzle tile size ({swizzle_elems}), but got {b_k_tiling}"
      )
    # See the explanation in the N-fastest case when n_tiles == 1.
    if n_tiles == 1:
      if n_tiling % 8:
        raise ValueError("N tile size must be a multiple of 8")
    elif n_tiling not in {8, swizzle_elems}:
      raise ValueError(
          "Column major RHS (K-fastest) requires the N tile size to be either"
          f" to the swizzle tile size ({swizzle_elems}) or 8, but got {n_tiling}"
      )
  else:
    raise ValueError(b_byte_strides)

  # Verify the shape and strides of A are as expected.
  if not a_in_smem:
    m = a_shape[0]
    a_order = m_tiling = None
  else:
    a_ty = ir.MemRefType(a.type)
    m_tiles, a_k_tiles, m_tiling, a_k_tiling = a_ty.shape
    m = m_tiles * m_tiling
    if a_k_tiling != swizzle_elems or a_k_tiles * a_k_tiling != k:
      raise ValueError(a_ty.shape)
    a_strides, _ = ir.MemRefType(a.type).get_strides_and_offset()
    a_byte_strides = [s * element_bytewidth for s in a_strides]
    if a_byte_strides[2:] == [swizzle, element_bytewidth]:
      a_order = WGMMALayout.ROW_MAJOR
    elif a_byte_strides[2:] == [element_bytewidth, swizzle]:
      a_order = WGMMALayout.COL_MAJOR
    else:
      raise ValueError(a_byte_strides)
    if a_order != WGMMALayout.ROW_MAJOR and m_tiling != swizzle_elems:
      # Not sure what the layout is like, since the tiles aren't square.
      raise NotImplementedError

  b_k_fastest = b_order == WGMMALayout.COL_MAJOR
  a_k_fastest = a_order == WGMMALayout.ROW_MAJOR
  # This is the number of rows until consecutive repeats of the swizzle pattern.
  swizzle_pattern_rows = swizzle // 16
  # A swizzle atom is a 2D matrix with the dimensions below.
  swizzle_atom_bytes = swizzle_pattern_rows * 128

  # Here "leading" refers to the fastest changing dimension. There are two
  # strides we have to define per value:
  #   Leading byte offset (LBO)
  #     K-fastest: ignored
  #     MN-fastest: stride between consecutive swizzle atoms that share the same
  #       K coordinate.
  #   Stride byte offset (SBO)
  #     As far as I can tell this is just the offset between two consecutive
  #     swizzle atoms along the non-leading dimension.
  IGNORED = 0
  a_desc_fields = dict(
      # I can't fully explain why WGMMA ignores LBO for A. For a_k_fastest, it
      # is documented in the PTX docs, and my best explanation for the other
      # case is that the instruction has a fixed shape and so it does not care
      # about strides. It's possible that it's an artifact of the fact that we
      # use tiling of 64.
      leading_byte_offset=IGNORED,
      stride_byte_offset=swizzle_atom_bytes,
      swizzle=swizzle,
      memory_space=3,
  )
  # If B is N-fastest, all swizzle atoms within a tile share the same N
  # coordinate, so we simply take the stride between consecutive N tiles.
  # If B is K-fastest, all swizzle atoms within a tile share the same K
  # coordinate, which forces us to lay out the tiles in N-fastest order or else
  # they would have uneven strides.
  b_desc_fields = dict(
      leading_byte_offset=IGNORED if b_k_fastest else b_n_byte_stride,
      stride_byte_offset=swizzle_atom_bytes,
      swizzle=swizzle,
      memory_space=3,
  )
  # The K strides indicate the stride between the consecutive places where all
  # coordinates are 0 except for K being incremented by the instruction width.
  # If an input is K-fastest, we increment the descriptor by 32 bytes, since
  # that is the K-width of all MMA instructions.
  if b_k_fastest:
    b_k_wgmma_stride = 32
    b_k_group_stride = b_k_byte_stride  # The tile has only one K swizzle atom.
  elif b_k_tiling == swizzle_elems:
    # When B is N-fastest and we use the large square tiling, the relevant
    # slices all fall within the first tile. A single MMA instruction for 16-bit
    # types reads a subtile of shape 16x(swizzle bytes), giving us the necessary
    # expression.
    assert n_tiling == swizzle_elems or n_tiles == 1
    b_k_wgmma_stride = swizzle * 16
    b_k_group_stride = b_k_byte_stride
  else:
    # If we use the small non-square tiling and N-fastest layout, each tile only
    # contains a single swizzle atom with the K coordinate, so we just look up
    # the next tile.
    b_k_wgmma_stride = b_k_byte_stride
    wgmma_in_group = swizzle // 32
    b_k_group_stride = b_k_byte_stride * wgmma_in_group
  wgmma_params = dict(
      a_transpose=not a_k_fastest,
      b_transpose=not b_k_fastest,
      # TODO(apaszke): This explanation is quite bad. We should better figure
      # out how to do LHS transposes.
      # We only support swizzle=128 for M-fastest A. In this case the tile is
      # swizzle x 64 (= swizzle elems) and so we just take a quarter of its size.
      a_k_stride=32 if a_k_fastest else swizzle * 16,
      b_k_stride=b_k_wgmma_stride,
      swizzle=swizzle,
      element_type=ir.FloatTF32Type.get()
      if ir.F32Type.isinstance(element_type)
      else element_type,
  )
  if not a_in_smem:
    wgmma_params["a_k_stride"] = wgmma_params["a_transpose"] = None
    a_k_group_stride = a_desc_base = None
  else:
    a_desc_base = create_descriptor(
        a, **a_desc_fields, const_init=descriptor_const_init
    )
    a_strides, _ = ir.MemRefType(a.type).get_strides_and_offset()
    a_k_group_stride = a_strides[1] * element_bytewidth
  b_desc_base = create_descriptor(
      b, **b_desc_fields, const_init=descriptor_const_init
  )

  return (
      a_desc_base,
      b_desc_base,
      (m, k, n),
      (m_tiling, n_tiling),
      element_type,
      wgmma_params,
      a_k_group_stride,
      b_k_group_stride,
  )


# TODO(apaszke): Remove WGMMALayout. Make input shapes logical and infer
# transpositions from memref strides.
def wgmma(
    acc: WGMMAAccumulator,
    a: fa.FragmentedArray | ir.Value,
    b: ir.Value,
    *,
    swizzle: int = 128,
):
  """Perform acc += a @ b using the WGMMA instruction.

  The expected memref shapes are:
    a: (m, k, 64, S)
    b: (k, n,  S, S)
  where S = swizzle // bytewidth(element_type).

  The refs must be contiguous or be contiguous except for having their two minor
  dimensions swapped.
  """
  a_in_regs = isinstance(a, fa.FragmentedArray)
  if not a_in_regs and not ir.MemRefType.isinstance(a.type):
    raise ValueError(f"Unsupported A type: {type(a)}")
  if not ir.MemRefType.isinstance(b.type):
    raise ValueError(f"B must be a memref, got: {b.type}")

  (
      a_desc_base,
      b_desc_base,
      (m, k, n),
      (m_tiling, _),
      element_type,
      wgmma_params,
      a_k_group_stride,
      b_k_group_stride,
  ) = _validate_mma(a, b, swizzle)

  if n > 256:
    raise ValueError(f"N must be smaller than 256, got {n}")

  if a_in_regs:
    if a.mlir_dtype != ir.F16Type.get() and a.mlir_dtype != ir.BF16Type.get():
      raise ValueError(
          f"Only 16-bit dtypes supported for A in registers, got {a.mlir_dtype}"
      )
    if a.shape[0] % 64:
      raise ValueError(f"m must be a multiple of 64, got: {a.shape[0]}")
    a_m_group_stride = None
  else:
    if m_tiling != 64:
      raise ValueError(f"A must have rows tiled by 64, got: {m_tiling}")
    a_strides, _ = ir.MemRefType(a.type).get_strides_and_offset()
    a_m_group_stride = a_strides[0] * bytewidth(element_type)

  k_group_width = swizzle // bytewidth(element_type)
  groups_k = k // k_group_width
  groups_m = m // 64

  expected_acc_shape = (groups_m * 64, n)
  if acc.value.shape != expected_acc_shape:
    raise ValueError(
        f"Accumulator shape mismatch: expected {expected_acc_shape}, got"
        f" {acc.value.shape}"
    )

  if a_in_regs:
    a = wgmma_fence(a)  # Make sure the registers are ready.

  i64 = ir.IntegerType.get_signless(64)
  new_acc_regs = acc.value.registers.copy()
  for mi in range(groups_m):
    for ki in range(groups_k):
      if a_in_regs:
        a_mk = a[
            mi * 64 : (mi + 1) * 64,
            ki * k_group_width : (ki + 1) * k_group_width
        ]
      else:
        a_mk = llvm_add(
            a_desc_base,
            c(wgmma_encode(mi * a_m_group_stride + ki * a_k_group_stride), i64),
        )
      b_k = llvm_add(b_desc_base, c(wgmma_encode(ki * b_k_group_stride), i64))
      new_acc_regs[mi : mi + 1] = wgmma_m64(
          new_acc_regs[mi : mi + 1], a_mk, b_k, n=n, **wgmma_params
      )
  return WGMMAAccumulator(
      _value=fa.FragmentedArray(
          _registers=new_acc_regs,
          _layout=fa.WGMMA_LAYOUT,
          _is_signed=acc.value.is_signed,
      ),
      _sync=False,
  )


def wgmma_fence(array: fa.FragmentedArray):
  """Fences the array construction from WGMMA instructions.

  LLVM treats in-register computation as pure and can move it after the fence,
  which is explicitly disallowed by the PTX programming model. For that reason,
  we insert an LLVM optimization barrier before the fence.
  """
  array = fa.optimization_barrier(array)
  nvvm.wgmma_fence_aligned()
  return array


def _as_fragmented_reg_ndarray(flat_regs, dtype: ir.Type, shape: tuple[int, ...]):
  vec_regs = []
  for first, second in zip(flat_regs[::2], flat_regs[1::2]):
    vec = llvm.mlir_undef(ir.VectorType.get((2,), dtype))
    vec = llvm.insertelement(vec, first, position=_lc(0))
    vec = llvm.insertelement(vec, second, position=_lc(1))
    vec_regs.append(vec)
  return np.asarray(vec_regs, dtype=object).reshape(shape)


def _as_i32_reg(v):
  i32 = ir.IntegerType.get_signless(32)
  return llvm.extractelement(
      vector.bitcast(ir.VectorType.get((1,), i32), v), _lc(0)
  )


def _lc(x):
  i32 = ir.IntegerType.get_signless(32)
  return llvm.ConstantOp(i32, ir.IntegerAttr.get(i32, x)).result
