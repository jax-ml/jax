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
    a_layout: WGMMALayout,
    b_layout: WGMMALayout,
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
  kn_tiling = swizzle // element_bytewidth

  # Verify the shape and strides of B are as expected.
  k_tiles, n_tiles, k_tiling, n_tiling = b_ty.shape
  if k_tiling != kn_tiling:
    raise ValueError(b_ty.shape)
  # Note that while this technically allows n to be smaller than kn_tile,
  # the stride checks above will still enforce that the memory region is padded.
  # It might be possible to relax that requirement, but I haven't tested it.
  if n_tiling > kn_tiling and n_tiling % kn_tiling:
    raise ValueError(n_tiling, kn_tiling)
  k = k_tiles * kn_tiling
  n = n_tiles * n_tiling

  b_strides, _ = b_ty.get_strides_and_offset()
  b_byte_strides = [s * element_bytewidth for s in b_strides]
  b_k_byte_stride = b_byte_strides[0]
  if b_byte_strides[1] != swizzle * kn_tiling:
    raise ValueError(b_byte_strides)
  if b_byte_strides[2:] == [swizzle, element_bytewidth]:
    b_order = WGMMALayout.ROW_MAJOR
  elif b_byte_strides[2:] == [element_bytewidth, swizzle]:
    b_order = WGMMALayout.COL_MAJOR
  else:
    raise ValueError(b_byte_strides)

  # Verify the shape and strides of A are as expected.
  if not a_in_smem:
    m = a_shape[0]
    a_order = m_tiling = None
  else:
    a_ty = ir.MemRefType(a.type)
    m_tiles, k_tiles, m_tiling, k_tiling = a_ty.shape
    m = m_tiles * m_tiling
    if k_tiling != kn_tiling or k_tiles * k_tiling != k:
      raise ValueError(a_ty.shape)
    a_strides, _ = ir.MemRefType(a.type).get_strides_and_offset()
    a_byte_strides = [s * element_bytewidth for s in a_strides]
    if a_byte_strides[2:] == [swizzle, element_bytewidth]:
      a_order = WGMMALayout.ROW_MAJOR
    elif a_byte_strides[2:] == [element_bytewidth, swizzle]:
      a_order = WGMMALayout.COL_MAJOR
    else:
      raise ValueError(a_byte_strides)
    if a_order != a_layout and m_tiling != kn_tiling:
      # Not sure what the layout is like, since the tiles aren't square.
      raise NotImplementedError

  tnsp_lbo = swizzle * (swizzle // 32)
  sbo = swizzle // 2
  a_desc_fields = dict(
      leading_byte_offset=(1 if a_order == a_layout else tnsp_lbo) << 4,
      stride_byte_offset=sbo << 4,
      swizzle=swizzle,
      memory_space=3,
  )
  b_desc_fields = dict(
      leading_byte_offset=(1 if b_order == b_layout else tnsp_lbo) << 4,
      stride_byte_offset=sbo << 4,
      swizzle=swizzle,
      memory_space=3,
  )
  wgmma_params = dict(
      a_transpose=a_order != a_layout,
      b_transpose=b_order != b_layout,
      a_k_stride=(2 if a_order == a_layout else swizzle) << 4,
      b_k_stride=(2 if b_order == b_layout else swizzle) << 4,
      swizzle=swizzle,
      element_type=ir.FloatTF32Type.get()
      if ir.F32Type.isinstance(element_type)
      else element_type,
  )
  if not a_in_smem:
    wgmma_params["a_k_stride"] = wgmma_params["a_transpose"] = None
    a_k_byte_stride = a_desc_base = None
  else:
    a_desc_base = create_descriptor(
        a, **a_desc_fields, const_init=descriptor_const_init
    )
    a_strides, _ = ir.MemRefType(a.type).get_strides_and_offset()
    a_k_byte_stride = a_strides[1] * element_bytewidth
  b_desc_base = create_descriptor(
      b, **b_desc_fields, const_init=descriptor_const_init
  )

  return (
      a_desc_base,
      b_desc_base,
      (m, k, n),
      (m_tiling, kn_tiling),
      element_type,
      wgmma_params,
      a_k_byte_stride,
      b_k_byte_stride,
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
      (m_tiling, kn_tiling),
      element_type,
      wgmma_params,
      a_k_byte_stride,
      b_k_byte_stride,
  ) = _validate_mma(a, b, swizzle, WGMMALayout.ROW_MAJOR, WGMMALayout.COL_MAJOR)

  if n > 256:
    raise ValueError(f"N must be smaller than 256, got {n}")

  if a_in_regs:
    if a.mlir_dtype != ir.F16Type.get() and a.mlir_dtype != ir.BF16Type.get():
      raise ValueError(
          f"Only 16-bit dtypes supported for A in registers, got {a.mlir_dtype}"
      )
    if a.shape[0] % 64:
      raise ValueError(f"m must be a multiple of 64, got: {a.shape[0]}")
    a_m_byte_stride = None
  else:
    if m_tiling != 64:
      raise ValueError(f"A must have rows tiled by 64, got: {m_tiling}")
    a_strides, _ = ir.MemRefType(a.type).get_strides_and_offset()
    a_m_byte_stride = a_strides[0] * bytewidth(element_type)

  groups_k = k // kn_tiling
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
        a_mk = a[mi * 64 : (mi + 1) * 64, ki * kn_tiling : (ki + 1) * kn_tiling]
      else:
        a_mk = llvm_add(
            a_desc_base,
            c(wgmma_encode(mi * a_m_byte_stride + ki * a_k_byte_stride), i64),
        )
      b_k = llvm_add(b_desc_base, c(wgmma_encode(ki * b_k_byte_stride), i64))
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
