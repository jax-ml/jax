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
import functools
import itertools
import math

import jax
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import nvvm
from jaxlib.mlir.dialects import vector
import numpy as np

from . import fragmented_array as fa
from . import mma_utils
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
    if _value.layout != fa.WGMMA_LAYOUT:
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
    if ir.IntegerType.isinstance(dtype):
      zero = arith.constant(dtype, ir.IntegerAttr.get(dtype, 0))
    else:
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


def _supported_wgmma_types(dtype, abtype) -> bool:
  input_types_are = lambda ty: ty.isinstance(abtype)
  f16_acc_types = (ir.F16Type, ir.Float8E5M2Type, ir.Float8E4M3FNType)
  if ir.F32Type.isinstance(dtype):
    return any(input_types_are(ty) for ty in (ir.FloatTF32Type, ir.BF16Type, *f16_acc_types))
  elif ir.F16Type.isinstance(dtype):
    return any(input_types_are(ty) for ty in f16_acc_types)
  elif ir.IntegerType.get_signless(32).isinstance(dtype):
    return input_types_are(ir.IntegerType.get_signless(8))
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
    raise ValueError(f"Unsupported wgmma types {(out_ty, element_type)=}")
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
    if a.layout != fa.WGMMA_LAYOUT or a.shape != (64, swizzle // 2):
      raise ValueError("Unsupported A register array layout")
    if a_k_stride is not None or a_transpose is not None:
      raise ValueError("Unsupported WGMMA features with A in registers")
  else:
    if a_k_stride is None or a_k_stride % 16:
      raise ValueError
    if a_transpose is None:
      raise ValueError

  if ir.F32Type.isinstance(out_ty) or out_ty == i32:
    num_acc_regs = n // 2
    out_ty_field = out_ty
    acc_regs = [  # pylint: disable=g-complex-comprehension
        vector.extractelement(reg, position=c(pos, index))
        for reg in acc.flat
        for pos in range(2)
    ]
    to_acc_vec_regs = functools.partial(
        _as_fragmented_reg_ndarray, dtype=out_ty, shape=acc.shape)
    acc_constraint = "r" if ir.IntegerType.isinstance(out_ty) else "f"
  elif ir.F16Type.isinstance(out_ty):
    num_acc_regs = n // 4
    out_ty_field = i32
    acc_regs = [_as_i32_reg(reg) for reg in acc.flat]
    vec_ty = ir.VectorType(acc.flat[0].type)
    to_acc_vec_regs = lambda regs : np.array([_unpack_i32(vec_ty, reg) for reg in regs]).reshape(acc.shape)
    acc_constraint = "r"
  else:
    raise ValueError(
        f"WGMMA instruction only supports f32, f16 and s32 out (got {out_ty})")

  if supports_transpose:
    num_imm_regs = 4
  elif out_ty == i32:
    num_imm_regs = 0
  else:
    num_imm_regs = 2

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
  # Immediate regs (scale, ...).
  imm_regs = "".join(f", {r}" for r in take_regs(num_imm_regs))
  assert next(reg_count) == len(reg_constraints_list)
  k_instr = 32 // bytewidth(element_type)
  el_ty = str(element_type)
  if ir.Float8E5M2Type.isinstance(element_type):
    el_ty = "e5m2"
  elif ir.Float8E4M3FNType.isinstance(element_type):
    el_ty = "e4m3"
  elif ir.IntegerType.get_signless(8).isinstance(element_type):
    # TODO(bchetioui): add u8 support in the future. Currently we always assume
    # that 8-bit integers are s8, and we would need to change the signature of
    # `wgmma` to indicate whether the input should be treated as signed or not.
    el_ty = "s8"

  out_ty_str = str(out_ty)
  if out_ty == i32:
    out_ty_str = "s32"

  wgmma_instr = (
      f"wgmma.mma_async.sync.aligned.m64n{n}k{k_instr}.{out_ty_str}.{el_ty}.{el_ty} "
      f"{acc_reg_vector}, {a_regs}, {b_desc_reg}, p{imm_regs};"
  )
  ptx = f"{{ .reg .pred p; setp.ne.b32 p, {use_out_reg}, 0; {wgmma_instr} }}\n"

  def lc(x):
    return llvm.ConstantOp(i32, ir.IntegerAttr.get(i32, x)).result

  use_out = scale_a = scale_b = lc(1)
  if out_ty == i32:
    imms = [use_out]
  else:
    imms = [use_out, scale_a, scale_b]

  if supports_transpose and a_transpose is not None:
    imms += [lc(int(a_transpose)), lc(int(b_transpose))]
  elif supports_transpose:
    imms += [lc(int(b_transpose))]

  assert len(imms) == num_imm_regs + 1  # +1 for the use_out_reg in setp.ne.b32

  if acc.ndim != 10 or acc.shape[0] != 1 or math.prod(acc.shape[2:]) != 2:
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
        a = _llvm_add(
            a,
            llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, a_k_stride >> 4)),
        )
      a_args = [a]
    # Advance the B descriptor.
    if i > 0:
      b_descriptor = _llvm_add(
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
  if swizzle == 16:
    raise NotImplementedError("No swizzle is not supported")
  # Step 1. Establish the shape and element type of the operation.
  if not ir.MemRefType.isinstance(b.type):
    raise ValueError(f"B must be a memref, got: {b.type}")
  (k, n), element_type = mma_utils.tiled_memref_shape(b)
  if a_in_regs := isinstance(a, fa.FragmentedArray):
    m, k2 = a.shape
    element_type2 = a.mlir_dtype
    if a.mlir_dtype != ir.F16Type.get() and a.mlir_dtype != ir.BF16Type.get():
      raise ValueError(
          f"Only 16-bit dtypes supported for A in registers, got {a.mlir_dtype}"
      )
  elif ir.MemRefType.isinstance(a.type):
    (m, k2), element_type2 = mma_utils.tiled_memref_shape(a)
  else:
    raise ValueError(f"Unsupported A type: {type(a)}")
  if k != k2:
    raise ValueError(
        "WGMMA requires A and B to have the same contraction dimension (K),"
        f" got: {k2} and {k}"
    )
  if element_type != element_type2:
    raise ValueError(
        "WGMMA requires A and B to have the same element type, got:"
        f" {element_type2} and {element_type}"
    )
  if acc.value.shape != (m, n):
    raise ValueError(
        f"Accumulator shape mismatch: expected {(m, n)}, got {acc.value.shape}"
    )
  f32 = ir.F32Type.get()
  f16 = ir.F16Type.get()
  i32 = ir.IntegerType.get_signless(32)
  i8 = ir.IntegerType.get_signless(8)
  if element_type == f32 or element_type == ir.BF16Type.get():
    if acc.value.mlir_dtype != f32:
      raise ValueError(
          f"WGMMA with element type {element_type} only supports accumulators"
          f" of type f32, but got: {acc.value.mlir_dtype}"
      )
  elif any(
      t.isinstance(element_type)
      for t in {ir.F16Type, ir.Float8E5M2Type, ir.Float8E4M3FNType}
  ):
    if acc.value.mlir_dtype != f16 and acc.value.mlir_dtype != f32:
      raise ValueError(
          f"WGMMA with element type {element_type} only supports accumulators "
          f"of type f32 or f16, but got: {acc.value.mlir_dtype}"
      )
  elif element_type == i8:
    if a_in_regs and not a.is_signed:
      raise NotImplementedError("WGMMA with lhs of type u8")
    if acc.value.mlir_dtype != i32 or not acc.value.is_signed:
      raise ValueError(
          f"WGMMA with element type {element_type} only supports accumulators "
          f"of type s32, but got: {acc.value.mlir_dtype}"
      )
  else:
    raise NotImplementedError(f"Unsupported element type: {element_type}")

  # Step 2. Decide on the instruction shapes we'll use. Note that with swizzles,
  # instructions must be issued in groups of the same width as the swizzle.
  m_group_elems = 64  # Hopper has a fixed M instruction shape.
  k_group_elems = swizzle // utils.bytewidth(element_type)
  if n > 256 or n % 8:
    raise ValueError(f"N must be a multiple of 8 and <= 256, got: {n}")
  n_group_elems = n  # We assume only one N group below.
  if m % m_group_elems:
    raise ValueError(f"M must be a multiple of {m_group_elems}, got: {m}")
  if k % k_group_elems:
    raise ValueError(f"K must be a multiple of {k_group_elems}, got: {k}")
  m_groups = m // m_group_elems
  k_groups = k // k_group_elems
  # TODO(apaszke): Require users to bitcast input refs to tf32 before WGMMA.
  wgmma_element_type = (
      ir.FloatTF32Type.get() if element_type == ir.F32Type.get() else element_type
  )

  # Step 3. Compute the operand descriptors.
  if a_in_regs:
    a_desc_base = a_m_group_stride = a_k_group_stride = None
    a_instr_params = dict(a_transpose=None, a_k_stride=None)
  else:
    (
        (a_desc_base, a_k_instr_stride),
        (a_m_group_stride, a_k_group_stride),
        a_fastest,
    ) = mma_utils.create_descriptor(
        a,
        swizzle=swizzle,
        large_tile=(m_group_elems, k_group_elems),
        group_size=(m_group_elems, k_group_elems),
        logical_k_major=False,
    )
    a_instr_params = dict(a_transpose=a_fastest != mma_utils.Dim.K,
                          a_k_stride=a_k_instr_stride)
  (
      (b_desc_base, b_k_instr_stride),
      (b_n_group_stride, b_k_group_stride),
      b_fastest,
  ) = mma_utils.create_descriptor(
      b,
      swizzle=swizzle,
      large_tile=(k_group_elems,) * 2,  # It's not a typo that we use k for n.
      group_size=(k_group_elems, n_group_elems),
      logical_k_major=True,
  )
  del b_n_group_stride  # We only support one N group.

  # Step 4. Issue the instructions.
  if a_in_regs:
    a = wgmma_fence(a)  # Make sure the registers are ready.

  i64 = ir.IntegerType.get_signless(64)
  new_acc_regs = acc.value.registers.copy()
  for mi in range(m_groups):
    for ki in range(k_groups):
      if a_in_regs:
        a_mk = a[
            mi * m_group_elems : (mi + 1) * m_group_elems,
            ki * k_group_elems : (ki + 1) * k_group_elems,
        ]
      else:
        a_group_offset = mi * a_m_group_stride + ki * a_k_group_stride
        a_mk = _llvm_add(
            a_desc_base, c(mma_utils.encode_addr(a_group_offset), i64),
        )
      b_k = _llvm_add(
          b_desc_base, c(mma_utils.encode_addr(ki * b_k_group_stride), i64)
      )
      new_acc_regs[mi : mi + 1] = wgmma_m64(
          new_acc_regs[mi : mi + 1],
          a_mk,
          b_k,
          swizzle=swizzle,
          n=n_group_elems,
          element_type=wgmma_element_type,
          b_transpose=b_fastest != mma_utils.Dim.K,
          b_k_stride=b_k_instr_stride,
          **a_instr_params,
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


def _llvm_add(x, y):
  return llvm.add(x, y, overflow_flags=llvm.IntegerOverflowFlags.none)


def _unpack_i32(vec_ty, r):
  i32 = ir.IntegerType.get_signless(32)
  return vector.bitcast(
      vec_ty, vector.splat(ir.VectorType.get((1,), i32), r)
  )
