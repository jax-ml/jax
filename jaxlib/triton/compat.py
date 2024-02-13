# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compatibility layer on top of Triton Python APIs."""

# TODO(slebedev): Enable type checking.
# mypy: ignore-errors

from __future__ import annotations

from collections.abc import Sequence
import functools
import threading
from typing import Any, Literal

from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith as arith_dialect
from jaxlib.mlir.dialects import math as math_dialect
import numpy as np
import triton.backends.nvidia.compiler as cb
import triton.language as tl

from . import dialect as tt_dialect


PointerType = tt_dialect.PointerType


_tls = threading.local()


def new_ir_context() -> ir.Context:
  ctx = ir.Context()
  tt_dialect.register_dialect(ctx)
  ctx.load_all_available_dialects()
  return ctx


class builder:

  @classmethod
  @property
  def current(cls) -> "builder":
    return _tls.builder

  def __init__(self, cuda_options: cb.CUDAOptions):
    self.context = new_ir_context()
    self.loc = ir.Location.unknown(self.context)
    self.options = cuda_options

  def __enter__(self):
    _tls.builder = self
    self.context.__enter__()
    self.loc.__enter__()
    return self

  def __exit__(self, *exc_info):
    self.loc.__exit__(*exc_info)
    self.context.__exit__(*exc_info)
    del _tls.builder

  def get_void_ty(self) -> ir.Type:
    return ir.NoneType.get()

  def get_int1_ty(self) -> ir.Type:
    return ir.IntegerType.get_signless(1)

  def get_int8_ty(self) -> ir.Type:
    return ir.IntegerType.get_signless(8)

  def get_int16_ty(self) -> ir.Type:
    return ir.IntegerType.get_signless(16)

  def get_int32_ty(self) -> ir.Type:
    return ir.IntegerType.get_signless(32)

  def get_int64_ty(self) -> ir.Type:
    return ir.IntegerType.get_signless(64)

  def get_fp8e4nv_ty(self) -> ir.Type:
    return ir.Float8E4M3FNUZType.get()

  def get_fp8e4b15_ty(self) -> ir.Type:
    return ir.Float8E4M3B11FNUZType.get()

  def get_fp8e4b15x4_ty(self) -> ir.Type:
    return ir.Float8E4M3FNType.get()

  def get_fp8e5_ty(self) -> ir.Type:
    return ir.Float8E5M2Type.get()

  def get_half_ty(self) -> ir.Type:
    return ir.F16Type.get()

  def get_bf16_ty(self) -> ir.Type:
    return ir.BF16Type.get()

  def get_float_ty(self) -> ir.Type:
    return ir.F32Type.get()

  def get_double_ty(self) -> ir.Type:
    return ir.F64Type.get()

  def get_ptr_ty(self, t: ir.Type, addr_space: int) -> ir.Type:
    return PointerType.get(t, addr_space)

  def get_block_ty(
      self, t: ir.Type, shape: Sequence[int]
  ) -> ir.RankedTensorType:
    return ir.RankedTensorType.get(shape, t)

  def get_function_ty(
      self, in_types: Sequence[ir.Type], out_types: Sequence[ir.Type]
  ) -> type[ir.FunctionType]:
    return ir.FunctionType.get(in_types, out_types)


# TODO(slebedev): Consider moving upstream.
_FLOAT_WIDTH = {
    ir.Float8E4M3FNUZType: 8,
    ir.Float8E4M3FNType: 8,
    ir.Float8E4M3B11FNUZType: 8,
    ir.Float8E5M2Type: 8,
    ir.BF16Type: 16,
    ir.F16Type: 16,
    ir.F32Type: 32,
    ir.F64Type: 64,
}
_FLOAT_TYPES = tuple(_FLOAT_WIDTH)


class FloatTypeMeta(type):

  def __instancecheck__(cls, instance: object) -> bool:
    return isinstance(instance, _FLOAT_TYPES)

  def __subclasscheck__(cls, subclass: type[object]) -> bool:
    return issubclass(subclass, _FLOAT_TYPES)


class FloatType(metaclass=FloatTypeMeta):
  """Fake base class for MLIR floating point types."""

  def __init__(self, type: ir.Type):
    assert isinstance(type, _FLOAT_TYPES)
    self.type = type

  @property
  def is_standard(self) -> bool:
    return isinstance(
        self.type, (ir.BF16Type, ir.F16Type, ir.F32Type, ir.F64Type)
    )

  @property
  def width(self) -> int:
    return _FLOAT_WIDTH[type(self.type)]


dtype = tl.core.dtype


class block_type(tl.core.block_type):
  def __init__(self, element_ty: dtype, shape: list[Any]) -> Any:
    super().__init__(element_ty, shape)
    self.shape = tuple(self.shape)


pointer_type = tl.core.pointer_type

void = tl.core.void
bfloat16 = tl.core.bfloat16
float16 = tl.core.float16
float32 = tl.core.float32
float64 = tl.core.float64
int1 = tl.core.int1
int8 = tl.core.int8
int32 = tl.core.int32
int64 = tl.core.int64
uint32 = tl.core.uint32
uint64 = tl.core.uint64


def _bool_block_like(v: tensor) -> dtype:
  if not v.type.is_block():
    return int1
  return block_type(int1, v.shape)


def _to_tensor(x: object, dtype: dtype | None = None) -> "tensor":
  if dtype is not None and isinstance(x, (bool, int, float)):
    return tensor(
        arith_dialect.constant(dtype.to_ir(builder.current), x), dtype
    )

  # We follow Triton conversion logic here, but it might better to use
  # mlir.ir_constant instead.
  #
  # Note that Triton uses singless integers for both int* and uint* types.
  if isinstance(x, bool):
    return tensor(
        arith_dialect.constant(ir.IntegerType.get_signless(1), x), int1
    )
  elif isinstance(x, int):
    if -(2**31) <= x < 2**31:
      return tensor(
          arith_dialect.constant(ir.IntegerType.get_signless(32), x), int32
      )
    elif 2**31 <= x < 2**32:
      return tensor(
          arith_dialect.constant(ir.IntegerType.get_signless(32), x), uint32
      )
    elif -(2**63) <= x < 2**63:
      return tensor(
          arith_dialect.constant(ir.IntegerType.get_signless(64), x), int64
      )
    elif 2**63 <= x < 2**64:
      return tensor(
          arith_dialect.constant(ir.IntegerType.get_signless(64), x), uint64
      )
    else:
      raise ValueError(f"integer overflow representing {x}")
  elif isinstance(x, float):
    fi = np.finfo(np.float32)
    if np.isinf(x) or np.isnan(x) or x == 0 or -fi.min <= x <= fi.max:
      return tensor(arith_dialect.constant(ir.F32Type.get(), x), float32)
    else:
      return tensor(arith_dialect.constant(ir.F64Type.get(), x), float64)
  else:
    raise ValueError(f"cannot convert {x} to tensor")


class tensor:

  def __init__(self, handle: ir.Value, type: dtype):
    self.handle = handle
    self.shape = tuple(type.shape) if type.is_block() else ()
    self.type = type
    self.dtype = type.scalar

  def __str__(self) -> str:
    return f"{self.dtype}[{', '.join(map(str, self.shape))}]"


def _program_id(axis: int) -> ir.Value:
  if axis not in range(3):
    raise ValueError(f"axis must be in [0, 3), but got: {axis}")
  return tt_dialect.get_program_id(axis)


def program_id(axis: int) -> tensor:
  if axis not in range(3):
    raise ValueError(f"axis must be in [0, 3), but got: {axis}")
  return tensor(tt_dialect.get_program_id(axis), int32)


_STR_TO_EVICTION_POLICY = {str(e): e for e in tt_dialect.EvictionPolicy}
_STR_TO_CACHE_MODIFIER = {str(c): c for c in tt_dialect.CacheModifier}


def _infer_load_return_type(ptr: ir.Value) -> ir.Type:
  if ir.RankedTensorType.isinstance(ptr.type):
    ptr_type = ir.RankedTensorType(ptr.type)
    element_type = PointerType(ptr_type.element_type)
    return ir.RankedTensorType.get(
        ptr_type.shape,
        element_type.pointee_type,
        ptr_type.encoding,
    )
  else:
    ptr_type = PointerType(ptr.type)
    return ptr_type.pointee_type


def _load(
    ptr: ir.Value,
    mask: ir.Value | None = None,
    other: ir.Value | None = None,
    *,
    cache_modifier: str | None = None,
    eviction_policy: str | None = None,
    is_volatile: bool = False,
) -> ir.Value:
  if cache_modifier is None:
    cache_modifier = tt_dialect.CacheModifier.NONE
  elif cache_modifier == ".ca" or cache_modifier == ".cg":
    cache_modifier = _STR_TO_CACHE_MODIFIER[cache_modifier]
  else:
    raise ValueError(f"unsupported cache modifier: {cache_modifier}")
  if eviction_policy is None:
    eviction_policy = tt_dialect.EvictionPolicy.NORMAL
  else:
    try:
      eviction_policy = _STR_TO_EVICTION_POLICY[eviction_policy]
    except KeyError:
      raise ValueError(
          f"unsupported eviction policy: {eviction_policy}"
      ) from None

  if PointerType.isinstance(ptr.type):
    ptr_type = PointerType(ptr.type)
    if ir.RankedTensorType.isinstance(ptr_type.pointee_type):
      raise NotImplementedError("loading from a block pointer is not supported")

  ptr_type = _element_type(ptr.type)
  if not PointerType.isinstance(ptr_type):
    raise ValueError(f"unsupported pointer type: {ptr_type}")
  ptr_type = PointerType(ptr_type)
  if other is not None and mask is None:
    raise ValueError("other requires mask to be provided")
  if not ir.RankedTensorType.isinstance(ptr.type):
    if other is not None and ir.RankedTensorType.isinstance(other.type):
      raise ValueError("other cannot be a block if pointer is not a block")
    if mask is not None and ir.RankedTensorType.isinstance(mask.type):
      raise ValueError("mask cannot be a block if pointer is not a block")

  pointee_type = ptr_type.pointee_type
  is_int1 = isinstance(pointee_type, ir.IntegerType) and pointee_type.width == 1
  if is_int1:
    pointee_type = ir.IntegerType.get_signless(8)
    ptr = _cast(ptr, PointerType.get(pointee_type, ptr_type.address_space))

  if other is not None:
    other = _cast(other, pointee_type)

  result = tt_dialect.load(
      _infer_load_return_type(ptr),
      ptr,
      mask=mask,
      other=other,
      cache=cache_modifier,
      evict=eviction_policy,
      is_volatile=is_volatile,
  )
  return (
      result if not is_int1 else _cast(result, ir.IntegerType.get_signless(1))
  )


def load(
    ptr: tensor,
    mask: tensor | None = None,
    other: tensor | None = None,
    *,
    cache_modifier: str | None = None,
    eviction_policy: str | None = None,
    is_volatile: bool = False,
) -> tensor:
  element_type = ptr.dtype.element_ty
  result_handle = _load(
      ptr.handle,
      mask.handle if mask is not None else None,
      other.handle if other is not None else None,
      cache_modifier=cache_modifier,
      eviction_policy=eviction_policy,
      is_volatile=is_volatile,
  )
  if ptr.type.is_block():
    return tensor(result_handle, block_type(element_type, ptr.type.shape))
  else:
    return tensor(result_handle, element_type)


def _store(
    ptr: ir.Value,
    value: ir.Value,
    mask: ir.Value | None = None,
    *,
    cache_modifier: str | None = None,
    eviction_policy: str | None = None,
) -> ir.Value:
  if cache_modifier is None:
    cache_modifier = tt_dialect.CacheModifier.NONE
  elif cache_modifier != ".ca":
    cache_modifier = _STR_TO_CACHE_MODIFIER[cache_modifier]
  else:
    raise ValueError(f"unsupported cache modifier: {cache_modifier}")
  if eviction_policy is None:
    eviction_policy = tt_dialect.EvictionPolicy.NORMAL
  else:
    try:
      eviction_policy = _STR_TO_EVICTION_POLICY[eviction_policy]
    except KeyError:
      raise ValueError(
          f"unsupported eviction policy: {eviction_policy}"
      ) from None

  if PointerType.isinstance(ptr.type):
    ptr_type = PointerType(ptr.type)
    if ir.RankedTensorType.isinstance(ptr_type.pointee_type):
      raise NotImplementedError("loading from a block pointer is not supported")

  ptr_type = _element_type(ptr.type)
  if not PointerType.isinstance(ptr_type):
    raise ValueError(f"unsupported pointer type: {ptr_type}")
  ptr_type = PointerType(ptr_type)
  if not ir.RankedTensorType.isinstance(ptr.type):
    if ir.RankedTensorType.isinstance(value.type):
      raise ValueError("value cannot be a block if pointer is not a block")
    if mask is not None and ir.RankedTensorType.isinstance(mask.type):
      raise ValueError("mask cannot be a block if pointer is not a block")

  pointee_type = ptr_type.pointee_type
  if isinstance(pointee_type, ir.IntegerType) and pointee_type.width == 1:
    pointee_type = ir.IntegerType.get_signless(8)
    ptr = _cast(ptr, PointerType.get(pointee_type, ptr_type.address_space))

  value = _cast(value, pointee_type)
  return tt_dialect.store(
      ptr, value, mask=mask, cache=cache_modifier, evict=eviction_policy
  )


def store(
    ptr: tensor,
    value: tensor,
    mask: tensor | None = None,
    *,
    cache_modifier: str | None = None,
    eviction_policy: str | None = None,
) -> tensor:
  return tensor(
      _store(
          ptr.handle,
          value.handle,
          mask.handle if mask is not None else None,
          cache_modifier=cache_modifier,
          eviction_policy=eviction_policy,
      ),
      void,
  )


def _make_range(start: int, end: int) -> ir.Value:
  if end <= start:
    raise ValueError(
        f"end must be greater than start, but got: {end} <= {start}"
    )
  if max(start, end) >= 2**32:
    raise ValueError("start and end must fit in int32")
  return tt_dialect.make_range(
      ir.RankedTensorType.get([end - start], ir.IntegerType.get_signless(32)),
      start,
      end,
  )


def arange(start: int, end: int) -> tensor:
  handle = _make_range(start, end)
  return tensor(handle, block_type(int32, [end - start]))


def _broadcast_to(x: ir.Value, shape: Sequence[int]) -> ir.Value:
  if not ir.RankedTensorType.isinstance(x.type):
    return _splat(x, shape)
  ty = ir.RankedTensorType(x.type)
  if tuple(ty.shape) == shape:
    return x
  return tt_dialect.broadcast(
      ir.RankedTensorType.get(shape, ty.element_type, ty.encoding), x
  )


def broadcast_to(x: tensor, shape: Sequence[int]) -> tensor:
  if not shape:
    return x
  return tensor(_broadcast_to(x.handle, shape), block_type(x.dtype, shape))


def _splat(x: ir.Value, shape: Sequence[int]) -> ir.Value:
  if ir.RankedTensorType.isinstance(x.type):
    raise TypeError("cannot splat a ranked tensor")
  if not shape:
    return x
  return tt_dialect.splat(ir.RankedTensorType.get(shape, x.type), x)


def _expand_dims(x: ir.Value, axis: int) -> ir.Value:
  if not ir.RankedTensorType.isinstance(x.type):
    shape = list(ir.RankedTensorType(x.type).shape)
    shape.insert(axis, 1)
    return _splat(x, shape)
  return tt_dialect.expand_dims(x, axis)


def expand_dims(x: tensor, axis: int) -> tensor:
  dst_shape = list(x.shape)
  dst_shape.insert(axis, 1)
  return tensor(_expand_dims(x.handle, axis), block_type(x.dtype, dst_shape))


def _reshape(x: ir.Value, shape: Sequence[int]) -> ir.Value:
  ty = ir.RankedTensorType(x.type)
  return tt_dialect.reshape(
      ir.RankedTensorType.get(shape, ty.element_type, ty.encoding),
      x,
      allow_reorder=False,
  )


def _check_dot_operands(
    x_type: ir.RankedTensorType, y_type: ir.RankedTensorType, options: Any
):
  # TODO(slebedev): Ensure that the dtypes are supported by CUDA.
  return


def _dot(
    x: ir.Value,
    y: ir.Value,
    acc: ir.Value | None = None,
    *,
    allow_tf32: bool = True,
    max_num_imprecise_acc: int | None = None,
    out_type: ir.Type | None = None,
) -> ir.Value:
  if out_type is None:
    out_type = ir.F32Type.get()
  elif isinstance(out_type, ir.BF16Type):
    raise NotImplementedError(f"unsupported output type: {out_type}")

  x_type = ir.RankedTensorType(x.type)
  y_type = ir.RankedTensorType(y.type)
  if min(*x_type.shape, *y_type.shape) < 16:
    raise ValueError("all dimensions of x and y must be >= 16 ")
  if x_type.element_type != y_type.element_type:
    raise ValueError(
        "x and y must have the same element type, but got:"
        f" {x_type.element_type} and {y_type.element_type}"
    )

  _check_dot_operands(x_type, y_type, object())

  element_type = x_type.element_type
  if isinstance(element_type, ir.IntegerType):
    if element_type.width != 8:
      raise TypeError(f"unsupported element type: {element_type}")
    element_type = ir.IntegerType.get_signless(32)
  elif isinstance(element_type, (ir.F32Type, ir.BF16Type)):
    element_type = ir.F32Type.get()
  else:
    element_type = out_type

  if element_type != out_type:
    raise TypeError(
        f"output type {out_type} does not match element type {element_type}"
    )

  m, _ = x_type.shape
  _, n = y_type.shape

  if acc is None:
    acc = _full(ir.RankedTensorType.get([m, n], element_type), 0)

  if max_num_imprecise_acc is None:
    if (
        FloatType(x_type.element_type).width == 8
        and FloatType(y_type.element_type).width == 8
    ):
      # TODO(slebedev): Fill in from options.
      raise NotImplementedError
    else:
      max_num_imprecise_acc = 0

  return tt_dialect.dot(x, y, acc, allow_tf32, max_num_imprecise_acc)


def dot(
    x: tensor,
    y: tensor,
    acc: tensor | None = None,
    allow_tf32: bool = True,
    max_num_imprecise_acc: int | None = None,
    out_dtype: dtype = float32,
) -> tensor:
  if x.dtype.is_int():
    if x.dtype != int8:
      raise TypeError(f"unsupported dtype: {x.dtype}")
    element_type = int32
  elif x.dtype.is_fp32() or x.dtype.is_bf16():
    element_type = float32
  else:
    element_type = out_dtype

  m, _ = x.shape
  _, n = y.shape
  result_type = block_type(element_type, [m, n])

  return tensor(
      _dot(
          x.handle,
          y.handle,
          acc.handle if acc is not None else None,
          allow_tf32=allow_tf32,
          max_num_imprecise_acc=max_num_imprecise_acc,
          out_type=out_dtype.to_ir(builder.current),
      ),
      result_type,
  )


def abs(x: tensor) -> tensor:
  dtype = x.dtype
  if dtype.is_floating():
    return tensor(math_dialect.absf(x.handle), x.type)
  elif dtype.is_int_signed():
    return tensor(math_dialect.absi(x.handle), x.type)
  elif dtype.is_int_unsigned():
    return x
  else:
    raise ValueError(f"unsupported dtype: {dtype}")


def multiple_of(x: tensor, values: Sequence[int]) -> tensor:
  assert max(1, len(x.shape)) == len(values)
  set_attr(
      x.handle,
      "tt.divisibility",
      ir.DenseIntElementsAttr.get(
          np.fromiter(map(int, values), dtype=np.uint32)
      ),
  )
  return x


def max_contiguous(x: tensor, values: Sequence[int]) -> tensor:
  assert len(x.shape) == len(values)
  set_attr(
      x.handle,
      "tt.contiguity",
      ir.DenseIntElementsAttr.get(
          np.fromiter(map(int, values), dtype=np.uint32)
      ),
  )
  return x


def set_attr(v: ir.Value, name: str, attr: ir.Attribute) -> None:
  if not ir.BlockArgument.isinstance(v):
    v.owner.attributes[name] = attr
    return

  arg = ir.BlockArgument(v)
  name += f"_arg{arg.arg_number}"
  owner = arg.owner
  is_entry = owner.region.blocks[0] == owner
  if not is_entry:
    return
  if (op := owner.owner.operation) and not isinstance(op, tt_dialect.FuncOp):
    op.attributes[name] = attr


def _element_type(t: ir.Type) -> ir.Type:
  if ir.RankedTensorType.isinstance(t):
    return ir.RankedTensorType(t).element_type
  else:
    return t


def _full(t: ir.Type, v: object) -> ir.Type:
  element_type = _element_type(t)
  if isinstance(element_type, ir.IntegerType):
    result = arith_dialect.constant(element_type, int(v))
  elif isinstance(element_type, FloatType):
    result = arith_dialect.constant(element_type, float(v))
  else:
    raise NotImplementedError

  if ir.RankedTensorType.isinstance(t):
    return tt_dialect.splat(t, result)
  else:
    return result


_UI_PREDICATES = {
    "==": arith_dialect.CmpIPredicate.eq,
    "!=": arith_dialect.CmpIPredicate.ne,
    "<": arith_dialect.CmpIPredicate.ult,
    "<=": arith_dialect.CmpIPredicate.ule,
    ">": arith_dialect.CmpIPredicate.ugt,
    ">=": arith_dialect.CmpIPredicate.uge,
}
_SI_PREDICATES = {
    "==": arith_dialect.CmpIPredicate.eq,
    "!=": arith_dialect.CmpIPredicate.ne,
    "<": arith_dialect.CmpIPredicate.slt,
    "<=": arith_dialect.CmpIPredicate.sle,
    ">": arith_dialect.CmpIPredicate.sgt,
    ">=": arith_dialect.CmpIPredicate.sge,
}
_F_PREDICATES = {
    "==": arith_dialect.CmpFPredicate.OEQ,
    "!=": arith_dialect.CmpFPredicate.UNE,
    "<": arith_dialect.CmpFPredicate.OLT,
    "<=": arith_dialect.CmpFPredicate.OLE,
    ">": arith_dialect.CmpFPredicate.OGT,
    ">=": arith_dialect.CmpFPredicate.OGE,
}


def _cmp(
    x: ir.Value, y: ir.Value, p: Literal["==", "!=", "<", "<=", ">", ">="]
) -> ir.Value:
  assert x.type == y.type
  x_element_type = _element_type(x.type)
  if isinstance(x_element_type, ir.IntegerType):
    if x_element_type.is_signed:
      op = _SI_PREDICATES[p]
    else:
      op = _UI_PREDICATES[p]
    return arith_dialect.cmpi(op, x, y)
  elif isinstance(x_element_type, FloatType):
    return arith_dialect.cmpf(_F_PREDICATES[p], x, y)
  else:
    raise NotImplementedError(f"unsupported types: {x.type} and {y.type}")


_equal = functools.partial(_cmp, p="==")
_not_equal = functools.partial(_cmp, p="!=")
_less_than = functools.partial(_cmp, p="<")
_less_equal = functools.partial(_cmp, p="<=")
_greater_than = functools.partial(_cmp, p=">")
_greater_equal = functools.partial(_cmp, p=">=")


def _float_float_cast(src: ir.Value, dst_type: ir.Type) -> ir.Value:
  src_element_type = FloatType(_element_type(src.type))
  dst_element_type = FloatType(_element_type(dst_type))
  if src_element_type.width == 8 or dst_element_type.width == 8:
    return tt_dialect.fp_to_fp(
        dst_type,
        src,
        rounding=tt_dialect.RoundingMode.RTNE,
    )
  if src_element_type.width > dst_element_type.width:
    return arith_dialect.truncf(dst_type, src)
  elif src_element_type.width < dst_element_type.width:
    return arith_dialect.extf(dst_type, src)
  else:
    raise NotImplementedError


def _int_int_cast(src: ir.Value, dst_type: ir.Type) -> ir.Value:
  src_element_type = ir.IntegerType(_element_type(src.type))
  dst_element_type = ir.IntegerType(_element_type(dst_type))
  assert src_element_type != dst_element_type
  if dst_element_type.width == 1:
    return _not_equal(src, _full(src.type, 0))

  is_signed = src_element_type.is_signed and src_element_type.width != 1
  if src_element_type.width == dst_element_type.width:
    return arith_dialect.bitcast(dst_type, src)
  elif src_element_type.width > dst_element_type.width:
    return arith_dialect.trunci(dst_type, src)
  elif is_signed:
    return arith_dialect.extsi(dst_type, src)
  else:
    return arith_dialect.extui(dst_type, src)


def _float_int_cast(src: ir.Value, dst_type: ir.Type) -> ir.Value:
  src_element_type = FloatType(_element_type(src.type))
  if not src_element_type.is_standard:
    raise NotImplementedError(f"cannot cast {src} tp {dst_type}")
  dst_element_type = ir.IntegerType(_element_type(dst_type))
  if dst_element_type.width == 1:
    return _not_equal(src, _full(src.type, 0))
  elif dst_element_type.is_signed:
    return arith_dialect.fptosi(dst_type, src)
  else:
    return arith_dialect.fptoui(dst_type, src)


def _int_float_cast(src: ir.Value, dst_type: ir.Type) -> ir.Value:
  src_element_type = ir.IntegerType(_element_type(src.type))
  dst_element_type = FloatType(_element_type(dst_type))
  if not dst_element_type.is_standard:
    raise NotImplementedError(f"cannot cast {src} tp {dst_type}")
  if src_element_type.width == 1 or not src_element_type.is_signed:
    return arith_dialect.uitofp(dst_type, src)
  else:
    return arith_dialect.sitofp(dst_type, src)


def _cast(src: ir.Value, dst_type: ir.Type) -> ir.Value:
  if ir.RankedTensorType.isinstance(
      src.type
  ) and not ir.RankedTensorType.isinstance(dst_type):
    src_type = ir.RankedTensorType(src.type)
    dst_type = ir.RankedTensorType.get(
        src_type.shape,
        dst_type,
        src_type.encoding,
    )
  if src.type == dst_type:
    return src

  src_element_type = _element_type(src.type)
  dst_element_type = _element_type(dst_type)
  if isinstance(src_element_type, ir.Float8E4M3FNUZType) or isinstance(
      dst_element_type, ir.Float8E4M3FNUZType
  ):
    # TODO(slebedev): Check the CUDA version and raise conditionally.
    raise NotImplementedError("cannot cast from or to float8_e4m3fnuz")

  if isinstance(src_element_type, (ir.F16Type, ir.BF16Type)) and not isinstance(
      dst_element_type, ir.F32Type
  ):
    return _cast(_cast(src, ir.F32Type.get()), dst_type)

  if isinstance(src_element_type, FloatType) and isinstance(
      dst_element_type, FloatType
  ):
    return _float_float_cast(src, dst_type)

  if isinstance(src_element_type, ir.IntegerType) and isinstance(
      dst_element_type, ir.IntegerType
  ):
    return _int_int_cast(src, dst_type)

  if isinstance(src_element_type, FloatType) and isinstance(
      dst_element_type, ir.IntegerType
  ):
    return _float_int_cast(src, dst_type)
  if isinstance(src_element_type, ir.IntegerType) and isinstance(
      dst_element_type, FloatType
  ):
    return _int_float_cast(src, dst_type)

  if PointerType.isinstance(src_element_type) and isinstance(
      dst_element_type, ir.IntegerType
  ):
    if dst_element_type.int_bitwidth == 64:
      return tt_dialect.ptr_to_int(dst_type, src)
    else:
      x = _cast(src, ir.IntegerType.get_signless(64))
      zero = _full(x.type, 0)
      return _cast(_not_equal(x, zero), dst_type)
  if isinstance(src_element_type, ir.IntegerType) and PointerType.isinstance(
      dst_element_type
  ):
    return tt_dialect.int_to_ptr(dst_type, src)
  if PointerType.isinstance(src_element_type) and PointerType.isinstance(
      dst_element_type
  ):
    return tt_dialect.bitcast(dst_type, src)

  raise NotImplementedError(f"cannot cast {src} to {dst_type}")


class semantic:

  @staticmethod
  def cast(x: tensor, dst_ty: dtype) -> tensor:
    return tensor(
        _cast(x.handle, dst_ty.to_ir(builder.current)),
        block_type(dst_ty, x.shape) if x.type.is_block() else dst_ty,
    )

  @staticmethod
  def permute(x: tensor, dims: Sequence[int]) -> tensor:
    if len(dims) != len(x.shape):
      raise ValueError("dims must have the same shape as x")
    if set(dims) != set(range(len(x.shape))):
      raise ValueError(f"dims must be a permutation of 0, ..., {len(x.shape)}-1")
    return tensor(
        tt_dialect.trans(x.handle, dims),
        block_type(x.dtype, [x.shape[i] for i in dims]),
    )

  @staticmethod
  def _minus(x: ir.Value) -> ir.Value:
    if PointerType.isinstance(_element_type(x.type)):
      raise NotImplementedError(f"unsupported dtype: {x.type}")
    return semantic._sub(_full(x.type, 0), x)

  @staticmethod
  def minus(x: tensor) -> tensor:
    return tensor(semantic._minus(x.handle), x.type)

  @staticmethod
  def _add(x: ir.Value, y: ir.Value) -> ir.Value:
    x_element_type = _element_type(x.type)
    y_element_type = _element_type(y.type)
    if PointerType.isinstance(y_element_type):
      assert not PointerType.isinstance(x_element_type)
      x, y = y, x
      x_element_type, y_element_type = y_element_type, x_element_type

    if PointerType.isinstance(x_element_type):
      return tt_dialect.addptr(x.type, x, y)

    assert x.type == y.type, (str(x.type), str(y.type))
    if isinstance(x_element_type, ir.IntegerType):
      return arith_dialect.addi(x, y)
    elif isinstance(x_element_type, FloatType):
      return arith_dialect.addf(x, y)
    else:
      raise NotImplementedError(f"unsupported dtypes: {x.type} and {y.type}")

  @staticmethod
  def add(x: tensor, y: tensor) -> tensor:
    return tensor(semantic._add(x.handle, y.handle), x.type)

  @staticmethod
  def _sub(x: ir.Value, y: ir.Value) -> ir.Value:
    x_element_type = _element_type(x.type)
    y_element_type = _element_type(y.type)
    if PointerType.isinstance(x_element_type):
      return tt_dialect.addptr(x.type, x, semantic._minus(y))
    elif not PointerType.isinstance(y_element_type):
      assert x.type == y.type
      if isinstance(x_element_type, ir.IntegerType):
        return arith_dialect.subi(x, y)
      elif isinstance(x_element_type, FloatType):
        return arith_dialect.subf(x, y)
    raise NotImplementedError(f"unsupported dtype: {y.type}")

  @staticmethod
  def sub(x: tensor, y: tensor) -> tensor:
    return tensor(semantic._sub(x.handle, y.handle), x.type)

  @staticmethod
  def _mul(x: ir.Value, y: ir.Value) -> ir.Value:
    assert x.type == y.type
    x_element_type = _element_type(x.type)
    if isinstance(x_element_type, ir.IntegerType):
      return arith_dialect.muli(x, y)
    elif isinstance(x_element_type, FloatType):
      return arith_dialect.mulf(x, y)
    raise NotImplementedError(f"unsupported types: {x.type} and {y.type}")

  @staticmethod
  def mul(x: tensor, y: tensor) -> tensor:
    return tensor(semantic._mul(x.handle, y.handle), x.type)

  @staticmethod
  def _floordiv(x: ir.Value, y: ir.Value) -> ir.Value:
    assert x.type == y.type
    x_element_type = _element_type(x.type)
    if not isinstance(x_element_type, ir.IntegerType):
      raise NotImplementedError(f"unsupported types: {x.type} and {y.type}")
    if x_element_type.is_signed:
      return arith_dialect.divsi(x, y)
    else:
      return arith_dialect.divui(x, y)

  @staticmethod
  def floordiv(x: tensor, y: tensor) -> tensor:
    return tensor(semantic._floordiv(x.handle, y.handle), x.type)

  @staticmethod
  def _truediv(x: ir.Value, y: ir.Value) -> ir.Value:
    assert x.type == y.type
    x_element_type = _element_type(x.type)
    if isinstance(x_element_type, ir.IntegerType):
      x_element_type = ir.F32Type.get()
      x = _int_float_cast(x, x_element_type)
      y = _int_float_cast(y, x_element_type)
    if isinstance(x_element_type, FloatType):
      return arith_dialect.divf(x, y)
    raise NotImplementedError(f"unsupported types: {x.type} and {y.type}")

  @staticmethod
  def truediv(x: tensor, y: tensor) -> tensor:
    return tensor(semantic._truediv(x.handle, y.handle), x.type)

  @staticmethod
  def _mod(x: ir.Value, y: ir.Value) -> ir.Value:
    assert x.type == y.type
    x_element_type = _element_type(x.type)
    if not isinstance(x_element_type, ir.IntegerType):
      raise NotImplementedError(f"unsupported types: {x.type} and {y.type}")
    if x_element_type.is_signed:
      return arith_dialect.remsi(x, y)
    else:
      return arith_dialect.remui(x, y)

  @staticmethod
  def mod(x: tensor, y: tensor) -> tensor:
    return tensor(semantic._mod(x.handle, y.handle), x.type)

  @staticmethod
  def invert(x: tensor) -> tensor:
    if not x.dtype.is_int():
      raise NotImplementedError(f"unsupported dtype: {x.dtype}")
    one = tensor(
        _full(x.type.to_ir(builder.current), 0xFFFFFFFFFFFFFFFF), x.type
    )
    return semantic.xor_(x, one)

  @staticmethod
  def and_(x: tensor, y: tensor) -> tensor:
    assert x.shape == y.shape and x.dtype == y.dtype
    return tensor(arith_dialect.andi(x.handle, y.handle), x.type)

  @staticmethod
  def or_(x: tensor, y: tensor) -> tensor:
    assert x.shape == y.shape and x.dtype == y.dtype
    return tensor(arith_dialect.ori(x.handle, y.handle), x.type)

  @staticmethod
  def xor_(x: tensor, y: tensor) -> tensor:
    assert x.shape == y.shape and x.dtype == y.dtype
    return tensor(arith_dialect.xori(x.handle, y.handle), x.type)

  @staticmethod
  def lshr(x: tensor, y: tensor) -> tensor:
    assert x.shape == y.shape and x.dtype == y.dtype
    return tensor(arith_dialect.shrui(x.handle, y.handle), x.type)

  @staticmethod
  def ashr(x: tensor, y: tensor) -> tensor:
    assert x.shape == y.shape and x.dtype == y.dtype
    return tensor(arith_dialect.shrsi(x.handle, y.handle), x.type)

  @staticmethod
  def shl(x: tensor, y: tensor) -> tensor:
    assert x.shape == y.shape and x.dtype == y.dtype
    return tensor(arith_dialect.shli(x.handle, y.handle), x.type)

  @staticmethod
  def equal(x: tensor, y: tensor) -> tensor:
    return tensor(_equal(x.handle, y.handle), _bool_block_like(x))

  @staticmethod
  def not_equal(x: tensor, y: tensor) -> tensor:
    return tensor(_not_equal(x.handle, y.handle), _bool_block_like(x))

  @staticmethod
  def greater_than(x: tensor, y: tensor) -> tensor:
    return tensor(_greater_than(x.handle, y.handle), _bool_block_like(x))

  @staticmethod
  def greater_equal(x: tensor, y: tensor) -> tensor:
    return tensor(_greater_equal(x.handle, y.handle), _bool_block_like(x))

  @staticmethod
  def less_than(x: tensor, y: tensor) -> tensor:
    return tensor(_less_than(x.handle, y.handle), _bool_block_like(x))

  @staticmethod
  def less_equal(x: tensor, y: tensor) -> tensor:
    return tensor(_less_equal(x.handle, y.handle), _bool_block_like(x))
