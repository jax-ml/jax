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

from collections.abc import Mapping, Sequence
import threading
from typing import Any

from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith as arith_dialect
from jaxlib.mlir.dialects import math as math_dialect
import numpy as np
import triton.backends.nvidia.compiler as cb
import triton.language as tl

from . import dialect as tt_dialect


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
    return tt_dialect.PointerType.get(t, addr_space)

  def get_block_ty(
      self, t: ir.Type, shape: Sequence[int]
  ) -> ir.RankedTensorType:
    return ir.RankedTensorType.get(shape, t)

  def get_function_ty(
      self, in_types: Sequence[ir.Type], out_types: Sequence[ir.Type]
  ) -> type[ir.FunctionType]:
    return ir.FunctionType.get(in_types, out_types)


_FLOAT_TYPES = (
    ir.Float8E4M3FNUZType,
    ir.Float8E4M3FNType,
    ir.Float8E4M3B11FNUZType,
    ir.Float8E5M2Type,
    ir.BF16Type,
    ir.F16Type,
    ir.F32Type,
    ir.F64Type,
)

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


def program_id(axis: int) -> tensor:
  if axis not in range(3):
    raise ValueError(f"axis must be in [0, 3), but got: {axis}")
  return tensor(tt_dialect.get_program_id(axis), int32)


_STR_TO_EVICTION_POLICY = {str(e): e for e in tt_dialect.EvictionPolicy}
_STR_TO_CACHE_MODIFIER = {str(c): c for c in tt_dialect.CacheModifier}


def _infer_load_return_type(ptr: ir.Value) -> ir.Type:
  if ir.RankedTensorType.isinstance(ptr.type):
    ptr_type = ir.RankedTensorType(ptr.type)
    element_type = tt_dialect.PointerType(ptr_type.element_type)
    return ir.RankedTensorType.get(
        ptr_type.shape,
        element_type.pointee_type,
        ptr_type.encoding,
    )
  else:
    ptr_type = tt_dialect.PointerType(ptr.type)
    return ptr_type.pointee_type


def load(
    ptr: tensor,
    mask: tensor | None = None,
    other: tensor | None = None,
    *,
    cache_modifier: str | None = None,
    eviction_policy: str | None = None,
    is_volatile: bool = False,
) -> tensor:
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

  if ptr.type.is_ptr() and ptr.type.element_ty.is_block():
    # TODO(slebedev): Support load from a block pointer.
    raise NotImplementedError("loading from a block pointer is not supported")
  if not ptr.dtype.is_ptr():
    raise ValueError(f"unsupported pointer dtype: {ptr.dtype}")
  if other is not None:
    if mask is None:
      raise ValueError("other requires mask to be provided")
    assert mask.shape == other.shape == ptr.shape, (
        mask.shape,
        other.shape,
        ptr.shape,
    )
  elif mask is not None:
    assert mask.shape == ptr.shape
  if not ptr.type.is_block():
    if other is not None and other.type.is_block():
      raise ValueError("other cannot be a block if pointer is not a block")
    if mask is not None and mask.type.is_block():
      raise ValueError("mask cannot be a block if pointer is not a block")

  ptr_type = ptr.dtype
  element_type = ptr_type.element_ty

  if element_type == int1:
    # TODO(slebedev): Cast the result back to int1 before returning.
    element_type = int8
    ptr_type = pointer_type(element_type, ptr_type.address_space)
    ptr = semantic.cast(ptr, ptr_type)

  if other is not None:
    other = semantic.cast(other, element_type)

  result_handle = tt_dialect.load(
      _infer_load_return_type(ptr.handle),
      ptr.handle,
      mask=mask.handle if mask is not None else None,
      other=other.handle if other is not None else None,
      cache=cache_modifier,
      evict=eviction_policy,
      is_volatile=is_volatile,
  )
  if ptr.type.is_block():
    return tensor(result_handle, block_type(element_type, ptr.type.shape))
  else:
    return tensor(result_handle, element_type)


def store(
    ptr: tensor,
    value: tensor,
    mask: tensor | None = None,
    *,
    cache_modifier: str | None = None,
    eviction_policy: str | None = None,
) -> tensor:
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

  if ptr.type.is_ptr() and ptr.type.element_ty.is_block():
    # TODO(slebedev): Support load from a block pointer.
    raise NotImplementedError("storing to a block pointer is not supported")

  if not ptr.dtype.is_ptr():
    raise ValueError(f"unsupported pointer dtype: {ptr.dtype}")
  assert value.shape == ptr.shape
  if mask is not None:
    assert mask.shape == ptr.shape
  if not ptr.type.is_block():
    if value.type.is_block():
      raise ValueError("other cannot be a block if pointer is not a block")
    if mask is not None and mask.type.is_block():
      raise ValueError("mask cannot be a block if pointer is not a block")

  ptr_type = ptr.dtype
  element_type = ptr_type.element_ty

  if element_type == int1:
    # TODO(slebedev): Cast the result back to int1 before returning.
    element_type = int8
    ptr_type = pointer_type(element_type, ptr_type.address_space)
    ptr = semantic.cast(ptr, ptr_type)

  value = semantic.cast(value, element_type)

  return tensor(
      tt_dialect.store(
          ptr.handle,
          value.handle,
          mask=mask.handle if mask is not None else None,
          cache=cache_modifier,
          evict=eviction_policy,
      ),
      void,
  )


def arange(start: int, end: int) -> tensor:
  if end <= start:
    raise ValueError(
        f"end must be greater than start, but got: {end} <= {start}"
    )
  if max(start, end) >= 2**32:
    raise ValueError("start and end must fit in int32")
  ty = block_type(int32, [end - start])
  ir_ty = ir.RankedTensorType.get(
      [end - start], ir.IntegerType.get_signless(32)
  )
  return tensor(tt_dialect.make_range(ir_ty, start, end), ty)


def broadcast_to(x: tensor, shape: Sequence[int]) -> tensor:
  if not x.type.is_block():
    return splat(x, shape)
  elif x.shape == shape:
    return x
  x_ir_type = ir.RankedTensorType(x.handle.type)
  result_ir_type = ir.RankedTensorType.get(
      shape, x_ir_type.element_type, x_ir_type.encoding
  )
  return tensor(
      tt_dialect.broadcast(result_ir_type, x.handle),
      block_type(x.dtype, shape),
  )


def splat(x: tensor, shape: Sequence[int]) -> tensor:
  if x.type.is_block():
    raise ValueError("cannot splat a block tensor")
  if len(shape) == 0:
    return x
  result_ir_type = ir.RankedTensorType.get(shape, x.handle.type)
  return tensor(
      tt_dialect.splat(result_ir_type, x.handle), block_type(x.dtype, shape)
  )


def expand_dims(x: tensor, axis: int) -> tensor:
  dst_shape = list(x.shape)
  dst_shape.insert(axis, 1)
  if not x.type.is_block():
    return splat(input, dst_shape)
  return tensor(
      tt_dialect.expand_dims(x.handle, axis),
      block_type(x.dtype, dst_shape),
  )


def reshape(x: tensor, dst_shape: Sequence[int]) -> tensor:
  x_ir_type = ir.RankedTensorType(x.handle.type)
  result_ir_type = ir.RankedTensorType.get(
      dst_shape, x_ir_type.element_type, x_ir_type.encoding
  )
  return tensor(
      tt_dialect.reshape(result_ir_type, x.handle, allow_reorder=False),
      block_type(x.dtype, dst_shape),
  )


def _check_dot_operands(x_dtype: dtype, y_dtype: dtype, options: Any):
  # TODO(slebedev): Ensure that the dtypes are supported by CUDA.
  return


def dot(
    x: tensor,
    y: tensor,
    acc: tensor | None = None,
    allow_tf32: bool = True,
    max_num_imprecise_acc: int | None = None,
    out_dtype: dtype = float32,
) -> tensor:
  if min(*x.shape, *y.shape) < 16:
    raise ValueError("all dimensions of x and y must be >= 16 ")
  if out_dtype.is_bf16():
    raise ValueError(f"out_dtype={out_dtype} is unsupported")
  b: builder = builder.current
  _check_dot_operands(x.dtype, y.dtype, b.options)
  if x.dtype.is_int():
    if x.dtype != int8:
      raise TypeError(f"unsupported dtype: {x.dtype}")
    element_type = int32
  elif x.dtype.is_fp32() or x.dtype.is_bf16():
    element_type = float32
  else:
    element_type = out_dtype

  if element_type != out_dtype:
    raise TypeError(f"out_dtype={out_dtype} does not match element type {element_type}")

  m, _ = x.shape
  _, n = y.shape
  result_type = block_type(element_type, [m, n])

  if acc is None:
    zero = 0 if element_type.is_int() else 0.0
    acc = splat(
        tensor(
            arith_dialect.constant(element_type.to_ir(builder.current), zero),
            element_type,
        ),
        [m, n],
    )
  else:
    assert acc.type == result_type

  if max_num_imprecise_acc is None:
    if x.dtype.is_fp8() and y.dtype.is_fp8():
      max_num_imprecise_acc = b.options.max_num_imprecise_acc_default
    else:
      max_num_imprecise_acc = 0

  return tensor(
      tt_dialect.dot(
          x.handle,
          y.handle,
          acc.handle if acc is not None else None,
          allow_tf32,
          max_num_imprecise_acc,
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


def libdevice_extern_elementwise(
    table: Mapping[tuple[dtype, ...], tuple[str, dtype]],
    is_pure: bool = True,
):

  def inner(arg: tensor, *rest: tensor) -> tensor:
    args = (arg, *rest)
    assert all(other.shape == arg.shape for other in rest)
    key = tuple(arg.dtype for arg in args)
    try:
      symbol, dtype = table[key]
    except KeyError:
      raise NotImplementedError(f"unsupported dtypes: {key}") from None

    return_type = dtype
    if arg.type.is_block():
      return_type = block_type(dtype, arg.shape)
    return tensor(
        tt_dialect.extern_elementwise(
            return_type.to_ir(builder.current),
            [arg.handle for arg in args],
            libname="",
            libpath="",
            symbol=symbol,
            pure=is_pure,
        ),
        return_type,
    )

  return inner


class math:
  @staticmethod
  def max(x: tensor, y: tensor) -> tensor:
    # TODO(slebedev): Consider allowing customizing nan behavior.
    assert x.shape == y.shape
    if x.dtype.is_floating():
      # TODO(slebedev): Triton promotes bfloat16 to float32 and back here.
      return tensor(arith_dialect.maxnumf(x.handle, y.handle), x.type)
    if not x.dtype.is_int():
      raise NotImplementedError(f"unsupported dtypes: {x.dtype} and {y.dtype}")
    elif x.dtype.is_int_signed():
      return tensor(arith_dialect.maxsi(x.handle, y.handle), x.type)
    else:
      return tensor(arith_dialect.maxui(x.handle, y.handle), x.type)

  @staticmethod
  def min(x: tensor, y: tensor) -> tensor:
    # TODO(slebedev): Consider allowing customizing nan behavior.
    assert x.shape == y.shape
    if x.dtype.is_floating():
      # TODO(slebedev): Triton promotes bfloat16 to float32 and back here.
      return tensor(arith_dialect.minnumf(x.handle, y.handle), x.type)
    if not x.dtype.is_int():
      raise NotImplementedError(f"unsupported dtypes: {x.dtype} and {y.dtype}")
    elif x.dtype.is_int_signed():
      return tensor(arith_dialect.minsi(x.handle, y.handle), x.type)
    else:
      return tensor(arith_dialect.minui(x.handle, y.handle), x.type)

  sin = libdevice_extern_elementwise({
      (float32,): ("__nv_sinf", float32),
      (float64,): ("__nv_sin", float64),
  })
  cos = libdevice_extern_elementwise({
      (float32,): ("__nv_cosf", float32),
      (float64,): ("__nv_cos", float64),
  })
  tan = libdevice_extern_elementwise({
      (float32,): ("__nv_tanf", float32),
      (float64,): ("__nv_tan", float64),
  })
  asin = libdevice_extern_elementwise({
      (float32,): ("__nv_asinf", float32),
      (float64,): ("__nv_asin", float64),
  })
  acos = libdevice_extern_elementwise({
      (float32,): ("__nv_acosf", float32),
      (float64,): ("__nv_acos", float64),
  })
  atan = libdevice_extern_elementwise({
      (float32,): ("__nv_atanf", float32),
      (float64,): ("__nv_atan", float64),
  })
  atan2 = libdevice_extern_elementwise({
      (float32,): ("__nv_atan2f", float32),
      (float64,): ("__nv_atan2", float64),
  })
  sinh = libdevice_extern_elementwise({
      (float32,): ("__nv_sinhf", float32),
      (float64,): ("__nv_sinh", float64),
  })
  cosh = libdevice_extern_elementwise({
      (float32,): ("__nv_coshf", float32),
      (float64,): ("__nv_cosh", float64),
  })
  tanh = libdevice_extern_elementwise({
      (float32,): ("__nv_tanhf", float32),
      (float64,): ("__nv_tanh", float64),
  })
  asinh = libdevice_extern_elementwise({
      (float32,): ("__nv_asinhf", float32),
      (float64,): ("__nv_asinh", float64),
  })
  acosh = libdevice_extern_elementwise({
      (float32,): ("__nv_acosf", float32),
      (float64,): ("__nv_acosh", float64),
  })
  atanh = libdevice_extern_elementwise({
      (float32,): ("__nv_atanhf", float32),
      (float64,): ("__nv_atanh", float64),
  })

  cbrt = libdevice_extern_elementwise({
      (float32,): ("__nv_cbrtf", float32),
      (float64,): ("__nv_cbrt", float64),
  })
  clz = libdevice_extern_elementwise({
      (int32,): ("__nv_clz", int32),
      (int64,): ("__nv_clzll", int64),
  })
  exp = libdevice_extern_elementwise({
      (float32,): ("__nv_expf", float32),
      (float64,): ("__nv_exp", float64),
  })
  exp2 = libdevice_extern_elementwise({
      (float32,): ("__nv_exp2f", float32),
      (float64,): ("__nv_exp2", float64),
  })
  expm1 = libdevice_extern_elementwise({
      (float32,): ("__nv_expm1f", float32),
      (float64,): ("__nv_expm1", float64),
  })
  log = libdevice_extern_elementwise({
      (float32,): ("__nv_logf", float32),
      (float64,): ("__nv_log", float64),
  })
  log1p = libdevice_extern_elementwise({
      (float32,): ("__nv_log1pf", float32),
      (float64,): ("__nv_log1p", float64),
  })
  floor = libdevice_extern_elementwise({
      (float32,): ("__nv_floorf", float32),
      (float64,): ("__nv_floor", float64),
  })
  ceil = libdevice_extern_elementwise({
      (float32,): ("__nv_ceilf", float32),
      (float64,): ("__nv_ceil", float64),
  })
  abs = libdevice_extern_elementwise({
      (int32,): ("__nv_abs", int32),
      (int64,): ("__nv_llabs", int64),
      (float32,): ("__nv_fabsf", float32),
      (float64,): ("__nv_fabs", float64),
  })
  nextafter = libdevice_extern_elementwise({
      (float32, float32): ("__nv_nextafterf", float32),
      (float64, float64): ("__nv_nextafter", float64),
  })
  popc = libdevice_extern_elementwise({
      (int32,): ("__nv_popc", int32),
      (int64,): ("__nv_popcll", int64),
  })
  pow = libdevice_extern_elementwise({
      (float32, int32): ("__nv_powif", float32),
      (float64, int32): ("__nv_powi", float64),
      (float32, float32): ("__nv_powf", float32),
      (float64, float64): ("__nv_pow", float64),
  })
  sqrt = libdevice_extern_elementwise({
      (float32,): ("__nv_sqrtf", float32),
      (float64,): ("__nv_sqrt", float64),
  })
  rsqrt = libdevice_extern_elementwise({
      (float32,): ("__nv_rsqrtf", float32),
      (float64,): ("__nv_rsqrt", float64),
  })


def _full(t: ir.Type, v: object) -> ir.Type:
  element_type = t
  if ir.RankedTensorType.isinstance(t):
    element_type = ir.RankedTensorType(t).element_type

  if isinstance(element_type, ir.IntegerType):
    result = arith_dialect.constant(element_type, int(v))
  elif isinstance(element_type, _FLOAT_TYPES):
    result = arith_dialect.constant(element_type, float(v))
  else:
    raise NotImplementedError

  if ir.RankedTensorType.isinstance(t):
    return tt_dialect.splat(t, result)
  else:
    return result


def _int_cast(src: ir.Value, dst_type: ir.Type, is_signed: bool) -> ir.Value:
  src_type = src.type
  if ir.RankedTensorType.isinstance(
      src_type
  ) and ir.RankedTensorType.isinstance(dst_type):
    src_element_type = ir.RankedTensorType(src_type).element_type
    dst_element_type = ir.RankedTensorType(dst_type).element_type
  else:
    src_element_type = src_type
    dst_element_type = dst_type
  src_width = ir.IntegerType(src_element_type).width
  dst_width = ir.IntegerType(dst_element_type).width
  if src_width == dst_width:
    return arith_dialect.bitcast(dst_type, src)
  elif src_width > dst_width:
    return arith_dialect.trunci(dst_type, src)
  elif is_signed:
    return arith_dialect.extsi(dst_type, src)
  else:
    return arith_dialect.extui(dst_type, src)


class semantic:

  @staticmethod
  def cast(x: tensor, dst_ty: dtype) -> tensor:
    src_ty = x.type
    if src_ty.is_block():
      dst_ty = block_type(dst_ty.scalar, x.shape)
    if src_ty == dst_ty:
      return x

    src_element_ty = src_ty.scalar
    dst_element_ty = dst_ty.scalar

    if src_element_ty.is_fp8e4nv() or dst_element_ty.is_fp8e4nv():
      # TODO(slebedev): Check the CUDA version and raise conditionally.
      raise NotImplementedError("cannot cast from or to float8_e4m3fnuz")

    if (
        src_element_ty.is_floating()
        and dst_element_ty.is_floating()
        and (src_element_ty.is_fp8() or dst_element_ty.is_fp8())
    ):
      return tensor(
          tt_dialect.fp_to_fp(
              dst_ty.to_ir(builder.current),
              x.handle,
              rounding=tt_dialect.RoundingMode.RTNE,
          ),
          dst_ty,
      )

    if (
        src_element_ty.is_fp16() or src_element_ty.is_bf16()
    ) and not dst_element_ty.is_fp32():
      return semantic.cast(
          semantic.cast(x, float32), dst_element_ty.to_ir(builder.current)
      )

    if src_element_ty.is_floating() and dst_element_ty.is_floating():
      src_width = src_element_ty.primitive_bitwidth
      dst_width = dst_element_ty.primitive_bitwidth
      if src_width > dst_width:
        return tensor(
            arith_dialect.truncf(dst_ty.to_ir(builder.current), x.handle),
            dst_ty,
        )
      elif src_width < dst_width:
        return tensor(
            arith_dialect.extf(dst_ty.to_ir(builder.current), x.handle), dst_ty
        )

    if (
        src_element_ty.is_int()
        and dst_element_ty.is_int()
        and (
            src_element_ty.int_bitwidth != dst_element_ty.int_bitwidth
            or src_element_ty.int_signedness != dst_element_ty.int_signedness
        )
    ):
      if dst_element_ty.is_bool():
        zero = tensor(_full(src_ty.to_ir(builder.current), 0), src_ty)
        return semantic.not_equal(x, zero)
      else:
        sign_extend = (
            src_element_ty.is_int_signed() and not src_element_ty.is_bool()
        )
        return tensor(
            _int_cast(x.handle, dst_ty.to_ir(builder.current), sign_extend),
            dst_ty,
        )

    if src_element_ty.is_standard_floating() and dst_element_ty.is_int():
      if dst_element_ty.is_bool():
        zero = tensor(_full(src_ty.to_ir(builder.current), 0), src_ty)
        return semantic.not_equal(x, zero)
      elif dst_element_ty.is_int_signed():
        return tensor(
            arith_dialect.fptosi(dst_ty.to_ir(builder.current), x.handle),
            dst_ty,
        )
      else:
        return tensor(
            arith_dialect.fptoui(dst_ty.to_ir(builder.current), x.handle),
            dst_ty,
        )

    if src_element_ty.is_int() and dst_element_ty.is_standard_floating():
      if src_element_ty.is_bool() or not src_element_ty.is_int_signed():
        return tensor(
            arith_dialect.uitofp(dst_ty.to_ir(builder.current), x.handle),
            dst_ty,
        )
      else:
        return tensor(
            arith_dialect.sitofp(dst_ty.to_ir(builder.current), x.handle),
            dst_ty,
        )

    if src_element_ty.is_ptr() and dst_element_ty.is_int():
      if dst_element_ty.int_bitwidth == 64:
        return tensor(
            tt_dialect.ptr_to_int(dst_ty.to_ir(builder.current), x.handle),
            dst_ty,
        )
      else:
        x = semantic.cast(x, int64)
        zero = tensor(_full(x.type.to_ir(builder.current), 0), x.type)
        return semantic.not_equal(x, zero)

    if src_element_ty.is_int() and dst_element_ty.is_ptr():
      return tensor(
          tt_dialect.int_to_ptr(dst_ty.to_ir(builder.current), x.handle), dst_ty
      )

    if src_element_ty.is_ptr() and dst_element_ty.is_ptr():
      return tensor(
          tt_dialect.bitcast(dst_ty.to_ir(builder.current), x.handle), dst_ty
      )

    raise NotImplementedError(f"cannot cast {x} to {dst_ty}")

  @staticmethod
  def where(cond: tensor, x: tensor, y: tensor) -> tensor:
    assert cond.shape == x.shape == y.shape
    return tensor(arith_dialect.select(cond.handle, x.handle, y.handle), x.type)

  @staticmethod
  def trans(x: tensor) -> tensor:
    if len(x.shape) != 2:
      raise NotImplementedError(f"unsupported shape: {x.shape}")
    return tensor(
        tt_dialect.trans(x.handle),
        block_type(x.dtype, [*reversed(x.shape)]),
    )

  @staticmethod
  def minus(x: tensor) -> tensor:
    if x.dtype.is_ptr():
      raise NotImplementedError(f"unsupported dtype: {x.dtype}")
    zero = tensor(_full(x.type.to_ir(builder.current), 0), x.type)
    return semantic.sub(zero, x)

  @staticmethod
  def add(x: tensor, y: tensor) -> tensor:
    assert x.shape == y.shape
    if y.dtype.is_ptr() and not x.dtype.is_ptr():
      x, y = y, x
    if x.dtype.is_ptr():
      return tensor(
          tt_dialect.addptr(x.handle.type, x.handle, y.handle), x.type
      )
    elif not y.dtype.is_ptr():
      assert x.dtype == y.dtype
      if x.dtype.is_floating():
        return tensor(arith_dialect.addf(x.handle, y.handle), x.type)
      elif x.dtype.is_int():
        return tensor(arith_dialect.addi(x.handle, y.handle), x.type)
    raise NotImplementedError(f"unsupported dtypes: {x.dtype} and {y.dtype}")

  @staticmethod
  def sub(x: tensor, y: tensor) -> tensor:
    assert x.shape == y.shape
    if x.dtype.is_ptr():
      return tensor(
          tt_dialect.addptr(x.handle.type, x.handle, semantic.minus(y).handle),
          x.type,
      )
    elif not y.dtype.is_ptr():
      assert x.dtype == y.dtype
      if y.dtype.is_floating():
        return tensor(arith_dialect.subf(x.handle, y.handle), x.type)
      elif y.dtype.is_int():
        return tensor(arith_dialect.subi(x.handle, y.handle), x.type)
    raise NotImplementedError(f"unsupported dtypes: {y.dtype} and {y.dtype}")

  @staticmethod
  def mul(x: tensor, y: tensor) -> tensor:
    assert x.shape == y.shape and x.dtype == y.dtype
    if x.dtype.is_floating():
      return tensor(arith_dialect.mulf(x.handle, y.handle), x.type)
    elif x.dtype.is_int():
      return tensor(arith_dialect.muli(x.handle, y.handle), x.type)
    raise NotImplementedError(f"unsupported dtypes: {x.dtype} and {y.dtype}")

  @staticmethod
  def floordiv(x: tensor, y: tensor) -> tensor:
    assert x.shape == y.shape and x.dtype == y.dtype
    if not x.dtype.is_int():
      raise NotImplementedError(f"unsupported dtypes: {x.dtype} and {y.dtype}")
    if x.dtype.is_int_signed():
      return tensor(arith_dialect.divsi(x.handle, y.handle), x.type)
    else:
      return tensor(arith_dialect.divui(x.handle, y.handle), x.type)

  @staticmethod
  def truediv(x: tensor, y: tensor) -> tensor:
    assert x.shape == y.shape and x.dtype == y.dtype
    if x.dtype.is_int():
      assert y.dtype.is_int()
      x = semantic.cast(x, float32)
      y = semantic.cast(y, float32)
    if x.dtype.is_floating():
      assert y.dtype.is_floating()
      return tensor(arith_dialect.divf(x.handle, y.handle), x.type)
    raise NotImplementedError(f"unsupported dtypes: {x.dtype} and {y.dtype}")

  @staticmethod
  def mod(x: tensor, y: tensor) -> tensor:
    assert x.shape == y.shape and x.dtype == y.dtype
    if not x.dtype.is_int():
      raise NotImplementedError(f"unsupported dtypes: {x.dtype} and {y.dtype}")
    if x.dtype.is_int_signed():
      return tensor(arith_dialect.remsi(x.handle, y.handle), x.type)
    else:
      return tensor(arith_dialect.remui(x.handle, y.handle), x.type)

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
    assert x.shape == y.shape and x.dtype == y.dtype
    if x.dtype.is_floating():
      return tensor(
          arith_dialect.cmpf(
              arith_dialect.CmpFPredicate.OEQ, x.handle, y.handle
          ),
          _bool_block_like(x),
      )
    elif x.dtype.is_int():
      return tensor(
          arith_dialect.cmpi(
              arith_dialect.CmpIPredicate.eq, x.handle, y.handle
          ),
          _bool_block_like(x),
      )
    raise NotImplementedError(f"unsupported dtypes: {x.dtype} and {y.dtype}")

  @staticmethod
  def not_equal(x: tensor, y: tensor) -> tensor:
    assert x.shape == y.shape and x.dtype == y.dtype
    if x.dtype.is_floating():
      return tensor(
          arith_dialect.cmpf(
              arith_dialect.CmpFPredicate.UNE, x.handle, y.handle
          ),
          _bool_block_like(x),
      )
    elif x.dtype.is_int():
      return tensor(
          arith_dialect.cmpi(
              arith_dialect.CmpIPredicate.ne, x.handle, y.handle
          ),
          _bool_block_like(x),
      )
    raise NotImplementedError(f"unsupported dtypes: {x.dtype} and {y.dtype}")

  @staticmethod
  def greater_than(x: tensor, y: tensor) -> tensor:
    assert x.shape == y.shape and x.dtype == y.dtype
    if x.dtype.is_floating():
      return tensor(
          arith_dialect.cmpf(
              arith_dialect.CmpFPredicate.OGT, x.handle, y.handle
          ),
          _bool_block_like(x),
      )
    elif x.dtype.is_int():
      if x.dtype.is_int_signed():
        return tensor(
            arith_dialect.cmpi(
                arith_dialect.CmpIPredicate.sgt, x.handle, y.handle
            ),
            _bool_block_like(x),
        )
      else:
        return tensor(
            arith_dialect.cmpi(
                arith_dialect.CmpIPredicate.ugt, x.handle, y.handle
            ),
            _bool_block_like(x),
        )
    raise NotImplementedError(f"unsupported dtypes: {x.dtype} and {y.dtype}")

  @staticmethod
  def greater_equal(x: tensor, y: tensor) -> tensor:
    assert x.shape == y.shape and x.dtype == y.dtype
    if x.dtype.is_floating():
      return tensor(
          arith_dialect.cmpf(
              arith_dialect.CmpFPredicate.OGE, x.handle, y.handle
          ),
          _bool_block_like(x),
      )
    elif x.dtype.is_int():
      if x.dtype.is_int_signed():
        return tensor(
            arith_dialect.cmpi(
                arith_dialect.CmpIPredicate.sge, x.handle, y.handle
            ),
            _bool_block_like(x),
        )
      else:
        return tensor(
            arith_dialect.cmpi(
                arith_dialect.CmpIPredicate.uge, x.handle, y.handle
            ),
            _bool_block_like(x),
        )
    raise NotImplementedError(f"unsupported dtypes: {x.dtype} and {y.dtype}")

  @staticmethod
  def less_than(x: tensor, y: tensor) -> tensor:
    assert x.shape == y.shape and x.dtype == y.dtype
    if x.dtype.is_floating():
      return tensor(
          arith_dialect.cmpf(
              arith_dialect.CmpFPredicate.OLT, x.handle, y.handle
          ),
          _bool_block_like(x),
      )
    elif x.dtype.is_int():
      if x.dtype.is_int_signed():
        return tensor(
            arith_dialect.cmpi(
                arith_dialect.CmpIPredicate.slt, x.handle, y.handle
            ),
            _bool_block_like(x),
        )
      else:
        return tensor(
            arith_dialect.cmpi(
                arith_dialect.CmpIPredicate.ult, x.handle, y.handle
            ),
            _bool_block_like(x),
        )
    raise NotImplementedError(f"unsupported dtypes: {x.dtype} and {y.dtype}")

  @staticmethod
  def less_equal(x: tensor, y: tensor) -> tensor:
    assert x.shape == y.shape and x.dtype == y.dtype
    if x.dtype.is_floating():
      return tensor(
          arith_dialect.cmpf(
              arith_dialect.CmpFPredicate.OLE, x.handle, y.handle
          ),
          _bool_block_like(x),
      )
    elif x.dtype.is_int():
      if x.dtype.is_int_signed():
        return tensor(
            arith_dialect.cmpi(
                arith_dialect.CmpIPredicate.sle, x.handle, y.handle
            ),
            _bool_block_like(x),
        )
      else:
        return tensor(
            arith_dialect.cmpi(
                arith_dialect.CmpIPredicate.ule, x.handle, y.handle
            ),
            _bool_block_like(x),
        )
    raise NotImplementedError(f"unsupported dtypes: {x.dtype} and {y.dtype}")
