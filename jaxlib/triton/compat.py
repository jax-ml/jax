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

from contextlib import contextmanager
from functools import partial, wraps
import threading
from typing import TypeAlias

from triton._C.libtriton import ir as tl_ir  # type: ignore
import triton.compiler.backends.cuda as cb  # type: ignore
import triton.language as tl  # type: ignore


builder: TypeAlias = tl_ir.builder
module: TypeAlias = tl_ir.module
context: TypeAlias = tl_ir.context


_tls = threading.local()


@contextmanager
def new_builder(cuda_options: cb.CUDAOptions) -> builder:
  context = tl_ir.context()
  context.load_triton()
  builder = tl_ir.builder(context)
  builder.options = cuda_options
  _tls.context = context
  _tls.builder = builder
  yield builder
  del _tls.context
  del _tls.builder


def get_context() -> context:
  return _tls.context


def get_builder() -> builder:
  return _tls.builder


dtype = tl.core.dtype

block_type = tl.core.block_type
function_type = tl.core.function_type
pointer_type = tl.core.pointer_type

bfloat16 = tl.core.bfloat16
float16 = tl.core.float16
float32 = tl.core.float32
float64 = tl.core.float64
int32 = tl.core.int32
int64 = tl.core.int64


def wrap_with_builder(fn):
  @wraps(fn)
  def inner(*args, **kwargs):
    if tl.core.is_builtin(fn):
      v = fn(*args, **kwargs, _builder=get_builder())
    else:
      v = fn(*args, **kwargs, builder=get_builder())
    if isinstance(v, tl.core.tensor):
      return _to_tensor(v)
    return v

  return inner


constexpr = tl.core.constexpr


def _to_tensor(v) -> "tensor":
  t = tl.core._to_tensor(v, get_builder())
  return tensor(t.handle, t.type)


class tensor(tl.core.tensor):

  def __add__(self, other):
    return semantic.add(self, _to_tensor(other))

  def __radd__(self, other):
    return self + other

  def __sub__(self, other):
    return semantic.sub(self, _to_tensor(other))

  def __rsub__(self, other):
    return semantic.sub(_to_tensor(other), self)

  def __mul__(self, other):
    return semantic.mul(self, _to_tensor(other))

  def __rmul__(self, other):
    return self * other

  def __truediv__(self, other):
    return semantic.truediv(self, _to_tensor(other))

  def __rtruediv__(self, other):
    return semantic.truediv(_to_tensor(other), self)

  def __floordiv__(self, other):
    return semantic.floordiv(self, _to_tensor(other))

  def __rfloordiv__(self, other):
    return semantic.floordiv(_to_tensor(other), self)

  def __mod__(self, other):
    return semantic.mod(self, _to_tensor(other))

  def __rmod__(self, other):
    return semantic.mod(_to_tensor(other), self)

  def __neg__(self):
    return semantic.minus(self)

  def __invert__(self):
    return semantic.invert(self)

  # TODO(slebedev): Override other comparison methods.
  def __eq__(self, other):
    return semantic.equal(self, _to_tensor(other))

  __getitem__ = wrap_with_builder(tl.tensor.__getitem__)

  to = wrap_with_builder(tl.tensor.to)


program_id = wrap_with_builder(tl.core.program_id)

load = wrap_with_builder(tl.core.load)
store = wrap_with_builder(tl.core.store)

arange = wrap_with_builder(tl.core.arange)
broadcast_to = wrap_with_builder(tl.core.broadcast_to)
expand_dims = wrap_with_builder(tl.core.expand_dims)
reshape = wrap_with_builder(tl.core.reshape)

dot = wrap_with_builder(tl.core.dot)

atomic_xchg = wrap_with_builder(tl.core.atomic_xchg)
atomic_add = wrap_with_builder(tl.core.atomic_add)
atomic_max = wrap_with_builder(tl.core.atomic_max)
atomic_min = wrap_with_builder(tl.core.atomic_min)
atomic_and = wrap_with_builder(tl.core.atomic_and)
atomic_or = wrap_with_builder(tl.core.atomic_or)
atomic_xor = wrap_with_builder(tl.core.atomic_xor)
atomic_cas = wrap_with_builder(tl.atomic_cas)

abs = wrap_with_builder(tl.abs)
exp = wrap_with_builder(tl.exp)
log = wrap_with_builder(tl.log)
sqrt = wrap_with_builder(tl.sqrt)
sin = wrap_with_builder(tl.sin)
cos = wrap_with_builder(tl.cos)
max_contiguous = wrap_with_builder(tl.max_contiguous)
multiple_of = wrap_with_builder(tl.multiple_of)


class math:
  acos = wrap_with_builder(tl.math.acos)
  acosh = wrap_with_builder(tl.math.acosh)
  asin = wrap_with_builder(tl.math.asin)
  asinh = wrap_with_builder(tl.math.asinh)
  atan = wrap_with_builder(tl.math.atan)
  atan2 = wrap_with_builder(tl.math.atan2)
  atanh = wrap_with_builder(tl.math.atanh)
  cbrt = wrap_with_builder(tl.math.cbrt)
  ceil = wrap_with_builder(tl.math.ceil)
  clz = wrap_with_builder(tl.math.clz)
  cosh = wrap_with_builder(tl.math.cosh)
  exp2 = wrap_with_builder(tl.math.exp2)
  expm1 = wrap_with_builder(tl.math.expm1)
  floor = wrap_with_builder(tl.math.floor)
  log1p = wrap_with_builder(tl.math.log1p)
  max = partial(
      wrap_with_builder(tl.math.max),
      propagate_nan=tl.PropagateNan.NONE,
  )
  min = partial(
      wrap_with_builder(tl.math.min),
      propagate_nan=tl.PropagateNan.NONE,
  )
  nextafter = wrap_with_builder(tl.math.nextafter)
  popc = wrap_with_builder(tl.math.popc)
  pow = wrap_with_builder(tl.math.pow)
  rsqrt = wrap_with_builder(tl.math.rsqrt)
  sinh = wrap_with_builder(tl.math.sinh)
  tan = wrap_with_builder(tl.math.tan)
  tanh = wrap_with_builder(tl.math.tanh)


class semantic:
  add = wrap_with_builder(tl.semantic.add)
  and_ = wrap_with_builder(tl.semantic.and_)
  ashr = wrap_with_builder(tl.semantic.ashr)
  cast = wrap_with_builder(tl.semantic.cast)
  equal = wrap_with_builder(tl.semantic.equal)
  expand_dims = wrap_with_builder(tl.semantic.expand_dims)
  floordiv = wrap_with_builder(tl.semantic.floordiv)
  greater_equal = wrap_with_builder(tl.semantic.greater_equal)
  greater_than = wrap_with_builder(tl.semantic.greater_than)
  invert = wrap_with_builder(tl.semantic.invert)
  less_equal = wrap_with_builder(tl.semantic.less_equal)
  less_than = wrap_with_builder(tl.semantic.less_than)
  lshr = wrap_with_builder(tl.semantic.lshr)
  minus = wrap_with_builder(tl.semantic.minus)
  mod = wrap_with_builder(tl.semantic.mod)
  mul = wrap_with_builder(tl.semantic.mul)
  not_equal = wrap_with_builder(tl.semantic.not_equal)
  or_ = wrap_with_builder(tl.semantic.or_)
  shl = wrap_with_builder(tl.semantic.shl)
  sub = wrap_with_builder(tl.semantic.sub)
  trans = wrap_with_builder(tl.semantic.trans)
  truediv = wrap_with_builder(tl.semantic.truediv)
  where = wrap_with_builder(tl.semantic.where)
  xor_ = wrap_with_builder(tl.semantic.xor_)
