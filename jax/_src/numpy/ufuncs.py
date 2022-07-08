# Copyright 2018 Google LLC
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

# pytype: skip-file
"""
Implements ufuncs for jax.numpy.
"""

from functools import partial
import operator
from textwrap import dedent

import numpy as np

import jax
from jax._src.api import jit, custom_jvp, vmap
from jax._src import dtypes
from jax._src.lax import lax as lax_internal
from jax._src.numpy import reductions
from jax._src.numpy.util import (
   _broadcast_to, _check_arraylike, _eliminate_deprecated_list_indexing, _promote_args,
   _promote_args_inexact, _promote_dtypes_inexact, _promote_shapes, _where, _wraps)
from jax._src.util import canonicalize_axis, moveaxis
from jax import core
from jax import lax

_lax_const = lax_internal._const

_INT_DTYPES = {
  16: np.int16,
  32: np.int32,
  64: np.int64,
}


_AT_INPLACE_WARNING = """\
Because JAX arrays are immutable, jnp.ufunc.at() cannot operate inplace like
np.ufunc.at(). Instead, you can pass inplace=False and capture the result; e.g.
>>> arr = jnp.add.at(arr, ind, val, inplace=False)
"""


def _constant_like(x, const):
  return np.array(const, dtype=dtypes.dtype(x))


def _asarray(a, dtype=None):
  # simplified version of jnp.asarray() for local use. This will not
  # properly handle list inputs, so it should only be used after calling
  # _check_arraylike.
  return lax.convert_element_type(a, dtype or dtypes.dtype(a, canonicalize=True))


def _result_dtype(op, *args):
  """Compute result dtype of applying op to arguments with given dtypes."""
  args = [np.ones((0,) * np.ndim(arg), dtypes.dtype(arg)) for arg in args]
  return dtypes.dtype(op(*args))


def _replace_inf(x):
  return lax.select(isposinf(real(x)), lax_internal._zeros(x), x)


def _one_to_one_unop(numpy_fn, lax_fn, promote_to_inexact=False, lax_doc=False):
  if promote_to_inexact:
    fn = lambda x: lax_fn(*_promote_args_inexact(numpy_fn.__name__, x))
  else:
    fn = lambda x: lax_fn(*_promote_args(numpy_fn.__name__, x))
  fn = jit(fn, inline=True)
  if lax_doc:
    doc = dedent('\n\n'.join(lax_fn.__doc__.split('\n\n')[1:])).strip()
    return _wraps(numpy_fn, lax_description=doc)(fn)
  else:
    return _wraps(numpy_fn)(fn)


def _one_to_one_binop(numpy_fn, lax_fn, promote_to_inexact=False, lax_doc=False):
  if promote_to_inexact:
    fn = lambda x1, x2: lax_fn(*_promote_args_inexact(numpy_fn.__name__, x1, x2))
  else:
    fn = lambda x1, x2: lax_fn(*_promote_args(numpy_fn.__name__, x1, x2))
  fn = jit(fn, inline=True)
  if lax_doc:
    doc = dedent('\n\n'.join(lax_fn.__doc__.split('\n\n')[1:])).strip()
    return _wraps(numpy_fn, lax_description=doc)(fn)
  else:
    return _wraps(numpy_fn)(fn)


def _maybe_bool_binop(numpy_fn, lax_fn, bool_lax_fn, lax_doc=False):
  def fn(x1, x2):
    x1, x2 = _promote_args(numpy_fn.__name__, x1, x2)
    return lax_fn(x1, x2) if x1.dtype != np.bool_ else bool_lax_fn(x1, x2)
  fn = jit(fn, inline=True)
  if lax_doc:
    doc = dedent('\n\n'.join(lax_fn.__doc__.split('\n\n')[1:])).strip()
    return _wraps(numpy_fn, lax_description=doc)(fn)
  else:
    return _wraps(numpy_fn)(fn)


def _comparison_op(numpy_fn, lax_fn):
  # TODO(https://github.com/google/jax/issues/6713): decorate this function with
  # jit, after fixing a surprising interaction with remat(..., concrete=True).
  def fn(x1, x2):
    x1, x2 =  _promote_args(numpy_fn.__name__, x1, x2)
    # Comparison on complex types are defined as a lexicographic ordering on
    # the (real, imag) pair.
    if dtypes.issubdtype(dtypes.dtype(x1), np.complexfloating):
      rx = lax.real(x1)
      ry = lax.real(x2)
      return lax.select(lax.eq(rx, ry), lax_fn(lax.imag(x1), lax.imag(x2)),
                        lax_fn(rx, ry))
    return lax_fn(x1, x2)
  return _wraps(numpy_fn)(fn)


def _logical_op(np_op, bitwise_op):
  @_wraps(np_op, update_doc=False)
  @partial(jit, inline=True)
  def op(*args):
    zero = lambda x: lax.full_like(x, shape=(), fill_value=0)
    args = (x if dtypes.issubdtype(dtypes.dtype(x), np.bool_) else lax.ne(x, zero(x))
            for x in args)
    return bitwise_op(*_promote_args(np_op.__name__, *args))
  return op

# Note: functions with preceding underscores here are used in ufunc definitions below.
# TODO(jakevdp): convert more of these to ufuncs.

# Unary Functions
_fabs = _one_to_one_unop(np.fabs, lax.abs, True)
_bitwise_not = _one_to_one_unop(np.bitwise_not, lax.bitwise_not)
_invert = _one_to_one_unop(np.invert, lax.bitwise_not)
_negative = _one_to_one_unop(np.negative, lax.neg)
_positive = _one_to_one_unop(np.positive, lambda x: x)
_floor = _one_to_one_unop(np.floor, lax.floor, True)
_ceil = _one_to_one_unop(np.ceil, lax.ceil, True)
exp = _one_to_one_unop(np.exp, lax.exp, True)
log = _one_to_one_unop(np.log, lax.log, True)
expm1 = _one_to_one_unop(np.expm1, lax.expm1, True)
log1p = _one_to_one_unop(np.log1p, lax.log1p, True)
sin = _one_to_one_unop(np.sin, lax.sin, True)
cos = _one_to_one_unop(np.cos, lax.cos, True)
tan = _one_to_one_unop(np.tan, lax.tan, True)
arcsin = _one_to_one_unop(np.arcsin, lax.asin, True)
arccos = _one_to_one_unop(np.arccos, lax.acos, True)
arctan = _one_to_one_unop(np.arctan, lax.atan, True)
sinh = _one_to_one_unop(np.sinh, lax.sinh, True)
cosh = _one_to_one_unop(np.cosh, lax.cosh, True)
arcsinh = _one_to_one_unop(np.arcsinh, lax.asinh, True)
tanh = _one_to_one_unop(np.tanh, lax.tanh, True)
arctanh = _one_to_one_unop(np.arctanh, lax.atanh, True)
sqrt = _one_to_one_unop(np.sqrt, lax.sqrt, True)
cbrt = _one_to_one_unop(np.cbrt, lax.cbrt, True)

# Binary Functions
_add = _maybe_bool_binop(np.add, lax.add, lax.bitwise_or)
_bitwise_and = _one_to_one_binop(np.bitwise_and, lax.bitwise_and)
_bitwise_or = _one_to_one_binop(np.bitwise_or, lax.bitwise_or)
_bitwise_xor = _one_to_one_binop(np.bitwise_xor, lax.bitwise_xor)
left_shift = _one_to_one_binop(np.left_shift, lax.shift_left)
equal = _one_to_one_binop(np.equal, lax.eq)
_multiply = _maybe_bool_binop(np.multiply, lax.mul, lax.bitwise_and)
not_equal = _one_to_one_binop(np.not_equal, lax.ne)
_subtract = _one_to_one_binop(np.subtract, lax.sub)
arctan2 = _one_to_one_binop(np.arctan2, lax.atan2, True)
_minimum = _one_to_one_binop(np.minimum, lax.min)
_maximum = _one_to_one_binop(np.maximum, lax.max)
float_power = _one_to_one_binop(np.float_power, lax.pow, True)
nextafter = _one_to_one_binop(np.nextafter, lax.nextafter, True, True)

# Comparison Functions
greater_equal = _comparison_op(np.greater_equal, lax.ge)
greater = _comparison_op(np.greater, lax.gt)
less_equal = _comparison_op(np.less_equal, lax.le)
less = _comparison_op(np.less, lax.lt)

# Logical Functions
logical_and = _logical_op(np.logical_and, lax.bitwise_and)
logical_not = _logical_op(np.logical_not, lax.bitwise_not)
logical_or = _logical_op(np.logical_or, lax.bitwise_or)
logical_xor = _logical_op(np.logical_xor, lax.bitwise_xor)


@_wraps(np.arccosh)
@jit
def arccosh(x):
  # Note: arccosh is multi-valued for complex input, and lax.acosh uses a different
  # convention than np.arccosh.
  out = lax.acosh(*_promote_args_inexact("arccosh", x))
  if dtypes.issubdtype(out.dtype, np.complexfloating):
    out = _where(real(out) < 0, lax.neg(out), out)
  return out


@_wraps(np.right_shift)
@partial(jit, inline=True)
def right_shift(x1, x2):
  x1, x2 = _promote_args(np.right_shift.__name__, x1, x2)
  lax_fn = lax.shift_right_logical if \
    np.issubdtype(x1.dtype, np.unsignedinteger) else lax.shift_right_arithmetic
  return lax_fn(x1, x2)


@_wraps(np.absolute)
@partial(jit, inline=True)
def absolute(x):
  _check_arraylike('absolute', x)
  dt = dtypes.dtype(x)
  return x if dt == np.bool_ or dtypes.issubdtype(dt, np.unsignedinteger) else lax.abs(x)
abs = _wraps(np.abs)(absolute)


@_wraps(np.rint)
@jit
def rint(x):
  _check_arraylike('rint', x)
  dtype = dtypes.dtype(x)
  if dtypes.issubdtype(dtype, np.integer):
    return lax.convert_element_type(x, dtypes.float_)
  if dtypes.issubdtype(dtype, np.complexfloating):
    return lax.complex(rint(lax.real(x)), rint(lax.imag(x)))
  return lax.round(x, lax.RoundingMethod.TO_NEAREST_EVEN)


@_wraps(np.sign)
@jit
def sign(x):
  _check_arraylike('sign', x)
  dtype = dtypes.dtype(x)
  if dtypes.issubdtype(dtype, np.complexfloating):
    re = lax.real(x)
    return lax.complex(
      lax.sign(_where(re != 0, re, lax.imag(x))), _constant_like(re, 0))
  return lax.sign(x)


@_wraps(np.copysign)
@jit
def copysign(x1, x2):
  x1, x2 = _promote_args_inexact("copysign", x1, x2)
  if dtypes.issubdtype(dtypes.dtype(x1), np.complexfloating):
    raise TypeError("copysign does not support complex-valued inputs")
  return _where(signbit(x2).astype(bool), -lax.abs(x1), lax.abs(x1))


@_wraps(np.true_divide)
@partial(jit, inline=True)
def _true_divide(x1, x2):
  x1, x2 = _promote_args_inexact("true_divide", x1, x2)
  return lax.div(x1, x2)


@_wraps(np.floor_divide)
@jit
def _floor_divide(x1, x2):
  x1, x2 = _promote_args("floor_divide", x1, x2)
  dtype = dtypes.dtype(x1)
  if dtypes.issubdtype(dtype, np.integer):
    quotient = lax.div(x1, x2)
    select = logical_and(lax.sign(x1) != lax.sign(x2), lax.rem(x1, x2) != 0)
    # TODO(mattjj): investigate why subtracting a scalar was causing promotion
    return _where(select, quotient - 1, quotient)
  elif dtypes.issubdtype(dtype, np.complexfloating):
    x1r = lax.real(x1)
    x1i = lax.imag(x1)
    x2r = lax.real(x2)
    x2i = lax.imag(x2)
    which = lax.ge(lax.abs(x2r), lax.abs(x2i))
    rat1 = _where(which, lax.full_like(x2i, 1), lax.div(x2r, x2i))
    rat2 = _where(which, lax.div(x2i, x2r), _lax_const(x2i, 1))
    out = lax.floor(lax.div(lax.add(lax.mul(x1r, rat1), lax.mul(x1i, rat2)),
                            lax.add(lax.mul(x2r, rat1), lax.mul(x2i, rat2))))
    return lax.convert_element_type(out, dtype)
  else:
    return _float_divmod(x1, x2)[0]


@_wraps(np.divmod)
@jit
def divmod(x1, x2):
  x1, x2 = _promote_args("divmod", x1, x2)
  if dtypes.issubdtype(dtypes.dtype(x1), np.integer):
    return floor_divide(x1, x2), remainder(x1, x2)
  else:
    return _float_divmod(x1, x2)


def _float_divmod(x1, x2):
  # see float_divmod in floatobject.c of CPython
  mod = lax.rem(x1, x2)
  div = lax.div(lax.sub(x1, mod), x2)

  ind = lax.bitwise_and(mod != 0, lax.sign(x2) != lax.sign(mod))
  mod = lax.select(ind, mod + x2, mod)
  div = lax.select(ind, div - _constant_like(div, 1), div)

  return lax.round(div), mod


@partial(jit, inline=True)
def _power(x1, x2):
  x1, x2 = _promote_args("power", x1, x2)
  dtype = dtypes.dtype(x1)
  if not dtypes.issubdtype(dtype, np.integer):
    return lax.pow(x1, x2)

  # Integer power => use binary exponentiation.

  # TODO(phawkins): add integer pow support to XLA.
  bits = 6  # Anything more would overflow for any x1 > 1
  zero = _constant_like(x2, 0)
  one = _constant_like(x2, 1)
  # Initialize acc carefully such that pow(0, x2) is zero for x2 != 0
  acc = _where(lax.bitwise_and(lax.eq(x1, zero), lax.ne(x2, zero)), zero, one)
  for _ in range(bits):
    acc = _where(lax.bitwise_and(x2, one), lax.mul(acc, x1), acc)
    x1 = lax.mul(x1, x1)
    x2 = lax.shift_right_logical(x2, one)
  return acc


@_wraps(np.power)
def power(x1, x2):
  # Special case for concrete integer scalars: use binary exponentiation.
  # Using lax.pow may be imprecise for floating-point values; the goal of this
  # code path is to make sure we end up with a precise output for the common
  # pattern ``x ** 2`` or similar.
  if isinstance(core.get_aval(x2), core.ConcreteArray):
    try:
      x2 = operator.index(x2)
    except TypeError:
      pass
    else:
      return lax.integer_pow(x1, x2)
  return _power(x1, x2)


@custom_jvp
@_wraps(np.logaddexp)
@jit
def logaddexp(x1, x2):
  x1, x2 = _promote_args_inexact("logaddexp", x1, x2)
  amax = lax.max(x1, x2)
  if dtypes.issubdtype(x1.dtype, np.floating):
    delta = lax.sub(x1, x2)
    return lax.select(lax_internal._isnan(delta),
                      lax.add(x1, x2),  # NaNs or infinities of the same sign.
                      lax.add(amax, lax.log1p(lax.exp(lax.neg(lax.abs(delta))))))
  else:
    delta = lax.sub(lax.add(x1, x2), lax.mul(amax, _constant_like(amax, 2)))
    out = lax.add(amax, lax.log1p(lax.exp(delta)))
    return lax.complex(lax.real(out), _wrap_between(lax.imag(out), np.pi))


def _wrap_between(x, _a):
  """Wraps `x` between `[-a, a]`."""
  a = _constant_like(x, _a)
  two_a = _constant_like(x, 2 * _a)
  zero = _constant_like(x, 0)
  rem = lax.rem(lax.add(x, a), two_a)
  rem = lax.select(lax.lt(rem, zero), lax.add(rem, two_a), rem)
  return lax.sub(rem, a)


@logaddexp.defjvp
def _logaddexp_jvp(primals, tangents):
  x1, x2 = primals
  t1, t2 = tangents
  x1, x2, t1, t2 = _promote_args_inexact("logaddexp_jvp", x1, x2, t1, t2)
  primal_out = logaddexp(x1, x2)
  tangent_out = lax.add(lax.mul(t1, exp(lax.sub(_replace_inf(x1), _replace_inf(primal_out)))),
                        lax.mul(t2, exp(lax.sub(_replace_inf(x2), _replace_inf(primal_out)))))
  return primal_out, tangent_out


@custom_jvp
@_wraps(np.logaddexp2)
@jit
def logaddexp2(x1, x2):
  x1, x2 = _promote_args_inexact("logaddexp2", x1, x2)
  amax = lax.max(x1, x2)
  if dtypes.issubdtype(x1.dtype, np.floating):
    delta = lax.sub(x1, x2)
    return lax.select(lax_internal._isnan(delta),
                      lax.add(x1, x2),  # NaNs or infinities of the same sign.
                      lax.add(amax, lax.div(lax.log1p(exp2(lax.neg(lax.abs(delta)))),
                                            _constant_like(x1, np.log(2)))))
  else:
    delta = lax.sub(lax.add(x1, x2), lax.mul(amax, _constant_like(amax, 2)))
    out = lax.add(amax, lax.div(lax.log1p(exp2(delta)), _constant_like(x1, np.log(2))))
    return lax.complex(lax.real(out), _wrap_between(lax.imag(out), np.pi / np.log(2)))


@logaddexp2.defjvp
def _logaddexp2_jvp(primals, tangents):
  x1, x2 = primals
  t1, t2 = tangents
  x1, x2, t1, t2 = _promote_args_inexact("logaddexp2_jvp", x1, x2, t1, t2)
  primal_out = logaddexp2(x1, x2)
  tangent_out = lax.add(lax.mul(t1, exp2(lax.sub(_replace_inf(x1), _replace_inf(primal_out)))),
                        lax.mul(t2, exp2(lax.sub(_replace_inf(x2), _replace_inf(primal_out)))))
  return primal_out, tangent_out


@_wraps(np.log2)
@partial(jit, inline=True)
def log2(x):
  x, = _promote_args_inexact("log2", x)
  return lax.div(lax.log(x), lax.log(_constant_like(x, 2)))


@_wraps(np.log10)
@partial(jit, inline=True)
def log10(x):
  x, = _promote_args_inexact("log10", x)
  return lax.div(lax.log(x), lax.log(_constant_like(x, 10)))


@_wraps(np.exp2)
@partial(jit, inline=True)
def exp2(x):
  x, = _promote_args_inexact("exp2", x)
  return lax.exp(lax.mul(lax.log(_constant_like(x, 2)), x))


@_wraps(np.signbit)
@jit
def signbit(x):
  x, = _promote_args("signbit", x)
  dtype = dtypes.dtype(x)
  if dtypes.issubdtype(dtype, np.integer):
    return lax.lt(x, _constant_like(x, 0))
  elif dtypes.issubdtype(dtype, np.bool_):
    return lax.full_like(x, False, dtype=np.bool_)
  elif not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(
        "jax.numpy.signbit is not well defined for %s" % dtype)

  # TPU supports BF16 but not S16 types, so as a workaround, convert BF16 to
  # F32.
  if dtype == dtypes.bfloat16:
    dtype = np.float32
    x = lax.convert_element_type(x, np.float32)

  info = dtypes.finfo(dtype)
  if info.bits not in _INT_DTYPES:
    raise NotImplementedError(
        "jax.numpy.signbit only supports 16, 32, and 64-bit types.")
  int_type = _INT_DTYPES[info.bits]
  x = lax.bitcast_convert_type(x, int_type)
  return lax.convert_element_type(x >> (info.nexp + info.nmant), np.bool_)


def _normalize_float(x):
  info = dtypes.finfo(dtypes.dtype(x))
  int_type = _INT_DTYPES[info.bits]
  cond = lax.abs(x) < info.tiny
  x1 = _where(cond, x * _lax_const(x, 1 << info.nmant), x)
  x2 = _where(cond, int_type(-info.nmant), int_type(0))
  return lax.bitcast_convert_type(x1, int_type), x2


@_wraps(np.ldexp)
@jit
def ldexp(x1, x2):
  _check_arraylike("ldexp", x1, x2)
  x1_dtype = dtypes.dtype(x1)
  x2_dtype = dtypes.dtype(x2)
  if (dtypes.issubdtype(x1_dtype, np.complexfloating)
      or dtypes.issubdtype(x2_dtype, np.inexact)):
    raise ValueError(f"ldexp not supported for input types {(x1_dtype, x2_dtype)}")

  x1, x2 = _promote_shapes("ldexp", x1, x2)

  dtype = dtypes.canonicalize_dtype(dtypes._to_inexact_dtype(x1_dtype))
  info = dtypes.finfo(dtype)
  int_type = _INT_DTYPES[info.bits]

  x1 = lax.convert_element_type(x1, dtype)
  x2 = lax.convert_element_type(x2, int_type)

  mask = (1 << info.nexp) - 1
  bias = ((1 << info.nexp) - 1) >> 1
  x, e = _normalize_float(x1)
  x2 += e + ((x >> info.nmant) & mask) - bias

  # find underflow/overflow before denormalization
  underflow_cond = x2 < -(bias + info.nmant)
  overflow_cond = x2 > bias

  m = lax.full_like(x, 1, dtype=dtype)

  # denormals
  cond = x2 < -bias + 1
  x2 = _where(cond, x2 + info.nmant, x2)
  m = _where(cond, m / (1 << info.nmant), m)

  x2 = lax.convert_element_type(x2, np.int32)
  x &= ~(mask << info.nmant)
  x |= ((lax.convert_element_type(x2, int_type) + bias) << info.nmant)

  x = lax.convert_element_type(m, dtype) * lax.bitcast_convert_type(x, dtype)

  # underflow
  x = _where(underflow_cond, lax.full_like(x, 0, dtype=dtype), x)
  # overflow
  x = _where(overflow_cond, lax.sign(x1) * lax.full_like(x, np.inf), x)
  # ldexp(x1, x2) = x1 for x1 = inf, -inf, nan, 0
  return _where(isinf(x1) | isnan(x1) | (x1 == 0), x1, x)


@_wraps(np.frexp)
@jit
def frexp(x):
  _check_arraylike("frexp", x)
  x, = _promote_dtypes_inexact(x)
  if dtypes.issubdtype(x.dtype, np.complexfloating):
    raise TypeError("frexp does not support complex-valued inputs")

  dtype = dtypes.dtype(x)
  info = dtypes.finfo(dtype)
  mask = (1 << info.nexp) - 1
  bias = ((1 << info.nexp) - 1) >> 1

  x1, x2 = _normalize_float(x)
  x2 += ((x1 >> info.nmant) & mask) - bias + 1
  x1 &= ~(mask << info.nmant)
  x1 |= (bias - 1) << info.nmant
  x1 = lax.bitcast_convert_type(x1, dtype)

  cond = isinf(x) | isnan(x) | (x == 0)
  x2 = _where(cond, lax_internal._zeros(x2), x2)
  return _where(cond, x, x1), lax.convert_element_type(x2, np.int32)


@_wraps(np.remainder)
@jit
def remainder(x1, x2):
  x1, x2 = _promote_args("remainder", x1, x2)
  zero = _constant_like(x1, 0)
  trunc_mod = lax.rem(x1, x2)
  trunc_mod_not_zero = lax.ne(trunc_mod, zero)
  do_plus = lax.bitwise_and(
      lax.ne(lax.lt(trunc_mod, zero), lax.lt(x2, zero)), trunc_mod_not_zero)
  return lax.select(do_plus, lax.add(trunc_mod, x2), trunc_mod)
mod = _wraps(np.mod)(remainder)


@_wraps(np.fmod)
@jit
def fmod(x1, x2):
  _check_arraylike("fmod", x1, x2)
  if dtypes.issubdtype(dtypes.result_type(x1, x2), np.integer):
    x2 = _where(x2 == 0, lax_internal._ones(x2), x2)
  return lax.rem(*_promote_args("fmod", x1, x2))


@_wraps(np.square)
@partial(jit, inline=True)
def square(x):
  _check_arraylike("square", x)
  return lax.integer_pow(x, 2)


@_wraps(np.deg2rad)
@partial(jit, inline=True)
def deg2rad(x):
  x, = _promote_args_inexact("deg2rad", x)
  return lax.mul(x, _lax_const(x, np.pi / 180))


@_wraps(np.rad2deg)
@partial(jit, inline=True)
def rad2deg(x):
  x, = _promote_args_inexact("rad2deg", x)
  return lax.mul(x, _lax_const(x, 180 / np.pi))


degrees = rad2deg
radians = deg2rad


@_wraps(np.conjugate)
@partial(jit, inline=True)
def conjugate(x):
  _check_arraylike("conjugate", x)
  return lax.conj(x) if np.iscomplexobj(x) else x
conj = conjugate


@_wraps(np.imag)
@partial(jit, inline=True)
def imag(val):
  _check_arraylike("imag", val)
  return lax.imag(val) if np.iscomplexobj(val) else lax.full_like(val, 0)


@_wraps(np.real)
@partial(jit, inline=True)
def real(val):
  _check_arraylike("real", val)
  return lax.real(val) if np.iscomplexobj(val) else val

@_wraps(np.modf, skip_params=['out'])
@jit
def modf(x, out=None):
  _check_arraylike("modf", x)
  x, = _promote_dtypes_inexact(x)
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.modf is not supported.")
  whole = _where(lax.ge(x, lax_internal._zero(x)), floor(x), ceil(x))
  return x - whole, whole


@_wraps(np.isfinite)
@jit
def isfinite(x):
  _check_arraylike("isfinite", x)
  dtype = dtypes.dtype(x)
  if dtypes.issubdtype(dtype, np.floating):
    return lax.is_finite(x)
  elif dtypes.issubdtype(dtype, np.complexfloating):
    return lax.bitwise_and(lax.is_finite(real(x)), lax.is_finite(imag(x)))
  else:
    return lax.full_like(x, True, dtype=np.bool_)


@_wraps(np.isinf)
@jit
def isinf(x):
  _check_arraylike("isinf", x)
  dtype = dtypes.dtype(x)
  if dtypes.issubdtype(dtype, np.floating):
    return lax.eq(lax.abs(x), _constant_like(x, np.inf))
  elif dtypes.issubdtype(dtype, np.complexfloating):
    re = lax.real(x)
    im = lax.imag(x)
    return lax.bitwise_or(lax.eq(lax.abs(re), _constant_like(re, np.inf)),
                          lax.eq(lax.abs(im), _constant_like(im, np.inf)))
  else:
    return lax.full_like(x, False, dtype=np.bool_)


def _isposneginf(infinity, x, out):
  if out is not None:
    raise NotImplementedError("The 'out' argument to isneginf/isposinf is not supported.")
  dtype = dtypes.dtype(x)
  if dtypes.issubdtype(dtype, np.floating):
    return lax.eq(x, _constant_like(x, infinity))
  elif dtypes.issubdtype(dtype, np.complexfloating):
    raise ValueError("isposinf/isneginf are not well defined for complex types")
  else:
    return lax.full_like(x, False, dtype=np.bool_)


isposinf = _wraps(np.isposinf, skip_params=['out'])(
  lambda x, out=None: _isposneginf(np.inf, x, out)
)


isneginf = _wraps(np.isneginf, skip_params=['out'])(
  lambda x, out=None: _isposneginf(-np.inf, x, out)
)


@_wraps(np.isnan)
@jit
def isnan(x):
  _check_arraylike("isnan", x)
  return lax.ne(x, x)


@_wraps(np.heaviside)
@jit
def heaviside(x1, x2):
  _check_arraylike("heaviside", x1, x2)
  x1, x2 = _promote_dtypes_inexact(x1, x2)
  zero = _lax_const(x1, 0)
  return _where(lax.lt(x1, zero), zero,
                _where(lax.gt(x1, zero), _lax_const(x1, 1), x2))


@_wraps(np.hypot)
@jit
def hypot(x1, x2):
  _check_arraylike("hypot", x1, x2)
  x1, x2 = _promote_dtypes_inexact(x1, x2)
  x1 = lax.abs(x1)
  x2 = lax.abs(x2)
  x1, x2 = maximum(x1, x2), minimum(x1, x2)
  return lax.select(x1 == 0, x1, x1 * lax.sqrt(1 + lax.square(lax.div(x2, lax.select(x1 == 0, lax_internal._ones(x1), x1)))))


@_wraps(np.reciprocal)
@partial(jit, inline=True)
def reciprocal(x):
  _check_arraylike("reciprocal", x)
  x, = _promote_dtypes_inexact(x)
  return lax.integer_pow(x, -1)


@_wraps(np.sinc, update_doc=False)
@jit
def sinc(x):
  _check_arraylike("sinc", x)
  x, = _promote_dtypes_inexact(x)
  eq_zero = lax.eq(x, _lax_const(x, 0))
  pi_x = lax.mul(_lax_const(x, np.pi), x)
  safe_pi_x = _where(eq_zero, _lax_const(x, 1), pi_x)
  return _where(eq_zero, _sinc_maclaurin(0, pi_x),
                lax.div(lax.sin(safe_pi_x), safe_pi_x))


@partial(custom_jvp, nondiff_argnums=(0,))
def _sinc_maclaurin(k, x):
  # compute the kth derivative of x -> sin(x)/x evaluated at zero (since we
  # compute the monomial term in the jvp rule)
  # TODO(mattjj): see https://github.com/google/jax/issues/10750
  if k % 2:
    return x * 0
  else:
    return x * 0 + _lax_const(x, (-1) ** (k // 2) / (k + 1))

@_sinc_maclaurin.defjvp
def _sinc_maclaurin_jvp(k, primals, tangents):
  (x,), (t,) = primals, tangents
  return _sinc_maclaurin(k, x), _sinc_maclaurin(k + 1, x) * t


class ufunc:
  """Functions that operate element-by-element on whole arrays.

  This is a class for LAX-backed implementations of numpy ufuncs.
  """
  identity = None
  nargs = None
  nin = None
  nout = None
  _call = None
  _reduce = None
  _accumulate = None
  _at = None
  _reduceat = None

  def __init__(self, name):
    self.__name__ = name

  def __repr__(self):
    return f"<jnp.ufunc '{self.__name__}'>"

  def __call__(self, *args, **kwargs):
    return self._call(*args, **kwargs)

  @_wraps(np.ufunc.reduce)
  def reduce(self, a, axis=0, dtype=None, out=None, keepdims=False, initial=None, where=True):
    if self.nin != 2:
      raise ValueError("reduce only supported for binary ufuncs")
    if self.nout != 1:
      raise ValueError("reduce only supported for functions returning a single value")
    if out is not None:
      raise NotImplementedError(f"out argument of {self.__name__}.reduce()")
    _reduce = self._reduce or self._reduce_via_scan
    return _reduce(a, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)

  @partial(jit, static_argnames=['self', 'axis', 'dtype', 'keepdims', 'where'])
  def _reduce_via_scan(self, arr, axis=0, dtype=None, keepdims=False, initial=None, where=True):
    assert self.nin == 2 and self.nout == 1
    _check_arraylike(f"{self.__name__}.reduce", arr)
    arr = _asarray(arr)
    if initial is None:
      initial = self.identity
    if dtype is None:
      dtype = jax.eval_shape(self, lax_internal._one(arr), lax_internal._one(arr)).dtype

    if isinstance(axis, tuple):
      axis = tuple(canonicalize_axis(a, arr.ndim) for a in axis)
      raise NotImplementedError("tuple of axes")
    elif axis is None:
      if keepdims:
        final_shape = (1,) * arr.ndim
      else:
        final_shape = ()
      arr = arr.ravel()
      axis = 0
    else:
      axis = canonicalize_axis(axis, arr.ndim)
      if keepdims:
        final_shape = (*arr.shape[:axis], 1, *arr.shape[axis + 1:])
      else:
        final_shape = (*arr.shape[:axis], *arr.shape[axis + 1:])

    # TODO: handle without transpose?
    if axis != 0:
      arr = moveaxis(arr, axis, 0)

    if where is not True:
      raise NotImplementedError("where argument")
    if initial is None and arr.shape[0] == 0:
      raise ValueError("zero-size array to reduction operation {self.__name__} which has no ideneity")

    def body_fun(i, val):
      return self(val, arr[i].astype(dtype))

    if initial is None:
      start = 1
      initial = arr[0]
    else:
      _check_arraylike(f"{self.__name__}.reduce", arr)
      start = 0

    initial = _broadcast_to(_asarray(initial, dtype=dtype), arr.shape[1:])

    result = lax.fori_loop(start, arr.shape[0], body_fun, initial)
    if keepdims:
      result = result.reshape(final_shape)
    return result

  @_wraps(np.ufunc.accumulate)
  def accumulate(self, a, axis=0, dtype=None, out=None):
    if self.nin != 2:
      raise ValueError("accumulate only supported for binary ufuncs")
    if self.nout != 1:
      raise ValueError("accumulate only supported for functions returning a single value")
    if out is not None:
      raise NotImplementedError(f"out argument of {self.__name__}.accumulate()")
    _accumulate = self._accumulate or self._accumulate_via_scan
    return _accumulate(a, axis=axis, dtype=dtype)

  @partial(jit, static_argnames=['self', 'axis', 'dtype'])
  def _accumulate_via_scan(self, arr, axis=0, dtype=None):
    assert self.nin == 2 and self.nout == 1
    _check_arraylike(f"{self.__name__}.accumulate", arr)
    arr = _asarray(arr)

    if dtype is None:
      dtype = jax.eval_shape(self, lax_internal._one(arr), lax_internal._one(arr)).dtype

    if axis is None or isinstance(axis, tuple):
      raise ValueError("accumulate does not allow multiple axes")
    axis = canonicalize_axis(axis, np.ndim(arr))

    arr = moveaxis(arr, axis, 0)
    def scan_fun(carry, _):
      i, x = carry
      y = _where(i == 0, arr[0].astype(dtype), self(x.astype(dtype), arr[i].astype(dtype)))
      return (i + 1, y), y
    _, result = lax.scan(scan_fun, (0, arr[0].astype(dtype)), None, length=arr.shape[0])
    return moveaxis(result, 0, axis)

  @_wraps(np.ufunc.accumulate)
  def at(self, a, indices, b=None, inplace=True):
    if inplace:
      raise NotImplementedError(_AT_INPLACE_WARNING)
    _at = self._at or self._at_via_scan
    if b is None:
      return _at(a, indices)
    else:
      return _at(a, indices, b)

  def _at_via_scan(self, a, indices, *args):
    _check_arraylike(f"{self.__name__}.at", a, *args)
    dtype = jax.eval_shape(self, lax_internal._one(a), *(lax_internal._one(arg) for arg in args)).dtype
    a = _asarray(a, dtype=dtype)
    args = tuple(_asarray(arg, dtype=dtype) for arg in args)
    indices = _eliminate_deprecated_list_indexing(indices)
    if not indices:
      return a

    shapes = [np.shape(i) for i in indices if not isinstance(i, slice)]
    shape = shapes and lax.broadcast_shapes(*shapes)
    if not shape:
      return a.at[indices].set(self(a.at[indices].get(), *args))

    args = tuple(_broadcast_to(arg, shape).ravel() for arg in args)
    indices = [idx if isinstance(idx, slice) else _broadcast_to(idx, shape).ravel() for idx in indices]

    def scan_fun(carry, x):
      i, a = carry
      idx = tuple(ind if isinstance(ind, slice) else ind[i] for ind in indices)
      a = a.at[idx].set(self(a.at[idx].get(), *(arg[i] for arg in args)))
      return (i + 1, a), x
    carry, _ = lax.scan(scan_fun, (0, a), None, len(indices[0]))
    return carry[1]

  @_wraps(np.ufunc.reduceat)
  def reduceat(self, a, indices, axis=0, dtype=None, out=None):
    if self.nin != 2:
      raise ValueError("reduceat only supported for binary ufuncs")
    if self.nout != 1:
      raise ValueError("reduceat only supported for functions returning a single value")
    if self._reduceat is None:
      raise NotImplementedError(f"reduceat() method of {self}")
    if out is not None:
      raise NotImplementedError(f"out argument of {self.__name__}.reduceat()")
    return self._reduceat(a, indices, axis=axis, dtype=dtype)

  @_wraps(np.ufunc.outer)
  def outer(self, A, B, **kwargs):
    if self.nin != 2:
      raise ValueError("outer only supported for binary ufuncs")
    if self.nout != 1:
      raise ValueError("outer only supported for functions returning a single value")
    _check_arraylike(f"{self.__name__}.outer", A, B)
    _ravel = lambda A: lax.reshape(A, (np.size(A),))
    result = vmap(vmap(partial(self, **kwargs), (None, 0)), (0, None))(_ravel(A), _ravel(B))
    return result.reshape(*np.shape(A), *np.shape(B))

def _make_reduceat(name, segment_op, promote_bool):
  def reduceat(a, indices, axis, dtype):
    int_ = dtypes.canonicalize_dtype(dtypes.int_)
    if a.ndim != 1:
      # TODO(jakevdp) implement this
      raise NotImplementedError(f"jnp.{name}.reduceat(a) for a.ndim > 1")
    assert axis in [-1, 0]
    segments = lax.full((a.shape[axis],), 0, dtype=int_).at[indices].add(1).cumsum()
    if dtype is not None:
      a = a.astype(dtype)
    elif promote_bool and a.dtype == bool:
      a = a.astype(int_)
    return getattr(jax.ops, segment_op)(a, segments, num_segments=len(indices) + 1)[1:]
  return reduceat

def _make_at(name):
  def at(a, indices, b):
    _check_arraylike(f"{name}.at", a, b)
    return getattr(_asarray(a).at[indices], name)(b)
  return at

def _make_ufunc(name, *, nargs, nin, nout, call, identity=None, reduce=None,
                accumulate=None, at=None, reduceat=None):
  uf = ufunc(name)
  uf.nargs = nargs
  uf.nin = nin
  uf.nout = nout
  uf.identity = identity
  uf._call = call
  uf._reduce = reduce
  uf._accumulate = accumulate
  uf._at = at
  uf._reduceat = reduceat
  return _wraps(getattr(np, name))(uf)

def _min_with_dtype(a, *, axis, dtype, keepdims, initial, where):
  if dtype is not None: a = a.astype(dtype)
  if where is True: where = None  # Ensure default where is not traced
  return reductions.min(a, axis=axis, keepdims=keepdims, initial=initial, where=where)

def _max_with_dtype(a, *, axis, dtype, keepdims, initial, where):
  if dtype is not None: a = a.astype(dtype)
  if where is True: where = None  # Ensure default where is not traced
  return reductions.max(a, axis=axis, keepdims=keepdims, initial=initial, where=where)

bitwise_not = _make_ufunc('invert', nargs=1, nin=1, nout=1, call=_bitwise_not)
fabs = _make_ufunc('fabs', nargs=1, nin=1, nout=1, call=_fabs)
invert = _make_ufunc('invert', nargs=1, nin=1, nout=1, call=_invert)
negative = _make_ufunc('negative', nargs=1, nin=1, nout=1, call=_negative)
positive = _make_ufunc('negative', nargs=1, nin=1, nout=1, call=_positive)
floor = _make_ufunc('negative', nargs=1, nin=1, nout=1, call=_floor)
ceil = _make_ufunc('negative', nargs=1, nin=1, nout=1, call=_ceil)

add = _make_ufunc('add', nargs=2, nin=2, nout=1, identity=0,
                  call=_add, accumulate=reductions.cumsum, reduce=reductions.sum,
                  at=_make_at('add'), reduceat=_make_reduceat('add', 'segment_sum', True))
# TODO(jakevdp): use fast monoidal reductions for bitwise_and, bitwise_or, bitwise_xor.
bitwise_and = _make_ufunc('bitwise_and', nargs=2, nin=2, nout=1, identity=-1, call=_bitwise_and)
bitwise_or = _make_ufunc('bitwise_or', nargs=2, nin=2, nout=1, identity=0, call=_bitwise_or)
bitwise_xor = _make_ufunc('bitwise_xor', nargs=2, nin=2, nout=1, identity=0, call=_bitwise_xor)
subtract = _make_ufunc('subtract', nargs=2, nin=2, nout=1, call=_subtract)
maximum = _make_ufunc('maximum', nargs=2, nin=2, nout=1, call=_maximum, reduce=_max_with_dtype,
                      at=_make_at('max'), reduceat=_make_reduceat('max', 'segment_max', False))
minimum = _make_ufunc('minimum', nargs=2, nin=2, nout=1, call=_minimum, reduce=_min_with_dtype,
                      at=_make_at('min'), reduceat=_make_reduceat('min', 'segment_min', False))
multiply = _make_ufunc('multiply', nargs=2, nin=2, nout=1, identity=1,
                       call=_multiply, accumulate=reductions.cumprod, reduce=reductions.prod,
                       at=_make_at('mul'), reduceat=_make_reduceat('multiply', 'segment_prod', True))
true_divide = _make_ufunc('divide', nargs=2, nin=2, nout=1, call=_true_divide)
divide = true_divide
floor_divide = _make_ufunc('floor_divide', nargs=2, nin=2, nout=1, call=_floor_divide)
