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
Implements the NumPy API, using the primitives in :mod:`jax.lax`.

NumPy operations are implemented in Python in terms of the primitive operations
in :mod:`jax.lax`. Since NumPy operations are not primitive and instead are
implemented in terms of :mod:`jax.lax` operations, we do not need to define
transformation rules such as gradient or batching rules. Instead,
transformations for NumPy primitives can be derived from the transformation
rules for the underlying :code:`lax` primitives.
"""

import builtins
import collections
import operator
import os
import types
from typing import Sequence, Set, Tuple, Union
import warnings

import numpy as np
import opt_einsum

from jax import jit, custom_jvp
from .vectorize import vectorize
from ._util import _wraps
from .. import core
from .. import dtypes
from ..abstract_arrays import UnshapedArray, ShapedArray, ConcreteArray, canonicalize_shape
from ..config import flags
from ..interpreters.xla import (DeviceArray, device_put, array_result_handler,
                                DeviceValue, abstractify)
from ..interpreters.masking import Poly
from .. import lax
from .. import ops
from ..util import (partial, unzip2, prod as _prod,
                    subvals, safe_zip)
from ..tree_util import tree_leaves, tree_flatten

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    'jax_numpy_rank_promotion', os.getenv('JAX_NUMPY_RANK_PROMOTION', 'allow'),
    enum_values=['allow', 'warn', 'raise'],
    help=
    'Control NumPy-style automatic rank promotion broadcasting '
    '("allow", "warn", or "raise").')

newaxis = None

# Common docstring additions:

_PRECISION_DOC = """\
In addition to the original NumPy arguments listed below, also supports
``precision`` for extra control over matrix-multiplication precision
on supported devices. ``precision`` may be set to ``None``, which means
default precision for the backend, or any ``jax.lax.Precision`` enum value
(``Precision.DEFAULT``, ``Precision.HIGH`` or ``Precision.HIGHEST``).
"""

# We replace some builtin names to follow Numpy's API, so we capture here.
_abs = builtins.abs
_all = builtins.all
_any = builtins.any
_max = builtins.max
_min = builtins.min
_sum = builtins.sum
_divmod = builtins.divmod

# NumPy constants

pi = np.pi
e = np.e
euler_gamma = np.euler_gamma
inf = np.inf
NINF = np.NINF
PZERO = np.PZERO
NZERO = np.NZERO
nan = np.nan

# And some numpy utility functions
set_printoptions = np.set_printoptions

# We want isinstance(x, np.ndarray) checks in user code to work with the our
# array-like types, including DeviceArray and UnshapedArray (i.e. the abstract
# array base class). We can override the isinstance behavior directly, without
# having the complexity of multiple inheritance on those classes, by defining
# the ndarray class to have a metaclass with special __instancecheck__ behavior.
_arraylike_types = (np.ndarray, UnshapedArray, DeviceArray)

class _ArrayMeta(type(np.ndarray)):  # type: ignore
  """Metaclass for overriding ndarray isinstance checks."""

  def __instancecheck__(self, instance):
    try:
      return isinstance(instance.aval, _arraylike_types)
    except AttributeError:
      return isinstance(instance, _arraylike_types)

class ndarray(np.ndarray, metaclass=_ArrayMeta):
  dtype: np.dtype
  shape: Tuple[int, ...]
  size: int

  def __init__(shape, dtype=None, buffer=None, offset=0, strides=None,
               order=None):
    raise TypeError("jax.numpy.ndarray() should not be instantiated explicitly."
                    " Use jax.numpy.array, or jax.numpy.zeros instead.")


iscomplexobj = np.iscomplexobj

shape = _shape = np.shape
ndim = _ndim = np.ndim
size = np.size
_dtype = dtypes.result_type

# At present JAX doesn't have a reason to distinguish between scalars and arrays
# in its object system. Further, we want JAX scalars to have the same type
# promotion behaviors as JAX arrays. Rather than introducing a new type of JAX
# scalar object with JAX promotion behaviors, instead we make the JAX scalar
# types return JAX arrays when instantiated.

class _ScalarMeta(type):
  def __hash__(self):
    return hash(self.dtype.type)

  def __eq__(self, other):
    return id(self) == id(other) or self.dtype.type == other

  def __ne__(self, other):
    return not (self == other)

  def __call__(self, x):
    return array(x, dtype=self.dtype)

def _make_scalar_type(np_scalar_type):
  return _ScalarMeta(np_scalar_type.__name__, (object,),
                     {"dtype": np.dtype(np_scalar_type)})

bool_ = _make_scalar_type(np.bool_)
uint8 = _make_scalar_type(np.uint8)
uint16 = _make_scalar_type(np.uint16)
uint32 = _make_scalar_type(np.uint32)
uint64 = _make_scalar_type(np.uint64)
int8 = _make_scalar_type(np.int8)
int16 = _make_scalar_type(np.int16)
int32 = _make_scalar_type(np.int32)
int64 = _make_scalar_type(np.int64)
bfloat16 = _make_scalar_type(dtypes.bfloat16)
float16 = _make_scalar_type(np.float16)
float32 = single = _make_scalar_type(np.float32)
float64 = double = _make_scalar_type(np.float64)
complex64 = csingle = _make_scalar_type(np.complex64)
complex128 = cdouble = _make_scalar_type(np.complex128)

int_ = int32 if dtypes.int_ == np.int32 else int64
float_ = float32 if dtypes.float_ == np.float32 else float64
complex_ = complex64 if dtypes.complex_ == np.complex64 else complex128

number = np.number
inexact = np.inexact
complexfloating = np.complexfloating
floating = np.floating
integer = np.integer
signedinteger = np.signedinteger
unsignedinteger = np.unsignedinteger

flexible = np.flexible
character = np.character
object_ = np.object_

iinfo = dtypes.iinfo

dtype = np.dtype
can_cast = dtypes.can_cast
issubsctype = dtypes.issubsctype
promote_types = dtypes.promote_types

ComplexWarning = np.ComplexWarning

array_str = np.array_str
array_repr = np.array_repr

save = np.save
savez = np.savez
load = np.load


### utility functions

_canonicalize_axis = lax._canonicalize_axis

_DEFAULT_TYPEMAP = {
  np.bool_: bool_,
  np.int_: int_,
  np.float_: float_,
  np.complex_: complex_
}

def _np_array(obj, dtype=None, **kwargs):
  """Return a properly-typed numpy array.

  `_np_array(obj, **kwds)` is equivalent to `np.array(obj, **kwds)`, with the
  exception that when obj.dtype is not defined and dtype is not specified, it
  uses Jax's default dtypes.
  """
  arr = np.array(obj, dtype=dtype, **kwargs)
  obj_dtype = getattr(obj, 'dtype', None)
  arr_dtype = np.dtype(arr.dtype).type
  if dtype is None and obj_dtype is None and arr_dtype in _DEFAULT_TYPEMAP:
    arr = arr.astype(_DEFAULT_TYPEMAP[arr_dtype])
  return arr

_np_asarray = partial(_np_array, copy=False)

def _promote_shapes(fun_name, *args):
  """Prepend implicit leading singleton dimensions for Numpy broadcasting."""
  if len(args) < 2:
    return args
  else:
    shapes = [shape(arg) for arg in args]
    nonscalar_ranks = [len(shp) for shp in shapes if shp]
    if not nonscalar_ranks or len(set(nonscalar_ranks)) == 1:
      return args
    else:
      if FLAGS.jax_numpy_rank_promotion != "allow":
        _rank_promotion_warning_or_error(fun_name, shapes)
      result_rank = len(lax.broadcast_shapes(*shapes))
      return [broadcast_to(arg, (1,) * (result_rank - len(shp)) + shp)
              for arg, shp in zip(args, shapes)]

def _rank_promotion_warning_or_error(fun_name, shapes):
  if FLAGS.jax_numpy_rank_promotion == "warn":
    msg = ("Following NumPy automatic rank promotion for {} on shapes {}. "
           "Set the jax_numpy_rank_promotion config option to 'allow' to "
           "disable this warning; for more information, see "
           "https://jax.readthedocs.io/en/latest/rank_promotion_warning.html.")
    warnings.warn(msg.format(fun_name, ' '.join(map(str, shapes))))
  elif FLAGS.jax_numpy_rank_promotion == "raise":
    msg = ("Operands could not be broadcast together for {} on shapes {} "
           "and with the config option jax_numpy_rank_promotion='raise'. "
           "For more information, see "
           "https://jax.readthedocs.io/en/latest/rank_promotion_warning.html.")
    raise ValueError(msg.format(fun_name, ' '.join(map(str, shapes))))

def _promote_dtypes(*args):
  """Convenience function to apply Numpy argument dtype promotion."""
  # TODO(dougalm,mattjj): This is a performance bottleneck. Consider memoizing.
  if len(args) < 2:
    return args
  else:
    to_dtype = result_type(*args)
    return [lax.convert_element_type(x, to_dtype) for x in args]

def _promote_dtypes_inexact(*args):
  """Convenience function to apply Numpy argument dtype promotion.

  Promotes arguments to an inexact type."""
  to_dtype = _to_inexact_dtype(result_type(*args))
  return [lax.convert_element_type(x, to_dtype) for x in args]


def _to_inexact_dtype(dtype):
  """Promotes a dtype into an inexact dtype, if it is not already one."""
  return dtype if issubdtype(dtype, inexact) else promote_types(dtype, float_)

def _complex_elem_type(dtype):
  """Returns the float type of the real/imaginary parts of a complex dtype."""
  return np.abs(np.zeros((), dtype)).dtype

def _result_dtype(op, *args):
  """Compute result dtype of applying op to arguments with given dtypes."""
  args = [np.ones((0,) * ndim(arg), _dtype(arg)) for arg in args]
  return _dtype(op(*args))


def _arraylike(x): return isinstance(x, ndarray) or isscalar(x)
def _check_arraylike(fun_name, *args):
  """Check if all args fit JAX's definition of arraylike (ndarray or scalar)."""
  if _any(not _arraylike(arg) for arg in args):
    pos, arg = next((i, arg) for i, arg in enumerate(args)
                    if not _arraylike(arg))
    msg = "{} requires ndarray or scalar arguments, got {} at position {}."
    raise TypeError(msg.format(fun_name, type(arg), pos))


def _promote_args(fun_name, *args):
  """Convenience function to apply Numpy argument shape and dtype promotion."""
  _check_arraylike(fun_name, *args)
  return _promote_shapes(fun_name, *_promote_dtypes(*args))

def _promote_args_inexact(fun_name, *args):
  """Convenience function to apply Numpy argument shape and dtype promotion.

  Promotes non-inexact types to an inexact type."""
  _check_arraylike(fun_name, *args)
  return _promote_shapes(fun_name, *_promote_dtypes_inexact(*args))

def _constant_like(x, const):
  return np.array(const, dtype=_dtype(x))

### implementations of numpy functions in terms of lax

@_wraps(np.fmin)
def fmin(x1, x2):
  return where((x1 < x2) | isnan(x2), x1, x2)

@_wraps(np.fmax)
def fmax(x1, x2):
  return where((x1 > x2) | isnan(x2), x1, x2)

@_wraps(np.finfo)
def finfo(dtype):
  return dtypes.finfo(dtype)

@_wraps(np.issubdtype)
def issubdtype(arg1, arg2):
  return dtypes.issubdtype(arg1, arg2)

@_wraps(np.isscalar)
def isscalar(element):
  return dtypes.is_python_scalar(element) or np.isscalar(element)

iterable = np.iterable

@_wraps(np.result_type)
def result_type(*args):
  return dtypes.result_type(*args)

def _one_to_one_unop(numpy_fn, lax_fn, promote_to_inexact=False):
  if promote_to_inexact:
    def fn(x):
      x = lax.convert_element_type(x, _to_inexact_dtype(_dtype(x)))
      return lax_fn(x)
  else:
    fn = lambda x: lax_fn(x)
  return _wraps(numpy_fn)(fn)

def _one_to_one_binop(numpy_fn, lax_fn, promote_to_inexact=False):
  if promote_to_inexact:
    fn = lambda x1, x2: lax_fn(*_promote_args_inexact(numpy_fn, x1, x2))
  else:
    fn = lambda x1, x2: lax_fn(*_promote_args(numpy_fn.__name__, x1, x2))
  return _wraps(numpy_fn)(fn)

def _maybe_bool_binop(numpy_fn, lax_fn, bool_lax_fn):
  def fn(x1, x2):
    x1, x2 = _promote_args(numpy_fn.__name__, x1, x2)
    return lax_fn(x1, x2) if x1.dtype != bool_ else bool_lax_fn(x1, x2)
  return _wraps(numpy_fn)(fn)

absolute = abs = _one_to_one_unop(np.absolute, lax.abs)
fabs = _one_to_one_unop(np.fabs, lax.abs, True)
bitwise_not = _one_to_one_unop(np.bitwise_not, lax.bitwise_not)
negative = _one_to_one_unop(np.negative, lax.neg)
positive = _one_to_one_unop(np.positive, lambda x: x)

floor = _one_to_one_unop(np.floor, lax.floor, True)
ceil = _one_to_one_unop(np.ceil, lax.ceil, True)
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
arcsinh = _one_to_one_unop(np.arcsinh, lax.asinh, True)
arccosh = _one_to_one_unop(np.arccosh, lax.acosh, True)
arctanh = _one_to_one_unop(np.arctanh, lax.atanh, True)
sqrt = _one_to_one_unop(np.sqrt, lax.sqrt, True)


add = _maybe_bool_binop(np.add, lax.add, lax.bitwise_or)
bitwise_and = _one_to_one_binop(np.bitwise_and, lax.bitwise_and)
bitwise_or = _one_to_one_binop(np.bitwise_or, lax.bitwise_or)
bitwise_xor = _one_to_one_binop(np.bitwise_xor, lax.bitwise_xor)
right_shift = _one_to_one_binop(np.right_shift, lax.shift_right_arithmetic)
left_shift = _one_to_one_binop(np.left_shift, lax.shift_left)
equal = _one_to_one_binop(np.equal, lax.eq)
multiply = _maybe_bool_binop(np.multiply, lax.mul, lax.bitwise_and)
not_equal = _one_to_one_binop(np.not_equal, lax.ne)
subtract = _one_to_one_binop(np.subtract, lax.sub)
arctan2 = _one_to_one_binop(np.arctan2, lax.atan2, True)
minimum = _one_to_one_binop(np.minimum, lax.min)
maximum = _one_to_one_binop(np.maximum, lax.max)
float_power = _one_to_one_binop(np.float_power, lax.pow, True)
nextafter = _one_to_one_binop(np.nextafter, lax.nextafter, True)


def _comparison_op(numpy_fn, lax_fn):
  def fn(x1, x2):
    x1, x2 =  _promote_args(numpy_fn.__name__, x1, x2)
    # Comparison on complex types are defined as a lexicographic ordering on
    # the (real, imag) pair.
    if issubdtype(_dtype(x1), complexfloating):
      rx = lax.real(x1)
      ry = lax.real(x2)
      return lax.select(lax.eq(rx, ry), lax_fn(lax.imag(x1), lax.imag(x2)),
                        lax_fn(rx, ry))
    return lax_fn(x1, x2)
  return _wraps(numpy_fn)(fn)

greater_equal = _comparison_op(np.greater_equal, lax.ge)
greater = _comparison_op(np.greater, lax.gt)
less_equal = _comparison_op(np.less_equal, lax.le)
less = _comparison_op(np.less, lax.lt)


def _logical_op(np_op, bitwise_op):
  @_wraps(np_op, update_doc=False)
  def op(*args):
    zero = lambda x: lax.full_like(x, shape=(), fill_value=0)
    args = (x if issubdtype(_dtype(x), bool_) else lax.ne(x, zero(x))
            for x in args)
    return bitwise_op(*_promote_args(np_op.__name__, *args))
  return op

logical_and = _logical_op(np.logical_and, lax.bitwise_and)
logical_not = _logical_op(np.logical_not, lax.bitwise_not)
logical_or = _logical_op(np.logical_or, lax.bitwise_or)
logical_xor = _logical_op(np.logical_xor, lax.bitwise_xor)


@_wraps(np.rint)
def rint(x):
  dtype = _dtype(x)
  if issubdtype(dtype, integer):
    return lax.convert_element_type(x, float_)
  if issubdtype(dtype, complexfloating):
    return lax.complex(rint(lax.real(x)), rint(lax.imag(x)))
  return _round_to_nearest_even(x)


@_wraps(np.sign)
def sign(x):
  dtype = _dtype(x)
  if issubdtype(dtype, complexfloating):
    re = lax.real(x)
    return lax.complex(
      lax.sign(where(re != 0, re, lax.imag(x))), _constant_like(re, 0))
  return lax.sign(x)


@_wraps(np.copysign)
def copysign(x1, x2):
  if issubdtype(_dtype(x1), complexfloating) or issubdtype(_dtype(x2), complexfloating):
    raise TypeError("copysign does not support complex-valued inputs")
  x1, x2 = _promote_shapes("copysign", x1, x2)
  return where(signbit(x2), -lax.abs(x1), lax.abs(x1))


@_wraps(np.true_divide)
def true_divide(x1, x2):
  x1, x2 = _promote_args_inexact("true_divide", x1, x2)
  return lax.div(x1, x2)


@_wraps(np.divide)
def divide(x1, x2):
  # decide whether to perform integer division based on Numpy result dtype, as a
  # way to check whether Python 3 style division is active in Numpy
  result_dtype = _result_dtype(np.divide, x1, x2)
  if issubdtype(result_dtype, integer):
    return floor_divide(x1, x2)
  else:
    return true_divide(x1, x2)


@_wraps(np.floor_divide)
def floor_divide(x1, x2):
  x1, x2 = _promote_args("floor_divide", x1, x2)
  dtype = _dtype(x1)
  if issubdtype(dtype, integer):
    quotient = lax.div(x1, x2)
    select = logical_and(lax.sign(x1) != lax.sign(x2), lax.rem(x1, x2) != 0)
    # TODO(mattjj): investigate why subtracting a scalar was causing promotion
    return where(select, quotient - np.array(1, _dtype(quotient)), quotient)
  elif issubdtype(dtype, complexfloating):
    x1r = lax.real(x1)
    x1i = lax.imag(x1)
    x2r = lax.real(x2)
    x2i = lax.imag(x2)
    which = lax.ge(lax.abs(x2r), lax.abs(x2i))
    rat1 = where(which, lax._const(x2i, 1), lax.div(x2r, x2i))
    rat2 = where(which, lax.div(x2i, x2r), lax._const(x2i, 1))
    out = lax.floor(lax.div(lax.add(lax.mul(x1r, rat1), lax.mul(x1i, rat2)),
                            lax.add(lax.mul(x2r, rat1), lax.mul(x2i, rat2))))
    return lax.convert_element_type(out, dtype)
  else:
    return _float_divmod(x1, x2)[0]


@_wraps(np.divmod)
def divmod(x1, x2):
  x1, x2 = _promote_args("divmod", x1, x2)
  if issubdtype(_dtype(x1), integer):
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


@_wraps(np.power)
def power(x1, x2):
  # Special case for small positive integer scalars: use binary exponentiation.
  # Using lax.pow may be imprecise for floating-point values; the goal of this
  # code path is to make sure we end up with a precise output for the common
  # pattern ``x ** 2`` or similar.
  if isinstance(x2, int):
    return lax.integer_pow(x1, x2)

  x1, x2 = _promote_args(np.power, x1, x2)
  dtype = _dtype(x1)
  if not issubdtype(dtype, integer):
    return lax.pow(x1, x2)

  # Integer power => use binary exponentiation.

  # TODO(phawkins): add integer pow support to XLA.
  bits = 6  # Anything more would overflow for any x1 > 1
  acc = ones(shape(x1), dtype=dtype)
  for _ in range(bits):
    acc = where(lax.bitwise_and(x2, _constant_like(x2, 1)),
                lax.mul(acc, x1), acc)
    x1 = lax.mul(x1, x1)
    x2 = lax.shift_right_logical(x2, _constant_like(x2, 1))
  return acc


@custom_jvp
@_wraps(np.logaddexp)
def logaddexp(x1, x2):
  x1, x2 = _promote_shapes("logaddexp", *_promote_dtypes_inexact(x1, x2))
  amax = lax.max(x1, x2)
  delta = lax.sub(x1, x2)
  return lax.select(isnan(delta),
                    lax.add(x1, x2),  # NaNs or infinities of the same sign.
                    lax.add(amax, lax.log1p(lax.exp(-lax.abs(delta)))))

@logaddexp.defjvp
def _logaddexp_jvp(primals, tangents):
  x1, x2 = primals
  t1, t2 = tangents
  x1, x2, t1, t2 = broadcast_arrays(x1, x2, t1, t2)
  primal_out = logaddexp(x1, x2)
  tangent_out = (t1 * exp(_replace_inf(x1) - _replace_inf(primal_out)) +
                 t2 * exp(_replace_inf(x2) - _replace_inf(primal_out)))
  return primal_out, tangent_out

def _replace_inf(x):
  return lax.select(isposinf(x), zeros_like(x), x)


@custom_jvp
@_wraps(np.logaddexp2)
def logaddexp2(x1, x2):
  x1, x2 = _promote_shapes("logaddexp2", *_promote_dtypes_inexact(x1, x2))
  amax = lax.max(x1, x2)
  delta = lax.sub(x1, x2)
  return lax.select(isnan(delta),
                    lax.add(x1, x2),  # NaNs or infinities of the same sign.
                    lax.add(amax, lax.div(lax.log1p(exp2(-lax.abs(delta))),
                                          _constant_like(x1, np.log(2)))))
@logaddexp2.defjvp
def _logaddexp2_jvp(primals, tangents):
  x1, x2 = primals
  t1, t2 = tangents
  x1, x2, t1, t2 = broadcast_arrays(x1, x2, t1, t2)
  primal_out = logaddexp2(x1, x2)
  tangent_out = (t1 * 2 ** (_replace_inf(x1) - _replace_inf(primal_out)) +
                 t2 * 2 ** (_replace_inf(x2) - _replace_inf(primal_out)))
  return primal_out, tangent_out


@_wraps(np.log2)
def log2(x):
  x, = _promote_dtypes_inexact(x)
  return lax.div(lax.log(x), lax.log(_constant_like(x, 2)))


@_wraps(np.log10)
def log10(x):
  x, = _promote_dtypes_inexact(x)
  return lax.div(lax.log(x), lax.log(_constant_like(x, 10)))


@_wraps(np.exp2)
def exp2(x):
  x, = _promote_dtypes_inexact(x)
  return lax.exp(lax.mul(lax.log(_constant_like(x, 2)), x))

@_wraps(np.signbit)
def signbit(x):
  x, = _promote_shapes("signbit", x)
  dtype = _dtype(x)
  if issubdtype(dtype, integer):
    return lax.lt(x, _constant_like(x, 0))
  elif issubdtype(dtype, bool_):
    return full_like(x, False, dtype=bool_)
  elif not issubdtype(dtype, floating):
    raise ValueError(
        "jax.numpy.signbit is not well defined for %s" % dtype)

  # TPU supports BF16 but not S16 types, so as a workaround, convert BF16 to
  # F32.
  if dtype == bfloat16:
    dtype = float32
    x = lax.convert_element_type(x, float32)

  info = finfo(dtype)
  if info.bits == 16:
    int_type = np.int16
  elif info.bits == 32:
    int_type = np.int32
  elif info.bits == 64:
    int_type = np.int64
  else:
    raise NotImplementedError(
        "jax.numpy.signbit only supports 16, 32, and 64-bit types.")

  x = lax.bitcast_convert_type(x, int_type)
  return lax.convert_element_type(x >> (info.nexp + info.nmant), np.bool_)



@_wraps(np.trapz)
def trapz(y, x=None, dx=1.0, axis=-1):
  y = moveaxis(y, axis, -1)
  if x is not None:
    if ndim(x) == 1:
      dx = diff(x)
    else:
      dx = moveaxis(diff(x, axis=axis), axis, -1)
  return 0.5 * (dx * (y[..., 1:] + y[..., :-1])).sum(-1)


@_wraps(np.trunc)
def trunc(x):
  return where(lax.lt(x, lax._const(x, 0)), lax.ceil(x), lax.floor(x))


def _conv(x, y, mode, op, precision):
  if issubdtype(x.dtype, complexfloating) or issubdtype(y.dtype, complexfloating):
    raise NotImplementedError(f"{op}() does not support complex inputs")
  if ndim(x) != 1 or ndim(y) != 1:
    raise ValueError(f"{op}() only support 1-dimensional inputs.")
  x, y = _promote_dtypes_inexact(x, y)
  if len(x) == 0 or len(y) == 0:
    raise ValueError(f"{op}: inputs cannot be empty, got shapes {x.shape} and {y.shape}.")

  out_order = slice(None)
  if len(x) < len(y):
    x, y = y, x
    if op == "correlate":
      out_order = slice(None, None, -1)
  if op == 'convolve':
    y = y[::-1]

  if mode == 'valid':
    padding = [(0, 0)]
  elif mode == 'same':
    padding = [(y.shape[0] // 2, y.shape[0] - y.shape[0] // 2 - 1)]
  elif mode == 'full':
    padding = [(y.shape[0] - 1, y.shape[0] - 1)]
  else:
    raise ValueError("mode must be one of ['full', 'same', 'valid']")

  result = lax.conv_general_dilated(x[None, None, :], y[None, None, :], (1,),
                                    padding, precision=precision)
  return result[0, 0, out_order]


@_wraps(np.convolve, lax_description=_PRECISION_DOC)
def convolve(a, v, mode='full', *, precision=None):
  return _conv(a, v, mode, 'convolve', precision)


@_wraps(np.correlate, lax_description=_PRECISION_DOC)
def correlate(a, v, mode='valid', *, precision=None):
  return _conv(a, v, mode, 'correlate', precision)


def _normalize_float(x):
    info = finfo(_dtype(x))
    cond = lax.abs(x) < info.tiny
    x1 = where(cond, x * (1 << info.nmant), x)
    x2 = where(cond,
               full_like(x, -info.nmant, dtype=np.int32),
               zeros_like(x, dtype=np.int32))
    return lax.convert_element_type(x1, _dtype(x)), x2

_INT_DTYPES = {
  16: np.int16,
  32: np.int32,
  64: np.int64,
}

@_wraps(np.ldexp)
@jit
def ldexp(x1, x2):
  dtype = _result_dtype(np.ldexp, x1, x2)
  x1, x2 = _promote_shapes("ldexp", x1, x2)
  x1 = lax.convert_element_type(x1, dtype)

  info = finfo(dtype)
  mask = (1 << info.nexp) - 1
  bias = ((1 << info.nexp) - 1) >> 1

  int_type = _INT_DTYPES[info.bits]

  x, e = _normalize_float(x1)
  x2 += lax.convert_element_type(e, np.int32)
  x = lax.bitcast_convert_type(x, int_type)
  x2 += ((x >> info.nmant) & mask) - bias

  # find underflow/overflow before denormalization
  underflow_cond = x2 < -(bias + info.nmant)
  overflow_cond = x2 > bias

  m = ones_like(x, dtype=dtype)

  # denormals
  cond = x2 < -bias + 1
  x2 = where(cond, x2 + info.nmant, x2)
  m = where(cond, m / (1 << info.nmant), m)

  x2 = lax.convert_element_type(x2, np.int32)
  x &= ~(mask << info.nmant)
  x |= ((lax.convert_element_type(x2, int_type) + bias) << info.nmant)

  x = lax.convert_element_type(m, dtype) * lax.bitcast_convert_type(x, dtype)

  # underflow
  x = where(underflow_cond, zeros_like(x, dtype=dtype), x)
  # overflow
  x = where(overflow_cond, lax.sign(x1) * full_like(x, np.inf), x)
  # ldexp(x1, x2) = x1 for x1 = inf, -inf, nan, 0
  return where(isinf(x1) | isnan(x1) | (x1 == 0), x1, x)


@_wraps(np.frexp)
@jit
def frexp(x):
  x = asarray(x)
  if issubdtype(x.dtype, complexfloating):
    raise TypeError("frexp does not support complex-valued inputs")
  elif not issubdtype(x.dtype, floating):
    x = lax.convert_element_type(x, float_)

  dtype = _dtype(x)
  info = finfo(dtype)
  mask = (1 << info.nexp) - 1
  bias = ((1 << info.nexp) - 1) >> 1

  int_type = _INT_DTYPES[info.bits]

  x1, x2 = _normalize_float(x)
  x1 = lax.bitcast_convert_type(x1, int_type)
  x2 += ((x1 >> info.nmant) & mask) - bias + 1
  x1 &= ~(mask << info.nmant)
  x1 |= (bias - 1) << info.nmant
  x1 = lax.bitcast_convert_type(x1, dtype)

  cond = isinf(x) | isnan(x) | (x == 0)
  x2 = where(cond, zeros_like(x2), x2)
  return where(cond, x, x1), lax.convert_element_type(x2, int32)


@_wraps(np.remainder)
def remainder(x1, x2):
  x1, x2 = _promote_args("remainder", x1, x2)
  zero = _constant_like(x1, 0)
  trunc_mod = lax.rem(x1, x2)
  trunc_mod_not_zero = lax.ne(trunc_mod, zero)
  do_plus = lax.bitwise_and(
      lax.ne(lax.lt(trunc_mod, zero), lax.lt(x2, zero)), trunc_mod_not_zero)
  return lax.select(do_plus, lax.add(trunc_mod, x2), trunc_mod)
mod = remainder
fmod = _wraps(np.fmod)(lambda x1, x2: lax.rem(x1, x2))


@_wraps(np.cbrt)
def cbrt(x):
  x, = _promote_dtypes_inexact(x)
  return lax.sign(x) * power(lax.abs(x), _constant_like(x, 1. / 3.))


@_wraps(np.square)
def square(x): return lax.integer_pow(x, 2)


@_wraps(np.deg2rad)
def deg2rad(x):
  x, = _promote_dtypes_inexact(x)
  return lax.mul(x, lax._const(x, pi / 180))


@_wraps(np.rad2deg)
def rad2deg(x):
  x, = _promote_dtypes_inexact(x)
  return lax.mul(x, lax._const(x, 180 / pi))


degrees = rad2deg
radians = deg2rad


@_wraps(np.histogram_bin_edges)
def histogram_bin_edges(a, bins=10, range=None, weights=None):
  if isinstance(bins, str):
    raise NotImplementedError("string values for `bins` not implemented.")
  a = ravel(a)
  b = array(bins)
  if b.ndim == 1:
    return b
  if range is None:
    range = (a.min(), a.max())
  assert len(range) == 2
  range = asarray(range)
  range = (where(ptp(range) == 0, range[0] - 0.5, range[0]),
           where(ptp(range) == 0, range[1] + 0.5, range[1]))
  dtype = _dtype(a)
  if issubdtype(dtype, integer):
    dtype = promote_types(dtype, float32)
  return linspace(range[0], range[1], bins + 1, dtype=dtype)


@_wraps(np.histogram)
def histogram(a, bins=10, range=None, weights=None, density=None):
  if weights is not None and a.shape != weights.shape:
    raise ValueError("weights should have the same shape as a.")
  a = ravel(a)
  if weights is not None:
    weights = ravel(weights)
  else:
    weights = ones_like(a)
  bin_edges = histogram_bin_edges(a, bins, range, weights)
  bin_idx = searchsorted(bin_edges, a, side='right')
  bin_idx = where(a == bin_edges[-1], len(bin_edges) - 1, bin_idx)
  counts = bincount(bin_idx, weights, length=len(bin_edges))[1:]
  if density:
    bin_widths = diff(bin_edges)
    counts = counts / bin_widths / counts.sum()
  return counts, bin_edges


@_wraps(np.heaviside)
def heaviside(x1, x2):
  x1, x2 = _promote_dtypes_inexact(x1, x2)
  zero = lax._const(x1, 0)
  return where(lax.lt(x1, zero), zero,
               where(lax.gt(x1, zero), lax._const(x1, 1), x2))


@_wraps(np.hypot)
def hypot(x1, x2):
  x1, x2 = _promote_dtypes_inexact(x1, x2)
  return lax.sqrt(x1*x1 + x2*x2)


@_wraps(np.reciprocal)
def reciprocal(x):
  x, = _promote_dtypes_inexact(x)
  return lax.integer_pow(x, -1)


@_wraps(np.sinc, update_doc=False)
def sinc(x):
  x, = _promote_dtypes_inexact(x)
  eq_zero = lax.eq(x, lax._const(x, 0))
  safe_x = where(eq_zero, lax._const(x, 0), x)
  pi_x = lax.mul(lax._const(x, pi), safe_x)
  return where(eq_zero,
               lax._const(x, 1), lax.div(lax.sin(pi_x), pi_x))


@_wraps(np.transpose)
def transpose(a, axes=None):
  axes = np.arange(ndim(a))[::-1] if axes is None else axes
  return lax.transpose(a, axes)


@_wraps(np.rot90)
def rot90(m, k=1, axes=(0, 1)):
  ax1, ax2 = axes
  ax1 = _canonicalize_axis(ax1, m.ndim)
  ax2 = _canonicalize_axis(ax2, m.ndim)
  if ax1 == ax2:
    raise ValueError("Axes must be different")  # same as numpy error
  k = k % 4
  if k == 0:
    return m
  elif k == 2:
    return flip(flip(m, ax1), ax2)
  else:
    perm = list(range(m.ndim))
    perm[ax1], perm[ax2] = perm[ax2], perm[ax1]
    if k == 1:
      return transpose(flip(m, ax2), perm)
    else:
      return flip(transpose(m, perm), ax2)


@_wraps(np.flip)
def flip(m, axis=None):
  if axis is None:
    return lax.rev(m, list(range(len(m.shape))))
  return lax.rev(m, [_canonicalize_axis(axis, len(m.shape))])


@_wraps(np.fliplr)
def fliplr(m):
  return flip(m, 1)


@_wraps(np.flipud)
def flipud(m):
  return flip(m, 0)


@_wraps(np.conjugate)
def conjugate(x):
  return lax.conj(x) if iscomplexobj(x) else x
conj = conjugate


@_wraps(np.imag)
def imag(val):
  return lax.imag(val) if iscomplexobj(val) else zeros_like(val)


@_wraps(np.real)
def real(val):
  return lax.real(val) if iscomplexobj(val) else val


@_wraps(np.iscomplex)
def iscomplex(x):
  i = imag(x)
  return lax.ne(i, lax._const(i, 0))

@_wraps(np.isreal)
def isreal(x):
  i = imag(x)
  return lax.eq(i, lax._const(i, 0))

@_wraps(np.angle)
def angle(z):
  re = real(z)
  im = imag(z)
  dtype = _dtype(re)
  if not issubdtype(dtype, inexact) or (
      issubdtype(_dtype(z), floating) and ndim(z) == 0):
    dtype = dtypes.canonicalize_dtype(float_)
    re = lax.convert_element_type(re, dtype)
    im = lax.convert_element_type(im, dtype)
  return lax.atan2(im, re)


@_wraps(np.diff)
def diff(a, n=1, axis=-1,):
  if not isinstance(a, ndarray) or a.ndim == 0:
    return a
  if n == 0:
    return a
  if n < 0:
    raise ValueError(
      "order must be non-negative but got " + repr(n))

  nd = a.ndim

  slice1 = [slice(None)] * nd
  slice2 = [slice(None)] * nd
  slice1[axis] = slice(1, None)
  slice2[axis] = slice(None, -1)
  slice1 = tuple(slice1)
  slice2 = tuple(slice2)

  op = not_equal if a.dtype == np.bool_ else subtract
  for _ in range(n):
    a = op(a[slice1], a[slice2])

  return a

_EDIFF1D_DOC = """\
Unlike NumPy's implementation of ediff1d, :py:func:`jax.numpy.ediff1d` will not
issue an error if casting ``to_end`` or ``to_begin`` to the type of ``ary``
loses precision.
"""

@_wraps(np.ediff1d, lax_description=_EDIFF1D_DOC)
def ediff1d(ary, to_end=None, to_begin=None):
  ary = ravel(asarray(ary))
  result = lax.sub(ary[1:], ary[:-1])
  if to_begin is not None:
    result = concatenate((ravel(asarray(to_begin, dtype=ary.dtype)), result))
  if to_end is not None:
    result = concatenate((result, ravel(asarray(to_end, dtype=ary.dtype))))
  return result


@partial(jit, static_argnums=2)
def _gradient(a, varargs, axis):
  def gradient_along_axis(a, h, axis):
    sliced = partial(lax.slice_in_dim, a, axis=axis)
    a_grad = concatenate((
      (sliced(1, 2) - sliced(0, 1)),  # upper edge
      (sliced(2, None) - sliced(None, -2)) * 0.5,  # inner
      (sliced(-1, None) - sliced(-2, -1)),  # lower edge
    ), axis)
    return a_grad / h

  if axis is None:
    axis = range(a.ndim)
  else:
    if isinstance(axis, int):
      axis = (axis,)
    if not isinstance(axis, tuple) and not isinstance(axis, list):
      raise ValueError("Give `axis` either as int or iterable")
    elif len(axis) == 0:
      return []
    axis = [_canonicalize_axis(i, a.ndim) for i in axis]

  if _min([s for i, s in enumerate(a.shape) if i in axis]) < 2:
    raise ValueError("Shape of array too small to calculate "
                     "a numerical gradient, "
                     "at least 2 elements are required.")
  len_axes = len(axis)
  n = len(varargs)
  if n == 0 or varargs is None:
    # no spacing
    dx = [1.0] * len_axes
  elif n == 1:
    # single value for all axes
    dx = varargs * len_axes
  elif n == len_axes:
    dx = varargs
  else:
    TypeError("Invalid number of spacing arguments %d" % n)

  if ndim(dx[0]) != 0:
    raise NotImplementedError("Non-constant spacing not implemented")

  # TODO: use jax.lax loop tools if possible
  a_grad = [gradient_along_axis(a, h, ax) for ax, h in zip(axis, dx)]

  if len(axis) == 1:
    a_grad = a_grad[0]

  return a_grad


@_wraps(np.gradient)
def gradient(f, *args, **kwargs):
  axis = kwargs.pop("axis", None)
  if not len(kwargs) == 0:
    raise ValueError("Only `axis` keyword is implemented")
  return _gradient(f, args, axis)


@_wraps(np.isrealobj)
def isrealobj(x):
  return not iscomplexobj(x)


@_wraps(np.reshape)
def reshape(a, newshape, order="C"):
  try:
    return a.reshape(newshape, order=order)  # forward to method for ndarrays
  except AttributeError:
    return _reshape(a, newshape, order=order)

def _compute_newshape(a, newshape):
  """Fixes a -1 value in newshape, if present."""
  # other errors, like having more than one -1, are caught downstream
  newsize = _prod(newshape)
  if newsize < 0:
    fix = a.size // -newsize
    return [d if d != -1 else fix for d in newshape]
  else:
    return newshape

def _reshape(a, newshape, order="C"):
  computed_newshape = _compute_newshape(a, newshape)
  if order == "C":
    return lax.reshape(a, computed_newshape, None)
  elif order == "F":
    dims = np.arange(ndim(a))[::-1]
    return lax.reshape(a, computed_newshape[::-1], dims).T
  elif order == "A":
    raise NotImplementedError("np.reshape order=A is not implemented.")
  else:
    raise ValueError("Unexpected value for 'order' argument: {}.".format(order))

def _reshape_method(a, *newshape, **kwargs):
  order = kwargs.pop("order", "C")
  if len(kwargs) == 1:
    invalid_kwarg, = kwargs
    msg = "'{}' is an invalid keyword argument for this function"
    raise TypeError(msg.format(invalid_kwarg))  # same as NumPy error
  elif kwargs:
    invalid_kwargs = "'{}'".format("'".join(kwargs))
    msg = "{} are invalid keyword arguments for this function"
    raise TypeError(msg.format(invalid_kwargs))  # different from NumPy error
  if len(newshape) == 1 and not isinstance(newshape[0], int):
    newshape = newshape[0]
  return _reshape(a, newshape, order=order)


@_wraps(np.ravel)
def ravel(a, order="C"):
  if order == "K":
    raise NotImplementedError("Ravel not implemented for order='K'.")
  return reshape(a, (size(a),), order)


_UNRAVEL_INDEX_DOC = """\
Unlike numpy's implementation of unravel_index, negative indices are accepted
and out-of-bounds indices are clipped.
"""

@_wraps(np.unravel_index, lax_description=_UNRAVEL_INDEX_DOC)
def unravel_index(indices, shape):
  indices = asarray(indices)
  sizes = pad(shape, (0, 1), constant_values=1)
  cumulative_sizes = cumprod(sizes[::-1])[::-1]
  total_size = cumulative_sizes[0]
  # Clip so raveling and unraveling an oob index will not change the behavior
  clipped_indices = clip(indices, -total_size, total_size - 1)
  # Add enough trailing dims to avoid conflict with flat_index
  cumulative_sizes = cumulative_sizes.reshape([-1] + [1] * indices.ndim)
  idx = clipped_indices % cumulative_sizes[:-1] // cumulative_sizes[1:]
  return tuple(idx)


@_wraps(np.squeeze)
def squeeze(a, axis: Union[int, Tuple[int, ...]] = None):
  if axis is None:
    a_shape = shape(a)
    axis = tuple(i for i, d in enumerate(a_shape) if d == 1)
  elif not isinstance(axis, tuple):
    axis = (axis,)
  return lax.squeeze(a, axis)


@_wraps(np.expand_dims)
def expand_dims(a, axis: Union[int, Tuple[int, ...]]):
  if not isinstance(axis, tuple):
    axis = (axis,)
  return lax.expand_dims(a, axis)


@_wraps(np.swapaxes)
def swapaxes(a, axis1, axis2):
  perm = np.arange(ndim(a))
  perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
  return lax.transpose(a, perm)


@_wraps(np.moveaxis)
def moveaxis(a, source, destination):
  if isinstance(source, int):
    source = (source,)
  if isinstance(destination, int):
    destination = (destination,)
  source = tuple(_canonicalize_axis(i, ndim(a)) for i in source)
  destination = tuple(_canonicalize_axis(i, ndim(a)) for i in destination)
  if len(source) != len(destination):
    raise ValueError("Inconsistent number of elements: {} vs {}"
                     .format(len(source), len(destination)))
  perm = [i for i in range(ndim(a)) if i not in source]
  for dest, src in sorted(zip(destination, source)):
    perm.insert(dest, src)
  return lax.transpose(a, perm)


@_wraps(np.isclose)
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
  a, b = _promote_args("isclose", asarray(a), asarray(b))
  dtype = _dtype(a)
  if issubdtype(dtype, inexact):
    if issubdtype(dtype, complexfloating):
      dtype = _complex_elem_type(dtype)
    rtol = lax.convert_element_type(rtol, dtype)
    atol = lax.convert_element_type(atol, dtype)
    out = lax.le(
      lax.abs(lax.sub(a, b)),
      lax.add(atol, lax.mul(rtol, lax.abs(b))))
    # This corrects the comparisons for infinite and nan values
    a_inf = isinf(a)
    b_inf = isinf(b)
    any_inf = logical_or(a_inf, b_inf)
    both_inf = logical_and(a_inf, b_inf)
    # Make all elements where either a or b are infinite to False
    out = logical_and(out, logical_not(any_inf))
    # Make all elements where both a or b are the same inf to True
    same_value = lax.eq(a, b)
    same_inf = logical_and(both_inf, same_value)
    out = logical_or(out, same_inf)

    # Make all elements where either a or b is NaN to False
    a_nan = isnan(a)
    b_nan = isnan(b)
    any_nan = logical_or(a_nan, b_nan)
    out = logical_and(out, logical_not(any_nan))
    if equal_nan:
      # Make all elements where both a and b is NaN to True
      both_nan = logical_and(a_nan, b_nan)
      out = logical_or(out, both_nan)
    return _maybe_numpy_1_13_isclose_behavior(a, out)
  else:
    return lax.eq(a, b)

numpy_version = tuple(map(int, np.version.version.split('.')[:2]))
if numpy_version < (1, 14):
  # see discussion at https://github.com/numpy/numpy/pull/9720
  def _maybe_numpy_1_13_isclose_behavior(a, out):
    if size(out) == 1 and issubdtype(_dtype(a), complexfloating):
      return lax.reshape(out, (1,))
    else:
      return out
else:
  def _maybe_numpy_1_13_isclose_behavior(a, out):
    return out


@_wraps(np.in1d, lax_description="""
In the JAX version, the `assume_unique` argument is not referenced.
""")
def in1d(ar1, ar2, assume_unique=False, invert=False):
  # TODO(vanderplas): use sorting-based approach for larger inputs.
  ar1 = ravel(ar1)
  ar2 = ravel(ar2)
  if invert:
    return (ar1[:, None] != ar2).all(-1)
  else:
    return (ar1[:, None] == ar2).any(-1)

@partial(jit, static_argnums=2)
def _intersect1d_sorted_mask(ar1, ar2, return_indices=False):
    """
    Helper function for intersect1d which is jit-able
    """
    ar = concatenate((ar1, ar2))
    if return_indices:
      iota = lax.broadcasted_iota(np.int64, shape(ar), dimension=0)
      aux, indices = lax.sort_key_val(ar, iota)
    else:
      aux = sort(ar)

    mask = aux[1:] == aux[:-1]
    if return_indices:
      return aux, mask, indices
    else:
      return aux, mask

@_wraps(np.intersect1d)
def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):

  if not assume_unique:
    if return_indices:
      ar1, ind1 = unique(ar1, return_index=True)
      ar2, ind2 = unique(ar2, return_index=True)
    else:
      ar1 = unique(ar1)
      ar2 = unique(ar2)
  else:
    ar1 = ravel(ar1)
    ar2 = ravel(ar2)

  if return_indices:
    aux, mask, aux_sort_indices = _intersect1d_sorted_mask(ar1, ar2, return_indices)
  else:
    aux, mask = _intersect1d_sorted_mask(ar1, ar2, return_indices)

  int1d = aux[:-1][mask]

  if return_indices:
    ar1_indices = aux_sort_indices[:-1][mask]
    ar2_indices = aux_sort_indices[1:][mask] - ar1.size
    if not assume_unique:
      ar1_indices = ind1[ar1_indices]
      ar2_indices = ind2[ar2_indices]

    return int1d, ar1_indices, ar2_indices
  else:
    return int1d


@_wraps(np.isin, lax_description="""
In the JAX version, the `assume_unique` argument is not referenced.
""")
def isin(element, test_elements, assume_unique=False, invert=False):
  result = in1d(element, test_elements, assume_unique=assume_unique, invert=invert)
  return result.reshape(shape(element))


# The `jit` on `where` exists to avoid materializing constants in cases like
# `np.where(np.zeros(1000), 7, 4)`. In op-by-op mode, we don't want to
# materialize the broadcast forms of scalar arguments.
@jit
def _where(condition, x=None, y=None):
  if x is None or y is None:
    raise ValueError("Either both or neither of the x and y arguments should "
                     "be provided to jax.numpy.where, got {} and {}."
                     .format(x, y))
  if not issubdtype(_dtype(condition), bool_):
    condition = lax.ne(condition, zeros_like(condition))
  x, y = _promote_dtypes(x, y)
  condition, x, y = broadcast_arrays(condition, x, y)
  return lax.select(condition, x, y) if np.size(x) else x


_WHERE_DOC = """\
At present, JAX does not support JIT-compilation of the single-argument form
of :py:func:`jax.numpy.where` because its output shape is data-dependent. The
three-argument form does not have a data-dependent shape and can be JIT-compiled
successfully.
"""

@_wraps(np.where, update_doc=False, lax_description=_WHERE_DOC)
def where(condition, x=None, y=None):
  if x is None and y is None:
    return nonzero(asarray(condition))
  else:
    return _where(condition, x, y)


@_wraps(np.select)
def select(condlist, choicelist, default=0):
  if len(condlist) != len(choicelist):
    msg = "condlist must have length equal to choicelist ({} vs {})"
    raise ValueError(msg.format(len(condlist), len(choicelist)))
  if len(condlist) == 0:
    raise ValueError("condlist must be non-empty")
  choices = _promote_dtypes(default, *choicelist)
  choicelist = choices[1:]
  output = choices[0]
  for cond, choice in zip(condlist[::-1], choicelist[::-1]):
    output = where(cond, choice, output)
  return output


@_wraps(np.bincount, lax_description="""\
Jax adds the optional `length` parameter which specifies the output length, and
defaults to ``x.max() + 1``. It must be specified for bincount to be compilable.
Values larger than the specified length will be discarded.

Additionally, while ``np.bincount`` raises an error if the input array contains
negative values, ``jax.numpy.bincount`` treats negative values as zero.
""")
def bincount(x, weights=None, minlength=0, *, length=None):
  if not issubdtype(_dtype(x), integer):
    msg = f"x argument to bincount must have an integer type; got {x.dtype}"
    raise TypeError(msg)
  if length is None:
    length = max(x) + 1
  length = _max(length, minlength)
  if ndim(x) != 1:
    raise ValueError("only 1-dimensional input supported.")
  if weights is None:
    weights = array(1, dtype=int32)
  else:
    if shape(x) != shape(weights):
      raise ValueError("shape of weights must match shape of x.")
  return ops.index_add(zeros((length,), _dtype(weights)), ops.index[clip(x, 0)], weights)


def broadcast_arrays(*args):
  """Like Numpy's broadcast_arrays but doesn't return views."""
  shapes = [shape(arg) for arg in args]
  if len(set(shapes)) == 1:
    return [arg if isinstance(arg, ndarray) or isscalar(arg) else array(arg)
            for arg in args]
  result_shape = lax.broadcast_shapes(*shapes)
  return [broadcast_to(arg, result_shape) for arg in args]


@_wraps(np.broadcast_to, lax_description="""\
The JAX version does not necessarily return a view of the input.
""")
def broadcast_to(arr, shape):
  arr = arr if isinstance(arr, ndarray) else array(arr)
  shape = canonicalize_shape(shape)  # check that shape is concrete
  arr_shape = _shape(arr)
  if arr_shape == shape:
    return arr
  else:
    nlead = len(shape) - len(arr_shape)
    compatible = np.equal(arr_shape, shape[nlead:]) | np.equal(arr_shape, 1)
    if nlead < 0 or not np.all(compatible):
      msg = "Incompatible shapes for broadcasting: {} and requested shape {}"
      raise ValueError(msg.format(arr_shape, shape))
    diff, = np.where(np.not_equal(shape[nlead:], arr_shape))
    new_dims = tuple(range(nlead)) + tuple(nlead + diff)
    kept_dims = tuple(np.delete(np.arange(len(shape)), new_dims))
    return lax.broadcast_in_dim(squeeze(arr, tuple(diff)), shape, kept_dims)


@_wraps(np.split)
def split(ary, indices_or_sections, axis=0):
  axis = core.concrete_or_error(int, axis, "in jax.numpy.split argument `axis`")
  size = ary.shape[axis]
  if isinstance(indices_or_sections, (tuple, list) + _arraylike_types):
    indices_or_sections = [core.concrete_or_error(int, i_s, "in jax.numpy.split argument 1")
                           for i_s in indices_or_sections]
    split_indices = np.concatenate([[0], indices_or_sections, [size]])
  else:
    indices_or_sections = core.concrete_or_error(int, indices_or_sections,
                                                 "in jax.numpy.split argument 1")
    part_size, r = _divmod(size, indices_or_sections)
    if r != 0:
      raise ValueError("array split does not result in an equal division")
    split_indices = np.arange(indices_or_sections + 1) * part_size
  starts, ends = [0] * ndim(ary), shape(ary)
  _subval = lambda x, i, v: subvals(x, [(i, v)])
  return [lax.slice(ary, _subval(starts, axis, start), _subval(ends, axis, end))
          for start, end in zip(split_indices[:-1], split_indices[1:])]

def _split_on_axis(np_fun, axis):
  @_wraps(np_fun, update_doc=False)
  def f(ary, indices_or_sections):
    return split(ary, indices_or_sections, axis=axis)
  return f

vsplit = _split_on_axis(np.vsplit, axis=0)
hsplit = _split_on_axis(np.hsplit, axis=1)
dsplit = _split_on_axis(np.dsplit, axis=2)


@_wraps(np.clip)
def clip(a, a_min=None, a_max=None):
  if a_min is None and a_max is None:
    raise ValueError("At most one of a_min and a_max may be None")
  if a_min is not None:
    if _dtype(a_min) != _dtype(a):
      a_min = lax.convert_element_type(a_min, _dtype(a))
    a = maximum(a_min, a)
  if a_max is not None:
    if _dtype(a_max) != _dtype(a):
      a_max = lax.convert_element_type(a_max, _dtype(a))
    a = minimum(a_max, a)
  return a


def _round_to_nearest_even(x):
  half = lax._const(x, 0.5)
  one = lax._const(x, 1)
  round_val = lax.floor(x)
  fraction = x - round_val
  nearest_even_int = lax.sub(
    round_val, lax.mul(lax._const(x, 2), lax.floor(lax.mul(half, x))))
  is_odd = lax.eq(nearest_even_int, one)
  return lax.select(
    lax.bitwise_or(lax.gt(fraction, half),
                   lax.bitwise_and(lax.eq(fraction, half), is_odd)),
    lax.add(round_val, one), round_val)

@_wraps(np.round, update_doc=False)
def round(a, decimals=0):
  dtype = _dtype(a)
  if issubdtype(dtype, integer):
    if decimals < 0:
      raise NotImplementedError(
        "integer np.round not implemented for decimals < 0")
    return a  # no-op on integer types

  def _round_float(x):
    if decimals == 0:
      return _round_to_nearest_even(x)

    # TODO(phawkins): the strategy of rescaling the value isn't necessarily a
    # good one since we may be left with an incorrectly rounded value at the
    # end due to precision problems. As a workaround for float16, convert to
    # float32,
    x = lax.convert_element_type(x, np.float32) if dtype == np.float16 else x
    factor = _constant_like(x, 10 ** decimals)
    out = lax.div(_round_to_nearest_even(lax.mul(x, factor)), factor)
    return lax.convert_element_type(out, dtype) if dtype == np.float16 else out

  if issubdtype(dtype, complexfloating):
    return lax.complex(_round_float(lax.real(a)), _round_float(lax.imag(a)))
  else:
    return _round_float(a)
around = round


@_wraps(np.fix)
def fix(x, out=None):
  if out is not None:
    raise ValueError("fix does not support the `out` argument.")
  zero = lax._const(x, 0)
  return where(lax.ge(x, zero), lax.floor(x), lax.ceil(x))

@_wraps(np.isfinite)
def isfinite(x):
  dtype = _dtype(x)
  if issubdtype(dtype, floating):
    return lax.is_finite(x)
  elif issubdtype(dtype, complexfloating):
    return lax.bitwise_and(lax.is_finite(real(x)), lax.is_finite(imag(x)))
  else:
    return full_like(x, True, dtype=bool_)

@_wraps(np.isinf)
def isinf(x):
  dtype = _dtype(x)
  if issubdtype(dtype, floating):
    return lax.eq(lax.abs(x), _constant_like(x, inf))
  elif issubdtype(dtype, complexfloating):
    re = lax.real(x)
    im = lax.imag(x)
    return lax.bitwise_or(lax.eq(lax.abs(re), _constant_like(re, inf)),
                          lax.eq(lax.abs(im), _constant_like(im, inf)))
  else:
    return full_like(x, False, dtype=bool_)

def _isposneginf(infinity, x):
  dtype = _dtype(x)
  if issubdtype(dtype, floating):
    return lax.eq(x, _constant_like(x, infinity))
  elif issubdtype(dtype, complexfloating):
    raise ValueError("isposinf/isneginf are not well defined for complex types")
  else:
    return full_like(x, False, dtype=bool_)

isposinf = _wraps(np.isposinf)(lambda x: _isposneginf(inf, x))

isneginf = _wraps(np.isneginf)(lambda x: _isposneginf(-inf, x))

@_wraps(np.isnan)
def isnan(x):
  return lax.bitwise_and(lax.bitwise_not(isfinite(x)),
                         lax.bitwise_not(isinf(x)))

@_wraps(np.nan_to_num)
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
  del copy
  dtype = _dtype(x)
  if issubdtype(dtype, complexfloating):
    return lax.complex(
      nan_to_num(lax.real(x), nan=nan, posinf=posinf, neginf=neginf),
      nan_to_num(lax.imag(x), nan=nan, posinf=posinf, neginf=neginf))
  info = finfo(dtypes.canonicalize_dtype(dtype))
  posinf = info.max if posinf is None else posinf
  neginf = info.min if neginf is None else neginf
  x = where(isnan(x), _constant_like(x, nan), x)
  x = where(isposinf(x), _constant_like(x, posinf), x)
  x = where(isneginf(x), _constant_like(x, neginf), x)
  return x

### Reducers


def _make_reduction(np_fun, op, init_val, preproc=None, bool_op=None,
                    upcast_f16_for_computation=False):
  """Creates reduction function given a binary operation and monoid identity."""

  bool_op = bool_op or op

  @_wraps(np_fun)
  def reduction(a, axis=None, dtype=None, out=None, keepdims=False):
    if out is not None:
      raise ValueError("reduction does not support the `out` argument.")

    if isinstance(a, (list, tuple)):
      msg = ("jax.numpy reductions won't accept lists and tuples in future "
             "versions, only scalars and ndarrays")
      warnings.warn(msg, category=FutureWarning)
    a = a if isinstance(a, ndarray) else asarray(a)
    a = preproc(a) if preproc else a
    dims = _reduction_dims(a, axis)
    result_dtype = dtype or _dtype(np_fun(np.ones((), dtype=_dtype(a))))
    if upcast_f16_for_computation and issubdtype(result_dtype, inexact):
      computation_dtype = promote_types(result_dtype, float32)
    else:
      computation_dtype = result_dtype
    a = lax.convert_element_type(a, computation_dtype)
    result = lax.reduce(a, _reduction_init_val(a, init_val),
                        op if computation_dtype != np.bool_ else bool_op, dims)
    if keepdims:
      result = expand_dims(result, dims)
    return lax.convert_element_type(result, dtype or result_dtype)

  return reduction

def _reduction_dims(a, axis):
  if axis is None:
    return tuple(range(ndim(a)))
  elif isinstance(axis, (np.ndarray, tuple, list)):
    if len(axis) != len(set(axis)):
      raise ValueError(f"duplicate value in 'axis': {axis}")
    return tuple(_canonicalize_axis(x, ndim(a)) for x in axis)
  elif isinstance(axis, int):
    return (_canonicalize_axis(axis, ndim(a)),)
  else:
    raise TypeError("Unexpected type of axis argument: {}".format(type(axis)))

def _reduction_init_val(a, init_val):
  a_dtype = dtypes.canonicalize_dtype(_dtype(a))
  if a_dtype == 'bool':
    return np.array(init_val > 0, dtype=a_dtype)
  try:
    return np.array(init_val, dtype=a_dtype)
  except OverflowError:
    assert issubdtype(a_dtype, integer)
    sign, info = np.sign(init_val), iinfo(a_dtype)
    return np.array(info.min if sign < 0 else info.max, dtype=a_dtype)

_cast_to_bool = partial(lax.convert_element_type, new_dtype=bool_)

sum = _make_reduction(np.sum, lax.add, 0, upcast_f16_for_computation=True,
                      bool_op=lax.bitwise_or)
product = prod = _make_reduction(np.prod, lax.mul, 1, bool_op=lax.bitwise_and,
                                 upcast_f16_for_computation=True)
amax = max = _make_reduction(np.max, lax.max, -np.inf)
amin = min = _make_reduction(np.min, lax.min, np.inf)
all = alltrue = _make_reduction(np.all, lax.bitwise_and, True, _cast_to_bool)
any = sometrue = _make_reduction(np.any, lax.bitwise_or, False, _cast_to_bool)


@_wraps(np.mean)
def mean(a, axis=None, dtype=None, out=None, keepdims=False):
  if out is not None:
    raise ValueError("mean does not support the `out` argument.")

  if axis is None:
    normalizer = size(a)
  else:
    normalizer = np.prod(np.take(shape(a), axis))
  if dtype is None:
    if issubdtype(_dtype(a), bool_) or issubdtype(_dtype(a), integer):
      dtype = float_
    else:
      dtype = _dtype(a)

  return lax.div(
      sum(a, axis, dtype=dtype, keepdims=keepdims),
      lax.convert_element_type(normalizer, dtype))

@_wraps(np.average)
def average(a, axis=None, weights=None, returned=False):
  a = asarray(a)

  if weights is None: # Treat all weights as 1
    avg = mean(a, axis=axis)
    if axis is None:
      weights_sum = full((), size(a), dtype=avg.dtype)
    else:
      weights_sum = full_like(avg, a.shape[axis], dtype=avg.dtype)
  else:
    weights = asarray(weights)

    if issubdtype(a.dtype, inexact):
      out_dtype = result_type(a.dtype, weights.dtype)
    else:
      out_dtype = result_type(a.dtype, weights.dtype, float_)
    out_dtype = dtypes.canonicalize_dtype(out_dtype)

    a_shape = shape(a)
    a_ndim = len(a_shape)
    weights_shape = shape(weights)
    axis = None if axis is None else _canonicalize_axis(axis, a_ndim)

    if a_shape != weights_shape:
      # Make sure the dimensions work out
      if axis is None:
        raise ValueError("Axis must be specified when shapes of a and "
                         "weights differ.")
      if len(weights_shape) != 1:
        raise ValueError("1D weights expected when shapes of a and "
                         "weights differ.")
      if weights_shape[0] != a_shape[axis]:
        raise ValueError("Length of weights not "
                         "compatible with specified axis.")

      weights = broadcast_to(weights, (a_ndim - 1) * (1,) + weights_shape)
      weights = moveaxis(weights, -1, axis)

    weights_sum = sum(weights, axis=axis, dtype=out_dtype)
    avg = sum(multiply(a, weights), axis=axis, dtype=out_dtype) / weights_sum

  if returned:
    if avg.shape != weights_sum.shape:
      weights_sum = broadcast_to(weights_sum, avg.shape)
    return avg, weights_sum
  return avg


@_wraps(np.var)
def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
  if out is not None:
    raise ValueError("var does not support the `out` argument.")

  a_dtype, dtype = _var_promote_types(_dtype(a), dtype)
  a_mean = mean(a, axis, dtype=a_dtype, keepdims=True)
  centered = a - a_mean
  if issubdtype(centered.dtype, complexfloating):
    centered = lax.real(lax.mul(centered, lax.conj(centered)))
  else:
    centered = lax.square(centered)

  if axis is None:
    normalizer = size(a)
  else:
    normalizer = np.prod(np.take(shape(a), axis))
  normalizer = normalizer - ddof

  result = sum(centered, axis, keepdims=keepdims)
  out = lax.div(result, lax.convert_element_type(normalizer, result.dtype))
  return lax.convert_element_type(out, dtype)


def _var_promote_types(a_dtype, dtype):
  if dtype:
    if (not issubdtype(dtype, complexfloating) and
        issubdtype(a_dtype, complexfloating)):
      msg = ("jax.numpy.var does not yet support real dtype parameters when "
             "computing the variance of an array of complex values. The "
             "semantics of numpy.var seem unclear in this case. Please comment "
             "on https://github.com/google/jax/issues/2283 if this behavior is "
             "important to you.")
      raise ValueError(msg)
    a_dtype = promote_types(a_dtype, dtype)
  else:
    if not issubdtype(a_dtype, inexact):
      dtype = a_dtype = float_
    else:
      dtype = _complex_elem_type(a_dtype)
      a_dtype = promote_types(a_dtype, float32)
  return a_dtype, dtype


@_wraps(np.std)
def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
  if out is not None:
    raise ValueError("std does not support the `out` argument.")
  return sqrt(var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims))


@_wraps(np.ptp)
def ptp(a, axis=None, out=None, keepdims=False):
  if out is not None:
    raise ValueError("ptp does not support the `out` argument.")
  x = amax(a, axis=axis, keepdims=keepdims)
  y = amin(a, axis=axis, keepdims=keepdims)
  return lax.sub(x, y)


@_wraps(np.allclose)
def allclose(a, b, rtol=1e-05, atol=1e-08):
  return all(isclose(a, b, rtol, atol))


@_wraps(np.count_nonzero)
def count_nonzero(a, axis=None, keepdims=False):
  return sum(lax.ne(a, _constant_like(a, 0)), axis=axis,
             dtype=dtypes.canonicalize_dtype(np.int_), keepdims=keepdims)


_NONZERO_DOC = """\
At present, JAX does not support JIT-compilation of :py:func:`jax.numpy.nonzero`
because its output shape is data-dependent.
"""

@_wraps(np.nonzero, lax_description=_NONZERO_DOC)
def nonzero(a):
  # Note: this function cannot be jitted because its output has a dynamic
  # shape.
  a = atleast_1d(a)
  dims = shape(a)
  ndims = len(dims)
  ds = [lax.broadcasted_iota(int_, dims + (1,), i) for i in range(ndims)]
  d = concatenate(ds, axis=-1)
  indexes = d[a != 0]
  return tuple(indexes[..., i] for i in range(ndims))


@_wraps(np.flatnonzero)
def flatnonzero(a):
  return nonzero(ravel(a))[0]


def _make_nan_reduction(np_reduction, jnp_reduction, init_val, nan_if_all_nan):
  @_wraps(np_reduction)
  def nan_reduction(a, axis=None, out=None, keepdims=False, **kwargs):
    out = jnp_reduction(where(isnan(a), _reduction_init_val(a, init_val), a),
                       axis=axis, out=out, keepdims=keepdims, **kwargs)
    if nan_if_all_nan:
      return where(all(isnan(a), axis=axis, keepdims=keepdims),
                   _constant_like(a, nan), out)
    else:
      return out

  return nan_reduction

nanmin = _make_nan_reduction(np.nanmin, min, inf, nan_if_all_nan=True)
nanmax = _make_nan_reduction(np.nanmax, max, -inf, nan_if_all_nan=True)
nansum = _make_nan_reduction(np.nansum, sum, 0, nan_if_all_nan=False)
nanprod = _make_nan_reduction(np.nanprod, prod, 1, nan_if_all_nan=False)

@_wraps(np.nanmean)
def nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
  if out is not None:
    raise ValueError("nanmean does not support the `out` argument.")
  if issubdtype(_dtype(a), bool_) or issubdtype(_dtype(a), integer):
    return mean(a, axis, dtype, out, keepdims)
  if dtype is None:
    dtype = _dtype(a)
  nan_mask = logical_not(isnan(a))
  normalizer = sum(nan_mask, axis=axis, dtype=int32, keepdims=keepdims)
  normalizer = lax.convert_element_type(normalizer, dtype)
  td = lax.div(nansum(a, axis, dtype=dtype, keepdims=keepdims), normalizer)
  return td


@_wraps(np.nanvar)
def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
  if out is not None:
    raise ValueError("nanvar does not support the `out` argument.")

  a_dtype, dtype = _var_promote_types(_dtype(a), dtype)
  a_mean = nanmean(a, axis, dtype=a_dtype, keepdims=True)
  centered = a - a_mean
  if issubdtype(centered.dtype, complexfloating):
    centered = lax.real(lax.mul(centered, lax.conj(centered)))
  else:
    centered = lax.square(centered)

  normalizer = sum(logical_not(isnan(a)), axis=axis, keepdims=keepdims)
  normalizer = normalizer - ddof
  zero = lax.full_like(normalizer, 0, shape=())
  normalizer_mask = lax.le(normalizer, zero)

  result = nansum(centered, axis, keepdims=keepdims)
  result = where(normalizer_mask, nan, result)
  divisor = where(normalizer_mask, 1, normalizer)
  out = lax.div(result, lax.convert_element_type(divisor, result.dtype))
  return lax.convert_element_type(out, dtype)


@_wraps(np.nanstd)
def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
  if out is not None:
    raise ValueError("nanstd does not support the `out` argument.")
  return sqrt(nanvar(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims))


def _make_cumulative_reduction(np_reduction, reduction, fill_nan=False, fill_value=0):
  # We want to allow XLA to fuse the pad and reduce-window operators to
  # avoid materializing the padded output.
  # Consider removing `jit` once again if reduce-window is generalized to
  # support arbitrary padding.
  @partial(jit, static_argnums=(1, 2))
  def _cumulative_reduction(a, axis, dtype):
    if axis is None or isscalar(a):
      a = ravel(a)
      axis = 0

    a_shape = list(shape(a))
    num_dims = len(a_shape)

    if axis < 0:
      axis = axis + num_dims
    if axis < 0 or axis >= num_dims:
      raise ValueError(
          "axis {} is out of bounds for array of dimension {}".format(
              axis, num_dims))

    if fill_nan:
      a = where(isnan(a), _constant_like(a, fill_value), a)

    if not dtype and _dtype(a) == bool_:
      dtype = int_
    if dtype:
      a = lax.convert_element_type(a, dtype)

    return reduction(a, axis)

  @_wraps(np_reduction)
  def cumulative_reduction(a, axis=None, dtype=None):
    # jit doesn't support kwargs as static_args.
    return _cumulative_reduction(a, axis, dtype)
  return cumulative_reduction


cumsum = _make_cumulative_reduction(np.cumsum, lax.cumsum, fill_nan=False)
cumprod = _make_cumulative_reduction(np.cumprod, lax.cumprod, fill_nan=False)
cumproduct = cumprod
nancumsum = _make_cumulative_reduction(np.nancumsum, lax.cumsum,
                                       fill_nan=True, fill_value=0)
nancumprod = _make_cumulative_reduction(np.nancumprod, lax.cumprod,
                                        fill_nan=True, fill_value=1)


@_wraps(np.unwrap)
def unwrap(p, discont=pi, axis=-1):
  dd = diff(p, axis=axis)
  ddmod = mod(dd + pi, 2 * pi) - pi
  ddmod = where((ddmod == -pi) & (dd > 0), pi, ddmod)

  ph_correct = where(abs(dd) < discont, 0, ddmod - dd)

  up = concatenate((
    lax.slice_in_dim(p, 0, 1, axis=axis),
    lax.slice_in_dim(p, 1, None, axis=axis) + cumsum(ph_correct, axis=axis)
  ), axis=axis)

  return up


### Array-creation functions

def _check_no_padding(axis_padding, mode):
  if (axis_padding[0] > 0 or axis_padding[1] > 0):
    msg = "Cannot apply '{}' padding to empty axis"
    raise ValueError(msg.format(mode))


def _pad_constant(array, pad_width, constant_values):
  nd = ndim(array)
  constant_values = broadcast_to(asarray(constant_values), (nd, 2))
  constant_values = lax.convert_element_type(constant_values, array.dtype)
  for i in range(nd):
    widths = [(0, 0, 0)] * nd
    widths[i] = (pad_width[i, 0], 0, 0)
    array = lax.pad(array, constant_values[i, 0], widths)
    widths[i] = (0, pad_width[i, 1], 0)
    array = lax.pad(array, constant_values[i, 1], widths)
  return array


def _pad_wrap(array, pad_width):
  for i in range(ndim(array)):
    if array.shape[i] == 0:
      _check_no_padding(pad_width[i], "wrap")
      continue
    size = array.shape[i]
    repeats, (left_remainder, right_remainder) = _divmod(pad_width[i], size)
    total_repeats = repeats.sum() + 1
    parts = []
    if left_remainder:
      parts += [lax.slice_in_dim(array, size - left_remainder, size, axis=i)]
    parts += total_repeats * [array]
    if right_remainder:
      parts += [lax.slice_in_dim(array, 0, right_remainder, axis=i)]
    array = lax.concatenate(parts, dimension=i)
  return array


def _pad_symmetric_or_reflect(array, pad_width, mode):
  assert mode in ("symmetric", "reflect")

  for i in range(ndim(array)):
    if array.shape[i] == 0:
      _check_no_padding(pad_width[i], mode)
      continue

    n = array.shape[i]
    rarray = lax.rev(array, dimensions=(i,))
    offset = 1 if (mode == "reflect" and n > 1) else 0

    def build_padding(padding, forward):
      xs = []
      delta = n - offset
      while padding > delta:
        padding -= delta
        p = array if forward else rarray
        xs.append(lax.slice_in_dim(p, offset, n, axis=i))
        forward = not forward
      if padding > 0:
        x = lax.slice_in_dim(array if forward else rarray, offset,
                             padding + offset, axis=i)
        xs.append(x)
      return xs

    parts = reversed(build_padding(pad_width[i, 0], forward=True))
    parts = [lax.rev(x, dimensions=(i,)) for x in parts]
    parts += [array]
    parts += build_padding(pad_width[i, 1], forward=False)
    array = lax.concatenate(parts, dimension=i)
  return array


def _pad_edge(array, pad_width):
  nd = ndim(array)
  for i in range(nd):
    if array.shape[i] == 0:
      _check_no_padding(pad_width[i], "edge")
      continue

    n = array.shape[i]
    npad_before, npad_after = pad_width[i]

    edge_before = lax.slice_in_dim(array, 0, 1, axis=i)
    pad_before = repeat(edge_before, npad_before, axis=i)

    edge_after = lax.slice_in_dim(array, n-1, n, axis=i)
    pad_after = repeat(edge_after, npad_after, axis=i)

    array = lax.concatenate([pad_before, array, pad_after], dimension=i)
  return array


@partial(jit, static_argnums=(1, 2))
def _pad(array, pad_width, mode, constant_values):
  array = asarray(array)
  nd = ndim(array)
  pad_width = np.broadcast_to(np.asarray(pad_width), (nd, 2))
  if np.any(pad_width < 0):
    raise ValueError("index can't contain negative values")

  if mode == "constant":
    return _pad_constant(array, pad_width, constant_values)

  elif mode == "wrap":
    return _pad_wrap(array, pad_width)

  elif mode in ("symmetric", "reflect"):
    return _pad_symmetric_or_reflect(array, pad_width, mode)

  elif mode == "edge":
    return _pad_edge(array, pad_width)

  else:
    msg = "Unimplemented padding mode '{}' for np.pad."
    raise NotImplementedError(msg.format(mode))

@_wraps(np.pad)
def pad(array, pad_width, mode='constant', constant_values=0):
  if isinstance(pad_width, list):
    pad_width = tuple(pad_width)
  return _pad(array, pad_width, mode, constant_values)


@_wraps(np.stack)
def stack(arrays, axis=0):
  if not len(arrays):
    raise ValueError("Need at least one array to stack.")
  shape0 = shape(arrays[0])
  axis = _canonicalize_axis(axis, len(shape0) + 1)
  new_arrays = []
  for a in arrays:
    if shape(a) != shape0:
      raise ValueError("All input arrays must have the same shape.")
    new_arrays.append(expand_dims(a, axis))
  return concatenate(new_arrays, axis=axis)

@_wraps(np.tile)
def tile(A, reps):
  if isinstance(reps, int):
    reps = (reps,)
  A = reshape(A, (1,) * (len(reps) - ndim(A)) + shape(A))
  reps = (1,) * (ndim(A) - len(reps)) + tuple(reps)
  for i, rep in enumerate(reps):
    A = concatenate([A] * int(rep), axis=i)
  return A

@_wraps(np.concatenate)
def concatenate(arrays, axis=0):
  if not len(arrays):
    raise ValueError("Need at least one array to concatenate.")
  if ndim(arrays[0]) == 0:
    raise ValueError("Zero-dimensional arrays cannot be concatenated.")
  if axis is None:
    return concatenate([ravel(a) for a in arrays], axis=0)
  axis = _canonicalize_axis(axis, ndim(arrays[0]))
  arrays = _promote_dtypes(*arrays)
  # lax.concatenate can be slow to compile for wide concatenations, so form a
  # tree of concatenations as a workaround especially for op-by-op mode.
  # (https://github.com/google/jax/issues/653).
  k = 16
  if len(arrays) == 1:
    return array(arrays[0])
  else:
    while len(arrays) > 1:
      arrays = [lax.concatenate(arrays[i:i+k], axis)
                for i in range(0, len(arrays), k)]
    return arrays[0]


@_wraps(np.vstack)
def vstack(tup):
  return concatenate([atleast_2d(m) for m in tup], axis=0)
row_stack = vstack


@_wraps(np.hstack)
def hstack(tup):
  arrs = [atleast_1d(m) for m in tup]
  if arrs[0].ndim == 1:
    return concatenate(arrs, 0)
  return concatenate(arrs, 1)


@_wraps(np.dstack)
def dstack(tup):
  return concatenate([atleast_3d(m) for m in tup], axis=2)


@_wraps(np.column_stack)
def column_stack(tup):
  arrays = []
  for v in tup:
    arr = array(v)
    if arr.ndim < 2:
      arr = atleast_2d(arr).T
    arrays.append(arr)
  return concatenate(arrays, 1)


def _atleast_nd(x, n):
  m = ndim(x)
  return lax.broadcast(x, (1,) * (n - m)) if m < n else x

def _block(xs):
  if isinstance(xs, tuple):
    raise ValueError("jax.numpy.block does not allow tuples, got {}"
                     .format(xs))
  elif isinstance(xs, list):
    if len(xs) == 0:
      raise ValueError("jax.numpy.block does not allow empty list arguments")
    xs, depths = unzip2([_block(x) for x in xs])
    if _any(d != depths[0] for d in depths[1:]):
      raise ValueError("Mismatched list depths in jax.numpy.block")
    rank = _max(depths[0], _max(ndim(x) for x in xs))
    xs = [_atleast_nd(x, rank) for x in xs]
    return concatenate(xs, axis=-depths[0]), depths[0] + 1
  else:
    return asarray(xs), 1

@_wraps(np.block)
@jit
def block(arrays):
  out, _ = _block(arrays)
  return out


@_wraps(np.atleast_1d, update_doc=False)
def atleast_1d(*arys):
  if len(arys) == 1:
    arr = array(arys[0])
    return arr if ndim(arr) >= 1 else reshape(arr, -1)
  else:
    return [atleast_1d(arr) for arr in arys]


@_wraps(np.atleast_2d, update_doc=False)
def atleast_2d(*arys):
  if len(arys) == 1:
    arr = array(arys[0])
    if ndim(arr) >= 2:
      return arr
    elif ndim(arr) == 1:
      return expand_dims(arr, axis=0)
    else:
      return expand_dims(arr, axis=(0, 1))
  else:
    return [atleast_2d(arr) for arr in arys]


@_wraps(np.atleast_3d, update_doc=False)
def atleast_3d(*arys):
  if len(arys) == 1:
    arr = array(arys[0])
    if ndim(arr) == 0:
      arr = expand_dims(arr, axis=(0, 1, 2))
    elif ndim(arr) == 1:
      arr = expand_dims(arr, axis=(0, 2))
    elif ndim(arr) == 2:
      arr = expand_dims(arr, axis=2)
    return arr
  else:
    return [atleast_3d(arr) for arr in arys]


@_wraps(np.array)
def array(object, dtype=None, copy=True, order="K", ndmin=0):
  if order is not None and order != "K":
    raise NotImplementedError("Only implemented for order='K'")
  lax._check_user_dtype_supported(dtype, "array")
  dtype = dtype and dtypes.canonicalize_dtype(dtype)

  if _can_call_numpy_array(object):
    object = _np_array(object, dtype=dtype, ndmin=ndmin)
  assert type(object) not in dtypes.python_scalar_dtypes

  if type(object) is np.ndarray:
    out = _device_put_raw(object)
    if dtype: assert _dtype(out) == dtype
  elif isinstance(object, (DeviceValue, core.Tracer)):
    if isinstance(object, DeviceArray) and copy:
      # We perform a copy by bouncing back to the host
      # TODO(phawkins): add a device runtime function to copy a buffer
      out = _device_put_raw(_np_asarray(object))
    else:
      out = object
  elif isinstance(object, (list, tuple)):
    if object:
      out = stack([array(elt, dtype=dtype) for elt in object])
    else:
      out = _device_put_raw(_np_array([], dtype=dtype))
  else:
    try:
      view = memoryview(object)
    except TypeError:
      pass  # `object` does not support the buffer interface.
    else:
      return array(_np_asarray(view), dtype, copy)

    raise TypeError("Unexpected input type for array: {}".format(type(object)))

  if dtype and _dtype(out) != dtype:
    out = lax.convert_element_type(out, dtype)

  if ndmin > ndim(out):
    out = lax.broadcast(out, (1,) * (ndmin - ndim(out)))
  return out

def _can_call_numpy_array(x):
  return _all(not isinstance(l, (core.Tracer, DeviceValue))
              for l in tree_leaves(x))

# TODO(mattjj): maybe move these two functions into xla.py
def _device_put_raw(x):
  return array_result_handler(None, abstractify(x))(device_put(x))


@_wraps(np.asarray)
def asarray(a, dtype=None, order=None):
  lax._check_user_dtype_supported(dtype, "asarray")
  return array(a, dtype=dtype, copy=False, order=order)


@_wraps(np.zeros_like)
def zeros_like(a, dtype=None):
  lax._check_user_dtype_supported(dtype, "zeros_like")
  return lax.full_like(a, 0, dtype)


@_wraps(np.ones_like)
def ones_like(a, dtype=None):
  lax._check_user_dtype_supported(dtype, "ones_like")
  return lax.full_like(a, 1, dtype)


@_wraps(np.full)
def full(shape, fill_value, dtype=None):
  lax._check_user_dtype_supported(dtype, "full")
  shape = (shape,) if ndim(shape) == 0 else shape
  return lax.full(shape, fill_value, dtype)


@_wraps(np.full_like)
def full_like(a, fill_value, dtype=None):
  lax._check_user_dtype_supported(dtype, "full_like")
  return lax.full_like(a, fill_value, dtype)


@_wraps(np.zeros)
def zeros(shape, dtype=None):
  if isinstance(shape, types.GeneratorType):
    raise TypeError("expected sequence object with len >= 0 or a single integer")
  lax._check_user_dtype_supported(dtype, "zeros")
  dtype = float_ if dtype is None else dtype
  shape = (shape,) if ndim(shape) == 0 else shape
  return lax.full(shape, 0, dtype)

@_wraps(np.ones)
def ones(shape, dtype=None):
  if isinstance(shape, types.GeneratorType):
    raise TypeError("expected sequence object with len >= 0 or a single integer")
  lax._check_user_dtype_supported(dtype, "ones")
  dtype = float_ if dtype is None else dtype
  shape = (shape,) if ndim(shape) == 0 else shape
  return lax.full(shape, 1, dtype)


@_wraps(np.array_equal)
def array_equal(a1, a2, equal_nan=False):
  try:
    a1, a2 = asarray(a1), asarray(a2)
  except Exception:
    return False
  if shape(a1) != shape(a2):
    return False
  eq = asarray(a1 == a2)
  if equal_nan:
    eq = logical_or(eq, logical_and(isnan(a1), isnan(a2)))
  return all(eq)


# We can't create uninitialized arrays in XLA; use zeros for empty.
empty_like = zeros_like
empty = zeros


@_wraps(np.eye)
def eye(N, M=None, k=0, dtype=None):
  lax._check_user_dtype_supported(dtype, "eye")
  dtype = float_ if dtype is None else dtype
  M = N if M is None else M
  k = int(k)
  if N < 0 or M < 0:
    msg = "negative dimensions are not allowed, got {} and {}"
    raise ValueError(msg.format(N, M))
  if k is not None:
    k_dtype = _dtype(k)
    if not issubdtype(k_dtype, integer):
      msg = "eye argument `k` must be of integer dtype, got {}"
      raise TypeError(msg.format(k_dtype))
  return lax._eye(dtype, (N, M), k)


@_wraps(np.identity)
def identity(n, dtype=None):
  lax._check_user_dtype_supported(dtype, "identity")
  return eye(n, dtype=dtype)


@_wraps(np.arange)
def arange(start, stop=None, step=None, dtype=None):
  lax._check_user_dtype_supported(dtype, "arange")
  require = partial(core.concrete_or_error, _np_asarray)
  msg = "in jax.numpy.arange argument `{}`".format
  if stop is None and step is None:
    start = require(start, msg("stop"))
    dtype = dtype or _dtype(start)
    return lax.iota(dtype, np.ceil(start)) # avoids materializing
  else:
    start = require(start, msg("start"))
    stop = None if stop is None else require(stop, msg("stop"))
    step = None if step is None else require(step, msg("step"))
    if dtype is None:
      dtype = _dtype(start, *filter(lambda x: x is not None, [stop, step]))
    return array(np.arange(start, stop=stop, step=step, dtype=dtype))


def _wrap_numpy_nullary_function(f):
  """Adapts `f` to return a DeviceArray instead of an np.ndarray.

  `f` cannot have any non-static array arguments.
  """
  @_wraps(f, update_doc=False)
  def wrapper(*args, **kwargs):
    return asarray(f(*args, **kwargs))
  return wrapper


@_wraps(np.linspace)
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None,
             axis=0):
  """Implementation of linspace differentiable in start and stop args."""
  lax._check_user_dtype_supported(dtype, "linspace")
  if num < 0:
    raise ValueError("Number of samples, %s, must be non-negative." % num)

  dtype = dtype or result_type(start, stop, dtypes.canonicalize_dtype(float_))
  computation_dtype = promote_types(dtype, dtypes.canonicalize_dtype(float_))
  start = asarray(start, dtype=computation_dtype)
  stop = asarray(stop, dtype=computation_dtype)

  bounds_shape = list(lax.broadcast_shapes(shape(start), shape(stop)))
  broadcast_start = broadcast_to(start, bounds_shape)
  broadcast_stop = broadcast_to(stop, bounds_shape)
  axis = len(bounds_shape) + axis + 1 if axis < 0 else axis
  bounds_shape.insert(axis, 1)
  iota_shape = [1,] * len(bounds_shape)
  iota_shape[axis] = num
  div = (num - 1) if endpoint else num
  if num > 1:
    delta = lax.convert_element_type(stop - start, computation_dtype) / div
    if issubdtype(dtype, integer):
      # This is similar to how numpy computes linspace, but it
      # can fail to recover the endpoints in float32 arithmetic.
      out = (reshape(broadcast_start, bounds_shape) +
        reshape(lax.iota(dtype, num), iota_shape) *
        reshape(delta, bounds_shape))
    else:
      # This approach recovers the endpoints with float32 arithmetic,
      # but can lead to rounding errors for integer outputs.
      step = reshape(lax.iota(computation_dtype, num), iota_shape) / div
      out = (reshape(broadcast_start, bounds_shape) * (1 - step) +
        reshape(broadcast_stop, bounds_shape) * step)
  elif num == 1:
    delta = nan if endpoint else stop - start
    out = reshape(broadcast_start, bounds_shape)
  else: # num == 0 degenerate case, match numpy behavior
    empty_shape = list(lax.broadcast_shapes(shape(start), shape(stop)))
    empty_shape.insert(axis, 0)
    delta = nan
    out = reshape(array([], dtype=dtype), empty_shape)
  if retstep:
    return lax.convert_element_type(out, dtype), delta
  else:
    return lax.convert_element_type(out, dtype)


@_wraps(np.logspace)
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
  """Implementation of logspace differentiable in start and stop args."""
  dtype = dtype or result_type(start, stop, dtypes.canonicalize_dtype(float_))
  computation_dtype = promote_types(dtype, dtypes.canonicalize_dtype(float_))
  start = asarray(start, dtype=computation_dtype)
  stop = asarray(stop, dtype=computation_dtype)
  lin = linspace(start, stop, num,
                 endpoint=endpoint, retstep=False, dtype=None, axis=axis)
  return lax.convert_element_type(power(base, lin), dtype)


@_wraps(np.geomspace)
def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
  """Implementation of geomspace differentiable in start and stop args."""
  dtype = dtype or result_type(start, stop, dtypes.canonicalize_dtype(float_))
  computation_dtype = promote_types(dtype, dtypes.canonicalize_dtype(float_))
  start = asarray(start, dtype=computation_dtype)
  stop = asarray(stop, dtype=computation_dtype)
  # follow the numpy geomspace convention for negative and complex endpoints
  signflip = 1 - (1 - sign(real(start))) * (1 - sign(real(stop))) // 2
  res = signflip * logspace(log10(signflip * start),
                            log10(signflip * stop), num,
                            endpoint=endpoint, base=10.0,
                            dtype=computation_dtype, axis=0)
  if axis != 0:
    res = moveaxis(res, 0, axis)
  return lax.convert_element_type(res, dtype)


@_wraps(np.meshgrid)
def meshgrid(*args, **kwargs):
  indexing = kwargs.get("indexing", "xy")
  sparse = kwargs.get("sparse", False)
  copy = kwargs.get("copy", True)
  if not copy:
    raise ValueError("jax.numpy.meshgrid only supports copy=True")

  args = list(args)
  if indexing == "xy":
    if len(args) >= 2:
      args[0], args[1] = args[1], args[0]
  elif indexing != "ij":
    raise ValueError("Valid values for indexing are 'xy' and 'ij', got {}"
                     .format(indexing))

  shape = []
  for i, a in enumerate(args):
    args[i] = a = asarray(a)
    if len(a.shape) != 1:
      msg = "Arguments to jax.numpy.meshgrid must be 1D, got shape {}"
      raise ValueError(msg.format(a.shape))
    shape.append(1 if sparse else a.shape[0])

  output = []
  for i, a in enumerate(args):
    a = asarray(a)
    s = shape
    if sparse:
      s = list(s)
      s[i] = a.shape[0]
    output.append(lax.broadcast_in_dim(a, s, (i,)))

  if indexing == "xy" and len(args) >= 2:
      output[0], output[1] = output[1], output[0]

  return output


@_wraps(np.ix_)
def ix_(*args):
  n = len(args)
  output = []
  for i, a in enumerate(args):
    a = asarray(a)
    if len(a.shape) != 1:
      msg = "Arguments to jax.numpy.ix_ must be 1-dimensional, got shape {}"
      raise ValueError(msg.format(a.shape))
    if _dtype(a) == bool_:
      raise NotImplementedError(
        "Boolean arguments to jax.numpy.ix_ are not implemented")
    shape = [1] * n
    shape[i] = a.shape[0]
    if a.size == 0:
      # Numpy uses an integer index type for empty arrays.
      output.append(lax.full(shape, np.zeros((), np.intp)))
    else:
      output.append(lax.broadcast_in_dim(a, shape, (i,)))
  return tuple(output)


@_wraps(np.indices)
def indices(dimensions, dtype=int32, sparse=False):
  dimensions = tuple(dimensions)
  N = len(dimensions)
  output = []
  s = dimensions
  for i, dim in enumerate(dimensions):
    idx = lax.iota(dtype, dim)
    if sparse:
      s = (1,)*i + (dim,) + (1,)*(N - i - 1)
    output.append(lax.broadcast_in_dim(idx, s, (i,)))
  if sparse:
      return tuple(output)
  return stack(output, 0) if output else array([], dtype=dtype)


_TOTAL_REPEAT_LENGTH_DOC = """\
Jax adds the optional `total_repeat_length` parameter which specifies the total
number of repeat, and defaults to sum(repeats). It must be specified for repeat
to be compilable. If `sum(repeats)` is larger than the specified
`total_repeat_length` the remaining values will be discarded. In the case of
`sum(repeats)` being smaller than the specified target length, the final value
will be repeated.
"""


@_wraps(np.repeat, lax_description=_TOTAL_REPEAT_LENGTH_DOC)
def repeat(a, repeats, axis=None, *, total_repeat_length=None):
  if axis is None:
    a = ravel(a)
    axis = 0

  # If total_repeat_length is not given, can't compile, use a default.
  if total_repeat_length is None:
    repeats = core.concrete_or_error(np.array, repeats, "jax.numpy.repeat")
    repeats = np.ravel(repeats)
    if ndim(a) != 0:
      repeats = np.broadcast_to(repeats, [a.shape[axis]])
    total_repeat_length = np.sum(repeats)
  else:
    repeats = ravel(repeats)
    if ndim(a) != 0:
      repeats = broadcast_to(repeats, [a.shape[axis]])

  # Special case when a is a scalar.
  if ndim(a) == 0:
    if repeats.shape == (1,):
      return full([total_repeat_length], a)
    else:
      raise ValueError('`repeat` with a scalar parameter `a` is only '
      'implemented for scalar values of the parameter `repeats`.')

  # Special case if total_repeat_length is zero.
  if total_repeat_length == 0:
    result_shape = list(a.shape)
    result_shape[axis] = 0
    return reshape(array([], dtype=a.dtype), result_shape)

  # If repeats is on a zero sized axis, then return the array.
  if a.shape[axis] == 0:
    return a

  # This implementation of repeat avoid having to instantiate a large.
  # intermediate tensor.

  # Modify repeats from e.g. [1,2,0,5] -> [0,1,2,0] for exclusive repeat.
  exclusive_repeats = roll(repeats, shift=1).at[0].set(0)
  # Cumsum to get indices of new number in repeated tensor, e.g. [0, 1, 3, 3]
  scatter_indices = cumsum(exclusive_repeats)
  # Scatter these onto a zero buffer, e.g. [1,1,0,2,0,0,0,0]
  block_split_indicators = ops.index_add(
      x=zeros([total_repeat_length], dtype=int32),
      idx=scatter_indices,
      y=1)
  # Cumsum again to get scatter indices for repeat, e.g. [0,1,1,3,3,3,3,3]
  gather_indices = cumsum(block_split_indicators) - 1
  return take(a, gather_indices, axis=axis)


@_wraps(np.tri)
def tri(N, M=None, k=0, dtype=None):
  lax._check_user_dtype_supported(dtype, "tri")
  M = M if M is not None else N
  dtype = dtype or float32
  return lax._tri(dtype, (N, M), k)


@_wraps(np.tril)
def tril(m, k=0):
  m_shape = shape(m)
  if len(m_shape) < 2:
    raise ValueError("Argument to jax.numpy.tril must be at least 2D")
  mask = tri(*m_shape[-2:], k=k, dtype=bool)
  return lax.select(lax.broadcast(mask, m_shape[:-2]), m, zeros_like(m))


@_wraps(np.triu, update_doc=False)
def triu(m, k=0):
  m_shape = shape(m)
  if len(m_shape) < 2:
    raise ValueError("Argument to jax.numpy.triu must be at least 2D")
  mask = tri(*m_shape[-2:], k=k - 1, dtype=bool)
  return lax.select(lax.broadcast(mask, m_shape[:-2]), zeros_like(m), m)


@_wraps(np.trace)
def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
  if out:
    raise NotImplementedError("The 'out' argument to trace is not supported.")
  lax._check_user_dtype_supported(dtype, "trace")

  axis1 = _canonicalize_axis(axis1, ndim(a))
  axis2 = _canonicalize_axis(axis2, ndim(a))

  a_shape = shape(a)
  if dtype is None:
    dtype = _dtype(a)
    if issubdtype(dtype, integer):
      default_int = dtypes.canonicalize_dtype(np.int_)
      if iinfo(dtype).bits < iinfo(default_int).bits:
        dtype = default_int

  # Move the axis? dimensions to the end.
  perm = [i for i in range(len(a_shape)) if i != axis1 and i != axis2]
  perm = perm + [axis1, axis2]
  a = lax.transpose(a, perm)

  # Mask out the diagonal and reduce.
  a = where(eye(a_shape[axis1], a_shape[axis2], k=offset, dtype=bool),
            a, zeros_like(a))
  return sum(a, axis=(-2, -1), dtype=dtype)


def _wrap_indices_function(f):
  @_wraps(f, update_doc=False)
  def wrapper(*args, **kwargs):
    return tuple(asarray(x) for x in f(*args, **kwargs))
  return wrapper

tril_indices = _wrap_indices_function(np.tril_indices)
triu_indices = _wrap_indices_function(np.triu_indices)
mask_indices = _wrap_indices_function(np.mask_indices)


@_wraps(np.triu_indices_from)
def triu_indices_from(arr, k=0):
  return triu_indices(arr.shape[-2], k=k, m=arr.shape[-1])


@_wraps(np.tril_indices_from)
def tril_indices_from(arr, k=0):
  return tril_indices(arr.shape[-2], k=k, m=arr.shape[-1])


@_wraps(np.diag_indices)
def diag_indices(n, ndim=2):
  if n < 0:
    raise ValueError("n argument to diag_indices must be nonnegative, got {}"
                     .format(n))
  if ndim < 0:
    raise ValueError("ndim argument to diag_indices must be nonnegative, got {}"
                     .format(ndim))
  return (lax.iota(int_, n),) * ndim

@_wraps(np.diag_indices_from)
def diag_indices_from(arr):
  if not arr.ndim >= 2:
    raise ValueError("input array must be at least 2-d")

  if len(set(arr.shape)) != 1:
    raise ValueError("All dimensions of input must be of equal length")

  return diag_indices(arr.shape[0], ndim=arr.ndim)

@_wraps(np.diagonal)
def diagonal(a, offset=0, axis1=0, axis2=1):
  a_shape = shape(a)
  a_ndims = len(a_shape)

  # Move the two dimensions to the end.
  axis1 = _canonicalize_axis(axis1, a_ndims)
  axis2 = _canonicalize_axis(axis2, a_ndims)
  perm = [i for i in range(a_ndims) if i != axis1 and i != axis2]
  perm = perm + [axis1, axis2]
  a = lax.transpose(a, perm)

  # Mask out the diagonal and reduce over one of the axes
  a = where(eye(a_shape[axis1], a_shape[axis2], k=offset, dtype=bool),
            a, zeros_like(a))
  reduce_axis = -2 if offset < 0 else -1
  d = sum(a, axis=reduce_axis, dtype=_dtype(a))

  # Slice out the correct diagonal size.
  diag_size = _max(0, _min(a_shape[axis1] + _min(offset, 0),
                           a_shape[axis2] - _max(offset, 0)))
  return lax.slice_in_dim(d, 0, diag_size, axis=-1)


@_wraps(np.diag)
def diag(v, k=0):
  v_shape = shape(v)
  if len(v_shape) == 1:
    zero = lambda x: lax.full_like(x, shape=(), fill_value=0)
    n = v_shape[0] + _abs(k)
    v = lax.pad(v, zero(v), ((_max(0, k), _max(0, -k), 0),))
    return where(eye(n, k=k, dtype=bool), v, zeros_like(v))
  elif len(v_shape) == 2:
    return diagonal(v, offset=k)
  else:
    raise ValueError("diag input must be 1d or 2d")

_SCALAR_VALUE_DOC="""\
This differs from np.diagflat for some scalar values of v,
jax always returns a two-dimensional array, whereas numpy may
return a scalar depending on the type of v.
"""

@_wraps(np.diagflat, lax_description=_SCALAR_VALUE_DOC)
def diagflat(v, k=0):
  v = ravel(v)
  v_length = len(v)
  adj_length = v_length + _abs(k)
  res = zeros(adj_length*adj_length, dtype=v.dtype)
  i = arange(0, adj_length-_abs(k))
  if (k >= 0):
    fi = i+k+i*adj_length
  else:
    fi = i+(i-k)*adj_length
  res = ops.index_update(res, ops.index[fi], v)
  res = res.reshape(adj_length,adj_length)
  return res


@_wraps(np.polyval)
def polyval(p, x):
  if isinstance(p, np.poly1d):
    p = np.asarray(p)
  if isinstance(x, np.poly1d):
    y = 0
  else:
    y = zeros_like(x)
  for i in range(len(p)):
    y = y * x + p[i]
  return y

@_wraps(np.polyadd)
def polyadd(a1, a2):
  a1 = asarray(a1)
  a2 = asarray(a2)

  if a2.shape[0] <= a1.shape[0]:
    return a1.at[-a2.shape[0]:].add(a2)
  else:
    return a2.at[-a1.shape[0]:].add(a1)


@_wraps(np.polyder)
def polyder(p, m=1):
  p = asarray(p)
  if m < 0:
    raise ValueError("Order of derivative must be positive")
  if m == 0:
    return p
  if m % 1:
    raise ValueError("m must be an integer")
  coeff = (arange(len(p), m, -1) - 1 - arange(m)[:, newaxis]).prod(0)
  return p[:-m] * coeff

def _trim_zeros(a):
  for i, v in enumerate(a):
    if v != 0:
      return a[i:]
  return a[:0]

_LEADING_ZEROS_DOC="""\
Setting trim_leading_zeros=True makes the output match that of numpy.
But prevents the function from being able to be used in compiled code.
"""

@_wraps(np.polymul, lax_description=_LEADING_ZEROS_DOC)
def polymul(a1, a2, *, trim_leading_zeros=False):
  if isinstance(a1, np.poly1d):
    a1 = asarray(a1)
  if isinstance(a2, np.poly1d):
    a2 = asarray(a2)
  if trim_leading_zeros and (len(a1) > 1 or len(a2) > 1):
    a1, a2 = _trim_zeros(a1), _trim_zeros(a2)
  if len(a1) == 0:
    a1 = asarray([0.])
  if len(a2) == 0:
    a2 = asarray([0.])
  val = convolve(a1, a2, mode='full')
  return val

@_wraps(np.polysub)
def polysub(a1, a2):
  return polyadd(asarray(a1), -asarray(a2))


@_wraps(np.append)
def append(arr, values, axis=None):
  if axis is None:
    return concatenate([ravel(arr), ravel(values)], 0)
  else:
    return concatenate([arr, values], axis=axis)


### Tensor contraction operations


@_wraps(np.dot, lax_description=_PRECISION_DOC)
def dot(a, b, *, precision=None):  # pylint: disable=missing-docstring
  _check_arraylike("dot", a, b)
  a, b = _promote_dtypes(a, b)
  a_ndim, b_ndim = ndim(a), ndim(b)
  if a_ndim == 0 or b_ndim == 0:
    return lax.mul(a, b)
  if _max(a_ndim, b_ndim) <= 2:
    return lax.dot(a, b, precision=precision)

  if b_ndim == 1:
    contract_dims = ((a_ndim - 1,), (0,))
  else:
    contract_dims = ((a_ndim - 1,), (b_ndim - 2,))
  batch_dims = ((), ())
  return lax.dot_general(a, b, (contract_dims, batch_dims), precision)


@_wraps(np.matmul, lax_description=_PRECISION_DOC)
def matmul(a, b, *, precision=None):  # pylint: disable=missing-docstring
  _check_arraylike("matmul", a, b)
  for i, x in enumerate((a, b)):
    if ndim(x) < 1:
      msg = (f"matmul input operand {i} must have ndim at least 1, "
             f"but it has ndim {ndim(x)}")
      raise ValueError(msg)

  a, b = _promote_dtypes(a, b)

  a_is_mat, b_is_mat = (ndim(a) > 1), (ndim(b) > 1)
  a_batch_dims = shape(a)[:-2] if a_is_mat else ()
  b_batch_dims = shape(b)[:-2] if b_is_mat else ()
  num_batch_dims = _max(len(a_batch_dims), len(b_batch_dims))
  a_batch_dims = (None,) * (num_batch_dims - len(a_batch_dims)) + a_batch_dims
  b_batch_dims = (None,) * (num_batch_dims - len(b_batch_dims)) + b_batch_dims

  # Dimensions to squeeze from the inputs.
  a_squeeze = []
  b_squeeze = []

  # Positions of batch dimensions in squeezed inputs.
  a_batch = []
  b_batch = []

  # Desired index in final output of each kind of dimension, in the order that
  # lax.dot_general will emit them.
  idx_batch = []
  idx_a_other = []  # other = non-batch, non-contracting.
  idx_b_other = []
  for i, (ba, bb) in enumerate(zip(a_batch_dims, b_batch_dims)):
    if ba is None:
      idx_b_other.append(i)
    elif bb is None:
      idx_a_other.append(i)
    elif ba == 1:
      idx_b_other.append(i)
      a_squeeze.append(len(idx_batch) + len(idx_a_other) + len(a_squeeze))
    elif bb == 1:
      idx_a_other.append(i)
      b_squeeze.append(len(idx_batch) + len(idx_b_other) + len(b_squeeze))
    elif ba == bb:
      a_batch.append(len(idx_batch) + len(idx_a_other))
      b_batch.append(len(idx_batch) + len(idx_b_other))
      idx_batch.append(i)
    else:
      raise ValueError("Incompatible shapes for matmul arguments: {} and {}"
                       .format(shape(a), shape(b)))

  if a_is_mat: idx_a_other.append(num_batch_dims)
  if b_is_mat: idx_b_other.append(num_batch_dims + a_is_mat)
  perm = np.argsort(np.concatenate([idx_batch, idx_a_other, idx_b_other]))

  a = lax.squeeze(a, tuple(a_squeeze))
  b = lax.squeeze(b, tuple(b_squeeze))
  out = lax.dot_general(
    a, b, (((ndim(a) - 1,), (ndim(b) - 1 - b_is_mat,)), (a_batch, b_batch)),
    precision=precision)
  return lax.transpose(out, perm)


@_wraps(np.vdot, lax_description=_PRECISION_DOC)
def vdot(a, b, *, precision=None):
  if issubdtype(_dtype(a), complexfloating):
    a = conj(a)
  return dot(a.ravel(), b.ravel(), precision=precision)


@_wraps(np.tensordot, lax_description=_PRECISION_DOC)
def tensordot(a, b, axes=2, *, precision=None):
  _check_arraylike("tensordot", a, b)
  a_ndim = ndim(a)
  b_ndim = ndim(b)

  a, b = _promote_dtypes(a, b)
  if type(axes) is int:
    if axes > _min(a_ndim, b_ndim):
      msg = "Number of tensordot axes (axes {}) exceeds input ranks ({} and {})"
      raise TypeError(msg.format(axes, a.shape, b.shape))
    contracting_dims = tuple(range(a_ndim - axes, a_ndim)), tuple(range(axes))
  elif type(axes) in (list, tuple) and len(axes) == 2:
    ax1, ax2 = axes
    if type(ax1) == type(ax2) == int:
      contracting_dims = ((_canonicalize_axis(ax1, a_ndim),),
                          (_canonicalize_axis(ax2, b_ndim),))
    elif type(ax1) in (list, tuple) and type(ax2) in (list, tuple):
      if len(ax1) != len(ax2):
        msg = "tensordot requires axes lists to have equal length, got {} and {}."
        raise TypeError(msg.format(ax1, ax2))
      contracting_dims = (tuple(_canonicalize_axis(i, a_ndim) for i in ax1),
                          tuple(_canonicalize_axis(i, b_ndim) for i in ax2))
    else:
        msg = "tensordot requires both axes lists to be either ints, tuples or lists, got {} and {}"
        raise TypeError(msg.format(ax1, ax2))
  else:
    msg = ("tensordot axes argument must be an int, a pair of ints, or a pair "
           "of lists/tuples of ints.")
    raise TypeError(msg)
  return lax.dot_general(a, b, (contracting_dims, ((), ())),
                         precision=precision)


@_wraps(np.einsum, lax_description=_PRECISION_DOC)
def einsum(*operands, optimize='greedy', precision=None):
  optimize = 'greedy' if optimize is True else optimize
  # using einsum_call=True here is an internal api for opt_einsum
  operands, contractions = opt_einsum.contract_path(
      *operands, einsum_call=True, use_blas=True, optimize=optimize)
  contractions = tuple(data[:3] for data in contractions)
  return _einsum(operands, contractions, precision)

@_wraps(np.einsum_path)
def einsum_path(subscripts, *operands, optimize='greedy'):
  # using einsum_call=True here is an internal api for opt_einsum
  return opt_einsum.contract_path(subscripts, *operands, optimize=optimize)

def _removechars(s, chars):
  return s.translate(str.maketrans(dict.fromkeys(chars)))

@partial(jit, static_argnums=(1, 2))
def _einsum(operands: Sequence,
            contractions: Sequence[Tuple[Tuple[int, ...], Set[str], str]],
            precision):
  operands = list(_promote_dtypes(*operands))
  def sum(x, axes):
    return lax.reduce(x, np.array(0, x.dtype),
                      lax.add if x.dtype != bool_ else lax.bitwise_or, axes)

  def sum_uniques(operand, names, uniques):
    if uniques:
      axes = [names.index(name) for name in uniques]
      operand = sum(operand, axes)
      names = _removechars(names, uniques)
    return operand, names

  def sum_repeats(operand, names, counts, keep_names):
    for name, count in counts.items():
      if count > 1:
        axes = [i for i, n in enumerate(names) if n == name]
        eye = lax._delta(operand.dtype, operand.shape, axes)
        if name not in keep_names:
          operand = sum(operand * eye, axes)
          names = names.replace(name, '')
        else:
          operand = sum(operand * eye, axes[:-1])
          names = names.replace(name, '', count - 1)
    return operand, names

  def filter_singleton_dims(operand, names, other_shape, other_names):
    s = shape(operand)
    new_shape = []
    new_names = []
    for i, d in enumerate(names):
      other_i = other_names.find(d)
      if s[i] != 1 or other_i == -1 or other_shape[other_i] == 1:
        new_shape.append(s[i])
        new_names.append(d)
    return reshape(operand, tuple(new_shape)), "".join(new_names)

  for operand_indices, contracted_names_set, einstr in contractions:
    contracted_names = sorted(contracted_names_set)
    input_str, result_names = einstr.split('->')
    input_names = input_str.split(',')

    # switch on the number of operands to be processed in this loop iteration.
    # every case here sets 'operand' and 'names'.
    if len(operand_indices) == 1:
      operand = operands.pop(operand_indices[0])
      names, = input_names
      counts = collections.Counter(names)

      # sum out unique contracted indices with a single reduce-sum
      uniques = [name for name in contracted_names if counts[name] == 1]
      operand, names = sum_uniques(operand, names, uniques)

      # for every repeated index, do a contraction against an identity matrix
      operand, names = sum_repeats(operand, names, counts, result_names)

    elif len(operand_indices) == 2:
      lhs, rhs = map(operands.pop, operand_indices)
      lhs_names, rhs_names = input_names

      # handle cases where one side of a contracting or batch dimension is 1
      # but its counterpart is not.
      lhs, lhs_names = filter_singleton_dims(lhs, lhs_names, shape(rhs),
                                             rhs_names)
      rhs, rhs_names = filter_singleton_dims(rhs, rhs_names, shape(lhs),
                                             lhs_names)

      lhs_counts = collections.Counter(lhs_names)
      rhs_counts = collections.Counter(rhs_names)

      # sum out unique contracted indices in lhs and rhs
      lhs_uniques = [name for name in contracted_names
                     if lhs_counts[name] == 1 and rhs_counts[name] == 0]
      lhs, lhs_names = sum_uniques(lhs, lhs_names, lhs_uniques)

      rhs_uniques = [name for name in contracted_names
                     if rhs_counts[name] == 1 and lhs_counts[name] == 0]
      rhs, rhs_names = sum_uniques(rhs, rhs_names, rhs_uniques)

      # for every repeated index, contract against an identity matrix
      lhs, lhs_names = sum_repeats(lhs, lhs_names, lhs_counts,
                                   result_names + rhs_names)
      rhs, rhs_names = sum_repeats(rhs, rhs_names, rhs_counts,
                                   result_names + lhs_names)

      lhs_or_rhs_names = set(lhs_names) | set(rhs_names)
      contracted_names = [x for x in contracted_names if x in lhs_or_rhs_names]
      lhs_and_rhs_names = set(lhs_names) & set(rhs_names)
      batch_names = [x for x in result_names if x in lhs_and_rhs_names]

      lhs_batch, rhs_batch = unzip2((lhs_names.find(n), rhs_names.find(n))
                                    for n in batch_names)

      # NOTE(mattjj): this can fail non-deterministically in python3, maybe
      # due to opt_einsum
      assert _all(
        name in lhs_names and name in rhs_names and
        lhs.shape[lhs_names.index(name)] == rhs.shape[rhs_names.index(name)]
        for name in contracted_names)

      # contract using lax.dot_general
      batch_names_str = ''.join(batch_names)
      lhs_cont, rhs_cont = unzip2((lhs_names.index(n), rhs_names.index(n))
                                  for n in contracted_names)
      dimension_numbers = ((lhs_cont, rhs_cont), (lhs_batch, rhs_batch))
      operand = lax.dot_general(lhs, rhs, dimension_numbers, precision)
      deleted_names = batch_names_str + ''.join(contracted_names)
      names = (batch_names_str + _removechars(lhs_names, deleted_names)
               + _removechars(rhs_names, deleted_names))
    else:
      raise NotImplementedError  # if this is actually reachable, open an issue!

    # the resulting 'operand' with axis labels 'names' should be a permutation
    # of the desired result
    assert len(names) == len(result_names) == len(set(names))
    assert set(names) == set(result_names)
    if names != result_names:
      perm = tuple([names.index(name) for name in result_names])
      operand = lax.transpose(operand, perm)
    operands.append(operand)  # used in next iteration

  return operands[0]


def _movechars(s, src, dst):
  """Helper for einsum string munging, like moveaxis on identifier strings."""
  chars = [c for i, c in enumerate(s) if i not in src]
  for i, j in sorted(zip(dst, src)):
    chars.insert(i, s[j])
  return ''.join(chars)


@_wraps(np.inner, lax_description=_PRECISION_DOC)
def inner(a, b, *, precision=None):
  if ndim(a) == 0 or ndim(b) == 0:
    return a * b
  return tensordot(a, b, (-1, -1), precision=precision)


@_wraps(np.outer)
def outer(a, b, out=None):
  if out:
    raise NotImplementedError("The 'out' argument to outer is not supported.")
  a, b = _promote_dtypes(a, b)
  return ravel(a)[:, None] * ravel(b)

@partial(jit, static_argnums=(2, 3, 4))
def _cross(a, b, axisa, axisb, axisc):
  a = moveaxis(a, axisa, -1)
  b = moveaxis(b, axisb, -1)

  if a.shape[-1] not in (2, 3) or b.shape[-1] not in (2, 3):
    raise ValueError("Dimension must be either 2 or 3 for cross product")

  if a.shape[-1] == 2 and b.shape[-1] == 2:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

  a0 = a[..., 0]
  a1 = a[..., 1]
  a2 = a[..., 2] if a.shape[-1] == 3 else zeros_like(a0)
  b0 = b[..., 0]
  b1 = b[..., 1]
  b2 = b[..., 2] if b.shape[-1] == 3 else zeros_like(b0)
  c = array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])
  return moveaxis(c, 0, axisc)

@_wraps(np.cross)
def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
  if axis is not None:
    axisa = axis
    axisb = axis
    axisc = axis
  return _cross(a, b, axisa, axisb, axisc)

@_wraps(np.kron)
def kron(a, b):
  a, b = _promote_dtypes(a, b)
  if ndim(a) < ndim(b):
    a = reshape(a, (1,) * (ndim(b) - ndim(a)) + shape(a))
  elif ndim(b) < ndim(a):
    b = reshape(b, (1,) * (ndim(a) - ndim(b)) + shape(b))
  a_reshaped = reshape(a, [i for d in shape(a) for i in (d, 1)])
  b_reshaped = reshape(b, [i for d in shape(b) for i in (1, d)])
  out_shape = tuple(np.multiply(shape(a), shape(b)))
  return reshape(lax.mul(a_reshaped, b_reshaped), out_shape)


@_wraps(np.vander)
def vander(x, N=None, increasing=False):
  x = asarray(x)
  dtype = _dtype(x)
  if ndim(x) != 1:
    raise ValueError("x must be a one-dimensional array")
  x_shape = shape(x)
  N = N or x_shape[0]
  if N < 0:
    raise ValueError("N must be nonnegative")

  iota = lax.iota(dtype, N)
  if not increasing:
    iota = lax.sub(lax._const(iota, N - 1), iota)

  return power(x[..., None], iota)


### Misc


@_wraps(np.argwhere)
def argwhere(a):
  result = transpose(vstack(nonzero(a)))
  if ndim(a) == 0:
    return result[:0].reshape(result.shape[0], 0)
  return result.reshape(result.shape[0], ndim(a))


@_wraps(np.argmax)
def argmax(a, axis=None):
  if axis is None:
    a = ravel(a)
    axis = 0
  if a.shape[axis] == 0:
    raise ValueError("attempt to get argmax of an empty sequence")
  return lax.argmax(a, _canonicalize_axis(axis, a.ndim), int64)

@_wraps(np.argmin)
def argmin(a, axis=None):
  if axis is None:
    a = ravel(a)
    axis = 0
  if a.shape[axis] == 0:
    raise ValueError("attempt to get argmin of an empty sequence")
  return lax.argmin(a, _canonicalize_axis(axis, a.ndim), int64)


_NANARG_DOC = """\
Warning: jax.numpy.arg{} returns -1 for all-NaN slices and does not raise
an error.
"""

@_wraps(np.nanargmax, lax_description=_NANARG_DOC.format("max"))
def nanargmax(a, axis=None):
  if not issubdtype(_dtype(a), inexact):
    return argmax(a, axis=axis)
  nan_mask = isnan(a)
  a = where(nan_mask, -inf, a)
  res = argmax(a, axis=axis)
  return where(all(nan_mask, axis=axis), -1, res)

@_wraps(np.nanargmin, lax_description=_NANARG_DOC.format("min"))
def nanargmin(a, axis=None):
  if not issubdtype(_dtype(a), inexact):
    return argmin(a, axis=axis)
  nan_mask = isnan(a)
  a = where(nan_mask, inf, a)
  res = argmin(a, axis=axis)
  return where(all(nan_mask, axis=axis), -1, res)


@_wraps(np.sort)
def sort(a, axis=-1, kind='quicksort', order=None):
  if kind != 'quicksort':
    warnings.warn("'kind' argument to sort is ignored.")
  if order is not None:
    raise ValueError("'order' argument to sort is not supported.")

  if axis is None:
    return lax.sort(a.ravel(), dimension=0)
  else:
    return lax.sort(a, dimension=_canonicalize_axis(axis, ndim(a)))

@_wraps(np.sort_complex)
def sort_complex(a):
  a = lax.sort(a, dimension=0)
  return lax.convert_element_type(a, result_type(a, dtypes.canonicalize_dtype(complex_)))

@_wraps(np.lexsort)
def lexsort(keys, axis=-1):
  keys = tuple(keys)
  if len(keys) == 0:
    raise TypeError("need sequence of keys with len > 0 in lexsort")
  if len(set(shape(key) for key in keys)) > 1:
    raise ValueError("all keys need to be the same shape")
  if ndim(keys[0]) == 0:
    return np.int64(0)
  axis = _canonicalize_axis(axis, ndim(keys[0]))
  iota = lax.broadcasted_iota(np.int64, shape(keys[0]), axis)
  return lax.sort((*keys[::-1], iota), dimension=axis, num_keys=len(keys))[-1]


@_wraps(np.argsort)
def argsort(a, axis=-1, kind='quicksort', order=None):
  if kind != 'quicksort':
    warnings.warn("'kind' argument to argsort is ignored.")
  if order is not None:
    raise ValueError("'order' argument to argsort is not supported.")

  if axis is None:
    return argsort(a.ravel(), 0)
  else:
    axis = _canonicalize_axis(axis, ndim(a))
    iota = lax.broadcasted_iota(np.int64, shape(a), axis)
    _, perm = lax.sort_key_val(a, iota, dimension=axis)
    return perm


@_wraps(np.msort)
def msort(a):
  return sort(a, axis=0)


@partial(jit, static_argnums=(2,))
def _roll(a, shift, axis):
  a = asarray(a)
  a_shape = shape(a)
  if axis is None:
    return lax.reshape(roll(ravel(a), shift, axis=0), a_shape)

  a_ndim = len(a_shape)
  shift = asarray(shift)
  axis = np.asarray(axis)
  b_shape = lax.broadcast_shapes(shift.shape, axis.shape, (1,))
  if len(b_shape) != 1:
    msg = "'shift' and 'axis' arguments to roll must be scalars or 1D arrays"
    raise ValueError(msg)

  for x, i in zip(broadcast_to(shift, b_shape),
                  np.broadcast_to(axis, b_shape)):
    i = _canonicalize_axis(i, a_ndim)
    x = remainder(x, (a_shape[i] or 1))
    a = lax.concatenate((a, a), i)
    a = lax.dynamic_slice_in_dim(a, a_shape[i] - x, a_shape[i], axis=i)
  return a


@_wraps(np.roll)
def roll(a, shift, axis=None):
  return _roll(a, shift, axis)


@_wraps(np.rollaxis)
def rollaxis(a, axis, start=0):
  a_ndim = ndim(a)
  if not (-a_ndim <= axis < a_ndim):
    raise ValueError(f"axis={axis} is out of bounds for array of dimension {a_ndim}")
  if not (-a_ndim <= start <= a_ndim):
    raise ValueError(f"start={start} must satisfy {-a_ndim}<=start<={a_ndim}")
  if start < 0:
    start += a_ndim
  if axis < 0:
    axis += a_ndim
  if start > axis:
    start -= 1
  return moveaxis(a, axis, start)


@_wraps(np.packbits)
def packbits(a, axis=None, bitorder='big'):
  a = asarray(a)
  if not (issubdtype(dtype(a), integer) or issubdtype(dtype(a), bool_)):
    raise TypeError('Expected an input array of integer or boolean data type')
  if bitorder not in ['little', 'big']:
    raise ValueError("'order' must be either 'little' or 'big'")
  a = (a > 0).astype('uint8')
  bits = arange(8, dtype='uint8')
  if bitorder == 'big':
    bits = bits[::-1]
  if axis is None:
    a = ravel(a)
    axis = 0
  a = swapaxes(a, axis, -1)

  remainder = a.shape[-1] % 8
  if remainder:
    a = pad(a, (a.ndim - 1) * [(0, 0)] + [(0, 8 - remainder)])

  a = a.reshape(a.shape[:-1] + (a.shape[-1] // 8, 8))
  packed = (a << bits).sum(-1).astype('uint8')
  return swapaxes(packed, axis, -1)


@_wraps(np.unpackbits)
def unpackbits(a, axis=None, count=None, bitorder='big'):
  a = asarray(a)
  if dtype(a) != uint8:
    raise TypeError("Expected an input array of unsigned byte data type")
  if bitorder not in ['little', 'big']:
    raise ValueError("'order' must be either 'little' or 'big'")
  bits = asarray(1) << arange(8, dtype='uint8')
  if bitorder == 'big':
    bits = bits[::-1]
  if axis is None:
    a = a.ravel()
    axis = 0
  a = swapaxes(a, axis, -1)
  unpacked = ((a[..., None] & bits) > 0).astype('uint8')
  unpacked = unpacked.reshape(unpacked.shape[:-2] + (-1,))[..., :count]
  return swapaxes(unpacked, axis, -1)


@_wraps(np.take)
def take(a, indices, axis=None, out=None, mode=None):
  if out:
    raise NotImplementedError("The 'out' argument to np.take is not supported.")

  a = asarray(a)
  indices = asarray(indices)

  if axis is None:
    a = ravel(a)
    axis = 0
  axis = _canonicalize_axis(axis, ndim(a))

  if mode == "raise":
    # TODO(phawkins): we have no way to report out of bounds errors yet.
    raise NotImplementedError("The 'raise' mode to np.take is not supported.")
  elif mode == "wrap":
    indices = mod(indices, _constant_like(indices, a.shape[axis]))
  elif mode != "clip" and mode is not None:
    raise ValueError("Invalid mode '{}' for np.take".format(mode))

  index_dims = len(shape(indices))
  slice_sizes = list(shape(a))
  slice_sizes[axis] = _min(indices.size, 1)
  dnums = lax.GatherDimensionNumbers(
    offset_dims=tuple(
      list(range(axis)) +
      list(range(axis + index_dims, len(a.shape) + index_dims - 1))),
    collapsed_slice_dims=(axis,),
    start_index_map=(axis,))
  return lax.gather(a, indices[..., None], dimension_numbers=dnums,
                    slice_sizes=tuple(slice_sizes))


def _normalize_index(index, axis_size):
  """Normalizes an index value in the range [-N, N) to the range [0, N)."""
  if type(axis_size) is Poly:
    return index + axis_size if index < 0 else index

  return lax.select(
    lax.lt(index, _constant_like(index, 0)),
    lax.add(index, _constant_like(index, axis_size)),
    index)

@partial(jit, static_argnums=(2,))
def _take_along_axis(arr, indices, axis):
  if axis is None:
    if ndim(indices) != 1:
      msg = "take_along_axis indices must be 1D if axis=None, got shape {}"
      raise ValueError(msg.format(indices.shape))
    return take_along_axis(arr.ravel(), indices, 0)
  rank = ndim(arr)
  if rank != ndim(indices):
    msg = "indices and arr must have the same number of dimensions; {} vs. {}"
    raise ValueError(msg.format(ndim(indices), ndim(arr)))
  axis = _canonicalize_axis(axis, rank)

  def replace(tup, val):
    lst = list(tup)
    lst[axis] = val
    return tuple(lst)

  bcast_shape = lax.broadcast_shapes(replace(arr.shape, 1), replace(indices.shape, 1))
  indices = broadcast_to(indices, replace(bcast_shape, indices.shape[axis]))
  arr     = broadcast_to(arr,     replace(bcast_shape, arr.shape[axis]))

  axis_size = arr.shape[axis]
  arr_shape = replace(arr.shape, 1)
  idx_shape = indices.shape
  out_shape = lax.broadcast_shapes(idx_shape, arr_shape)

  index_dims = [i for i, idx in enumerate(idx_shape) if i == axis or idx != 1]

  gather_index_shape = tuple(np.array(out_shape)[index_dims]) + (1,)
  gather_indices = []
  slice_sizes = []
  offset_dims = []
  start_index_map = []
  collapsed_slice_dims = []
  j = 0
  for i in range(rank):
    if i == axis:
      indices = _normalize_index(indices, axis_size)
      gather_indices.append(lax.reshape(indices, gather_index_shape))
      slice_sizes.append(1)
      start_index_map.append(i)
      collapsed_slice_dims.append(i)
      j += 1
    elif idx_shape[i] != 1:
      iota = lax.iota(_dtype(indices), out_shape[i])
      iota = lax.tie_in(arr, iota)
      iota = lax.broadcast_in_dim(iota, gather_index_shape, (j,))
      gather_indices.append(iota)
      slice_sizes.append(1)
      start_index_map.append(i)
      collapsed_slice_dims.append(i)
      j += 1
    else:
      # If idx_shape[i] == 1, we can just take the entirety of the arr's axis
      # and avoid forming an iota index.
      offset_dims.append(i)
      slice_sizes.append(arr_shape[i])

  gather_indices = lax.concatenate(gather_indices, dimension=j)
  dnums = lax.GatherDimensionNumbers(
    offset_dims=tuple(offset_dims),
    collapsed_slice_dims=tuple(collapsed_slice_dims),
    start_index_map=tuple(start_index_map))
  return lax.gather(arr, gather_indices, dnums, tuple(slice_sizes))


@_wraps(getattr(np, "take_along_axis", None), update_doc=False)
def take_along_axis(arr, indices, axis):
  return _take_along_axis(arr, indices, axis)


### SetOps

@partial(jit, static_argnums=1)
def _unique1d_sorted_mask(ar, optional_indices=False):
  """
  Helper function for unique which is jit-able
  """

  ar = asarray(ar).flatten()

  if optional_indices:
    perm = ar.argsort()
    aux = ar[perm]
  else:
    aux = ar.sort()

  mask = empty(aux.shape, dtype=bool_)
  mask = ops.index_update(mask, ops.index[:1], True)
  mask = ops.index_update(mask, ops.index[1:], aux[1:] != aux[:-1])

  if optional_indices:
    return aux, mask, perm
  else:
    return aux, mask

def _unique1d(ar, return_index=False, return_inverse=False,
              return_counts=False):
  """
  Find the unique elements of an array, ignoring shape.
  """

  optional_indices = return_index or return_inverse

  if optional_indices:
    aux, mask, perm = _unique1d_sorted_mask(ar, optional_indices)
  else:
    aux, mask = _unique1d_sorted_mask(ar, optional_indices)

  ret = (aux[mask],)
  if return_index:
    ret += (perm[mask],)
  if return_inverse:
    imask = cumsum(mask) - 1
    inv_idx = zeros(mask.shape, dtype=dtypes.canonicalize_dtype(int_))
    inv_idx = ops.index_update(inv_idx, perm, imask)
    ret += (inv_idx,)
  if return_counts:
    idx = concatenate(nonzero(mask) + (array([mask.size]),))
    ret += (diff(idx),)
  return ret

@_wraps(np.unique)
def unique(ar, return_index=False, return_inverse=False,
           return_counts=False, axis=None):

  if iscomplexobj(ar):
    raise NotImplementedError(
          "np.unique is not implemented for complex valued arrays")

  if axis is None:
    ret = _unique1d(ar, return_index, return_inverse, return_counts)
    if len(ret) == 1:
      return ret[0]
    else:
      return ret

  raise NotImplementedError(
        "np.unique is not implemented for the axis argument")

### Indexing

def _rewriting_take(arr, idx):
  # Computes arr[idx].
  # All supported cases of indexing can be implemented as an XLA gather,
  # followed by an optional reverse and broadcast_in_dim.
  arr = asarray(arr)
  treedef, static_idx, dynamic_idx = _split_index_for_jit(idx)
  return _gather(arr, treedef, static_idx, dynamic_idx)

# TODO(phawkins): re-enable jit after fixing excessive recompilation for
# slice indexes (e.g., slice(0, 5, None), slice(10, 15, None), etc.).
# @partial(jit, static_argnums=(1, 2))
def _gather(arr, treedef, static_idx, dynamic_idx):
  idx = _merge_static_and_dynamic_indices(treedef, static_idx, dynamic_idx)
  indexer = _index_to_gather(shape(arr), idx)  # shared with _scatter_update
  y = arr

  # Avoid calling gather if the slice shape is empty, both as a fast path and to
  # handle cases like zeros(0)[array([], int32)].
  if _prod(indexer.slice_shape) == 0:
    return zeros(indexer.slice_shape, dtype=y.dtype)

  # We avoid generating a gather when indexer.gather_indices.size is empty.
  if indexer.gather_indices.size:
    y = lax.gather(y, indexer.gather_indices, indexer.dnums,
                   indexer.gather_slice_shape)

  # Reverses axes with negative strides.
  if indexer.reversed_y_dims:
    y = lax.rev(y, indexer.reversed_y_dims)

  # This adds np.newaxis/None dimensions.
  return expand_dims(y, indexer.newaxis_dims)

_Indexer = collections.namedtuple("_Indexer", [
  # The expected shape of the slice output.
  "slice_shape",

  # The slice shape to pass to lax.gather().
  "gather_slice_shape",

  # The gather indices to use.
  "gather_indices",

  # A GatherDimensionNumbers object describing the gather to perform.
  "dnums",

  # Slice dimensions that have negative strides, and so must be reversed after
  # the gather.
  "reversed_y_dims",

  # Keep track of any axes created by `newaxis`. These must be inserted for
  # gathers and eliminated for scatters.
  "newaxis_dims",
])

def _split_index_for_jit(idx):
  """Splits indices into necessarily-static and dynamic parts.

  Used to pass indices into `jit`-ted function.
  """
  # Convert list indices to tuples in cases (deprecated by NumPy.)
  idx = _eliminate_deprecated_list_indexing(idx)

  # Expand any (concrete) boolean indices. We can then use advanced integer
  # indexing logic to handle them.
  idx = _expand_bool_indices(idx)

  leaves, treedef = tree_flatten(idx)
  dynamic = [None] * len(leaves)
  static = [None] * len(leaves)
  for i, x in enumerate(leaves):
    if x is Ellipsis:
      static[i] = x
    elif isinstance(x, slice):
      # slice objects aren't hashable.
      static[i] = (x.start, x.stop, x.step)
    else:
      dynamic[i] = x
  return treedef, tuple(static), dynamic

def _merge_static_and_dynamic_indices(treedef, static_idx, dynamic_idx):
  """Recombines indices that were split by _split_index_for_jit."""
  idx = []
  for s, d in zip(static_idx, dynamic_idx):
    if d is not None:
      idx.append(d)
    elif isinstance(s, tuple):
      idx.append(slice(s[0], s[1], s[2]))
    else:
      idx.append(s)
  return treedef.unflatten(idx)

def _int(aval):
  return not aval.shape and issubdtype(aval.dtype, integer)

def _index_to_gather(x_shape, idx):
  # Remove ellipses and add trailing slice(None)s.
  idx = _canonicalize_tuple_index(len(x_shape), idx)

  # Check for advanced indexing:
  # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing

  # Do the advanced indexing axes appear contiguously? If not, NumPy semantics
  # move the advanced axes to the front.
  advanced_axes_are_contiguous = False

  advanced_indexes = None

  # The positions of the advanced indexing axes in `idx`.
  idx_advanced_axes = []

  # The positions of the advanced indexes in x's shape.
  # collapsed, after None axes have been removed. See below.
  x_advanced_axes = None

  if _is_advanced_int_indexer(idx):
    idx_no_nones = [(i, d) for i, d in enumerate(idx) if d is not None]
    advanced_pairs = (
      (asarray(e), i, j) for j, (i, e) in enumerate(idx_no_nones)
      if (isinstance(e, Sequence) or isinstance(e, ndarray)))
    advanced_pairs = ((_normalize_index(e, x_shape[j]), i, j)
                      for e, i, j in advanced_pairs)
    advanced_indexes, idx_advanced_axes, x_advanced_axes = zip(*advanced_pairs)
    advanced_axes_are_contiguous = np.all(np.diff(idx_advanced_axes) == 1)

  x_axis = 0  # Current axis in x.
  y_axis = 0  # Current axis in y, before collapsing. See below.
  collapsed_y_axis = 0  # Current axis in y, after collapsing.

  # Scatter dimension numbers.
  offset_dims = []
  collapsed_slice_dims = []
  start_index_map = []

  use_64bit_index = _any([type(d) is Poly or d >= (1 << 31) for d in x_shape])
  index_dtype = int64 if use_64bit_index else int32
  gather_indices = np.zeros((0,), dtype=index_dtype)  # use np to save a compilation

  # We perform three transformations to y before the scatter op, in order:
  # First, y is broadcast to slice_shape. In general `y` only need broadcast to
  # the right shape.
  slice_shape = []

  # Next, y is squeezed to remove newaxis_dims. This removes np.newaxis/`None`
  # indices, which the scatter cannot remove itself.
  newaxis_dims = []

  # Finally, we reverse reversed_y_dims to handle slices with negative strides.
  reversed_y_dims = []

  gather_slice_shape = []

  for idx_pos, i in enumerate(idx):
    # Handle the advanced indices here if:
    # * the advanced indices were not contiguous and we are the start.
    # * we are at the position of the first advanced index.
    if (advanced_indexes is not None and
        (advanced_axes_are_contiguous and idx_pos == idx_advanced_axes[0] or
         not advanced_axes_are_contiguous and idx_pos == 0)):
      advanced_indexes = broadcast_arrays(*advanced_indexes)
      shape = advanced_indexes[0].shape
      ndim = len(shape)
      advanced_indexes = [
        lax.convert_element_type(lax.reshape(a, shape + (1,)), index_dtype)
        for a in advanced_indexes]

      # Broadcast gather_indices from [..., k] to [..., 1, 1, ..., 1, k].
      gather_indices = lax.broadcast_in_dim(
        gather_indices, np.insert(gather_indices.shape, -1, shape),
        tuple(range(gather_indices.ndim - 1)) + (gather_indices.ndim + ndim - 1,))
      gather_indices = concatenate([gather_indices] + advanced_indexes, -1)
      start_index_map.extend(x_advanced_axes)
      collapsed_slice_dims.extend(x_advanced_axes)
      slice_shape.extend(shape)
      y_axis += ndim
      collapsed_y_axis += ndim

    # Per-index bookkeeping for advanced indexes.
    if idx_pos in idx_advanced_axes:
      x_axis += 1
      gather_slice_shape.append(1)
      continue

    try:
      abstract_i = core.get_aval(i)
    except TypeError:
      abstract_i = None
    # Handle basic int indexes.
    if (isinstance(abstract_i, ConcreteArray) or
        isinstance(abstract_i, ShapedArray)) and _int(abstract_i):
      if x_shape[x_axis] == 0:
        # XLA gives error when indexing into an axis of size 0
        raise IndexError(f"index is out of bounds for axis {x_axis} with size 0")
      i = _normalize_index(i, x_shape[x_axis])
      if type(i) is Poly:
        # dummy index if i is polynomial, doesn't matter for shape inference
        # TODO(mattjj,j-towns,juliuskunze): revise this logic
        i = 0
      i = lax.convert_element_type(i, index_dtype)
      i = broadcast_to(i, tuple(gather_indices.shape[:-1]) + (1,))
      gather_indices = concatenate((gather_indices, i), -1)
      collapsed_slice_dims.append(x_axis)
      gather_slice_shape.append(1)
      start_index_map.append(x_axis)
      x_axis += 1
    # Handle np.newaxis (None)
    elif i is None:
      slice_shape.append(1)
      newaxis_dims.append(y_axis)
      y_axis += 1
    # Handle slice(None)
    elif _is_slice_none(i):
      slice_shape.append(x_shape[x_axis])
      gather_slice_shape.append(x_shape[x_axis])
      offset_dims.append(collapsed_y_axis)
      collapsed_y_axis += 1
      y_axis += 1
      x_axis += 1
    # Handle slice index (only static, otherwise an error is raised)
    elif isinstance(i, slice):
      if not _all(elt is None or type(elt) is Poly
                  or type(core.get_aval(elt)) is ConcreteArray
                  for elt in (i.start, i.stop, i.step)):
        msg = ("Array slice indices must have static start/stop/step to be used "
               "with NumPy indexing syntax. To index a statically sized "
               "array at a dynamic position, try lax.dynamic_slice/"
               "dynamic_update_slice (JAX does not support dynamically sized "
               "arrays within JIT compiled functions).")
        raise IndexError(msg)
      start, limit, stride, needs_rev = _static_idx(i, x_shape[x_axis])
      if needs_rev:
        reversed_y_dims.append(collapsed_y_axis)
      if stride == 1:
        i = lax.convert_element_type(start, index_dtype)
        i = broadcast_to(i, tuple(gather_indices.shape[:-1]) + (1,))
        gather_indices = concatenate((gather_indices, i), -1)
        slice_shape.append(limit - start)
        gather_slice_shape.append(limit - start)
        offset_dims.append(collapsed_y_axis)
        start_index_map.append(x_axis)
      else:
        i = arange(start, limit, stride, dtype=index_dtype)
        size = i.shape[0]
        slice_shape.append(size)
        gather_slice_shape.append(1)
        gather_indices_shape = tuple(gather_indices.shape[:-1]) + (size,)
        i = lax.broadcast_in_dim(
            i, shape=gather_indices_shape + (1,),
            broadcast_dimensions=(len(gather_indices_shape) - 1,))
        gather_indices = lax.broadcast_in_dim(
            gather_indices,
            shape=gather_indices_shape + (len(start_index_map),),
            broadcast_dimensions=(
              tuple(range(len(gather_indices_shape) - 1)) +
              (len(gather_indices_shape),)))
        gather_indices = concatenate(
          (gather_indices, i), len(gather_indices_shape))
        start_index_map.append(x_axis)
        collapsed_slice_dims.append(x_axis)

      collapsed_y_axis += 1
      y_axis += 1
      x_axis += 1
    else:
      if (abstract_i is not None and
          not (issubdtype(abstract_i.dtype, integer) or issubdtype(abstract_i.dtype, bool_))):
        msg = ("Indexer must have integer or boolean type, got indexer "
               "with type {} at position {}, indexer value {}")
        raise TypeError(msg.format(abstract_i.dtype.name, idx_pos, i))

      msg = "Indexing mode not yet supported. Open a feature request!\n{}"
      raise IndexError(msg.format(idx))

  dnums = lax.GatherDimensionNumbers(
    offset_dims = tuple(offset_dims),
    collapsed_slice_dims = tuple(sorted(collapsed_slice_dims)),
    start_index_map = tuple(start_index_map)
  )
  return _Indexer(
    slice_shape=slice_shape,
    newaxis_dims=tuple(newaxis_dims),
    gather_slice_shape=gather_slice_shape,
    reversed_y_dims=reversed_y_dims,
    dnums=dnums,
    gather_indices=gather_indices)

def _should_unpack_list_index(x):
  """Helper for _eliminate_deprecated_list_indexing."""
  return (isinstance(x, ndarray) and np.ndim(x) != 0
          or isinstance(x, Sequence)
          or isinstance(x, slice) or x is Ellipsis or x is None)

def _eliminate_deprecated_list_indexing(idx):
  # "Basic slicing is initiated if the selection object is a non-array,
  # non-tuple sequence containing slice objects, [Ellipses, or newaxis
  # objects]". Detects this case and canonicalizes to a tuple. This case is
  # deprecated by NumPy and exists for backward compatibility.
  if not isinstance(idx, tuple):
    if isinstance(idx, Sequence) and not isinstance(idx, ndarray):
      if _any(_should_unpack_list_index(i) for i in idx):
        idx = tuple(idx)
      else:
        idx = (idx,)
    else:
      idx = (idx,)
  return idx

def _expand_bool_indices(idx):
  """Converts concrete bool indexes into advanced integer indexes."""
  out = []
  for i in idx:
    try:
      abstract_i = core.get_aval(i)
    except TypeError:
      abstract_i = None
    if (isinstance(abstract_i, ShapedArray) and issubdtype(abstract_i.dtype, bool_)
          or isinstance(i, list) and _all(not _shape(e) and issubdtype(_dtype(e), bool_)
                                          for e in i)):
      if isinstance(i, list):
        i = array(i)
        abstract_i = core.get_aval(i)

      if not type(abstract_i) is ConcreteArray:
        msg = ("Array boolean indices must be static (e.g. no dependence on an "
               "argument to a jit or vmap function).")
        raise IndexError(msg)
      else:
        out.extend(np.where(i))
    else:
      out.append(i)
  return tuple(out)

def _is_slice_none(idx):
  """Return True if idx is equal to slice(None), False otherwise."""
  if isinstance(idx, slice):
    return idx.start is None and idx.stop is None and idx.step is None

# TODO(mattjj): clean up this logic
def _is_advanced_int_indexer(idx):
  """Returns True if idx should trigger int array indexing, False otherwise."""
  # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
  assert isinstance(idx, tuple)
  if _all(np.ndim(elt) == 0 for elt in idx):
    return False
  return _all(e is None or e is Ellipsis or isinstance(e, slice)
              or _is_int_arraylike(e) for e in idx)

def _is_int_arraylike(x):
  """Returns True if x is array-like with integer dtype, False otherwise."""
  return (isinstance(x, int) and not isinstance(x, bool)
          or issubdtype(getattr(x, "dtype", None), np.integer)
          or isinstance(x, (list, tuple)) and _all(_is_int_arraylike(e) for e in x))


def _canonicalize_tuple_index(arr_ndim, idx):
  """Helper to remove Ellipsis and add in the implicit trailing slice(None)."""
  len_without_none = _sum(1 for e in idx if e is not None and e is not Ellipsis)
  if len_without_none > arr_ndim:
    msg = "Too many indices for array: {} non-None/Ellipsis indices for dim {}."
    raise IndexError(msg.format(len_without_none, arr_ndim))
  ellipses = (i for i, elt in enumerate(idx) if elt is Ellipsis)
  ellipsis_index = next(ellipses, None)
  if ellipsis_index is not None:
    if next(ellipses, None) is not None:
      msg = "Multiple ellipses (...) not supported: {}."
      raise IndexError(msg.format(list(map(type, idx))))
    colons = (slice(None),) * (arr_ndim - len_without_none)
    idx = idx[:ellipsis_index] + colons + idx[ellipsis_index + 1:]
  elif len_without_none < arr_ndim:
    colons = (slice(None),) * (arr_ndim - len_without_none)
    idx = tuple(idx) + colons
  return idx

def _polymorphic_slice_indices(idx: slice, size: Union[int, Poly]):
  # like idx.indices(size), but allows for polymorphic indices and size
  # see https://github.com/python/cpython/blob/6d6508765514c7c10719478a0430f5e47c9a96ac/Objects/sliceobject.c#L372
  assert isinstance(idx, slice)

  step = 1 if idx.step is None else idx.step
  step_is_negative = step < 0
  lower = -1 if step_is_negative else 0
  upper = size + lower

  def sanitize(index, default):
    if index is None:
      return default
    elif type(index) is Poly:
      return index
    elif index < 0:
      return _max(index + size, lower)
    else:
      return _min(index, upper)

  start = sanitize(idx.start, default=upper if step_is_negative else lower)
  stop = sanitize(idx.stop, default=lower if step_is_negative else upper)
  return start, stop, step

def _static_idx(idx: slice, size: Union[int, Poly]):
  """Helper function to compute the static slice start/limit/stride values."""
  if _any(type(s) is Poly for s in (idx.start, idx.stop, idx.step, size)):
    start, stop, step = _polymorphic_slice_indices(idx, size)
  elif isinstance(size, int):
    start, stop, step = idx.indices(size)
  else:
    raise TypeError(size)

  if type(start) is not Poly and type(stop) is not Poly:
    if (step < 0 and stop >= start) or (step > 0 and start >= stop):
      return 0, 0, 1, False  # sliced to size zero

  if step > 0:
    return start, stop, step, False
  else:
    k  = (start - stop - 1) % (-step)
    return stop + k + 1, start + 1, -step, True


blackman = _wrap_numpy_nullary_function(np.blackman)
bartlett = _wrap_numpy_nullary_function(np.bartlett)
hamming = _wrap_numpy_nullary_function(np.hamming)
hanning = _wrap_numpy_nullary_function(np.hanning)
# TODO: lower `kaiser` via lax to allow non-constant beta values.
kaiser = _wrap_numpy_nullary_function(np.kaiser)

def _gcd_cond_fn(xs):
  x1, x2 = xs
  return any(x2 != 0)

def _gcd_body_fn(xs):
  x1, x2 = xs
  x1, x2 = (where(x2 != 0, x2, x1),
            where(x2 != 0, lax.rem(x1, x2), lax._const(x2, 0)))
  return (where(x1 < x2, x2, x1), where(x1 < x2, x1, x2))

@_wraps(getattr(np, "gcd", None))
def gcd(x1, x2):
  if (not issubdtype(_dtype(x1), integer) or
      not issubdtype(_dtype(x2), integer)):
    raise ValueError("Arguments to gcd must be integers.")
  x1, x2 = _promote_dtypes(x1, x2)
  x1, x2 = broadcast_arrays(x1, x2)
  gcd, _ = lax.while_loop(_gcd_cond_fn, _gcd_body_fn,
                          (lax.abs(x1), lax.abs(x2)))
  return gcd


@_wraps(getattr(np, "lcm", None))
def lcm(x1, x2):
  x1, x2 = _promote_dtypes(x1, x2)
  d = gcd(x1, x2)
  return where(d == 0, lax._const(d, 0),
               lax.div(lax.abs(multiply(x1, x2)), d))


@_wraps(np.extract)
def extract(condition, arr):
  return compress(ravel(condition), ravel(arr))


@_wraps(np.compress)
def compress(condition, a, axis=None, out=None):
  if out is not None:
    raise NotImplementedError("out argument is not supported.")
  if ndim(condition) != 1:
    raise ValueError("condition must be a 1D array")
  condition = array(condition).astype(bool)
  a = array(a)
  if axis is None:
    axis = 0
    a = ravel(a)
  else:
    a = moveaxis(a, axis, 0)
  condition, extra = condition[:a.shape[0]], condition[a.shape[0]:]
  if any(extra):
    raise ValueError("condition contains entries that are out of bounds")
  a = a[:condition.shape[0]]
  return moveaxis(a[condition], 0, axis)


@_wraps(np.cov)
def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None,
        aweights=None):
  msg = ("jax.numpy.cov not implemented for nontrivial {}. "
         "Open a feature request at https://github.com/google/jax/issues !")
  if y is not None: raise NotImplementedError(msg.format('y'))
  # These next two are actually implemented, just not tested.
  if fweights is not None: raise NotImplementedError(msg.format('fweights'))
  if aweights is not None: raise NotImplementedError(msg.format('aweights'))

  if m.ndim > 2:
    raise ValueError("m has more than 2 dimensions")  # same as numpy error
  X = array(m, ndmin=2, dtype=dtypes.canonicalize_dtype(result_type(m, float_)))
  if not rowvar and X.shape[0] != 1:
    X = X.T
  if X.shape[0] == 0:
    return array([]).reshape(0, 0)
  if ddof is None:
    ddof = 1 if bias == 0 else 0

  w = None
  if fweights is not None:
    if np.ndim(fweights) > 1:
      raise RuntimeError("cannot handle multidimensional fweights")
    if np.shape(fweights)[0] != X.shape[1]:
      raise RuntimeError("incompatible numbers of samples and fweights")
    w = asarray(fweights)
  if aweights is not None:
    if np.ndim(aweights) > 1:
      raise RuntimeError("cannot handle multidimensional aweights")
    if np.shape(aweights)[0] != X.shape[1]:
      raise RuntimeError("incompatible numbers of samples and aweights")
    w = aweights if w is None else w * aweights

  avg, w_sum = average(X, axis=1, weights=w, returned=True)
  w_sum = w_sum[0]

  if w is None:
    f = X.shape[1] - ddof
  elif ddof == 0:
    f = w_sum
  elif aweights is None:
    f = w_sum - ddof
  else:
    f = w_sum - ddof * sum(w * aweights) / w_sum

  X = X - avg[:, None]
  X_T = X.T if w is None else (X * w).T
  return true_divide(dot(X, X_T.conj()), f).squeeze()


@_wraps(np.corrcoef)
def corrcoef(x, y=None, rowvar=True):
  c = cov(x, y, rowvar)
  if len(shape(c)) == 0:
      # scalar - this should yield nan for values (nan/nan, inf/inf, 0/0), 1 otherwise
      return divide(c, c)
  d = diag(c)
  stddev = sqrt(real(d))
  c = divide(c, stddev[:,None])
  c = divide(c, stddev[None,:])

  real_part = clip(real(c), -1, 1)
  if iscomplexobj(c):
      complex_part = clip(imag(c), -1, 1)
      c = lax.complex(real_part, complex_part)
  else:
      c = real_part
  return c


@_wraps(getattr(np, "quantile", None))
def quantile(a, q, axis=None, out=None, overwrite_input=False,
             interpolation="linear", keepdims=False):
  if overwrite_input or out is not None:
    msg = ("jax.numpy.quantile does not support overwrite_input=True or "
           "out != None")
    raise ValueError(msg)
  return _quantile(a, q, axis, interpolation, keepdims, False)

@_wraps(getattr(np, "nanquantile", None))
def nanquantile(a, q, axis=None, out=None, overwrite_input=False,
                interpolation="linear", keepdims=False):
  if overwrite_input or out is not None:
    msg = ("jax.numpy.nanquantile does not support overwrite_input=True or "
           "out != None")
    raise ValueError(msg)
  return _quantile(a, q, axis, interpolation, keepdims, True)


@partial(jit, static_argnums=(2, 3, 4, 5))
def _quantile(a, q, axis, interpolation, keepdims, squash_nans):
  if interpolation not in ["linear", "lower", "higher", "midpoint", "nearest"]:
    raise ValueError("interpolation can only be 'linear', 'lower', 'higher', "
                     "'midpoint', or 'nearest'")
  a = asarray(a, dtype=promote_types(_dtype(a), float32))
  q = asarray(q, dtype=promote_types(_dtype(q), float32))
  if axis is None:
    a = ravel(a)
    axis = 0
  elif isinstance(axis, tuple):
    raise NotImplementedError("Tuple values for axis are not implemented")
  else:
    axis = _canonicalize_axis(axis, ndim(a))

  q_shape = shape(q)
  q_ndim = ndim(q)
  if q_ndim > 1:
    raise ValueError("q must be have rank <= 1, got shape {}".format(shape(q)))

  a_shape = shape(a)
  a = lax.sort(a, dimension=axis)

  if squash_nans:
    counts = sum(logical_not(isnan(a)), axis=axis, dtype=q.dtype,
                 keepdims=keepdims)
    shape_after_reduction = counts.shape
    q = lax.expand_dims(
      q, tuple(range(q_ndim, len(shape_after_reduction) + q_ndim)))
    counts = lax.expand_dims(counts, tuple(range(q_ndim)))
    q = lax.mul(q, lax.sub(counts, _constant_like(q, 1)))
    low = lax.floor(q)
    high = lax.ceil(q)
    high_weight = lax.sub(q, low)
    low_weight = lax.sub(_constant_like(high_weight, 1), high_weight)

    low = lax.max(_constant_like(low, 0), lax.min(low, counts - 1))
    high = lax.max(_constant_like(high, 0), lax.min(high, counts - 1))
    low = lax.convert_element_type(low, int64)
    high = lax.convert_element_type(high, int64)
    out_shape = q_shape + shape_after_reduction
    index = [lax.broadcasted_iota(int64, out_shape, dim + q_ndim)
             for dim in range(len(shape_after_reduction))]
    if keepdims:
      index[axis] = low
    else:
      index.insert(axis, low)
    low_value = a[tuple(index)]
    index[axis] = high
    high_value = a[tuple(index)]
  else:
    n = a_shape[axis]
    q = lax.mul(q, _constant_like(q, n - 1))
    low = lax.floor(q)
    high = lax.ceil(q)
    high_weight = lax.sub(q, low)
    low_weight = lax.sub(_constant_like(high_weight, 1), high_weight)

    low = lax.clamp(_constant_like(low, 0), low, _constant_like(low, n - 1))
    high = lax.clamp(_constant_like(high, 0), high, _constant_like(high, n - 1))
    low = lax.convert_element_type(low, int64)
    high = lax.convert_element_type(high, int64)

    slice_sizes = list(a_shape)
    slice_sizes[axis] = 1
    dnums = lax.GatherDimensionNumbers(
      offset_dims=tuple(range(
        q_ndim,
        len(a_shape) + q_ndim if keepdims else len(a_shape) + q_ndim - 1)),
      collapsed_slice_dims=() if keepdims else (axis,),
      start_index_map=(axis,))
    low_value = lax.gather(a, low[..., None], dimension_numbers=dnums,
                           slice_sizes=slice_sizes)
    high_value = lax.gather(a, high[..., None], dimension_numbers=dnums,
                            slice_sizes=slice_sizes)
    if q_ndim == 1:
      low_weight = lax.broadcast_in_dim(low_weight, low_value.shape,
                                        broadcast_dimensions=(0,))
      high_weight = lax.broadcast_in_dim(high_weight, high_value.shape,
                                        broadcast_dimensions=(0,))

  if interpolation == "linear":
    result = lax.add(lax.mul(low_value.astype(q.dtype), low_weight),
                     lax.mul(high_value.astype(q.dtype), high_weight))
  elif interpolation == "lower":
    result = low_value
  elif interpolation == "higher":
    result = high_value
  elif interpolation == "nearest":
    pred = lax.le(high_weight, _constant_like(high_weight, 0.5))
    result = lax.select(pred, low_value, high_value)
  elif interpolation == "midpoint":
    result = lax.mul(lax.add(low_value, high_value), _constant_like(low_value, 0.5))
  else:
    raise ValueError(f"interpolation={interpolation!r} not recognized")

  return lax.convert_element_type(result, a.dtype)


@partial(jit, static_argnums=2)
def _searchsorted(a, v, side='left'):
  assert side in ['left', 'right']
  op = operator.lt if side == "right" else operator.le
  v_shape = shape(v)
  v = ravel(v)

  def f(carry, x):
    low, high = carry
    mid = (low + high) // 2
    mask = op(v, a[mid])
    low = where(mask, low, mid)
    high = where(mask, mid, high)
    return (low, high), x

  low = zeros(len(v), dtype=int_)
  high = full(len(v), len(a), dtype=int_)
  n_levels = int(np.ceil(np.log2(len(a) + 1)))
  (low, high), _ = lax.scan(f, (low, high), None, length=n_levels)

  return high.reshape(v_shape)

def searchsorted(a, v, side='left', sorter=None):
  if side not in ['left', 'right']:
    raise ValueError(f"{side!r} is an invalid value for keyword 'side'")
  if sorter is not None:
    raise NotImplementedError("sorter is not implemented")
  a = asarray(a)
  v = asarray(v)
  if ndim(a) != 1:
    raise ValueError("a should be 1-dimensional")
  if size(a) == 0 or size(v) == 0:
    return zeros_like(v, dtype=int)
  return _searchsorted(a, v, side)


@_wraps(np.digitize)
def digitize(x, bins, right=False):
  if len(bins) == 0:
    return zeros(x, dtype=dtypes.canonicalize_dtype(int_))
  side = 'right' if not right else 'left'
  return where(
    bins[-1] >= bins[0],
    searchsorted(bins, x, side=side),
    len(bins) - searchsorted(bins[::-1], x, side=side)
  )


@_wraps(np.percentile)
def percentile(a, q, axis=None, out=None, overwrite_input=False,
               interpolation="linear", keepdims=False):
  q = true_divide(asarray(q), float32(100.0))
  return quantile(a, q, axis=axis, out=out, overwrite_input=overwrite_input,
                  interpolation=interpolation, keepdims=keepdims)

@_wraps(np.nanpercentile)
def nanpercentile(a, q, axis=None, out=None, overwrite_input=False,
                  interpolation="linear", keepdims=False):
  q = true_divide(asarray(q), float32(100.0))
  return nanquantile(a, q, axis=axis, out=out, overwrite_input=overwrite_input,
                     interpolation=interpolation, keepdims=keepdims)

@_wraps(np.median)
def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
  return quantile(a, 0.5, axis=axis, out=out, overwrite_input=overwrite_input,
                  keepdims=keepdims, interpolation='midpoint')

@_wraps(np.nanmedian)
def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
  return nanquantile(a, 0.5, axis=axis, out=out,
                     overwrite_input=overwrite_input, keepdims=keepdims,
                     interpolation='midpoint')


def _astype(arr, dtype):
  lax._check_user_dtype_supported(dtype, "astype")
  return lax.convert_element_type(arr, dtype)

def _view(arr, dtype=None, type=None):
  if type is not None:
    raise NotImplementedError("`type` argument of array.view()")
  if dtype is None:
    return arr
  arr_dtype = _dtype(arr)
  if arr_dtype == dtype:
    return arr
  # bool is implemented as lax:PRED, which is not compatible with lax.bitcast_convert_type.
  # We work around this by casting bool to uint8.
  if arr_dtype == bool_:
    arr = arr.astype(uint8)
  nbits_in = 8 * arr_dtype.itemsize
  nbits_out = 8 * _dtype(dtype).itemsize
  if nbits_in == nbits_out:
    if dtype == bool_:
      return lax.bitcast_convert_type(arr, uint8).astype(dtype)
    return lax.bitcast_convert_type(arr, dtype)
  if nbits_out > nbits_in and (shape(arr)[-1] * nbits_in) % nbits_out != 0:
    raise ValueError("When changing to a larger dtype, its size must be a divisor "
                     "of the total size in bytes of the last axis of the array.")
  byte_dtypes = {8: uint8, 16: uint16, 32: uint32, 64: uint64}
  if nbits_in not in byte_dtypes:
    raise NotImplementedError(f"arr.view() for arr.dtype={arr_dtype}")
  if nbits_out not in byte_dtypes:
    raise NotImplementedError(f"arr.view(dtype) for dtype={dtype}")
  dt_in = byte_dtypes[nbits_in]
  dt_out = byte_dtypes[nbits_out]
  arr_bytes = lax.bitcast_convert_type(arr, dt_in)
  if nbits_in < nbits_out:
    shifts = arange(0, nbits_out, nbits_in, dtype=dt_out)
    arr_bytes = arr_bytes.reshape(arr.shape[:-1] + (-1, nbits_out // nbits_in)).astype(dt_out)
    arr_bytes = (arr_bytes << shifts).sum(-1).astype(dt_out)
  else:
    shifts = arange(0, nbits_in, nbits_out, dtype=dt_in)
    arr_bytes = ((arr_bytes[..., newaxis] >> shifts) & iinfo(dt_out).max).astype(dt_out)
    arr_bytes = arr_bytes.reshape(arr_bytes.shape[:-2] + (-1,))
  if dtype == bool_:
    return lax.bitcast_convert_type(arr_bytes, uint8).astype(dtype)
  return lax.bitcast_convert_type(arr_bytes, dtype)

### track unimplemented functions

def _not_implemented(fun):
  @_wraps(fun)
  def wrapped(*args, **kwargs):
    msg = "Numpy function {} not yet implemented"
    raise NotImplementedError(msg.format(fun))
  return wrapped


### add method and operator overloads to arraylike classes

# We add operator overloads to DeviceArray and ShapedArray. These method and
# operator overloads mainly just forward calls to the corresponding lax_numpy
# functions, which can themselves handle instances from any of these classes.

_scalar_types = (int, float, complex, np.generic)

def _defer_to_unrecognized_arg(binary_op):
  # Ensure that other array types have the chance to override arithmetic.
  def deferring_binary_op(self, other):
    if not isinstance(other, _scalar_types + _arraylike_types + (core.Tracer,)):
      return NotImplemented
    return binary_op(self, other)
  return deferring_binary_op

def _swap_args(f):
  return lambda x, y: f(y, x)

def _unimplemented_setitem(self, i, x):
  msg = ("'{}' object does not support item assignment. JAX arrays are "
         "immutable; perhaps you want jax.ops.index_update or "
         "jax.ops.index_add instead?")
  raise TypeError(msg.format(type(self)))

def _operator_round(number, ndigits=None):
  out = round(number, decimals=ndigits or 0)
  # If `ndigits` is None, for a builtin float round(7.5) returns an integer.
  return out.astype(int_) if ndigits is None else out

_operators = {
    "getitem": _rewriting_take,
    "setitem": _unimplemented_setitem,
    "neg": negative,
    "pos": positive,
    "eq": _defer_to_unrecognized_arg(equal),
    "ne": _defer_to_unrecognized_arg(not_equal),
    "lt": _defer_to_unrecognized_arg(less),
    "le": _defer_to_unrecognized_arg(less_equal),
    "gt": _defer_to_unrecognized_arg(greater),
    "ge": _defer_to_unrecognized_arg(greater_equal),
    "abs": abs,
    "add": _defer_to_unrecognized_arg(add),
    "radd": _defer_to_unrecognized_arg(add),
    "sub": _defer_to_unrecognized_arg(subtract),
    "rsub": _defer_to_unrecognized_arg(_swap_args(subtract)),
    "mul": _defer_to_unrecognized_arg(multiply),
    "rmul": _defer_to_unrecognized_arg(multiply),
    "div": _defer_to_unrecognized_arg(divide),
    "rdiv": _defer_to_unrecognized_arg(_swap_args(divide)),
    "truediv": _defer_to_unrecognized_arg(true_divide),
    "rtruediv": _defer_to_unrecognized_arg(_swap_args(true_divide)),
    "floordiv": _defer_to_unrecognized_arg(floor_divide),
    "rfloordiv": _defer_to_unrecognized_arg(_swap_args(floor_divide)),
    "divmod": _defer_to_unrecognized_arg(divmod),
    "rdivmod": _defer_to_unrecognized_arg(_swap_args(divmod)),
    "mod": _defer_to_unrecognized_arg(mod),
    "rmod": _defer_to_unrecognized_arg(_swap_args(mod)),
    "pow": _defer_to_unrecognized_arg(power),
    "rpow": _defer_to_unrecognized_arg(_swap_args(power)),
    "matmul": _defer_to_unrecognized_arg(matmul),
    "rmatmul": _defer_to_unrecognized_arg(_swap_args(matmul)),
    "and": _defer_to_unrecognized_arg(bitwise_and),
    "rand": _defer_to_unrecognized_arg(bitwise_and),
    "or": _defer_to_unrecognized_arg(bitwise_or),
    "ror": _defer_to_unrecognized_arg(bitwise_or),
    "xor": _defer_to_unrecognized_arg(bitwise_xor),
    "rxor": _defer_to_unrecognized_arg(bitwise_xor),
    "invert": bitwise_not,
    "lshift": _defer_to_unrecognized_arg(left_shift),
    "rshift": _defer_to_unrecognized_arg(right_shift),
    "round": _operator_round,
}

# These numpy.ndarray methods are just refs to an equivalent numpy function
_nondiff_methods = ["all", "any", "argmax", "argmin", "argpartition", "argsort",
                    "nonzero", "searchsorted", "round"]
_diff_methods = ["clip", "conj", "conjugate", "cumprod", "cumsum",
                 "diagonal", "dot", "max", "mean", "min", "prod", "ptp",
                 "ravel", "repeat", "sort", "squeeze", "std", "sum",
                 "swapaxes", "take", "tile", "trace", "transpose", "var"]

# These methods are mentioned explicitly by nondiff_methods, so we create
# _not_implemented implementations of them here rather than in __init__.py.
# TODO(phawkins): implement these.
argpartition = _not_implemented(np.argpartition)
_NOT_IMPLEMENTED = ['argpartition']

# Set up operator, method, and property forwarding on Tracer instances containing
# ShapedArray avals by following the forwarding conventions for Tracer.
# Forward operators using a single-underscore-prefix naming convention:
for operator_name, function in _operators.items():
  setattr(ShapedArray, "_{}".format(operator_name), staticmethod(function))
# Forward methods and properties using core.aval_method and core.aval_property:
for method_name in _nondiff_methods + _diff_methods:
  setattr(ShapedArray, method_name, core.aval_method(globals()[method_name]))
setattr(ShapedArray, "reshape", core.aval_method(_reshape_method))
setattr(ShapedArray, "flatten", core.aval_method(ravel))
setattr(ShapedArray, "T", core.aval_property(transpose))
setattr(ShapedArray, "real", core.aval_property(real))
setattr(ShapedArray, "imag", core.aval_property(imag))
setattr(ShapedArray, "astype", core.aval_method(_astype))
setattr(ShapedArray, "view", core.aval_method(_view))


# Forward operators, methods, and properties on DeviceArray to lax_numpy
# functions (with no Tracers involved; this forwarding is direct)
for operator_name, function in _operators.items():
  setattr(DeviceArray, "__{}__".format(operator_name), function)
for method_name in _nondiff_methods + _diff_methods:
  setattr(DeviceArray, method_name, globals()[method_name])
setattr(DeviceArray, "reshape", _reshape_method)
setattr(DeviceArray, "flatten", ravel)
setattr(DeviceArray, "T", property(transpose))
setattr(DeviceArray, "real", property(real))
setattr(DeviceArray, "imag", property(imag))
setattr(DeviceArray, "astype", _astype)
setattr(DeviceArray, "view", _view)


# Extra methods that are handy
setattr(ShapedArray, "broadcast", core.aval_method(lax.broadcast))
setattr(ShapedArray, "broadcast_in_dim", core.aval_method(lax.broadcast_in_dim))
setattr(ShapedArray, "split", core.aval_method(split))
setattr(DeviceArray, "broadcast", lax.broadcast)
setattr(DeviceArray, "broadcast_in_dim", lax.broadcast_in_dim)
setattr(DeviceArray, "split", split)

def _compress_method(a, condition, axis=None, out=None):
  return compress(condition, a, axis, out)

setattr(ShapedArray, "compress", _compress_method)
setattr(DeviceArray, "compress", _compress_method)

@partial(jit, static_argnums=(1,2,3))
def _multi_slice(arr: DeviceArray,
                 start_indices: Tuple[Tuple[int, ...]],
                 limit_indices: Tuple[Tuple[int, ...]],
                 removed_dims: Tuple[Tuple[int, ...]]):
  """Extracts multiple slices from `arr`.

  This is used to shard DeviceArray arguments to pmap. It's implemented as a
  DeviceArray method here to avoid circular imports.
  """
  results = []
  for starts, limits, removed in safe_zip(start_indices, limit_indices, removed_dims):
    sliced = lax.slice(arr, starts, limits)
    if removed:
      sliced = sliced.reshape(np.delete(sliced.shape, removed_dims))
    results.append(sliced)
  return results
setattr(DeviceArray, "_multi_slice", _multi_slice)


# Syntactic sugar for scatter operations.
class _IndexUpdateHelper:
  # Note: this docstring will appear as the docstring for the `at` property.
  """Indexable helper object to call indexed update functions.

  The `at` property is syntactic sugar for calling the indexed update functions
  defined in :mod:`jax.ops`, and acts as a pure equivalent of in-place
  modificatons.

  In particular:
  - ``x = x.at[idx].set(y)`` is a pure equivalent of ``x[idx] = y``.
  - ``x = x.at[idx].add(y)`` is a pure equivalent of ``x[idx] += y``.
  - ``x = x.at[idx].mul(y)`` is a pure equivalent of ``x[idx] *= y``.
  - ``x = x.at[idx].min(y)`` is a pure equivalent of
      ``x[idx] = minimum(x[idx], y)``.
  - ``x = x.at[idx].max(y)`` is a pure equivalent of
      ``x[idx] = maximum(x[idx], y)``.
  """
  __slots__ = ("array",)

  def __init__(self, array):
    self.array = array

  def __getitem__(self, index):
    return _IndexUpdateRef(self.array, index)

  def __repr__(self):
    return f"_IndexUpdateHelper({repr(self.array)})"


class _IndexUpdateRef:
  """Helper object to call indexed update functions for an (advanced) index.

  This object references a source array and a specific indexer into that array.
  Methods on this object return copies of the source array that have been
  modified at the positions specified by the indexer.
  """
  __slots__ = ("array", "index")

  def __init__(self, array, index):
    self.array = array
    self.index = index

  def __repr__(self):
    return f"_IndexUpdateRef({repr(self.array)}, {repr(self.index)})"

  def set(self, values, indices_are_sorted=False, unique_indices=False):
    """Pure equivalent of ``x[idx] = y``.

    ``x.at[idx].set(y)`` is syntactic sugar for
    ``jax.ops.index_update(x, jax.ops.index[idx], y)``, and
    returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] = y``.

    See :mod:`jax.ops` for details.
    """
    return ops.index_update(self.array, self.index, values,
                            indices_are_sorted=indices_are_sorted,
                            unique_indices=unique_indices)

  def add(self, values, indices_are_sorted=False, unique_indices=False):
    """Pure equivalent of ``x[idx] += y``.

    ``x.at[idx].add(y)`` is syntactic sugar for
    ``jax.ops.index_add(x, jax.ops.index[idx], y)``, and
    returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] += y``.

    See :mod:`jax.ops` for details.
    """
    return ops.index_add(self.array, self.index, values,
                         indices_are_sorted=indices_are_sorted,
                         unique_indices=unique_indices)

  def mul(self, values, indices_are_sorted=False, unique_indices=False):
    """Pure equivalent of ``x[idx] += y``.

    ``x.at[idx].mul(y)`` is syntactic sugar for
    ``jax.ops.index_mul(x, jax.ops.index[idx], y)``, and
    returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] *= y``.

    See :mod:`jax.ops` for details.
    """
    return ops.index_mul(self.array, self.index, values,
                         indices_are_sorted=indices_are_sorted,
                         unique_indices=unique_indices)

  def min(self, values, indices_are_sorted=False, unique_indices=False):
    """Pure equivalent of ``x[idx] = minimum(x[idx], y)``.

    ``x.at[idx].min(y)`` is syntactic sugar for
    ``jax.ops.index_min(x, jax.ops.index[idx], y)``, and
    returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>`
    ``x[idx] = minimum(x[idx], y)``.

    See :mod:`jax.ops` for details.
    """
    return ops.index_min(self.array, self.index, values,
                         indices_are_sorted=indices_are_sorted,
                         unique_indices=unique_indices)

  def max(self, values, indices_are_sorted=False, unique_indices=False):
    """Pure equivalent of ``x[idx] = maximum(x[idx], y)``.

    ``x.at[idx].max(y)`` is syntactic sugar for
    ``jax.ops.index_max(x, jax.ops.index[idx], y)``, and
    returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>`
    ``x[idx] = maximum(x[idx], y)``.

    See :mod:`jax.ops` for details.
    """
    return ops.index_max(self.array, self.index, values,
                         indices_are_sorted=indices_are_sorted,
                         unique_indices=unique_indices)

setattr(DeviceArray, "at", property(_IndexUpdateHelper))
setattr(ShapedArray, "at", core.aval_property(_IndexUpdateHelper))
