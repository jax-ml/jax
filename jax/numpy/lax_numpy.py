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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import re
import string
import warnings
import types

import numpy as onp
import opt_einsum
import six
from six.moves import builtins, xrange

from jax import jit
from .. import core
from ..abstract_arrays import UnshapedArray, ShapedArray, ConcreteArray
from ..interpreters.xla import DeviceArray
from .. import lax
from ..util import memoize, partial, get_module_functions, unzip2, prod as _prod
from ..lib import xla_bridge
from ..lib.xla_bridge import xla_client

if six.PY3:
  def removechars(s, chars):
    return s.translate(str.maketrans(dict.fromkeys(chars)))
else:
  def removechars(s, chars):
    return s.translate(None, ''.join(chars))

newaxis = None

# We replace some builtin names to follow Numpy's API, so we capture here.
_abs = builtins.abs
_all = builtins.all
_any = builtins.any
_max = builtins.max
_min = builtins.min
_sum = builtins.sum

# We need some numpy scalars
pi = onp.pi
e = onp.e
inf = onp.inf
NINF = onp.NINF
nan = onp.nan

# And some numpy utility functions
set_printoptions = onp.set_printoptions

# We want isinstance(x, np.ndarray) checks in user code to work with the our
# array-like types, including DeviceArray and UnshapedArray (i.e. the abstract
# array base class). We can override the isinstance behavior directly, without
# having the complexity of multiple inheritance on those classes, by defining
# the ndarray class to have a metaclass with special __instancecheck__ behavior.
_arraylike_types = (onp.ndarray, UnshapedArray, DeviceArray)

class _ArrayMeta(type(onp.ndarray)):
  """Metaclass for overriding ndarray isinstance checks."""

  def __instancecheck__(self, instance):
    try:
      return isinstance(instance.aval, _arraylike_types)
    except AttributeError:
      return isinstance(instance, _arraylike_types)

# pylint: disable=invalid-name
class ndarray(six.with_metaclass(_ArrayMeta, onp.ndarray)):
  pass
# pylint: enable=invalid-name


isscalar = onp.isscalar
iscomplexobj = onp.iscomplexobj
result_type = onp.result_type
shape = _shape = onp.shape
ndim = _ndim = onp.ndim
size = onp.size
_dtype = lax.dtype

bool_ = onp.bool_
uint8 = onp.uint8
uint16 = onp.uint16
uint32 = onp.uint32
uint64 = onp.uint64
int8 = onp.int8
int16 = onp.int16
int32 = onp.int32
int64 = onp.int64
float16 = onp.float16
float32 = single = onp.float32
float64 = double = onp.float64
complex64 = csingle = onp.complex64
complex128 = cdouble = onp.complex128

flexible = onp.flexible
character = onp.character
object_ = onp.object_
number = onp.number
inexact = onp.inexact
complexfloating = onp.complexfloating
floating = onp.floating
integer = onp.integer
signedinteger = onp.signedinteger
unsignedinteger = onp.unsignedinteger

iinfo = onp.iinfo
finfo = onp.finfo

issubdtype = onp.issubdtype
issubsctype = onp.issubsctype

ComplexWarning = onp.ComplexWarning

array_str = onp.array_str
array_repr = onp.array_repr

### utility functions


def _promote_shapes(*args):
  """Prepend implicit leading singleton dimensions for Numpy broadcasting."""
  if len(args) < 2:
    return args
  else:
    shapes = [shape(arg) for arg in args]
    nd = len(lax.broadcast_shapes(*shapes))
    return [lax.reshape(arg, (1,) * (nd - len(shp)) + shp)
            if len(shp) != nd else arg for arg, shp in zip(args, shapes)]

def _promote_dtypes(*args):
  """Convenience function to apply Numpy argument dtype promotion."""
  # TODO(dougalm,mattjj): This is a performance bottleneck. Consider memoizing.
  if len(args) < 2:
    return args
  else:
    from_dtypes = (x if type(x) in _builtin_numeric_types else _dtype(x)
                   for x in args)
    to_dtype = xla_bridge.canonicalize_dtype(result_type(*from_dtypes))
    return [lax.convert_element_type(x, to_dtype)
            if _dtype(x) != to_dtype else x for x in args]

if six.PY3:
  _builtin_numeric_types = (int, float, complex)
else:
  _builtin_numeric_types = (int, float, long, complex)

def _promote_to_result_dtype(op, *args):
  """Convenience function to promote args directly to the op's result dtype."""
  to_dtype = _result_dtype(op, *args)
  return [lax.convert_element_type(arg, to_dtype) for arg in args]


def _result_dtype(op, *args):
  """Compute result dtype of applying op to arguments with given dtypes."""
  args = (onp.ones((0,) * ndim(arg), _dtype(arg)) for arg in args)
  return _dtype(op(*args))


def _check_arraylike(fun_name, *args):
  """Check if all args fit JAX's definition of arraylike (ndarray or scalar)."""
  not_array = lambda x: not isinstance(x, ndarray) and not onp.isscalar(x)
  if _any(not_array(arg) for arg in args):
    pos, arg = next((i, arg) for i, arg in enumerate(args) if not_array(arg))
    msg = "{} requires ndarray or scalar arguments, got {} at position {}."
    raise TypeError(msg.format(fun_name, type(arg), pos))


def _promote_args(fun_name, *args):
  """Convenience function to apply Numpy argument shape and dtype promotion."""
  _check_arraylike(fun_name, *args)
  return _promote_shapes(*_promote_dtypes(*args))


def _promote_args_like(op, *args):
  """Convenience function to apply shape and dtype promotion to result type."""
  _check_arraylike(op.__name__, *args)
  return _promote_shapes(*_promote_to_result_dtype(op, *args))


def _constant_like(x, const):
  return onp.array(const, dtype=_dtype(x))

_numpy_signature_re = re.compile(r'^([\w., ]+=)?\s*[\w\.]+\(.*\)$')

def _wraps(fun):
  """Like functools.wraps but works with numpy.ufuncs."""
  def wrap(op):
    try:
      # Numpy doc comments have the form:
      # fn(x, y, z)          (optional)
      #
      # A one-line summary
      #
      # ... everything else ...
      # We (a) move the summary to the top, since it is what the Sphinx
      # autosummary extension expects, and (b) add a comment below the summary
      # to the effect that this is a LAX wrapper of a Numpy function.
      sections = fun.__doc__.split("\n\n")

      signatures = []
      summary = None
      for i in xrange(len(sections)):
        if _numpy_signature_re.match(sections[i]):
          signatures.append(sections[i])
        else:
          summary = sections[i].strip()
          break
      body = "\n\n".join(signatures + sections[i + 1:])
      docstr = (
        "{summary}\n\nLAX-backend implementation of :func:`{fun}`. "
        "Original docstring below.\n\n{body}".format(
          summary=summary, fun=fun.__name__, body=body))
      op.__name__ = fun.__name__
      op.__doc__ = docstr
    finally:
      return op
  return wrap


### implementations of numpy functions in terms of lax


def _one_to_one_unop(numpy_fn, lax_fn, promote_like=False):
  if promote_like:
    fn = lambda x: lax_fn(lax.convert_element_type(x, _result_dtype(numpy_fn, x)))
  else:
    fn = lambda x: lax_fn(x)
  return _wraps(numpy_fn)(fn)

def _one_to_one_binop(numpy_fn, lax_fn, promote_like=False):
  if promote_like:
    fn = lambda x, y: lax_fn(*_promote_args_like(numpy_fn, x, y))
  else:
    fn = lambda x, y: lax_fn(*_promote_args(numpy_fn.__name__, x, y))
  return _wraps(numpy_fn)(fn)

absolute = abs = _one_to_one_unop(onp.absolute, lax.abs)
fabs = _one_to_one_unop(onp.fabs, lax.abs, True)
bitwise_not = _one_to_one_unop(onp.bitwise_not, lax.bitwise_not)
negative = _one_to_one_unop(onp.negative, lax.neg)
positive = _one_to_one_unop(onp.positive, lambda x: x)
sign = _one_to_one_unop(onp.sign, lax.sign)

floor = _one_to_one_unop(onp.floor, lax.floor, True)
ceil = _one_to_one_unop(onp.ceil, lax.ceil, True)
exp = _one_to_one_unop(onp.exp, lax.exp, True)
log = _one_to_one_unop(onp.log, lax.log, True)
expm1 = _one_to_one_unop(onp.expm1, lax.expm1, True)
log1p = _one_to_one_unop(onp.log1p, lax.log1p, True)
sin = _one_to_one_unop(onp.sin, lax.sin, True)
cos = _one_to_one_unop(onp.cos, lax.cos, True)
tan = _one_to_one_unop(onp.tan, lax.tan, True)
arcsin = _one_to_one_unop(onp.arcsin, lax.asin, True)
arccos = _one_to_one_unop(onp.arccos, lax.acos, True)
arctan = _one_to_one_unop(onp.arctan, lax.atan, True)
sinh = _one_to_one_unop(onp.sinh, lax.sinh, True)
cosh = _one_to_one_unop(onp.cosh, lax.cosh, True)
tanh = _one_to_one_unop(onp.tanh, lax.tanh, True)
arcsinh = _one_to_one_unop(onp.arcsinh, lax.asinh, True)
arccosh = _one_to_one_unop(onp.arccosh, lax.acosh, True)
arctanh = _one_to_one_unop(onp.arctanh, lax.atanh, True)


add = _one_to_one_binop(onp.add, lax.add)
bitwise_and = _one_to_one_binop(onp.bitwise_and, lax.bitwise_and)
bitwise_or = _one_to_one_binop(onp.bitwise_or, lax.bitwise_or)
bitwise_xor = _one_to_one_binop(onp.bitwise_xor, lax.bitwise_xor)
right_shift = _one_to_one_binop(onp.right_shift, lax.shift_right_arithmetic)
left_shift = _one_to_one_binop(onp.left_shift, lax.shift_left)
equal = _one_to_one_binop(onp.equal, lax.eq)
multiply = _one_to_one_binop(onp.multiply, lax.mul)
not_equal = _one_to_one_binop(onp.not_equal, lax.ne)
subtract = _one_to_one_binop(onp.subtract, lax.sub)
arctan2 = _one_to_one_binop(onp.arctan2, lax.atan2, True)
minimum = _one_to_one_binop(onp.minimum, lax.min)
maximum = _one_to_one_binop(onp.maximum, lax.max)
float_power = _one_to_one_binop(onp.float_power, lax.pow, True)


def _comparison_op(numpy_fn, lax_fn):
  def fn(x, y):
    x, y =  _promote_args(numpy_fn.__name__, x, y)
    # Comparison on complex types are defined as a lexicographic ordering on
    # the (real, imag) pair.
    if issubdtype(_dtype(x), complexfloating):
      rx = lax.real(x)
      ry = lax.real(y)
      return lax.select(lax.eq(rx, ry), lax_fn(lax.imag(x), lax.imag(y)),
                        lax_fn(rx, ry))
    return lax_fn(x, y)
  return _wraps(numpy_fn)(fn)

greater_equal = _comparison_op(onp.greater_equal, lax.ge)
greater = _comparison_op(onp.greater, lax.gt)
less_equal = _comparison_op(onp.less_equal, lax.le)
less = _comparison_op(onp.less, lax.lt)


def _logical_op(np_op, bitwise_op):
  @_wraps(np_op)
  def op(*args):
    zero = lambda x: lax.full_like(x, shape=(), fill_value=0)
    args = (x if onp.issubdtype(_dtype(x), onp.bool_) else lax.ne(x, zero(x))
            for x in args)
    return bitwise_op(*_promote_args(np_op.__name__, *args))
  return op

logical_and = _logical_op(onp.logical_and, lax.bitwise_and)
logical_not = _logical_op(onp.logical_not, lax.bitwise_not)
logical_or = _logical_op(onp.logical_or, lax.bitwise_or)
logical_xor = _logical_op(onp.logical_xor, lax.bitwise_xor)


@_wraps(onp.true_divide)
def true_divide(x1, x2):
  result_dtype = _result_dtype(onp.true_divide, x1, x2)
  x1, x2 = _promote_shapes(x1, x2)
  return lax.div(lax.convert_element_type(x1, result_dtype),
                 lax.convert_element_type(x2, result_dtype))


@_wraps(onp.divide)
def divide(x1, x2):
  # decide whether to perform integer division based on Numpy result dtype, as a
  # way to check whether Python 3 style division is active in Numpy
  result_dtype = _result_dtype(onp.divide, x1, x2)
  if onp.issubdtype(result_dtype, onp.integer):
    return floor_divide(x1, x2)
  else:
    return true_divide(x1, x2)


@_wraps(onp.floor_divide)
def floor_divide(x1, x2):
  x1, x2 = _promote_args("floor_divide", x1, x2)
  dtype = _dtype(x1)
  if issubdtype(dtype, integer):
    quotient = lax.div(x1, x2)
    select = logical_and(lax.sign(x1) != lax.sign(x2), lax.rem(x1, x2) != 0)
    # TODO(mattjj): investigate why subtracting a scalar was causing promotion
    return where(select, quotient - onp.array(1, _dtype(quotient)), quotient)
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


@_wraps(onp.divmod)
def divmod(x1, x2):
  x1, x2 = _promote_args("divmod", x1, x2)
  if onp.issubdtype(_dtype(x1), onp.integer):
    return floor_divide(x1, x2), remainder(x1, x2)
  else:
    return _float_divmod(x1, x2)


def _float_divmod(x1, x2):
  # see float_divmod in floatobject.c of CPython
  mod = lax.rem(x1, x2)
  div = lax.div(lax.sub(x1, mod), x2)

  ind = lax.bitwise_and(mod != 0, lax.sign(x2) != lax.sign(mod))
  mod = lax.select(ind, mod + x1, mod)
  div = lax.select(ind, div - _constant_like(div, 1), div)

  return lax.round(div), mod


@_wraps(onp.power)
def power(x1, x2):
  x1 = asarray(x1)
  x2 = asarray(x2)
  x1, x2 = _promote_args_like(onp.power, x1, x2)
  dtype = _dtype(x1)
  if not issubdtype(dtype, integer):
    return lax.pow(x1, x2)

  # Integer power => use binary exponentiation.

  # TODO(phawkins): add integer pow support to XLA.
  bits = 6  # Anything more would overflow for any x1 > 1
  acc = ones(shape(x1), dtype=dtype)
  for _ in xrange(bits):
    acc = where(lax.bitwise_and(x2, _constant_like(x2, 1)),
                lax.mul(acc, x1), acc)
    x1 = lax.mul(x1, x1)
    x2 = lax.shift_right_logical(x2, _constant_like(x2, 1))
  return acc


@_wraps(onp.logaddexp)
def logaddexp(x1, x2):
  x1, x2 = _promote_shapes(*_promote_to_result_dtype(onp.logaddexp, x1, x2))
  amax = lax.max(x1, x2)
  return lax.add(amax, lax.log(lax.add(lax.exp(lax.sub(x1, amax)),
                                       lax.exp(lax.sub(x2, amax)))))


@_wraps(onp.logaddexp2)
def logaddexp2(x1, x2):
  x1, x2 = _promote_shapes(*_promote_to_result_dtype(onp.logaddexp2, x1, x2))
  amax = lax.max(x1, x2)
  return lax.add(amax, log2(lax.add(exp2(lax.sub(x1, amax)),
                                    exp2(lax.sub(x2, amax)))))


@_wraps(onp.log2)
def log2(x):
  x, = _promote_to_result_dtype(onp.log2, x)
  return lax.div(lax.log(x), lax.log(_constant_like(x, 2)))


@_wraps(onp.log10)
def log10(x):
  x, = _promote_to_result_dtype(onp.log10, x)
  return lax.div(lax.log(x), lax.log(_constant_like(x, 10)))


@_wraps(onp.exp2)
def exp2(x):
  x, = _promote_to_result_dtype(onp.exp2, x)
  return lax.exp(lax.mul(lax.log(_constant_like(x, 2)), x))


@_wraps(onp.remainder)
def remainder(x1, x2):
  x1, x2 = _promote_args("remainder", x1, x2)
  return lax.rem(lax.add(lax.rem(x1, x2), x2), x2)
mod = remainder
fmod = _wraps(onp.fmod)(lambda x, y: lax.rem(x, y))


@_wraps(onp.cbrt)
def cbrt(x):
  x, = _promote_to_result_dtype(onp.cbrt, x)
  return lax.sign(x) * power(lax.abs(x), _constant_like(x, 1. / 3.))


@_wraps(onp.sqrt)
def sqrt(x):
  x, = _promote_to_result_dtype(onp.sqrt, x)
  return power(x, _constant_like(x, 0.5))


@_wraps(onp.square)
def square(x):
  x, = _promote_to_result_dtype(onp.square, x)
  return x * x


@_wraps(onp.deg2rad)
def deg2rad(x):
  x, = _promote_to_result_dtype(onp.deg2rad, x)
  return lax.mul(x, lax._const(x, pi / 180))


@_wraps(onp.rad2deg)
def rad2deg(x):
  x, = _promote_to_result_dtype(onp.rad2deg, x)
  return lax.mul(x, lax._const(x, 180 / pi))


degrees = rad2deg
radians = deg2rad


@_wraps(onp.heaviside)
def heaviside(x, y):
  x, y = _promote_to_result_dtype(onp.heaviside, x, y)
  zero = lax._const(x, 0)
  return where(lax.lt(x, zero), zero,
               where(lax.gt(x, zero), lax._const(x, 1), y))


@_wraps(onp.hypot)
def hypot(x, y):
  x, y = _promote_to_result_dtype(onp.hypot, x, y)
  return lax.sqrt(x*x + y*y)


@_wraps(onp.reciprocal)
def reciprocal(x):
  x, = _promote_to_result_dtype(onp.reciprocal, x)
  return lax.div(lax._const(x, 1), x)


@_wraps(onp.sinc)
def sinc(x):
  x, = _promote_to_result_dtype(onp.sinc, x)
  pi_x = lax.mul(lax._const(x, pi), x)
  return where(lax.eq(x, lax._const(x, 0)),
               lax._const(x, 1), lax.div(lax.sin(pi_x), pi_x))


@_wraps(onp.transpose)
def transpose(x, axis=None):
  axis = onp.arange(ndim(x))[::-1] if axis is None else axis
  return lax.transpose(x, axis)


@_wraps(onp.rot90)
def rot90(m, k=1, axes=(0, 1)):
  ax1, ax2 = axes
  if ax1 % m.ndim == ax2 % m.ndim:
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


@_wraps(onp.flip)
def flip(m, axis):
  # Negative axes wrap around
  if axis < 0:
    rank = len(m.shape)
    assert axis >= -rank, "axis={} is invalid for the {}-dimensional input array".format(axis, rank)
    return lax.rev(m, [axis % rank])
  else:
    return lax.rev(m, [axis])


@_wraps(onp.fliplr)
def fliplr(m):
  return flip(m, 1)


@_wraps(onp.flipud)
def flipud(m):
  return flip(m, 0)


@_wraps(onp.conjugate)
def conjugate(x):
  return lax.conj(x) if iscomplexobj(x) else x
conj = conjugate


@_wraps(onp.imag)
def imag(x):
  return lax.imag(x) if iscomplexobj(x) else zeros_like(x)


@_wraps(onp.real)
def real(x):
  return lax.real(x) if iscomplexobj(x) else x


@_wraps(onp.iscomplex)
def iscomplex(x):
  i = imag(x)
  return lax.ne(i, lax._const(i, 0))

@_wraps(onp.isreal)
def isreal(x):
  i = imag(x)
  return lax.eq(i, lax._const(i, 0))

@_wraps(onp.angle)
def angle(x):
  re = real(x)
  im = imag(x)
  dtype = _dtype(re)
  if not issubdtype(dtype, inexact) or (
      issubdtype(_dtype(x), floating) and ndim(x) == 0):
    dtype = xla_bridge.canonicalize_dtype(float64)
    re = lax.convert_element_type(re, dtype)
    im = lax.convert_element_type(im, dtype)
  return lax.atan2(im, re)


@_wraps(onp.diff)
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

  op = not_equal if a.dtype == onp.bool_ else subtract
  for _ in range(n):
    a = op(a[slice1], a[slice2])

  return a


@_wraps(onp.isrealobj)
def isrealobj(a):
  return not iscomplexobj(a)


@_wraps(onp.reshape)
def reshape(a, newshape, order="C"):
  try:
    return a.reshape(newshape, order=order)
  except AttributeError:
    return _reshape(a, newshape, order=order)

def _reshape(a, newshape, order="C"):
  dummy_val = onp.broadcast_to(0, shape(a))  # zero strides
  computed_newshape = onp.reshape(dummy_val, newshape).shape

  if order == "C":
    return lax.reshape(a, computed_newshape, None)
  elif order == "F":
    dims = onp.arange(ndim(a))[::-1]
    return lax.reshape(a, computed_newshape[::-1], dims).T
  elif order == "A":
    raise NotImplementedError("np.reshape order=A is not implemented.")
  else:
    raise ValueError("Unexpected value for 'order' argument: {}.".format(order))


@_wraps(onp.ravel)
def ravel(a, order="C"):
  if order == "K":
    raise NotImplementedError("Ravel not implemented for order='K'.")
  return reshape(a, (size(a),), order)


@_wraps(onp.squeeze)
def squeeze(a, axis=None):
  if 1 not in shape(a):
    return a
  if axis is None:
    newshape = [d for d in shape(a) if d != 1]
  else:
    axis = frozenset(onp.mod(axis, ndim(a)).reshape(-1))
    newshape = [d for i, d in enumerate(shape(a))
                if d != 1 or i not in axis]
  return lax.reshape(a, newshape)


@_wraps(onp.expand_dims)
def expand_dims(a, axis):
  shape = _shape(a)
  axis = axis % (ndim(a) + 1)  # pylint: disable=g-no-augmented-assignment
  return lax.reshape(a, shape[:axis] + (1,) + shape[axis:])


@_wraps(onp.swapaxes)
def swapaxes(a, axis1, axis2):
  perm = onp.arange(ndim(a))
  perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
  return lax.transpose(a, perm)


@_wraps(onp.moveaxis)
def moveaxis(a, source, destination):
  source = onp.mod(source, ndim(a)).reshape(-1)
  destination = onp.mod(destination, ndim(a)).reshape(-1)
  if len(source) != len(destination):
    raise ValueError("Inconsistent number of elements: {} vs {}"
                     .format(len(source), len(destination)))
  perm = [i for i in range(ndim(a)) if i not in source]
  for dest, src in sorted(zip(destination, source)):
    perm.insert(dest, src)
  return lax.transpose(a, perm)


@_wraps(onp.isclose)
def isclose(a, b, rtol=1e-05, atol=1e-08):
  a, b = _promote_args("isclose", asarray(a), asarray(b))
  dtype = _dtype(a)
  if issubdtype(dtype, inexact):
    if issubdtype(dtype, complexfloating):
      dtype = _result_dtype(real, a)
    rtol = lax.convert_element_type(rtol, dtype)
    atol = lax.convert_element_type(atol, dtype)
    out = lax.le(
      lax.abs(lax.sub(a, b)),
      lax.add(atol, lax.mul(rtol, lax.abs(b))))
    return _maybe_numpy_1_13_isclose_behavior(a, out)
  else:
    return lax.eq(a, b)

numpy_version = tuple(map(int, onp.version.version.split('.')))
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


@_wraps(onp.where)
def where(condition, x=None, y=None):
  if x is None or y is None:
    raise ValueError("Must use the three-argument form of where().")
  if not onp.issubdtype(_dtype(condition), onp.bool_):
    condition = lax.ne(condition, zeros_like(condition))
  condition, x, y = broadcast_arrays(condition, x, y)
  if not onp.size(x):
    empty, _ = _promote_dtypes(x, y)
    return empty
  else:
    return lax.select(condition, *_promote_dtypes(x, y))


def broadcast_arrays(*args):
  """Like Numpy's broadcast_arrays but doesn't return views."""
  shapes = [shape(arg) for arg in args]
  if len(set(shapes)) == 1:
    return [arg if isinstance(arg, ndarray) or isscalar(arg) else array(arg)
            for arg in args]
  result_shape = lax.broadcast_shapes(*shapes)
  return [broadcast_to(arg, result_shape) for arg in args]


def broadcast_to(arr, shape):
  """Like Numpy's broadcast_to but doesn't necessarily return views."""
  arr = arr if isinstance(arr, ndarray) or isscalar(arr) else array(arr)
  if _shape(arr) != shape:
    # TODO(mattjj): revise this to call lax.broadcast_in_dim rather than
    # lax.broadcast and lax.transpose
    lax.broadcast_shapes(shape, _shape(arr))  # error checking
    nlead = len(shape) - len(_shape(arr))
    diff, = onp.where(onp.not_equal(shape[nlead:], _shape(arr)))

    new_dims = tuple(range(nlead)) + tuple(nlead + diff)
    kept_dims = tuple(onp.delete(onp.arange(len(shape)), new_dims))
    perm = onp.argsort(new_dims + kept_dims)

    broadcast_dims = onp.take(shape, new_dims)
    squeezed_array = squeeze(arr, diff)
    return lax.transpose(lax.broadcast(squeezed_array, broadcast_dims), perm)
  else:
    return arr


@_wraps(onp.split)
def split(ary, indices_or_sections, axis=0):
  dummy_val = onp.broadcast_to(0, ary.shape)  # zero strides
  subarrays = onp.split(dummy_val, indices_or_sections, axis)  # shapes
  split_indices = onp.cumsum([0] + [onp.shape(sub)[axis] for sub in subarrays])
  starts, ends = [0] * ndim(ary), shape(ary)
  _subval = lambda x, i, v: lax.subvals(x, [(i, v)])
  return [lax.slice(ary, _subval(starts, axis, start), _subval(ends, axis, end))
          for start, end in zip(split_indices[:-1], split_indices[1:])]

def _split_on_axis(onp_fun, axis):
  @_wraps(onp_fun)
  def f(ary, indices_or_sections):
    return split(ary, indices_or_sections, axis=axis)
  return f

vsplit = _split_on_axis(onp.vsplit, axis=0)
hsplit = _split_on_axis(onp.hsplit, axis=1)
dsplit = _split_on_axis(onp.dsplit, axis=2)


@_wraps(onp.clip)
def clip(a, a_min=None, a_max=None):
  if a_min is None and a_max is None:
    raise "At most one of a_min and a_max may be None"
  if a_min is not None:
    if _dtype(a_min) != _dtype(a):
      a_min = lax.convert_element_type(a_min, _dtype(a))
    a = lax.max(a_min, a)
  if a_max is not None:
    if _dtype(a_max) != _dtype(a):
      a_max = lax.convert_element_type(a_max, _dtype(a))
    a = lax.min(a_max, a)
  return a


def _dtype_info(dtype):
  """Helper function for to get dtype info needed for clipping."""
  if onp.issubdtype(dtype, onp.integer):
    return onp.iinfo(dtype)
  return onp.finfo(dtype)


@_wraps(onp.round)
def round(a, decimals=0):
  dtype = _dtype(a)
  if issubdtype(dtype, integer):
    if decimals < 0:
      raise NotImplementedError(
        "integer np.round not implemented for decimals < 0")
    return a  # no-op on integer types

  def _round_float(x):
    if decimals == 0:
      return lax.round(x)

    factor = _constant_like(x, 10 ** decimals)
    return lax.div(lax.round(lax.mul(x, factor)), factor)

  if issubdtype(dtype, complexfloating):
    return lax.complex(_round_float(lax.real(a)), _round_float(lax.imag(a)))
  else:
    return _round_float(a)
around = round


@_wraps(onp.fix)
def fix(x, out=None):
  if out is not None:
    raise ValueError("fix does not support the `out` argument.")
  zero = lax._const(x, 0)
  return where(lax.ge(x, zero), lax.floor(x), lax.ceil(x))

@_wraps(onp.isfinite)
def isfinite(x):
  dtype = _dtype(x)
  if issubdtype(dtype, floating):
    return lax.is_finite(x)
  elif issubdtype(dtype, complexfloating):
    return lax.bitwise_and(lax.is_finite(real(x)), lax.is_finite(imag(x)))
  else:
    return full_like(x, True, dtype=bool_)

@_wraps(onp.isinf)
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

isposinf = _wraps(onp.isposinf)(partial(_isposneginf, inf))
isneginf = _wraps(onp.isneginf)(partial(_isposneginf, -inf))

@_wraps(onp.isnan)
def isnan(x):
  return lax.bitwise_and(lax.bitwise_not(isfinite(x)),
                         lax.bitwise_not(isinf(x)))

@_wraps(onp.nan_to_num)
def nan_to_num(x, copy=True):
  del copy
  dtype = _dtype(x)
  if issubdtype(dtype, complexfloating):
    return lax.complex(nan_to_num(lax.real(x)), nan_to_num(lax.imag(x)))
  info = finfo(xla_bridge.canonicalize_dtype(dtype))
  x = where(isnan(x), _constant_like(x, 0), x)
  x = where(isposinf(x), _constant_like(x, info.max), x)
  x = where(isneginf(x), _constant_like(x, info.min), x)
  return x

### Reducers


def _make_reduction(np_fun, op, init_val, preproc=None):
  """Creates reduction function given a binary operation and monoid identity."""

  @_wraps(np_fun)
  def reduction(a, axis=None, dtype=None, out=None, keepdims=False):
    if out is not None:
      raise ValueError("reduction does not support the `out` argument.")

    a = a if isinstance(a, ndarray) else asarray(a)
    a = preproc(a) if preproc else a
    dims = _reduction_dims(a, axis)
    result_dtype = _dtype(np_fun(onp.ones((), dtype=dtype or _dtype(a))))
    if _dtype(a) != result_dtype:
      a = lax.convert_element_type(a, result_dtype)
    result = lax.reduce(a, _reduction_init_val(a, init_val), op, dims)
    if keepdims:
      shape_with_singletons = lax.subvals(shape(a), zip(dims, (1,) * len(dims)))
      result = lax.reshape(result, shape_with_singletons)
    if dtype and onp.dtype(dtype) != onp.dtype(result_dtype):
      result = lax.convert_element_type(result, dtype)
    return result

  return reduction

def _reduction_dims(a, axis):
  if axis is None:
    return onp.arange(ndim(a))
  elif isinstance(axis, (onp.ndarray, tuple, list)):
    return onp.mod(onp.asarray(axis), ndim(a))
  elif isinstance(axis, int):
    return onp.mod([axis], ndim(a))
  else:
    raise TypeError("Unexpected type of axis argument: {}".format(type(axis)))

def _reduction_init_val(a, init_val):
  a_dtype = xla_bridge.canonicalize_dtype(_dtype(a))
  try:
    return onp.array(init_val, dtype=a_dtype)
  except OverflowError:
    assert onp.issubdtype(a_dtype, onp.integer)
    sign, iinfo = onp.sign(init_val), onp.iinfo(a_dtype)
    return onp.array(iinfo.min if sign < 0 else iinfo.max, dtype=a_dtype)

_cast_to_bool = partial(lax.convert_element_type, new_dtype=onp.bool_)

sum = _make_reduction(onp.sum, lax.add, 0)
product = prod = _make_reduction(onp.prod, lax.mul, 1)
amax = max = _make_reduction(onp.max, lax.max, -onp.inf)
amin = min = _make_reduction(onp.min, lax.min, onp.inf)
all = alltrue = _make_reduction(onp.all, lax.bitwise_and, True, _cast_to_bool)
any = sometrue = _make_reduction(onp.any, lax.bitwise_or, False, _cast_to_bool)


@_wraps(onp.mean)
def mean(a, axis=None, dtype=None, out=None, keepdims=False):
  if out is not None:
    raise ValueError("mean does not support the `out` argument.")

  if axis is None:
    normalizer = size(a)
  else:
    normalizer = onp.prod(onp.take(shape(a), axis))
  if dtype is None:
    if (onp.issubdtype(_dtype(a), onp.bool_) or
        onp.issubdtype(_dtype(a), onp.integer)):
      dtype = xla_bridge.canonicalize_dtype(onp.float64)
    else:
      dtype = _dtype(a)

  return lax.div(
      sum(a, axis, dtype=dtype, keepdims=keepdims),
      lax.convert_element_type(normalizer, dtype))

@_wraps(onp.average)
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

        if issubdtype(a.dtype, integer) or issubdtype(a.dtype, bool_):
            out_dtype = xla_bridge.canonicalize_dtype(result_type(a.dtype,
                                                                  weights.dtype,
                                                                  floating))
        else:
            out_dtype = xla_bridge.canonicalize_dtype(result_type(a.dtype, weights.dtype))

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


@_wraps(onp.var)
def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
  if out is not None:
    raise ValueError("var does not support the `out` argument.")

  if ddof != 0:
    raise NotImplementedError("Only implemented for ddof=0.")
  if dtype is None:
    if (onp.issubdtype(_dtype(a), onp.bool_) or
        onp.issubdtype(_dtype(a), onp.integer)):
      dtype = xla_bridge.canonicalize_dtype(onp.float64)
  centered = subtract(a, mean(a, axis, dtype=dtype, keepdims=True))
  if iscomplexobj(centered):
    centered = lax.abs(centered)
  return mean(lax.mul(centered, centered), axis, dtype=dtype, keepdims=keepdims)


@_wraps(onp.std)
def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
  if out is not None:
    raise ValueError("std does not support the `out` argument.")
  return sqrt(var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims))


@_wraps(onp.ptp)
def ptp(a, axis=None, out=None, keepdims=False):
  if out is not None:
    raise ValueError("ptp does not support the `out` argument.")
  x = amax(a, axis=axis, keepdims=keepdims)
  y = amin(a, axis=axis, keepdims=keepdims)
  return lax.sub(x, y)


@_wraps(onp.allclose)
def allclose(a, b, rtol=1e-05, atol=1e-08):
  return all(isclose(a, b, rtol, atol))


@_wraps(onp.count_nonzero)
def count_nonzero(a, axis=None):
  return sum(lax.ne(a, _constant_like(a, 0)), axis=axis,
             dtype=xla_bridge.canonicalize_dtype(onp.int_))


def _make_nan_reduction(onp_reduction, np_reduction, init_val, nan_if_all_nan):
  @_wraps(onp_reduction)
  def nan_reduction(a, axis=None, out=None, keepdims=False, **kwargs):
    out = np_reduction(where(isnan(a), _reduction_init_val(a, init_val), a),
                       axis=axis, out=out, keepdims=keepdims, **kwargs)
    if nan_if_all_nan:
      return where(all(isnan(a), axis=axis, keepdims=keepdims),
                   _constant_like(a, nan), out)
    else:
      return out

  return nan_reduction

nanmin = _make_nan_reduction(onp.nanmin, min, inf, nan_if_all_nan=True)
nanmax = _make_nan_reduction(onp.nanmax, max, -inf, nan_if_all_nan=True)
nansum = _make_nan_reduction(onp.nansum, sum, 0, nan_if_all_nan=False)
nanprod = _make_nan_reduction(onp.nanprod, prod, 1, nan_if_all_nan=False)


def _make_cumulative_reduction(onp_reduction, window_reduce, init_val,
                               squash_nan=False):
  @_wraps(onp_reduction)
  def cumulative_reduction(a, axis=None, dtype=None):
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

    if squash_nan:
      a = where(isnan(a), _constant_like(a, init_val), a)

    if dtype:
      a = lax.convert_element_type(a, dtype)

    if a_shape[axis] == 0:
      return a

    padding = [(0, 0, 0)] * num_dims
    padding[axis] = (a_shape[axis] - 1, 0, 0)
    a = lax.pad(a, _constant_like(a, init_val), padding)
    strides = [1] * num_dims
    window_dims = [1] * num_dims
    window_dims[axis] = a_shape[axis]
    return window_reduce(
       a, window_dims, strides, xla_client.PaddingType.VALID)

  return cumulative_reduction


cumsum = _make_cumulative_reduction(
  onp.cumsum, lax._reduce_window_sum, 0, squash_nan=False)
cumprod = _make_cumulative_reduction(
  onp.cumprod, lax._reduce_window_prod, 1, squash_nan=False)
cumproduct = cumprod
nancumsum = _make_cumulative_reduction(
  onp.nancumsum, lax._reduce_window_sum, 0, squash_nan=True)
nancumprod = _make_cumulative_reduction(
  onp.nancumprod, lax._reduce_window_prod, 1, squash_nan=True)


### Array-creation functions

# TODO(phawkins): use this helper everywhere.
def _canonicalize_axis(axis, num_dims):
  """Canonicalize an axis in (-num_dims, num_dims) to [0, num_dims)."""
  axis = int(axis)
  if axis < 0:
    axis = axis + num_dims
  if axis < 0 or axis >= num_dims:
      raise ValueError(
          "axis {} is out of bounds for array of dimension {}".format(
              axis, num_dims))
  return axis


@_wraps(onp.pad)
def pad(array, pad_width, mode, constant_values=0):
  if mode != "constant":
    msg = "Only the 'constant' case of np.pad is implemented, got mode={}."
    raise NotImplementedError(msg.format(mode))

  array = asarray(array)
  pad_width = onp.broadcast_to(onp.asarray(pad_width), (array.ndim, 2))
  constant_values = broadcast_to(asarray(constant_values), (array.ndim, 2))
  for i in xrange(array.ndim):
    widths = [(0, 0, 0)] * array.ndim
    widths[i] = (pad_width[i, 0], 0, 0)
    array = lax.pad(array, constant_values[i, 0], widths)
    widths[i] = (0, pad_width[i, 1], 0)
    array = lax.pad(array, constant_values[i, 1], widths)
  return array


@_wraps(onp.stack)
def stack(arrays, axis=0):
  if not arrays:
    raise ValueError("Need at least one array to stack.")
  shape0 = shape(arrays[0])
  axis = _canonicalize_axis(axis, len(shape0) + 1)
  new_shape = list(shape0)
  new_shape.insert(axis, 1)
  new_arrays = []
  for a in arrays:
    if shape(a) != shape0:
      raise ValueError("All input arrays must have the same shape.")
    new_arrays.append(reshape(a, new_shape))
  return concatenate(new_arrays, axis=axis)

@_wraps(onp.tile)
def tile(a, reps):
    if isinstance(reps, int):
        reps = (reps,)
    a = a[(None,) * (len(reps) - a.ndim)]
    reps = (1,) * (a.ndim - len(reps)) + reps
    for i, rep in enumerate(reps):
        a = concatenate([a] * rep, axis=i)
    return a

@_wraps(onp.concatenate)
def concatenate(arrays, axis=0):
  if not arrays:
    raise ValueError("Need at least one array to concatenate.")
  if ndim(arrays[0]) == 0:
    raise ValueError("Zero-dimensional arrays cannot be concatenated.")
  return lax.concatenate(_promote_dtypes(*arrays), axis % ndim(arrays[0]))


@_wraps(onp.vstack)
def vstack(tup):
  return concatenate([atleast_2d(m) for m in tup], axis=0)
row_stack = vstack


@_wraps(onp.hstack)
def hstack(tup):
  arrs = [atleast_1d(m) for m in tup]
  if arrs[0].ndim == 1:
    return concatenate(arrs, 0)
  return concatenate(arrs, 1)


@_wraps(onp.dstack)
def dstack(tup):
  return concatenate([atleast_3d(m) for m in tup], axis=2)


@_wraps(onp.column_stack)
def column_stack(tup):
  arrays = []
  for v in tup:
    arr = array(v)
    if arr.ndim < 2:
      arr = arr.reshape((-1, 1))
    arrays.append(arr)
  return concatenate(arrays, 1)


@_wraps(onp.atleast_1d)
def atleast_1d(*arys):
  if len(arys) == 1:
    arr = array(arys[0])
    return arr if ndim(arr) >= 1 else reshape(arr, -1)
  else:
    return [atleast_1d(arr) for arr in arys]


@_wraps(onp.atleast_2d)
def atleast_2d(*arys):
  if len(arys) == 1:
    arr = array(arys[0])
    return arr if ndim(arr) >= 2 else reshape(arr, (1, -1))
  else:
    return [atleast_2d(arr) for arr in arys]


@_wraps(onp.atleast_3d)
def atleast_3d(*arys):
  if len(arys) == 1:
    arr = array(arys[0])
    if ndim(arr) <= 1:
      arr = reshape(arr, (1, -1, 1))
    elif ndim(arr) == 2:
      arr = reshape(arr, shape(arr) + (1,))
    return arr
  else:
    return [atleast_3d(arr) for arr in arys]


@_wraps(onp.array)
def array(object, dtype=None, copy=True, order="K", ndmin=0):
  del copy  # Unused.
  if ndmin != 0 or order != "K":
    raise NotImplementedError("Only implemented for order='K', ndmin=0.")

  if isinstance(object, ndarray):
    if dtype and _dtype(object) != dtype:
      return lax.convert_element_type(object, dtype)
    else:
      return object
  elif hasattr(object, '__array__'):
    # this case is for duck-typed handling of objects that implement `__array__`
    return array(object.__array__(), dtype)
  elif isinstance(object, (list, tuple)):
    if object:
      return stack([array(elt, dtype=dtype) for elt in object])
    else:
      return onp.array([], dtype)
  elif isscalar(object):
    out = lax.reshape(object, ())
    if dtype and _dtype(out) != dtype:
      return lax.convert_element_type(out, dtype)
    else:
      return out
  else:
    raise TypeError("Unexpected input type for array: {}".format(type(object)))
asarray = array


@_wraps(onp.zeros_like)
def zeros_like(x, dtype=None):
  return lax.full_like(x, 0, dtype)


@_wraps(onp.ones_like)
def ones_like(x, dtype=None):
  return lax.full_like(x, 1, dtype)


@_wraps(onp.full)
def full(shape, fill_value, dtype=None):
  return lax.full(shape, fill_value, dtype)


@_wraps(onp.full_like)
def full_like(a, fill_value, dtype=None):
  return lax.full_like(a, fill_value, dtype)


@_wraps(onp.zeros)
def zeros(shape, dtype=onp.dtype("float64")):
  if isinstance(shape, types.GeneratorType):
    raise TypeError("expected sequence object with len >= 0 or a single integer")
  shape = (shape,) if onp.isscalar(shape) else shape
  return lax.full(shape, 0, dtype)

@_wraps(onp.ones)
def ones(shape, dtype=onp.dtype("float64")):
  shape = (shape,) if onp.isscalar(shape) else shape
  return lax.full(shape, 1, dtype)


@_wraps(onp.array_equal)
def array_equal(a1, a2):
  try:
    a1, a2 = asarray(a1), asarray(a2)
  except Exception:
    return False
  if a1.shape != a2.shape:
    return False
  return asarray(a1==a2).all()


# We can't create uninitialized arrays in XLA; use zeros for empty.
empty_like = zeros_like
empty = zeros


@_wraps(onp.eye)
def eye(N, M=None, k=None, dtype=onp.dtype("float64")):
  M = N if M is None else M
  if N < 0 or M < 0:
    msg = "negative dimensions are not allowed, got {} and {}"
    raise ValueError(msg.format(N, M))
  if k is None:
    return lax.broadcasted_eye(dtype, (N, M), (0, 1))
  else:
    k_dtype = _dtype(k)
    if not onp.issubdtype(k_dtype, onp.integer):
      msg = "eye argument `k` must be of integer dtype, got {}"
      raise TypeError(msg.format(k_dtype))
    rows = k + lax.broadcasted_iota(k_dtype, (N, M), 0)
    cols = lax.broadcasted_iota(k_dtype, (N, M), 1)
    return lax.convert_element_type(lax.eq(rows, cols), dtype)


@_wraps(onp.identity)
def identity(n, dtype=None):
  return eye(n, dtype=dtype)


@_wraps(onp.arange)
def arange(*args, **kwargs):
  dtype = kwargs.pop("dtype", None)
  if not args:
    raise TypeError("Required argument 'start' (pos 1) not found")  # same as numpy error

  # If called like np.arange(N), we create a lazy lax._IotaConstant.
  if len(args) == 1 and not kwargs:
    stop, = args
    dtype = dtype or _dtype(stop)
    if onp.issubdtype(dtype, onp.integer):
      return lax.iota(dtype, stop)  # avoids materializing

  # Fall back to instantiating an ndarray in host memory
  return onp.arange(*args, **kwargs)

linspace = onp.linspace
logspace = onp.logspace
geomspace = onp.geomspace
meshgrid = onp.meshgrid

@_wraps(onp.repeat)
def repeat(a, repeats, axis=None):
  if not isscalar(repeats):
    raise NotImplementedError(
        "np.repeat implementation only supports scalar repeats")
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

  # Broadcasts to [..., X, repeats, ...] and reshapes to [..., X * repeats, ...]
  broadcast_shape = list(a_shape)
  broadcast_shape.insert(axis + 1, repeats)
  broadcast_dims = onp.concatenate((onp.arange(0, axis + 1),
                                    onp.arange(axis + 2, num_dims + 1)))
  a_shape[axis] *= repeats
  return lax.reshape(
      lax.broadcast_in_dim(a, broadcast_shape, broadcast_dims),
      a_shape)


@_wraps(onp.tri)
def tri(N, M=None, k=0, dtype=None):
  M = M if M is not None else N
  dtype = dtype or float32
  x = arange(N, dtype=int32)
  y = arange(M, dtype=int32)
  mask = lax.ge(
      (lax.broadcast_in_dim(x, shape=(N, M), broadcast_dimensions=(0,)) +
       int32(k)),
      lax.broadcast(y, [N]))
  return lax.convert_element_type(mask, dtype)


@_wraps(onp.tril)
def tril(m, k=0):
  mask = tri(*shape(m)[-2:], k=k, dtype=bool)
  return where(mask, m, zeros_like(m))


@_wraps(onp.triu)
def triu(m, k=0):
  mask = tri(*shape(m)[-2:], k=k - 1, dtype=bool)
  return where(mask, zeros_like(m), m)


@_wraps(onp.trace)
def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
  if out:
    raise NotImplementedError("The 'out' argument to trace is not supported.")

  a_shape = shape(a)
  if dtype is None:
    dtype = _dtype(a)
    if issubdtype(dtype, integer):
      default_int = xla_bridge.canonicalize_dtype(onp.int_)
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


diag_indices = onp.diag_indices


@_wraps(onp.diagonal)
def diagonal(a, offset=0, axis1=0, axis2=1):
  a_shape = shape(a)
  a_ndims = len(a_shape)

  # Move the two dimensions to the end.
  axis1 %= a_ndims
  axis2 %= a_ndims
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


@_wraps(onp.diag)
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


@_wraps(onp.polyval)
def polyval(p, x):
  if isinstance(p, onp.poly1d):
    p = onp.asarray(p)
  if isinstance(x, onp.poly1d):
    y = 0
  else:
    y = zeros_like(x)
  for i in range(len(p)):
    y = y * x + p[i]
  return y


@_wraps(onp.append)
def append(arr, values, axis=None):
  if axis is None:
    return concatenate([ravel(arr), ravel(values)], 0)
  else:
    return concatenate([arr, values], axis=axis)


### Tensor contraction operations


@_wraps(onp.dot)
def dot(a, b):  # pylint: disable=missing-docstring
  _check_arraylike("dot", a, b)
  a, b = _promote_dtypes(a, b)
  a_ndim, b_ndim = ndim(a), ndim(b)
  if a_ndim == 0 or b_ndim == 0:
    return lax.mul(a, b)
  if _max(a_ndim, b_ndim) <= 2:
    return lax.dot(a, b)

  if b_ndim == 1:
    contract_dims = ((a_ndim - 1,), (0,))
  else:
    contract_dims = ((a_ndim - 1,), (b_ndim - 2,))
  batch_dims = ((), ())
  return lax.dot_general(a, b, (contract_dims, batch_dims))


@_wraps(onp.matmul)
def matmul(a, b):  # pylint: disable=missing-docstring
  _check_arraylike("matmul", a, b)
  a_is_vec, b_is_vec = (ndim(a) == 1), (ndim(b) == 1)
  a = lax.reshape(a, (1,) + shape(a)) if a_is_vec else a
  b = lax.reshape(b, shape(b) + (1,)) if b_is_vec else b

  a, b = _promote_dtypes(a, b)
  batch_shape = lax.broadcast_shapes(shape(a)[:-2], shape(b)[:-2])
  a = broadcast_to(a, batch_shape + shape(a)[-2:])
  b = broadcast_to(b, batch_shape + shape(b)[-2:])
  batch_dims = tuple(range(len(batch_shape)))
  result = lax.dot_general(a, b, (((ndim(a) - 1,), (ndim(b) - 2,)),
                                  (batch_dims, batch_dims)))

  if a_is_vec or b_is_vec:
    m, n = shape(result)[-2:]
    new_m = () if a_is_vec else (m,)
    new_n = () if b_is_vec else (n,)
    return lax.reshape(result, batch_shape + new_m + new_n)
  else:
    return result


@_wraps(onp.vdot)
def vdot(a, b):
  if onp.issubdtype(_dtype(a), onp.complexfloating):
    a = conj(a)
  return dot(a.ravel(), b.ravel())


@_wraps(onp.tensordot)
def tensordot(a, b, axes=2):
  _check_arraylike("tensordot", a, b)
  if not (ndim(a) >= 1 and ndim(b) >= 1):
    msg = "tensordot requires a.ndim and b.dim to be at least 1, got {} and {}."
    raise TypeError(msg.format(ndim(a), ndim(b)))

  if type(axes) is int:
    a, b = _promote_dtypes(a, b)
    a_reshape = lax.reshape(a, (_prod(a.shape[:-axes]), _prod(a.shape[-axes:])))
    b_reshape = lax.reshape(b, (_prod(b.shape[:axes]), _prod(b.shape[axes:])))
    out_reshape = lax.dot(a_reshape, b_reshape)
    return lax.reshape(out_reshape, a.shape[:-axes] + b.shape[axes:])
  elif type(axes) in (list, tuple) and len(axes) == 2:
    ax1, ax2 = axes
    if type(ax1) == type(ax2) == int:
      a_transposed = moveaxis(a, ax1, -1) if ax1 != a.ndim - 1 else a
      b_transposed = moveaxis(b, ax2, 0) if ax2 != 0 else b
      return tensordot(a_transposed, b_transposed, 1)
    elif type(ax1) in (list, tuple) and type(ax2) in (list, tuple):
      if len(ax1) != len(ax2):
        msg = "tensordot requires axes lists to have equal length, got {} and {}."
        raise TypeError(msg.format(ax1, ax2))
      num_axes = len(ax1)
      a_transposed = moveaxis(a, ax1, tuple(range(a.ndim - num_axes, a.ndim)))
      b_transposed = moveaxis(b, ax2, tuple(range(num_axes)))
      return tensordot(a_transposed, b_transposed, num_axes)
  msg = ("tensordot axes argument must be an int, a pair of ints, or a pair of "
         "lists/tuples of ints.")
  raise TypeError(msg)


@_wraps(onp.einsum)
def einsum(*operands, **kwargs):
  optimize = kwargs.pop('optimize', 'auto')
  optimize = 'greedy' if optimize is True else optimize
  if kwargs:
    msg = 'invalid keyword arguments for einsum: {}'
    raise TypeError(msg.format(', '.join(kwargs)))
  # using einsum_call=True here is an internal api for opt_einsum
  operands, contractions = opt_einsum.contract_path(
      *operands, einsum_call=True, use_blas=True, optimize=optimize)
  contractions = tuple(data[:3] for data in contractions)
  return _einsum(operands, contractions)

@_wraps(onp.einsum_path)
def einsum_path(subscripts, *operands, **kwargs):
  optimize = kwargs.pop('optimize', 'greedy')
  # using einsum_call=True here is an internal api for opt_einsum
  return opt_einsum.contract_path(subscripts, *operands, optimize=optimize)

@partial(jit, static_argnums=(1,))
def _einsum(operands, contractions):
  operands = list(_promote_dtypes(*operands))
  sum = lambda x, axes: lax.reduce(x, onp.array(0, x.dtype), lax.add, axes)

  def sum_uniques(operand, names, uniques):
    if uniques:
      axes = [names.index(name) for name in uniques]
      operand = sum(operand, axes)
      names = removechars(names, uniques)
    return operand, names

  def sum_repeats(operand, names, counts, keep_names):
    for name, count in counts.items():
      if count > 1:
        axes = [i for i, n in enumerate(names) if n == name]
        eye = lax.broadcasted_eye(operand.dtype, operand.shape, axes)
        if name not in keep_names:
          operand = sum(operand * eye, axes)
          names = names.replace(name, '')
        else:
          operand = sum(operand * eye, axes[:-1])
          names = names.replace(name, '', count - 1)
    return operand, names

  for operand_indices, contracted_names, einstr in contractions:
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
      lhs_counts, rhs_counts = map(collections.Counter, input_names)
      lhs_names, rhs_names = input_names

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

      contracted_names = contracted_names & (set(lhs_names) | set(rhs_names))
      batch_names = (set(lhs_names) & set(rhs_names)) - contracted_names
      lhs_batch, rhs_batch = unzip2((lhs_names.find(n), rhs_names.find(n))
                                    for n in batch_names)

      # NOTE(mattjj): this can fail non-deterministically in python3, maybe
      # due to opt_einsum
      assert _all(name in lhs_names and name in rhs_names and
                  lhs.shape[lhs_names.index(name)] == rhs.shape[rhs_names.index(name)]
                  for name in contracted_names)

      # move batch dims to the front (required by lax.dot_general, and easier)
      batch_dims = tuple(range(len(batch_names)))
      if lhs_batch != rhs_batch or set(lhs_batch) != set(batch_dims):
        lhs = moveaxis(lhs, lhs_batch, batch_dims)
        lhs_names = _movechars(lhs_names, lhs_batch, batch_dims)
        rhs = moveaxis(rhs, rhs_batch, batch_dims)
        rhs_names = _movechars(rhs_names, rhs_batch, batch_dims)
        batch_names = ''.join(batch_names)
      else:
        batch_dims = tuple(lhs_batch)
        batch_names = ''.join(lhs_names[i] for i in range(len(lhs_names))
                              if i in batch_dims)

      if contracted_names:
        # contract using lax.dot_general
        lhs_cont, rhs_cont = unzip2((lhs_names.index(n), rhs_names.index(n))
                                    for n in contracted_names)
        operand = _dot_general(lhs, rhs, lhs_cont, rhs_cont, len(batch_dims))
        deleted_names = batch_names + ''.join(contracted_names)
        names = (batch_names + removechars(lhs_names, deleted_names)
                 + removechars(rhs_names, deleted_names))
      else:
        # no contraction, just a tensor product
        nbatch = len(batch_names)
        assert lhs.shape[:nbatch] == rhs.shape[:nbatch]
        names = batch_names + lhs_names[nbatch:] + rhs_names[nbatch:]
        lhs_shape = lhs.shape + (1,) * (rhs.ndim - nbatch)
        rhs_shape = rhs.shape[:nbatch] + (1,) * (lhs.ndim - nbatch) + rhs.shape[nbatch:]
        operand = lax.reshape(lhs, lhs_shape) * lax.reshape(rhs, rhs_shape)

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


def _dot_general(lhs, rhs, lhs_cont, rhs_cont, nbatch):
  """Helper for einsum contractions."""
  # lax.dot_general has some tight constraints on dimension_numbers that this
  # wrapper loosens via transposes and reshapes
  assert len(lhs_cont) == len(rhs_cont) > 0
  ncont = len(lhs_cont)
  lhs_ntensor = lhs.ndim - nbatch - ncont
  rhs_ntensor = rhs.ndim - nbatch - ncont
  batch_dims = tuple(range(nbatch))

  if ncont == 1 and 0 <= lhs_ntensor <= 1 and 0 <= rhs_ntensor <= 1:
    dimension_numbers = [(lhs_cont, rhs_cont), (batch_dims, batch_dims)]
    return lax.dot_general(lhs, rhs, dimension_numbers)
  else:
    # move contracting dimensions to the end. lax.dot_general only allows one
    # contracting dimension, so if there's more than one we collapse them.
    if ncont > 1:
      lhs_cdims = tuple(range(lhs.ndim - ncont, lhs.ndim))
      lhs = moveaxis(lhs, lhs_cont, lhs_cdims)
      lhs = lhs.reshape(lhs.shape[:-ncont] + (-1,))

      rhs_cdims = tuple(range(rhs.ndim - ncont, rhs.ndim))
      rhs = moveaxis(rhs, rhs_cont, rhs_cdims)
      rhs = rhs.reshape(rhs.shape[:-ncont] + (-1,))
    else:
      lhs = moveaxis(lhs, lhs_cont[0], -1)
      rhs = moveaxis(rhs, rhs_cont[0], -1)

    # lax.dot_general only allows zero or one tensor product dims per operand,
    # so if there's more than one we collapse them.
    result_shape = lhs.shape[:nbatch] + lhs.shape[nbatch:-1] + rhs.shape[nbatch:-1]

    if lhs_ntensor > 1:
      lhs = lhs.reshape(lhs.shape[:nbatch] + (-1,) + lhs.shape[-1:])

    if rhs_ntensor > 1:
      rhs = rhs.reshape(rhs.shape[:nbatch] + (-1,) + rhs.shape[-1:])

    lhs_cont, rhs_cont = [lhs.ndim - 1], [rhs.ndim - 1]
    dimension_numbers = [(lhs_cont, rhs_cont), (batch_dims, batch_dims)]
    result = lax.dot_general(lhs, rhs, dimension_numbers)
    return lax.reshape(result, result_shape)


def _movechars(s, src, dst):
  """Helper for einsum string munging, like moveaxis on identifier strings."""
  chars = [c for i, c in enumerate(s) if i not in src]
  for i, j in sorted(zip(dst, src)):
    chars.insert(i, s[j])
  return ''.join(chars)


@_wraps(onp.inner)
def inner(a, b):
  if ndim(a) == 0 or ndim(b) == 0:
    return a * b
  return tensordot(a, b, (-1, -1))


@_wraps(onp.outer)
def outer(a, b, out=None):
  if out:
    raise NotImplementedError("The 'out' argument to outer is not supported.")
  return ravel(a)[:, None] * ravel(b)

@_wraps(onp.cross)
def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    if axis is not None:
        axisa = axis
        axisb = axis
        axisc = axis

    a_ndims = len(shape(a))
    b_ndims = len(shape(b))
    axisa = _canonicalize_axis(axisa, a_ndims)
    axisb = _canonicalize_axis(axisb, b_ndims)
    a = moveaxis(a, axisa, -1)
    b = moveaxis(b, axisb, -1)
    a_shape = shape(a)
    b_shape = shape(b)

    if a_shape[-1] not in (2, 3) or b_shape[-1] not in (2, 3):
        raise ValueError("Dimension must be either 2 or 3 for cross product")

    if a_shape[-1] == 2 and b_shape[-1] == 2:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    if a_shape[-1] == 2:
        a = concatenate((a, zeros(a_shape[:-1] + (1,), dtype=a.dtype)), axis=-1)
    elif b_shape[-1] == 2:
        b = concatenate((b, zeros(b_shape[:-1] + (1,), dtype=b.dtype)), axis=-1)

    a0 = a[..., 0]
    a1 = a[..., 1]
    a2 = a[..., 2]
    b0 = b[..., 0]
    b1 = b[..., 1]
    b2 = b[..., 2]

    c = array([a1 * b2 - a2 * b1,
               a2 * b0 - a0 * b2,
               a0 * b1 - a1 * b0])

    c_ndims = len(shape(c))
    axisc = _canonicalize_axis(axisc, c_ndims)

    return moveaxis(c, 0, axisc)

@_wraps(onp.kron)
def kron(a, b):
  a_shape = shape(a)
  b_shape = shape(b)
  a_ndims = len(a_shape)
  b_ndims = len(b_shape)
  a = array(a)
  b = array(b)
  d = _min(a_ndims, b_ndims)
  if d == 0:
    return a * b
  a_broadcast_dims = list(range(a_ndims - d, a_ndims + d, 2))
  a_broadcast_shape = onp.ones(a_ndims + d, dtype=onp.int64)
  a_broadcast_shape[:-2*d] = a_shape[:-d]
  a_broadcast_shape[a_broadcast_dims] = a_shape[-d:]

  b_broadcast_dims = list(range(b_ndims -d + 1, b_ndims + d + 1, 2))
  b_broadcast_shape = onp.ones(b_ndims + d, dtype=onp.int64)
  b_broadcast_shape[:-2*d] = b_shape[:-d]
  b_broadcast_shape[b_broadcast_dims] = b_shape[-d:]

  if a_ndims > b_ndims:
    out_shape = onp.array(a_shape, dtype=onp.int64)
    out_shape[-d:] *= onp.array(b_shape, dtype=onp.int64)
  else:
    out_shape = onp.array(b_shape, dtype=onp.int64)
    out_shape[-d:] *= onp.array(a_shape, dtype=onp.int64)

  a_broadcast = lax.broadcast_in_dim(
    a, a_broadcast_shape, list(range(a_ndims - d)) + a_broadcast_dims)
  b_broadcast = lax.broadcast_in_dim(
    b, b_broadcast_shape, list(range(b_ndims - d)) + b_broadcast_dims)
  return lax.reshape(a_broadcast * b_broadcast, out_shape)


@_wraps(onp.vander)
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


@_wraps(onp.argmax)
def argmax(a, axis=None):
  if axis is None:
    a = ravel(a)
    axis = 0
  return _argminmax(max, a, axis)


@_wraps(onp.argmin)
def argmin(a, axis=None):
  if axis is None:
    a = ravel(a)
    axis = 0
  return _argminmax(min, a, axis)


# TODO(mattjj): redo this lowering with a call to variadic lax.reduce
def _argminmax(op, a, axis):
  shape = [1] * a.ndim
  shape[axis] = a.shape[axis]
  idxs = onp.arange(a.shape[axis]).reshape(shape)
  maxval = onp.iinfo(xla_bridge.canonicalize_dtype(idxs.dtype)).max
  mask_idxs = where(lax._eq_meet(a, op(a, axis, keepdims=True)), idxs, maxval)
  return min(mask_idxs, axis)


@_wraps(onp.sort)
def sort(a, axis=-1, kind='quicksort', order=None):
  if kind != 'quicksort':
    warnings.warn("'kind' argument to sort is ignored.")
  if order is not None:
    raise ValueError("'order' argument to sort is not supported.")

  if axis is None:
    return lax.sort(a.ravel(), 0)
  else:
    return lax.sort(a, axis % ndim(a))


@_wraps(onp.argsort)
def argsort(a, axis=-1, kind='quicksort', order=None):
  if kind != 'quicksort':
    warnings.warn("'kind' argument to argsort is ignored.")
  if order is not None:
    raise ValueError("'order' argument to argsort is not supported.")

  if axis is None:
    return argsort(a.ravel(), 0)
  else:
    axis = axis % ndim(a)
    iota = lax.broadcasted_iota(onp.int64, shape(a), axis)
    _, perm = lax.sort_key_val(a, iota, dimension=axis)
    return perm


@_wraps(onp.roll)
def roll(a, shift, axis=None):
  a = asarray(a)
  a_shape = shape(a)
  if axis is None:
    return lax.reshape(roll(ravel(a), shift, axis=0), a_shape)

  a_ndim = len(a_shape)
  if isinstance(shift, tuple):
    if isinstance(axis, tuple):
      if len(axis) != len(shift):
        msg = "Mismatched lengths between shift ({}) and axis ({}) for np.roll."
        raise ValueError(msg.format(len(shift), len(axis)))
      axis = tuple(a for a in axis)
    else:
      axis = (axis,) * len(shift)
  elif isinstance(axis, tuple):
    shift = (shift,) * len(axis)
  else:
    shift = (shift,)
    axis = (axis,)

  for offset, i in zip(shift, axis):
    i = _canonicalize_axis(i, a_ndim)
    offset = offset % (a_shape[i] or 1)
    slices = [slice(None)] * a_ndim
    slices[i] = slice(None, -offset)
    before = a[tuple(slices)]
    slices[i] = slice(-offset, None)
    after = a[tuple(slices)]
    a = lax.concatenate((after, before), i)

  return a


@_wraps(onp.take)
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
  slice_sizes[axis] = 1
  dnums = lax.GatherDimensionNumbers(
    offset_dims=tuple(
      list(range(axis)) +
      list(range(axis + index_dims, len(a.shape) + index_dims - 1))),
    collapsed_slice_dims=(axis,),
    start_index_map=(axis,))
  return lax.gather(a, indices[..., None], dimension_numbers=dnums,
                    slice_sizes=tuple(slice_sizes))


@_wraps(getattr(onp, "take_along_axis", None))
def take_along_axis(arr, indices, axis):
  if axis is None and ndim(arr) != 1:
    return take_along_axis(arr.ravel(), indices.ravel(), 0)
  elif ndim(arr) == 1:
    return lax.index_take(arr, (indices,), (0,))
  else:
    # TODO(mattjj): if we lower directly to lax.gather here, we might be able to
    # avoid the reshape on the output.
    all_indices = [lax.broadcasted_iota(_dtype(indices), shape(indices), i)
                   for i in range(ndim(arr))]
    all_indices[axis] = indices
    all_indices = tuple(map(ravel, all_indices))
    out_flat = lax.index_take(arr, all_indices, tuple(range(ndim(arr))))
    return reshape(out_flat, shape(indices))


### Indexing


def _rewriting_take(arr, idx, axis=0):
  """A function like numpy.take that handles boxes and rewrites to LAX."""

  # Handle special indexers: (), Ellipsis, slice(None), and None.
  # TODO(mattjj): don't compare empty tuple identity (though works for CPython)
  if idx is () or idx is Ellipsis or _is_slice_none(idx):  # pylint: disable=literal-comparison
    return arr
  elif idx is None:
    return expand_dims(arr, 0)


  # Handle int index
  _int = lambda aval: not aval.shape and onp.issubdtype(aval.dtype, onp.integer)
  try:
    abstract_idx = core.get_aval(idx)
  except TypeError:
    abstract_idx = None

  if isinstance(abstract_idx, ConcreteArray) and _int(abstract_idx):
    return lax.index_in_dim(arr, idx, axis, False)
  elif isinstance(abstract_idx, ShapedArray) and _int(abstract_idx):
    idx = mod(idx, _constant_like(idx, arr.shape[axis]))
    return lax.dynamic_index_in_dim(arr, idx, axis, False)

  # Handle slice index (only static, otherwise an error is raised)
  elif isinstance(idx, slice):
    if not _all(elt is None or type(core.get_aval(elt)) is ConcreteArray
                for elt in (idx.start, idx.stop, idx.step)):
      msg = ("Array slice indices must have static start/stop/step to be used "
             "with Numpy indexing syntax. Try lax.dynamic_slice instead.")
      raise IndexError(msg)
    else:
      start, limit, stride, needs_rev = _static_idx(idx, arr.shape[axis])
      result = lax.slice_in_dim(arr, start, limit, stride, axis=axis)
      return lax.rev(result, [axis]) if needs_rev else result

  # Handle non-advanced bool index (only static, otherwise an error is raised)
  elif (isinstance(abstract_idx, ShapedArray) and onp.issubdtype(abstract_idx.dtype, onp.bool_)
        or isinstance(idx, list) and _all(not _shape(e) and onp.issubdtype(_dtype(e), onp.bool_)
                                          for e in idx)):
    if isinstance(idx, list):
      idx = array(idx)
      abstract_idx = core.get_aval(idx)

    if not type(abstract_idx) is ConcreteArray:
      msg = ("Array boolean indices must be static (e.g. no dependence on an "
             "argument to a jit or vmap function).")
      raise IndexError(msg)
    else:
      if idx.ndim > arr.ndim or idx.shape != arr.shape[:idx.ndim]:
        msg = "Boolean index shape did not match indexed array shape prefix."
        raise IndexError(msg)
      else:
        reshaped_arr = arr.reshape((-1,) + arr.shape[idx.ndim:])
        int_idx, = onp.where(idx.ravel())
        return lax.index_take(reshaped_arr, (int_idx,), (0,))

  # Handle non-advanced tuple indices by recursing once
  elif isinstance(idx, tuple) and _all(onp.ndim(elt) == 0 for elt in idx):
    canonical_idx = _canonicalize_tuple_index(arr, idx)
    result, axis = arr, 0
    # TODO(mattjj): could generate a single HLO here, rather than one for each
    # elt in canonical idx. For example, x[0, :, 0] generates three HLOs now.
    for elt in (elt for elt in canonical_idx if elt is not None):
      result = _rewriting_take(result, elt, axis=axis)
      axis += isinstance(elt, slice)   # advance axis index if not eliminated
    unexpanded_shape_itr = iter(result.shape)
    result_shape = tuple(1 if elt is None else next(unexpanded_shape_itr)
                         for elt in canonical_idx if isinstance(elt, (type(None), slice)))
    return lax.reshape(result, result_shape) if result_shape else result

  # Handle advanced indexing (non-tuple sequence, ndarray of dtype int or bool,
  # or a tuple with at least one sequence object).
  # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
  # https://gist.github.com/seberg/976373b6a2b7c4188591

  # Handle integer array indexing *without* ellipsis/slices/nones
  # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#integer-array-indexing
  if _is_advanced_int_indexer_without_slices(idx):
    if isinstance(idx, (tuple, list)):
      if _any(_shape(e) for e in idx):
        # At least one sequence element in the index list means broadcasting.
        idx = broadcast_arrays(*idx)
      else:
        # The index list is a flat list of integers.
        idx = [lax.concatenate([lax.reshape(e, (1,)) for e in idx], 0)]
    else:
      # The indexer is just a single integer array.
      idx = [idx]

    flat_idx = tuple([mod(ravel(x), _constant_like(x, arr.shape[i]))
                      for i, x in enumerate(idx)])
    # TODO(mattjj): if we instead lower directly to lax.gather, we can probably
    # eliminate the reshape here.
    out = lax.index_take(arr, flat_idx, tuple(range(len(idx))))
    return lax.reshape(out, idx[0].shape + _shape(arr)[len(idx):])

  # Handle integer array indexing *with* ellipsis/slices/nones by recursing once
  # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#combining-advanced-and-basic-indexing
  elif _is_advanced_int_indexer(idx):
    canonical_idx = _canonicalize_tuple_index(arr, tuple(idx))
    idx_noadvanced = [slice(None) if _is_int_arraylike(e) else e
                      for e in canonical_idx]
    arr_sliced = _rewriting_take(arr, tuple(idx_noadvanced))

    advanced_pairs = ((e, i) for i, e in enumerate(canonical_idx) if _is_int_arraylike(e))
    idx_advanced, axes = zip(*advanced_pairs)
    idx_advanced = broadcast_arrays(*idx_advanced)

    flat_idx = tuple(mod(ravel(x), _constant_like(x, arr_sliced.shape[i]))
                     for i, x in zip(axes, idx_advanced))
    # TODO(mattjj): if we instead lower directly to lax.gather, we can probably
    # eliminate the reshape here.
    out = lax.index_take(arr_sliced, flat_idx, axes)
    shape_suffix = tuple(onp.delete(_shape(arr_sliced), axes))
    out = lax.reshape(out, idx_advanced[0].shape + shape_suffix)

    axes_are_contiguous = onp.all(onp.diff(axes) == 1)
    if axes_are_contiguous:
      start = axes[0]
      naxes = idx_advanced[0].ndim
      out = moveaxis(out, list(range(naxes)), list(range(start, start + naxes)))
    return out

  msg = "Indexing mode not yet supported. Open a feature request!\n{}"
  raise IndexError(msg.format(idx))


def _is_slice_none(idx):
  """Return True if idx is equal to slice(None), False otherwise."""
  if isinstance(idx, slice):
    return idx.start is None and idx.stop is None and idx.step is None


def _is_advanced_int_indexer(idx):
  """Returns True if idx should trigger int array indexing, False otherwise."""
  # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
  if isinstance(idx, (tuple, list)) and _any(onp.ndim(elt) != 0 for elt in idx):
    return _all(e is None or e is Ellipsis or isinstance(e, slice)
                or _is_int_arraylike(e) for e in idx)
  else:
    return _is_int_arraylike(idx)


def _is_advanced_int_indexer_without_slices(idx):
  """Returns True iff idx is an advanced int idx without slice/ellipsis/none."""
  if _is_advanced_int_indexer(idx):
    if isinstance(idx, (tuple, list)):
      return not _any(e is None or e is Ellipsis or isinstance(e, slice)
                      for e in idx)
    else:
      return True


def _is_int_arraylike(x):
  """Returns True if x is array-like with integer dtype, False otherwise."""
  return (isinstance(x, int) and not isinstance(x, bool)
          or onp.issubdtype(getattr(x, "dtype", None), onp.integer)
          or isinstance(x, (list, tuple)) and _all(_is_int_arraylike(e) for e in x))


def _canonicalize_tuple_index(arr, idx):
  """Helper to remove Ellipsis and add in the implicit trailing slice(None)."""
  len_without_none = _sum(1 for e in idx if e is not None and e is not Ellipsis)
  if len_without_none > arr.ndim:
    msg = "Too many indices for array: {} non-None/Ellipsis indices for dim {}."
    raise IndexError(msg.format(len_without_none, arr.ndim))
  ellipses = (i for i, elt in enumerate(idx) if elt is Ellipsis)
  ellipsis_index = next(ellipses, None)
  if ellipsis_index is not None:
    if next(ellipses, None) is not None:
      msg = "Multiple ellipses (...) not supported: {}."
      raise IndexError(msg.format(list(map(type, idx))))
    colons = (slice(None),) * (arr.ndim - len_without_none)
    idx = idx[:ellipsis_index] + colons + idx[ellipsis_index + 1:]
  elif len_without_none < arr.ndim:
    colons = (slice(None),) * (arr.ndim - len_without_none)
    idx = tuple(idx) + colons
  return idx


def _static_idx(idx, size):
  """Helper function to compute the static slice start/limit/stride values."""
  indices = onp.arange(size)[idx]  # get shape statically
  if not len(indices):  # pylint: disable=g-explicit-length-test
    return 0, 0, 1, False  # sliced to size zero
  start, stop_inclusive = indices[0], indices[-1]
  step = 1 if idx.step is None else idx.step
  if step > 0:
    end = _min(stop_inclusive + step, size)
    return start, end, step, False
  else:
    end = _min(start - step, size)
    return stop_inclusive, end, -step, True


blackman = onp.blackman
bartlett = onp.bartlett
hamming = onp.hamming
hanning = onp.hanning
kaiser = onp.kaiser  # TODO: lower via lax to allow non-constant beta.


@_wraps(getattr(onp, "gcd", None))
def gcd(x1, x2):
  if (not issubdtype(_dtype(x1), integer) or
      not issubdtype(_dtype(x2), integer)):
    raise ValueError("Arguments to gcd must be integers.")
  def cond_fn(xs):
    x1, x2 = xs
    return any(x2 != 0)
  def body_fn(xs):
    x1, x2 = xs
    x1, x2 = (where(x2 != 0, x2, x1),
              where(x2 != 0, lax.rem(x1, x2), lax._const(x2, 0)))
    return (where(x1 < x2, x2, x1), where(x1 < x2, x1, x2))
  x1, x2 = _promote_dtypes(lax.abs(x1), lax.abs(x2))
  x1, x2 = broadcast_arrays(x1, x2)
  gcd, _ = lax.while_loop(cond_fn, body_fn, (x1, x2))
  return gcd


@_wraps(getattr(onp, "lcm", None))
def lcm(x1, x2):
  d = gcd(x1, x2)
  return where(d == 0, lax._const(d, 0),
               lax.div(lax.abs(multiply(x1, x2)), d))


### track unimplemented functions

def _not_implemented(fun):
  @_wraps(fun)
  def wrapped(*args, **kwargs):
    msg = "Numpy function {} not yet implemented"
    raise NotImplementedError(msg.format(fun))
  return wrapped

# Build a set of all unimplemented NumPy functions.
for func in get_module_functions(onp):
  if func.__name__ not in globals():
    globals()[func.__name__] = _not_implemented(func)


### add method and operator overloads to arraylike classes

# We add operator overloads to DeviceArray and ShapedArray. These method and
# operator overloads mainly just forward calls to the corresponding lax_numpy
# functions, which can themselves handle instances from any of these classes.


def _swap_args(f):
  return lambda x, y: f(y, x)

_operators = {
    "getitem": _rewriting_take,
    "neg": negative,
    "eq": equal,
    "ne": not_equal,
    "lt": less,
    "le": less_equal,
    "gt": greater,
    "ge": greater_equal,
    "abs": abs,
    "add": add,
    "radd": add,
    "sub": subtract,
    "rsub": _swap_args(subtract),
    "mul": multiply,
    "rmul": multiply,
    "div": divide,
    "rdiv": _swap_args(divide),
    "truediv": true_divide,
    "rtruediv": _swap_args(true_divide),
    "floordiv": floor_divide,
    "rfloordiv": _swap_args(floor_divide),
    "divmod": divmod,
    "rdivmod": _swap_args(divmod),
    "mod": mod,
    "rmod": _swap_args(mod),
    "pow": power,
    "rpow": _swap_args(power),
    "matmul": matmul,
    "rmatmul": _swap_args(matmul),
    "and": bitwise_and,
    "rand": bitwise_and,
    "or": bitwise_or,
    "ror": bitwise_or,
    "xor": bitwise_xor,
    "rxor": bitwise_xor,
    "invert": bitwise_not,
    "lshift": left_shift,
    "rshift": right_shift,
}

# These numpy.ndarray methods are just refs to an equivalent numpy function
_nondiff_methods = ["all", "any", "argmax", "argmin", "argpartition", "argsort",
                    "nonzero", "searchsorted", "round"]
_diff_methods = ["clip", "compress", "conj", "conjugate", "cumprod", "cumsum",
                 "diagonal", "dot", "max", "mean", "min", "prod", "ptp",
                 "ravel", "repeat", "sort", "squeeze", "std", "sum",
                 "swapaxes", "take", "tile", "trace", "transpose", "var"]


# Set up operator, method, and property forwarding on Tracer instances containing
# ShapedArray avals by following the forwarding conventions for Tracer.
# Forward operators using a single-underscore-prefix naming convention:
for operator_name, function in _operators.items():
  setattr(ShapedArray, "_{}".format(operator_name), staticmethod(function))
# Forward methods and properties using core.aval_method and core.aval_property:
for method_name in _nondiff_methods + _diff_methods:
  setattr(ShapedArray, method_name, core.aval_method(globals()[method_name]))
setattr(ShapedArray, "reshape", core.aval_method(_reshape))
setattr(ShapedArray, "flatten", core.aval_method(ravel))
setattr(ShapedArray, "T", core.aval_property(transpose))
setattr(ShapedArray, "real", core.aval_property(real))
setattr(ShapedArray, "imag", core.aval_property(imag))
setattr(ShapedArray, "astype", core.aval_method(lax.convert_element_type))


# Forward operators, methods, and properties on DeviceArray to lax_numpy
# functions (with no Tracers involved; this forwarding is direct)
for operator_name, function in _operators.items():
  setattr(DeviceArray, "__{}__".format(operator_name), function)
for method_name in _nondiff_methods + _diff_methods:
  setattr(DeviceArray, method_name, globals()[method_name])
setattr(DeviceArray, "reshape", _reshape)
setattr(DeviceArray, "flatten", ravel)
setattr(DeviceArray, "T", property(transpose))
setattr(DeviceArray, "real", property(real))
setattr(DeviceArray, "imag", property(imag))
setattr(DeviceArray, "astype", lax.convert_element_type)


# Extra methods that are handy
setattr(ShapedArray, "broadcast", core.aval_method(lax.broadcast))
setattr(ShapedArray, "split", core.aval_method(split))
setattr(DeviceArray, "broadcast", lax.broadcast)
setattr(DeviceArray, "split", split)
