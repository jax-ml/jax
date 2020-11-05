# Copyright 2019 Google LLC
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

# Array type functions.
#
# JAX dtypes differ from NumPy in both:
# a) their type promotion rules, and
# b) the set of supported types (e.g., bfloat16),
# so we need our own implementation that deviates from NumPy in places.


from distutils.util import strtobool
import functools
import os

import numpy as np

from . import util
from .config import flags
from .lib import xla_client

from ._src import traceback_util
traceback_util.register_exclusion(__file__)

FLAGS = flags.FLAGS
flags.DEFINE_bool('jax_enable_x64',
                  strtobool(os.getenv('JAX_ENABLE_X64', 'False')),
                  'Enable 64-bit types to be used.')

# bfloat16 support
bfloat16 = xla_client.bfloat16
_bfloat16_dtype = np.dtype(bfloat16)

class _bfloat16_finfo(object):
  bits = 16
  eps = bfloat16(float.fromhex("0x1p-7"))
  epsneg = bfloat16(float.fromhex("0x1p-8"))
  machep = -7
  negep = -8
  max = bfloat16(float.fromhex("0x1.FEp127"))
  min = -max
  nexp = 8
  nmant = 7
  iexp = nexp
  precision = 2
  resolution = 10 ** -2
  tiny = bfloat16(float.fromhex("0x1p-126"))

# Default types.

bool_ = np.bool_
int_ = np.int64
float_ = np.float64
complex_ = np.complex128

# TODO(phawkins): change the above defaults to:
# int_ = np.int32
# float_ = np.float32
# complex_ = np.complex64

# Trivial vectorspace datatype needed for tangent values of int/bool primals
float0 = np.dtype([('float0', np.void, 0)])

_dtype_to_32bit_dtype = {
    np.dtype('int64'): np.dtype('int32'),
    np.dtype('uint64'): np.dtype('uint32'),
    np.dtype('float64'): np.dtype('float32'),
    np.dtype('complex128'): np.dtype('complex64'),
}

@util.memoize
def canonicalize_dtype(dtype):
  """Convert from a dtype to a canonical dtype based on FLAGS.jax_enable_x64."""
  if isinstance(dtype, str) and dtype == "bfloat16":
    dtype = bfloat16
  try:
    dtype = np.dtype(dtype)
  except TypeError as e:
    raise TypeError(f'dtype {dtype!r} not understood') from e

  if FLAGS.jax_enable_x64:
    return dtype
  else:
    return _dtype_to_32bit_dtype.get(dtype, dtype)


# Default dtypes corresponding to Python scalars.
python_scalar_dtypes = {
  bool: np.dtype(bool_),
  int: np.dtype(int_),
  float: np.dtype(float_),
  complex: np.dtype(complex_),
  float0: float0
}

def scalar_type_of(x):
  typ = dtype(x)
  if np.issubdtype(typ, np.bool_):
    return bool
  elif np.issubdtype(typ, np.integer):
    return int
  elif np.issubdtype(typ, np.floating):
    return float
  elif np.issubdtype(typ, np.complexfloating):
    return complex
  else:
    raise TypeError("Invalid scalar value {}".format(x))

def coerce_to_array(x):
  """Coerces a scalar or NumPy array to an np.array.

  Handles Python scalar type promotion according to JAX's rules, not NumPy's
  rules.
  """
  dtype = python_scalar_dtypes.get(type(x), None)
  return np.array(x, dtype) if dtype else np.array(x)

iinfo = np.iinfo

def finfo(dtype):
  # Since NumPy doesn't consider bfloat16 a floating-point type, we have to
  # provide an alternative implementation of finfo that does so.
  if ((isinstance(dtype, str) and dtype == "bfloat16") or
      np.result_type(dtype) == _bfloat16_dtype):
    return _bfloat16_finfo
  else:
    return np.finfo(dtype)

def _issubclass(a, b):
  """Determines if ``a`` is a subclass of ``b``.

  Similar to issubclass, but returns False instead of an exception if `a` is not
  a class.
  """
  try:
    return issubclass(a, b)
  except TypeError:
    return False

def issubdtype(a, b):
  if a == bfloat16:
    if isinstance(b, np.dtype):
      return b == _bfloat16_dtype
    else:
      return b in [bfloat16, np.floating, np.inexact, np.number]
  if not _issubclass(b, np.generic):
    # Workaround for JAX scalar types. NumPy's issubdtype has a backward
    # compatibility behavior for the second argument of issubdtype that
    # interacts badly with JAX's custom scalar types. As a workaround,
    # explicitly cast the second argument to a NumPy type object.
    b = np.dtype(b).type
  return np.issubdtype(a, b)

can_cast = np.can_cast
issubsctype = np.issubsctype


# List of all valid JAX dtypes, in the order they appear in the type promotion
# table.
_weak_types = [int, float, complex]

_jax_types = [
  np.dtype('bool'),
  np.dtype('uint8'),
  np.dtype('uint16'),
  np.dtype('uint32'),
  np.dtype('uint64'),
  np.dtype('int8'),
  np.dtype('int16'),
  np.dtype('int32'),
  np.dtype('int64'),
  np.dtype(bfloat16),
  np.dtype('float16'),
  np.dtype('float32'),
  np.dtype('float64'),
  np.dtype('complex64'),
  np.dtype('complex128'),
] + _weak_types

def _jax_type(value):
  """Return the jax type for a value or type."""
  # Note: `value in _weak_types` can return false positives due to dtype comparator overloading.
  if any(value is typ for typ in _weak_types):
    return value
  dtype_ = dtype(value)
  if is_weakly_typed(value):
    pytype = type(dtype_.type(0).item())
    if pytype in _weak_types:
      return pytype
  return dtype_

# Mapping from types to their type numbers.
_jax_type_nums = {t: i for i, t in enumerate(_jax_types)}


def _make_type_promotion_table():
  # Note: this promotion table is generated via the least upper bounds over a type
  # promotion lattice. See testPromotionTableLattice for details of this.
  b1, u1, u2, u4, u8, i1, i2, i4, i8, bf, f2, f4, f8, c4, c8, i_, f_, c_ = _jax_types
  #  b1, u1, u2, u4, u8, i1, i2, i4, i8, bf, f2, f4, f8, c4, c8, s*, f*, c*
  return np.array([
    [b1, u1, u2, u4, u8, i1, i2, i4, i8, bf, f2, f4, f8, c4, c8, i_, f_, c_],  # b1
    [u1, u1, u2, u4, u8, i2, i2, i4, i8, bf, f2, f4, f8, c4, c8, u1, f_, c_],  # u1
    [u2, u2, u2, u4, u8, i4, i4, i4, i8, bf, f2, f4, f8, c4, c8, u2, f_, c_],  # u2
    [u4, u4, u4, u4, u8, i8, i8, i8, i8, bf, f2, f4, f8, c4, c8, u4, f_, c_],  # u4
    [u8, u8, u8, u8, u8, f_, f_, f_, f_, bf, f2, f4, f8, c4, c8, u8, f_, c_],  # u8
    [i1, i2, i4, i8, f_, i1, i2, i4, i8, bf, f2, f4, f8, c4, c8, i1, f_, c_],  # i1
    [i2, i2, i4, i8, f_, i2, i2, i4, i8, bf, f2, f4, f8, c4, c8, i2, f_, c_],  # i2
    [i4, i4, i4, i8, f_, i4, i4, i4, i8, bf, f2, f4, f8, c4, c8, i4, f_, c_],  # i4
    [i8, i8, i8, i8, f_, i8, i8, i8, i8, bf, f2, f4, f8, c4, c8, i8, f_, c_],  # i8
    [bf, bf, bf, bf, bf, bf, bf, bf, bf, bf, f4, f4, f8, c4, c8, bf, bf, c4],  # bf
    [f2, f2, f2, f2, f2, f2, f2, f2, f2, f4, f2, f4, f8, c4, c8, f2, f2, c4],  # f2
    [f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f8, c4, c8, f4, f4, c4],  # f4
    [f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, c8, c8, f8, f8, c8],  # f8
    [c4, c4, c4, c4, c4, c4, c4, c4, c4, c4, c4, c4, c8, c4, c8, c4, c4, c4],  # c4
    [c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8],  # c8
    [i_, u1, u2, u4, u8, i1, i2, i4, i8, bf, f2, f4, f8, c4, c8, i_, f_, c_],  # s*
    [f_, f_, f_, f_, f_, f_, f_, f_, f_, bf, f2, f4, f8, c4, c8, f_, f_, c_],  # f*
    [c_, c_, c_, c_, c_, c_, c_, c_, c_, c4, c4, c4, c8, c4, c8, c_, c_, c_],  # c*
  ])

_type_promotion_table = _make_type_promotion_table()

def promote_types(a, b):
  """Returns the type to which a binary operation should cast its arguments.

  For details of JAX's type promotion semantics, see :ref:`type-promotion`.

  Args:
    a: a :class:`numpy.dtype` or a dtype specifier.
    b: a :class:`numpy.dtype` or a dtype specifier.

  Returns:
    A :class:`numpy.dtype` object.
  """
  return np.dtype(_promote_types_raw(np.dtype(a), np.dtype(b)))

def _promote_types_raw(a, b):
  # Note: `x in _weak_types` can return false positives due to dtype comparator overloading.
  a = a if any(a is t for t in _weak_types) else np.dtype(a)
  b = b if any(b is t for t in _weak_types) else np.dtype(b)
  try:
    return _type_promotion_table[_jax_type_nums[a], _jax_type_nums[b]]
  except KeyError:
    pass
  raise TypeError("Invalid type promotion of {} and {}".format(a, b))

def is_weakly_typed(x):
  try:
    return x.aval.weak_type
  except AttributeError:
    return type(x) in python_scalar_dtypes

def is_python_scalar(x):
  return is_weakly_typed(x) and np.ndim(x) == 0

def dtype(x):
  if type(x) in python_scalar_dtypes:
    return python_scalar_dtypes[type(x)]
  return np.result_type(x)

def result_type(*args):
  """Convenience function to apply Numpy argument dtype promotion."""
  # TODO(dougalm,mattjj): This is a performance bottleneck. Consider memoizing.
  if len(args) < 2:
    return canonicalize_dtype(dtype(args[0]))
  result_type = functools.reduce(_promote_types_raw, (_jax_type(arg) for arg in args))
  # TODO(jakevdp): propagate weak_type to the result when necessary.
  return canonicalize_dtype(result_type)
