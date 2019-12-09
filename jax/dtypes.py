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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from distutils.util import strtobool
import functools
import os

import numpy as onp
import six

from . import util
from .config import flags
from .lib import xla_client

FLAGS = flags.FLAGS
flags.DEFINE_bool('jax_enable_x64',
                  strtobool(os.getenv('JAX_ENABLE_X64', 'False')),
                  'Enable 64-bit types to be used.')

# bfloat16 support
bfloat16 = xla_client.bfloat16
_bfloat16_dtype = onp.dtype(bfloat16)

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

bool_ = onp.bool_
int_ = onp.int64
float_ = onp.float64
complex_ = onp.complex128

# TODO(phawkins): change the above defaults to:
# int_ = onp.int32
# float_ = onp.float32
# complex_ = onp.complex64


_dtype_to_32bit_dtype = {
    onp.dtype('int64'): onp.dtype('int32'),
    onp.dtype('uint64'): onp.dtype('uint32'),
    onp.dtype('float64'): onp.dtype('float32'),
    onp.dtype('complex128'): onp.dtype('complex64'),
}

@util.memoize
def canonicalize_dtype(dtype):
  """Convert from a dtype to a canonical dtype based on FLAGS.jax_enable_x64."""
  dtype = onp.dtype(dtype)

  if FLAGS.jax_enable_x64:
    return dtype
  else:
    return _dtype_to_32bit_dtype.get(dtype, dtype)


# Default dtypes corresponding to Python scalars.
python_scalar_dtypes = {
  bool: onp.dtype(bool_),
  int: onp.dtype(int_),
  float: onp.dtype(float_),
  complex: onp.dtype(complex_),
}

if six.PY2:
  python_scalar_dtypes[long] = onp.dtype(int_)  # noqa: F821

def scalar_type_of(x):
  typ = dtype(x)
  if onp.issubdtype(typ, onp.bool_):
    return bool
  elif onp.issubdtype(typ, onp.integer):
    return int
  elif onp.issubdtype(typ, onp.floating):
    return float
  elif onp.issubdtype(typ, onp.complexfloating):
    return complex
  else:
    raise TypeError("Invalid scalar value {}".format(x))

def coerce_to_array(x):
  """Coreces a scalar or NumPy array to an onp.array.

  Handles Python scalar type promotion according to JAX's rules, not NumPy's
  rules.
  """
  dtype = python_scalar_dtypes.get(type(x), None)
  return onp.array(x, dtype) if dtype else onp.array(x)

iinfo = onp.iinfo

def finfo(dtype):
  # Since NumPy doesn't consider bfloat16 a floating-point type, we have to
  # provide an alternative implementation of finfo that does so.
  if onp.result_type(dtype) == _bfloat16_dtype:
    return _bfloat16_finfo
  else:
    return onp.finfo(dtype)


def issubdtype(a, b):
  if a == bfloat16:
    return b in [onp.floating, onp.inexact, onp.number]
  return onp.issubdtype(a, b)

can_cast = onp.can_cast
issubsctype = onp.issubsctype


# List of all valid JAX dtypes, in the order they appear in the type promotion
# table.
_jax_types = [
  onp.dtype('bool'),
  onp.dtype('uint8'),
  onp.dtype('uint16'),
  onp.dtype('uint32'),
  onp.dtype('uint64'),
  onp.dtype('int8'),
  onp.dtype('int16'),
  onp.dtype('int32'),
  onp.dtype('int64'),
  onp.dtype(bfloat16),
  onp.dtype('float16'),
  onp.dtype('float32'),
  onp.dtype('float64'),
  onp.dtype('complex64'),
  onp.dtype('complex128'),
]

# Mapping from types to their type numbers.
_jax_type_nums = {t: i for i, t in enumerate(_jax_types)}

def _make_type_promotion_table():
  b1, u1, u2, u4, u8, s1, s2, s4, s8, bf, f2, f4, f8, c4, c8 = _jax_types
  #  b1, u1, u2, u4, u8, s1, s2, s4, s8, bf, f2, f4, f8, c4, c8
  return onp.array([
    [b1, u1, u2, u4, u8, s1, s2, s4, s8, bf, f2, f4, f8, c4, c8],  # b1
    [u1, u1, u2, u4, u8, s2, s2, s4, s8, bf, f2, f4, f8, c4, c8],  # u1
    [u2, u2, u2, u4, u8, s4, s4, s4, s8, bf, f2, f4, f8, c4, c8],  # u2
    [u4, u4, u4, u4, u8, s8, s8, s8, s8, bf, f2, f4, f8, c4, c8],  # u4
    [u8, u8, u8, u8, u8, f8, f8, f8, f8, bf, f2, f4, f8, c4, c8],  # u8
    [s1, s2, s4, s8, f8, s1, s2, s4, s8, bf, f2, f4, f8, c4, c8],  # s1
    [s2, s2, s4, s8, f8, s2, s2, s4, s8, bf, f2, f4, f8, c4, c8],  # s2
    [s4, s4, s4, s8, f8, s4, s4, s4, s8, bf, f2, f4, f8, c4, c8],  # s4
    [s8, s8, s8, s8, f8, s8, s8, s8, s8, bf, f2, f4, f8, c4, c8],  # s8
    [bf, bf, bf, bf, bf, bf, bf, bf, bf, bf, f4, f4, f8, c4, c8],  # bf
    [f2, f2, f2, f2, f2, f2, f2, f2, f2, f4, f2, f4, f8, c4, c8],  # f2
    [f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f8, c4, c8],  # f4
    [f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, c8, c8],  # f8
    [c4, c4, c4, c4, c4, c4, c4, c4, c4, c4, c4, c4, c8, c4, c8],  # c4
    [c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8],  # c8
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
  a = onp.dtype(a)
  b = onp.dtype(b)
  try:
    return _type_promotion_table[_jax_type_nums[a], _jax_type_nums[b]]
  except KeyError:
    pass
  raise TypeError("Invalid type promotion of {} and {}".format(a, b))


def is_python_scalar(x):
  try:
    return x.aval.weak_type and onp.ndim(x) == 0
  except AttributeError:
    return type(x) in python_scalar_dtypes

def _dtype_priority(dtype):
  if issubdtype(dtype, onp.bool_):
    return 0
  elif issubdtype(dtype, onp.integer):
    return 1
  elif issubdtype(dtype, onp.floating):
    return 2
  elif issubdtype(dtype, onp.complexfloating):
    return 3
  else:
    raise TypeError("Dtype {} is not supported by JAX".format(dtype))

def dtype(x):
  if type(x) in python_scalar_dtypes:
    return python_scalar_dtypes[type(x)]
  return onp.result_type(x)

def result_type(*args):
  """Convenience function to apply Numpy argument dtype promotion."""
  # TODO(dougalm,mattjj): This is a performance bottleneck. Consider memoizing.
  if len(args) < 2:
    return dtype(args[0])
  scalars = []
  dtypes = []
  for x in args:
    (scalars if is_python_scalar(x) else dtypes).append(dtype(x))
  array_priority = max(map(_dtype_priority, dtypes)) if dtypes else -1
  dtypes += [x for x in scalars if _dtype_priority(x) > array_priority]
  return canonicalize_dtype(functools.reduce(promote_types, dtypes))
