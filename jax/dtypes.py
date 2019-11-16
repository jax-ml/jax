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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from distutils.util import strtobool
import os

import numpy as onp
import six

from .config import flags
from . import util

FLAGS = flags.FLAGS
flags.DEFINE_bool('jax_enable_x64',
                  strtobool(os.getenv('JAX_ENABLE_X64', 'False')),
                  'Enable 64-bit types to be used.')


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
  python_scalar_dtypes[long] = int_  # noqa: F821

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
finfo = onp.finfo

can_cast = onp.can_cast
issubdtype = onp.issubdtype
issubsctype = onp.issubsctype
promote_types = onp.promote_types


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
  return canonicalize_dtype(onp.result_type(*dtypes))