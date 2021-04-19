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


import functools
from typing import Any, Dict

import numpy as np

from jax._src import util
from jax._src.config import flags, config
from jax.lib import xla_client

from jax._src import traceback_util
traceback_util.register_exclusion(__file__)

FLAGS = flags.FLAGS

# bfloat16 support
bfloat16: type = xla_client.bfloat16
_bfloat16_dtype: np.dtype = np.dtype(bfloat16)

# Default types.

bool_ = np.bool_
int_: np.dtype = np.int64  # type: ignore
float_: np.dtype = np.float64  # type: ignore
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
  """Convert from a dtype to a canonical dtype based on config.x64_enabled."""
  try:
    dtype = np.dtype(dtype)
  except TypeError as e:
    raise TypeError(f'dtype {dtype!r} not understood') from e

  if config.x64_enabled:
    return dtype
  else:
    return _dtype_to_32bit_dtype.get(dtype, dtype)


# Default dtypes corresponding to Python scalars.
python_scalar_dtypes : dict = {
  bool: np.dtype(bool_),
  int: np.dtype(int_),
  float: np.dtype(float_),
  complex: np.dtype(complex_),
}

def scalar_type_of(x):
  typ = dtype(x)
  if typ == bfloat16:
    return float
  elif np.issubdtype(typ, np.bool_):
    return bool
  elif np.issubdtype(typ, np.integer):
    return int
  elif np.issubdtype(typ, np.floating):
    return float
  elif np.issubdtype(typ, np.complexfloating):
    return complex
  else:
    raise TypeError("Invalid scalar value {}".format(x))


def _scalar_type_to_dtype(typ: type, value: Any = None):
  """Return the numpy dtype for the given scalar type.

  Raises
  ------
  OverflowError: if `typ` is `int` and the value is too large for int64.

  Examples
  --------
  >>> _scalar_type_to_dtype(int)
  dtype('int32')
  >>> _scalar_type_to_dtype(float)
  dtype('float32')
  >>> _scalar_type_to_dtype(complex)
  dtype('complex64')
  >>> _scalar_type_to_dtype(int)
  dtype('int32')
  >>> _scalar_type_to_dtype(int, 0)
  dtype('int32')
  >>> _scalar_type_to_dtype(int, 1 << 63)  # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
  OverflowError: Python int 9223372036854775808 too large to convert to int32
  """
  dtype = canonicalize_dtype(python_scalar_dtypes[typ])
  if typ is int and value is not None:
    if value < np.iinfo(dtype).min or value > np.iinfo(dtype).max:
      raise OverflowError(f"Python int {value} too large to convert to {dtype}")
  return dtype


def coerce_to_array(x, dtype=None):
  """Coerces a scalar or NumPy array to an np.array.

  Handles Python scalar type promotion according to JAX's rules, not NumPy's
  rules.
  """
  if dtype is None and type(x) in python_scalar_dtypes:
    dtype = _scalar_type_to_dtype(type(x), x)
  return np.asarray(x, dtype)

iinfo = np.iinfo

class finfo(np.finfo):
  __doc__ = np.finfo.__doc__
  _finfo_cache: Dict[np.dtype, np.finfo] = {}
  @staticmethod
  def _bfloat16_finfo():
    def float_to_str(f):
      return "%12.4e" % float(f)

    bfloat16 = _bfloat16_dtype.type
    tiny = float.fromhex("0x1p-126")
    resolution = 0.01
    eps = float.fromhex("0x1p-7")
    epsneg = float.fromhex("0x1p-8")
    max = float.fromhex("0x1.FEp127")

    obj = object.__new__(np.finfo)
    obj.dtype = _bfloat16_dtype
    obj.bits = 16
    obj.eps = bfloat16(eps)
    obj.epsneg = bfloat16(epsneg)
    obj.machep = -7
    obj.negep = -8
    obj.max = bfloat16(max)
    obj.min = bfloat16(-max)
    obj.nexp = 8
    obj.nmant = 7
    obj.iexp = obj.nexp
    obj.precision = 2
    obj.resolution = bfloat16(resolution)
    obj.tiny = bfloat16(tiny)
    obj.machar = None  # np.core.getlimits.MachArLike does not support bfloat16.

    obj._str_tiny = float_to_str(tiny)
    obj._str_max = float_to_str(max)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    return obj

  def __new__(cls, dtype):
    if isinstance(dtype, str) and dtype == 'bfloat16' or dtype == _bfloat16_dtype:
      if _bfloat16_dtype not in cls._finfo_cache:
        cls._finfo_cache[_bfloat16_dtype] = cls._bfloat16_finfo()
      return cls._finfo_cache[_bfloat16_dtype]
    return super().__new__(cls, dtype)

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

# Return the type holding the real part of the input type
def dtype_real(typ):
  if np.issubdtype(typ, np.complexfloating):
    if typ == np.dtype('complex64'):
      return np.dtype('float32')
    elif typ == np.dtype('complex128'):
      return np.dtype('float64')
    else:
      raise TypeError("Unknown complex floating type {}".format(typ))
  else:
    return typ

# Enumeration of all valid JAX types in order.
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
] + _weak_types  # type: ignore[operator]

def _jax_type(dtype, weak_type):
  """Return the jax type for a dtype and weak type."""
  return type(dtype.type(0).item()) if (weak_type and dtype != bool) else dtype

def _dtype_and_weaktype(value):
  """Return a (dtype, weak_type) tuple for the given input."""
  return dtype(value), any(value is typ for typ in _weak_types) or is_weakly_typed(value)

def _type_promotion_lattice():
  """
  Return the type promotion lattice in the form of a DAG.
  This DAG maps each type to its immediately higher type on the lattice.
  """
  b1, u1, u2, u4, u8, i1, i2, i4, i8, bf, f2, f4, f8, c4, c8, i_, f_, c_ = _jax_types
  return {
    b1: [i_],
    u1: [i2, u2], u2: [i4, u4], u4: [i8, u8], u8: [f_],
    i_: [u1, i1], i1: [i2], i2: [i4], i4: [i8], i8: [f_],
    f_: [bf, f2, c_], bf: [f4], f2: [f4], f4: [f8, c4], f8: [c8],
    c_: [c4], c4: [c8], c8: [],
  }

def _make_lattice_upper_bounds():
  lattice = _type_promotion_lattice()
  upper_bounds = {node: {node} for node in lattice}
  for n in lattice:
    while True:
      new_upper_bounds = set().union(*(lattice[b] for b in upper_bounds[n]))
      if n in new_upper_bounds:
        raise ValueError(f"cycle detected in type promotion lattice for node {n}")
      if new_upper_bounds.issubset(upper_bounds[n]):
        break
      upper_bounds[n] |= new_upper_bounds
  return upper_bounds
_lattice_upper_bounds = _make_lattice_upper_bounds()

@functools.lru_cache(512)  # don't use util.memoize because there is no X64 dependence.
def _least_upper_bound(*nodes):
  """Compute the least upper bound of a set of nodes.

  Args:
    nodes: sequence of entries from _jax_types
  Returns:
    the _jax_type representing the least upper bound of the input nodes
      on the promotion lattice.
  """
  # This function computes the least upper bound of a set of nodes N within a partially
  # ordered set defined by the lattice generated above.
  # Given a partially ordered set S, let the set of upper bounds of n ∈ S be
  #   UB(n) ≡ {m ∈ S | n ≤ m}
  # Further, for a set of nodes N ⊆ S, let the set of common upper bounds be given by
  #   CUB(N) ≡ {a ∈ S | ∀ b ∈ N: a ∈ UB(b)}
  # Then the least upper bound of N is defined as
  #   LUB(N) ≡ {c ∈ CUB(N) | ∀ d ∈ CUB(N), c ≤ d}
  # The definition of an upper bound implies that c ≤ d if and only if d ∈ UB(c),
  # so the LUB can be expressed:
  #   LUB(N) = {c ∈ CUB(N) | ∀ d ∈ CUB(N): d ∈ UB(c)}
  # or, equivalently:
  #   LUB(N) = {c ∈ CUB(N) | CUB(N) ⊆ UB(c)}
  # By definition, LUB(N) has a cardinality of 1 for a partially ordered set.
  # Note a potential algorithmic shortcut: from the definition of CUB(N), we have
  #   ∀ c ∈ N: CUB(N) ⊆ UB(c)
  # So if N ∩ CUB(N) is nonempty, if follows that LUB(N) = N ∩ CUB(N).
  N = set(nodes)
  UB = _lattice_upper_bounds
  CUB = set.intersection(*(UB[n] for n in N))
  LUB = (CUB & N) or {c for c in CUB if CUB.issubset(UB[c])}
  if len(LUB) == 1:
    return LUB.pop()
  else:
    raise ValueError(f"{nodes} do not have a unique least upper bound.")

def promote_types(a, b):
  """Returns the type to which a binary operation should cast its arguments.

  For details of JAX's type promotion semantics, see :ref:`type-promotion`.

  Args:
    a: a :class:`numpy.dtype` or a dtype specifier.
    b: a :class:`numpy.dtype` or a dtype specifier.

  Returns:
    A :class:`numpy.dtype` object.
  """
  a = a if any(a is t for t in _weak_types) else np.dtype(a)
  b = b if any(b is t for t in _weak_types) else np.dtype(b)
  return np.dtype(_least_upper_bound(a, b))

def is_weakly_typed(x):
  try:
    return x.aval.weak_type
  except AttributeError:
    return type(x) in _weak_types

def is_python_scalar(x):
  try:
    return x.aval.weak_type and np.ndim(x) == 0
  except AttributeError:
    return type(x) in python_scalar_dtypes

def dtype(x):
  if type(x) in python_scalar_dtypes:
    return python_scalar_dtypes[type(x)]
  return np.result_type(x)

def _lattice_result_type(*args):
  dtypes, weak_types = zip(*(_dtype_and_weaktype(arg) for arg in args))
  if len(dtypes) == 1:
    return dtypes[0], weak_types[0]

  # If all inputs are weakly typed, we compute the bound of the strongly-typed
  # counterparts and apply the weak type at the end. This avoids returning the
  # incorrect result with non-canonical weak types (e.g. weak int16).
  if all(weak_types):
    result_type = _least_upper_bound(*{_jax_type(dtype, False) for dtype in dtypes})
    return dtype(result_type), True
  else:
    result_type = _least_upper_bound(*{_jax_type(d, w) for d, w in zip(dtypes, weak_types)})
    return dtype(result_type), any(result_type is t for t in _weak_types)

def result_type(*args):
  """Convenience function to apply JAX argument dtype promotion."""
  if len(args) == 0:
    raise ValueError("at least one array or dtype is required")
  return canonicalize_dtype(_lattice_result_type(*args)[0])
