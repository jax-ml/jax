# Copyright 2019 The JAX Authors.
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

from __future__ import annotations

import abc
import dataclasses
import functools
import types
from typing import cast, overload, Any, Callable, Literal, Union
import warnings

import ml_dtypes
import numpy as np

from jax._src import config
from jax._src import literals
from jax._src.typing import Array, DType, DTypeLike
from jax._src.util import set_module, StrictABC

from jax._src import traceback_util
traceback_util.register_exclusion(__file__)

try:
  _ml_dtypes_version = tuple(map(int, ml_dtypes.__version__.split('.')[:3]))
except:
  pass
else:
  if _ml_dtypes_version < (0, 5):
    raise ValueError("JAX requires ml_dtypes version 0.5 or newer; "
                     f"installed version is {ml_dtypes.__version__}.")

export = set_module('jax.dtypes')

@export
class extended(np.generic):
  """Scalar class for extended dtypes.

  This is an abstract class that should never be instantiated, but rather
  exists for the sake of `jnp.issubdtype`.

  Examples:
    >>> from jax import random
    >>> from jax import dtypes
    >>> key = random.key(0)
    >>> jnp.issubdtype(key.dtype, dtypes.extended)
    True
  """


@export
class prng_key(extended):
  """Scalar class for PRNG Key dtypes.

  This is an abstract class that should never be instantiated, but rather
  exists for the sake of `jnp.issubdtype`.

  Examples:
    >>> from jax import random
    >>> from jax import dtypes
    >>> key = random.key(0)
    >>> jnp.issubdtype(key.dtype, dtypes.prng_key)
    True
  """


class ExtendedDType(StrictABC):
  """Abstract Base Class for extended dtypes"""
  @property
  @abc.abstractmethod
  def type(self) -> type: ...


# fp8 support
float8_e3m4: type[np.generic] = ml_dtypes.float8_e3m4
float8_e4m3: type[np.generic] = ml_dtypes.float8_e4m3
float8_e8m0fnu: type[np.generic] = ml_dtypes.float8_e8m0fnu
float8_e4m3b11fnuz: type[np.generic] = ml_dtypes.float8_e4m3b11fnuz
float8_e4m3fn: type[np.generic] = ml_dtypes.float8_e4m3fn
float8_e4m3fnuz: type[np.generic] = ml_dtypes.float8_e4m3fnuz
float8_e5m2: type[np.generic] = ml_dtypes.float8_e5m2
float8_e5m2fnuz: type[np.generic] = ml_dtypes.float8_e5m2fnuz

_float8_e3m4_dtype: np.dtype = np.dtype(float8_e3m4)
_float8_e4m3_dtype: np.dtype = np.dtype(float8_e4m3)
_float8_e8m0fnu_dtype: np.dtype = np.dtype(float8_e8m0fnu)
_float8_e4m3b11fnuz_dtype: np.dtype = np.dtype(float8_e4m3b11fnuz)
_float8_e4m3fn_dtype: np.dtype = np.dtype(float8_e4m3fn)
_float8_e4m3fnuz_dtype: np.dtype = np.dtype(float8_e4m3fnuz)
_float8_e5m2_dtype: np.dtype = np.dtype(float8_e5m2)
_float8_e5m2fnuz_dtype: np.dtype = np.dtype(float8_e5m2fnuz)

# fp4 support
float4_e2m1fn: type[np.generic] = ml_dtypes.float4_e2m1fn

_float4_e2m1fn_dtype: np.dtype = np.dtype(float4_e2m1fn)

def supports_inf(dtype: DTypeLike) -> bool:
  """Return true if the dtype supports infinity, else return False."""
  typ = np.dtype(dtype).type
  if typ in {float8_e4m3b11fnuz, float8_e4m3fn, float8_e4m3fnuz, float8_e5m2fnuz}:
    return False
  return issubdtype(dtype, np.inexact)

# bfloat16 support
bfloat16: type[np.generic] = ml_dtypes.bfloat16
_bfloat16_dtype: np.dtype = np.dtype(bfloat16)

_custom_float_scalar_types = [
    float4_e2m1fn,
    float8_e3m4,
    float8_e4m3,
    float8_e8m0fnu,
    float8_e4m3b11fnuz,
    float8_e4m3fn,
    float8_e4m3fnuz,
    float8_e5m2,
    float8_e5m2fnuz,
    bfloat16,
]
_custom_float_dtypes = [
    _float4_e2m1fn_dtype,
    _float8_e3m4_dtype,
    _float8_e4m3_dtype,
    _float8_e8m0fnu_dtype,
    _float8_e4m3b11fnuz_dtype,
    _float8_e4m3fn_dtype,
    _float8_e4m3fnuz_dtype,
    _float8_e5m2_dtype,
    _float8_e5m2fnuz_dtype,
    _bfloat16_dtype,
]
_float8_dtypes = [
    _float8_e3m4_dtype,
    _float8_e4m3_dtype,
    _float8_e8m0fnu_dtype,
    _float8_e4m3b11fnuz_dtype,
    _float8_e4m3fn_dtype,
    _float8_e4m3fnuz_dtype,
    _float8_e5m2_dtype,
    _float8_e5m2fnuz_dtype,
]

_float4_dtypes: list[np.dtype] = [
    _float4_e2m1fn_dtype,
]

int2: type[np.generic] = ml_dtypes.int2
uint2: type[np.generic] = ml_dtypes.uint2

_int2_dtype: np.dtype = np.dtype(int2)
_uint2_dtype: np.dtype = np.dtype(uint2)

# 4-bit integer support
int4: type[np.generic] = ml_dtypes.int4
uint4: type[np.generic] = ml_dtypes.uint4
_int4_dtype = np.dtype(int4)
_uint4_dtype = np.dtype(uint4)

_intn_dtypes = [
    _int2_dtype,
    _uint2_dtype,
    _int4_dtype,
    _uint4_dtype,
]

# Default types.
bool_ = np.bool_
int_: type[Any]
uint: type[Any]
float_: type[Any]
complex_: type[Any]
if config.default_dtype_bits.value == '32':
  int_ = np.int32
  uint = np.uint32
  float_ = np.float32
  complex_ = np.complex64
else:
  int_ = np.int64
  uint = np.uint64
  float_ = np.float64
  complex_ = np.complex128


# Default dtypes. These are intended to have the same semantics as, say,
# canonicalize_dtype(np.float64), but are preparing for the reduction in the
# number of places we perform dtype canonicalization.


def default_int_dtype() -> DType:
  return (
      np.dtype(np.int64)
      if config.enable_x64.value and config.default_dtype_bits.value == '64'
      else np.dtype(np.int32)
  )


def default_uint_dtype() -> DType:
  return (
      np.dtype(np.uint64)
      if config.enable_x64.value and config.default_dtype_bits.value == '64'
      else np.dtype(np.uint32)
  )


def default_float_dtype() -> DType:
  return (
      np.dtype(np.float64)
      if config.enable_x64.value and config.default_dtype_bits.value == '64'
      else np.dtype(np.float32)
  )


def default_complex_dtype() -> DType:
  return (
      np.dtype(np.complex128)
      if config.enable_x64.value and config.default_dtype_bits.value == '64'
      else np.dtype(np.complex64)
  )


default_types: dict[str, Callable[[], DType]] = {
    'b': lambda: np.dtype(bool),
    'i': default_int_dtype,
    'u': default_uint_dtype,
    'f': default_float_dtype,
    'c': default_complex_dtype,
}

def jax_dtype(obj: DTypeLike | None, *, align: bool = False,
              copy: bool = False) -> DType:
  """Cast an object to a dtype, respecting JAX dtype defaults.

  Arguments mirror those of :func:`numpy.dtype`.
  """
  if obj is None:
    obj = default_float_dtype()
  elif issubdtype(obj, extended):
    return obj  # type: ignore[return-value]
  elif isinstance(obj, type) and (f := _DEFAULT_TYPEMAP.get(obj)) is not None:
    obj = f()
  return np.dtype(obj, align=align, copy=copy)

_DEFAULT_TYPEMAP: dict[type, Callable[[], np.dtype]] = {
  bool: lambda: np.dtype(bool),
  int: default_int_dtype,
  float: default_float_dtype,
  complex: default_complex_dtype,
}

def bit_width(dtype: DTypeLike) -> int:
  """Number of bits per element for the dtype."""
  # Note: we cannot use dtype.itemsize here because this is
  # incorrect for sub-byte integer types.
  if dtype == np.dtype(bool):
    return 8  # physical bit layout for boolean dtype
  elif issubdtype(dtype, np.integer):
    return iinfo(dtype).bits
  elif issubdtype(dtype, np.floating):
    return finfo(dtype).bits
  elif issubdtype(dtype, np.complexfloating):
    return 2 * finfo(dtype).bits
  else:
    raise ValueError(f"unexpected input: {dtype=}")

# Trivial vectorspace datatype needed for tangent values of int/bool primals
float0: np.dtype = np.dtype([('float0', np.void, 0)])

_dtype_to_32bit_dtype: dict[DType, DType] = {
    np.dtype('int64'): np.dtype('int32'),
    np.dtype('uint64'): np.dtype('uint32'),
    np.dtype('float64'): np.dtype('float32'),
    np.dtype('complex128'): np.dtype('complex64'),
}

# Note: we promote narrow types to float32 here for backward compatibility
# with earlier approaches. We might consider revisiting this, or perhaps
# tying the logic more closely to the type promotion lattice.
_dtype_to_inexact: dict[DType, DType] = {
    np.dtype(k): np.dtype(v) for k, v in [
        ('bool', 'float32'),
        ('uint4', 'float32'), ('int4', 'float32'),
        ('uint8', 'float32'), ('int8', 'float32'),
        ('uint16', 'float32'), ('int16', 'float32'),
        ('uint32', 'float32'), ('int32', 'float32'),
        ('uint64', 'float64'), ('int64', 'float64')
    ]
}

def to_numeric_dtype(dtype: DTypeLike) -> DType:
  """Promotes a dtype into an numeric dtype, if it is not already one."""
  dtype_ = np.dtype(dtype)
  return np.dtype('int32') if dtype_ == np.dtype('bool') else dtype_


def to_inexact_dtype(dtype: DTypeLike) -> DType:
  """Promotes a dtype into an inexact dtype, if it is not already one."""
  dtype_ = np.dtype(dtype)
  return _dtype_to_inexact.get(dtype_, dtype_)


def to_floating_dtype(dtype: DTypeLike) -> DType:
  """Promotes a dtype to a non-complex floating dtype."""
  dtype_ = np.dtype(dtype)
  return finfo(_dtype_to_inexact.get(dtype_, dtype_)).dtype


def to_complex_dtype(dtype: DTypeLike) -> DType:
  ftype = to_inexact_dtype(dtype)
  if ftype in [np.dtype('float64'), np.dtype('complex128')]:
    return np.dtype('complex128')
  return np.dtype('complex64')


@functools.cache
def _canonicalize_dtype(x64_enabled: bool, allow_extended_dtype: bool, dtype: Any) -> DType | ExtendedDType:
  if issubdtype(dtype, extended):
    if not allow_extended_dtype:
      raise ValueError(f"Internal: canonicalize_dtype called on extended dtype {dtype} "
                       "with allow_extended_dtype=False")
    return dtype
  try:
    dtype_ = np.dtype(dtype)
  except TypeError as e:
    raise TypeError(f'dtype {dtype!r} not understood') from e

  if x64_enabled:
    return dtype_
  else:
    return _dtype_to_32bit_dtype.get(dtype_, dtype_)

@overload
def canonicalize_dtype(dtype: Any, allow_extended_dtype: Literal[False] = False) -> DType: ...

@overload
def canonicalize_dtype(dtype: Any, allow_extended_dtype: bool = False) -> DType | ExtendedDType: ...

@export
def canonicalize_dtype(dtype: Any, allow_extended_dtype: bool = False) -> DType | ExtendedDType:
  """Convert from a dtype to a canonical dtype based on config.x64_enabled."""
  return _canonicalize_dtype(config.enable_x64.value, allow_extended_dtype, dtype)  # pytype: disable=bad-return-type

class InvalidInputException(Exception):
  pass

canonicalize_value_handlers: dict[Any, Callable] = {}


# TODO(mattjj): try to remove this canonicalize_dtype stuff
def canonicalize_value(x):
  typ = type(x)
  handler = canonicalize_value_handlers.get(typ)
  if handler:
    return handler(x)
  for typ in typ.__mro__:
    handler = canonicalize_value_handlers.get(typ)
    if handler:
      return handler(x)
  if hasattr(x, '__jax_array__'):
    raise ValueError(
        'Triggering __jax_array__() during abstractification is no longer'
        ' supported. To avoid this error, either explicitly convert your object'
        ' using jax.numpy.array(), or register your object as a pytree.'
    )
  raise InvalidInputException(
      f"Argument '{x}' of type {type(x)} is not a valid JAX type."
  )


# The list of all known Python scalar types.
python_scalar_types: set[type] = {bool, int, float, complex}

# Default dtypes corresponding to Python scalars.
python_scalar_types_to_dtypes: dict[type, DType] = {
  bool: np.dtype('bool'),
  int: np.dtype('int64'),
  float: np.dtype('float64'),
  complex: np.dtype('complex128'),
}

@export
def scalar_type_of(x: Any) -> type:
  """Return the scalar type associated with a JAX value."""
  typ = dtype(x)
  if typ in _custom_float_dtypes:
    return float
  elif typ in _intn_dtypes:
    return int
  elif np.issubdtype(typ, np.bool_):
    return bool
  elif np.issubdtype(typ, np.integer):
    return int
  elif np.issubdtype(typ, np.floating):
    return float
  elif np.issubdtype(typ, np.complexfloating):
    return complex
  else:
    raise TypeError(f"Invalid scalar value {x}")


def scalar_type_to_dtype(typ: type, value: Any = None) -> DType:
  """Return the numpy dtype for the given scalar type.

  Raises
  ------
  OverflowError: if `typ` is `int` and the value is too large for int64.

  Examples
  --------
  >>> scalar_type_to_dtype(int)
  dtype('int32')
  >>> scalar_type_to_dtype(float)
  dtype('float32')
  >>> scalar_type_to_dtype(complex)
  dtype('complex64')
  >>> scalar_type_to_dtype(int)
  dtype('int32')
  >>> scalar_type_to_dtype(int, 0)
  dtype('int32')
  >>> scalar_type_to_dtype(int, 1 << 63)  # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
  OverflowError: Python int 9223372036854775808 too large to convert to int32
  """
  dtype = canonicalize_dtype(python_scalar_types_to_dtypes[typ])
  if typ is int and value is not None:
    iinfo = np.iinfo(dtype)
    if value < iinfo.min or value > iinfo.max:
      raise OverflowError(f"Python int {value} too large to convert to {dtype}")
  return dtype


def coerce_to_array(x: Any, dtype: DTypeLike | None = None) -> np.ndarray:
  """Coerces a scalar or NumPy array to an np.array.

  Handles Python scalar type promotion according to JAX's rules, not NumPy's
  rules.
  """
  if dtype is None and type(x) in python_scalar_types:
    dtype = scalar_type_to_dtype(type(x), x)
  return np.asarray(x, dtype)

iinfo = ml_dtypes.iinfo
finfo = ml_dtypes.finfo

def _issubclass(a: Any, b: Any) -> bool:
  """Determines if ``a`` is a subclass of ``b``.

  Similar to issubclass, but returns False instead of an exception if `a` is not
  a class.
  """
  try:
    return issubclass(a, b)
  except TypeError:
    return False


_types_for_issubdtype = (type, np.dtype, ExtendedDType)

# TODO(jakevdp): consider whether to disallow None here. We allow it
# because np.issubdtype allows it (and treats it as equivalent to float64).
@set_module('jax.numpy')
def issubdtype(a: DTypeLike | ExtendedDType | None,
               b: DTypeLike | ExtendedDType | None) -> bool:
  """Returns True if first argument is a typecode lower/equal in type hierarchy.

  This is like :func:`numpy.issubdtype`, but can handle dtype extensions such as
  :obj:`jax.dtypes.bfloat16` and `jax.dtypes.prng_key`.
  """
  # Main departures from np.issubdtype are:
  # - "extended" dtypes (like prng key types) are not normal numpy dtypes, so we
  #   need to handle them specifically. However, their scalar types do conform to
  #   the numpy scalar type hierarchy.
  # - custom dtypes (like bfloat16, int4, etc.) are normal numpy dtypes, but they
  #   don't conform to the standard numpy type hierarchy (e.g. the bfloat16 scalar
  #   type is not a subclass of np.floating) so we must also handle these specially.

  # We cannot use the cached version directly for all inputs, because some may be
  # unhashable (e.g. custom objects with a dtype attribute). The following check is
  # fast and covers the majority of calls to this function within JAX library code.
  return _issubdtype_cached(
    a if isinstance(a, _types_for_issubdtype) else np.dtype(a),  # type: ignore[arg-type]
    b if isinstance(b, _types_for_issubdtype) else np.dtype(b),  # type: ignore[arg-type]
  )


@functools.lru_cache(512)  # don't use util.memoize because there is no X64 dependence.
def _issubdtype_cached(a: type | np.dtype | ExtendedDType,
                       b: type | np.dtype | ExtendedDType) -> bool:
  # First handle extended dtypes, which require their own logic.
  a_is_type = isinstance(a, type)
  b_is_type = isinstance(b, type)
  if b_is_type and _issubclass(b, extended):
    if isinstance(a, ExtendedDType):
      return _issubclass(a.type, b)
    if a_is_type and _issubclass(a, np.generic):
      return _issubclass(a, b)
    return _issubclass(np.dtype(a).type, b)
  if isinstance(b, ExtendedDType):
    return isinstance(a, ExtendedDType) and a == b
  if isinstance(a, ExtendedDType):
    a = a.type
    a_is_type = isinstance(a, type)

  # For all others, normalize inputs to scalar types.
  a_sctype = a if a_is_type and _issubclass(a, np.generic) else np.dtype(a).type
  b_sctype = b if b_is_type and _issubclass(b, np.generic) else np.dtype(b).type

  # Now do special handling of custom float and int types, as they don't conform
  # to the normal scalar type hierarchy.
  if a_sctype in _custom_float_scalar_types:
    return b_sctype in {a_sctype, np.floating, np.inexact, np.number, np.generic}
  if a_sctype in [int2, int4]:
    return b_sctype in {a_sctype, np.signedinteger, np.integer, np.number, np.generic}
  if a_sctype in [uint2, uint4]:
    return b_sctype in {a_sctype, np.unsignedinteger, np.integer, np.number, np.generic}

  # Otherwise, fall back to numpy.issubdtype
  return bool(np.issubdtype(a_sctype, b_sctype))

can_cast = np.can_cast

JAXType = Union[type, DType]

# Enumeration of all valid JAX types in order.
_weak_types: list[JAXType] = [int, float, complex]
_bool_types: list[JAXType] = [np.dtype(bool)]
_signed_types: list[JAXType]
_unsigned_types: list[JAXType]
_int_types: list[JAXType]
_unsigned_types = [
    np.dtype(uint2),
    np.dtype(uint4),
    np.dtype('uint8'),
    np.dtype('uint16'),
    np.dtype('uint32'),
    np.dtype('uint64'),
]
_signed_types = [
    np.dtype(int2),
    np.dtype(int4),
    np.dtype('int8'),
    np.dtype('int16'),
    np.dtype('int32'),
    np.dtype('int64'),
]

_int_types = _unsigned_types + _signed_types

_float_types: list[JAXType] = [
    *_custom_float_dtypes,
    np.dtype('float16'),
    np.dtype('float32'),
    np.dtype('float64'),
]
_complex_types: list[JAXType] = [
    np.dtype('complex64'),
    np.dtype('complex128'),
]


# We add the StringDType only to `_jax_dtype_set` but not to `_jax_types` and
# `_dtype_kinds`. This is because, in spite of a very similar sounding name,
# `_jax_types` is only meant for the promotion related logic, and StringDType
# does not participate in promotions at the moment. Similarly, `_dtype_kinds` is
# only meant for the `jnp.isdtype` and we want to be conservative and not allow
# StringDType to be used in there.
_string_types: list[JAXType] = []
if hasattr(np.dtypes, 'StringDType'):
  _string_types: list[JAXType] = [np.dtypes.StringDType()]  # type: ignore

_jax_dtype_set = {
    float0,
    *_bool_types,
    *_int_types,
    *_float_types,
    *_complex_types,
    *_string_types,
}

_jax_types = (_bool_types + _int_types + _float_types + _complex_types)

_dtype_kinds: dict[str, set] = {
    'bool': {*_bool_types},
    'signed integer': {*_signed_types},
    'unsigned integer': {*_unsigned_types},
    'integral': {*_signed_types, *_unsigned_types},
    'real floating': {*_float_types},
    'complex floating': {*_complex_types},
    'numeric': {*_signed_types, *_unsigned_types, *_float_types, *_complex_types},
}


@set_module('jax.numpy')
def isdtype(dtype: DTypeLike, kind: str | DTypeLike | tuple[str | DTypeLike, ...]) -> bool:
  """Returns a boolean indicating whether a provided dtype is of a specified kind.

  Args:
    dtype : the input dtype
    kind : the data type kind.
      If ``kind`` is dtype-like, return ``dtype = kind``.
      If ``kind`` is a string, then return True if the dtype is in the specified category:

      - ``'bool'``: ``{bool}``
      - ``'signed integer'``: ``{int4, int8, int16, int32, int64}``
      - ``'unsigned integer'``: ``{uint4, uint8, uint16, uint32, uint64}``
      - ``'integral'``: shorthand for ``('signed integer', 'unsigned integer')``
      - ``'real floating'``: ``{float8_*, float16, bfloat16, float32, float64}``
      - ``'complex floating'``: ``{complex64, complex128}``
      - ``'numeric'``: shorthand for ``('integral', 'real floating', 'complex floating')``

      If ``kind`` is a tuple, then return True if dtype matches any entry of the tuple.

  Returns:
    True or False
  """
  the_dtype = np.dtype(dtype)
  kind_tuple: tuple[str | DTypeLike, ...] = (
    kind if isinstance(kind, tuple) else (kind,)
  )
  options: set[DType] = set()
  for kind in kind_tuple:
    if isinstance(kind, str) and kind in _dtype_kinds:
      options.update(_dtype_kinds[kind])
      continue
    try:
      _dtype = np.dtype(kind)
    except TypeError as e:
      if isinstance(kind, str):
        raise ValueError(
          f"Unrecognized {kind=} expected one of {list(_dtype_kinds.keys())}, "
          "or a compatible input for jnp.dtype()")
      raise TypeError(
        f"Expected kind to be a dtype, string, or tuple; got {kind=}"
      ) from e
    options.add(_dtype)
  return the_dtype in options


def _jax_type(dtype: DType, weak_type: bool) -> JAXType:
  """Return the jax type for a dtype and weak type."""
  if weak_type:
    if dtype == bool:
      return dtype
    if dtype in _custom_float_dtypes:
      return float
    return type(dtype.type(0).item())
  return dtype

def _dtype_and_weaktype(value: Any) -> tuple[DType, bool]:
  """Return a (dtype, weak_type) tuple for the given input."""
  return dtype(value), any(value is typ for typ in _weak_types) or is_weakly_typed(value)

def _type_promotion_lattice(strict: bool, x64: bool) -> dict[JAXType, list[JAXType]]:
  """
  Return the type promotion lattice in the form of a DAG.
  This DAG maps each type to its immediately higher types on the lattice.

  Args:
    strict: use strict promotion lattice?
    x64: allow promotions that form x64 types from non-x64 inputs?
  """
  b1, = _bool_types
  u2, u4, u8, u16, u32, u64, i2, i4, i8, i16, i32, i64 = _int_types
  *small_float_types, bf16, f16, f32, f64 = _float_types
  c64, c128 = _complex_types
  i_, f_, c_ = _weak_types
  if not strict:
    out: dict[JAXType, list[JAXType]] = {
        b1: [i_],
        i_: [u8, u2, u4, i8, i2, i4],
        u2: [],
        u4: [],
        u8: [i16, u16],
        u16: [i32, u32],
        u32: [i64, u64],
        u64: [f_],
        i2: [],
        i4: [],
        i8: [i16],
        i16: [i32],
        i32: [i64],
        i64: [f_],
        f_: [*small_float_types, bf16, f16, c_],
        **{t: [] for t in small_float_types},
        bf16: [f32],
        f16: [f32],
        f32: [f64, c64],
        f64: [c128],
        c_: [c64],
        c64: [c128],
        c128: [],
    }
    # If x64 mode is not enabled, then we want to avoid any promotions that form
    # 64-bit types from non-64-bit inputs. There's only one of these in the
    # entire promotion lattice, namely u4xi4->i8, which we can avoid by
    # replacing it with u4xi4->i4.
    if not x64:
      out[u32] = [i32, u64]
    return out
  else:
    return {
      i_: [f_] + _int_types,
      f_: [c_] + _float_types,
      c_: _complex_types,
      **{t: [] for t in _jax_types}
    }

def _make_lattice_upper_bounds(strict: bool, x64: bool) -> dict[JAXType, set[JAXType]]:
  lattice = _type_promotion_lattice(strict, x64)
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

_standard_x64_lattice_ubs = _make_lattice_upper_bounds(strict=False, x64=True)
_standard_x32_lattice_ubs = _make_lattice_upper_bounds(strict=False, x64=False)
_strict_lattice_ubs = _make_lattice_upper_bounds(strict=True, x64=True)

class TypePromotionError(ValueError):
  pass

# We don't use util.memoize because there is no implicit X64 dependence.
@functools.lru_cache(512)
def _least_upper_bound(jax_numpy_dtype_promotion: config.NumpyDtypePromotion,
                       x64: bool, *nodes: JAXType) -> JAXType:
  """Compute the least upper bound of a set of nodes.

  Args:
    nodes: sequence of entries from _jax_types + _weak_types
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
  if jax_numpy_dtype_promotion == config.NumpyDtypePromotion.STRICT:
    UB = _strict_lattice_ubs
  elif jax_numpy_dtype_promotion == config.NumpyDtypePromotion.STANDARD:
    if x64:
      UB = _standard_x64_lattice_ubs
    else:
      UB = _standard_x32_lattice_ubs
  else:
    raise ValueError(
      f"Unexpected value of jax_numpy_dtype_promotion={jax_numpy_dtype_promotion!r}")
  try:
    bounds = [UB[n] for n in N]
  except KeyError:
    dtype = next(n for n in N if n not in UB)
    raise ValueError(f"{dtype=} is not a valid dtype for JAX type promotion.")
  CUB = set.intersection(*bounds)
  LUB = (CUB & N) or {c for c in CUB if CUB.issubset(UB[c])}
  if len(LUB) == 1:
    return LUB.pop()
  elif len(LUB) == 0:
    if config.numpy_dtype_promotion.value == config.NumpyDtypePromotion.STRICT:
      msg = (
        f"Input dtypes {tuple(str(n) for n in nodes)} have no available implicit dtype "
        "promotion path when jax_numpy_dtype_promotion=strict. Try explicitly casting "
        "inputs to the desired output type, or set jax_numpy_dtype_promotion=standard.")
    elif any(n in _float8_dtypes for n in nodes):
      msg = (
        f"Input dtypes {tuple(str(n) for n in nodes)} have no available implicit dtype "
        "promotion path. To avoid unintended promotion, 8-bit floats do not support "
        "implicit promotion. If you'd like your inputs to be promoted to another type, "
        "you can do so explicitly using e.g. x.astype('float32')")
    elif any(n in _float4_dtypes for n in nodes):
      msg = (
        f"Input dtypes {tuple(str(n) for n in nodes)} have no available implicit dtype "
        "promotion path. To avoid unintended promotion, 4-bit floats do not support "
        "implicit promotion. If you'd like your inputs to be promoted to another type, "
        "you can do so explicitly using e.g. x.astype('float32')")
    elif any(n in _intn_dtypes for n in nodes):
      msg = (
        f"Input dtypes {tuple(str(n) for n in nodes)} have no available implicit dtype "
        "promotion path. To avoid unintended promotion, 2-bit and 4-bit integers do not "
        "support implicit promotion. If you'd like your inputs to be promoted to another "
        "type, you can do so explicitly using e.g. x.astype('int32')")
    else:
      msg = (
        f"Input dtypes {tuple(str(n) for n in nodes)} have no available implicit dtype "
        "promotion path. Try explicitly casting inputs to the desired output type.")
    raise TypePromotionError(msg)
  else:
    # If we get here, it means the lattice is ill-formed.
    raise TypePromotionError(
      f"Internal Type Promotion error: {nodes} do not have a unique least upper bound "
      f"on the specified lattice; options are {LUB}. This is an unexpected error in "
      "JAX's internal logic; please report it to the JAX maintainers."
    )

@set_module('jax.numpy')
def promote_types(a: DTypeLike, b: DTypeLike) -> DType:
  """Returns the type to which a binary operation should cast its arguments.

  JAX implementation of :func:`numpy.promote_types`. For details of JAX's
  type promotion semantics, see :ref:`type-promotion`.

  Args:
    a: a :class:`numpy.dtype` or a dtype specifier.
    b: a :class:`numpy.dtype` or a dtype specifier.

  Returns:
    A :class:`numpy.dtype` object.

  Examples:
    Type specifiers may be strings, dtypes, or scalar types, and the return
    value is always a dtype:

    >>> jnp.promote_types('int32', 'float32')  # strings
    dtype('float32')
    >>> jnp.promote_types(jnp.dtype('int32'), jnp.dtype('float32'))  # dtypes
    dtype('float32')
    >>> jnp.promote_types(jnp.int32, jnp.float32)  # scalar types
    dtype('float32')

    Built-in scalar types (:type:`int`, :type:`float`, or :type:`complex`) are
    treated as weakly-typed and will not change the bit width of a strongly-typed
    counterpart (see discussion in :ref:`type-promotion`):

    >>> jnp.promote_types('uint8', int)
    dtype('uint8')
    >>> jnp.promote_types('float16', float)
    dtype('float16')

    This differs from the NumPy version of this function, which treats built-in scalar
    types as equivalent to 64-bit types:

    >>> import numpy
    >>> numpy.promote_types('uint8', int)
    dtype('int64')
    >>> numpy.promote_types('float16', float)
    dtype('float64')
  """
  # Note: we deliberately avoid `if a in _weak_types` here because we want to check
  # object identity, not object equality, due to the behavior of np.dtype.__eq__
  a_tp = cast(JAXType, a if any(a is t for t in _weak_types) else np.dtype(a))
  b_tp = cast(JAXType, b if any(b is t for t in _weak_types) else np.dtype(b))
  return np.dtype(_least_upper_bound(
      config.numpy_dtype_promotion.value, config.enable_x64.value, a_tp, b_tp))


def register_weak_scalar_type(typ: type):
  """Register a scalar type as a weak type."""
  _registered_weak_types.add(typ)

_registered_weak_types: set[JAXType] = {
    literals.TypedInt,
    literals.TypedFloat,
    literals.TypedComplex,
}


def is_weakly_typed(x: Any) -> bool:
  if type(x) in _weak_types or type(x) in _registered_weak_types:
    return True
  if isinstance(x, literals.TypedNdArray):
    return x.weak_type
  try:
    return x.aval.weak_type
  except AttributeError:
    return False

def is_python_scalar(x: Any) -> bool:
  try:
    return x.aval.weak_type and np.ndim(x) == 0
  except AttributeError:
    return type(x) in python_scalar_types

def check_valid_dtype(dtype: DType) -> None:
  if dtype not in _jax_dtype_set:
    raise TypeError(f"Dtype {dtype} is not a valid JAX array "
                    "type. Only arrays of numeric types are supported by JAX.")

def _maybe_canonicalize_explicit_dtype(dtype: DType, fun_name: str) -> DType:
  "Canonicalizes explicitly requested dtypes, per explicit_x64_dtypes."
  allow = config.explicit_x64_dtypes.value
  if allow == config.ExplicitX64Mode.ALLOW or config.enable_x64.value:
    return dtype
  canonical_dtype = canonicalize_dtype(dtype)
  if canonical_dtype == dtype:
    return dtype
  fun_name = f" requested in {fun_name}" if fun_name else ""
  if allow == config.ExplicitX64Mode.ERROR:
    msg = ("Explicitly requested dtype {}{} is not available. To enable more "
           "dtypes, set the jax_enable_x64 or allow_explicit_x64_dtypes "
           "configuration options."
          "See https://github.com/jax-ml/jax#current-gotchas for more.")
    msg = msg.format(dtype, fun_name, canonical_dtype.name)
    raise ValueError(msg)
  else:  # WARN
    msg = ("Explicitly requested dtype {}{} is not available, "
          "and will be truncated to dtype {}. To enable more dtypes, set the "
          "jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell "
          "environment variable. "
          "See https://github.com/jax-ml/jax#current-gotchas for more.")
    msg = msg.format(dtype, fun_name, canonical_dtype.name)
    warnings.warn(msg, stacklevel=4)
    return canonical_dtype


_types_whose_dtype_should_not_be_canonicalized = (
    Array,
    literals.TypedNdArray,
    literals.TypedInt,
    literals.TypedFloat,
    literals.TypedComplex,
)

def dtype(x: Any) -> DType:
  """Return the dtype object for a user-provided value or type.

  Python scalars, Python scalar types, NumPy scalar type, NumPy dtypes, and
  non-JAX arrays will have their dtypes canonicalized.

  This is not the function you want to call for an internally-derived dtype,
  since it will raise an error if explicit x64 types are observed in a non-x64
  mode. You should use this function for user-derived types or values only.

  Note: this is not the same function as jax.numpy.dtype, which simply aliases
  numpy.dtype."""
  # TODO(phawkins): in the future, we would like to:
  # - return the default dtype for Python scalar types and values
  # - canonicalize NumPy array and scalar types
  # - return NumPy dtypes as-is, uncanonicalized.
  if x is None:
    raise ValueError(f"Invalid argument to dtype: {x}.")
  if isinstance(x, type):
    # Python scalar types, e.g., int, float
    if (dt := python_scalar_types_to_dtypes.get(x)) is not None:
      return canonicalize_dtype(dt)

    # Numpy scalar types, e.g., np.int32, np.float32
    if _issubclass(x, np.generic):
      dt = np.dtype(x)
      return _maybe_canonicalize_explicit_dtype(dt, "dtype")

  # Python scalar values, e.g., int(3), float(3.14)
  elif (dt := python_scalar_types_to_dtypes.get(type(x))) is not None:
    return canonicalize_dtype(dt)
  # Jax Arrays, literal arrays, and scalars.
  # We intentionally do not canonicalize these types: once we've formed an x64
  # value, that is something we respect irrespective of the x64 mode.
  elif isinstance(x, _types_whose_dtype_should_not_be_canonicalized):
    return x.dtype

  if isinstance(x, str):
    x = np.dtype(x)

  if isinstance(x, np.dtype):
    if x not in _jax_dtype_set and not issubdtype(x, extended):
      raise TypeError(f"Value '{x}' with dtype {dt} is not a valid JAX array "
                      "type. Only arrays of numeric types are supported by JAX.")
    return _maybe_canonicalize_explicit_dtype(x, "dtype")

  if issubdtype(getattr(x, 'dtype', None), extended):
    dt = x.dtype
  else:
    try:
      dt = np.result_type(x)
    except TypeError as err:
      raise TypeError(f"Cannot determine dtype of {x}") from err
  if dt not in _jax_dtype_set and not issubdtype(dt, extended):
    raise TypeError(f"Value '{x}' with dtype {dt} is not a valid JAX array "
                    "type. Only arrays of numeric types are supported by JAX.")
  # TODO(jakevdp): fix return type annotation and remove this ignore.
  return canonicalize_dtype(dt, allow_extended_dtype=True)  # type: ignore[return-value]

def lattice_result_type(*args: Any) -> tuple[DType, bool]:
  dtypes, weak_types = zip(*(_dtype_and_weaktype(arg) for arg in args))
  if len(dtypes) == 1:
    out_dtype = dtypes[0]
    out_weak_type = weak_types[0]
  elif len(set(dtypes)) == 1 and not all(weak_types):
    # Trivial promotion case. This allows extended dtypes through.
    out_dtype = dtypes[0]
    out_weak_type = False
  elif all(weak_types) and config.numpy_dtype_promotion.value != config.NumpyDtypePromotion.STRICT:
    # If all inputs are weakly typed, we compute the bound of the strongly-typed
    # counterparts and apply the weak type at the end. This avoids returning the
    # incorrect result with non-canonical weak types (e.g. weak int16).
    # TODO(jakevdp): explore removing this special case.
    result_type = _least_upper_bound(
        config.numpy_dtype_promotion.value, config.enable_x64.value,
        *{_jax_type(dtype, False) for dtype in dtypes})
    # We do not call dtype(result_type) on dtypes here because we do not want to
    # error on explicit x64 types in a non-x64 mode.
    out_dtype = result_type if isinstance(result_type, DType) else dtype(result_type)
    out_weak_type = True
  else:
    result_type = _least_upper_bound(
        config.numpy_dtype_promotion.value, config.enable_x64.value,
        *{_jax_type(d, w) for d, w in zip(dtypes, weak_types)})
    # We do not call dtype(result_type) on dtypes here because we do not want to
    # error on explicit x64 types in a non-x64 mode.
    out_dtype = result_type if isinstance(result_type, DType) else dtype(result_type)
    out_weak_type = any(result_type is t for t in _weak_types)
  return out_dtype, (out_dtype != bool_) and out_weak_type

@overload
def result_type(*args: Any, return_weak_type_flag: Literal[True]) -> tuple[DType, bool]: ...

@overload
def result_type(*args: Any, return_weak_type_flag: Literal[False] = False) -> DType: ...

@overload
def result_type(*args: Any, return_weak_type_flag: bool = False) -> DType | tuple[DType, bool]: ...

@export
def result_type(*args: Any, return_weak_type_flag: bool = False) -> DType | tuple[DType, bool]:
  """Convenience function to apply JAX argument dtype promotion.

  Args:
    return_weak_type_flag : if True, then return a ``(dtype, weak_type)`` tuple.
      If False, just return `dtype`

  Returns:
    dtype or (dtype, weak_type) depending on the value of the ``return_weak_type`` argument.
  """
  if len(args) == 0:
    raise ValueError("at least one array or dtype is required")
  dtype: DType | ExtendedDType
  dtype, weak_type = lattice_result_type(*(default_float_dtype() if arg is None else arg for arg in args))
  if weak_type:
    dtype = default_types['f' if dtype in _custom_float_dtypes else dtype.kind]()
  # TODO(jakevdp): fix return type annotation and remove this ignore.
  return (dtype, weak_type) if return_weak_type_flag else dtype  # type: ignore[return-value]

def check_and_canonicalize_user_dtype(dtype, fun_name=None) -> DType:
  """Checks validity of a user-provided dtype, and returns its canonical form.

  For Python scalar types this function returns the corresponding default dtype.
  """
  if dtype is None:
    raise ValueError("dtype must be specified.")
  if isinstance(dtype, Array):
    raise ValueError("Passing an array as a dtype argument is no longer "
                     "supported; instead of dtype=arr use dtype=arr.dtype.")
  if issubdtype(dtype, extended):
    return dtype
  # Avoid using `dtype in [...]` because of numpy dtype equality overloading.
  if isinstance(dtype, type) and (f := _DEFAULT_TYPEMAP.get(dtype)) is not None:
    return f()
  np_dtype = np.dtype(dtype)
  if np_dtype not in _jax_dtype_set:
    msg = (
        f'JAX only supports number, bool, and string dtypes, got dtype {dtype}'
    )
    msg += f" in {fun_name}" if fun_name else ""
    raise TypeError(msg)
  return _maybe_canonicalize_explicit_dtype(np_dtype, fun_name)

def safe_to_cast(input_dtype_or_value: Any,
                 output_dtype_or_value: Any) -> bool:
  """Check if a dtype/value is safe to cast to another dtype/value

  Args:
    input_dtype_or_value: a dtype or value (to be passed to result_type)
      representing the source dtype.
    output_dtype_or_value: a dtype or value (to be passed to result_type)
      representing the target dtype.

  Returns:
    boolean representing whether the values are safe to cast according to
    default type promotion semantics.

  Raises:
    TypePromotionError: if the inputs have differing types and no type promotion
    path under the current jax_numpy_dtype_promotion setting.

  Examples:

    >>> safe_to_cast('int16', 'float32')
    True
    >>> safe_to_cast('float32', 'int16')
    False
    >>> safe_to_cast('float32', 'complex64')
    True
    >>> safe_to_cast('complex64', 'float32')
    False
  """
  input_dtype = dtype(input_dtype_or_value)
  output_dtype = dtype(output_dtype_or_value)
  if input_dtype == output_dtype:
    return True
  # We deliberately use output_dtype rather than output_dtype_or_value here:
  # this effectively treats the output dtype as always strongly-typed.
  return result_type(input_dtype_or_value, output_dtype) == output_dtype

def primal_tangent_dtype(primal_dtype, tangent_dtype,
                         name: str | None = None) -> ExtendedDType:
  primal_dtype, tangent_dtype = map(dtype, (primal_dtype, tangent_dtype))
  name_ = name or (f'PrimalTangentDType{{{short_dtype_name(primal_dtype)}'
                   f'/{short_dtype_name(tangent_dtype)}}}')
  rules = types.SimpleNamespace(
      physical_element_aval=
      lambda dtype: types.SimpleNamespace(shape=(), dtype=primal_dtype),
      tangent_dtype=lambda dtype: tangent_dtype,
      allow_conversion=True)

  class primal_tangent_dtype_scalar(extended): ...

  @dataclasses.dataclass(frozen=True)
  class PrimalTangentDType(ExtendedDType):
    name = name_
    _rules = rules
    type = primal_tangent_dtype_scalar
    __repr__ = lambda _: name_

  return PrimalTangentDType()

@functools.cache
def short_dtype_name(dtype) -> str:
  if isinstance(dtype, ExtendedDType):
    return str(dtype)
  else:
    return (dtype.name.replace('float', 'f').replace('uint'   , 'u')
                      .replace('int'  , 'i').replace('complex', 'c'))


def is_string_dtype(dtype: DTypeLike | None) -> bool:
  return dtype in _string_types
