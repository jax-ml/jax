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


import builtins
import functools
from typing import (cast, overload, Any, Dict, List, Literal, Optional, Set,
                    Tuple, Type, Union)
import warnings

import ml_dtypes
import numpy as np

from jax._src.config import flags, config
from jax._src.typing import DType, DTypeLike, OpaqueDType

from jax._src import traceback_util
traceback_util.register_exclusion(__file__)

FLAGS = flags.FLAGS

# TODO(frostig,mattjj): achieve this w/ a protocol instead of registry?
opaque_dtypes: Set[OpaqueDType] = set()

def is_opaque_dtype(dtype: Any) -> bool:
  return type(dtype) in opaque_dtypes

# fp8 support
float8_e4m3fn: Type[np.generic] = ml_dtypes.float8_e4m3fn
float8_e5m2: Type[np.generic] = ml_dtypes.float8_e5m2
_float8_e4m3fn_dtype: np.dtype = np.dtype(float8_e4m3fn)
_float8_e5m2_dtype: np.dtype = np.dtype(float8_e5m2)

# bfloat16 support
bfloat16: Type[np.generic] = ml_dtypes.bfloat16
_bfloat16_dtype: np.dtype = np.dtype(bfloat16)

int4: Optional[Type[np.generic]] = None
_int4_dtype: Optional[np.dtype] = None
uint4: Optional[Type[np.generic]] = None
_uint4_dtype: Optional[np.dtype] = None

if hasattr(ml_dtypes, "int4"):
  int4 = ml_dtypes.int4
  uint4 = ml_dtypes.uint4
  _int4_dtype = np.dtype(int4)
  _uint4_dtype = np.dtype(uint4)

_custom_float_dtypes = (_bfloat16_dtype, _float8_e4m3fn_dtype,
                        _float8_e5m2_dtype)

# Default types.
bool_: type = np.bool_
int_: type = np.int32 if config.jax_default_dtype_bits == '32' else np.int64
uint: type = np.uint32 if config.jax_default_dtype_bits == '32' else np.uint64
float_: type = np.float32 if config.jax_default_dtype_bits == '32' else np.float64
complex_: type = np.complex64 if config.jax_default_dtype_bits == '32' else np.complex128
_default_types: Dict[str, type] = {'b': bool_, 'i': int_, 'u': uint, 'f': float_, 'c': complex_}

# Trivial vectorspace datatype needed for tangent values of int/bool primals
float0: np.dtype = np.dtype([('float0', np.void, 0)])

_dtype_to_32bit_dtype: Dict[DType, DType] = {
    np.dtype('int64'): np.dtype('int32'),
    np.dtype('uint64'): np.dtype('uint32'),
    np.dtype('float64'): np.dtype('float32'),
    np.dtype('complex128'): np.dtype('complex64'),
}

# Note: we promote narrow types to float32 here for backward compatibility
# with earlier approaches. We might consider revisiting this, or perhaps
# tying the logic more closely to the type promotion lattice.
_dtype_to_inexact: Dict[DType, DType] = {
    np.dtype(k): np.dtype(v) for k, v in [
        ('bool', 'float32'),
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


def to_complex_dtype(dtype: DTypeLike) -> DType:
  ftype = to_inexact_dtype(dtype)
  if ftype in [np.dtype('float64'), np.dtype('complex128')]:
    return np.dtype('complex128')
  return np.dtype('complex64')


@functools.lru_cache(maxsize=None)
def _canonicalize_dtype(x64_enabled: bool, allow_opaque_dtype: bool, dtype: Any) -> Union[DType, OpaqueDType]:
  if is_opaque_dtype(dtype):
    if not allow_opaque_dtype:
      raise ValueError(f"Internal: canonicalize_dtype called on opaque dtype {dtype} "
                       "with allow_opaque_dtype=False")
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
def canonicalize_dtype(dtype: Any, allow_opaque_dtype: Literal[False] = False) -> DType: ...

@overload
def canonicalize_dtype(dtype: Any, allow_opaque_dtype: bool = False) -> Union[DType, OpaqueDType]: ...

def canonicalize_dtype(dtype: Any, allow_opaque_dtype: bool = False) -> Union[DType, OpaqueDType]:
  """Convert from a dtype to a canonical dtype based on config.x64_enabled."""
  return _canonicalize_dtype(config.x64_enabled, allow_opaque_dtype, dtype)

# Default dtypes corresponding to Python scalars.
python_scalar_dtypes : Dict[type, DType] = {
  bool: np.dtype('bool'),
  int: np.dtype('int64'),
  float: np.dtype('float64'),
  complex: np.dtype('complex128'),
}

def scalar_type_of(x: Any) -> type:
  """Return the scalar type associated with a JAX value."""
  typ = dtype(x)
  if typ == bfloat16:
    return float
  elif typ == float8_e4m3fn:
    return float
  elif typ == float8_e5m2:
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
    raise TypeError(f"Invalid scalar value {x}")


def _scalar_type_to_dtype(typ: type, value: Any = None) -> DType:
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


def coerce_to_array(x: Any, dtype: Optional[DTypeLike] = None) -> np.ndarray:
  """Coerces a scalar or NumPy array to an np.array.

  Handles Python scalar type promotion according to JAX's rules, not NumPy's
  rules.
  """
  if dtype is None and type(x) in python_scalar_dtypes:
    dtype = _scalar_type_to_dtype(type(x), x)
  return np.asarray(x, dtype)

try:
  iinfo = ml_dtypes.iinfo
except AttributeError:
  iinfo = np.iinfo

try:
  finfo = ml_dtypes.finfo
except AttributeError as err:
  _ml_dtypes_version = getattr(ml_dtypes, "__version__", "<unknown>")
  raise ImportError("JAX requires package ml_dtypes>=0.1.0. "
                    f"Installed version is {_ml_dtypes_version}.") from err

def _issubclass(a: Any, b: Any) -> bool:
  """Determines if ``a`` is a subclass of ``b``.

  Similar to issubclass, but returns False instead of an exception if `a` is not
  a class.
  """
  try:
    return issubclass(a, b)
  except TypeError:
    return False

_type_classes = {
    np.generic,
    np.number,
    np.flexible,
    np.character,
    np.integer,
    np.signedinteger,
    np.unsignedinteger,
    np.inexact,
    np.floating,
    np.complexfloating,
}

def _is_typeclass(a: Any) -> bool:
  try:
    return a in _type_classes
  except TypeError:
    return False

def issubdtype(a: DTypeLike, b: DTypeLike) -> bool:
  """Returns True if first argument is a typecode lower/equal in type hierarchy.

  This is like :func:`numpy.issubdtype`, but can handle dtype extensions such as
  :obj:`jax.dtypes.bfloat16`.
  """
  if is_opaque_dtype(a):
    return a == b
  # Canonicalizes all concrete types to np.dtype instances
  a = a if _is_typeclass(a) else np.dtype(a)
  b = b if _is_typeclass(b) else np.dtype(b)
  if isinstance(a, np.dtype):
    if a in _custom_float_dtypes:
      # Avoid implicitly casting list elements below to a dtype.
      if isinstance(b, np.dtype):
        return a == b
      return b in [np.floating, np.inexact, np.number]
    # TODO(phawkins): remove the "_int4_dtype is not None" tests after requiring
    # an ml_dtypes version that has int4 and uint4.
    if _int4_dtype is not None and a == _int4_dtype:
      if isinstance(b, np.dtype):
        return a == b
      return b in [np.signedinteger, np.integer, np.number]
    if _uint4_dtype is not None and a == _uint4_dtype:
      if isinstance(b, np.dtype):
        return a == b
      return b in [np.unsignedinteger, np.integer, np.number]
  return np.issubdtype(a, b)

can_cast = np.can_cast
issubsctype = np.issubsctype

JAXType = Union[type, DType]

# Enumeration of all valid JAX types in order.
_weak_types: List[JAXType] = [int, float, complex]
_bool_types: List[JAXType] = [np.dtype(bool)]
_int_types: List[JAXType]
if int4 is not None:
  _int_types = [
      np.dtype(uint4),
      np.dtype('uint8'),
      np.dtype('uint16'),
      np.dtype('uint32'),
      np.dtype('uint64'),
      np.dtype(int4),
      np.dtype('int8'),
      np.dtype('int16'),
      np.dtype('int32'),
      np.dtype('int64'),
  ]
else:
  _int_types = [
      np.dtype('uint8'),
      np.dtype('uint16'),
      np.dtype('uint32'),
      np.dtype('uint64'),
      np.dtype('int8'),
      np.dtype('int16'),
      np.dtype('int32'),
      np.dtype('int64'),
  ]

_float_types: List[JAXType]
_float_types = [
  np.dtype(float8_e4m3fn),
  np.dtype(float8_e5m2),
  np.dtype(bfloat16),
  np.dtype('float16'),
  np.dtype('float32'),
  np.dtype('float64'),
]
_complex_types: List[JAXType] = [
    np.dtype('complex64'),
    np.dtype('complex128'),
]
_jax_types = _bool_types + _int_types + _float_types + _complex_types
_jax_dtype_set = {float0, *_bool_types, *_int_types, *_float_types, *_complex_types}

def _jax_type(dtype: DType, weak_type: bool) -> JAXType:
  """Return the jax type for a dtype and weak type."""
  if weak_type:
    if dtype == bool:
      return dtype
    if dtype in _custom_float_dtypes:
      return float
    return type(dtype.type(0).item())
  return dtype

def _dtype_and_weaktype(value: Any) -> Tuple[DType, bool]:
  """Return a (dtype, weak_type) tuple for the given input."""
  return dtype(value), any(value is typ for typ in _weak_types) or is_weakly_typed(value)

def _type_promotion_lattice(jax_numpy_dtype_promotion: str) -> Dict[JAXType, List[JAXType]]:
  """
  Return the type promotion lattice in the form of a DAG.
  This DAG maps each type to its immediately higher type on the lattice.
  """
  b1, = _bool_types
  if int4 is not None:
    _uint4, u1, u2, u4, u8, _int4, i1, i2, i4, i8 = _int_types  # pytype: disable=bad-unpacking
  else:
    u1, u2, u4, u8, i1, i2, i4, i8 = _int_types  # pytype: disable=bad-unpacking
  f1_e4m3fn, f1_e5m2, bf, f2, f4, f8 = _float_types
  c4, c8 = _complex_types
  i_, f_, c_ = _weak_types
  if jax_numpy_dtype_promotion == 'standard':
    out: Dict[JAXType, List[JAXType]]
    out = {
      b1: [i_],
      u1: [i2, u2], u2: [i4, u4], u4: [i8, u8], u8: [f_],
      i_: [u1, i1], i1: [i2], i2: [i4], i4: [i8], i8: [f_],
      f_: [f1_e4m3fn, f1_e5m2, bf, f2, c_], f1_e4m3fn: [], f1_e5m2: [], bf: [f4], f2: [f4], f4: [f8, c4], f8: [c8],
      c_: [c4], c4: [c8], c8: [],
    }
    if _int4_dtype is not None:
      out[i_].append(_int4_dtype)
      out[_int4_dtype] = []
    if _uint4_dtype is not None:
      out[i_].append(_uint4_dtype)
      out[_uint4_dtype] = []
    return out
  elif jax_numpy_dtype_promotion == 'strict':
    return {
      i_: [f_] + _int_types,
      f_: [c_] + _float_types,
      c_: _complex_types,
      **{t: [] for t in _jax_types}
    }
  else:
    raise ValueError(
      f"Unexpected value of jax_numpy_dtype_promotion={jax_numpy_dtype_promotion!r}")

def _make_lattice_upper_bounds(jax_numpy_dtype_promotion: str) -> Dict[JAXType, Set[JAXType]]:
  lattice = _type_promotion_lattice(jax_numpy_dtype_promotion)
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

_lattice_upper_bounds: Dict[str, Dict[JAXType, Set[JAXType]]] = {
  'standard': _make_lattice_upper_bounds('standard'),
  'strict': _make_lattice_upper_bounds('strict'),
}

class TypePromotionError(ValueError):
  pass

@functools.lru_cache(512)  # don't use util.memoize because there is no X64 dependence.
def _least_upper_bound(jax_numpy_dtype_promotion: str, *nodes: JAXType) -> JAXType:
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
  UB = _lattice_upper_bounds[jax_numpy_dtype_promotion]
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
    if config.jax_numpy_dtype_promotion == 'strict':
      msg = (
        f"Input dtypes {tuple(str(n) for n in nodes)} have no available implicit dtype "
        "promotion path when jax_numpy_dtype_promotion=strict. Try explicitly casting "
        "inputs to the desired output type, or set jax_numpy_dtype_promotion=standard.")
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

def promote_types(a: DTypeLike, b: DTypeLike) -> DType:
  """Returns the type to which a binary operation should cast its arguments.

  For details of JAX's type promotion semantics, see :ref:`type-promotion`.

  Args:
    a: a :class:`numpy.dtype` or a dtype specifier.
    b: a :class:`numpy.dtype` or a dtype specifier.

  Returns:
    A :class:`numpy.dtype` object.
  """
  # Note: we deliberately avoid `if a in _weak_types` here because we want to check
  # object identity, not object equality, due to the behavior of np.dtype.__eq__
  a_tp = cast(JAXType, a if any(a is t for t in _weak_types) else np.dtype(a))
  b_tp = cast(JAXType, b if any(b is t for t in _weak_types) else np.dtype(b))
  return np.dtype(_least_upper_bound(config.jax_numpy_dtype_promotion, a_tp, b_tp))

def is_weakly_typed(x: Any) -> bool:
  try:
    return x.aval.weak_type
  except AttributeError:
    return type(x) in _weak_types

def is_python_scalar(x: Any) -> bool:
  try:
    return x.aval.weak_type and np.ndim(x) == 0
  except AttributeError:
    return type(x) in python_scalar_dtypes

def check_valid_dtype(dtype: DType) -> None:
  if dtype not in _jax_dtype_set:
    raise TypeError(f"Dtype {dtype} is not a valid JAX array "
                    "type. Only arrays of numeric types are supported by JAX.")

def dtype(x: Any, *, canonicalize: bool = False) -> DType:
  """Return the dtype object for a value or type, optionally canonicalized based on X64 mode."""
  if x is None:
    raise ValueError(f"Invalid argument to dtype: {x}.")
  elif isinstance(x, type) and x in python_scalar_dtypes:
    dt = python_scalar_dtypes[x]
  elif type(x) in python_scalar_dtypes:
    dt = python_scalar_dtypes[type(x)]
  elif is_opaque_dtype(getattr(x, 'dtype', None)):
    dt = x.dtype
  else:
    try:
      dt = np.result_type(x)
    except TypeError as err:
      raise TypeError(f"Cannot determine dtype of {x}") from err
  if dt not in _jax_dtype_set and not is_opaque_dtype(dt):
    raise TypeError(f"Value '{x}' with dtype {dt} is not a valid JAX array "
                    "type. Only arrays of numeric types are supported by JAX.")
  return canonicalize_dtype(dt, allow_opaque_dtype=True) if canonicalize else dt

def _lattice_result_type(*args: Any) -> Tuple[DType, bool]:
  dtypes, weak_types = zip(*(_dtype_and_weaktype(arg) for arg in args))
  if len(dtypes) == 1:
    out_dtype = dtypes[0]
    out_weak_type = weak_types[0]
  elif len(set(dtypes)) == 1 and not all(weak_types):
    # Trivial promotion case. This allows opaque dtypes through.
    out_dtype = dtypes[0]
    out_weak_type = False
  elif all(weak_types) and config.jax_numpy_dtype_promotion != 'strict':
    # If all inputs are weakly typed, we compute the bound of the strongly-typed
    # counterparts and apply the weak type at the end. This avoids returning the
    # incorrect result with non-canonical weak types (e.g. weak int16).
    # TODO(jakevdp): explore removing this special case.
    result_type = _least_upper_bound(config.jax_numpy_dtype_promotion,
                                     *{_jax_type(dtype, False) for dtype in dtypes})
    out_dtype = dtype(result_type)
    out_weak_type = True
  else:
    result_type = _least_upper_bound(config.jax_numpy_dtype_promotion,
                                     *{_jax_type(d, w) for d, w in zip(dtypes, weak_types)})
    out_dtype = dtype(result_type)
    out_weak_type = any(result_type is t for t in _weak_types)
  return out_dtype, (out_dtype != bool_) and out_weak_type

@overload
def result_type(*args: Any, return_weak_type_flag: Literal[True]) -> Tuple[DType, bool]: ...

@overload
def result_type(*args: Any, return_weak_type_flag: Literal[False] = False) -> DType: ...

@overload
def result_type(*args: Any, return_weak_type_flag: bool = False) -> Union[DType, Tuple[DType, bool]]: ...

def result_type(*args: Any, return_weak_type_flag: bool = False) -> Union[DType, Tuple[DType, bool]]:
  """Convenience function to apply JAX argument dtype promotion.

  Args:
    return_weak_type_flag : if True, then return a ``(dtype, weak_type)`` tuple.
      If False, just return `dtype`

  Returns:
    dtype or (dtype, weak_type) depending on the value of the ``return_weak_type`` argument.
  """
  if len(args) == 0:
    raise ValueError("at least one array or dtype is required")
  dtype, weak_type = _lattice_result_type(*(float_ if arg is None else arg for arg in args))
  if weak_type:
    dtype = canonicalize_dtype(
      _default_types['f' if dtype in [_float8_e4m3fn_dtype, _float8_e5m2_dtype, _bfloat16_dtype] else dtype.kind])
  else:
    dtype = canonicalize_dtype(dtype, allow_opaque_dtype=True)
  return (dtype, weak_type) if return_weak_type_flag else dtype

def check_user_dtype_supported(dtype, fun_name=None):
  if is_opaque_dtype(dtype):
    return
  # Avoid using `dtype in [...]` because of numpy dtype equality overloading.
  if isinstance(dtype, type) and dtype in {bool, int, float, builtins.complex}:
    return
  np_dtype = np.dtype(dtype)
  if int4 is not None:
    is_custom_dtype = np_dtype.type in [
        float8_e4m3fn, float8_e5m2, bfloat16, int4, uint4]
  else:
    is_custom_dtype = np_dtype.type in [float8_e4m3fn, float8_e5m2, bfloat16]
  if np_dtype.kind not in "biufc" and not is_custom_dtype:
    msg = f"JAX only supports number and bool dtypes, got dtype {dtype}"
    msg += f" in {fun_name}" if fun_name else ""
    raise TypeError(msg)
  if dtype is not None and np_dtype != canonicalize_dtype(dtype):
    msg = ("Explicitly requested dtype {} {} is not available, "
           "and will be truncated to dtype {}. To enable more dtypes, set the "
           "jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell "
           "environment variable. "
           "See https://github.com/google/jax#current-gotchas for more.")
    fun_name = f"requested in {fun_name}" if fun_name else ""
    truncated_dtype = canonicalize_dtype(dtype).name
    warnings.warn(msg.format(dtype, fun_name, truncated_dtype), stacklevel=3)
