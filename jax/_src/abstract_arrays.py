# Copyright 2018 The JAX Authors.
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

from __future__ import annotations

import numpy as np
import threading
from typing import TYPE_CHECKING
import weakref

from jax._src import config
from jax._src import core
from jax._src import literals
from jax._src import dtypes
from jax._src.lib import jaxlib_extension_version
from jax._src.lib import weakref_lru_cache

from jax._src import traceback_util
traceback_util.register_exclusion(__file__)

ShapedArray = core.ShapedArray
AbstractToken = core.AbstractToken
abstract_token = core.abstract_token
canonicalize_shape = core.canonicalize_shape

numpy_scalar_types: set[type] = {  # pylint: disable=g-bare-generic
    dtypes.int4, np.int8, np.int16, np.int32, np.int64,
    dtypes.uint4, np.uint8, np.uint16, np.uint32, np.uint64,
    np.complex64, np.complex128,
    np.bool_, np.longlong, np.intc,
} | {np.dtype(dt).type for dt in dtypes._float_types}

if dtypes.int2 is not None:
  assert dtypes.uint2 is not None
  numpy_scalar_types.add(dtypes.int2)
  numpy_scalar_types.add(dtypes.uint2)

if dtypes.int1 is not None:
  assert dtypes.uint1 is not None
  numpy_scalar_types.add(dtypes.int1)
  numpy_scalar_types.add(dtypes.uint1)

array_types: set[type] = {literals.TypedNdArray, np.ndarray} | numpy_scalar_types  # pylint: disable=g-bare-generic


def masked_array_error(*args, **kwargs):
  raise ValueError(
      "numpy masked arrays are not supported as direct inputs to JAX functions."
      " Use arr.filled() to convert the value to a standard numpy array.")

core.pytype_aval_mappings[np.ma.MaskedArray] = masked_array_error


def _make_shaped_array_for_numpy_array(x: np.ndarray) -> ShapedArray:
  dtype = x.dtype
  dtypes.check_valid_dtype(dtype)
  return ShapedArray(x.shape, dtypes.canonicalize_dtype(dtype), sharding=None)

core.pytype_aval_mappings[np.ndarray] = _make_shaped_array_for_numpy_array


def _make_shaped_array_for_typed_ndarray(
    x: literals.TypedNdArray,
) -> ShapedArray:
  dtype = x.dtype
  dtypes.check_valid_dtype(dtype)
  return ShapedArray(x.shape, dtype, sharding=None, weak_type=x.weak_type)


core.pytype_aval_mappings[literals.TypedNdArray] = _make_shaped_array_for_typed_ndarray


def _make_shaped_array_for_numpy_scalar(x: np.generic) -> ShapedArray:
  dtype = np.dtype(x)
  dtypes.check_valid_dtype(dtype)
  shape = np.shape(x)
  return ShapedArray(shape, dtypes.canonicalize_dtype(dtype), sharding=None)

for t in numpy_scalar_types:
  core.pytype_aval_mappings[t] = _make_shaped_array_for_numpy_scalar

core.literalable_types.update(array_types)


core.literalable_types.add(literals.TypedNdArray)

_int32_min = np.iinfo(np.int32).min
_int32_max = np.iinfo(np.int32).max
_int64_min = np.iinfo(np.int64).min
_int64_max = np.iinfo(np.int64).max

# Note: all python scalar types are weak except bool, because bool only
# comes in a single width.
_bool_aval = ShapedArray((), dtype=np.dtype(bool))
_int32_aval = ShapedArray((), dtype=np.dtype(np.int32), weak_type=True)
_int64_aval = ShapedArray((), dtype=np.dtype(np.int64), weak_type=True)
_float32_aval = ShapedArray((), dtype=np.dtype(np.float32), weak_type=True)
_float64_aval = ShapedArray((), dtype=np.dtype(np.float64), weak_type=True)
_complex64_aval = ShapedArray((), dtype=np.dtype(np.complex64), weak_type=True)
_complex128_aval = ShapedArray((), dtype=np.dtype(np.complex128), weak_type=True)

core.pytype_aval_mappings[bool] = lambda v: _bool_aval

def _int_aval(value):
  if config.enable_x64.value:
    if value < _int64_min or value > _int64_max:
      raise OverflowError(f"Python int {value} too large to convert to int64")
    return _int64_aval
  else:
    if value < _int32_min or value > _int32_max:
      raise OverflowError(f"Python int {value} too large to convert to int32")
    return _int32_aval
core.pytype_aval_mappings[int] = _int_aval

_float_aval = lambda v: _float64_aval if config.enable_x64.value else _float32_aval
core.pytype_aval_mappings[float] = _float_aval

_complex_aval = lambda v: _complex128_aval if config.enable_x64.value else _complex64_aval
core.pytype_aval_mappings[complex] = _complex_aval

core.literalable_types.update(dtypes.python_scalar_types)


def _aval_for_typed_scalar(x):
  return ShapedArray((), x.dtype, weak_type=True, sharding=None)

for t in literals.typed_scalar_types:
  core.pytype_aval_mappings[t] = _aval_for_typed_scalar
core.literalable_types.update(literals.typed_scalar_types)

_ndarray_dtype_cache = {}
_ndarray_dtype_cache_lock = threading.Lock()

if jaxlib_extension_version < 420 and not TYPE_CHECKING:
  # This is a temporary shim implementation of a weakly-keyed, weakly value
  # cache that should go away after jaxlib 0.9.2 is the minimum.
  def _canonicalize_ndarray_dtype(x):
    x_id = id(x)
    with _ndarray_dtype_cache_lock:
      entry = _ndarray_dtype_cache.get(x_id)
      if entry is not None:
        xref, ansref = entry
        ans = ansref()
        if xref() is x and ans is not None:
          return ans

    dtype = dtypes.canonicalize_dtype(x.dtype)
    ans = literals.TypedNdArray(np.asarray(x, dtype), weak_type=False)

    def clear_cache(wr, key=x_id):
      with _ndarray_dtype_cache_lock:
        val = _ndarray_dtype_cache.get(key)
        if val is not None and (val[0] is wr or val[1] is wr):
          del _ndarray_dtype_cache[key]

    xref = weakref.ref(x, clear_cache)
    ansref = weakref.ref(ans, clear_cache)
    with _ndarray_dtype_cache_lock:
      _ndarray_dtype_cache[x_id] = (xref, ansref)
    return ans

else:
  # We use a weakly-keyed, weakly-valued cache to memoize the result of
  # canonicalizing ndarray dtypes. The goal is that as long as both the key
  # and the value are alive, we will produce the same object from dtype
  # canonicalization. This avoids duplication of large constants when forming
  # a jaxpr.
  @weakref_lru_cache.weak_key_weak_value_cache
  def _canonicalize_ndarray_dtype(x):
    dtype = dtypes.canonicalize_dtype(x.dtype)
    return literals.TypedNdArray(np.asarray(x, dtype), weak_type=False)

dtypes.canonicalize_value_handlers[np.ndarray] = _canonicalize_ndarray_dtype


def _canonicalize_masked_array_dtype(x):
  raise ValueError("numpy masked arrays are not supported as direct inputs to JAX functions. "
                   "Use arr.filled() to convert the value to a standard numpy array.")

def _canonicalize_numpy_scalar(x):
  dtype = dtypes.canonicalize_dtype(x.dtype)
  return literals.TypedNdArray(np.asarray(x, dtype), weak_type=False)

dtypes.canonicalize_value_handlers.update(
    (t, _canonicalize_numpy_scalar) for t in numpy_scalar_types)


dtypes.canonicalize_value_handlers[literals.TypedNdArray] = lambda x: x

dtypes.canonicalize_value_handlers[np.ma.MaskedArray] = _canonicalize_masked_array_dtype

def _canonicalize_python_scalar(literal_type, typ):
  def canonicalize_scalar(x):
    return literal_type(x, dtypes.scalar_type_to_dtype(typ, x))  # pytype: disable=wrong-arg-types
  return canonicalize_scalar

dtypes.canonicalize_value_handlers[bool] = lambda x: x
dtypes.canonicalize_value_handlers[int] = _canonicalize_python_scalar(
    literals.TypedInt, int)
dtypes.canonicalize_value_handlers[float] = _canonicalize_python_scalar(
    literals.TypedFloat, float)
dtypes.canonicalize_value_handlers[complex] = _canonicalize_python_scalar(
    literals.TypedComplex, complex)

dtypes.canonicalize_value_handlers[literals.TypedInt] = lambda x: x
dtypes.canonicalize_value_handlers[literals.TypedFloat] = lambda x: x
dtypes.canonicalize_value_handlers[literals.TypedComplex] = lambda x: x
