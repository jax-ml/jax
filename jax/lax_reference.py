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

from typing import Sequence

import numpy as np

from . import dtypes, lax
from .lax.lax import padtype_to_pads, DType, Shape
# flake8: noqa: F401
from .interpreters.numpy_eval import (
  neg, sign, floor, ceil, round, nextafter, is_finite, exp, expm1, log, log1p,
  tanh, sin, cos, atan2, sqrt, rsqrt, sinh, cosh, asinh, acosh, atanh, betainc,
  lgamma, digamma, igamma, igammac, erf, erfc, erf_inv, bessel_i0e, bessel_i1e,
  real, imag, conj, complex, abs, pow, integer_pow, bitwise_not, bitwise_and,
  bitwise_or, bitwise_xor, add, sub, mul, div, rem, max, min, shift_left,
  shift_right_arithmetic, shift_right_logical, population_count, eq, ne, ge, gt,
  le, lt, convert_element_type, bitcast_convert_type, clamp,
  conv_with_general_padding, dot_general, broadcast_in_dim, squeeze,
  reshape, pad, rev, select, slice, transpose, sort_key_val, top_k, cummin,
  cummax, cumsum, cumprod, gather, numpy_eval, _conv
)

square = np.square
reciprocal = np.reciprocal
tan = np.tan
asin = np.arcsin
acos = np.arccos
atan = np.arctan
full = np.full

def dot(a, b, *, precision=None): return np.dot(a, b)

def broadcast(operand, sizes):
  return np.broadcast_to(operand, sizes + np.shape(operand))

def conv(lhs, rhs, window_strides: Sequence[int], padding: str, precision=None):
  pads = padtype_to_pads(lhs.shape[2:], rhs.shape[2:], window_strides, padding)
  return _conv(lhs, rhs, window_strides, pads)

def broadcasted_iota(dtype: DType, shape: Shape, dimension: int):
  arr = np.arange(shape[dimension], dtype=dtypes.canonicalize_dtype(dtype))
  singleton_shape = [1] * len(shape)
  singleton_shape[dimension] = shape[dimension]
  return np.broadcast_to(arr.reshape(singleton_shape), shape)

def _delta(dtype: DType, shape: Shape, axes: Sequence[int]):
  base = np.zeros(np.take(shape, axes), dtype=dtype)
  base[np.diag_indices(np.min(base.shape), ndim=base.ndim)] = 1
  broadcast_axes = [a for a in range(len(shape)) if a not in axes]
  for a in broadcast_axes:
    base = np.expand_dims(base, a)  # not all at once to support NumPy 1.16
  return np.broadcast_to(base, shape)

argmin = numpy_eval()(lax.argmin)
argmax = numpy_eval()(lax.argmax)
batch_matmul = numpy_eval()(lax.batch_matmul)
concatenate = numpy_eval()(lax.concatenate)
conv_general_dilated = numpy_eval()(lax.conv_general_dilated)
dynamic_slice = numpy_eval()(lax.dynamic_slice)
dynamic_update_slice = numpy_eval()(lax.dynamic_update_slice)
expand_dims = numpy_eval()(lax.expand_dims)
index_take = numpy_eval()(lax.index_take)
reduce = numpy_eval()(lax.reduce)
reduce_window = numpy_eval()(lax.reduce_window)
sort = numpy_eval()(lax.sort)
