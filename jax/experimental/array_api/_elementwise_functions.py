# Copyright 2023 The JAX Authors.
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

import jax
from jax.experimental.array_api._data_type_functions import (
    result_type as _result_type,
    isdtype as _isdtype,
)
import numpy as np


def _promote_dtypes(name, *args):
  assert isinstance(name, str)
  if not all(isinstance(arg, jax.Array) for arg in args):
    raise ValueError(f"{name}: inputs must be arrays; got types {[type(arg) for arg in args]}")
  dtype = _result_type(*args)
  return [arg.astype(dtype) for arg in args]


def abs(x, /):
  """Calculates the absolute value for each element x_i of the input array x."""
  x, = _promote_dtypes("abs", x)
  return jax.numpy.abs(x)


def acos(x, /):
  """Calculates an implementation-dependent approximation of the principal value of the inverse cosine for each element x_i of the input array x."""
  x, = _promote_dtypes("acos", x)
  return jax.numpy.acos(x)

def acosh(x, /):
  """Calculates an implementation-dependent approximation to the inverse hyperbolic cosine for each element x_i of the input array x."""
  x, = _promote_dtypes("acos", x)
  return jax.numpy.acosh(x)


def add(x1, x2, /):
  """Calculates the sum for each element x1_i of the input array x1 with the respective element x2_i of the input array x2."""
  x1, x2 = _promote_dtypes("add", x1, x2)
  return jax.numpy.add(x1, x2)


def asin(x, /):
  """Calculates an implementation-dependent approximation of the principal value of the inverse sine for each element x_i of the input array x."""
  x, = _promote_dtypes("asin", x)
  return jax.numpy.asin(x)


def asinh(x, /):
  """Calculates an implementation-dependent approximation to the inverse hyperbolic sine for each element x_i in the input array x."""
  x, = _promote_dtypes("asinh", x)
  return jax.numpy.asinh(x)


def atan(x, /):
  """Calculates an implementation-dependent approximation of the principal value of the inverse tangent for each element x_i of the input array x."""
  x, = _promote_dtypes("atan", x)
  return jax.numpy.atan(x)


def atan2(x1, x2, /):
  """Calculates an implementation-dependent approximation of the inverse tangent of the quotient x1/x2, having domain [-infinity, +infinity] x [-infinity, +infinity] (where the x notation denotes the set of ordered pairs of elements (x1_i, x2_i)) and codomain [-π, +π], for each pair of elements (x1_i, x2_i) of the input arrays x1 and x2, respectively."""
  x1, x2 = _promote_dtypes("atan2", x1, x2)
  return jax.numpy.arctan2(x1, x2)


def atanh(x, /):
  """Calculates an implementation-dependent approximation to the inverse hyperbolic tangent for each element x_i of the input array x."""
  x, = _promote_dtypes("atanh", x)
  return jax.numpy.atanh(x)


def bitwise_and(x1, x2, /):
  """Computes the bitwise AND of the underlying binary representation of each element x1_i of the input array x1 with the respective element x2_i of the input array x2."""
  x1, x2 = _promote_dtypes("bitwise_and", x1, x2)
  return jax.numpy.bitwise_and(x1, x2)


def bitwise_left_shift(x1, x2, /):
  """Shifts the bits of each element x1_i of the input array x1 to the left by appending x2_i (i.e., the respective element in the input array x2) zeros to the right of x1_i."""
  x1, x2 = _promote_dtypes("bitwise_left_shift", x1, x2)
  return jax.numpy.bitwise_left_shift(x1, x2)


def bitwise_invert(x, /):
  """Inverts (flips) each bit for each element x_i of the input array x."""
  x, = _promote_dtypes("bitwise_invert", x)
  return jax.numpy.bitwise_invert(x)


def bitwise_or(x1, x2, /):
  """Computes the bitwise OR of the underlying binary representation of each element x1_i of the input array x1 with the respective element x2_i of the input array x2."""
  x1, x2 = _promote_dtypes("bitwise_or", x1, x2)
  return jax.numpy.bitwise_or(x1, x2)


def bitwise_right_shift(x1, x2, /):
  """Shifts the bits of each element x1_i of the input array x1 to the right according to the respective element x2_i of the input array x2."""
  x1, x2 = _promote_dtypes("bitwise_right_shift", x1, x2)
  return jax.numpy.bitwise_right_shift(x1, x2)


def bitwise_xor(x1, x2, /):
  """Computes the bitwise XOR of the underlying binary representation of each element x1_i of the input array x1 with the respective element x2_i of the input array x2."""
  x1, x2 = _promote_dtypes("bitwise_xor", x1, x2)
  return jax.numpy.bitwise_xor(x1, x2)


def ceil(x, /):
  """Rounds each element x_i of the input array x to the smallest (i.e., closest to -infinity) integer-valued number that is not less than x_i."""
  x, = _promote_dtypes("ceil", x)
  if _isdtype(x.dtype, "integral"):
    return x
  return jax.numpy.ceil(x)


def conj(x, /):
  """Returns the complex conjugate for each element x_i of the input array x."""
  x, = _promote_dtypes("conj", x)
  return jax.numpy.conj(x)


def cos(x, /):
  """Calculates an implementation-dependent approximation to the cosine for each element x_i of the input array x."""
  x, = _promote_dtypes("cos", x)
  return jax.numpy.cos(x)


def cosh(x, /):
  """Calculates an implementation-dependent approximation to the hyperbolic cosine for each element x_i in the input array x."""
  x, = _promote_dtypes("cosh", x)
  return jax.numpy.cosh(x)


def divide(x1, x2, /):
  """Calculates the division of each element x1_i of the input array x1 with the respective element x2_i of the input array x2."""
  x1, x2 = _promote_dtypes("divide", x1, x2)
  return jax.numpy.divide(x1, x2)


def equal(x1, x2, /):
  """Computes the truth value of x1_i == x2_i for each element x1_i of the input array x1 with the respective element x2_i of the input array x2."""
  x1, x2 = _promote_dtypes("equal", x1, x2)
  return jax.numpy.equal(x1, x2)


def exp(x, /):
  """Calculates an implementation-dependent approximation to the exponential function for each element x_i of the input array x (e raised to the power of x_i, where e is the base of the natural logarithm)."""
  x, = _promote_dtypes("exp", x)
  return jax.numpy.exp(x)


def expm1(x, /):
  """Calculates an implementation-dependent approximation to exp(x)-1 for each element x_i of the input array x."""
  x, = _promote_dtypes("expm1", x)
  return jax.numpy.expm1(x)


def floor(x, /):
  """Rounds each element x_i of the input array x to the greatest (i.e., closest to +infinity) integer-valued number that is not greater than x_i."""
  x, = _promote_dtypes("floor", x)
  if _isdtype(x.dtype, "integral"):
    return x
  return jax.numpy.floor(x)


def floor_divide(x1, x2, /):
  """Rounds the result of dividing each element x1_i of the input array x1 by the respective element x2_i of the input array x2 to the greatest (i.e., closest to +infinity) integer-value number that is not greater than the division result."""
  x1, x2 = _promote_dtypes("floor_divide", x1, x2)
  return jax.numpy.floor_divide(x1, x2)


def greater(x1, x2, /):
  """Computes the truth value of x1_i > x2_i for each element x1_i of the input array x1 with the respective element x2_i of the input array x2."""
  x1, x2 = _promote_dtypes("greater", x1, x2)
  return jax.numpy.greater(x1, x2)


def greater_equal(x1, x2, /):
  """Computes the truth value of x1_i >= x2_i for each element x1_i of the input array x1 with the respective element x2_i of the input array x2."""
  x1, x2 = _promote_dtypes("greater_equal", x1, x2)
  return jax.numpy.greater_equal(x1, x2)


def imag(x, /):
  """Returns the imaginary component of a complex number for each element x_i of the input array x."""
  x, = _promote_dtypes("imag", x)
  return jax.numpy.imag(x)


def isfinite(x, /):
  """Tests each element x_i of the input array x to determine if finite."""
  x, = _promote_dtypes("isfinite", x)
  return jax.numpy.isfinite(x)


def isinf(x, /):
  """Tests each element x_i of the input array x to determine if equal to positive or negative infinity."""
  x, = _promote_dtypes("isinf", x)
  return jax.numpy.isinf(x)


def isnan(x, /):
  """Tests each element x_i of the input array x to determine whether the element is NaN."""
  x, = _promote_dtypes("isnan", x)
  return jax.numpy.isnan(x)


def less(x1, x2, /):
  """Computes the truth value of x1_i < x2_i for each element x1_i of the input array x1 with the respective element x2_i of the input array x2."""
  x1, x2 = _promote_dtypes("less", x1, x2)
  return jax.numpy.less(x1, x2)


def less_equal(x1, x2, /):
  """Computes the truth value of x1_i <= x2_i for each element x1_i of the input array x1 with the respective element x2_i of the input array x2."""
  x1, x2 = _promote_dtypes("less_equal", x1, x2)
  return jax.numpy.less_equal(x1, x2)


def log(x, /):
  """Calculates an implementation-dependent approximation to the natural (base e) logarithm for each element x_i of the input array x."""
  x, = _promote_dtypes("log", x)
  return jax.numpy.log(x)


def log1p(x, /):
  """Calculates an implementation-dependent approximation to log(1+x), where log refers to the natural (base e) logarithm, for each element x_i of the input array x."""
  x, = _promote_dtypes("log", x)
  return jax.numpy.log1p(x)


def log2(x, /):
  """Calculates an implementation-dependent approximation to the base 2 logarithm for each element x_i of the input array x."""
  x, = _promote_dtypes("log2", x)
  return jax.numpy.log2(x)


def log10(x, /):
  """Calculates an implementation-dependent approximation to the base 10 logarithm for each element x_i of the input array x."""
  x, = _promote_dtypes("log10", x)
  return jax.numpy.log10(x)


def logaddexp(x1, x2, /):
  """Calculates the logarithm of the sum of exponentiations log(exp(x1) + exp(x2)) for each element x1_i of the input array x1 with the respective element x2_i of the input array x2."""
  x1, x2 = _promote_dtypes("logaddexp", x1, x2)
  return jax.numpy.logaddexp(x1, x2)


def logical_and(x1, x2, /):
  """Computes the logical AND for each element x1_i of the input array x1 with the respective element x2_i of the input array x2."""
  x1, x2 = _promote_dtypes("logical_and", x1, x2)
  return jax.numpy.logical_and(x1, x2)


def logical_not(x, /):
  """Computes the logical NOT for each element x_i of the input array x."""
  x, = _promote_dtypes("logical_not", x)
  return jax.numpy.logical_not(x)


def logical_or(x1, x2, /):
  """Computes the logical OR for each element x1_i of the input array x1 with the respective element x2_i of the input array x2."""
  x1, x2 = _promote_dtypes("logical_or", x1, x2)
  return jax.numpy.logical_or(x1, x2)


def logical_xor(x1, x2, /):
  """Computes the logical XOR for each element x1_i of the input array x1 with the respective element x2_i of the input array x2."""
  x1, x2 = _promote_dtypes("logical_xor", x1, x2)
  return jax.numpy.logical_xor(x1, x2)


def multiply(x1, x2, /):
  """Calculates the product for each element x1_i of the input array x1 with the respective element x2_i of the input array x2."""
  x1, x2 = _promote_dtypes("multiply", x1, x2)
  return jax.numpy.multiply(x1, x2)


def negative(x, /):
  """Computes the numerical negative of each element x_i (i.e., y_i = -x_i) of the input array x."""
  x, = _promote_dtypes("negative", x)
  return jax.numpy.negative(x)


def not_equal(x1, x2, /):
  """Computes the truth value of x1_i != x2_i for each element x1_i of the input array x1 with the respective element x2_i of the input array x2."""
  x1, x2 = _promote_dtypes("not_equal", x1, x2)
  return jax.numpy.not_equal(x1, x2)


def positive(x, /):
  """Computes the numerical positive of each element x_i (i.e., y_i = +x_i) of the input array x."""
  x, = _promote_dtypes("positive", x)
  return x


def pow(x1, x2, /):
  """Calculates an implementation-dependent approximation of exponentiation by raising each element x1_i (the base) of the input array x1 to the power of x2_i (the exponent), where x2_i is the corresponding element of the input array x2."""
  x1, x2 = _promote_dtypes("pow", x1, x2)
  return jax.numpy.pow(x1, x2)


def real(x, /):
  """Returns the real component of a complex number for each element x_i of the input array x."""
  x, = _promote_dtypes("real", x)
  return jax.numpy.real(x)


def remainder(x1, x2, /):
  """Returns the remainder of division for each element x1_i of the input array x1 and the respective element x2_i of the input array x2."""
  x1, x2 = _promote_dtypes("remainder", x1, x2)
  return jax.numpy.remainder(x1, x2)


def round(x, /):
  """Rounds each element x_i of the input array x to the nearest integer-valued number."""
  x, = _promote_dtypes("round", x)
  return jax.numpy.round(x)


def sign(x, /):
  """Returns an indication of the sign of a number for each element x_i of the input array x."""
  x, = _promote_dtypes("sign", x)
  if _isdtype(x.dtype, "complex floating"):
    return x / abs(x)
  return jax.numpy.sign(x)


def sin(x, /):
  """Calculates an implementation-dependent approximation to the sine for each element x_i of the input array x."""
  x, = _promote_dtypes("sin", x)
  return jax.numpy.sin(x)


def sinh(x, /):
  """Calculates an implementation-dependent approximation to the hyperbolic sine for each element x_i of the input array x."""
  x, = _promote_dtypes("sin", x)
  return jax.numpy.sinh(x)


def square(x, /):
  """Squares each element x_i of the input array x."""
  x, = _promote_dtypes("square", x)
  return jax.numpy.square(x)


def sqrt(x, /):
  """Calculates the principal square root for each element x_i of the input array x."""
  x, = _promote_dtypes("sqrt", x)
  return jax.numpy.sqrt(x)


def subtract(x1, x2, /):
  """Calculates the difference for each element x1_i of the input array x1 with the respective element x2_i of the input array x2."""
  x1, x2 = _promote_dtypes("subtract", x1, x2)
  return jax.numpy.subtract(x1, x2)


def tan(x, /):
  """Calculates an implementation-dependent approximation to the tangent for each element x_i of the input array x."""
  x, = _promote_dtypes("tan", x)
  return jax.numpy.tan(x)


def tanh(x, /):
  """Calculates an implementation-dependent approximation to the hyperbolic tangent for each element x_i of the input array x."""
  x, = _promote_dtypes("tanh", x)
  return jax.numpy.tanh(x)


def trunc(x, /):
  """Rounds each element x_i of the input array x to the nearest integer-valued number that is closer to zero than x_i."""
  x, = _promote_dtypes("trunc", x)
  if _isdtype(x.dtype, "integral"):
    return x
  return jax.numpy.trunc(x)
