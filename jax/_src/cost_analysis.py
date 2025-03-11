# Copyright 2025 The JAX Authors.
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


import functools
import typing
import warnings
import jax
import jax.core
import jax.extend.core


ShapedArray = jax._src.core.ShapedArray


class _CountFlops:
  """Visitor for counting the number of floating point operations."""

  def _shape_size(self, shape: typing.Iterable):
    return functools.reduce(lambda x, y: x * y, shape, 1)

  def _unary(self, x: ShapedArray):
    # Default unary handling is one flop for each input/output element.
    return self._shape_size(x.shape)

  def _binary(self, x: ShapedArray, y: ShapedArray):
    # Default binary handling is one flop for each input element, plus one for
    # each output element.
    output_shape = [max(xd, yd) for xd, yd in zip(x.shape, y.shape)]
    return (
        self._shape_size(x.shape)
        + self._shape_size(y.shape)
        + self._shape_size(output_shape)
    )

  def visit_dot_general(self, x: ShapedArray, y: ShapedArray):
    # Cost of (n, ..., m) @ (m, ..., k) is (n * ... * m * ... * k) * 2
    return 2 * self._shape_size(x.shape + y.shape[1:])

  def visit_max(self, x: ShapedArray, y: ShapedArray):
    # Cost of matmul + cost of max (which is the size of the output matrix)
    return 2 * self._shape_size(x.shape) * self._shape_size(
        y.shape[1:]
    ) + self._shape_size(x.shape[:-1] + x.shape[1:])

  def visit_reduce_sum(self, x: ShapedArray):
    return self._shape_size(x.shape) - 1

  visit_add = _binary
  visit_mul = _binary
  visit_sin = _unary
  visit_convert_element_type = _unary


class _CountBytesAccessed:
  """Visitor for counting the number of bytes of memory read or written."""

  def __init__(self, word_size: int):
    self.word_size = word_size  # in bytes

  def _size_in_bytes(self, shape: typing.Iterable):
    return functools.reduce(lambda c, v: c * v, shape, self.word_size)

  def _binary(self, x: ShapedArray, y: ShapedArray):
    output_shape = [max(xd, yd) for xd, yd in zip(x.shape, y.shape)]
    return (
        self._size_in_bytes(x.shape)
        + self._size_in_bytes(y.shape)
        + self._size_in_bytes(output_shape)
    )

  def _unary(self, x: ShapedArray):
    return 2 * self._size_in_bytes(x.shape)

  def visit_dot_general(self, x: ShapedArray, y: ShapedArray):
    # Read both inputs, write the result.
    return (
        self._size_in_bytes(x.shape)
        + self._size_in_bytes(y.shape)
        + self._size_in_bytes(x.shape[:-1] + y.shape[1:])
    )

  def visit_max(self, x: ShapedArray, y: ShapedArray):
    # Access all inputs and a scalar, and also:
    # 1) write and read the output of matmul
    # 2) write and read a matrix of zeros
    # 3) write a matrix of results
    return (
        self._size_in_bytes(x.shape)
        + self._size_in_bytes(y.shape)
        + 5 * self._size_in_bytes(x.shape[:-1] + y.shape[1:])
        + self.word_size
    )

  def visit_reduce_sum(self, x: ShapedArray):
    return self._size_in_bytes(x.shape) + 2 * self.word_size

  visit_add = _binary
  visit_mul = _binary
  visit_sin = _unary


def _visit(visitor, jaxpr: jax.extend.core.Jaxpr):
  total = 0
  for eq in jaxpr.eqns:
    if eq.primitive.name == "pjit":
      for nested in jax.core.jaxprs_in_params(eq.params):
        total += _visit(visitor, nested)
    elif f := getattr(visitor, "visit_" + eq.primitive.name, None):
      total += f(*[i.aval for i in eq.invars])
    else:
      warnings.warn(
          "Warning: No cost analysis implementation for %r" % eq.primitive.name
      )
  return total


def count_flops(jaxpr: jax.extend.core.Jaxpr):
  return _visit(_CountFlops(), jaxpr)


def count_bytes_accessed(jaxpr: jax.extend.core.Jaxpr):
  word_size = 64 if jax.config.read("jax_enable_x64") else 32
  accessed = _visit(_CountBytesAccessed(word_size // 8), jaxpr)
  return accessed
