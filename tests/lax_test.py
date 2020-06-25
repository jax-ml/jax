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


import collections
from functools import partial
import itertools
from unittest import SkipTest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp

import jax
from jax import api
from jax import core
from jax import dtypes
from jax import lax
from jax import test_util as jtu
from jax import lax_reference
from jax.test_util import check_grads
import jax.util

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS


def num_float_bits(dtype):
  return dtypes.finfo(dtypes.canonicalize_dtype(dtype)).bits


### lax tests

# For standard unops and binops, we can generate a large number of tests on
# arguments of appropriate shapes and dtypes using the following table.

def supported_dtypes(dtypes):
  return [t for t in dtypes if t in jtu.supported_dtypes()]

float_dtypes = supported_dtypes([dtypes.bfloat16, onp.float16, onp.float32,
                                 onp.float64])
complex_elem_dtypes = supported_dtypes([onp.float32, onp.float64])
complex_dtypes = supported_dtypes([onp.complex64, onp.complex128])
inexact_dtypes = float_dtypes + complex_dtypes
int_dtypes = supported_dtypes([onp.int32, onp.int64])
uint_dtypes = supported_dtypes([onp.uint32, onp.uint64])
bool_dtypes = [onp.bool_]
default_dtypes = float_dtypes + int_dtypes
all_dtypes = float_dtypes + complex_dtypes + int_dtypes + bool_dtypes

compatible_shapes = [[(3,)], [(3, 4), (3, 1), (1, 4)], [(2, 3, 4), (2, 1, 4)]]


OpRecord = collections.namedtuple(
    "OpRecord", ["op", "nargs", "dtypes", "rng_factory", "tol"])

def op_record(op, nargs, dtypes, rng_factory, tol=None):
  return OpRecord(op, nargs, dtypes, rng_factory, tol)

LAX_OPS = [
    op_record("neg", 1, default_dtypes + complex_dtypes, jtu.rand_small),
    op_record("sign", 1, default_dtypes + uint_dtypes, jtu.rand_small),
    op_record("floor", 1, float_dtypes, jtu.rand_small),
    op_record("ceil", 1, float_dtypes, jtu.rand_small),
    op_record("round", 1, float_dtypes, jtu.rand_default),
    op_record("nextafter", 2, [f for f in float_dtypes if f != dtypes.bfloat16],
              jtu.rand_default, tol=0),

    op_record("is_finite", 1, float_dtypes, jtu.rand_small),

    op_record("exp", 1, float_dtypes + complex_dtypes, jtu.rand_small),
    # TODO(b/142975473): on CPU, expm1 for float64 is only accurate to ~float32
    # precision.
    op_record("expm1", 1, float_dtypes + complex_dtypes, jtu.rand_small,
              {onp.float64: 1e-8}),
    op_record("log", 1, float_dtypes + complex_dtypes, jtu.rand_positive),
    op_record("log1p", 1, float_dtypes + complex_dtypes, jtu.rand_positive),
    # TODO(b/142975473): on CPU, tanh for complex128 is only accurate to
    # ~float32 precision.
    # TODO(b/143135720): on GPU, tanh has only ~float32 precision.
    op_record("tanh", 1, float_dtypes + complex_dtypes, jtu.rand_small,
              {onp.float64: 1e-9, onp.complex128: 1e-7}),
    op_record("sin", 1, float_dtypes + complex_dtypes, jtu.rand_default),
    op_record("cos", 1, float_dtypes + complex_dtypes, jtu.rand_default),
    op_record("atan2", 2, float_dtypes, jtu.rand_default),

    op_record("sqrt", 1, float_dtypes + complex_dtypes, jtu.rand_positive),
    op_record("rsqrt", 1, float_dtypes + complex_dtypes, jtu.rand_positive),
    op_record("square", 1, float_dtypes + complex_dtypes, jtu.rand_default),
    op_record("reciprocal", 1, float_dtypes + complex_dtypes, jtu.rand_positive),
    op_record("tan", 1, float_dtypes, jtu.rand_default, {onp.float32: 3e-5}),
    op_record("asin", 1, float_dtypes, jtu.rand_small),
    op_record("acos", 1, float_dtypes, jtu.rand_small),
    op_record("atan", 1, float_dtypes, jtu.rand_small),
    op_record("asinh", 1, float_dtypes, jtu.rand_default),
    op_record("acosh", 1, float_dtypes, jtu.rand_positive),
    # TODO(b/155331781): atanh has only ~float precision
    op_record("atanh", 1, float_dtypes, jtu.rand_small, {onp.float64: 1e-9}),
    op_record("sinh", 1, float_dtypes + complex_dtypes, jtu.rand_default),
    op_record("cosh", 1, float_dtypes + complex_dtypes, jtu.rand_default),
    op_record("lgamma", 1, float_dtypes, jtu.rand_positive,
              {onp.float32: 1e-3 if jtu.device_under_test() == "tpu" else 1e-5,
               onp.float64: 1e-14}),
    op_record("digamma", 1, float_dtypes, jtu.rand_positive,
              {onp.float64: 1e-14}),
    op_record("betainc", 3, float_dtypes, jtu.rand_positive,
              {onp.float64: 1e-14}),
    op_record("igamma", 2,
              [f for f in float_dtypes if f not in [dtypes.bfloat16, onp.float16]],
              jtu.rand_positive, {onp.float64: 1e-14}),
    op_record("igammac", 2,
              [f for f in float_dtypes if f not in [dtypes.bfloat16, onp.float16]],
              jtu.rand_positive, {onp.float64: 1e-14}),
    op_record("erf", 1, float_dtypes, jtu.rand_small),
    op_record("erfc", 1, float_dtypes, jtu.rand_small),
    # TODO(b/142976030): the approximation of erfinf used by XLA is only
    # accurate to float32 precision.
    op_record("erf_inv", 1, float_dtypes, jtu.rand_small,
              {onp.float64: 1e-9}),
    op_record("bessel_i0e", 1, float_dtypes, jtu.rand_default),
    op_record("bessel_i1e", 1, float_dtypes, jtu.rand_default),

    op_record("real", 1, complex_dtypes, jtu.rand_default),
    op_record("imag", 1, complex_dtypes, jtu.rand_default),
    op_record("complex", 2, complex_elem_dtypes, jtu.rand_default),
    op_record("conj", 1, complex_elem_dtypes + complex_dtypes,
              jtu.rand_default),
    op_record("abs", 1, default_dtypes + complex_dtypes, jtu.rand_default),
    op_record("pow", 2, float_dtypes + complex_dtypes, jtu.rand_positive),

    op_record("bitwise_and", 2, bool_dtypes, jtu.rand_small),
    op_record("bitwise_not", 1, bool_dtypes, jtu.rand_small),
    op_record("bitwise_or", 2, bool_dtypes, jtu.rand_small),
    op_record("bitwise_xor", 2, bool_dtypes, jtu.rand_small),
    op_record("population_count", 1, uint_dtypes, partial(jtu.rand_int,
                                                          high=1 << 32)),

    op_record("add", 2, default_dtypes + complex_dtypes, jtu.rand_small),
    op_record("sub", 2, default_dtypes + complex_dtypes, jtu.rand_small),
    op_record("mul", 2, default_dtypes + complex_dtypes, jtu.rand_small),
    op_record("div", 2, default_dtypes + complex_dtypes, jtu.rand_nonzero),
    op_record("rem", 2, default_dtypes, jtu.rand_nonzero),

    op_record("max", 2, all_dtypes, jtu.rand_small),
    op_record("min", 2, all_dtypes, jtu.rand_small),

    op_record("eq", 2, all_dtypes, jtu.rand_some_equal),
    op_record("ne", 2, all_dtypes, jtu.rand_small),
    op_record("ge", 2, default_dtypes, jtu.rand_small),
    op_record("gt", 2, default_dtypes, jtu.rand_small),
    op_record("le", 2, default_dtypes, jtu.rand_small),
    op_record("lt", 2, default_dtypes, jtu.rand_small),
]

CombosWithReplacement = itertools.combinations_with_replacement


class LaxTest(jtu.JaxTestCase):
  """Numerical tests for LAX operations."""

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(
            rec.op, shapes, itertools.repeat(dtype)),
         "op_name": rec.op, "rng_factory": rec.rng_factory, "shapes": shapes,
         "dtype": dtype}
        for shape_group in compatible_shapes
        for shapes in CombosWithReplacement(shape_group, rec.nargs)
        for dtype in rec.dtypes)
      for rec in LAX_OPS))
  def testOp(self, op_name, rng_factory, shapes, dtype):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype) for shape in shapes]
    op = getattr(lax, op_name)
    self._CompileAndCheck(op, args_maker)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(
            rec.op, shapes, itertools.repeat(dtype)),
         "op_name": rec.op, "rng_factory": rec.rng_factory, "shapes": shapes,
         "dtype": dtype, "tol": rec.tol}
        for shape_group in compatible_shapes
        for shapes in CombosWithReplacement(shape_group, rec.nargs)
        for dtype in rec.dtypes)
      for rec in LAX_OPS))
  def testOpAgainstNumpy(self, op_name, rng_factory, shapes, dtype, tol):
    if (not FLAGS.jax_enable_x64 and op_name == "nextafter"
        and dtype == onp.float64):
      raise SkipTest("64-bit mode disabled")
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype) for shape in shapes]
    op = getattr(lax, op_name)
    numpy_op = getattr(lax_reference, op_name)
    self._CheckAgainstNumpy(op, numpy_op, args_maker, tol=tol)

  # TODO test shift_left, shift_right_arithmetic, shift_right_logical

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_from_dtype={}_to_dtype={}".format(
          from_dtype, to_dtype),
       "from_dtype": from_dtype, "to_dtype": to_dtype, "rng_factory": rng_factory}
      for from_dtype, to_dtype in itertools.product(
          [onp.float32, onp.int32, "float32", "int32"], repeat=2)
      for rng_factory in [jtu.rand_default]))
  def testConvertElementType(self, from_dtype, to_dtype, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng((2, 3), from_dtype)]
    op = lambda x: lax.convert_element_type(x, to_dtype)
    self._CompileAndCheck(op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_from_dtype={}_to_dtype={}"
       .format(from_dtype, to_dtype),
       "from_dtype": from_dtype, "to_dtype": to_dtype, "rng_factory": rng_factory}
      for from_dtype, to_dtype in itertools.product(
          [onp.float32, onp.int32, "float32", "int32"], repeat=2)
      for rng_factory in [jtu.rand_default]))
  def testConvertElementTypeAgainstNumpy(self, from_dtype, to_dtype, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng((2, 3), from_dtype)]
    op = lambda x: lax.convert_element_type(x, to_dtype)
    numpy_op = lambda x: lax_reference.convert_element_type(x, to_dtype)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_from_dtype={}_to_dtype={}"
       .format(from_dtype, to_dtype),
       "from_dtype": from_dtype, "to_dtype": to_dtype, "rng_factory": rng_factory}
      for from_dtype, to_dtype in itertools.product(
          [onp.float32, onp.int32, "float32", "int32"], repeat=2)
      for rng_factory in [jtu.rand_default]))
  def testBitcastConvertType(self, from_dtype, to_dtype, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng((2, 3), from_dtype)]
    op = lambda x: lax.bitcast_convert_type(x, to_dtype)
    self._CompileAndCheck(op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_from_dtype={}_to_dtype={}"
       .format(from_dtype, to_dtype),
       "from_dtype": from_dtype, "to_dtype": to_dtype, "rng_factory": rng_factory}
      for from_dtype, to_dtype in itertools.product(
          [onp.float32, onp.int32, "float32", "int32"], repeat=2)
      for rng_factory in [jtu.rand_default]))
  def testBitcastConvertTypeAgainstNumpy(self, from_dtype, to_dtype, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng((2, 3), from_dtype)]
    op = lambda x: lax.bitcast_convert_type(x, to_dtype)
    numpy_op = lambda x: lax_reference.bitcast_convert_type(x, to_dtype)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_min_shape={}_operand_shape={}_max_shape={}".format(
          jtu.format_shape_dtype_string(min_shape, dtype),
          jtu.format_shape_dtype_string(operand_shape, dtype),
          jtu.format_shape_dtype_string(max_shape, dtype)),
       "min_shape": min_shape, "operand_shape": operand_shape,
       "max_shape": max_shape, "dtype": dtype, "rng_factory": rng_factory}
      for min_shape, operand_shape, max_shape in [
          [(), (2, 3), ()],
          [(2, 3), (2, 3), ()],
          [(), (2, 3), (2, 3)],
          [(2, 3), (2, 3), (2, 3)],
      ]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testClamp(self, min_shape, operand_shape, max_shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    shapes = [min_shape, operand_shape, max_shape]
    args_maker = lambda: [rng(shape, dtype) for shape in shapes]
    self._CompileAndCheck(lax.clamp, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_min_shape={}_operand_shape={}_max_shape={}".format(
          jtu.format_shape_dtype_string(min_shape, dtype),
          jtu.format_shape_dtype_string(operand_shape, dtype),
          jtu.format_shape_dtype_string(max_shape, dtype)),
       "min_shape": min_shape, "operand_shape": operand_shape,
       "max_shape": max_shape, "dtype": dtype, "rng_factory": rng_factory}
      for min_shape, operand_shape, max_shape in [
          [(), (2, 3), ()],
          [(2, 3), (2, 3), ()],
          [(), (2, 3), (2, 3)],
          [(2, 3), (2, 3), (2, 3)],
      ]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testClampAgainstNumpy(self, min_shape, operand_shape, max_shape, dtype,
                            rng_factory):
    rng = rng_factory(self.rng())
    shapes = [min_shape, operand_shape, max_shape]
    args_maker = lambda: [rng(shape, dtype) for shape in shapes]
    self._CheckAgainstNumpy(lax.clamp, lax_reference.clamp, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dim={}_baseshape=[{}]_dtype={}_narrs={}".format(
          dim, ",".join(str(d) for d in base_shape), onp.dtype(dtype).name,
          num_arrs),
       "dim": dim, "base_shape": base_shape, "dtype": dtype,
       "num_arrs": num_arrs, "rng_factory": rng_factory}
      for num_arrs in [3]
      for dtype in default_dtypes
      for base_shape in [(4,), (3, 4), (2, 3, 4)]
      for dim in range(len(base_shape))
      for rng_factory in [jtu.rand_default]))
  def testConcatenate(self, dim, base_shape, dtype, num_arrs, rng_factory):
    rng = rng_factory(self.rng())
    shapes = [base_shape[:dim] + (size,) + base_shape[dim+1:]
              for size, _ in zip(itertools.cycle([3, 1, 4]), range(num_arrs))]
    args_maker = lambda: [rng(shape, dtype) for shape in shapes]
    op = lambda *args: lax.concatenate(args, dim)
    self._CompileAndCheck(op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dim={}_baseshape=[{}]_dtype={}_narrs={}".format(
          dim, ",".join(str(d) for d in base_shape), onp.dtype(dtype).name,
          num_arrs),
       "dim": dim, "base_shape": base_shape, "dtype": dtype,
       "num_arrs": num_arrs, "rng_factory": rng_factory}
      for num_arrs in [3]
      for dtype in default_dtypes
      for base_shape in [(4,), (3, 4), (2, 3, 4)]
      for dim in range(len(base_shape))
      for rng_factory in [jtu.rand_default]))
  def testConcatenateAgainstNumpy(self, dim, base_shape, dtype, num_arrs, rng_factory):
    rng = rng_factory(self.rng())
    shapes = [base_shape[:dim] + (size,) + base_shape[dim+1:]
              for size, _ in zip(itertools.cycle([3, 1, 4]), range(num_arrs))]
    args_maker = lambda: [rng(shape, dtype) for shape in shapes]
    op = lambda *args: lax.concatenate(args, dim)
    numpy_op = lambda *args: lax_reference.concatenate(args, dim)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_strides={}_padding={}".format(
           jtu.format_shape_dtype_string(lhs_shape, dtype),
           jtu.format_shape_dtype_string(rhs_shape, dtype), strides, padding),
          "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
          "strides": strides, "padding": padding, "rng_factory": rng_factory}
      for lhs_shape, rhs_shape in [
          ((b, i, 9, 10), (j, i, 4, 5))
          for b, i, j in itertools.product([2, 3], repeat=3)]
      for dtype in float_dtypes
      for strides in [(1, 1), (1, 2), (2, 1)]
      for padding in ["VALID", "SAME"]
      for rng_factory in [jtu.rand_small]))
  def testConv(self, lhs_shape, rhs_shape, dtype, strides, padding, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    def fun(lhs, rhs):
      return lax.conv(lhs, rhs, strides, padding)

    self._CompileAndCheck(fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_strides={}_padding={}".format(
           jtu.format_shape_dtype_string(lhs_shape, dtype),
           jtu.format_shape_dtype_string(rhs_shape, dtype), strides, padding),
          "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
          "strides": strides, "padding": padding, "rng_factory": rng_factory}
      for lhs_shape, rhs_shape in [
          ((b, i, 9, 10), (j, i, 4, 5))
          for b, i, j in itertools.product([2, 3], repeat=3)]
      for dtype in float_dtypes
      for strides in [(1, 1), (1, 2), (2, 1)]
      for padding in ["VALID", "SAME"]
      for rng_factory in [jtu.rand_small]))
  def testConvAgainstNumpy(self, lhs_shape, rhs_shape, dtype, strides, padding,
                           rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    op = lambda lhs, rhs: lax.conv(lhs, rhs, strides, padding)
    numpy_op = lambda lhs, rhs: lax_reference.conv(lhs, rhs, strides, padding)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_lhs_shape={}_rhs_shape={}_strides={}_padding={}"
       "_lhs_dilation={}_rhs_dilation={}".format(
           jtu.format_shape_dtype_string(lhs_shape, dtype),
           jtu.format_shape_dtype_string(rhs_shape, dtype),
           strides, padding, lhs_dilation, rhs_dilation),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "strides": strides, "padding": padding, "lhs_dilation": lhs_dilation,
       "rhs_dilation": rhs_dilation, "rng_factory": rng_factory}
      for lhs_shape, rhs_shape in [
          ((b, i, 9, 10), (j, i, 4, 5))
          for b, i, j in itertools.product([1, 2, 3], repeat=3)]
      for dtype in float_dtypes
      for strides in [(1, 1), (1, 2), (2, 1)]
      for padding in [((0, 0), (0, 0)), ((1, 2), (2, 0))]
      for lhs_dilation, rhs_dilation in itertools.product(
          [(1, 1), (1, 2), (2, 2)], repeat=2)
      for rng_factory in [jtu.rand_small]))
  def testConvWithGeneralPadding(self, lhs_shape, rhs_shape, dtype, strides,
                                 padding, lhs_dilation, rhs_dilation, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    def fun(lhs, rhs):
      return lax.conv_with_general_padding(
          lhs, rhs, strides, padding, lhs_dilation, rhs_dilation)

    self._CompileAndCheck(fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_lhs_shape={}_rhs_shape={}_strides={}_padding={}"
       "_lhs_dilation={}_rhs_dilation={}".format(
           jtu.format_shape_dtype_string(lhs_shape, dtype),
           jtu.format_shape_dtype_string(rhs_shape, dtype),
           strides, padding, lhs_dilation, rhs_dilation),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "strides": strides, "padding": padding, "lhs_dilation": lhs_dilation,
       "rhs_dilation": rhs_dilation, "rng_factory": rng_factory}
      for lhs_shape, rhs_shape in [
          ((b, i, 9, 10), (j, i, 4, 5))
          for b, i, j in itertools.product([1, 2, 3], repeat=3)]
      for dtype in [onp.float32] for strides in [(1, 1), (1, 2), (2, 1)]
      for padding in [((0, 0), (0, 0)), ((1, 2), (2, 0))]
      for lhs_dilation, rhs_dilation in itertools.product(
          [(1, 1), (1, 2), (2, 2)], repeat=2)
      for rng_factory in [jtu.rand_small]))
  def testConvWithGeneralPaddingAgainstNumpy(
      self, lhs_shape, rhs_shape, dtype, strides, padding, lhs_dilation,
      rhs_dilation, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    def fun(lhs, rhs):
      return lax.conv_with_general_padding(
          lhs, rhs, strides, padding, lhs_dilation, rhs_dilation,
          precision=lax.Precision.HIGHEST)

    def numpy_fun(lhs, rhs):
      return lax_reference.conv_with_general_padding(
          lhs, rhs, strides, padding, lhs_dilation, rhs_dilation)

    self._CheckAgainstNumpy(numpy_fun, fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_lhs_shape={}_rhs_shape={}_strides={}_padding={}"
       "_lhs_dilation={}_rhs_dilation={}"
       "_dims={}".format(
           jtu.format_shape_dtype_string(lhs_shape, dtype),
           jtu.format_shape_dtype_string(rhs_shape, dtype),
           strides, padding, lhs_dilation, rhs_dilation,
           ",".join(dim_nums)),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "strides": strides, "padding": padding, "lhs_dilation": lhs_dilation,
       "rhs_dilation": rhs_dilation, "dimension_numbers": dim_nums,
       "feature_group_count": feature_group_count,
       "batch_group_count": batch_group_count,
       "perms": perms, "rng_factory": rng_factory}
      for batch_group_count, feature_group_count in [(1, 1), (2, 1), (1, 2)]
      for lhs_shape, rhs_shape in [
          ((b * batch_group_count, i * feature_group_count, 9, w),
           (j * feature_group_count * batch_group_count, i, 4, 5))
          for w in [0, 10]
          for b, i, j in itertools.product([2, 3], repeat=3)]
      for dtype in float_dtypes for strides in [(1, 1), (2, 1)]
      for padding in [((1, 2), (2, 0)), ((10, 8), (7, 13))]
      for lhs_dilation, rhs_dilation in itertools.product(
          [(1, 1), (1, 2), (1, 4)], repeat=2)
      for rng_factory in [jtu.rand_small]
      for dim_nums, perms in [
        (("NCHW", "OIHW", "NCHW"), ([0, 1, 2, 3], [0, 1, 2, 3])),
        (("NHWC", "HWIO", "NHWC"), ([0, 2, 3, 1], [2, 3, 1, 0])),
        (("NCHW", "HWIO", "NHWC"), ([0, 1, 2, 3], [2, 3, 1, 0])),
      ]))
  def testConvGeneralDilated(self, lhs_shape, rhs_shape, dtype, strides,
                             padding, lhs_dilation, rhs_dilation,
                             feature_group_count, batch_group_count,
                             dimension_numbers, perms, rng_factory):
    rng = rng_factory(self.rng())
    lhs_perm, rhs_perm = perms  # permute to compatible shapes

    def args_maker():
      return [lax.transpose(rng(lhs_shape, dtype), lhs_perm),
              lax.transpose(rng(rhs_shape, dtype), rhs_perm)]

    def fun(lhs, rhs):
      return lax.conv_general_dilated(
          lhs, rhs, strides, padding, lhs_dilation, rhs_dilation,
          dimension_numbers, feature_group_count=feature_group_count,
          batch_group_count=batch_group_count)

    self._CompileAndCheck(fun, args_maker)

  # TODO(mattjj): test conv_general_dilated against numpy

  def testConv0DIsDot(self):
    rng = jtu.rand_default(self.rng())
    def args_maker():
      return [rng((10, 5), onp.float32), rng((5, 7), onp.float32)]
    jnp_fun = partial(lax.conv_general_dilated, window_strides=(),
                      padding='VALID', dimension_numbers=('NC', 'IO', 'NC'))
    self._CompileAndCheck(jnp_fun, args_maker)
    self._CheckAgainstNumpy(jnp_fun, onp.dot, args_maker, tol=.1)


  @staticmethod
  def _conv_transpose_via_grad(data, kernel, strides, padding,
                               rhs_dilation=None, dimension_numbers=None):
    """Helper method: calculates conv transpose via grad for testing."""
    assert len(data.shape) == len(kernel.shape)
    nspatial = len(data.shape) - 2
    one = (1,) * nspatial
    rhs_dilation = rhs_dilation or one
    dn = lax.conv_dimension_numbers(data.shape, kernel.shape,
                                    dimension_numbers)
    in_shape = onp.take(data.shape, dn.lhs_spec)
    in_sdims = in_shape[2:]
    k_shape = onp.take(kernel.shape, dn.rhs_spec)
    k_sdims = k_shape[2:]
    e_k_sdims = [(k-1) * r + 1 for k, r in zip(k_sdims, rhs_dilation)]
    if padding == 'VALID':
      o_sdims = [in_sdims[i]*strides[i] + max(e_k_sdims[i]-strides[i],0)
                 for i in range(nspatial)]
    elif padding == 'SAME':
      o_sdims = [in_sdims[i]*strides[i] for i in range(nspatial)]
    o_shape =  [in_shape[0], k_shape[1]] + o_sdims
    out_spec_inv = [x[0] for x in
                    sorted(enumerate(dn.out_spec), key=lambda x: x[1])]
    o_layout = onp.take(onp.array(o_shape), out_spec_inv)
    placeholder = onp.ones(o_layout, data.dtype)
    conv = lambda x: lax.conv_general_dilated(x, kernel, strides, padding,
                                              one, rhs_dilation, dn)
    _, g = api.vjp(conv, placeholder)
    return g(data)[0]

  @staticmethod
  def _transpose_conv_kernel(data, kernel, dimension_numbers):
    dn = lax.conv_dimension_numbers(data.shape, kernel.shape,
                                    dimension_numbers)
    spatial_axes = onp.array(dn.rhs_spec)[2:]
    for axis in spatial_axes:
      kernel = onp.flip(kernel, axis)
    kernel = onp.swapaxes(kernel, dn.rhs_spec[0], dn.rhs_spec[1])
    return kernel

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_strides={}_padding={}_rhs_dilation={}".format(
           jtu.format_shape_dtype_string(lhs_shape, dtype),
           jtu.format_shape_dtype_string(rhs_shape, dtype), strides, padding, rhs_dilation),
          "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
          "strides": strides, "padding": padding, "rhs_dilation": rhs_dilation,
          "rng_factory": rng_factory, 'dspec': dspec}
      for lhs_shape, rhs_shape in [
          ((b, 9, 10, i), (k, k, j, i))  # NB: i,j flipped in RHS for transpose
          for b, i, j, k in itertools.product([2,3],[2,3],[2,3],[3,4,5])]
      for dtype in float_dtypes
      for strides in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3)]
      for padding in ["VALID", "SAME"]
      for dspec in [('NHWC', 'HWIO', 'NHWC'),]
      for rhs_dilation in [None, (2, 2)]
      for rng_factory in [jtu.rand_small]))
  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def testConvTranspose2DT(self, lhs_shape, rhs_shape, dtype, strides,
                          padding, dspec, rhs_dilation, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    # NB: this test calculates conv_transpose performing identically to the
    # lhs-grad of conv.
    def fun(lhs, rhs):
      return lax.conv_transpose(lhs, rhs, strides, padding,
                                rhs_dilation=rhs_dilation,
                                dimension_numbers=dspec,
                                transpose_kernel=True)

    def fun_via_grad(lhs, rhs):
      return self._conv_transpose_via_grad(lhs, rhs, strides, padding,
                                           rhs_dilation=rhs_dilation,
                                           dimension_numbers=dspec)

    # NB: below just checks for agreement, we're not calling numpy.
    self._CheckAgainstNumpy(fun, fun_via_grad, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_strides={}_padding={}_rhs_dilation={}".format(
           jtu.format_shape_dtype_string(lhs_shape, dtype),
           jtu.format_shape_dtype_string(rhs_shape, dtype), strides, padding, rhs_dilation),
          "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
          "strides": strides, "padding": padding, "rhs_dilation": rhs_dilation,
          "rng_factory": rng_factory, 'dspec': dspec}
      for lhs_shape, rhs_shape in [
          ((b, 9, 10, i), (k, k, i, j))
          for b, i, j, k in itertools.product([2,3],[2,3],[2,3],[3,4,5])]
      for dtype in float_dtypes
      for strides in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3)]
      for padding in ["VALID", "SAME"]
      for dspec in [('NHWC', 'HWIO', 'NHWC'),]
      for rhs_dilation in [None, (2, 2)]
      for rng_factory in [jtu.rand_small]))
  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def testConvTranspose2D(self, lhs_shape, rhs_shape, dtype, strides,
                          padding, dspec, rhs_dilation, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    def fun(lhs, rhs):
      return lax.conv_transpose(lhs, rhs, strides, padding,
                                rhs_dilation=rhs_dilation,
                                dimension_numbers=dspec,
                                transpose_kernel=False)

    def fun_via_grad(lhs, rhs):
      rhs_t = self._transpose_conv_kernel(lhs, rhs, dimension_numbers=dspec)
      return self._conv_transpose_via_grad(lhs, rhs_t, strides, padding,
                                           rhs_dilation=rhs_dilation,
                                           dimension_numbers=dspec)

    # NB: below just checks for agreement, we're not calling numpy.
    self._CheckAgainstNumpy(fun, fun_via_grad, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_strides={}_padding={}_rhs_dilation={}".format(
           jtu.format_shape_dtype_string(lhs_shape, dtype),
           jtu.format_shape_dtype_string(rhs_shape, dtype), strides, padding, rhs_dilation),
          "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
          "strides": strides, "padding": padding, "rhs_dilation": rhs_dilation,
          "rng_factory": rng_factory, 'dspec': dspec}
      for lhs_shape, rhs_shape in [
          ((b, 10, i), (k, i, j))
          for b, i, j, k in itertools.product([2,3],[2,3],[2,3],[3,4,5])]
      for dtype in float_dtypes
      for strides in [(1,), (2,), (3,)]
      for padding in ["VALID", "SAME"]
      for dspec in [('NHC', 'HIO', 'NHC'),]
      for rhs_dilation in [None, (2,)]
      for rng_factory in [jtu.rand_small]))
  def testConvTranspose1D(self, lhs_shape, rhs_shape, dtype, strides,
                          padding, dspec, rhs_dilation, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    def fun(lhs, rhs):
      return lax.conv_transpose(lhs, rhs, strides, padding,
                                dimension_numbers=dspec,
                                rhs_dilation=rhs_dilation,
                                transpose_kernel=False)

    def fun_via_grad(lhs, rhs):
      rhs_t = self._transpose_conv_kernel(lhs, rhs, dimension_numbers=dspec)
      return self._conv_transpose_via_grad(lhs, rhs_t, strides, padding,
                                           rhs_dilation=rhs_dilation,
                                           dimension_numbers=dspec)

    # NB: below just checks for agreement, we're not calling numpy.
    self._CheckAgainstNumpy(fun, fun_via_grad, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_lhs_shape={}_rhs_shape={}_precision={}".format(
          jtu.format_shape_dtype_string(lhs_shape, dtype),
          jtu.format_shape_dtype_string(rhs_shape, dtype),
          precision),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "precision": precision, "rng_factory": rng_factory}
      for lhs_shape in [(3,), (4, 3)] for rhs_shape in [(3,), (3, 6)]
      for dtype in all_dtypes
      for precision in [None, lax.Precision.DEFAULT, lax.Precision.HIGH,
                        lax.Precision.HIGHEST]
      for rng_factory in [jtu.rand_default]))
  def testDot(self, lhs_shape, rhs_shape, dtype, precision, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    self._CompileAndCheck(partial(lax.dot, precision=precision), args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_lhs_shape={}_rhs_shape={}".format(
          jtu.format_shape_dtype_string(lhs_shape, dtype),
          jtu.format_shape_dtype_string(rhs_shape, dtype)),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "rng_factory": rng_factory}
      for lhs_shape in [(3,), (4, 3)] for rhs_shape in [(3,), (3, 6)]
      for dtype in all_dtypes
      for rng_factory in [jtu.rand_default]))
  def testDotAgainstNumpy(self, lhs_shape, rhs_shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    tol = {
      onp.float16: 1e-2,
      onp.float64: max(jtu.default_tolerance()[onp.dtype(onp.float64)], 1e-14),
      onp.complex128: max(jtu.default_tolerance()[onp.dtype(onp.complex128)],
                          1e-14)
    }
    lax_op = partial(lax.dot, precision=lax.Precision.HIGHEST)
    self._CheckAgainstNumpy(lax_op, lax_reference.dot, args_maker, tol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_lhs_contracting={}_rhs_contracting={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               lhs_contracting, rhs_contracting),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "lhs_contracting": lhs_contracting, "rhs_contracting": rhs_contracting,
       "rng_factory": rng_factory}
      for lhs_shape, rhs_shape, lhs_contracting, rhs_contracting in [
          [(3, 5), (2, 5), [1], [1]],
          [(5, 3), (5, 2), [0], [0]],
          [(5, 3, 2), (5, 2, 4), [0], [0]],
          [(5, 3, 2), (5, 2, 4), [0,2], [0,1]],
          [(1, 2, 2, 3), (1, 2, 3, 1), [1], [1]],
          [(3, 2), (2, 4), [1], [0]],
      ]
      for dtype in all_dtypes
      for rng_factory in [jtu.rand_small]))
  def testDotGeneralContractOnly(self, lhs_shape, rhs_shape, dtype,
                                 lhs_contracting, rhs_contracting, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    dimension_numbers = ((lhs_contracting, rhs_contracting), ([], []))

    def fun(lhs, rhs):
      return lax.dot_general(lhs, rhs, dimension_numbers)

    self._CompileAndCheck(fun, args_maker, check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_dimension_numbers={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               dimension_numbers),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "dimension_numbers": dimension_numbers, "rng_factory": rng_factory}
      for lhs_shape, rhs_shape, dimension_numbers in [
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0]))),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1]))),
      ]
      for dtype in all_dtypes
      for rng_factory in [jtu.rand_small]))
  def testDotGeneralContractAndBatch(self, lhs_shape, rhs_shape, dtype,
                                     dimension_numbers, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    def fun(lhs, rhs):
      return lax.dot_general(lhs, rhs, dimension_numbers)

    self._CompileAndCheck(fun, args_maker, check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_dimension_numbers={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               dimension_numbers),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "dimension_numbers": dimension_numbers, "rng_factory": rng_factory}
      for lhs_shape, rhs_shape, dimension_numbers in [
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0]))),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1]))),
      ]
      for dtype in all_dtypes
      for rng_factory in [jtu.rand_small]))
  def testDotGeneralAgainstNumpy(self, lhs_shape, rhs_shape, dtype,
                                 dimension_numbers, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    op = lambda x, y: lax.dot_general(x, y, dimension_numbers)
    numpy_op = lambda x, y: lax_reference.dot_general(x, y, dimension_numbers)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_dtype={}_broadcast_sizes={}".format(
          shape, onp.dtype(dtype).name, broadcast_sizes),
       "shape": shape, "dtype": dtype, "broadcast_sizes": broadcast_sizes,
       "rng_factory": rng_factory}
      for shape in [(), (2, 3)]
      for dtype in default_dtypes
      for broadcast_sizes in [(), (2,), (1, 2)]
      for rng_factory in [jtu.rand_default]))
  def testBroadcast(self, shape, dtype, broadcast_sizes, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.broadcast(x, broadcast_sizes)
    self._CompileAndCheck(op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_broadcast_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), broadcast_sizes),
       "shape": shape, "dtype": dtype, "broadcast_sizes": broadcast_sizes,
       "rng_factory": rng_factory}
      for shape in [(), (2, 3)]
      for dtype in default_dtypes
      for broadcast_sizes in [(), (2,), (1, 2)]
      for rng_factory in [jtu.rand_default]))
  def testBroadcastAgainstNumpy(self, shape, dtype, broadcast_sizes, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.broadcast(x, broadcast_sizes)
    numpy_op = lambda x: lax_reference.broadcast(x, broadcast_sizes)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}_bcdims={}".format(
          jtu.format_shape_dtype_string(inshape, dtype),
          outshape, broadcast_dimensions),
       "inshape": inshape, "dtype": dtype, "outshape": outshape,
       "dimensions": broadcast_dimensions, "rng_factory": rng_factory}
      for inshape, outshape, broadcast_dimensions in [
          ([2], [2, 2], [0]),
          ([2], [2, 2], [1]),
          ([2], [2, 3], [0]),
          ([], [2, 3], []),
          ([1], [2, 3], [1]),
      ]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testBroadcastInDim(self, inshape, dtype, outshape, dimensions, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(inshape, dtype)]
    op = lambda x: lax.broadcast_in_dim(x, outshape, dimensions)
    self._CompileAndCheck(op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_inshape={}_outshape={}_bcdims={}".format(
      jtu.format_shape_dtype_string(inshape, onp.float32),
      outshape, broadcast_dimensions),
      "inshape": inshape, "outshape": outshape,
      "broadcast_dimensions": broadcast_dimensions, "err_msg": err_msg}
    for inshape, outshape, broadcast_dimensions, err_msg in [
      ([2], [2, 2], [0, 1], ('broadcast_dimensions must have length equal to '
                              'operand ndim')),
      ([2, 2], [2], [0, 1], ('target broadcast shape must have equal or higher rank '
                             'to the operand shape')),
      ([2], [2, 3], [2], ('broadcast_in_dim broadcast_dimensions must be a subset of output '
                          'dimensions')),
      ([2], [3], [0], ('operand dimension sizes must either be 1, or be '
                       'equal to their corresponding dimensions in the target broadcast shape')),
      ([2, 2], [2, 2], [1, 0], ('broadcast_dimensions must be strictly increasing')),
    ]))
  def testBroadcastInDimShapeCheck(self, inshape, outshape, broadcast_dimensions, err_msg):
    rng = jtu.rand_default(self.rng())
    x = rng(inshape, onp.float32)
    with self.assertRaisesRegex(TypeError, err_msg):
      lax.broadcast_in_dim(x, shape=outshape, broadcast_dimensions=broadcast_dimensions)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}_bcdims={}".format(
          jtu.format_shape_dtype_string(inshape, dtype),
          outshape, broadcast_dimensions),
       "inshape": inshape, "dtype": dtype, "outshape": outshape,
       "dimensions": broadcast_dimensions, "rng_factory": rng_factory}
      for inshape, outshape, broadcast_dimensions in [
          ([2], [2, 2], [0]),
          ([2], [2, 2], [1]),
          ([2], [2, 3], [0]),
          ([], [2, 3], []),
          ([1], [2, 3], [1]),
      ]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testBroadcastInDimAgainstNumpy(self, inshape, dtype, outshape,
                                     dimensions, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(inshape, dtype)]
    op = lambda x: lax.broadcast_in_dim(x, outshape, dimensions)
    numpy_op = lambda x: lax_reference.broadcast_in_dim(x, outshape, dimensions)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_inshape={}_dimensions={}".format(
      jtu.format_shape_dtype_string(inshape, onp.float32), dimensions),
      "inshape": inshape, "dimensions": dimensions, "error_type": error_type,
      "err_msg": err_msg}
    for inshape, dimensions, error_type, err_msg in [
      ((1, 2, 3), (0, 0), ValueError, 'dimensions are not unique'),
      ((1, 2, 3), (3,), ValueError, 'axis 3 is out of bounds'),
      ((1, 2, 3), (-4,), ValueError, 'axis -4 is out of bounds'),
      ((1, 2, 3), (1,), ValueError, 'cannot select an axis to squeeze out'),
      ((1, 2, 3), (None,), TypeError, 'cannot be interpreted as an integer'),
    ]))
  def testSqueezeShapeCheck(self, inshape, dimensions, error_type, err_msg):
    rng = jtu.rand_default(self.rng())
    x = rng(inshape, onp.float32)
    with self.assertRaisesRegex(error_type, err_msg):
      lax.squeeze(x, dimensions=dimensions)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_dimensions={}".format(
          jtu.format_shape_dtype_string(arg_shape, onp.float32), dimensions),
       "arg_shape": arg_shape, "dimensions": dimensions,
       "rng_factory": rng_factory}
      for arg_shape, dimensions in [
          [(1,), (0,)],
          [(1,), (-1,)],
          [(2, 1, 4), (1,)],
          [(2, 1, 3, 1), (1,)],
          [(2, 1, 3, 1), (1, 3)],
          [(2, 1, 3, 1), (3,)],
      ]
      for rng_factory in [jtu.rand_default]))
  def testSqueeze(self, arg_shape, dimensions, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(arg_shape, onp.float32)]
    op = lambda x: lax.squeeze(x, dimensions)
    numpy_op = lambda x: lax_reference.squeeze(x, dimensions)
    self._CompileAndCheck(op, args_maker)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)
    check_grads(op, args_maker(), 2, ["fwd", "rev"], eps=1.)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          jtu.format_shape_dtype_string(out_shape, dtype)),
       "arg_shape": arg_shape, "out_shape": out_shape, "dtype": dtype,
       "rng_factory": rng_factory}
      for dtype in default_dtypes
      for arg_shape, out_shape in [
          [(3, 4), (12,)], [(2, 1, 4), (8,)], [(2, 2, 4), (2, 8)]
      ]
      for rng_factory in [jtu.rand_default]))
  def testReshape(self, arg_shape, out_shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(arg_shape, dtype)]
    op = lambda x: lax.reshape(x, out_shape)
    self._CompileAndCheck(op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          jtu.format_shape_dtype_string(out_shape, dtype)),
       "arg_shape": arg_shape, "out_shape": out_shape, "dtype": dtype,
       "rng_factory": rng_factory}
      for dtype in default_dtypes
      for arg_shape, out_shape in [
          [(3, 4), (12,)], [(2, 1, 4), (8,)], [(2, 2, 4), (2, 8)]
      ]
      for rng_factory in [jtu.rand_default]))
  def testReshapeAgainstNumpy(self, arg_shape, out_shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(arg_shape, dtype)]
    op = lambda x: lax.reshape(x, out_shape)
    numpy_op = lambda x: lax_reference.reshape(x, out_shape)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_pads={}"
       .format(jtu.format_shape_dtype_string(shape, dtype), pads),
       "shape": shape, "dtype": dtype, "pads": pads, "rng_factory": jtu.rand_small}
      for shape in [(2, 3)]
      for dtype in default_dtypes
      for pads in [[(1, 2, 1), (0, 1, 0)]]))
  def testPad(self, shape, dtype, pads, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    fun = lambda operand: lax.pad(operand, onp.array(0, dtype), pads)
    self._CompileAndCheck(fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_pads={}"
       .format(jtu.format_shape_dtype_string(shape, dtype), pads),
       "shape": shape, "dtype": dtype, "pads": pads, "rng_factory": jtu.rand_small}
      for shape in [(2, 3)]
      for dtype in default_dtypes
      for pads in [
        [(0, 0, 0), (0, 0, 0)],  # no padding
        [(1, 1, 0), (2, 2, 0)],  # only positive edge padding
        [(1, 2, 1), (0, 1, 0)],  # edge padding and interior padding
        [(0, 0, 0), (-1, -1, 0)],  # negative padding
        [(0, 0, 0), (-2, -2, 4)],  # add big dilation then remove from edges
        [(0, 0, 0), (-2, -3, 1)],  # remove everything in one dimension
      ]))
  def testPadAgainstNumpy(self, shape, dtype, pads, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.pad(x, onp.array(0, dtype), pads)
    numpy_op = lambda x: lax_reference.pad(x, onp.array(0, dtype), pads)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  def testReverse(self):
    rev = api.jit(lambda operand: lax.rev(operand, dimensions))

    dimensions = []
    self.assertAllClose(onp.array([0, 1, 2, 3]), rev(onp.array([0, 1, 2, 3])),
                        check_dtypes=False)

    dimensions = [0]
    self.assertAllClose(onp.array([3, 2, 1]), rev(onp.array([1, 2, 3])),
                        check_dtypes=False)

    dimensions = [0, 1]
    self.assertAllClose(onp.array([[6, 5, 4], [3, 2, 1]]),
                        rev(onp.array([[1, 2, 3], [4, 5, 6]])),
                        check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_predshape={}_argshapes={}".format(
          jtu.format_shape_dtype_string(pred_shape, onp.bool_),
          jtu.format_shape_dtype_string(arg_shape, arg_dtype)),
       "pred_shape": pred_shape, "arg_shape": arg_shape, "arg_dtype": arg_dtype,
       "rng_factory": rng_factory}
      for arg_shape in [(), (3,), (2, 3)]
      for pred_shape in ([(), arg_shape] if arg_shape else [()])
      for arg_dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testSelect(self, pred_shape, arg_shape, arg_dtype, rng_factory):
    def args_maker():
      return [rng(pred_shape, onp.bool_), rng(arg_shape, arg_dtype),
              rng(arg_shape, arg_dtype)]
    rng = rng_factory(self.rng())
    return self._CompileAndCheck(lax.select, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_predshape={}_argshapes={}".format(
          jtu.format_shape_dtype_string(pred_shape, onp.bool_),
          jtu.format_shape_dtype_string(arg_shape, arg_dtype)),
       "pred_shape": pred_shape, "arg_shape": arg_shape, "arg_dtype": arg_dtype,
       "rng_factory": rng_factory}
      for arg_shape in [(), (3,), (2, 3)]
      for pred_shape in ([(), arg_shape] if arg_shape else [()])
      for arg_dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testSelectAgainstNumpy(self, pred_shape, arg_shape, arg_dtype, rng_factory):
    def args_maker():
      return [rng(pred_shape, onp.bool_), rng(arg_shape, arg_dtype),
              rng(arg_shape, arg_dtype)]
    rng = rng_factory(self.rng())
    return self._CheckAgainstNumpy(lax.select, lax_reference.select, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}_start_indices={}_limit_indices={}_strides={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, limit_indices, strides),
       "shape": shape, "dtype": dtype, "starts": start_indices,
       "limits": limit_indices, "strides": strides, "rng_factory": rng_factory}
      for shape, start_indices, limit_indices, strides in [
        [(3,), (1,), (2,), None],
        [(7,), (4,), (7,), None],
        [(5,), (1,), (5,), (2,)],
        [(8,), (1,), (6,), (2,)],
        [(5, 3), (1, 1), (3, 2), None],
        [(5, 3), (1, 1), (3, 1), None],
        [(7, 5, 3), (4, 0, 1), (7, 1, 3), None],
        [(5, 3), (1, 1), (2, 1), (1, 1)],
        [(5, 3), (1, 1), (5, 3), (2, 1)],
      ]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testSlice(self, shape, dtype, starts, limits, strides, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.slice(x, starts, limits, strides)
    self._CompileAndCheck(op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}_start_indices={}_limit_indices={}_strides={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, limit_indices, strides),
       "shape": shape, "dtype": dtype, "starts": start_indices,
       "limits": limit_indices, "strides": strides, "rng_factory": rng_factory}
      for shape, start_indices, limit_indices, strides in [
        [(3,), (1,), (2,), None],
        [(7,), (4,), (7,), None],
        [(5,), (1,), (5,), (2,)],
        [(8,), (1,), (6,), (2,)],
        [(5, 3), (1, 1), (3, 2), None],
        [(5, 3), (1, 1), (3, 1), None],
        [(7, 5, 3), (4, 0, 1), (7, 1, 3), None],
        [(5, 3), (1, 1), (2, 1), (1, 1)],
        [(5, 3), (1, 1), (5, 3), (2, 1)],
      ]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testSliceAgainstNumpy(self, shape, dtype, starts, limits,
                            strides, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.slice(x, starts, limits, strides)
    numpy_op = lambda x: lax_reference.slice(x, starts, limits, strides)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_start_indices={}_size_indices={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, size_indices),
       "shape": shape, "dtype": dtype, "start_indices": start_indices,
       "size_indices": size_indices, "rng_factory": rng_factory}
      for shape, start_indices, size_indices in [
        [(3,), onp.array((1,)), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(5, 3), onp.array((1, 1)), (3, 1)],
        [(7, 5, 3), onp.array((4, 1, 0)), (2, 0, 1)],
      ]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testDynamicSlice(self, shape, dtype, start_indices, size_indices, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype), onp.array(start_indices)]
    op = lambda x, starts: lax.dynamic_slice(x, starts, size_indices)
    self._CompileAndCheck(op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_start_indices={}_size_indices={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, size_indices),
       "shape": shape, "dtype": dtype, "start_indices": start_indices,
       "size_indices": size_indices, "rng_factory": rng_factory}
      for shape, start_indices, size_indices in [
        [(3,), (1,), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
      ]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testDynamicSliceAgainstNumpy(self, shape, dtype, start_indices,
                                   size_indices, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype), onp.array(start_indices)]
    op = lambda x, s: lax.dynamic_slice(x, s, size_indices)
    numpy_op = lambda x, s: lax_reference.dynamic_slice(x, s, size_indices)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  def testDynamicSliceInDim(self):
    # Regression test for mixed type problem in dynamic_slice_in_dim.
    rng = jtu.rand_default(self.rng())
    x = rng((6, 7), onp.int32)
    onp.testing.assert_equal(lax.dynamic_slice_in_dim(x, 2, 3), x[2:5])

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_start_indices={}_update_shape={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, update_shape),
       "shape": shape, "dtype": dtype, "start_indices": start_indices,
       "update_shape": update_shape, "rng_factory": rng_factory}
      for shape, start_indices, update_shape in [
        [(3,), (1,), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
      ]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testDynamicUpdateSlice(self, shape, dtype, start_indices, update_shape,
                             rng_factory):
    rng = rng_factory(self.rng())

    def args_maker():
      return [rng(shape, dtype), rng(update_shape, dtype),
              onp.array(start_indices)]

    self._CompileAndCheck(lax.dynamic_update_slice, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_start_indices={}_update_shape={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, update_shape),
       "shape": shape, "dtype": dtype, "start_indices": start_indices,
       "update_shape": update_shape, "rng_factory": rng_factory}
      for shape, start_indices, update_shape in [
        [(3,), (1,), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
      ]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testDynamicUpdateSliceAgainstNumpy(self, shape, dtype, start_indices,
                                         update_shape, rng_factory):
    rng = rng_factory(self.rng())

    def args_maker():
      return [rng(shape, dtype), rng(update_shape, dtype),
              onp.array(start_indices)]

    self._CheckAgainstNumpy(lax.dynamic_update_slice,
                            lax_reference.dynamic_update_slice, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_perm={}".format(
          jtu.format_shape_dtype_string(shape, dtype), perm),
       "shape": shape, "dtype": dtype, "perm": perm, "rng_factory": rng_factory}
      for shape, perm in [
        [(3, 4), (1, 0)],
        [(3, 4), (0, 1)],
        [(3, 4, 5), (2, 1, 0)],
        [(3, 4, 5), (1, 0, 2)],
      ]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testTranspose(self, shape, dtype, perm, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.transpose(x, perm)
    self._CompileAndCheck(op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_perm={}".format(
          jtu.format_shape_dtype_string(shape, dtype), perm),
       "shape": shape, "dtype": dtype, "perm": perm, "rng_factory": rng_factory}
      for shape, perm in [
        [(3, 4), (1, 0)],
        [(3, 4), (0, 1)],
        [(3, 4, 5), (2, 1, 0)],
        [(3, 4, 5), (1, 0, 2)],
      ]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testTransposeAgainstNumpy(self, shape, dtype, perm, rng_factory):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.transpose(x, perm)
    numpy_op = lambda x: lax_reference.transpose(x, perm)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_inshape={}_reducedims={}_initval={}"
       .format(op.__name__, jtu.format_shape_dtype_string(shape, dtype), dims,
               init_val),
       "op": op, "init_val": init_val, "shape": shape, "dtype": dtype,
       "dims": dims, "rng_factory": rng_factory}
      for init_val, op, types in [
          (0, lax.add, default_dtypes),
          (1, lax.mul, default_dtypes),
          (0, lax.max, all_dtypes), # non-monoidal
          (-onp.inf, lax.max, float_dtypes),
          (dtypes.iinfo(onp.int32).min, lax.max, [onp.int32]),
          # (dtypes.iinfo(onp.int64).min, lax.max, [onp.int64]),  # TODO fails
          (dtypes.iinfo(onp.uint32).min, lax.max, [onp.uint32]),
          (dtypes.iinfo(onp.uint64).min, lax.max, [onp.uint64]),
          (onp.inf, lax.min, float_dtypes),
          (dtypes.iinfo(onp.int32).max, lax.min, [onp.int32]),
          # (dtypes.iinfo(onp.int64).max, lax.min, [onp.int64]),  # TODO fails
          (dtypes.iinfo(onp.uint32).max, lax.min, [onp.uint32]),
          (dtypes.iinfo(onp.uint64).max, lax.min, [onp.uint64]),
      ]
      for dtype in types
      for shape, dims in [
          [(3, 4, 5), (0,)], [(3, 4, 5), (1, 2)],
          [(3, 4, 5), (0, 2)], [(3, 4, 5), (0, 1, 2)]
      ]
      for rng_factory in [
          jtu.rand_default if dtypes.issubdtype(dtype, onp.integer)
          else jtu.rand_small]))
  def testReduce(self, op, init_val, shape, dtype, dims, rng_factory):
    rng = rng_factory(self.rng())
    init_val = onp.asarray(init_val, dtype=dtype)
    fun = lambda operand, init_val: lax.reduce(operand, init_val, op, dims)
    args_maker = lambda: [rng(shape, dtype), init_val]
    self._CompileAndCheck(fun, args_maker)

    # we separately test the version that uses a concrete init_val because it
    # can hit different code paths
    fun = lambda operand: lax.reduce(operand, init_val, op, dims)
    args_maker = lambda: [rng(shape, dtype)]
    self._CompileAndCheck(fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_dtype={}_padding={}"
       .format(op.__name__, onp.dtype(dtype).name, padding),
       "op": op, "init_val": init_val, "dtype": dtype, "padding": padding,
       "rng_factory": rng_factory}
      for init_val, op, dtypes in [
          (0, lax.add, [onp.float32]),
          (-onp.inf, lax.max, [onp.float32]),
          (onp.inf, lax.min, [onp.float32]),
      ]
      for dtype in dtypes
      for padding in ["VALID", "SAME"]
      for rng_factory in [jtu.rand_small]))
  def testReduceWindow(self, op, init_val, dtype, padding, rng_factory):
    rng = rng_factory(self.rng())
    init_val = onp.asarray(init_val, dtype=dtype)

    all_configs = itertools.chain(
        itertools.product(
            [(4, 6)],
            [(2, 1), (1, 2)],
            [(1, 1), (2, 1), (1, 2)]),
        itertools.product(
            [(3, 2, 4, 6)], [(1, 1, 2, 1), (2, 1, 2, 1)],
            [(1, 2, 2, 1), (1, 1, 1, 1)]))

    def fun(operand, init_val):
      return lax.reduce_window(operand, init_val, op, dims, strides, padding)

    # pylint: disable=cell-var-from-loop
    for shape, dims, strides in all_configs:
      args_maker = lambda: [rng(shape, dtype), init_val]
      self._CompileAndCheck(fun, args_maker)
    # pylint: enable=cell-var-from-loop

    # we separately test the version that uses a concrete init_val because it
    # can hit different code paths
    def fun(operand):
      return lax.reduce_window(operand, init_val, op, dims, strides, padding)

    # pylint: disable=cell-var-from-loop
    for shape, dims, strides in all_configs:
      args_maker = lambda: [rng(shape, dtype)]
      self._CompileAndCheck(fun, args_maker)
    # pylint: enable=cell-var-from-loop

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_shape={}_axis={}"
       .format(op.__name__, jtu.format_shape_dtype_string(shape, dtype), axis),
       "op": op, "onp_op": onp_op, "shape": shape, "dtype": dtype,
       "axis": axis, "rng_factory": rng_factory}
      for op, onp_op, types in [
          (lax.cumsum, onp.cumsum, default_dtypes),
          (lax.cumprod, onp.cumprod, default_dtypes),
      ]
      for dtype in types
      for shape in [[10], [3, 4, 5]]
      for axis in range(len(shape))
      for rng_factory in [
          jtu.rand_default if dtypes.issubdtype(dtype, onp.integer)
          else jtu.rand_small]))
  def testCumulativeReduce(self, op, onp_op, shape, dtype, axis, rng_factory):
    rng = rng_factory(self.rng())
    fun = partial(op, axis=axis)
    onp_fun = partial(onp_op, axis=axis, dtype=dtype)
    args_maker = lambda: [rng(shape, dtype)]
    self._CompileAndCheck(fun, args_maker)
    self._CheckAgainstNumpy(fun, onp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "shape": shape, "dtype": dtype, "axis": axis}
      for dtype in all_dtypes
      for shape in [(5,), (5, 7)]
      for axis in [-1, len(shape) - 1]))
  def testSort(self, shape, dtype, axis):
    # TODO(b/141131288): enable complex-valued sorts on TPU.
    if (onp.issubdtype(dtype, onp.complexfloating) and (
        (jtu.device_under_test() == "cpu" and jax.lib.version <= (0, 1, 47)) or
         jtu.device_under_test() == "tpu")):
      raise SkipTest("Complex-valued sort not implemented")
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    fun = lambda x: lax.sort(x, dimension=axis)
    self._CompileAndCheck(fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
        "shape": shape, "dtype": dtype, "axis": axis}
      for dtype in all_dtypes
      for shape in [(5,), (5, 7)]
      for axis in [-1, len(shape) - 1]))
  def testSortAgainstNumpy(self, shape, dtype, axis):
    # TODO(b/141131288): enable complex-valued sorts on TPU.
    if (onp.issubdtype(dtype, onp.complexfloating) and (
        (jtu.device_under_test() == "cpu" and jax.lib.version <= (0, 1, 47)) or
         jtu.device_under_test() == "tpu")):
      raise SkipTest("Complex-valued sort not implemented")
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.sort(x, dimension=axis)
    numpy_op = lambda x: lax_reference.sort(x, axis)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_keyshape={}_valshape={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, key_dtype),
          jtu.format_shape_dtype_string(shape, val_dtype),
          axis),
       "shape": shape, "key_dtype": key_dtype, "val_dtype": val_dtype,
       "axis": axis}
      for key_dtype in float_dtypes + complex_dtypes + int_dtypes + uint_dtypes
      for val_dtype in [onp.float32, onp.int32, onp.uint32]
      for shape in [(3,), (5, 3)]
      for axis in [-1, len(shape) - 1]))
  def testSortKeyVal(self, shape, key_dtype, val_dtype, axis):
    # TODO(b/141131288): enable complex-valued sorts on TPU.
    if (onp.issubdtype(key_dtype, onp.complexfloating) and (
        (jtu.device_under_test() == "cpu" and jax.lib.version <= (0, 1, 47)) or
         jtu.device_under_test() == "tpu")):
      raise SkipTest("Complex-valued sort not implemented")
    rng = jtu.rand_default(self.rng())
    # This test relies on the property that wherever keys are tied, values are
    # too, since we don't guarantee the same ordering of values with equal keys.
    # To avoid that case, we generate unique keys (globally in the key array).
    def args_maker():
      flat_keys = onp.arange(onp.prod(shape, dtype=int), dtype=key_dtype)
      keys = self.rng().permutation(flat_keys).reshape(shape)
      values = rng(shape, val_dtype)
      return keys, values

    fun = lambda keys, values: lax.sort_key_val(keys, values, axis)
    self._CompileAndCheck(fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_keyshape={}_valshape={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, key_dtype),
          jtu.format_shape_dtype_string(shape, val_dtype),
          axis),
       "shape": shape, "key_dtype": key_dtype, "val_dtype": val_dtype,
       "axis": axis}
      for key_dtype in float_dtypes + complex_dtypes + int_dtypes + uint_dtypes
      for val_dtype in [onp.float32, onp.int32, onp.uint32]
      for shape in [(3,), (5, 3)]
      for axis in [-1, len(shape) - 1]))
  def testSortKeyValAgainstNumpy(self, shape, key_dtype, val_dtype, axis):
    # TODO(b/141131288): enable complex-valued sorts on TPU.
    if (onp.issubdtype(key_dtype, onp.complexfloating) and (
        (jtu.device_under_test() == "cpu" and jax.lib.version <= (0, 1, 47)) or
         jtu.device_under_test() == "tpu")):
      raise SkipTest("Complex-valued sort not implemented")
    rng = jtu.rand_default(self.rng())
    # This test relies on the property that wherever keys are tied, values are
    # too, since we don't guarantee the same ordering of values with equal keys.
    # To avoid that case, we generate unique keys (globally in the key array).
    def args_maker():
      flat_keys = onp.arange(onp.prod(shape, dtype=int), dtype=key_dtype)
      keys = self.rng().permutation(flat_keys).reshape(shape)
      values = rng(shape, val_dtype)
      return keys, values

    op = lambda ks, vs: lax.sort_key_val(ks, vs, axis)
    numpy_op = lambda ks, vs: lax_reference.sort_key_val(ks, vs, axis)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_k={}".format(
          jtu.format_shape_dtype_string(shape, dtype), k),
       "rng_factory": rng_factory, "shape": shape, "dtype": dtype, "k": k}
      for dtype in [onp.float32, onp.int32, onp.uint32]
      for shape in [(3,), (5, 3)]
      for k in [1, 3]
      for rng_factory in [jtu.rand_default]))
  def testTopK(self, shape, dtype, k, rng_factory):
    def args_maker():
      flat_values = onp.arange(onp.prod(shape, dtype=int), dtype=dtype)
      values = self.rng().permutation(flat_values).reshape(shape)
      return [values]
    def reference_top_k(x):
      bcast_idxs = onp.broadcast_to(onp.arange(shape[-1], dtype=onp.int32), shape)
      sorted_vals, sorted_idxs = lax_reference.sort_key_val(x, bcast_idxs)
      return sorted_vals[..., :-k-1:-1], sorted_idxs[..., :-k-1:-1]
    op = lambda vs: lax.top_k(vs, k=k)
    self._CheckAgainstNumpy(op, reference_top_k, args_maker)
    self._CompileAndCheck(op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_lhs_shape={}_rhs_shape={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype)),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "rng_factory": rng_factory}
      for lhs_shape, rhs_shape in [((3, 2), (2, 4)),
                                   ((5, 3, 2), (5, 2, 4)),
                                   ((1, 2, 2, 3), (1, 2, 3, 1))]
      for dtype in float_dtypes
      for rng_factory in [jtu.rand_small]))
  def testBatchMatMul(self, lhs_shape, rhs_shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    arg_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    self._CompileAndCheck(lax.batch_matmul, arg_maker)

  def testCollapse(self):

    @api.jit
    def collapse_first_two(x):
      return lax.collapse(x, 0, 2)

    self.assertEqual((6,), collapse_first_two(onp.zeros((2, 3))).shape)
    self.assertEqual((6, 4), collapse_first_two(onp.zeros((2, 3, 4))).shape)
    self.assertEqual((2, 3, 4),
                     collapse_first_two(onp.zeros((1, 2, 3, 4))).shape)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_idxs={}_axes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), idxs, axes),
       "shape": shape, "dtype": dtype, "idxs": idxs, "axes": axes, "rng_factory": rng_factory}
      for dtype in all_dtypes
      for shape, idxs, axes in [
          [(3, 4, 5), (onp.array([0, 2, 1]),), (0,)],
          [(3, 4, 5), (onp.array([-1, -2]),), (0,)],
          [(3, 4, 5), (onp.array([0, 2]), onp.array([1, 3])), (0, 1)],
          [(3, 4, 5), (onp.array([0, 2]), onp.array([1, 3])), (0, 2)],
      ]
      for rng_factory in [jtu.rand_default]))
  def testIndexTake(self, shape, dtype, idxs, axes, rng_factory):
    rng = rng_factory(self.rng())
    rand_idxs = lambda: tuple(rng(e.shape, e.dtype) for e in idxs)
    args_maker = lambda: [rng(shape, dtype), rand_idxs()]
    fun = lambda src, idxs: lax.index_take(src, idxs, axes)
    self._CompileAndCheck(fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_idxs={}_dnums={}_slice_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), idxs, dnums,
          slice_sizes),
       "shape": shape, "dtype": dtype, "idxs": idxs, "dnums": dnums,
       "slice_sizes": slice_sizes, "rng_factory": rng_factory,
       "rng_idx_factory": rng_idx_factory}
      for dtype in all_dtypes
      for shape, idxs, dnums, slice_sizes in [
          ((5,), onp.array([[0], [2]]), lax.GatherDimensionNumbers(
            offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)),
            (1,)),
          ((10,), onp.array([[0], [0], [0]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)),
            (2,)),
          ((10, 5,), onp.array([[0], [2], [1]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)),
            (1, 3)),
          ((10, 5), onp.array([[0, 2], [1, 0]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0, 1)),
            (1, 3)),
      ]
      for rng_idx_factory in [partial(jtu.rand_int, high=max(shape))]
      for rng_factory in [jtu.rand_default]))
  def testGather(self, shape, dtype, idxs, dnums, slice_sizes, rng_factory,
                 rng_idx_factory):
    rng = rng_factory(self.rng())
    rng_idx = rng_idx_factory(self.rng())
    rand_idxs = lambda: rng_idx(idxs.shape, idxs.dtype)
    args_maker = lambda: [rng(shape, dtype), rand_idxs()]
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    self._CompileAndCheck(fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_idxs={}_update={}_dnums={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          idxs, update_shape, dnums),
       "arg_shape": arg_shape, "dtype": dtype, "idxs": idxs,
       "update_shape": update_shape, "dnums": dnums,
       "rng_factory": rng_factory, "rng_idx_factory": rng_idx_factory}
      for dtype in float_dtypes
      for arg_shape, idxs, update_shape, dnums in [
          ((5,), onp.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
            update_window_dims=(), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,))),
          ((10,), onp.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(),
            scatter_dims_to_operand_dims=(0,))),
          ((10, 5,), onp.array([[0], [2], [1]]), (3, 3), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,))),
      ]
      for rng_idx_factory in [partial(jtu.rand_int, high=max(arg_shape))]
      for rng_factory in [jtu.rand_default]))
  def testScatterAdd(self, arg_shape, dtype, idxs, update_shape, dnums,
                     rng_factory, rng_idx_factory):
    rng = rng_factory(self.rng())
    rng_idx = rng_idx_factory(self.rng())
    rand_idxs = lambda: rng_idx(idxs.shape, idxs.dtype)
    args_maker = lambda: [rng(arg_shape, dtype), rand_idxs(),
                          rng(update_shape, dtype)]
    fun = partial(lax.scatter_add, dimension_numbers=dnums)
    self._CompileAndCheck(fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_idxs={}_update={}_dnums={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          idxs, update_shape, dnums),
       "arg_shape": arg_shape, "dtype": dtype, "idxs": idxs,
       "update_shape": update_shape, "dnums": dnums,
       "rng_factory": rng_factory, "rng_idx_factory": rng_idx_factory}
      for dtype in float_dtypes
      for arg_shape, idxs, update_shape, dnums in [
          ((5,), onp.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
            update_window_dims=(), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,))),
          ((10,), onp.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(),
            scatter_dims_to_operand_dims=(0,))),
          ((10, 5,), onp.array([[0], [2], [1]]), (3, 3), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,))),
      ]
      for rng_idx_factory in [partial(jtu.rand_int, high=max(arg_shape))]
      for rng_factory in [jtu.rand_default]))
  def testScatterMin(self, arg_shape, dtype, idxs, update_shape, dnums,
                     rng_factory, rng_idx_factory):
    rng = rng_factory(self.rng())
    rng_idx = rng_idx_factory(self.rng())
    rand_idxs = lambda: rng_idx(idxs.shape, idxs.dtype)
    args_maker = lambda: [rng(arg_shape, dtype), rand_idxs(),
                          rng(update_shape, dtype)]
    fun = partial(lax.scatter_min, dimension_numbers=dnums)
    self._CompileAndCheck(fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_idxs={}_update={}_dnums={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          idxs, update_shape, dnums),
       "arg_shape": arg_shape, "dtype": dtype, "idxs": idxs,
       "update_shape": update_shape, "dnums": dnums,
       "rng_factory": rng_factory, "rng_idx_factory": rng_idx_factory}
      for dtype in float_dtypes
      for arg_shape, idxs, update_shape, dnums in [
          ((5,), onp.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
            update_window_dims=(), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,))),
          ((10,), onp.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(),
            scatter_dims_to_operand_dims=(0,))),
          ((10, 5,), onp.array([[0], [2], [1]]), (3, 3), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,))),
      ]
      for rng_idx_factory in [partial(jtu.rand_int, high=max(arg_shape))]
      for rng_factory in [jtu.rand_default]))
  def testScatterMax(self, arg_shape, dtype, idxs, update_shape, dnums,
                     rng_factory, rng_idx_factory):
    rng = rng_factory(self.rng())
    rng_idx = rng_idx_factory(self.rng())
    rand_idxs = lambda: rng_idx(idxs.shape, idxs.dtype)
    args_maker = lambda: [rng(arg_shape, dtype), rand_idxs(),
                          rng(update_shape, dtype)]
    fun = partial(lax.scatter_max, dimension_numbers=dnums)
    self._CompileAndCheck(fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_idxs={}_update={}_dnums={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          idxs, update_shape, dnums),
       "arg_shape": arg_shape, "dtype": dtype, "idxs": idxs,
       "update_shape": update_shape, "dnums": dnums,
       "rng_factory": rng_factory, "rng_idx_factory": rng_idx_factory}
      for dtype in float_dtypes
      for arg_shape, idxs, update_shape, dnums in [
          ((5,), onp.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
            update_window_dims=(), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,))),
          ((10,), onp.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(),
            scatter_dims_to_operand_dims=(0,))),
          ((10, 5,), onp.array([[0], [2], [1]]), (3, 3), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,))),
      ]
      for rng_idx_factory in [partial(jtu.rand_int, high=max(arg_shape))]
      for rng_factory in [jtu.rand_default]))
  def testScatter(self, arg_shape, dtype, idxs, update_shape, dnums,
                     rng_factory, rng_idx_factory):
    rng = rng_factory(self.rng())
    rng_idx = rng_idx_factory(self.rng())
    rand_idxs = lambda: rng_idx(idxs.shape, idxs.dtype)
    args_maker = lambda: [rng(arg_shape, dtype), rand_idxs(),
                          rng(update_shape, dtype)]
    fun = partial(lax.scatter, dimension_numbers=dnums)
    self._CompileAndCheck(fun, args_maker)

  def testIssue831(self):
    # Tests the DeviceTuple constant handler
    def f(x):
      g = lambda *args: args[1]
      return api.jit(lax.fori_loop, static_argnums=(2,))( 0, 10, g, x)

    api.jit(f)(1.)  # doesn't crash

  def testReshapeWithUnusualShapes(self):
    ans = lax.reshape(onp.ones((3,), onp.float32), (lax.add(1, 2), 1))
    self.assertAllClose(ans, onp.ones((3, 1), onp.float32))

    self.assertRaisesRegex(
      TypeError,
      "Shapes must be 1D sequences of concrete values of integer type.*",
      lambda: lax.reshape(onp.ones(3,), (onp.array([3, 1]),)))

    self.assertRaisesRegex(
      TypeError,
      "Shapes must be 1D sequences of concrete values of integer type.*",
      lambda: lax.reshape(onp.ones(3,), (1.5, 2.0)))

  @jtu.skip_on_devices("tpu")  # S16 not supported on TPU
  def testDynamicSliceTypeErrors(self):
    self.assertRaisesRegex(
      TypeError,
      "index arguments to dynamic_slice must be integers of the same type",
      lambda: lax.dynamic_slice(onp.ones((3, 4), dtype=onp.float32),
                                (onp.int32(1), onp.int16(2)), (2, 2)))

  @jtu.skip_on_devices("tpu")  # S16 not supported on TPU
  def testDynamicUpdateSliceTypeErrors(self):
    self.assertRaisesRegex(
      TypeError,
      "index arguments to dynamic_update_slice must be integers of the same "
      "type",
      lambda: lax.dynamic_update_slice(onp.ones((3, 4), dtype=onp.float32),
                                       onp.zeros((2, 2), dtype=onp.float32),
                                       (onp.int32(1), onp.int16(2))))

  def test_tie_in_error(self):
    with core.skipping_checks():
      with self.assertRaisesRegex(
          TypeError, ".* of type .*tuple.* is not a valid JAX type"):
        api.make_jaxpr(lambda x: lax.tie_in((x, x), 1))(1.)

  def test_primitive_jaxtype_error(self):
    with core.skipping_checks():
      with self.assertRaisesRegex(
          TypeError, "Argument .* of type .* is not a valid JAX type"):
        lax.add(1, 'hi')

class LazyConstantTest(jtu.JaxTestCase):
  def _Check(self, make_const, expected):
    # check casting to ndarray works
    asarray_result = onp.asarray(make_const())

    # check passing as an argument works (should hit constant handler)
    zero = onp.array(0, expected.dtype)
    argument_result = lax.add(zero, make_const())

    # check looping into a compiled computation works
    jit_result = api.jit(lambda x: lax.add(x, make_const()))(zero)

    # ensure they're all the same
    self.assertAllClose(asarray_result, expected)
    self.assertAllClose(argument_result, expected)
    self.assertAllClose(jit_result, expected)

    # ensure repr doesn't crash
    repr(make_const())

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_fill={}".format(
          jtu.format_shape_dtype_string(shape, dtype) if dtype else shape,
          fill_value),
       "shape": shape, "dtype": dtype, "fill_value": fill_value}
      for dtype in itertools.chain(default_dtypes, [None])
      for shape in [(), (3,), (2, 3), (2, 3, 4), (1001, 1001)]
      for fill_value in [0, 1, onp.pi]))
  def testFilledConstant(self, shape, fill_value, dtype):
    make_const = lambda: lax.full(shape, fill_value, dtype)
    expected = onp.full(shape, fill_value,
                        dtype or dtypes.result_type(fill_value))
    self._Check(make_const, expected)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_dim={}".format(
          jtu.format_shape_dtype_string(shape, dtype), dimension),
       "shape": shape, "dtype": dtype, "dimension": dimension}
      for dtype in default_dtypes
      for shape in [(), (3,), (2, 3), (2, 3, 4),
                    # TODO(mattjj): re-enable
                    # (1001, 1001), (101, 101, 101),
                    ]
      for dimension in range(len(shape))))
  def testIotaConstant(self, dtype, shape, dimension):
    make_const = lambda: lax.broadcasted_iota(dtype, shape, dimension)

    arr = onp.arange(shape[dimension], dtype=dtypes.canonicalize_dtype(dtype))
    singleton_shape = [1] * len(shape)
    singleton_shape[dimension] = shape[dimension]
    expected = onp.broadcast_to(arr.reshape(singleton_shape), shape)

    self._Check(make_const, expected)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axes),
       "shape": shape, "dtype": dtype, "axes": axes}
      for dtype in default_dtypes
      for shape, axes in [
          [(2, 3), (0, 1)],
          [(2, 3, 4), (0, 1)],
          [(2, 3, 4), (0, 2)],
          [(2, 3, 4), (1, 2)],
          [(2, 3, 4), (0, 1, 2)],
          [(2, 3, 4, 2), (0, 1, 2)],
          [(2, 3, 4, 2), (0, 2, 3)],
          [(1001, 1001), (0, 1)],
      ]))
  @jtu.skip_on_devices("tpu")  # TODO(mattjj): investigate failure
  def testDeltaConstant(self, dtype, shape, axes):
    make_const = lambda: lax._delta(dtype, shape, axes)
    # don't check the asarray case, just assume it's right
    expected = onp.asarray(make_const())
    self._Check(make_const, expected)

  def testBroadcastInDim(self):
    arr = lax.full((2, 1), 1.) + 1.
    arr_onp = onp.full((2, 1), 1.) + 1.
    expected = lax_reference.broadcast_in_dim(arr_onp, (2, 1, 3), (0, 2))
    make_const = lambda: lax.broadcast_in_dim(arr, (2, 1, 3), (0, 2))
    self._Check(make_const, expected)


if __name__ == '__main__':
  absltest.main()
