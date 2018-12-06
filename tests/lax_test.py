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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
from functools import partial
import itertools

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp
import numpy.random as npr

from jax import api
from jax import core
from jax import lax
from jax import test_util as jtu
from jax import lax_reference
from jax.config import config
from jax.interpreters import xla
from jax.lib import xla_bridge

config.parse_flags_with_absl()
FLAGS = config.FLAGS


def num_float_bits(dtype):
  return onp.finfo(xla_bridge.canonicalize_dtype(dtype)).bits


### lax tests

# For standard unops and binops, we can generate a large number of tests on
# arguments of appropriate shapes and dtypes using the following table.

float_dtypes = [onp.float32, onp.float64]
complex_dtypes = [onp.complex64]
int_dtypes = [onp.int32, onp.int64]
bool_dtypes = [onp.bool_]
default_dtypes = float_dtypes + int_dtypes
all_dtypes = float_dtypes + complex_dtypes + int_dtypes + bool_dtypes

compatible_shapes = [[(3,)], [(3, 4), (3, 1), (1, 4)], [(2, 3, 4), (2, 1, 4)]]

OpRecord = collections.namedtuple("OpRecord",
                                  ["op", "nargs", "dtypes", "rng", "tol"])


def op_record(op, nargs, dtypes, rng, tol=1e-5):
  return OpRecord(op, nargs, dtypes, rng, tol)

LAX_OPS = [
    op_record(lax.neg, 1, default_dtypes + complex_dtypes, jtu.rand_small()),
    op_record(lax.sign, 1, default_dtypes, jtu.rand_small()),
    op_record(lax.floor, 1, float_dtypes, jtu.rand_small()),
    op_record(lax.ceil, 1, float_dtypes, jtu.rand_small()),
    op_record(lax.round, 1, float_dtypes, jtu.rand_default()),

    op_record(lax.is_finite, 1, float_dtypes, jtu.rand_small()),

    op_record(lax.exp, 1, float_dtypes + complex_dtypes, jtu.rand_small()),
    op_record(lax.expm1, 1, float_dtypes + complex_dtypes, jtu.rand_small()),
    op_record(lax.log, 1, float_dtypes + complex_dtypes, jtu.rand_positive()),
    op_record(lax.log1p, 1, float_dtypes + complex_dtypes, jtu.rand_positive()),
    op_record(lax.tanh, 1, float_dtypes + complex_dtypes, jtu.rand_small()),
    op_record(lax.sin, 1, float_dtypes + complex_dtypes, jtu.rand_default()),
    op_record(lax.cos, 1, float_dtypes + complex_dtypes, jtu.rand_default()),
    op_record(lax.atan2, 2, float_dtypes, jtu.rand_default()),

    op_record(lax.sqrt, 1, float_dtypes + complex_dtypes, jtu.rand_positive()),
    op_record(lax.rsqrt, 1, float_dtypes + complex_dtypes, jtu.rand_positive()),
    op_record(lax.square, 1, float_dtypes + complex_dtypes, jtu.rand_default()),
    op_record(lax.reciprocal, 1, float_dtypes + complex_dtypes, jtu.rand_positive()),
    op_record(lax.tan, 1, float_dtypes, jtu.rand_default()),
    op_record(lax.asin, 1, float_dtypes, jtu.rand_small()),
    op_record(lax.acos, 1, float_dtypes, jtu.rand_small()),
    op_record(lax.atan, 1, float_dtypes, jtu.rand_small()),
    op_record(lax.sinh, 1, float_dtypes + complex_dtypes, jtu.rand_default()),
    op_record(lax.cosh, 1, float_dtypes + complex_dtypes, jtu.rand_default()),
    op_record(lax.asinh, 1, float_dtypes + complex_dtypes, jtu.rand_positive()),
    op_record(lax.acosh, 1, float_dtypes + complex_dtypes, jtu.rand_positive()),

    op_record(lax.lgamma, 1, float_dtypes, jtu.rand_positive()),
    op_record(lax.digamma, 1, float_dtypes, jtu.rand_positive()),
    op_record(lax.erf, 1, float_dtypes, jtu.rand_small()),
    op_record(lax.erfc, 1, float_dtypes, jtu.rand_small()),
    op_record(lax.erf_inv, 1, float_dtypes, jtu.rand_small(), tol=1e-2),

    op_record(lax.real, 1, complex_dtypes, jtu.rand_default()),
    op_record(lax.imag, 1, complex_dtypes, jtu.rand_default()),
    op_record(lax.complex, 2, [onp.float32], jtu.rand_default()),
    op_record(lax.conj, 1, [onp.float32] + complex_dtypes, jtu.rand_default()),
    op_record(lax.abs, 1, default_dtypes + complex_dtypes, jtu.rand_default()),
    op_record(lax.pow, 2, float_dtypes + complex_dtypes, jtu.rand_positive()),

    op_record(lax.bitwise_and, 2, bool_dtypes, jtu.rand_small()),
    op_record(lax.bitwise_not, 1, bool_dtypes, jtu.rand_small()),
    op_record(lax.bitwise_or, 2, bool_dtypes, jtu.rand_small()),
    op_record(lax.bitwise_xor, 2, bool_dtypes, jtu.rand_small()),

    op_record(lax.add, 2, default_dtypes + complex_dtypes, jtu.rand_small()),
    op_record(lax.sub, 2, default_dtypes + complex_dtypes, jtu.rand_small()),
    op_record(lax.mul, 2, default_dtypes + complex_dtypes, jtu.rand_small()),
    op_record(lax.div, 2, default_dtypes + complex_dtypes, jtu.rand_nonzero()),
    op_record(lax.rem, 2, default_dtypes, jtu.rand_nonzero()),

    op_record(lax.max, 2, default_dtypes, jtu.rand_small()),
    op_record(lax.min, 2, default_dtypes, jtu.rand_small()),

    op_record(lax.eq, 2, all_dtypes, jtu.rand_some_equal()),
    op_record(lax.ne, 2, all_dtypes, jtu.rand_small()),
    op_record(lax.ge, 2, default_dtypes, jtu.rand_small()),
    op_record(lax.gt, 2, default_dtypes, jtu.rand_small()),
    op_record(lax.le, 2, default_dtypes, jtu.rand_small()),
    op_record(lax.lt, 2, default_dtypes, jtu.rand_small()),
]

CombosWithReplacement = itertools.combinations_with_replacement


class LaxTest(jtu.JaxTestCase):
  """Numerical tests for LAX operations."""

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": jtu.format_test_name_suffix(
          rec.op.__name__, shapes, itertools.repeat(dtype)),
       "op": rec.op, "rng": rec.rng, "shapes": shapes, "dtype": dtype}
      for rec in LAX_OPS
      for shape_group in compatible_shapes
      for shapes in CombosWithReplacement(shape_group, rec.nargs)
      for dtype in rec.dtypes))
  def testOp(self, op, rng, shapes, dtype):
    args_maker = lambda: [rng(shape, dtype) for shape in shapes]
    self._CompileAndCheck(op, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": jtu.format_test_name_suffix(
          rec.op.__name__, shapes, itertools.repeat(dtype)),
       "op": rec.op, "rng": rec.rng, "shapes": shapes, "dtype": dtype,
       "tol": rec.tol}
      for rec in LAX_OPS
      for shape_group in compatible_shapes
      for shapes in CombosWithReplacement(shape_group, rec.nargs)
      for dtype in rec.dtypes))
  def testOpAgainstNumpy(self, op, rng, shapes, dtype, tol):
    args_maker = lambda: [rng(shape, dtype) for shape in shapes]
    numpy_op = getattr(lax_reference, op.__name__)
    self._CheckAgainstNumpy(op, numpy_op, args_maker, tol=tol)

  # TODO test shift_left, shift_right_arithmetic, shift_right_logical

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_from_dtype={}_to_dtype={}".format(
          from_dtype, to_dtype),
       "from_dtype": from_dtype, "to_dtype": to_dtype, "rng": rng}
      for from_dtype, to_dtype in itertools.product(
          [onp.float32, onp.int32, "float32", "int32"], repeat=2)
      for rng in [jtu.rand_default()]))
  def testConvertElementType(self, from_dtype, to_dtype, rng):
    args_maker = lambda: [rng((2, 3), from_dtype)]
    op = lambda x: lax.convert_element_type(x, to_dtype)
    self._CompileAndCheck(op, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_from_dtype={}_to_dtype={}"
       .format(from_dtype, to_dtype),
       "from_dtype": from_dtype, "to_dtype": to_dtype, "rng": rng}
      for from_dtype, to_dtype in itertools.product(
          [onp.float32, onp.int32, "float32", "int32"], repeat=2)
      for rng in [jtu.rand_default()]))
  def testConvertElementTypeAgainstNumpy(self, from_dtype, to_dtype, rng):
    args_maker = lambda: [rng((2, 3), from_dtype)]
    op = lambda x: lax.convert_element_type(x, to_dtype)
    numpy_op = lambda x: lax_reference.convert_element_type(x, to_dtype)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_from_dtype={}_to_dtype={}"
       .format(from_dtype, to_dtype),
       "from_dtype": from_dtype, "to_dtype": to_dtype, "rng": rng}
      for from_dtype, to_dtype in itertools.product(
          [onp.float32, onp.int32, "float32", "int32"], repeat=2)
      for rng in [jtu.rand_default()]))
  def testBitcastConvertType(self, from_dtype, to_dtype, rng):
    args_maker = lambda: [rng((2, 3), from_dtype)]
    op = lambda x: lax.bitcast_convert_type(x, to_dtype)
    self._CompileAndCheck(op, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_from_dtype={}_to_dtype={}"
       .format(from_dtype, to_dtype),
       "from_dtype": from_dtype, "to_dtype": to_dtype, "rng": rng}
      for from_dtype, to_dtype in itertools.product(
          [onp.float32, onp.int32, "float32", "int32"], repeat=2)
      for rng in [jtu.rand_default()]))
  def testBitcastConvertTypeAgainstNumpy(self, from_dtype, to_dtype, rng):
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
       "max_shape": max_shape, "dtype": dtype, "rng": rng}
      for min_shape, operand_shape, max_shape in [
          [(), (2, 3), ()],
          [(2, 3), (2, 3), ()],
          [(), (2, 3), (2, 3)],
          [(2, 3), (2, 3), (2, 3)],
      ]
      for dtype in default_dtypes
      for rng in [jtu.rand_default()]))
  def testClamp(self, min_shape, operand_shape, max_shape, dtype, rng):
    shapes = [min_shape, operand_shape, max_shape]
    args_maker = lambda: [rng(shape, dtype) for shape in shapes]
    self._CompileAndCheck(lax.clamp, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_min_shape={}_operand_shape={}_max_shape={}".format(
          jtu.format_shape_dtype_string(min_shape, dtype),
          jtu.format_shape_dtype_string(operand_shape, dtype),
          jtu.format_shape_dtype_string(max_shape, dtype)),
       "min_shape": min_shape, "operand_shape": operand_shape,
       "max_shape": max_shape, "dtype": dtype, "rng": rng}
      for min_shape, operand_shape, max_shape in [
          [(), (2, 3), ()],
          [(2, 3), (2, 3), ()],
          [(), (2, 3), (2, 3)],
          [(2, 3), (2, 3), (2, 3)],
      ]
      for dtype in default_dtypes
      for rng in [jtu.rand_default()]))
  def testClampAgainstNumpy(self, min_shape, operand_shape, max_shape, dtype,
                            rng):
    shapes = [min_shape, operand_shape, max_shape]
    args_maker = lambda: [rng(shape, dtype) for shape in shapes]
    self._CheckAgainstNumpy(lax.clamp, lax_reference.clamp, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dim={}_baseshape=[{}]_dtype={}_narrs={}".format(
          dim, ",".join(str(d) for d in base_shape), onp.dtype(dtype).name,
          num_arrs),
       "dim": dim, "base_shape": base_shape, "dtype": dtype,
       "num_arrs": num_arrs, "rng": rng}
      for num_arrs in [3]
      for dtype in default_dtypes
      for base_shape in [(4,), (3, 4), (2, 3, 4)]
      for dim in range(len(base_shape))
      for rng in [jtu.rand_default()]))
  def testConcatenate(self, dim, base_shape, dtype, num_arrs, rng):
    shapes = [base_shape[:dim] + (size,) + base_shape[dim+1:]
              for size, _ in zip(itertools.cycle([3, 1, 4]), range(num_arrs))]
    args_maker = lambda: [rng(shape, dtype) for shape in shapes]
    op = lambda *args: lax.concatenate(args, dim)
    self._CompileAndCheck(op, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dim={}_baseshape=[{}]_dtype={}_narrs={}".format(
          dim, ",".join(str(d) for d in base_shape), onp.dtype(dtype).name,
          num_arrs),
       "dim": dim, "base_shape": base_shape, "dtype": dtype,
       "num_arrs": num_arrs, "rng": rng}
      for num_arrs in [3]
      for dtype in default_dtypes
      for base_shape in [(4,), (3, 4), (2, 3, 4)]
      for dim in range(len(base_shape))
      for rng in [jtu.rand_default()]))
  def testConcatenateAgainstNumpy(self, dim, base_shape, dtype, num_arrs, rng):
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
          "strides": strides, "padding": padding, "rng": rng}
      for lhs_shape, rhs_shape in [
          ((b, i, 9, 10), (j, i, 4, 5))
          for b, i, j in itertools.product([2, 3], repeat=3)]
      for dtype in [onp.float32]
      for strides in [(1, 1), (1, 2), (2, 1)]
      for padding in ["VALID", "SAME"]
      for rng in [jtu.rand_small()]))
  def testConv(self, lhs_shape, rhs_shape, dtype, strides, padding, rng):
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    def fun(lhs, rhs):
      return lax.conv(lhs, rhs, strides, padding)

    self._CompileAndCheck(fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_strides={}_padding={}".format(
           jtu.format_shape_dtype_string(lhs_shape, dtype),
           jtu.format_shape_dtype_string(rhs_shape, dtype), strides, padding),
          "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
          "strides": strides, "padding": padding, "rng": rng}
      for lhs_shape, rhs_shape in [
          ((b, i, 9, 10), (j, i, 4, 5))
          for b, i, j in itertools.product([2, 3], repeat=3)]
      for dtype in [onp.float32]
      for strides in [(1, 1), (1, 2), (2, 1)]
      for padding in ["VALID", "SAME"]
      for rng in [jtu.rand_small()]))
  def testConvAgainstNumpy(self, lhs_shape, rhs_shape, dtype, strides, padding,
                           rng):
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
       "rhs_dilation": rhs_dilation, "rng": rng}
      for lhs_shape, rhs_shape in [
          ((b, i, 9, 10), (j, i, 4, 5))
          for b, i, j in itertools.product([1, 2, 3], repeat=3)]
      for dtype in [onp.float32] for strides in [(1, 1), (1, 2), (2, 1)]
      for padding in [((0, 0), (0, 0)), ((1, 2), (2, 0))]
      for lhs_dilation, rhs_dilation in itertools.product(
          [(1, 1), (1, 2), (2, 2)], repeat=2)
      for rng in [jtu.rand_small()]))
  def testConvWithGeneralPadding(self, lhs_shape, rhs_shape, dtype, strides,
                                 padding, lhs_dilation, rhs_dilation, rng):
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    def fun(lhs, rhs):
      return lax.conv_with_general_padding(
          lhs, rhs, strides, padding, lhs_dilation, rhs_dilation)

    self._CompileAndCheck(fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_lhs_shape={}_rhs_shape={}_strides={}_padding={}"
       "_lhs_dilation={}_rhs_dilation={}".format(
           jtu.format_shape_dtype_string(lhs_shape, dtype),
           jtu.format_shape_dtype_string(rhs_shape, dtype),
           strides, padding, lhs_dilation, rhs_dilation),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "strides": strides, "padding": padding, "lhs_dilation": lhs_dilation,
       "rhs_dilation": rhs_dilation, "rng": rng}
      for lhs_shape, rhs_shape in [
          ((b, i, 9, 10), (j, i, 4, 5))
          for b, i, j in itertools.product([1, 2, 3], repeat=3)]
      for dtype in [onp.float32] for strides in [(1, 1), (1, 2), (2, 1)]
      for padding in [((0, 0), (0, 0)), ((1, 2), (2, 0))]
      for lhs_dilation, rhs_dilation in itertools.product(
          [(1, 1), (1, 2), (2, 2)], repeat=2)
      for rng in [jtu.rand_small()]))
  def DISABLED_testConvWithGeneralPaddingAgainstNumpy(
      self, lhs_shape, rhs_shape, dtype, strides, padding, lhs_dilation,
      rhs_dilation, rng):
    # TODO(mattjj): make this test pass
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    def fun(lhs, rhs):
      return lax.conv_with_general_padding(
          lhs, rhs, strides, padding, lhs_dilation, rhs_dilation)

    def numpy_fun(lhs, rhs):
      return lax_reference.conv_with_general_padding(
          lhs, rhs, strides, padding, lhs_dilation, rhs_dilation)

    self._CheckAgainstNumpy(fun, numpy_fun, args_maker)

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
       "perms": perms, "rng": rng}
      for lhs_shape, rhs_shape in [
          ((b, i, 9, 10), (j, i, 4, 5))
          for b, i, j in itertools.product([2, 3], repeat=3)]
      for dtype in [onp.float32] for strides in [(1, 1), (2, 1)]
      for padding in [((1, 2), (2, 0))]
      for lhs_dilation, rhs_dilation in itertools.product(
          [(1, 1), (1, 2)], repeat=2)
      for rng in [jtu.rand_small()]
      for dim_nums, perms in [(("NCHW", "OIHW", "NCHW"),
                              ([0, 1, 2, 3], [0, 1, 2, 3])),
                             (("NHWC", "HWIO", "NHWC"),
                              ([0, 2, 3, 1], [2, 3, 1, 0]))]))
  def testConvGeneralDilated(self, lhs_shape, rhs_shape, dtype, strides,
                             padding, lhs_dilation, rhs_dilation,
                             dimension_numbers, perms, rng):
    lhs_perm, rhs_perm = perms  # permute to compatible shapes

    def args_maker():
      return [lax.transpose(rng(lhs_shape, dtype), lhs_perm),
              lax.transpose(rng(rhs_shape, dtype), rhs_perm)]

    def fun(lhs, rhs):
      return lax.conv_general_dilated(
          lhs, rhs, strides, padding, lhs_dilation, rhs_dilation,
          dimension_numbers)

    self._CompileAndCheck(fun, args_maker, check_dtypes=True)

  # TODO(mattjj): test conv_general_dilated against numpy

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_lhs_shape={}_rhs_shape={}".format(
          jtu.format_shape_dtype_string(lhs_shape, dtype),
          jtu.format_shape_dtype_string(rhs_shape, dtype)),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "rng": rng}
      for lhs_shape in [(3,), (4, 3)] for rhs_shape in [(3,), (3, 6)]
      for dtype in float_dtypes
      for rng in [jtu.rand_default()]))
  def testDot(self, lhs_shape, rhs_shape, dtype, rng):
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    self._CompileAndCheck(lax.dot, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_lhs_shape={}_rhs_shape={}".format(
          jtu.format_shape_dtype_string(lhs_shape, dtype),
          jtu.format_shape_dtype_string(rhs_shape, dtype)),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "rng": rng}
      for lhs_shape in [(3,), (4, 3)] for rhs_shape in [(3,), (3, 6)]
      for dtype in float_dtypes
      for rng in [jtu.rand_default()]))
  def testDotAgainstNumpy(self, lhs_shape, rhs_shape, dtype, rng):
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    self._CheckAgainstNumpy(lax.dot, lax_reference.dot, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_lhs_contracting={}_rhs_contracting={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               lhs_contracting, rhs_contracting),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "lhs_contracting": lhs_contracting, "rhs_contracting": rhs_contracting,
       "rng": rng}
      for lhs_shape, rhs_shape, lhs_contracting, rhs_contracting in [
          # these all fail with "RuntimeError: Unimplemented: Dot with
          # non-standard contracting dimensions not implemented."
          # [(3, 5), (2, 5), [1], [1]],
          # [(5, 3), (5, 2), [0], [0]],
          # [(5, 3, 2), (5, 2, 4), [0], [0]],
          # [(5, 3, 2), (5, 2, 4), [0,2], [0,1]],
          # [(1, 2, 2, 3), (1, 2, 3, 1), [1], [1]],
          [(3, 2), (2, 4), [1], [0]],
      ]
      for dtype in float_dtypes
      for rng in [jtu.rand_small()]))
  def testDotGeneralContractOnly(self, lhs_shape, rhs_shape, dtype,
                                 lhs_contracting, rhs_contracting, rng):
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
       "dimension_numbers": dimension_numbers, "rng": rng}
      for lhs_shape, rhs_shape, dimension_numbers in [
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0]))),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1]))),
      ]
      for dtype in float_dtypes
      for rng in [jtu.rand_small()]))
  def testDotGeneralContractAndBatch(self, lhs_shape, rhs_shape, dtype,
                                     dimension_numbers, rng):
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
       "dimension_numbers": dimension_numbers, "rng": rng}
      for lhs_shape, rhs_shape, dimension_numbers in [
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0]))),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1]))),
      ]
      for dtype in float_dtypes
      for rng in [jtu.rand_small()]))
  def testDotGeneralAgainstNumpy(self, lhs_shape, rhs_shape, dtype,
                                 dimension_numbers, rng):
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    op = lambda x, y: lax.dot_general(x, y, dimension_numbers)
    numpy_op = lambda x, y: lax_reference.dot_general(x, y, dimension_numbers)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_dtype={}_broadcast_sizes={}".format(
          shape, onp.dtype(dtype).name, broadcast_sizes),
       "shape": shape, "dtype": dtype, "broadcast_sizes": broadcast_sizes,
       "rng": rng}
      for shape in [(), (2, 3)]
      for dtype in default_dtypes
      for broadcast_sizes in [(), (2,), (1, 2)]
      for rng in [jtu.rand_default()]))
  def testBroadcast(self, shape, dtype, broadcast_sizes, rng):
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.broadcast(x, broadcast_sizes)
    self._CompileAndCheck(op, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_broadcast_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), broadcast_sizes),
       "shape": shape, "dtype": dtype, "broadcast_sizes": broadcast_sizes,
       "rng": rng}
      for shape in [(), (2, 3)]
      for dtype in default_dtypes
      for broadcast_sizes in [(), (2,), (1, 2)]
      for rng in [jtu.rand_default()]))
  def testBroadcastAgainstNumpy(self, shape, dtype, broadcast_sizes, rng):
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.broadcast(x, broadcast_sizes)
    numpy_op = lambda x: lax_reference.broadcast(x, broadcast_sizes)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}_bcdims={}".format(
          jtu.format_shape_dtype_string(inshape, dtype),
          outshape, broadcast_dimensions),
       "inshape": inshape, "dtype": dtype, "outshape": outshape,
       "dimensions": broadcast_dimensions, "rng": rng}
      for inshape, outshape, broadcast_dimensions in [
          ([2], [2, 2], [0]),
          ([2], [2, 2], [1]),
          ([2], [2, 3], [0]),
          ([], [2, 3], []),
      ]
      for dtype in default_dtypes
      for rng in [jtu.rand_default()]))
  def testBroadcastInDim(self, inshape, dtype, outshape, dimensions, rng):
    args_maker = lambda: [rng(inshape, dtype)]
    op = lambda x: lax.broadcast_in_dim(x, outshape, dimensions)
    self._CompileAndCheck(op, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}_bcdims={}".format(
          jtu.format_shape_dtype_string(inshape, dtype),
          outshape, broadcast_dimensions),
       "inshape": inshape, "dtype": dtype, "outshape": outshape,
       "dimensions": broadcast_dimensions, "rng": rng}
      for inshape, outshape, broadcast_dimensions in [
          ([2], [2, 2], [0]),
          ([2], [2, 2], [1]),
          ([2], [2, 3], [0]),
          ([], [2, 3], []),
      ]
      for dtype in default_dtypes
      for rng in [jtu.rand_default()]))
  def testBroadcastInDimAgainstNumpy(self, inshape, dtype, outshape,
                                     dimensions, rng):
    args_maker = lambda: [rng(inshape, dtype)]
    op = lambda x: lax.broadcast_in_dim(x, outshape, dimensions)
    numpy_op = lambda x: lax_reference.broadcast_in_dim(x, outshape, dimensions)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          jtu.format_shape_dtype_string(out_shape, dtype)),
       "arg_shape": arg_shape, "out_shape": out_shape, "dtype": dtype,
       "rng": rng}
      for dtype in default_dtypes
      for arg_shape, out_shape in [
          [(3, 4), (12,)], [(2, 1, 4), (8,)], [(2, 2, 4), (2, 8)]
      ]
      for rng in [jtu.rand_default()]))
  def testReshape(self, arg_shape, out_shape, dtype, rng):
    args_maker = lambda: [rng(arg_shape, dtype)]
    op = lambda x: lax.reshape(x, out_shape)
    self._CompileAndCheck(op, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          jtu.format_shape_dtype_string(out_shape, dtype)),
       "arg_shape": arg_shape, "out_shape": out_shape, "dtype": dtype,
       "rng": rng}
      for dtype in default_dtypes
      for arg_shape, out_shape in [
          [(3, 4), (12,)], [(2, 1, 4), (8,)], [(2, 2, 4), (2, 8)]
      ]
      for rng in [jtu.rand_default()]))
  def testReshapeAgainstNumpy(self, arg_shape, out_shape, dtype, rng):
    args_maker = lambda: [rng(arg_shape, dtype)]
    op = lambda x: lax.reshape(x, out_shape)
    numpy_op = lambda x: lax_reference.reshape(x, out_shape)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_pads={}"
       .format(jtu.format_shape_dtype_string(shape, dtype), pads),
       "shape": shape, "dtype": dtype, "pads": pads, "rng": jtu.rand_small()}
      for shape in [(2, 3)]
      for dtype in default_dtypes
      for pads in [[(1, 2, 1), (0, 1, 0)]]))
  def testPad(self, shape, dtype, pads, rng):
    args_maker = lambda: [rng(shape, dtype)]
    fun = lambda operand: lax.pad(operand, onp.array(0, dtype), pads)
    self._CompileAndCheck(fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_pads={}"
       .format(jtu.format_shape_dtype_string(shape, dtype), pads),
       "shape": shape, "dtype": dtype, "pads": pads, "rng": jtu.rand_small()}
      for shape in [(2, 3)]
      for dtype in default_dtypes
      for pads in [[(1, 2, 1), (0, 1, 0)]]))
  def testPadAgainstNumpy(self, shape, dtype, pads, rng):
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.pad(x, onp.array(0, dtype), pads)
    numpy_op = lambda x: lax_reference.pad(x, onp.array(0, dtype), pads)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  def testReverse(self):
    rev = api.jit(lambda operand: lax.rev(operand, dimensions))

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
       "rng": rng}
      for arg_shape in [(), (3,), (2, 3)]
      for pred_shape in ([(), arg_shape] if arg_shape else [()])
      for arg_dtype in default_dtypes
      for rng in [jtu.rand_default()]))
  def testSelect(self, pred_shape, arg_shape, arg_dtype, rng):

    def args_maker():
      return [rng(pred_shape, onp.bool_), rng(arg_shape, arg_dtype),
              rng(arg_shape, arg_dtype)]

    return self._CompileAndCheck(lax.select, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_predshape={}_argshapes={}".format(
          jtu.format_shape_dtype_string(pred_shape, onp.bool_),
          jtu.format_shape_dtype_string(arg_shape, arg_dtype)),
       "pred_shape": pred_shape, "arg_shape": arg_shape, "arg_dtype": arg_dtype,
       "rng": rng}
      for arg_shape in [(), (3,), (2, 3)]
      for pred_shape in ([(), arg_shape] if arg_shape else [()])
      for arg_dtype in default_dtypes
      for rng in [jtu.rand_default()]))
  def testSelectAgainstNumpy(self, pred_shape, arg_shape, arg_dtype, rng):

    def args_maker():
      return [rng(pred_shape, onp.bool_), rng(arg_shape, arg_dtype),
              rng(arg_shape, arg_dtype)]

    return self._CheckAgainstNumpy(lax.select, lax_reference.select, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}_start_indices={}_limit_indices={}_strides={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, limit_indices, strides),
       "shape": shape, "dtype": dtype, "starts": start_indices,
       "limits": limit_indices, "strides": strides, "rng": rng}
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
      for rng in [jtu.rand_default()]))
  def testSlice(self, shape, dtype, starts, limits, strides, rng):
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.slice(x, starts, limits, strides)
    self._CompileAndCheck(op, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}_start_indices={}_limit_indices={}_strides={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, limit_indices, strides),
       "shape": shape, "dtype": dtype, "starts": start_indices,
       "limits": limit_indices, "strides": strides, "rng": rng}
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
      for rng in [jtu.rand_default()]))
  def testSliceAgainstNumpy(self, shape, dtype, starts, limits,
                            strides, rng):
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.slice(x, starts, limits, strides)
    numpy_op = lambda x: lax_reference.slice(x, starts, limits, strides)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_start_indices={}_size_indices={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, size_indices),
       "shape": shape, "dtype": dtype, "start_indices": start_indices,
       "size_indices": size_indices, "rng": rng}
      for shape, start_indices, size_indices in [
        [(3,), (1,), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
      ]
      for dtype in default_dtypes
      for rng in [jtu.rand_default()]))
  def testDynamicSlice(self, shape, dtype, start_indices, size_indices, rng):
    args_maker = lambda: [rng(shape, dtype), onp.array(start_indices)]
    op = lambda x, starts: lax.dynamic_slice(x, starts, size_indices)
    self._CompileAndCheck(op, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_start_indices={}_size_indices={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, size_indices),
       "shape": shape, "dtype": dtype, "start_indices": start_indices,
       "size_indices": size_indices, "rng": rng}
      for shape, start_indices, size_indices in [
        [(3,), (1,), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
      ]
      for dtype in default_dtypes
      for rng in [jtu.rand_default()]))
  def testDynamicSliceAgainstNumpy(self, shape, dtype, start_indices,
                                   size_indices, rng):
    args_maker = lambda: [rng(shape, dtype), onp.array(start_indices)]
    op = lambda x, s: lax.dynamic_slice(x, s, size_indices)
    numpy_op = lambda x, s: lax_reference.dynamic_slice(x, s, size_indices)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_start_indices={}_update_shape={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, update_shape),
       "shape": shape, "dtype": dtype, "start_indices": start_indices,
       "update_shape": update_shape, "rng": rng}
      for shape, start_indices, update_shape in [
        [(3,), (1,), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
      ]
      for dtype in default_dtypes
      for rng in [jtu.rand_default()]))
  def testDynamicUpdateSlice(self, shape, dtype, start_indices, update_shape,
                             rng):

    def args_maker():
      return [rng(shape, dtype), rng(update_shape, dtype),
              onp.array(start_indices)]

    self._CompileAndCheck(lax.dynamic_update_slice, args_maker,
                          check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_start_indices={}_update_shape={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, update_shape),
       "shape": shape, "dtype": dtype, "start_indices": start_indices,
       "update_shape": update_shape, "rng": rng}
      for shape, start_indices, update_shape in [
        [(3,), (1,), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
      ]
      for dtype in default_dtypes
      for rng in [jtu.rand_default()]))
  def testDynamicUpdateSliceAgainstNumpy(self, shape, dtype, start_indices,
                                         update_shape, rng):

    def args_maker():
      return [rng(shape, dtype), rng(update_shape, dtype),
              onp.array(start_indices)]

    self._CheckAgainstNumpy(lax.dynamic_update_slice,
                            lax_reference.dynamic_update_slice, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_perm={}".format(
          jtu.format_shape_dtype_string(shape, dtype), perm),
       "shape": shape, "dtype": dtype, "perm": perm, "rng": rng}
      for shape, perm in [
        [(3, 4), (1, 0)],
        [(3, 4), (0, 1)],
        [(3, 4, 5), (2, 1, 0)],
        [(3, 4, 5), (1, 0, 2)],
      ]
      for dtype in default_dtypes
      for rng in [jtu.rand_default()]))
  def testTranspose(self, shape, dtype, perm, rng):
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.transpose(x, perm)
    self._CompileAndCheck(op, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_perm={}".format(
          jtu.format_shape_dtype_string(shape, dtype), perm),
       "shape": shape, "dtype": dtype, "perm": perm, "rng": rng}
      for shape, perm in [
        [(3, 4), (1, 0)],
        [(3, 4), (0, 1)],
        [(3, 4, 5), (2, 1, 0)],
        [(3, 4, 5), (1, 0, 2)],
      ]
      for dtype in default_dtypes
      for rng in [jtu.rand_default()]))
  def testTransposeAgainstNumpy(self, shape, dtype, perm, rng):
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.transpose(x, perm)
    numpy_op = lambda x: lax_reference.transpose(x, perm)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_inshape={}_reducedims={}"
       .format(op.__name__, jtu.format_shape_dtype_string(shape, dtype), dims),
       "op": op, "init_val": init_val, "shape": shape, "dtype": dtype,
       "dims": dims, "rng": rng}
      for init_val, op, dtypes in [
          (0, lax.add, default_dtypes),
          (-onp.inf, lax.max, float_dtypes),
          (onp.iinfo(onp.int32).min, lax.max, [onp.int32]),
          (onp.iinfo(onp.int64).min, lax.max, [onp.int64]),
          (onp.iinfo(onp.uint32).min, lax.max, [onp.uint32]),
          (onp.iinfo(onp.uint64).min, lax.max, [onp.uint64]),
          (onp.inf, lax.min, float_dtypes),
          (onp.iinfo(onp.int32).max, lax.min, [onp.int32]),
          (onp.iinfo(onp.int64).max, lax.min, [onp.int64]),
          (onp.iinfo(onp.uint32).max, lax.min, [onp.uint32]),
          (onp.iinfo(onp.uint64).max, lax.min, [onp.uint64]),
      ]
      for dtype in dtypes
      for shape, dims in [
          [(3, 4, 5), (0,)], [(3, 4, 5), (1, 2)],
          [(3, 4, 5), (0, 2)], [(3, 4, 5), (0, 1, 2)]
      ]
      for rng in [jtu.rand_small()]))
  def testReduce(self, op, init_val, shape, dtype, dims, rng):
    init_val = onp.asarray(init_val, dtype=dtype)
    fun = lambda operand, init_val: lax.reduce(operand, init_val, op, dims)
    args_maker = lambda: [rng(shape, dtype), init_val]
    self._CompileAndCheck(fun, args_maker, check_dtypes=True)

    # we separately test the version that uses a concrete init_val because it
    # can hit different code paths
    fun = lambda operand: lax.reduce(operand, init_val, op, dims)
    args_maker = lambda: [rng(shape, dtype)]
    self._CompileAndCheck(fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_dtype={}_padding={}"
       .format(op.__name__, onp.dtype(dtype).name, padding),
       "op": op, "init_val": init_val, "dtype": dtype, "padding": padding,
       "rng": rng}
      for init_val, op, dtypes in [
          (0, lax.add, [onp.float32]),
          (-onp.inf, lax.max, [onp.float32]),
          (onp.inf, lax.min, [onp.float32]),
      ]
      for dtype in dtypes
      for padding in ["VALID", "SAME"]
      for rng in [jtu.rand_small()]))
  def testReduceWindow(self, op, init_val, dtype, padding, rng):
    init_val = onp.asarray(init_val, dtype=dtype)

    # We need this conditional and the corresponding loop logic to be in the
    # test method, rather than at the parameterized test level, because it
    # depends on FLAGS for the device under test.
    if FLAGS.jax_test_dut == "tpu":
      all_configs = [((4, 6), (2, 1), (1, 2))]
    else:
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
      self._CompileAndCheck(fun, args_maker, check_dtypes=True)
    # pylint: enable=cell-var-from-loop

    # we separately test the version that uses a concrete init_val because it
    # can hit different code paths
    def fun(operand):
      return lax.reduce_window(operand, init_val, op, dims, strides, padding)

    # pylint: disable=cell-var-from-loop
    for shape, dims, strides in all_configs:
      args_maker = lambda: [rng(shape, dtype)]
      self._CompileAndCheck(fun, args_maker, check_dtypes=True)
    # pylint: enable=cell-var-from-loop

  # TODO(b/205052657): enable more tests when supported
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "rng": rng, "shape": shape, "dtype": dtype, "axis": axis}
      for dtype in [onp.float32, onp.int32, onp.uint32]
      for shape in [(5,), (5, 7)]
      for axis in [-1, len(shape) - 1]
      for rng in [jtu.rand_default()]))
  def testSort(self, shape, dtype, axis, rng):
    if len(shape) > 1 and not FLAGS.jax_test_dut.startswith("tpu"):
      msg = "sort only implemented for R1 on non-TPU backends"
      return absltest.unittest.skip(msg)

    args_maker = lambda: [rng(shape, dtype)]
    fun = lambda x: lax.sort(x, axis)
    self._CompileAndCheck(fun, args_maker, check_dtypes=True)

  # TODO(b/205052657): enable more tests when supported
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "rng": rng, "shape": shape, "dtype": dtype, "axis": axis}
      for dtype in [onp.float32, onp.int32, onp.uint32]
      for shape in [(5,), (5, 7)]
      for axis in [-1, len(shape) - 1]
      for rng in [jtu.rand_default()]))
  def testSortAgainstNumpy(self, shape, dtype, axis, rng):
    if len(shape) > 1 and not FLAGS.jax_test_dut.startswith("tpu"):
      msg = "sort only implemented for R1 on non-TPU backends"
      return absltest.unittest.skip(msg)

    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.sort(x, axis)
    numpy_op = lambda x: lax_reference.sort(x, axis)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  # TODO(b/205052657): enable more tests when supported
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_keyshape={}_valshape={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, key_dtype),
          jtu.format_shape_dtype_string(shape, val_dtype),
          axis),
       "rng": rng, "shape": shape,
       "key_dtype": key_dtype, "val_dtype": val_dtype, "axis": axis}
      for key_dtype in [onp.float32, onp.int32, onp.uint32]
      for val_dtype in [onp.float32, onp.int32, onp.uint32]
      for shape in [(3,), (5, 3)]
      for axis in [-1, len(shape) - 1]
      for rng in [jtu.rand_default()]))
  def testSortKeyVal(self, shape, key_dtype, val_dtype, axis, rng):
    if len(shape) > 1 and not FLAGS.jax_test_dut.startswith("tpu"):
      msg = "sort_key_val only implemented for R1 non-TPU backends"
      return absltest.unittest.skip(msg)

    # This test relies on the property that wherever keys are tied, values are
    # too, since we don't guarantee the same ordering of values with equal keys.
    # To avoid that case, we generate unique keys (globally in the key array).
    perm_rng = onp.random.RandomState(0)
    def args_maker():
      flat_keys = onp.arange(onp.prod(shape, dtype=int), dtype=key_dtype)
      keys = perm_rng.permutation(flat_keys).reshape(shape)
      values = rng(shape, val_dtype)
      return keys, values

    fun = lambda keys, values: lax.sort_key_val(keys, values, axis)
    self._CompileAndCheck(fun, args_maker, check_dtypes=True)

  # TODO(b/205052657): enable more tests when supported
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_keyshape={}_valshape={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, key_dtype),
          jtu.format_shape_dtype_string(shape, val_dtype),
          axis),
       "rng": rng, "shape": shape,
       "key_dtype": key_dtype, "val_dtype": val_dtype, "axis": axis}
      for key_dtype in [onp.float32, onp.int32, onp.uint32]
      for val_dtype in [onp.float32, onp.int32, onp.uint32]
      for shape in [(3,), (5, 3)]
      for axis in [-1, len(shape) - 1]
      for rng in [jtu.rand_default()]))
  def testSortKeyValAgainstNumpy(self, shape, key_dtype, val_dtype, axis, rng):
    if len(shape) > 1 and not FLAGS.jax_test_dut.startswith("tpu"):
      msg = "sort_key_val only implemented for R1 non-TPU backends"
      return absltest.unittest.skip(msg)

    # This test relies on the property that wherever keys are tied, values are
    # too, since we don't guarantee the same ordering of values with equal keys.
    # To avoid that case, we generate unique keys (globally in the key array).
    perm_rng = onp.random.RandomState(0)
    def args_maker():
      flat_keys = onp.arange(onp.prod(shape, dtype=int), dtype=key_dtype)
      keys = perm_rng.permutation(flat_keys).reshape(shape)
      values = rng(shape, val_dtype)
      return keys, values

    op = lambda ks, vs: lax.sort_key_val(ks, vs, axis)
    numpy_op = lambda ks, vs: lax_reference.sort_key_val(ks, vs, axis)
    self._CheckAgainstNumpy(op, numpy_op, args_maker)

  def testWhileWithTuple(self):
    limit = 10

    def loop_cond(state):
      pos, _ = state
      return lax.lt(pos, limit)

    def loop_body(state):
      pos, count = state
      return (lax.add(pos, 1), lax.add(count, 1))

    def loop(init):
      result = lax._while_loop(loop_cond, loop_body, (init, 0))
      _, count = result
      return count

    cloop = api.jit(loop)

    self.assertEqual(loop(2), limit - 2)
    self.assertEqual(cloop(2), limit - 2)
    self.assertEqual(cloop(2), limit - 2)
    self.assertEqual(cloop(3), limit - 3)

  def testNestedWhile(self):

    def outer_loop(num):  # pylint: disable=missing-docstring
      def cond_fun(state):
        num, i, _ = state
        return lax.lt(i, num)

      def body_fun(state):
        num, i, count = state
        return (num, lax.add(i, 1), inner_loop(i, count))

      init_val = (num, 0, 0)
      _, i, count = lax._while_loop(cond_fun, body_fun, init_val)
      return (i, count)

    def inner_loop(i, count):  # pylint: disable=missing-docstring
      def cond_fun(state):
        i, j, _ = state
        return lax.le(j, i)

      def body_fun(state):
        i, j, count = state
        return (i, lax.add(j, 1), lax.add(count, 1))

      init_val = (i, 0, count)
      _, _, count = lax._while_loop(cond_fun, body_fun, init_val)
      return count

    cloop = api.jit(outer_loop)

    self.assertEqual(outer_loop(3), (3, 6))
    self.assertEqual(cloop(3), (3, 6))
    self.assertEqual(cloop(3), (3, 6))
    self.assertEqual(cloop(2), (2, 3))
    self.assertEqual(cloop(4), (4, 10))

  def testNestedWhileWithDynamicUpdateSlice(self):
    num = 5

    def update_entry(arr, val, i, j):
      val = lax.reshape(val, [1, 1])
      return lax.dynamic_update_slice(arr, val, (i, j))

    def outer_loop(arr):  # pylint: disable=missing-docstring

      def cond_fun(state):
        i, num, _, _ = state
        return lax.lt(i, num)

      def body_fun(state):
        i, num, arr, out = state
        return (lax.add(i, 1), num, arr, inner_loop(i, arr, out))

      out = onp.zeros(arr.shape, dtype=arr.dtype)
      init_val = (0, num, arr, out)
      _, _, _, out = lax._while_loop(cond_fun, body_fun, init_val)
      return out

    def inner_loop(i, arr, out):  # pylint: disable=missing-docstring

      def cond_fun(state):
        i, j, _, _ = state
        return lax.le(j, i)

      def body_fun(state):
        i, j, arr, out = state
        arr_i = lax.dynamic_index_in_dim(arr, i, 0, False)
        arr_i_j = lax.dynamic_index_in_dim(arr_i, j, 0, False)
        out = update_entry(out, arr_i_j, i, j)
        return (i, lax.add(j, 1), arr, out)

      init_val = (i, 0, arr, out)
      _, _, _, out = lax._while_loop(cond_fun, body_fun, init_val)
      return out

    cloop = api.jit(outer_loop)
    arr = npr.RandomState(0).randn(5, 5)
    self.assertAllClose(outer_loop(arr), onp.tril(arr), check_dtypes=False)
    self.assertAllClose(cloop(arr), onp.tril(arr), check_dtypes=False)
    self.assertAllClose(cloop(arr), onp.tril(arr), check_dtypes=False)

  def testLoopWithConjunctionCondition(self):
    def sum_first_n(arr, num):  # pylint: disable=missing-docstring
      def cond_fun(state):
        arr, num, i, _ = state
        return lax.bitwise_and(lax.lt(i, num), lax.lt(i, arr.shape[0]))

      def body_fun(state):
        arr, num, i, total = state
        arr_i = lax.dynamic_index_in_dim(arr, i, 0, False)
        return (arr, num, lax.add(i, 1), lax.add(total, arr_i))

      init_val = (arr, num, 0, 0.)
      _, _, _, total = lax._while_loop(cond_fun, body_fun, init_val)
      return total

    cfun = api.jit(sum_first_n)
    x = npr.RandomState(0).randn(10)

    for num in [0, 5, 10, 15]:
      self.assertAllClose(sum_first_n(x, num), onp.sum(x[:num]),
                          check_dtypes=False)
      self.assertAllClose(cfun(x, num), onp.sum(x[:num]), check_dtypes=False)
      self.assertAllClose(cfun(x, num), onp.sum(x[:num]), check_dtypes=False)

  def testForiLoopBasic(self):
    def count(num):
      def body_fun(i, tot):
        return lax.add(tot, i)
      return lax.fori_loop(0, num, body_fun, 0)

    cfun = api.jit(count)

    self.assertEqual(count(2), 1)
    self.assertEqual(count(2), cfun(2))
    self.assertEqual(count(3), 3)
    self.assertEqual(count(3), cfun(3))
    self.assertEqual(count(4), 6)
    self.assertEqual(count(4), cfun(4))

  def testForiLoopTupleState(self):
    def sum_first_n(arr, num):
      def body_fun(i, state):
        arr, total = state
        arr_i = lax.dynamic_index_in_dim(arr, i, 0, False)
        return (arr, lax.add(total, arr_i))

      init_val = (arr, 0.)
      _, total = lax.fori_loop(0, lax.min(arr.shape[0], num), body_fun,
                               init_val)
      return total

    cfun = api.jit(sum_first_n)
    x = npr.RandomState(0).randn(10)

    for num in [0, 5, 10, 15]:
      self.assertAllClose(sum_first_n(x, num), onp.sum(x[:num]),
                          check_dtypes=False)
      self.assertAllClose(cfun(x, num), onp.sum(x[:num]), check_dtypes=False)
      self.assertAllClose(cfun(x, num), onp.sum(x[:num]), check_dtypes=False)

  def testForiLoopDictState(self):
    def sum_first_n(arr, num):
      def body_fun(i, state):
        arr, total = state['arr'], state['total']
        arr_i = lax.dynamic_index_in_dim(arr, i, 0, False)
        return {'arr': arr, 'total': lax.add(total, arr_i)}

      init_val = {'arr': arr, 'total': 0.}
      out_val = lax.fori_loop(0, lax.min(arr.shape[0], num), body_fun, init_val)
      return out_val['total']

    cfun = api.jit(sum_first_n)
    x = npr.RandomState(0).randn(10)

    for num in [0, 5, 10, 15]:
      self.assertAllClose(sum_first_n(x, num), onp.sum(x[:num]),
                          check_dtypes=False)
      self.assertAllClose(cfun(x, num), onp.sum(x[:num]), check_dtypes=False)
      self.assertAllClose(cfun(x, num), onp.sum(x[:num]), check_dtypes=False)

  def testForiLoopEmptyTupleInState(self):
    def sum_first_n(arr, num):
      def body_fun(i, state):
        arr, total, _ = state
        arr_i = lax.dynamic_index_in_dim(arr, i, 0, False)
        return (arr, lax.add(total, arr_i), ())

      init_val = (arr, 0., ())
      _, tot, _ = lax.fori_loop(0, lax.min(arr.shape[0], num), body_fun, init_val)
      return tot

    cfun = api.jit(sum_first_n)
    x = npr.RandomState(0).randn(10)

    for num in [0, 5, 10, 15]:
      self.assertAllClose(sum_first_n(x, num), onp.sum(x[:num]),
                          check_dtypes=False)
      self.assertAllClose(cfun(x, num), onp.sum(x[:num]), check_dtypes=False)
      self.assertAllClose(cfun(x, num), onp.sum(x[:num]), check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_lhs_shape={}_rhs_shape={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype)),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "rng": rng}
      for lhs_shape, rhs_shape in [((3, 2), (2, 4)),
                                   ((5, 3, 2), (5, 2, 4)),
                                   ((1, 2, 2, 3), (1, 2, 3, 1))]
      for dtype in float_dtypes
      for rng in [jtu.rand_small()]))
  def testBatchMatMul(self, lhs_shape, rhs_shape, dtype, rng):
    arg_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    self._CompileAndCheck(lax.batch_matmul, arg_maker, check_dtypes=True)

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
       "shape": shape, "dtype": dtype, "idxs": idxs, "axes": axes, "rng": rng}
      for dtype in all_dtypes
      for shape, idxs, axes in [
          [(3, 4, 5), (onp.array([0, 2, 1]),), (0,)],
          [(3, 4, 5), (onp.array([-1, -2]),), (0,)],
          [(3, 4, 5), (onp.array([0, 2]), onp.array([1, 3])), (0, 1)],
          [(3, 4, 5), (onp.array([0, 2]), onp.array([1, 3])), (0, 2)],
      ]
      for rng in [jtu.rand_default()]))
  def testIndexTake(self, shape, dtype, idxs, axes, rng):
    rand_idxs = lambda: tuple(rng(e.shape, e.dtype) for e in idxs)
    args_maker = lambda: [rng(shape, dtype), rand_idxs()]
    fun = lambda src, idxs: lax.index_take(src, idxs, axes)
    self._CompileAndCheck(fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dst_shape={}_idxs={}_axes={}".format(
          jtu.format_shape_dtype_string(dst_shape, dtype), idxs, axes),
       "dst_shape": dst_shape, "dtype": dtype, "idxs": idxs, "axes": axes,
       "rng": rng}
      for dtype in default_dtypes
      for dst_shape, idxs, axes in [
          [(3, 4, 5), (onp.array([0, 2, 1]),), (0,)],
          [(3, 4, 5), (onp.array([-1, -2]),), (0,)],
          [(3, 4, 5), (onp.array([0, 2]), onp.array([1, 3])), (0, 1)],
          [(3, 4, 5), (onp.array([0, 2]), onp.array([1, 3])), (0, 2)],
      ]
      for rng in [jtu.rand_default()]))
  def testIndexUntake(self, dst_shape, dtype, idxs, axes, rng):
    # We call lax.index_take to get the shapes right
    src_shape = lax.index_take(rng(dst_shape, dtype), idxs, axes).shape
    ridxs = lambda: tuple(rng(e.shape, e.dtype) for e in idxs)
    args_maker = lambda: [rng(src_shape, dtype), rng(dst_shape, dtype), ridxs()]
    fun = lambda src, dst, idxs: lax.index_untake(src, dst, idxs, axes)
    self._CompileAndCheck(fun, args_maker, check_dtypes=True)


GradTestSpec = collections.namedtuple(
    "GradTestSpec", ["op", "nargs", "order", "rng", "dtypes"])

LAX_GRAD_OPS = [
    GradTestSpec(lax.neg, nargs=1, order=2, rng=jtu.rand_default(),
                 dtypes=[onp.float64, onp.complex64]),
    GradTestSpec(lax.floor, nargs=1, order=2, rng=jtu.rand_default(),
                 dtypes=[onp.float64]),
    GradTestSpec(lax.ceil, nargs=1, order=2, rng=jtu.rand_default(),
                 dtypes=[onp.float64]),
    GradTestSpec(lax.round, nargs=1, order=2, rng=jtu.rand_default(),
                 dtypes=[onp.float64]),
    # GradTestSpec(lax.rem, nargs=2, order=2, rng=jtu.rand_default(),
    #              dtypes=[onp.float64]),  # TODO(mattjj): enable

    GradTestSpec(lax.exp, nargs=1, order=2, rng=jtu.rand_small(),
                 dtypes=[onp.float64, onp.complex64]),
    GradTestSpec(lax.expm1, nargs=1, order=2, rng=jtu.rand_default(),
                 dtypes=[onp.float64, onp.complex64]),
    GradTestSpec(lax.log, nargs=1, order=2, rng=jtu.rand_positive(),
                 dtypes=[onp.float64, onp.complex64]),
    GradTestSpec(lax.log1p, nargs=1, order=2, rng=jtu.rand_positive(),
                 dtypes=[onp.float64, onp.complex64]),
    GradTestSpec(lax.tanh, nargs=1, order=2, rng=jtu.rand_default(),
                 dtypes=[onp.float64, onp.complex64]),
    GradTestSpec(lax.sin, nargs=1, order=2, rng=jtu.rand_default(),
                 dtypes=[onp.float64]),
    GradTestSpec(lax.cos, nargs=1, order=2, rng=jtu.rand_default(),
                 dtypes=[onp.float64]),

    GradTestSpec(lax.erf, nargs=1, order=2, rng=jtu.rand_small(),
                 dtypes=[onp.float64]),
    GradTestSpec(lax.erfc, nargs=1, order=2, rng=jtu.rand_small(),
                 dtypes=[onp.float64]),
    GradTestSpec(lax.erf_inv, nargs=1, order=2, rng=jtu.rand_small(),
                 dtypes=[onp.float64]),
    # GradTestSpec(lax.lgamma, nargs=1, order=2, rng=jtu.rand_small(),
    #              dtypes=[onp.float64]),  # TODO(mattjj): enable

    GradTestSpec(lax.real, nargs=1, order=2, rng=jtu.rand_default(),
                 dtypes=[onp.complex64]),
    GradTestSpec(lax.imag, nargs=1, order=2, rng=jtu.rand_default(),
                 dtypes=[onp.complex64]),
    # GradTestSpec(lax.complex, nargs=2, order=2, rng=jtu.rand_default(),
    #              dtypes=[onp.float32]),  # TODO(mattjj): enable
    GradTestSpec(lax.conj, nargs=1, order=2, rng=jtu.rand_default(),
                 dtypes=[onp.float32, onp.complex64]),
    GradTestSpec(lax.abs, nargs=1, order=2, rng=jtu.rand_positive(),
                 dtypes=[onp.float64, onp.complex64]),
    GradTestSpec(lax.pow, nargs=2, order=2, rng=jtu.rand_positive(),
                 dtypes=[onp.float64, onp.complex64]),

    GradTestSpec(lax.add, nargs=2, order=2, rng=jtu.rand_default(),
                 dtypes=[onp.float64, onp.complex64]),
    GradTestSpec(lax.sub, nargs=2, order=2, rng=jtu.rand_default(),
                 dtypes=[onp.float64, onp.complex64]),
    GradTestSpec(lax.mul, nargs=2, order=2, rng=jtu.rand_default(),
                 dtypes=[onp.float64, onp.complex64]),
    GradTestSpec(lax.div, nargs=2, order=1, rng=jtu.rand_not_small(),
                 dtypes=[onp.float64, onp.complex64]),

    GradTestSpec(lax.max, nargs=2, order=2, rng=jtu.rand_some_equal(),
                 dtypes=[onp.float64]),
    GradTestSpec(lax.min, nargs=2, order=2, rng=jtu.rand_some_equal(),
                 dtypes=[onp.float64]),
]


def check_grads(f, args, order, atol=None, rtol=None, eps=None):
  # TODO(mattjj,dougalm): add higher-order check
  default_tol = 1e-6 if FLAGS.jax_enable_x64 else 1e-2
  atol = atol or default_tol
  rtol = rtol or default_tol
  eps = eps or default_tol
  jtu.check_jvp(f, partial(api.jvp, f), args, atol, rtol, eps)
  jtu.check_vjp(f, partial(api.vjp, f), args, atol, rtol, eps)


def check_grads_bilinear(f, args, order, atol=None, rtol=None):
  # Can use large eps to make up for numerical inaccuracies since the op is
  # bilinear (relying on the fact that we only check one arg at a time)
  lhs, rhs = args
  check_grads(lambda lhs: f(lhs, rhs), (lhs,), order, atol, rtol, eps=1.)
  check_grads(lambda rhs: f(lhs, rhs), (rhs,), order, atol, rtol, eps=1.)


class LaxAutodiffTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": jtu.format_test_name_suffix(
          rec.op.__name__, shapes, itertools.repeat(dtype)),
       "op": rec.op, "rng": rec.rng, "shapes": shapes, "dtype": dtype,
       "order": rec.order}
      for rec in LAX_GRAD_OPS
      for shape_group in compatible_shapes
      for shapes in CombosWithReplacement(shape_group, rec.nargs)
      for dtype in rec.dtypes
  ))
  def testOpGrad(self, op, rng, shapes, dtype, order):
    if FLAGS.jax_test_dut and FLAGS.jax_test_dut.startswith("tpu"):
      if dtype is onp.complex64:
        return absltest.unittest.skip("complex grads unimplemented on tpu")
      if op is lax.pow:
        return absltest.unittest.skip("pow grad imprecise on tpu")
    tol = 1e-1 if num_float_bits(dtype) == 32 else None
    args = tuple(rng(shape, dtype) for shape in shapes)
    check_grads(op, args, order, tol, tol, tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_from_dtype={}_to_dtype={}".format(
          from_dtype, to_dtype),
       "from_dtype": from_dtype, "to_dtype": to_dtype, "rng": rng}
      for from_dtype, to_dtype in itertools.product(
          [onp.float32, onp.float64], repeat=2)
      for rng in [jtu.rand_default()]))
  def testConvertElementTypeGrad(self, from_dtype, to_dtype, rng):
    args = (rng((2, 3), from_dtype),)
    convert_element_type = lambda x: lax.convert_element_type(x, to_dtype)
    check_grads(convert_element_type, args, 1, 1e-3, 1e-3, 1e-3)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_min_shape={}_operand_shape={}_max_shape={}".format(
          jtu.format_shape_dtype_string(min_shape, dtype),
          jtu.format_shape_dtype_string(operand_shape, dtype),
          jtu.format_shape_dtype_string(max_shape, dtype)),
       "min_shape": min_shape, "operand_shape": operand_shape,
       "max_shape": max_shape, "dtype": dtype, "rng": rng}
      for min_shape, operand_shape, max_shape in [
          [(), (), ()],
          [(), (2, 3), ()],
          [(2, 3), (2, 3), (2, 3)],
      ]
      for dtype in float_dtypes
      for rng in [jtu.rand_default()]))
  def testClampGrad(self, min_shape, operand_shape, max_shape, dtype, rng):
    tol = 1e-2 if onp.finfo(dtype).bits == 32 else None
    shapes = [min_shape, operand_shape, max_shape]
    min, operand, max = (rng(shape, dtype) for shape in shapes)
    min, max = onp.minimum(min, max), onp.maximum(min, max)  # broadcast
    check_grads(lax.clamp, (min, operand, max), 2, tol, tol, tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dim={}_baseshape=[{}]_dtype={}_narrs={}".format(
          dim, ",".join(str(d) for d in base_shape), onp.dtype(dtype).name,
          num_arrs),
       "dim": dim, "base_shape": base_shape, "dtype": dtype,
       "num_arrs": num_arrs, "rng": rng}
      for num_arrs in [3]
      for dtype in float_dtypes
      for base_shape in [(4,), (3, 4), (2, 3, 4)]
      for dim in range(len(base_shape))
      for rng in [jtu.rand_default()]))
  def testConcatenateGrad(self, dim, base_shape, dtype, num_arrs, rng):
    tol = 1e-2 if onp.finfo(dtype).bits == 32 else None
    shapes = [base_shape[:dim] + (size,) + base_shape[dim+1:]
              for size, _ in zip(itertools.cycle([3, 1, 4]), range(num_arrs))]
    operands = tuple(rng(shape, dtype) for shape in shapes)
    concatenate = lambda *args: lax.concatenate(args, dim)
    check_grads(concatenate, operands, 2, tol, tol, tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_strides={}_padding={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               strides, padding),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "strides": strides, "padding": padding, "rng": rng,}
       for lhs_shape, rhs_shape, all_strides in itertools.chain(
           [((b, i, 3, 4), (j, i, 1, 2), [(1, 1), (1, 2), (2, 1)])
            for b, i, j in itertools.product([2, 3], repeat=3)],
           [((4, 2, 1), (3, 2, 1), [(1,)])])
       for strides in all_strides
       for dtype in [onp.float32]
       for padding in ["VALID", "SAME"]
       for rng in [jtu.rand_small()]))
  @jtu.skip_on_devices("tpu")
  def testConvGrad(self, lhs_shape, rhs_shape, dtype, strides, padding, rng):
    lhs = rng(lhs_shape, dtype)
    rhs = rng(rhs_shape, dtype)
    conv = partial(lax.conv, window_strides=strides, padding=padding)
    check_grads_bilinear(conv, (lhs, rhs), order=2, atol=1e-2, rtol=1e-2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_strides={}_padding={}_lhs_dilation={}_"
       "rhs_dilation={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               strides, padding, lhs_dil, rhs_dil),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "strides": strides, "padding": padding, "lhs_dil": lhs_dil,
       "rhs_dil": rhs_dil, "rng": rng}
       for lhs_shape, rhs_shape, all_strides, all_pads, lhs_dils, rhs_dils in
       itertools.chain(
           [((b, i, 3, 4), (j, i, 1, 2), [(1, 1), (1, 2), (2, 1)],
             [((0, 0), (0, 0)), ((-1, 0), (0, -1)), ((1, 0), (0, 1))],
             [(1, 1), (2, 1)], [(1, 1)])
            for b, i, j in itertools.product([2, 3], repeat=3)],
           [((4, 2, 1), (3, 2, 1), [(1,)], [((1, 1),), ((0, 0),)],
             [(1,), (2,)], [(1,), (2,)])])
       for strides in all_strides
       for rhs_dil in rhs_dils
       for lhs_dil in lhs_dils
       for dtype in [onp.float32]
       for padding in all_pads
       for rng in [jtu.rand_small()]))
  @jtu.skip_on_devices("tpu")
  def testConvWithGeneralPaddingGrad(self, lhs_shape, rhs_shape, dtype, strides,
                                     padding, lhs_dil, rhs_dil, rng):
    lhs = rng(lhs_shape, dtype)
    rhs = rng(rhs_shape, dtype)
    conv = partial(lax.conv_with_general_padding, window_strides=strides,
                   padding=padding, lhs_dilation=lhs_dil, rhs_dilation=rhs_dil)
    check_grads_bilinear(conv, (lhs, rhs), order=2, atol=1e-2, rtol=1e-2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_strides={}_padding={}_lhs_dilation={}_"
       "rhs_dilation={}_dims={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               strides, padding, lhs_dil, rhs_dil, ",".join(dim_nums)),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "strides": strides, "padding": padding, "lhs_dil": lhs_dil,
       "rhs_dil": rhs_dil, "rng": rng, "dimension_numbers": dim_nums,
       "perms": perms}
      for lhs_shape, rhs_shape, all_strides, all_pads, lhs_dils, rhs_dils in [
          ((b, i, 3, 4), (j, i, 1, 2), [(1, 1), (1, 2), (2, 1)],
          [((0, 0), (0, 0)), ((-1, 0), (0, -1)), ((1, 0), (0, 1))],
          [(1, 1), (2, 1)], [(1, 1)])
          for b, i, j in itertools.product([1, 2], repeat=3)]
      for strides in all_strides
      for rhs_dil in rhs_dils
      for lhs_dil in lhs_dils
      for dtype in [onp.float32]
      for padding in all_pads
      for rng in [jtu.rand_default()]
      for dim_nums, perms in [
          (("NCHW", "OIHW", "NCHW"), ([0, 1, 2, 3], [0, 1, 2, 3])),
          (("NHWC", "HWIO", "NHWC"), ([0, 2, 3, 1], [2, 3, 1, 0]))]))
  @jtu.skip_on_devices("tpu")
  def testConvGeneralDilatedGrad(self, lhs_shape, rhs_shape, dtype, strides,
                                 padding, lhs_dil, rhs_dil, dimension_numbers,
                                 perms, rng):
    tol = 1e-1 if onp.finfo(dtype).bits == 32 else 1e-3
    lhs_perm, rhs_perm = perms  # permute to compatible shapes
    lhs = onp.transpose(rng(lhs_shape, dtype), lhs_perm)
    rhs = onp.transpose(rng(rhs_shape, dtype), rhs_perm)
    conv = partial(lax.conv_general_dilated, window_strides=strides,
                   padding=padding, lhs_dilation=lhs_dil, rhs_dilation=rhs_dil,
                   dimension_numbers=dimension_numbers)
    check_grads_bilinear(conv, (lhs, rhs), order=2, atol=tol, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_lhs_shape={}_rhs_shape={}".format(
          jtu.format_shape_dtype_string(lhs_shape, dtype),
          jtu.format_shape_dtype_string(rhs_shape, dtype)),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "rng": jtu.rand_default()}
      for lhs_shape in [(2,), (3, 2)] for rhs_shape in [(2,), (2, 4)]
      for dtype in float_dtypes))
  @jtu.skip_on_flag("jax_xla_backend", "xrt")
  def testDotGrad(self, lhs_shape, rhs_shape, dtype, rng):
    tol = 1e-1 if num_float_bits(dtype) == 32 else 1e-3
    lhs = rng(lhs_shape, dtype)
    rhs = rng(rhs_shape, dtype)
    check_grads_bilinear(lax.dot, (lhs, rhs), order=2, atol=tol, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_dimension_numbers={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               dimension_numbers),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "dimension_numbers": dimension_numbers, "rng": jtu.rand_small()}
      for lhs_shape, rhs_shape, dimension_numbers in [
          ((3, 2), (2, 4), (([1], [0]), ([], []))),
          ((3, 5), (2, 5), (([1], [1]), ([], []))),
          ((5, 3), (5, 2), (([0], [0]), ([], []))),
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0]))),
      ]
      for dtype in float_dtypes))
  @jtu.skip_on_devices("tpu")
  def testDotGeneralContractAndBatchGrads(self, lhs_shape, rhs_shape, dtype,
                                          dimension_numbers, rng):
    tol = 1e-1 if onp.finfo(dtype).bits == 32 else 1e-2
    lhs = rng(lhs_shape, dtype)
    rhs = rng(rhs_shape, dtype)
    dot_general = partial(lax.dot_general, dimension_numbers=dimension_numbers)
    check_grads_bilinear(dot_general, (lhs, rhs), order=2, atol=tol, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_dtype={}_broadcast_sizes={}".format(
          shape, onp.dtype(dtype).name, broadcast_sizes),
       "shape": shape, "dtype": dtype, "broadcast_sizes": broadcast_sizes,
       "rng": rng}
      for shape in [(), (2, 3)]
      for dtype in float_dtypes
      for broadcast_sizes in [(), (2,), (1, 2)]
      for rng in [jtu.rand_default()]))
  def testBroadcastGrad(self, shape, dtype, broadcast_sizes, rng):
    tol = 1e-2 if onp.finfo(dtype).bits == 32 else None
    args = (rng(shape, dtype),)
    broadcast = lambda x: lax.broadcast(x, broadcast_sizes)
    check_grads(broadcast, args, 2, tol, tol, tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}_bcdims={}".format(
          jtu.format_shape_dtype_string(inshape, dtype),
          outshape, broadcast_dimensions),
       "inshape": inshape, "dtype": dtype, "outshape": outshape,
       "dimensions": broadcast_dimensions, "rng": rng}
      for inshape, outshape, broadcast_dimensions in [
          ([2], [2, 2], [0]),
          ([2], [2, 2], [1]),
          ([2], [2, 3], [0]),
          ([], [2, 3], []),
      ]
      for dtype in float_dtypes
      for rng in [jtu.rand_default()]))
  def testBroadcastInDimGrad(self, inshape, dtype, outshape, dimensions, rng):
    tol = 1e-2 if onp.finfo(dtype).bits == 32 else None
    operand = rng(inshape, dtype)
    broadcast_in_dim = lambda x: lax.broadcast_in_dim(x, outshape, dimensions)
    check_grads(broadcast_in_dim, (operand,), 2, tol, tol, tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          jtu.format_shape_dtype_string(out_shape, dtype)),
       "arg_shape": arg_shape, "out_shape": out_shape, "dtype": dtype,
       "rng": rng}
      for dtype in float_dtypes
      for arg_shape, out_shape in [
          [(3, 4), (12,)], [(2, 1, 4), (8,)], [(2, 2, 4), (2, 8)]
      ]
      for rng in [jtu.rand_default()]))
  def testReshapeGrad(self, arg_shape, out_shape, dtype, rng):
    tol = 1e-2 if onp.finfo(dtype).bits == 32 else None
    operand = rng(arg_shape, dtype)
    reshape = lambda x: lax.reshape(x, out_shape)
    check_grads(reshape, (operand,), 2, tol, tol, tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_pads={}"
       .format(jtu.format_shape_dtype_string(shape, dtype), pads),
       "shape": shape, "dtype": dtype, "pads": pads, "rng": jtu.rand_small()}
      for shape in [(2, 3)]
      for dtype in float_dtypes
      for pads in [[(1, 2, 1), (0, 1, 0)]]))
  def testPadGrad(self, shape, dtype, pads, rng):
    tol = 1e-2 if onp.finfo(dtype).bits == 32 else None

    operand = rng(shape, dtype)
    pad = lambda operand: lax.pad(operand, onp.array(0, dtype), pads)
    check_grads(pad, (operand,), 2, tol, tol, tol)

    operand = rng(shape, dtype)
    padding_value = onp.array(0., dtype)
    pad = lambda operand, padding_value: lax.pad(operand, padding_value, pads)
    check_grads(pad, (operand, padding_value), 2, tol, tol, tol)

  def testReverseGrad(self):
    rev = lambda operand: lax.rev(operand, dimensions)

    dimensions = [0]
    check_grads(rev, (onp.array([3., 2., 1.]),), 2)

    dimensions = [0, 1]
    check_grads(rev, (onp.array([[6., 5., 4.], [3., 2., 1.]]),), 2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_predshape={}_argshapes={}".format(
          jtu.format_shape_dtype_string(pred_shape, onp.bool_),
          jtu.format_shape_dtype_string(arg_shape, dtype)),
       "pred_shape": pred_shape, "arg_shape": arg_shape, "dtype": dtype,
       "rng": rng}
      for arg_shape in [(), (3,), (2, 3)]
      for pred_shape in ([(), arg_shape] if arg_shape else [()])
      for dtype in float_dtypes
      for rng in [jtu.rand_default()]))
  def testSelectGrad(self, pred_shape, arg_shape, dtype, rng):
    tol = 1e-2 if onp.finfo(dtype).bits == 32 else None
    pred = rng(pred_shape, onp.bool_)
    on_true = rng(arg_shape, dtype)
    on_false = rng(arg_shape, dtype)
    select = lambda on_true, on_false: lax.select(pred, on_true, on_false)
    check_grads(select, (on_true, on_false), 2, tol, tol, tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}_start_indices={}_limit_indices={}_strides={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, limit_indices, strides),
       "shape": shape, "dtype": dtype, "starts": start_indices,
       "limits": limit_indices, "strides": strides, "rng": rng}
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
      for dtype in float_dtypes
      for rng in [jtu.rand_default()]))
  def testSliceGrad(self, shape, dtype, starts, limits, strides, rng):
    tol = 1e-2 if onp.finfo(dtype).bits == 32 else None
    operand = rng(shape, dtype)
    slice = lambda x: lax.slice(x, starts, limits, strides)
    check_grads(slice, (operand,), 2, tol, tol, tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_start_indices={}_size_indices={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, size_indices),
       "shape": shape, "dtype": dtype, "start_indices": start_indices,
       "size_indices": size_indices, "rng": rng}
      for shape, start_indices, size_indices in [
        [(3,), (1,), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
      ]
      for dtype in float_dtypes
      for rng in [jtu.rand_default()]))
  def testDynamicSliceGrad(self, shape, dtype, start_indices, size_indices,
                           rng):
    tol = 1e-2 if onp.finfo(dtype).bits == 32 else None
    operand = rng(shape, dtype)
    dynamic_slice = lambda x: lax.dynamic_slice(x, start_indices, size_indices)
    check_grads(dynamic_slice, (operand,), 2, tol, tol, tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_start_indices={}_update_shape={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, update_shape),
       "shape": shape, "dtype": dtype, "start_indices": start_indices,
       "update_shape": update_shape, "rng": rng}
      for shape, start_indices, update_shape in [
        [(3,), (1,), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
      ]
      for dtype in float_dtypes
      for rng in [jtu.rand_default()]))
  def testDynamicUpdateSliceGrad(self, shape, dtype, start_indices,
                                 update_shape, rng):
    tol = 1e-2 if onp.finfo(dtype).bits == 32 else None
    operand = rng(shape, dtype)
    update = rng(update_shape, dtype)
    start_indices = onp.array(start_indices)

    dus = lambda x, y: lax.dynamic_update_slice(x, y, start_indices)
    check_grads(dus, (operand, update), 2, tol, tol, tol)

    dus = lambda x: lax.dynamic_update_slice(x, update, start_indices)
    check_grads(dus, (operand,), 2, tol, tol, tol)

    dus = lambda y: lax.dynamic_update_slice(operand, y, start_indices)
    check_grads(dus, (update,), 2, tol, tol, tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_perm={}".format(
          jtu.format_shape_dtype_string(shape, dtype), perm),
       "shape": shape, "dtype": dtype, "perm": perm, "rng": rng}
      for shape, perm in [
        [(3, 4), (1, 0)],
        [(3, 4), (0, 1)],
        [(3, 4, 5), (2, 1, 0)],
        [(3, 4, 5), (1, 0, 2)],
      ]
      for dtype in float_dtypes
      for rng in [jtu.rand_default()]))
  def testTransposeGrad(self, shape, dtype, perm, rng):
    tol = 1e-2 if onp.finfo(dtype).bits == 32 else None
    operand = rng(shape, dtype)
    transpose = lambda x: lax.transpose(x, perm)
    check_grads(transpose, (operand,), 2, tol, tol, tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_inshape={}_reducedims={}"
       .format(op.__name__, jtu.format_shape_dtype_string(shape, dtype), dims),
       "op": op, "init_val": init_val, "shape": shape, "dtype": dtype,
       "dims": dims, "rng": rng}
      for init_val, op, dtypes in [
          (0, lax.add, float_dtypes),
          (-onp.inf, lax.max, float_dtypes),
          (onp.inf, lax.min, float_dtypes),
      ]
      for dtype in dtypes
      for shape, dims in [
          [(3, 4, 5), (0,)],
          [(3, 4, 5), (1, 2)],
          [(3, 4, 5), (0, 2)],
          [(3, 4, 5), (0, 1, 2)]
      ]
      for rng in [jtu.rand_small()]))
  def testReduceGrad(self, op, init_val, shape, dtype, dims, rng):
    tol = 1e-2 if onp.finfo(dtype).bits == 32 else None
    operand = rng(shape, dtype)
    init_val = onp.asarray(init_val, dtype=dtype)
    reduce = lambda operand: lax.reduce(operand, init_val, op, dims)
    check_grads(reduce, (operand,), 1, tol, tol, tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_dtype={}_padding={}"
       .format(op.__name__, onp.dtype(dtype).name, padding),
       "op": op, "init_val": init_val, "dtype": dtype, "padding": padding,
       "rng": rng}
      for init_val, op, dtypes, rng in [
          (0, lax.add, [onp.float32], jtu.rand_small()),
          (-onp.inf, lax.max, [onp.float32], jtu.rand_default()),
          (onp.inf, lax.min, [onp.float32], jtu.rand_default()),
      ]
      for dtype in dtypes
      for padding in ["VALID", "SAME"]
      for rng in [jtu.rand_default()]))
  def testReduceWindowGrad(self, op, init_val, dtype, padding, rng):
    init_val = onp.asarray(init_val, dtype=dtype)

    # We need this conditional and the corresponding loop logic to be in the
    # test method, rather than at the parameterized test level, because it
    # depends on FLAGS for the device under test.
    if FLAGS.jax_test_dut == "tpu":
      all_configs = [((6, 5, 4, 3), (2, 2, 1, 1), (1, 2, 1, 1))]
    else:
      all_configs = itertools.chain(
          itertools.product(
              [(4, 6)],  # shapes
              [(2, 1), (1, 2)],  # window_dimensions
              [(1, 1), (2, 1), (1, 2)]  # strides
          ),
          itertools.product(
              [(3, 2, 4, 6)],  # shapes
              [(1, 1, 2, 1), (2, 1, 2, 1)],  # window_dimensions
              [(1, 2, 2, 1), (1, 1, 1, 1)]),  # strides
      )

    def fun(operand):
      return lax.reduce_window(operand, init_val, op, dims, strides, padding)

    # pylint: disable=cell-var-from-loop
    for shape, dims, strides in all_configs:
      operand = rng(shape, dtype)
      if op is lax.add:
        check_grads(fun, (operand,), 1, 1e-2, 1e-2, 1e-2)
      else:
        # this test can fail if there are duplicates in operand
        self.assertEqual(onp.unique(operand).size, operand.size,
                         msg="test requires operand elements to be unique.")
        jtu.check_vjp(fun, partial(api.vjp, fun), (operand,),
                            1e-2, 1e-2, 1e-2)
    # pylint: enable=cell-var-from-loop

  # TODO(b/205052657): enable more tests when supported
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "rng": rng, "shape": shape, "dtype": dtype, "axis": axis}
      for dtype in [onp.float32]
      for shape in [(5,), (5, 7)]
      for axis in [len(shape) - 1]
      for rng in [jtu.rand_default()]))
  def testSortGrad(self, shape, dtype, axis, rng):
    tol = 1e-2 if onp.finfo(dtype).bits == 32 else None
    operand = rng(shape, dtype)
    sort = lambda x: lax.sort(x, axis)
    check_grads(sort, (operand,), 2, tol, tol, tol)

  # TODO(b/205052657): enable more tests when supported
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_keyshape={}_valshape={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, key_dtype),
          jtu.format_shape_dtype_string(shape, val_dtype),
          axis),
       "rng": rng, "shape": shape,
       "key_dtype": key_dtype, "val_dtype": val_dtype, "axis": axis}
      for key_dtype in [onp.float32]
      for val_dtype in [onp.float32]
      for shape in [(3,), (5, 3)]
      for axis in [len(shape) - 1]
      for rng in [jtu.rand_default()]))
  def testSortKeyValGrad(self, shape, key_dtype, val_dtype, axis, rng):
    # This test relies on the property that wherever keys are tied, values are
    # too, since we don't guarantee the same ordering of values with equal keys.
    # To avoid that case, we generate unique keys (globally in the key array).
    perm_rng = onp.random.RandomState(0)
    def args_maker():
      flat_keys = onp.arange(onp.prod(shape, dtype=int), dtype=key_dtype)
      keys = perm_rng.permutation(flat_keys).reshape(shape)
      values = rng(shape, val_dtype)
      return keys, values
    keys, values = args_maker()

    fun = lambda keys, values: lax.sort_key_val(keys, values, axis)
    check_grads(fun, (keys, values), 2, 1e-2, 1e-2, 1e-2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_idxs={}_axes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), idxs, axes),
       "shape": shape, "dtype": dtype, "idxs": idxs, "axes": axes, "rng": rng}
      for dtype in float_dtypes
      for shape, idxs, axes in [
          [(3, 4, 5), (onp.array([0, 2, 1]),), (0,)],
          [(3, 4, 5), (onp.array([-1, -2]),), (0,)],
          [(3, 4, 5), (onp.array([0, 2]), onp.array([1, 3])), (0, 1)],
          [(3, 4, 5), (onp.array([0, 2]), onp.array([1, 3])), (0, 2)],
      ]
      for rng in [jtu.rand_default()]))
  def testIndexTakeGrad(self, shape, dtype, idxs, axes, rng):
    idxs = tuple(rng(e.shape, e.dtype) for e in idxs)
    src = rng(shape, dtype)
    index_take = lambda src: lax.index_take(src, idxs, axes)
    check_grads(index_take, (src,), 2, 1e-2, 1e-2, 1e-2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dst_shape={}_idxs={}_axes={}".format(
          jtu.format_shape_dtype_string(dst_shape, dtype), idxs, axes),
       "dst_shape": dst_shape, "dtype": dtype, "idxs": idxs, "axes": axes,
       "rng": rng}
      for dtype in float_dtypes
      for dst_shape, idxs, axes in [
          [(3, 4, 5), (onp.array([0, 2, 1]),), (0,)],
          [(3, 4, 5), (onp.array([-1, -2]),), (0,)],
          [(3, 4, 5), (onp.array([0, 2]), onp.array([1, 3])), (0, 1)],
          [(3, 4, 5), (onp.array([0, 2]), onp.array([1, 3])), (0, 2)],
      ]
      for rng in [jtu.rand_default()]))
  def testIndexUntakeGrad(self, dst_shape, dtype, idxs, axes, rng):
    # We call lax.index_take to get the shapes right
    src_shape = lax.index_take(rng(dst_shape, dtype), idxs, axes).shape

    idxs = tuple(rng(e.shape, e.dtype) for e in idxs)
    src = rng(src_shape, dtype)
    dst = rng(dst_shape, dtype)
    index_untake = lambda src, dst: lax.index_untake(src, dst, idxs, axes)
    check_grads(index_untake, (src, dst), 2, 1e-2, 1e-2, 1e-2)


if __name__ == '__main__':
  absltest.main()
