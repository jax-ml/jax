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
import itertools
from unittest import skip

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp

from jax import api
from jax import lax
from jax import numpy as lnp
from jax import test_util as jtu

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS

nonempty_nonscalar_array_shapes = [(4,), (3, 4), (3, 1), (1, 4), (2, 1, 4), (2, 3, 4)]
nonempty_array_shapes = [()] + nonempty_nonscalar_array_shapes
empty_array_shapes = [(0,), (0, 4), (3, 0),]

scalar_shapes = [jtu.NUMPY_SCALAR_SHAPE]
array_shapes = nonempty_array_shapes + empty_array_shapes
nonempty_shapes = scalar_shapes + nonempty_array_shapes
all_shapes =  scalar_shapes + array_shapes

float_dtypes = [onp.float32, onp.float64]
complex_dtypes = [onp.complex64, onp.complex128]
int_dtypes = [onp.int32, onp.int64]
unsigned_dtypes = [onp.uint32, onp.uint64]
bool_dtypes = [onp.bool_]
default_dtypes = float_dtypes + int_dtypes
inexact_dtypes = float_dtypes + complex_dtypes
number_dtypes = float_dtypes + complex_dtypes + int_dtypes
all_dtypes = number_dtypes + bool_dtypes

OpRecord = collections.namedtuple(
  "OpRecord",
  ["name", "nargs", "dtypes", "shapes", "rng", "diff_modes", "test_name"])


def op_record(name, nargs, dtypes, shapes, rng, diff_modes, test_name=None):
  test_name = test_name or name
  return OpRecord(name, nargs, dtypes, shapes, rng, diff_modes, test_name)

JAX_ONE_TO_ONE_OP_RECORDS = [
    op_record("abs", 1, number_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("add", 2, number_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("ceil", 1, float_dtypes, all_shapes, jtu.rand_default(), []),
    op_record("conj", 1, number_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("equal", 2, all_dtypes, all_shapes, jtu.rand_some_equal(), []),
    op_record("exp", 1, number_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("fabs", 1, float_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("floor", 1, float_dtypes, all_shapes, jtu.rand_default(), []),
    op_record("greater", 2, number_dtypes, all_shapes, jtu.rand_some_equal(), []),
    op_record("greater_equal", 2, number_dtypes, all_shapes, jtu.rand_some_equal(), []),
    op_record("isfinite", 1, number_dtypes, all_shapes, jtu.rand_some_inf(), []),
    op_record("less", 2, number_dtypes, all_shapes, jtu.rand_some_equal(), []),
    op_record("less_equal", 2, number_dtypes, all_shapes, jtu.rand_some_equal(), []),
    op_record("log", 1, number_dtypes, all_shapes, jtu.rand_positive(), ["rev"]),
    op_record("logical_and", 2, all_dtypes, all_shapes, jtu.rand_bool(), []),
    op_record("logical_not", 1, all_dtypes, all_shapes, jtu.rand_bool(), []),
    op_record("logical_or", 2, all_dtypes, all_shapes, jtu.rand_bool(), []),
    op_record("logical_xor", 2, all_dtypes, all_shapes, jtu.rand_bool(), []),
    op_record("maximum", 2, number_dtypes, all_shapes, jtu.rand_some_inf(), []),
    op_record("minimum", 2, number_dtypes, all_shapes, jtu.rand_some_inf(), []),
    op_record("multiply", 2, number_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("negative", 1, number_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("not_equal", 2, number_dtypes, all_shapes, jtu.rand_some_equal(), ["rev"]),
    op_record("power", 2, inexact_dtypes, all_shapes, jtu.rand_positive(), ["rev"]),
    op_record("reciprocal", 1, inexact_dtypes, all_shapes, jtu.rand_default(), []),
    op_record("subtract", 2, number_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("sin", 1, number_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("cos", 1, number_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("tan", 1, number_dtypes, all_shapes, jtu.rand_uniform(-1.5, 1.5),
              ["rev"]),
    op_record("sinh", 1, number_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("cosh", 1, number_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("tanh", 1, number_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("arcsin", 1, float_dtypes, all_shapes, jtu.rand_small(), ["rev"]),
    op_record("arccos", 1, float_dtypes, all_shapes, jtu.rand_small(), ["rev"]),
    op_record("arctan", 1, float_dtypes, all_shapes, jtu.rand_small(), ["rev"]),
    op_record("arctan2", 2, float_dtypes, all_shapes, jtu.rand_small(), ["rev"]),
    op_record("arcsinh", 1, number_dtypes, all_shapes, jtu.rand_positive(), ["rev"]),
    op_record("arccosh", 1, number_dtypes, all_shapes, jtu.rand_positive(), ["rev"]),
    op_record("arctanh", 1, number_dtypes, all_shapes, jtu.rand_small(), ["rev"]),
]

JAX_COMPOUND_OP_RECORDS = [
    op_record("angle", 1, number_dtypes, all_shapes, jtu.rand_default(), []),
    op_record("conjugate", 1, number_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("deg2rad", 1, float_dtypes, all_shapes, jtu.rand_default(), []),
    op_record("divide", 2, number_dtypes, all_shapes, jtu.rand_nonzero(), ["rev"]),
    op_record("exp2", 1, number_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("expm1", 1, number_dtypes, all_shapes, jtu.rand_positive(), [],
              test_name="expm1_large"),
    op_record("expm1", 1, number_dtypes, all_shapes, jtu.rand_small_positive(), []),
    op_record("floor_divide", 2, number_dtypes, all_shapes, jtu.rand_nonzero(), ["rev"]),
    op_record("hypot", 2, default_dtypes, all_shapes, jtu.rand_default(), []),
    op_record("kron", 2, number_dtypes, nonempty_shapes, jtu.rand_default(), []),
    op_record("outer", 2, number_dtypes, all_shapes, jtu.rand_default(), []),
    op_record("imag", 1, number_dtypes, all_shapes, jtu.rand_some_inf(), []),
    op_record("isclose", 2, all_dtypes, all_shapes, jtu.rand_small_positive(), []),
    op_record("iscomplex", 1, number_dtypes, all_shapes, jtu.rand_some_inf(), []),
    op_record("isreal", 1, number_dtypes, all_shapes, jtu.rand_some_inf(), []),
    op_record("log2", 1, number_dtypes, all_shapes, jtu.rand_positive(), ["rev"]),
    op_record("log10", 1, number_dtypes, all_shapes, jtu.rand_positive(), ["rev"]),
    op_record("log1p", 1, number_dtypes, all_shapes, jtu.rand_positive(), [],
              test_name="log1p_large"),
    op_record("log1p", 1, number_dtypes, all_shapes, jtu.rand_small_positive(), []),
    op_record("logaddexp", 2, float_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("logaddexp2", 2, float_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("polyval", 2, number_dtypes, nonempty_nonscalar_array_shapes, jtu.rand_default(), []),
    op_record("positive", 1, number_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("rad2deg", 1, float_dtypes, all_shapes, jtu.rand_default(), []),
    op_record("ravel", 1, all_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("real", 1, number_dtypes, all_shapes, jtu.rand_some_inf(), []),
    op_record("remainder", 2, default_dtypes, all_shapes, jtu.rand_nonzero(), []),
    op_record("sinc", 1, number_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("square", 1, number_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("sqrt", 1, number_dtypes, all_shapes, jtu.rand_positive(), ["rev"]),
    op_record("transpose", 1, all_dtypes, all_shapes, jtu.rand_default(), ["rev"]),
    op_record("true_divide", 2, all_dtypes, all_shapes, jtu.rand_nonzero(), ["rev"]),
    op_record("where", 3, (onp.float32, onp.int64), all_shapes, jtu.rand_some_zero(), []),
]

JAX_BITWISE_OP_RECORDS = [
    op_record("bitwise_and", 2, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_bool(), []),
    op_record("bitwise_not", 1, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_bool(), []),
    op_record("bitwise_or", 2, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_bool(), []),
    op_record("bitwise_xor", 2, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_bool(), []),
]

JAX_REDUCER_RECORDS = [
    op_record("mean", 1, number_dtypes, nonempty_shapes, jtu.rand_default(), []),
    op_record("prod", 1, number_dtypes, all_shapes, jtu.rand_small_positive(), []),
    op_record("sum", 1, number_dtypes, all_shapes, jtu.rand_default(), []),
    op_record("var", 1, number_dtypes, nonempty_shapes, jtu.rand_default(), []),
    op_record("std", 1, inexact_dtypes, nonempty_shapes, jtu.rand_default(), []),
]

JAX_REDUCER_NO_DTYPE_RECORDS = [
    op_record("all", 1, all_dtypes, all_shapes, jtu.rand_some_zero(), []),
    op_record("any", 1, all_dtypes, all_shapes, jtu.rand_some_zero(), []),
    op_record("max", 1, all_dtypes, nonempty_shapes, jtu.rand_default(), []),
    op_record("min", 1, all_dtypes, nonempty_shapes, jtu.rand_default(), []),
]

numpy_version = tuple(map(int, onp.version.version.split('.')))
if numpy_version >= (1, 15):
  JAX_REDUCER_NO_DTYPE_RECORDS += [
      op_record("ptp", 1, number_dtypes, nonempty_shapes, jtu.rand_default(), []),
  ]


JAX_ARGMINMAX_RECORDS = [
    op_record("argmin", 1, all_dtypes, nonempty_shapes, jtu.rand_some_equal(), []),
    op_record("argmax", 1, all_dtypes, nonempty_shapes, jtu.rand_some_equal(), []),
]

CombosWithReplacement = itertools.combinations_with_replacement


def _dtypes_are_compatible_for_bitwise_ops(args):
  if len(args) <= 1:
    return True
  is_signed = lambda dtype: onp.issubdtype(dtype, onp.signedinteger)
  width = lambda dtype: onp.iinfo(dtype).bits
  x, y = args
  if width(x) > width(y):
    x, y = y, x
  # The following condition seems a little ad hoc, but seems to capture what
  # numpy actually implements.
  return (
      is_signed(x) == is_signed(y)
      or (width(x) == 32 and width(y) == 32)
      or (width(x) == 32 and width(y) == 64 and is_signed(y)))

def _shapes_are_broadcast_compatible(shapes):
  accumulator = onp.zeros([])
  for shape in shapes:
    try:
      accumulator = accumulator + onp.zeros(shape)
    except ValueError:
      return False
  return True


class LaxBackedNumpyTests(jtu.JaxTestCase):
  """Tests for LAX-backed Numpy implementation."""

  def _GetArgsMaker(self, rng, shapes, dtypes):
    return lambda: [rng(shape, dtype) for shape, dtype in zip(shapes, dtypes)]

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(rec.test_name, shapes,
                                                      dtypes),
         "rng": rec.rng, "shapes": shapes, "dtypes": dtypes,
         "onp_op": getattr(onp, rec.name), "lnp_op": getattr(lnp, rec.name)}
        for shapes in filter(
          _shapes_are_broadcast_compatible,
          CombosWithReplacement(rec.shapes, rec.nargs))
        for dtypes in CombosWithReplacement(rec.dtypes, rec.nargs))
      for rec in itertools.chain(JAX_ONE_TO_ONE_OP_RECORDS, JAX_COMPOUND_OP_RECORDS)))
  def testOp(self, onp_op, lnp_op, rng, shapes, dtypes):
    args_maker = self._GetArgsMaker(rng, shapes, dtypes)
    self._CheckAgainstNumpy(onp_op, lnp_op, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_op, args_maker, check_dtypes=True)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(
            rec.test_name, shapes, dtypes),
         "rng": rec.rng, "shapes": shapes, "dtypes": dtypes,
         "onp_op": getattr(onp, rec.name), "lnp_op": getattr(lnp, rec.name)}
        for shapes in filter(
          _shapes_are_broadcast_compatible,
          CombosWithReplacement(rec.shapes, rec.nargs))
        for dtypes in filter(
          _dtypes_are_compatible_for_bitwise_ops,
          CombosWithReplacement(rec.dtypes, rec.nargs)))
      for rec in JAX_BITWISE_OP_RECORDS))
  def testBitwiseOp(self, onp_op, lnp_op, rng, shapes, dtypes):
    if not FLAGS.jax_enable_x64 and any(
        onp.iinfo(dtype).bits == 64 for dtype in dtypes):
      self.skipTest("x64 types are disabled by jax_enable_x64")
    args_maker = self._GetArgsMaker(rng, shapes, dtypes)
    self._CheckAgainstNumpy(onp_op, lnp_op, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_op, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "{}_inshape={}_axis={}_dtype={}_keepdims={}".format(
          rec.test_name.capitalize(),
          jtu.format_shape_dtype_string(shape, dtype), axis,
          "None" if out_dtype is None else onp.dtype(out_dtype).name, keepdims),
       "rng": rec.rng, "shape": shape, "dtype": dtype, "out_dtype": out_dtype,
       "onp_op": getattr(onp, rec.name), "lnp_op": getattr(lnp, rec.name),
       "axis": axis, "keepdims": keepdims}
      for rec in JAX_REDUCER_RECORDS
      for shape in rec.shapes for dtype in rec.dtypes
      for out_dtype in [None] + rec.dtypes
      for axis in set(range(-len(shape), len(shape))) | set([None])
      for keepdims in [False, True]))
  def testReducer(self, onp_op, lnp_op, rng, shape, dtype, out_dtype, axis, keepdims):
    onp_fun = lambda x: onp_op(x, axis, dtype=out_dtype, keepdims=keepdims)
    lnp_fun = lambda x: lnp_op(x, axis, dtype=out_dtype, keepdims=keepdims)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "{}_inshape={}_axis={}_keepdims={}".format(
          rec.test_name.capitalize(),
          jtu.format_shape_dtype_string(shape, dtype), axis, keepdims),
       "rng": rec.rng, "shape": shape, "dtype": dtype,
       "onp_op": getattr(onp, rec.name), "lnp_op": getattr(lnp, rec.name),
       "axis": axis, "keepdims": keepdims}
      for rec in JAX_REDUCER_NO_DTYPE_RECORDS
      for shape in rec.shapes for dtype in rec.dtypes
      for axis in set(range(-len(shape), len(shape))) | set([None])
      for keepdims in [False, True]))
  def testReducerNoDtype(self, onp_op, lnp_op, rng, shape, dtype, axis, keepdims):
    onp_fun = lambda x: onp_op(x, axis, keepdims=keepdims)
    lnp_fun = lambda x: lnp_op(x, axis, keepdims=keepdims)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "shape": shape, "dtype": dtype, "axis": axis}
      for shape in all_shapes for dtype in all_dtypes
      for axis in set(range(-len(shape), len(shape))) | set([None])))
  def testCountNonzero(self, shape, dtype, axis):
    rng = jtu.rand_some_zero()
    onp_fun = lambda x: onp.count_nonzero(x, axis)
    lnp_fun = lambda x: lnp.count_nonzero(x, axis)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "{}_inshape={}_axis={}".format(
          rec.test_name.capitalize(),
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "rng": rec.rng, "shape": shape, "dtype": dtype,
       "onp_op": getattr(onp, rec.name), "lnp_op": getattr(lnp, rec.name),
       "axis": axis}
      for rec in JAX_ARGMINMAX_RECORDS
      for shape in rec.shapes for dtype in rec.dtypes
      for axis in range(-len(shape), len(shape))))
  def testArgMinMax(self, onp_op, lnp_op, rng, shape, dtype, axis):

    def onp_fun(array_to_reduce):
      return onp_op(array_to_reduce, axis)

    def lnp_fun(array_to_reduce):
      return lnp_op(array_to_reduce, axis)

    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_{}".format(
          name,
          jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
          jtu.format_shape_dtype_string(rhs_shape, rhs_dtype)),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
       "rng": rng}
      for rng in [jtu.rand_default()]
      for name, lhs_shape, rhs_shape in [
          ("matrix-scalar", (3, 3), ()),
          ("scalar-matrix", (), (3, 3)),
          ("matrix-vector", (4, 5), (5,)),
          ("vector-matrix", (6,), (6, 4)),
          ("matrix-matrix", (3, 4), (4, 5)),
          ("tensor-vector", (4, 3, 2), (2,)),
          ("vector-tensor", (2,), (3, 2, 4)),
          ("tensor-matrix", (4, 3, 2), (2, 5)),
          ("matrix-tensor", (5, 2), (3, 2, 4)),
          ("tensor-tensor", (2, 3, 4), (5, 4, 1))]
      for lhs_dtype, rhs_dtype in CombosWithReplacement(inexact_dtypes, 2)))
  def testDot(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, rng):
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    self._CheckAgainstNumpy(onp.dot, lnp.dot, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp.dot, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_{}".format(
          name,
          jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
          jtu.format_shape_dtype_string(rhs_shape, rhs_dtype)),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
       "rng": rng}
      for rng in [jtu.rand_default()]
      for name, lhs_shape, rhs_shape in [
          ("vector-vector", (3,), (3,)),
          ("matrix-vector", (3, 3), (3,)),
          ("vector-matrix", (3,), (3, 3)),
          ("matrix-matrix", (3, 3), (3, 3)),
          ("vector-tensor", (3,), (5, 3, 2)),
          ("tensor-vector", (5, 3, 2), (2,)),
          ("matrix-tensor", (5, 2), (3, 2, 4)),
          ("tensor-matrix", (5, 2, 3), (3, 2)),
          ("tensor-tensor", (5, 3, 4), (5, 4, 1)),
          ("tensor-tensor-broadcast", (3, 1, 3, 4), (5, 4, 1))]
      for lhs_dtype, rhs_dtype in CombosWithReplacement(inexact_dtypes, 2)))
  def testMatmul(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, rng):
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    self._CheckAgainstNumpy(onp.matmul, lnp.matmul, args_maker,
                            check_dtypes=True)
    self._CompileAndCheck(lnp.matmul, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_{}".format(
          jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
          jtu.format_shape_dtype_string(rhs_shape, rhs_dtype),
          axes),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
       "axes": axes, "rng": rng}
      for rng in [jtu.rand_default()]
      for lhs_shape, rhs_shape, axes in [
          [(2, 3, 4), (3, 4, 5, 6), 2],
          [(2, 3, 4), (5, 4, 3, 6), [1, 2]],
          [(2, 3, 4), (5, 4, 3, 6), [[1, 2], [2, 1]]],
          [(1, 2, 3, 4), (4, 5, 3, 6), [[2, 3], [2, 0]]],
      ]
      for lhs_dtype, rhs_dtype in CombosWithReplacement(inexact_dtypes, 2)))
  def testTensordot(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, axes, rng):
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    lnp_fun = lambda a, b: lnp.tensordot(a, b, axes)
    onp_fun = lambda a, b: onp.tensordot(a, b, axes)
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}".format(
          jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
          jtu.format_shape_dtype_string(rhs_shape, rhs_dtype)),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
       "rng": jtu.rand_default()}
      # TODO(phawkins): support integer dtypes too.
      for lhs_dtype, rhs_dtype in CombosWithReplacement(inexact_dtypes, 2)
      for lhs_shape, rhs_shape in [
        (l, r) for l, r in CombosWithReplacement(all_shapes, 2)
        if len(jtu._dims_of_shape(l)) == 0
        or len(jtu._dims_of_shape(r)) == 0
        or l[-1] == r[-1]]))
  def testInner(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, rng):
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    onp_fun = lambda lhs, rhs: onp.inner(lhs, rhs)
    lnp_fun = lambda lhs, rhs: lnp.inner(lhs, rhs)
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_amin={}_amax={}".format(
          jtu.format_shape_dtype_string(shape, dtype), a_min, a_max),
       "shape": shape, "dtype": dtype, "a_min": a_min, "a_max": a_max,
       "rng": jtu.rand_default()}
      for shape in all_shapes for dtype in number_dtypes
      for a_min, a_max in [(-1, None), (None, 1), (-1, 1)]))
  def testClipStaticBounds(self, shape, dtype, a_min, a_max, rng):
    onp_fun = lambda x: onp.clip(x, a_min=a_min, a_max=a_max)
    lnp_fun = lambda x: lnp.clip(x, a_min=a_min, a_max=a_max)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_decimals={}".format(
          jtu.format_shape_dtype_string(shape, dtype), decimals),
       "shape": shape, "dtype": dtype, "decimals": decimals,
       "rng": jtu.rand_default()}
      for shape in all_shapes for dtype in number_dtypes
      for decimals in [0, 1, -2]))
  def testRoundStaticDecimals(self, shape, dtype, decimals, rng):
    if onp.issubdtype(dtype, onp.integer) and decimals < 0:
      self.skipTest("Integer rounding with decimals < 0 not implemented")
    onp_fun = lambda x: onp.round(x, decimals=decimals)
    lnp_fun = lambda x: lnp.round(x, decimals=decimals)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_rpadwidth={}_rconstantvalues={}".format(
          jtu.format_shape_dtype_string(shape, dtype), pad_width_rank,
          constant_values_rank),
       "shape": shape, "dtype": dtype, "pad_width_rank": pad_width_rank,
       "constant_values_rank": constant_values_rank, "rng": jtu.rand_default(),
       "irng": jtu.rand_int(3)}
      for shape in all_shapes for dtype in all_dtypes
      for pad_width_rank in range(3)
      for constant_values_rank in range(3)))
  def testPad(self, shape, dtype, pad_width_rank, constant_values_rank, rng,
              irng):
    pad_width = irng([len(shape), 2][2 - pad_width_rank:], onp.int32)
    def onp_fun(x, constant_vals):
      if pad_width.size == 0:
        return x
      return onp.pad(x, pad_width, mode='constant',
                     constant_values=constant_vals)
    def lnp_fun(x, constant_vals):
      return lnp.pad(x, pad_width, mode='constant',
                     constant_values=constant_vals)

    def args_maker():
      constant_vals = rng([len(shape), 2][2 - constant_values_rank:], dtype)
      return rng(shape, dtype), constant_vals

    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_axis={}_baseshape=[{}]_dtypes=[{}]".format(
          axis, ",".join(str(d) for d in base_shape),
          ",".join(onp.dtype(dtype).name for dtype in dtypes)),
       "axis": axis, "base_shape": base_shape, "dtypes": dtypes,
       "rng": jtu.rand_default()}
      for num_arrs in [3]
      for dtypes in CombosWithReplacement(default_dtypes, num_arrs)
      for base_shape in [(4,), (3, 4), (2, 3, 4)]
      for axis in range(-len(base_shape)+1, len(base_shape))))
  def testConcatenate(self, axis, base_shape, dtypes, rng):
    wrapped_axis = axis % len(base_shape)
    shapes = [base_shape[:wrapped_axis] + (size,) + base_shape[wrapped_axis+1:]
              for size, _ in zip(itertools.cycle([3, 1, 4]), dtypes)]
    onp_fun = lambda *args: onp.concatenate(args, axis=axis)
    lnp_fun = lambda *args: lnp.concatenate(args, axis=axis)

    def args_maker():
      return [rng(shape, dtype) for shape, dtype in zip(shapes, dtypes)]

    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_axis={}_baseshape=[{}]_dtypes=[{}]".format(
          axis, ",".join(str(d) for d in base_shape),
          ",".join(onp.dtype(dtype).name for dtype in dtypes)),
       "axis": axis, "base_shape": base_shape, "dtypes": dtypes,
       "rng": jtu.rand_default()}
      for dtypes in CombosWithReplacement(default_dtypes, 2)
      for base_shape in [(4,), (3, 4), (2, 3, 4)]
      for axis in range(-len(base_shape)+1, len(base_shape))))
  def testAppend(self, axis, base_shape, dtypes, rng):
    wrapped_axis = axis % len(base_shape)
    shapes = [base_shape[:wrapped_axis] + (size,) + base_shape[wrapped_axis+1:]
              for size, _ in zip(itertools.cycle([3, 1, 4]), dtypes)]
    onp_fun = lambda arr, values: onp.append(arr, values, axis=axis)
    lnp_fun = lambda arr, values: lnp.append(arr, values, axis=axis)

    def args_maker():
      return [rng(shape, dtype) for shape, dtype in zip(shapes, dtypes)]

    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape=[{}]_axis={}_repeats={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, repeats),
       "axis": axis, "shape": shape, "dtype": dtype, "repeats": repeats,
       "rng": jtu.rand_default()}
      for repeats in [0, 1, 2]
      for dtype in default_dtypes
      for shape in all_shapes
      for axis in [None] + list(range(-len(shape), len(shape)))))
  def testRepeat(self, axis, shape, dtype, repeats, rng):
    onp_fun = lambda arg: onp.repeat(arg, repeats=repeats, axis=axis)
    lnp_fun = lambda arg: lnp.repeat(arg, repeats=repeats, axis=axis)

    args_maker = lambda: [rng(shape, dtype)]

    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "op={}_shape=[{}]_axis={}_out_dtype={}".format(
          op, jtu.format_shape_dtype_string(shape, dtype), axis, out_dtype),
       "axis": axis, "shape": shape, "dtype": dtype, "out_dtype": out_dtype,
       "rng": jtu.rand_default(), "lnp_op": getattr(lnp, op),
       "onp_op": getattr(onp, op)}
      for op in ["cumsum", "cumprod"]
      # TODO(phawkins): replace both type lists with default_dtypes after a
      # Jaxlib update includes
      # https://github.com/google/jax/commit/86f5d189cf563b027c3cd00eea38072c003905c8
      for dtype in [onp.float32, onp.int32]
      for out_dtype in [onp.float32, onp.int32]
      for shape in all_shapes
      for axis in [None] + list(range(-len(shape), len(shape)))))
  def testCumSumProd(self, axis, shape, dtype, out_dtype, onp_op, lnp_op, rng):
    onp_fun = lambda arg: onp_op(arg, axis=axis, dtype=out_dtype)
    lnp_fun = lambda arg: lnp_op(arg, axis=axis, dtype=out_dtype)

    args_maker = lambda: [rng(shape, dtype)]

    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dtype={}_m={}_n={}_k={}".format(
          onp.dtype(dtype).name, m, n, k),
       "m": m, "n": n, "k": k, "dtype": dtype, "rng": jtu.rand_default()}
      for dtype in default_dtypes
      for n in [0, 4]
      for m in [None, 0, 1, 3, 4]
      for k in list(range(-4, 4))))
  def testTri(self, m, n, k, dtype, rng):
    onp_fun = lambda: onp.tri(n, M=m, k=k, dtype=dtype)
    lnp_fun = lambda: lnp.tri(n, M=m, k=k, dtype=dtype)
    args_maker = lambda: []
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_shape={}_k={}".format(
          op, jtu.format_shape_dtype_string(shape, dtype), k),
       "dtype": dtype, "shape": shape, "op": op, "k": k,
       "rng": jtu.rand_default()}
      for dtype in default_dtypes
      for shape in [shape for shape in all_shapes if len(shape) >= 1]
      for op in ["tril", "triu"]
      for k in list(range(-3, 3))))
  def testTriLU(self, dtype, shape, op, k, rng):
    onp_fun = lambda arg: getattr(onp, op)(arg, k=k)
    lnp_fun = lambda arg: getattr(lnp, op)(arg, k=k)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_k={}".format(
          jtu.format_shape_dtype_string(shape, dtype), k),
       "dtype": dtype, "shape": shape, "k": k, "rng": jtu.rand_default()}
      for dtype in default_dtypes
      for shape in [shape for shape in all_shapes if len(shape) in (1, 2)]
      for k in list(range(-4, 4))))
  def testDiag(self, shape, dtype, k, rng):
    onp_fun = lambda arg: onp.diag(arg, k)
    lnp_fun = lambda arg: lnp.diag(arg, k)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_offset={}_axis1={}_axis2={}".format(
          jtu.format_shape_dtype_string(shape, dtype), offset, axis1, axis2),
       "dtype": dtype, "shape": shape, "offset": offset, "axis1": axis1,
       "axis2": axis2, "rng": jtu.rand_default()}
      for dtype in default_dtypes
      for shape in [shape for shape in all_shapes if len(shape) >= 2]
      for axis1 in range(-len(shape), len(shape))
      for axis2 in [a for a in range(-len(shape), len(shape))
                    if a % len(shape) != axis1 % len(shape)]
      for offset in list(range(-4, 4))))
  def testDiagonal(self, shape, dtype, offset, axis1, axis2, rng):
    onp_fun = lambda arg: onp.diagonal(arg, offset, axis1, axis2)
    lnp_fun = lambda arg: lnp.diagonal(arg, offset, axis1, axis2)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_n={}".format(onp.dtype(dtype).name, n),
       "dtype": dtype, "n": n}
      for dtype in default_dtypes
      for n in list(range(4))))
  def testIdentity(self, n, dtype):
    onp_fun = lambda: onp.identity(n, dtype)
    lnp_fun = lambda: lnp.identity(n, dtype)
    args_maker = lambda: []
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_dtype_{}_offset={}_axis1={}_axis2={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          out_dtype, offset, axis1, axis2),
       "dtype": dtype, "out_dtype": out_dtype, "shape": shape, "offset": offset,
       "axis1": axis1, "axis2": axis2, "rng": jtu.rand_default()}
      for dtype in default_dtypes
      for out_dtype in [None] + number_dtypes
      for shape in [shape for shape in all_shapes if len(shape) >= 2]
      for (axis1, axis2) in itertools.combinations(range(len(shape)), 2)
      for offset in list(range(-4, 4))))
  def testTrace(self, shape, dtype, out_dtype, offset, axis1, axis2, rng):
    onp_fun = lambda arg: onp.trace(arg, offset, axis1, axis2, out_dtype)
    lnp_fun = lambda arg: lnp.trace(arg, offset, axis1, axis2, out_dtype)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(
          jtu.format_test_name_suffix("", [shape] * len(dtypes), dtypes)),
       "shape": shape, "dtypes": dtypes, "rng": rng}
      for dtypes in [
        [onp.float32],
        [onp.float32, onp.float32],
        [onp.float32, onp.int32, onp.float32],
        [onp.float32, onp.int64, onp.float32],
        [onp.float32, onp.int32, onp.float64],
      ]
      for shape in [(), (2,), (3, 4), (1, 100)]
      for rng in [jtu.rand_default()]))
  def testStack(self, shape, dtypes, rng):
    args_maker = lambda: [[rng(shape, dtype) for dtype in dtypes]]
    self._CheckAgainstNumpy(lnp.stack, onp.stack, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outdtype={}".format(
          jtu.format_shape_dtype_string(shape, fill_value_dtype),
          onp.dtype(out_dtype).name),
       "shape": shape, "fill_value_dtype": fill_value_dtype,
       "out_dtype": out_dtype, "rng": jtu.rand_default()}
      for shape in array_shapes
      for fill_value_dtype in default_dtypes
      for out_dtype in default_dtypes))
  def testFull(self, shape, fill_value_dtype, out_dtype, rng):
    onp_fun = lambda fill_value: onp.full(shape, fill_value, dtype=out_dtype)
    lnp_fun = lambda fill_value: lnp.full(shape, fill_value, dtype=out_dtype)
    args_maker = lambda: [rng((), fill_value_dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_filldtype={}_outdtype={}".format(
          jtu.format_shape_dtype_string(shape, in_dtype),
          onp.dtype(fill_value_dtype).name,
          onp.dtype(out_dtype).name),
       "shape": shape, "in_dtype": in_dtype,
       "fill_value_dtype": fill_value_dtype, "out_dtype": out_dtype,
       "rng": jtu.rand_default()}
      for shape in array_shapes
      for in_dtype in default_dtypes
      for fill_value_dtype in default_dtypes
      for out_dtype in default_dtypes))
  def testFullLike(self, shape, in_dtype, fill_value_dtype, out_dtype, rng):
    onp_fun = lambda x, fill_value: onp.full_like(x, fill_value, dtype=out_dtype)
    lnp_fun = lambda x, fill_value: lnp.full_like(x, fill_value, dtype=out_dtype)
    args_maker = lambda: [rng(shape, in_dtype), rng((), fill_value_dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_{}sections".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, num_sections),
       "shape": shape, "num_sections": num_sections, "axis": axis,
       "dtype": dtype, "rng": jtu.rand_default()}
      for shape, axis, num_sections in [
          ((3,), 0, 3), ((12,), 0, 3), ((12, 4), 0, 4), ((12, 4), 1, 2),
          ((2, 3, 4), -1, 2), ((2, 3, 4), -2, 3)]
      for dtype in default_dtypes))
  def testSplitStaticInt(self, shape, num_sections, axis, dtype, rng):
    onp_fun = lambda x: onp.split(x, num_sections, axis=axis)
    lnp_fun = lambda x: lnp.split(x, num_sections, axis=axis)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_{}sections".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, num_sections),
       "shape": shape, "num_sections": num_sections, "axis": axis,
       "dtype": dtype, "rng": jtu.rand_default()}
      for shape, axis, num_sections in [
          ((12, 4), 0, 4), ((12, 4), 1, 2),
          ((2, 3, 4), 2, 2), ((4, 3, 4), 0, 2)]
      for dtype in default_dtypes))
  def testHVDSplit(self, shape, num_sections, axis, dtype, rng):
    def fn(module, axis):
      if axis == 0:
        return module.vsplit
      elif axis == 1:
        return module.hsplit
      else:
        assert axis == 2
        return module.dsplit

    onp_fun = lambda x: fn(onp, axis)(x, num_sections)
    lnp_fun = lambda x: fn(lnp, axis)(x, num_sections)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          jtu.format_shape_dtype_string(out_shape, dtype)),
       "arg_shape": arg_shape, "out_shape": out_shape, "dtype": dtype,
       "rng": jtu.rand_default()}
      for dtype in default_dtypes
      for arg_shape, out_shape in [
          (jtu.NUMPY_SCALAR_SHAPE, (1, 1, 1)),
          ((), (1, 1, 1)),
          ((7, 0), (0, 42, 101)),
          ((3, 4), 12),
          ((3, 4), (12,)),
          ((3, 4), -1),
          ((2, 1, 4), (-1,)),
          ((2, 2, 4), (2, 8))
      ]))
  def testReshape(self, arg_shape, out_shape, dtype, rng):
    onp_fun = lambda x: onp.reshape(x, out_shape)
    lnp_fun = lambda x: lnp.reshape(x, out_shape)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_expanddim={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype), dim),
       "arg_shape": arg_shape, "dtype": dtype, "dim": dim,
       "rng": jtu.rand_default()}
      for arg_shape in [(), (3,), (3, 4)]
      for dtype in default_dtypes
      for dim in range(-len(arg_shape)+1, len(arg_shape))))
  def testExpandDimsStaticDim(self, arg_shape, dtype, dim, rng):
    onp_fun = lambda x: onp.expand_dims(x, dim)
    lnp_fun = lambda x: lnp.expand_dims(x, dim)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_axes=({},{})".format(
          jtu.format_shape_dtype_string(arg_shape, dtype), ax1, ax2),
       "arg_shape": arg_shape, "dtype": dtype, "ax1": ax1, "ax2": ax2,
       "rng": jtu.rand_default()}
      for arg_shape, ax1, ax2 in [
          ((3, 4), 0, 1), ((3, 4), 1, 0), ((3, 4, 5), 1, 2),
          ((3, 4, 5), -1, -2), ((3, 4, 5), 0, 1)]
      for dtype in default_dtypes))
  def testSwapAxesStaticAxes(self, arg_shape, dtype, ax1, ax2, rng):
    onp_fun = lambda x: onp.swapaxes(x, ax1, ax2)
    lnp_fun = lambda x: lnp.swapaxes(x, ax1, ax2)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_axis={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype), ax),
       "arg_shape": arg_shape, "dtype": dtype, "ax": ax,
       "rng": jtu.rand_default()}
      for arg_shape, ax in [
          ((3, 1), None),
          ((3, 1), 1),
          ((1, 3, 1), (0, 2)),
          ((1, 4, 1), (0,))]
      for dtype in default_dtypes))
  def testSqueeze(self, arg_shape, dtype, ax, rng):
    onp_fun = lambda x: onp.squeeze(x, ax)
    lnp_fun = lambda x: lnp.squeeze(x, ax)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_arg{}".format(i), "arg": arg}
      for i, arg in enumerate([
          [1, 2, 3], [1., 2., 3.],
          [[1, 2], [3, 4], [5, 6]], [[1, 2.], [3, 4], [5, 6]],
          [[3, onp.array(2), 1], onp.arange(3.)],
      ])))
  def testArray(self, arg):
    args_maker = lambda: [arg]
    self._CheckAgainstNumpy(onp.array, lnp.array, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp.array, args_maker, check_dtypes=True)

  def testArrayAsarrayMethod(self):
    class arraylike(object):
      def __asarray__(self, dtype=None):
        return 3.
    a = arraylike()
    ans = lnp.array(a)
    assert ans == 3.

  def testAllClose(self):
    rng = onp.random.RandomState(0)
    x = rng.randn(2, 2)
    y = rng.randn(2)

    def same(list1, list2):
      allclose = functools.partial(lnp.allclose, atol=1e-3, rtol=1e-3)
      elements_close = list(map(allclose, list1, list2))
      return lnp.all(lnp.array(elements_close))

    csame = api.jit(same)

    a1 = same((x, y), (x, y))
    a2 = csame((x, y), (x, y))
    a3 = csame((x, y), (x, 2 * y))

    self.assertTrue(a1)
    self.assertTrue(a2)
    self.assertFalse(a3)

  @jtu.skip_on_devices("tpu")  # TODO(mattjj): investigate this failure
  def testOnesBroadcastingConstantHandler(self):
    # TODO(mattjj): update this test for jax3
    self.skipTest("test needs jax3 update")

    def fun(x):
      ones = lnp.ones((3, 4))
      assert isinstance(ones, onp.ndarray) and ones.strides == (0, 0)

      # To check that the constant handler generates a Broadcast for stride-zero
      # arrays, we monkey-patch the client instance.
      # TODO(mattjj): once we have better HLO dumping and inspecting facilities,
      # we can check the HLO more directly.
      c = x._node.c
      Broadcast = c.Broadcast  # pylint: disable=invalid-name
      was_called = []
      c.Broadcast = lambda *args: was_called.append(True) or Broadcast(*args)
      out = x + ones  # the ndarray constant handler should call Broadcast here
      assert was_called, "Broadcast was not called."

      return out

    fun = api.jit(fun)
    out_val = fun(lnp.ones(4))
    self.assertAllClose(out_val, onp.full((3, 4), 2.), check_dtypes=False)

  def testZeroStridesConstantHandler(self):
    raw_const = onp.random.RandomState(0).randn(1, 2, 1, 1, 5, 1)
    const = onp.broadcast_to(raw_const, (3, 2, 3, 4, 5, 6))

    def fun(x):
      return x * const

    fun = api.jit(fun)
    out_val = fun(3.)
    self.assertAllClose(out_val, 3. * const, check_dtypes=False)

  def testIsInstanceNdarrayDuringTracing(self):
    arr = onp.ones(3)

    @api.jit
    def f(x):
      self.assertIsInstance(x, lnp.ndarray)
      return lnp.sum(x)

    f(arr)


  def testNonArrayErrorMessage(self):
    x = [1., 2.]
    y = onp.array([3., 4.])

    def g(x, y):
      return lnp.add(x, y)

    def f(x, y):
      return lnp.dot(x, y)

    self.assertRaises(TypeError, lambda: g(x, y))
    self.assertRaises(TypeError, lambda: f(x, y))
    self.assertRaises(TypeError, lambda: api.jit(g)(x, y))
    self.assertRaises(TypeError, lambda: api.jit(f)(x, y))

  def testAbstractionErrorMessage(self):

    @api.jit
    def f(x, n):
      for _ in range(n):
        x = x * x
      return x

    self.assertRaises(TypeError, lambda: f(3., 3))

    @api.jit
    def g(x):
      if x > 0.:
        return x * 2
      else:
        return x + 2

    self.assertRaises(TypeError, lambda: g(3.))

  def testTracingPrimitiveWithNoTranslationErrorMessage(self):
    # TODO(mattjj): update this for jax3
    self.skipTest("test needs jax3 update")
    foo = lnp._not_implemented(lambda x: x)

    # No error if there's no tracing.
    foo(onp.arange(3))

    cfoo = api.jit(foo)
    self.assertRaises(NotImplementedError, lambda: cfoo(onp.arange(3)))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "rng": rng, "shape": shape, "dtype": dtype, "axis": axis}
      for shape in [(3,), (2, 3)]
      for dtype in default_dtypes
      for axis in range(-len(shape), len(shape))  # Test negative axes
      for rng in [jtu.rand_default()]))
  def testFlip(self, shape, dtype, axis, rng):
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    lnp_op = lambda x: lnp.flip(x, axis)
    onp_op = lambda x: onp.flip(x, axis)
    self._CheckAgainstNumpy(onp_op, lnp_op, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_op, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "rng": rng, "shape": shape, "dtype": dtype}
      for shape in [(3,), (2, 3), (3, 2, 4)]
      for dtype in default_dtypes
      for rng in [jtu.rand_default()]))
  def testFlipud(self, shape, dtype, rng):
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    lnp_op = lambda x: lnp.flipud(x)
    onp_op = lambda x: onp.flipud(x)
    self._CheckAgainstNumpy(onp_op, lnp_op, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_op, args_maker, check_dtypes=True)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "rng": rng, "shape": shape, "dtype": dtype}
      for shape in [(3, 2), (2, 3), (3, 2, 4)]
      for dtype in default_dtypes
      for rng in [jtu.rand_default()]))
  def testFliplr(self, shape, dtype, rng):
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    lnp_op = lambda x: lnp.fliplr(x)
    onp_op = lambda x: onp.fliplr(x)
    self._CheckAgainstNumpy(onp_op, lnp_op, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_op, args_maker, check_dtypes=True)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_k={}_axes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), k, axes),
       "rng": rng, "shape": shape, "dtype": dtype, "k": k, "axes": axes}
      for shape, axes in [
          [(2, 3), (0, 1)],
          [(2, 3), (1, 0)],
          [(4, 3, 2), (0, 2)],
          [(4, 3, 2), (2, 1)],
      ]
      for k in range(-3, 4)
      for dtype in default_dtypes
      for rng in [jtu.rand_default()]))
  def testRot90(self, shape, dtype, k, axes, rng):
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    lnp_op = lambda x: lnp.rot90(x, k, axes)
    onp_op = lambda x: onp.rot90(x, k, axes)
    self._CheckAgainstNumpy(onp_op, lnp_op, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_op, args_maker, check_dtypes=True)

  # TODO(mattjj): test infix operator overrides

  def testRavel(self):
    rng = onp.random.RandomState(0)
    args_maker = lambda: [rng.randn(3, 4).astype("float32")]
    self._CompileAndCheck(lambda x: x.ravel(), args_maker, check_dtypes=True)

  def testAstype(self):
    rng = onp.random.RandomState(0)
    args_maker = lambda: [rng.randn(3, 4).astype("float32")]
    op = lambda x: x.astype(lnp.int32)
    self._CheckAgainstNumpy(op, op, args_maker, check_dtypes=True)
    self._CompileAndCheck(op, args_maker, check_dtypes=True)

  # TODO(mattjj): test other ndarray-like method overrides

  def testOnpMean(self):
    # from https://github.com/google/jax/issues/125
    x = lax.add(lnp.eye(3), 0.)
    ans = onp.mean(x)
    self.assertAllClose(ans, onp.array([1./3, 1./3, 1./3]), check_dtypes=False)

  # TODO(mattjj): more exhaustive arange tests
  def testArangeOnFloats(self):
    # from https://github.com/google/jax/issues/145
    expected = onp.arange(0.0, 1.0, 0.1)
    ans = lnp.arange(0.0, 1.0, 0.1)
    self.assertAllClose(expected, ans, check_dtypes=True)

  def testSortManually(self):
    # manual tests for sort are nice because we don't have to worry about ties.
    # lax.sort is tested combinatorially.
    ans = lnp.sort(onp.array([16, 15, 23, 42, 8, 4]))
    expected = onp.array([4, 8, 15, 16, 23, 42])
    self.assertAllClose(expected, ans, check_dtypes=True)

    a = onp.array([[1, 4], [3, 1]])
    ans = lnp.sort(a, axis=None)
    expected = onp.array([[1, 1, 3, 4]])
    self.assertAllClose(expected, ans, check_dtypes=True)

    a = onp.array([[1, 4], [3, 1]])
    ans = lnp.sort(a)  # last axis
    expected = onp.array([[1, 4], [1, 3]])
    self.assertAllClose(expected, ans, check_dtypes=True)

    a = onp.array([[1, 4], [3, 1]])
    ans = lnp.sort(a, axis=0)
    expected = onp.array([[1, 1], [3, 4]])
    self.assertAllClose(expected, ans, check_dtypes=True)

  def testArgsortManually(self):
    x = onp.array([16, 15, 23, 42, 8, 4])
    ans = lnp.argsort(x)
    expected = onp.argsort(x)
    self.assertAllClose(expected, ans, check_dtypes=False)

    x = onp.array([[16, 15, 23], [42, 8, 4]])
    ans = lnp.argsort(x, axis=0)
    expected = onp.argsort(x, axis=0)
    self.assertAllClose(expected, ans, check_dtypes=False)

    x = onp.array([[16, 15, 23], [42, 8, 4]])
    ans = lnp.argsort(x, axis=1)
    expected = onp.argsort(x, axis=1)
    self.assertAllClose(expected, ans, check_dtypes=False)

    x = onp.array([[16, 15, 23], [42, 8, 4]])
    ans = lnp.argsort(x, axis=None)
    expected = onp.argsort(x, axis=None)
    self.assertAllClose(expected, ans, check_dtypes=False)

    x = onp.array([[16, 15, 23], [42, 8, 4]])
    ans = lnp.argsort(x)
    expected = onp.argsort(x)
    self.assertAllClose(expected, ans, check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_index={}_axis={}_mode={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          jtu.format_shape_dtype_string(index_shape, index_dtype),
          axis, mode),
       "rng": rng, "rng_indices": rng_indices, "shape": shape,
       "index_shape": index_shape, "dtype": dtype, "index_dtype": index_dtype,
       "axis": axis, "mode": mode}
      for shape in [(3,), (3, 4), (3, 4, 5)]
      for index_shape in scalar_shapes + [(3,), (2, 1, 3)]
      for axis in itertools.chain(range(-len(shape), len(shape)), [None])
      for dtype in all_dtypes
      for index_dtype in int_dtypes
      for mode in ['wrap', 'clip']
      for rng in [jtu.rand_default()]
      for rng_indices in [jtu.rand_int(-5, 5)]))
  def testTake(self, shape, dtype, index_shape, index_dtype, axis, mode, rng,
               rng_indices):
    if (FLAGS.jax_test_dut.startswith("tpu")
        and onp.issubdtype(dtype, onp.complexfloating)):
      self.skipTest("skipping complex dtype on TPU")  # TODO(mattjj): investigate failures

    def args_maker():
      x = rng(shape, dtype)
      i = rng_indices(index_shape, index_dtype)
      return x, i

    lnp_op = lambda x, i: lnp.take(x, i, axis=axis, mode=mode)
    onp_op = lambda x, i: onp.take(x, i, axis=axis, mode=mode)
    self._CheckAgainstNumpy(lnp_op, onp_op, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_op, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "rng": rng, "shape": shape, "dtype": dtype, "axis": axis}
      for shape in [(3,), (3, 4), (3, 4, 5)]
      for axis in itertools.chain(range(len(shape)), [-1], [None])
      for dtype in default_dtypes
      for rng in [jtu.rand_default()]))
  def testTakeAlongAxis(self, shape, dtype, axis, rng):
    def args_maker():
      x = rng(shape, dtype)
      i = onp.argsort(x, axis=axis)
      return x, i

    lnp_op = lambda x, i: lnp.take_along_axis(x, i, axis=axis)

    if hasattr(onp, "take_along_axis"):
      onp_op = lambda x, i: onp.take_along_axis(x, i, axis=axis)
      self._CheckAgainstNumpy(lnp_op, onp_op, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_op, args_maker, check_dtypes=True)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_n={}_increasing={}".format(
          jtu.format_shape_dtype_string([shape], dtype),
          n, increasing),
       "dtype": dtype, "shape": shape, "n": n, "increasing": increasing,
       "rng": jtu.rand_default()}
      for dtype in inexact_dtypes
      for shape in [0, 5]
      for n in [2, 4]
      for increasing in [False, True]))
  def testVander(self, shape, dtype, n, increasing, rng):
    onp_fun = lambda arg: onp.vander(arg, N=n, increasing=increasing)
    lnp_fun = lambda arg: lnp.vander(arg, N=n, increasing=increasing)
    args_maker = lambda: [rng([shape], dtype)]
    # np.vander seems to return float64 for all floating types. We could obey
    # those semantics, but they seem like a bug.
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=False)


if __name__ == "__main__":
  absltest.main()
