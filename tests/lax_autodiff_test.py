# Copyright 2020 Google LLC
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
from jax.test_util import check_grads

from tests.lax_test import compatible_shapes, num_float_bits

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS


GradTestSpec = collections.namedtuple(
    "GradTestSpec",
    ["op", "nargs", "order", "rng_factory", "dtypes", "name", "tol"])
def grad_test_spec(op, nargs, order, rng_factory, dtypes, name=None, tol=None):
  return GradTestSpec(
      op, nargs, order, rng_factory, dtypes, name or op.__name__, tol)

grad_inexact_dtypes = jtu.basic_float_dtypes + jtu.complex_dtypes

LAX_GRAD_OPS = [
    grad_test_spec(lax.neg, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.floor, nargs=1, order=2,
                   rng_factory=partial(jtu.rand_uniform, low=0.1, high=0.4),
                   dtypes=jtu.basic_float_dtypes),
    grad_test_spec(lax.ceil, nargs=1, order=2,
                   rng_factory=partial(jtu.rand_uniform, low=0.1, high=0.4),
                   dtypes=jtu.basic_float_dtypes),
    grad_test_spec(lax.round, nargs=1, order=2,
                   rng_factory=partial(jtu.rand_uniform, low=0.1, high=0.4),
                   dtypes=jtu.basic_float_dtypes),

    grad_test_spec(lax.exp, nargs=1, order=2, rng_factory=jtu.rand_small,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.expm1, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.log, nargs=1, order=2, rng_factory=jtu.rand_positive,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.log1p, nargs=1, order=2, rng_factory=jtu.rand_positive,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.sinh, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=jtu.basic_float_dtypes + [onp.complex64], tol=1e-5),
    grad_test_spec(lax.cosh, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes, tol=1e-5),
    grad_test_spec(lax.tanh, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes, tol=1e-5),
    grad_test_spec(lax.sin, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes, tol={onp.float32: 5e-1}),
    grad_test_spec(lax.cos, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.tan, nargs=1, order=2,
                   rng_factory=partial(jtu.rand_uniform, low=-1.3, high=1.3),
                   dtypes=grad_inexact_dtypes, tol=1e-3),
    grad_test_spec(lax.asin, nargs=1, order=2,
                   rng_factory=partial(jtu.rand_uniform, low=-1.3, high=1.3),
                   dtypes=jtu.basic_float_dtypes, tol=1e-3),
    grad_test_spec(lax.acos, nargs=1, order=2,
                   rng_factory=partial(jtu.rand_uniform, low=-1.3, high=1.3),
                   dtypes=jtu.basic_float_dtypes, tol=2e-2),
    # TODO(proteneer): atan2 input is already a representation of a
    # complex number. Need to think harder about what this even means
    # if each input itself is a complex number.
    grad_test_spec(lax.atan2, nargs=2, order=2, rng_factory=jtu.rand_default,
                   dtypes=jtu.basic_float_dtypes),

    grad_test_spec(lax.erf, nargs=1, order=2, rng_factory=jtu.rand_small,
                   dtypes=jtu.basic_float_dtypes),
    grad_test_spec(lax.erfc, nargs=1, order=2, rng_factory=jtu.rand_small,
                   dtypes=jtu.basic_float_dtypes),
    grad_test_spec(lax.erf_inv, nargs=1, order=2, rng_factory=jtu.rand_small,
                   dtypes=jtu.basic_float_dtypes),
    # grad_test_spec(lax.lgamma, nargs=1, order=2, rng_factory=jtu.rand_small,
    #                dtypes=jtu.basic_float_dtypes),  # TODO(mattjj): enable
    grad_test_spec(lax.bessel_i0e, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=jtu.basic_float_dtypes),
    grad_test_spec(lax.bessel_i1e, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=jtu.basic_float_dtypes),

    grad_test_spec(lax.real, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=jtu.complex_dtypes),
    grad_test_spec(lax.imag, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=jtu.complex_dtypes),
    grad_test_spec(lax.complex, nargs=2, order=2, rng_factory=jtu.rand_default,
                   dtypes=jtu.basic_float_dtypes),
    grad_test_spec(lax.conj, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.abs, nargs=1, order=2, rng_factory=jtu.rand_positive,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.pow, nargs=2, order=2, rng_factory=jtu.rand_positive,
                   dtypes=grad_inexact_dtypes, tol={onp.float32: 3e-1}),

    grad_test_spec(lax.add, nargs=2, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.sub, nargs=2, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.mul, nargs=2, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.div, nargs=2, order=1, rng_factory=jtu.rand_not_small,
                   dtypes=grad_inexact_dtypes),

    grad_test_spec(lax.max, nargs=2, order=2, rng_factory=jtu.rand_default,
                   dtypes=jtu.basic_float_dtypes),
    grad_test_spec(lax.min, nargs=2, order=2, rng_factory=jtu.rand_default,
                   dtypes=jtu.basic_float_dtypes),
    # TODO(mattjj): make some-equal checks more robust, enable second-order
    # grad_test_spec(lax.max, nargs=2, order=1, rng_factory=jtu.rand_some_equal,
    #                dtypes=jtu.basic_float_dtypes, name="MaxSomeEqual"),
    # grad_test_spec(lax.min, nargs=2, order=1, rng_factory=jtu.rand_some_equal,
    #                dtypes=jtu.basic_float_dtypes, name="MinSomeEqual"),
]

GradSpecialValuesTestSpec = collections.namedtuple(
    "GradSpecialValuesTestSpec", ["op", "values", "tol"])
def grad_special_values_test_spec(op, values, tol=None):
  return GradSpecialValuesTestSpec(op, values, tol)

LAX_GRAD_SPECIAL_VALUE_TESTS = [
    grad_special_values_test_spec(
      lax.sinh, [0.],
      tol={onp.float32: 1e-2} if jtu.device_under_test() == "tpu" else None),
    grad_special_values_test_spec(
      lax.cosh, [0.],
      tol={onp.float32: 1e-2} if jtu.device_under_test() == "tpu" else None),
    grad_special_values_test_spec(lax.tanh, [0., 1000.]),
    grad_special_values_test_spec(lax.sin, [0., onp.pi, onp.pi/2., onp.pi/4.]),
    grad_special_values_test_spec(lax.cos, [0., onp.pi, onp.pi/2., onp.pi/4.]),
    grad_special_values_test_spec(lax.tan, [0.]),
    grad_special_values_test_spec(lax.asin, [0.]),
    grad_special_values_test_spec(lax.acos, [0.]),
    grad_special_values_test_spec(lax.atan, [0., 1000.]),
    grad_special_values_test_spec(lax.erf, [0., 10.]),
    grad_special_values_test_spec(lax.erfc, [0., 10.]),
]


def check_grads_bilinear(f, args, order,
                         modes=["fwd", "rev"], atol=None, rtol=None):
  # Can use large eps to make up for numerical inaccuracies since the op is
  # bilinear (relying on the fact that we only check one arg at a time)
  lhs, rhs = args
  check_grads(lambda lhs: f(lhs, rhs), (lhs,), order,
              modes=modes, atol=atol, rtol=rtol, eps=1.)
  check_grads(lambda rhs: f(lhs, rhs), (rhs,), order,
              modes=modes, atol=atol, rtol=rtol, eps=1.)

class LaxAutodiffTest(jtu.JaxTestCase):

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(
            rec.name, shapes, itertools.repeat(dtype)),
         "op": rec.op, "rng_factory": rec.rng_factory, "shapes": shapes, "dtype": dtype,
         "order": rec.order, "tol": rec.tol}
        for shape_group in compatible_shapes
        for shapes in itertools.combinations_with_replacement(shape_group, rec.nargs)
        for dtype in rec.dtypes)
      for rec in LAX_GRAD_OPS))
  def testOpGrad(self, op, rng_factory, shapes, dtype, order, tol):
    rng = rng_factory(self.rng())
    if jtu.device_under_test() == "tpu" and op is lax.pow:
      raise SkipTest("pow grad imprecise on tpu")
    tol = jtu.join_tolerance(1e-1, tol) if num_float_bits(dtype) == 32 else tol
    args = tuple(rng(shape, dtype) for shape in shapes)
    check_grads(op, args, order, ["fwd", "rev"], tol, tol)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
          {"testcase_name": "_{}_{}".format(rec.op.__name__, special_value),
           "op": rec.op, "special_value": special_value, "tol": rec.tol}
          for special_value in rec.values)
      for rec in LAX_GRAD_SPECIAL_VALUE_TESTS))
  def testOpGradSpecialValue(self, op, special_value, tol):
    check_grads(op, (special_value,), 2, ["fwd", "rev"], rtol=tol, atol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_from_dtype={}_to_dtype={}".format(
          jtu.dtype_str(from_dtype), jtu.dtype_str(to_dtype)),
       "from_dtype": from_dtype, "to_dtype": to_dtype, "rng_factory": rng_factory}
      for from_dtype, to_dtype in itertools.product(
          jtu.float_dtypes + jtu.complex_dtypes, repeat=2)
      for rng_factory in [jtu.rand_default]))
  def testConvertElementTypeGrad(self, from_dtype, to_dtype, rng_factory):
    rng = rng_factory(self.rng())
    tol = max(jtu.tolerance(to_dtype, jtu.default_gradient_tolerance),
              jtu.tolerance(from_dtype, jtu.default_gradient_tolerance))
    args = (rng((2, 3), from_dtype),)
    convert_element_type = lambda x: lax.convert_element_type(x, to_dtype)
    convert_element_type = jtu.ignore_warning(category=onp.ComplexWarning)(
      convert_element_type)
    check_grads(convert_element_type, args, 2, ["fwd", "rev"], tol, tol, eps=1.)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype,
       "rng_factory": rng_factory}
      for shape in [(), (2, 3)]
      for dtype in jtu.basic_float_dtypes
      for rng_factory in [jtu.rand_default]))
  def testClampGrad(self, shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    operand = rng(shape, dtype)
    low = operand - dtype(10)
    high = operand + dtype(10)
    # Avoids points near the boundary where the gradient may be inaccurate.
    check_grads(lax.clamp, (operand, low, high), 2, ["fwd", "rev"], eps=1e-2)
    check_grads(lax.clamp, (low, operand, high), 2, ["fwd", "rev"], eps=1e-2)
    check_grads(lax.clamp, (low, high, operand), 2, ["fwd", "rev"], eps=1e-2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dim={}_baseshape=[{}]_dtype={}_narrs={}".format(
          dim, ",".join(str(d) for d in base_shape), onp.dtype(dtype).name,
          num_arrs),
       "dim": dim, "base_shape": base_shape, "dtype": dtype,
       "num_arrs": num_arrs, "rng_factory": rng_factory}
      for num_arrs in [3]
      for dtype in jtu.float_dtypes
      for base_shape in [(4,), (3, 4), (2, 3, 4)]
      for dim in range(len(base_shape))
      for rng_factory in [jtu.rand_default]))
  def testConcatenateGrad(self, dim, base_shape, dtype, num_arrs, rng_factory):
    rng = rng_factory(self.rng())
    shapes = [base_shape[:dim] + (size,) + base_shape[dim+1:]
              for size, _ in zip(itertools.cycle([3, 1, 4]), range(num_arrs))]
    operands = tuple(rng(shape, dtype) for shape in shapes)
    concatenate = lambda *args: lax.concatenate(args, dim)
    check_grads(concatenate, operands, 2, ["fwd", "rev"], eps=1.)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_strides={}_padding={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               strides, padding),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "strides": strides, "padding": padding, "rng_factory": rng_factory,}
       for lhs_shape, rhs_shape, all_strides in itertools.chain(
           [((b, i, 3, 4), (j, i, 1, 2), [(1, 1), (1, 2), (2, 1)])
            for b, i, j in itertools.product([2, 3], repeat=3)],
           [((4, 2, 1), (3, 2, 1), [(1,)])])
       for strides in all_strides
       for dtype in jtu.float_dtypes
       for padding in ["VALID", "SAME"]
       for rng_factory in [jtu.rand_small]))
  def testConvGrad(self, lhs_shape, rhs_shape, dtype, strides, padding, rng_factory):
    rng = rng_factory(self.rng())
    lhs = rng(lhs_shape, dtype)
    rhs = rng(rhs_shape, dtype)
    conv = partial(lax.conv, window_strides=strides, padding=padding,
                   precision=lax.Precision.HIGHEST)
    check_grads_bilinear(conv, (lhs, rhs), order=2, modes=["fwd", "rev"],
                         atol=1e-2, rtol=1e-2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_strides={}_padding={}_lhs_dilation={}_"
       "rhs_dilation={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               strides, padding, lhs_dil, rhs_dil),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "strides": strides, "padding": padding, "lhs_dil": lhs_dil,
       "rhs_dil": rhs_dil, "rng_factory": rng_factory}
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
       for dtype in jtu.float_dtypes
       for padding in all_pads
       for rng_factory in [jtu.rand_small]))
  def testConvWithGeneralPaddingGrad(self, lhs_shape, rhs_shape, dtype, strides,
                                     padding, lhs_dil, rhs_dil, rng_factory):
    rng = rng_factory(self.rng())
    lhs = rng(lhs_shape, dtype)
    rhs = rng(rhs_shape, dtype)
    conv = partial(lax.conv_with_general_padding, window_strides=strides,
                   padding=padding, lhs_dilation=lhs_dil, rhs_dilation=rhs_dil,
                   precision=lax.Precision.HIGHEST)
    check_grads_bilinear(conv, (lhs, rhs), order=2, modes=["fwd", "rev"],
                         atol=1e-2, rtol=1e-2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_strides={}_padding={}_lhs_dilation={}_"
       "rhs_dilation={}_dims={}_feature_group_count={}_batch_group_count={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               strides, padding, lhs_dil, rhs_dil, ",".join(dim_nums),
               feature_group_count, batch_group_count),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "strides": strides, "padding": padding, "lhs_dil": lhs_dil,
       "rhs_dil": rhs_dil, "rng_factory": rng_factory, "dimension_numbers": dim_nums,
       "perms": perms, "feature_group_count": feature_group_count,
       "batch_group_count": batch_group_count}
      for batch_group_count, feature_group_count in ([(1, 1), (2, 1), (1, 2)])
      for lhs_shapes, rhs_shape, all_strides, lhs_dils, rhs_dils in [
          ([(b * batch_group_count, i * feature_group_count, 6, 7),
            (b * batch_group_count, i * feature_group_count, 0, 4)],  # lhs_shape
           (j * batch_group_count * feature_group_count, i, 1, 2),  # rhs_shape
           [(1, 1), (1, 2), (2, 1)],  # strides
           [(1, 1), (2, 1)],  # lhs_dils
           [(1, 1), (2, 2)])  # rhs_dils
          for b, i, j in itertools.product([1, 2], repeat=3)]
      for lhs_shape in lhs_shapes
      for strides in all_strides
      for rhs_dil in rhs_dils
      for lhs_dil in lhs_dils
      for dtype in jtu.basic_float_dtypes
      for padding in ([((0, 0), (0, 0)), ((1, 0), (0, 1))] +
        ([((0, -1), (0, 0))] if lhs_shape[2] != 0 else []))
      for dim_nums, perms in [
          (("NCHW", "OIHW", "NCHW"), ([0, 1, 2, 3], [0, 1, 2, 3])),
          (("NHWC", "HWIO", "NHWC"), ([0, 2, 3, 1], [2, 3, 1, 0])),
          (("NHWC", "OIHW", "NCHW"), ([0, 2, 3, 1], [0, 1, 2, 3]))]
      for rng_factory in [jtu.rand_default]
  ))
  def testConvGeneralDilatedGrad(self, lhs_shape, rhs_shape, dtype, strides,
                                 padding, lhs_dil, rhs_dil, dimension_numbers,
                                 perms, feature_group_count, batch_group_count,
                                 rng_factory):
    if dtype == onp.float16:
      raise SkipTest("float16 numerical issues")  # TODO(mattjj): resolve

    rng = rng_factory(self.rng())
    tol = {dtypes.bfloat16: 1e-0, onp.float16: 5e-1, onp.float32: 2e-4}

    # permute shapes to match dim_spec, scale by feature_group_count
    lhs_perm, rhs_perm = perms
    lhs_shape = list(onp.take(lhs_shape, lhs_perm))
    rhs_shape = list(onp.take(rhs_shape, rhs_perm))

    lhs = rng(lhs_shape, dtype)
    rhs = rng(rhs_shape, dtype)
    conv = partial(lax.conv_general_dilated, window_strides=strides,
                   padding=padding, lhs_dilation=lhs_dil, rhs_dilation=rhs_dil,
                   dimension_numbers=dimension_numbers,
                   feature_group_count=feature_group_count,
                   batch_group_count=batch_group_count,
                   precision=lax.Precision.HIGHEST)
    check_grads_bilinear(conv, (lhs, rhs), order=2, modes=["fwd", "rev"],
                         atol=tol, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_lhs_shape={}_rhs_shape={}".format(
          jtu.format_shape_dtype_string(lhs_shape, dtype),
          jtu.format_shape_dtype_string(rhs_shape, dtype)),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "rng_factory": jtu.rand_default}
      for lhs_shape in [(2,), (3, 2)] for rhs_shape in [(2,), (2, 4)]
      for dtype in jtu.float_dtypes))
  def testDotGrad(self, lhs_shape, rhs_shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    tol = {onp.float16: 1e-1, onp.float32: 1e-4}
    lhs = rng(lhs_shape, dtype)
    rhs = rng(rhs_shape, dtype)
    dot = partial(lax.dot, precision=lax.Precision.HIGHEST)
    check_grads_bilinear(dot, (lhs, rhs), order=2, modes=["fwd", "rev"],
                         atol=tol, rtol=tol)
    # check that precision config is preserved
    result, pullback = api.vjp(dot, lhs, rhs)
    gresult = lax.zeros_like_array(result)
    s = str(api.make_jaxpr(pullback)(gresult))
    assert "precision=HIGHEST" in s

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_dimension_numbers={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               dimension_numbers),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "dimension_numbers": dimension_numbers, "rng_factory": jtu.rand_small}
      for lhs_shape, rhs_shape, dimension_numbers in [
          ((3, 2), (2, 4), (([1], [0]), ([], []))),
          ((3, 5), (2, 5), (([1], [1]), ([], []))),
          ((5, 3), (5, 2), (([0], [0]), ([], []))),
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0]))),
      ]
      for dtype in jtu.float_dtypes))
  def testDotGeneralContractAndBatchGrads(self, lhs_shape, rhs_shape, dtype,
                                          dimension_numbers, rng_factory):
    rng = rng_factory(self.rng())
    lhs = rng(lhs_shape, dtype)
    rhs = rng(rhs_shape, dtype)
    dot_general = partial(lax.dot_general, dimension_numbers=dimension_numbers,
                          precision=lax.Precision.HIGHEST)
    check_grads_bilinear(dot_general, (lhs, rhs), order=2, modes=["fwd", "rev"])
    # check that precision config is preserved
    result, pullback = api.vjp(dot_general, lhs, rhs)
    gresult = lax.zeros_like_array(result)
    s = str(api.make_jaxpr(pullback)(gresult))
    assert "precision=HIGHEST" in s

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_dtype={}_broadcast_sizes={}".format(
          shape, onp.dtype(dtype).name, broadcast_sizes),
       "shape": shape, "dtype": dtype, "broadcast_sizes": broadcast_sizes,
       "rng_factory": rng_factory}
      for shape in [(), (2, 3)]
      for dtype in jtu.float_dtypes
      for broadcast_sizes in [(), (2,), (1, 2)]
      for rng_factory in [jtu.rand_default]))
  def testBroadcastGrad(self, shape, dtype, broadcast_sizes, rng_factory):
    rng = rng_factory(self.rng())
    args = (rng(shape, dtype),)
    broadcast = lambda x: lax.broadcast(x, broadcast_sizes)
    check_grads(broadcast, args, 2, ["fwd", "rev"], eps=1.)

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
      ]
      for dtype in jtu.float_dtypes
      for rng_factory in [jtu.rand_default]))
  def testBroadcastInDimGrad(self, inshape, dtype, outshape, dimensions, rng_factory):
    rng = rng_factory(self.rng())
    operand = rng(inshape, dtype)
    broadcast_in_dim = lambda x: lax.broadcast_in_dim(x, outshape, dimensions)
    check_grads(broadcast_in_dim, (operand,), 2, ["fwd", "rev"], eps=1.)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}_perm={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          jtu.format_shape_dtype_string(out_shape, dtype),
          permutation),
       "arg_shape": arg_shape, "out_shape": out_shape, "dtype": dtype,
       "rng_factory": rng_factory, "permutation": permutation}
      for dtype in jtu.float_dtypes
      for arg_shape, out_shape, permutation in [
          [(3, 4), (12,), None],
          [(2, 1, 4), (8,), None],
          [(2, 2, 4), (2, 8), None],
          [(3, 4), (12,), (0, 1)],
          [(3, 4), (12,), (1, 0)],
          [(2, 1, 4), (8,), (0, 2, 1)],
          [(2, 1, 4), (8,), (2, 0, 1)],
          [(2, 2, 4), (2, 8), (0, 2, 1)],
          [(2, 2, 4), (2, 8), (2, 0, 1)],
      ]
      for rng_factory in [jtu.rand_default]))
  def testReshapeGrad(self, arg_shape, out_shape, permutation, dtype, rng_factory):
    rng = rng_factory(self.rng())
    operand = rng(arg_shape, dtype)
    reshape = lambda x: lax.reshape(x, out_shape, permutation)
    check_grads(reshape, (operand,), 2, ["fwd", "rev"], eps=1.)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_pads={}"
       .format(jtu.format_shape_dtype_string(shape, dtype), pads),
       "shape": shape, "dtype": dtype, "pads": pads, "rng_factory": jtu.rand_small}
      for shape in [(2, 3)]
      for dtype in jtu.float_dtypes
      for pads in [[(1, 2, 1), (0, 1, 0)], [(-1, 0, 0), (-1, 0, 2)]]))
  def testPadGrad(self, shape, dtype, pads, rng_factory):
    rng = rng_factory(self.rng())
    operand = rng(shape, dtype)
    pad = lambda operand: lax.pad(operand, onp.array(0, dtype), pads)
    check_grads(pad, (operand,), 2, ["fwd", "rev"], eps=1.)

    operand = rng(shape, dtype)
    padding_value = onp.array(0., dtype)
    pad = lambda operand, padding_value: lax.pad(operand, padding_value, pads)
    check_grads(pad, (operand, padding_value), 2, ["fwd", "rev"], eps=1.)

  def testReverseGrad(self):
    rev = lambda operand: lax.rev(operand, dimensions)

    dimensions = [0]
    check_grads(rev, (onp.array([3., 2., 1.]),), 2)

    dimensions = [0, 1]
    check_grads(rev, (onp.array([[6., 5., 4.], [3., 2., 1.]]),), 2,
                rtol={onp.float32: 3e-3})

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_predshape={}_argshapes={}".format(
          jtu.format_shape_dtype_string(pred_shape, onp.bool_),
          jtu.format_shape_dtype_string(arg_shape, dtype)),
       "pred_shape": pred_shape, "arg_shape": arg_shape, "dtype": dtype,
       "rng_factory": rng_factory}
      for arg_shape in [(), (3,), (2, 3)]
      for pred_shape in ([(), arg_shape] if arg_shape else [()])
      for dtype in jtu.float_dtypes
      for rng_factory in [jtu.rand_default]))
  def testSelectGrad(self, pred_shape, arg_shape, dtype, rng_factory):
    rng = rng_factory(self.rng())
    pred = rng(pred_shape, onp.bool_)
    on_true = rng(arg_shape, dtype)
    on_false = rng(arg_shape, dtype)
    select = lambda on_true, on_false: lax.select(pred, on_true, on_false)
    check_grads(select, (on_true, on_false), 2, ["fwd", "rev"], eps=1.)

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
      for dtype in jtu.float_dtypes
      for rng_factory in [jtu.rand_default]))
  def testSliceGrad(self, shape, dtype, starts, limits, strides, rng_factory):
    rng = rng_factory(self.rng())
    operand = rng(shape, dtype)
    slice = lambda x: lax.slice(x, starts, limits, strides)
    check_grads(slice, (operand,), 2, ["fwd", "rev"], eps=1.)

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
      for dtype in jtu.float_dtypes
      for rng_factory in [jtu.rand_default]))
  def testDynamicSliceGrad(self, shape, dtype, start_indices, size_indices,
                           rng_factory):
    rng = rng_factory(self.rng())
    operand = rng(shape, dtype)
    dynamic_slice = lambda x: lax.dynamic_slice(x, start_indices, size_indices)
    check_grads(dynamic_slice, (operand,), 2, ["fwd", "rev"], eps=1.)

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
      for dtype in jtu.float_dtypes
      for rng_factory in [jtu.rand_default]))
  def testDynamicUpdateSliceGrad(self, shape, dtype, start_indices,
                                 update_shape, rng_factory):
    rng = rng_factory(self.rng())
    operand = rng(shape, dtype)
    update = rng(update_shape, dtype)
    start_indices = onp.array(start_indices)

    dus = lambda x, y: lax.dynamic_update_slice(x, y, start_indices)
    check_grads(dus, (operand, update), 2, ["fwd", "rev"], eps=1.)

    dus = lambda x: lax.dynamic_update_slice(x, update, start_indices)
    check_grads(dus, (operand,), 2, ["fwd", "rev"], eps=1.)

    dus = lambda y: lax.dynamic_update_slice(operand, y, start_indices)
    check_grads(dus, (update,), 2, ["fwd", "rev"], eps=1.)

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
      for dtype in jtu.float_dtypes
      for rng_factory in [jtu.rand_default]))
  def testTransposeGrad(self, shape, dtype, perm, rng_factory):
    rng = rng_factory(self.rng())
    operand = rng(shape, dtype)
    transpose = lambda x: lax.transpose(x, perm)
    check_grads(transpose, (operand,), 2, ["fwd", "rev"], eps=1.)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_inshape={}_reducedims={}"
       .format(op.__name__, jtu.format_shape_dtype_string(shape, dtype), dims),
       "op": op, "init_val": init_val, "shape": shape, "dtype": dtype,
       "dims": dims, "rng_factory": rng_factory}
      for init_val, op, dtypes, rng_factory in [
          (0, lax.add, jtu.inexact_dtypes, jtu.rand_default),
          (-onp.inf, lax.max, grad_inexact_dtypes, jtu.rand_unique_int),
          (onp.inf, lax.min, grad_inexact_dtypes, jtu.rand_unique_int),
          (1, lax.mul, jtu.basic_float_dtypes, partial(jtu.rand_default, scale=1)),
      ]
      for dtype in dtypes
      for shape, dims in [
          [(3, 4, 5), ()],
          [(3, 4, 5), (0,)],
          [(3, 4, 5), (1, 2)],
          [(3, 4, 5), (0, 2)],
          [(3, 4, 5), (0, 1, 2)],
          [(3, 1), (1,)],
      ]))
  def testReduceGrad(self, op, init_val, shape, dtype, dims, rng_factory):
    rng = rng_factory(self.rng())
    if jtu.device_under_test() == "tpu" and op is lax.mul:
      raise SkipTest("unimplemented case")
    tol = {dtypes.bfloat16: 2e-1, onp.float16: 1e-1, onp.float32: 1e-1,
           onp.float64: 1e-3, onp.complex64: 1e-1}
    operand = rng(shape, dtype)
    init_val = onp.asarray(init_val, dtype=dtype)
    reduce = lambda operand: lax.reduce(operand, init_val, op, dims)
    eps = (1.0 if dtypes.finfo(dtype).bits == 16 and op is lax.add else
           1e-1 if dtype == dtypes.bfloat16 else
           1e-2 if dtypes.finfo(dtype).bits == 32 else None)
    check_grads(reduce, (operand,), 2, ["fwd", "rev"], tol, tol, eps)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_dtype={}_padding={}"
       .format(op.__name__, onp.dtype(dtype).name, padding),
       "op": op, "init_val": init_val, "dtype": dtype, "padding": padding,
       "rng_factory": rng_factory}
      for init_val, op, dtypes, rng_factory in [
          (0, lax.add, jtu.basic_float_dtypes, jtu.rand_small),
          (-onp.inf, lax.max, jtu.basic_float_dtypes, jtu.rand_unique_int),
          (onp.inf, lax.min, jtu.basic_float_dtypes, jtu.rand_unique_int),
      ]
      for dtype in dtypes
      for padding in ["VALID", "SAME"]))
  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  @jtu.ignore_warning(category=UserWarning,
                      message="Using reduced precision for gradient.*")
  def testReduceWindowGrad(self, op, init_val, dtype, padding, rng_factory):
    rng = rng_factory(self.rng())
    init_val = onp.asarray(init_val, dtype=dtype)

    # We need this conditional and the corresponding loop logic to be in the
    # test method, rather than at the parameterized test level, because it
    # depends on FLAGS for the device under test.
    # TODO(b/31565929): enable when fixed.
    if jtu.device_under_test() == "tpu" and op is not lax.add:
      all_configs = [((6, 5, 4, 3), (2, 2, 1, 1), (1, 2, 1, 1))]

      # TODO(b/73062247): need variadic reduce-window for better precision.
      gradient_order = 1
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
      gradient_order = 3

    def fun(operand):
      return lax.reduce_window(operand, init_val, op, dims, strides, padding)

    for shape, dims, strides in all_configs:
      operand = rng(shape, dtype)
      if op is lax.add:
        eps = 1.
        tol = None
      else:
        # this test can fail if there are duplicates in operand
        self.assertEqual(onp.unique(operand).size, operand.size,
                         msg="test requires operand elements to be unique.")
        eps = 1e-2
        tol = {onp.float16: 1e-1, onp.float32: 6e-2, onp.float64: 6e-2}
      check_grads(fun, (operand,), gradient_order, ["fwd", "rev"], tol, tol,
                  eps)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_shape={}_axis={}"
       .format(op.__name__, jtu.format_shape_dtype_string(shape, dtype), axis),
       "op": op, "shape": shape, "dtype": dtype,
       "axis": axis, "rng_factory": rng_factory}
      for op, types in [
          (lax.cumsum, [onp.float32, onp.float64]),
          (lax.cumprod, [onp.float32, onp.float64]),
      ]
      for dtype in types
      for shape in [[10], [3, 4, 5]]
      for axis in range(len(shape))
      for rng_factory in [
          jtu.rand_default if dtypes.issubdtype(dtype, onp.integer)
          else jtu.rand_small]))
  def testCumulativeReduceGrad(self, op, shape, dtype, axis, rng_factory):
    rng = rng_factory(self.rng())
    check_grads(partial(op, axis=axis), (rng(shape, dtype),), order=2)


  # TODO(b/205052657): enable more tests when supported
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "rng_factory": rng_factory, "shape": shape, "dtype": dtype, "axis": axis}
      for dtype in [onp.float32]
      for shape in [(5,), (5, 7)]
      for axis in [len(shape) - 1]
      for rng_factory in [jtu.rand_default]))
  def testSortGrad(self, shape, dtype, axis, rng_factory):
    rng = rng_factory(self.rng())
    operand = rng(shape, dtype)
    sort = lambda x: lax.sort(x, dimension=axis)
    check_grads(sort, (operand,), 2, ["fwd", "rev"], eps=1e-2)

  # TODO(b/205052657): enable more tests when supported
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_keyshape={}_valshape={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, key_dtype),
          jtu.format_shape_dtype_string(shape, val_dtype),
          axis),
       "rng_factory": rng_factory, "shape": shape,
       "key_dtype": key_dtype, "val_dtype": val_dtype, "axis": axis}
      for key_dtype in [onp.float32]
      for val_dtype in [onp.float32]
      for shape in [(3,), (5, 3)]
      for axis in [len(shape) - 1]
      for rng_factory in [jtu.rand_default]))
  def testSortKeyValGrad(self, shape, key_dtype, val_dtype, axis, rng_factory):
    rng = rng_factory(self.rng())
    # This test relies on the property that wherever keys are tied, values are
    # too, since we don't guarantee the same ordering of values with equal keys.
    # To avoid that case, we generate unique keys (globally in the key array).
    def args_maker():
      flat_keys = onp.arange(onp.prod(shape, dtype=int), dtype=key_dtype)
      keys = self.rng().permutation(flat_keys).reshape(shape)
      values = rng(shape, val_dtype)
      return keys, values
    keys, values = args_maker()

    fun = lambda keys, values: lax.sort_key_val(keys, values, axis)
    check_grads(fun, (keys, values), 2, ["fwd", "rev"], 1e-2, 1e-2, 1e-2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_k={}".format(
          jtu.format_shape_dtype_string(shape, dtype), k),
       "rng_factory": rng_factory, "shape": shape, "dtype": dtype, "k": k}
      for dtype in [onp.float32,]
      for shape in [(4,), (5, 5), (2, 1, 4)]
      for k in [1, 3]
      for rng_factory in [jtu.rand_default]))
  def testTopKGrad(self, shape, dtype, k, rng_factory):
    flat_values = onp.arange(onp.prod(shape, dtype=int), dtype=dtype)
    values = self.rng().permutation(flat_values).reshape(shape)
    fun = lambda vs: lax.top_k(vs, k=k)[0]
    check_grads(fun, (values,), 2, ["fwd", "rev"], eps=1e-2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_idxs={}_axes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), idxs, axes),
       "shape": shape, "dtype": dtype, "idxs": idxs, "axes": axes,
       "rng_factory": rng_factory}
      for dtype in jtu.float_dtypes
      for shape, idxs, axes in [
          [(3, 4, 5), (onp.array([0, 2, 1]),), (0,)],
          [(3, 4, 5), (onp.array([-1, -2]),), (0,)],
          [(3, 4, 5), (onp.array([0, 2]), onp.array([1, 3])), (0, 1)],
          [(3, 4, 5), (onp.array([0, 2]), onp.array([1, 3])), (0, 2)],
      ]
      for rng_factory in [jtu.rand_default]))
  def testIndexTakeGrad(self, shape, dtype, idxs, axes, rng_factory):
    rng = rng_factory(self.rng())
    src = rng(shape, dtype)
    index_take = lambda src: lax.index_take(src, idxs, axes)
    check_grads(index_take, (src,), 2, ["fwd", "rev"], eps=1.)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_idxs={}_dnums={}_slice_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), idxs, dnums,
          slice_sizes),
       "shape": shape, "dtype": dtype, "idxs": idxs, "dnums": dnums,
       "slice_sizes": slice_sizes, "rng_factory": rng_factory,
       "rng_idx_factory": rng_idx_factory}
      for dtype in jtu.basic_float_dtypes
      for shape, idxs, dnums, slice_sizes, max_idx in [
          ((5,), onp.array([[0], [2]]), lax.GatherDimensionNumbers(
            offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)),
            (1,), 5),
          ((10,), onp.array([[0], [0], [0]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)),
            (2,), 9),
          ((10, 5,), onp.array([[0], [2], [1]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)),
            (1, 3), 3),
      ]
      for rng_idx_factory in [partial(jtu.rand_int, high=max_idx)]
      for rng_factory in [jtu.rand_default]))
  def testGatherGrad(self, shape, dtype, idxs, dnums, slice_sizes, rng_factory,
                     rng_idx_factory):
    rng = rng_factory(self.rng())
    rng_idx = rng_idx_factory(self.rng())
    idxs = rng_idx(idxs.shape, idxs.dtype)
    gather = lambda x: lax.gather(x, idxs, dimension_numbers=dnums,
                                  slice_sizes=slice_sizes)
    x = rng(shape, dtype)
    check_grads(gather, (x,), 2, ["fwd", "rev"], 1e-2, 1e-2, 1.)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_idxs={}_update={}_dnums={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          idxs, update_shape, dnums),
       "arg_shape": arg_shape, "dtype": dtype, "idxs": idxs,
       "update_shape": update_shape, "dnums": dnums, "rng_factory": rng_factory,
       "rng_idx_factory": rng_idx_factory}
      for dtype in jtu.basic_float_dtypes
      for arg_shape, idxs, update_shape, dnums, max_idx in [
          ((5,), onp.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
            update_window_dims=(), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,)), 4),
          ((10,), onp.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(),
            scatter_dims_to_operand_dims=(0,)), 9),
          ((10, 5,), onp.array([[0], [2], [1]]), (3, 3), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,)), 3),
      ]
      for rng_idx_factory in [partial(jtu.rand_int, high=max_idx)]
      for rng_factory in [jtu.rand_default]))
  def testScatterAddGrad(self, arg_shape, dtype, idxs, update_shape, dnums,
                         rng_factory, rng_idx_factory):
    rng = rng_factory(self.rng())
    rng_idx = rng_idx_factory(self.rng())
    idxs = rng_idx(idxs.shape, idxs.dtype)
    scatter_add = lambda x, y: lax.scatter_add(x, idxs, y,
                                               dimension_numbers=dnums)
    x = rng(arg_shape, dtype)
    y = rng(update_shape, dtype)
    check_grads(scatter_add, (x, y), 2, ["fwd", "rev"], 1e-2, 1e-2, 1.)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_idxs={}_update={}_dnums={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          idxs, update_shape, dnums),
       "arg_shape": arg_shape, "dtype": dtype, "idxs": idxs,
       "update_shape": update_shape, "dnums": dnums, "rng_factory": rng_factory,
       "rng_idx_factory": rng_idx_factory}
      for dtype in jtu.basic_float_dtypes
      for arg_shape, idxs, update_shape, dnums, max_idx in [
          ((5,), onp.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
            update_window_dims=(), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,)), 4),
          ((10,), onp.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(),
            scatter_dims_to_operand_dims=(0,)), 9),
          ((10, 5,), onp.array([[0], [2], [1]]), (3, 3), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,)), 3),
      ]
      # Scatters with conflicting indices are not deterministic on GPU, so we
      # use indices that do not collide.
      for rng_idx_factory in [partial(jtu.rand_unique_int, high=max_idx)]
      for rng_factory in [jtu.rand_default]))
  def testScatterGrad(self, arg_shape, dtype, idxs, update_shape, dnums,
                      rng_factory, rng_idx_factory):
    rng = rng_factory(self.rng())
    rng_idx = rng_idx_factory(self.rng())
    idxs = rng_idx(idxs.shape, idxs.dtype)
    scatter = lambda x, y: lax.scatter(x, idxs, y, dimension_numbers=dnums)
    x = rng(arg_shape, dtype)
    y = rng(update_shape, dtype)
    check_grads(scatter, (x, y), 2, ["fwd", "rev"], 1e-2, 1e-2, 1.)

  def testScatterGradSymbolicZeroUpdate(self):
    # https://github.com/google/jax/issues/1901
    def f(x):
      n = x.shape[0]
      y = onp.arange(n, dtype=x.dtype)
      return jax.ops.index_update(x, onp.diag_indices(n), y)
    rng = jtu.rand_default(self.rng())
    check_grads(f, (rng((5, 5), onp.float32),), 2, ["fwd", "rev"], 1e-2, 1e-2,
                1.)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_idxs={}_update={}_dnums={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          idxs, update_shape, dnums),
       "arg_shape": arg_shape, "dtype": dtype, "idxs": idxs,
       "update_shape": update_shape, "dnums": dnums,
       "rng_factory": rng_factory, "rng_idx_factory": rng_idx_factory}
      for dtype in jtu.basic_float_dtypes
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
    idxs = rng_idx(idxs.shape, idxs.dtype)
    scatter_max = lambda x, y: lax.scatter_max(x, idxs, y, dnums)
    x = rng(arg_shape, dtype)
    y = rng(update_shape, dtype)
    check_grads(scatter_max, (x, y), 2, ["fwd", "rev"], 1e-2, 1e-2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_idxs={}_update={}_dnums={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          idxs, update_shape, dnums),
       "arg_shape": arg_shape, "dtype": dtype, "idxs": idxs,
       "update_shape": update_shape, "dnums": dnums,
       "rng_factory": rng_factory, "rng_idx_factory": rng_idx_factory}
      for dtype in jtu.basic_float_dtypes
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
    idxs = rng_idx(idxs.shape, idxs.dtype)
    scatter_min = lambda x, y: lax.scatter_min(x, idxs, y, dnums)
    x = rng(arg_shape, dtype)
    y = rng(update_shape, dtype)
    check_grads(scatter_min, (x, y), 2, ["fwd", "rev"], 1e-2, 1e-2)

  def testStopGradient(self):
    def f(x):
      return lax.sin(x) * lax.cos(lax.stop_gradient(x))

    def f2(x, y):
      return lax.sin(x) * lax.cos(y)

    x = 3.14
    ans = api.grad(f)(x)
    expected = api.grad(f2)(x, x)
    self.assertAllClose(ans, expected)

    ans = api.grad(api.grad(f))(x)
    expected = api.grad(api.grad(f2))(x, x)
    self.assertAllClose(ans, expected)

    ans = api.grad(lambda x: lax.stop_gradient({'foo':x})['foo'])(3.)
    expected = onp.array(0.0)
    self.assertAllClose(ans, expected, check_dtypes=False)

    with core.skipping_checks():
      with self.assertRaises(TypeError):
        lax.stop_gradient(lambda x: x)

  # TODO(mattjj): make this a more systematic test
  def testRemainder(self):
    rng = onp.random.RandomState(0)
    x = rng.uniform(-0.9, 9, size=(3, 4))
    y = rng.uniform(0.7, 1.9, size=(3, 1))
    assert not set(onp.unique(x)) & set(onp.unique(y))
    tol = 1e-1 if num_float_bits(onp.float64) == 32 else 1e-3
    check_grads(lax.rem, (x, y), 2, ["fwd", "rev"], tol, tol)

    rng = onp.random.RandomState(0)
    x = rng.uniform(-0.9, 9, size=(1, 4))
    y = rng.uniform(0.7, 1.9, size=(3, 4))
    assert not set(onp.unique(x)) & set(onp.unique(y))
    tol = 1e-1 if num_float_bits(onp.float64) == 32 else 1e-3
    check_grads(lax.rem, (x, y), 2, ["fwd", "rev"], tol, tol)

  def testHigherOrderGradientOfReciprocal(self):
    # Regression test for https://github.com/google/jax/issues/3136
    def inv(x):
      # N.B.: intentionally written as 1/x, not x ** -1 or reciprocal(x)
      return 1 / x
    grad_fn = jax.grad(jax.grad(jax.grad(jax.grad(jax.grad(jax.grad(inv))))))
    self.assertAllClose(onp.float32(0.0439453125), grad_fn(onp.float32(4.)))


if __name__ == '__main__':
  absltest.main()
