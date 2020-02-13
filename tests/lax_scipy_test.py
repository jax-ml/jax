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
import functools
from functools import partial
import itertools
import unittest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp
import scipy.special as osp_special
import scipy.stats as osp_stats

from jax import api
from jax import lib
from jax import test_util as jtu
from jax.scipy import special as lsp_special
from jax.scipy import stats as lsp_stats

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS

all_shapes = [(), (4,), (3, 4), (3, 1), (1, 4), (2, 1, 4)]

float_dtypes = [onp.float32, onp.float64]
complex_dtypes = [onp.complex64]
int_dtypes = [onp.int32, onp.int64]
bool_dtypes = [onp.bool_]
default_dtypes = float_dtypes + int_dtypes
numeric_dtypes = float_dtypes + complex_dtypes + int_dtypes


OpRecord = collections.namedtuple(
    "OpRecord",
    ["name", "nargs", "dtypes", "rng_factory", "test_autodiff", "test_name"])


def op_record(name, nargs, dtypes, rng_factory, test_grad, test_name=None):
  test_name = test_name or name
  return OpRecord(name, nargs, dtypes, rng_factory, test_grad, test_name)

JAX_SPECIAL_FUNCTION_RECORDS = [
    # TODO: digamma has no JVP implemented.
    op_record("betaln", 2, float_dtypes, jtu.rand_positive, False),
    op_record("betainc", 3, float_dtypes, jtu.rand_positive, False),
    op_record("digamma", 1, float_dtypes, jtu.rand_positive, False),
    op_record("gammainc", 2, float_dtypes, jtu.rand_positive, False),
    op_record("gammaincc", 2, float_dtypes, jtu.rand_positive, False),
    op_record("erf", 1, float_dtypes, jtu.rand_small_positive, True),
    op_record("erfc", 1, float_dtypes, jtu.rand_small_positive, True),
    op_record("erfinv", 1, float_dtypes, jtu.rand_small_positive, True),
    op_record("expit", 1, float_dtypes, jtu.rand_small_positive, True),
    # TODO: gammaln has slightly high error.
    op_record("gammaln", 1, float_dtypes, jtu.rand_positive, False),
    op_record("logit", 1, float_dtypes, jtu.rand_small_positive, False),
    op_record("log_ndtr", 1, float_dtypes, jtu.rand_default, True),
    op_record("ndtri", 1, float_dtypes, partial(jtu.rand_uniform, 0.05, 0.95),
              True),
    op_record("ndtr", 1, float_dtypes, jtu.rand_default, True),
    # TODO(phawkins): gradient of entr yields NaNs.
    op_record("entr", 1, float_dtypes, jtu.rand_default, False),
]

CombosWithReplacement = itertools.combinations_with_replacement


class LaxBackedScipyTests(jtu.JaxTestCase):
  """Tests for LAX-backed Scipy implementation."""

  def _GetArgsMaker(self, rng, shapes, dtypes):
    return lambda: [rng(shape, dtype) for shape, dtype in zip(shapes, dtypes)]

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_axis={}_keepdims={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, keepdims),
       # TODO(b/133842870): re-enable when exp(nan) returns NaN on CPU.
       "rng_factory": jtu.rand_some_inf_and_nan
                      if jtu.device_under_test() != "cpu"
                      else jtu.rand_default,
       "shape": shape, "dtype": dtype,
       "axis": axis, "keepdims": keepdims}
      for shape in all_shapes for dtype in float_dtypes
      for axis in range(-len(shape), len(shape))
      for keepdims in [False, True]))
  @jtu.skip_on_flag("jax_xla_backend", "xrt")
  def testLogSumExp(self, rng_factory, shape, dtype, axis, keepdims):
    rng = rng_factory()
    # TODO(mattjj): test autodiff
    def scipy_fun(array_to_reduce):
      return osp_special.logsumexp(array_to_reduce, axis, keepdims=keepdims)

    def lax_fun(array_to_reduce):
      return lsp_special.logsumexp(array_to_reduce, axis, keepdims=keepdims)

    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(itertools.chain.from_iterable(
    jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(
            rec.test_name, shapes, dtypes),
         "rng_factory": rec.rng_factory, "shapes": shapes, "dtypes": dtypes,
         "test_autodiff": rec.test_autodiff,
         "scipy_op": getattr(osp_special, rec.name),
         "lax_op": getattr(lsp_special, rec.name)}
        for shapes in CombosWithReplacement(all_shapes, rec.nargs)
        for dtypes in CombosWithReplacement(rec.dtypes, rec.nargs))
      for rec in JAX_SPECIAL_FUNCTION_RECORDS))
  def testScipySpecialFun(self, scipy_op, lax_op, rng_factory, shapes, dtypes,
                          test_autodiff):
    rng = rng_factory()
    args_maker = self._GetArgsMaker(rng, shapes, dtypes)
    args = args_maker()
    self.assertAllClose(scipy_op(*args), lax_op(*args), atol=1e-3, rtol=1e-3,
                        check_dtypes=False)
    self._CompileAndCheck(lax_op, args_maker, check_dtypes=True, rtol=1e-5)

    if test_autodiff:
      jtu.check_grads(lax_op, args, order=1, atol=1e-3, rtol=3e-3, eps=1e-3)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_d={}".format(
          jtu.format_shape_dtype_string(shape, dtype), d),
       "rng_factory": jtu.rand_positive, "shape": shape, "dtype": dtype,
       "d": d}
      for shape in all_shapes
      for dtype in float_dtypes
      for d in [1, 2, 5]))
  def testMultigammaln(self, rng_factory, shape, dtype, d):
    def scipy_fun(a):
      return osp_special.multigammaln(a, d)

    def lax_fun(a):
      return lsp_special.multigammaln(a, d)

    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype) + (d - 1) / 2.]
    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=True,
                            tol={onp.float32: 1e-3, onp.float64: 1e-14})
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)

  def testIssue980(self):
    x = onp.full((4,), -1e20, dtype=onp.float32)
    self.assertAllClose(onp.zeros((4,), dtype=onp.float32),
                        lsp_special.expit(x), check_dtypes=True)


if __name__ == "__main__":
  absltest.main()
