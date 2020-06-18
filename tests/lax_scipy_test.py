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

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp
import scipy.special as osp_special

from jax import api
from jax import test_util as jtu
from jax.scipy import special as lsp_special

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
    ["name", "nargs", "dtypes", "rng_factory", "test_autodiff", "nondiff_argnums", "test_name"])


def op_record(name, nargs, dtypes, rng_factory, test_grad, nondiff_argnums=(), test_name=None):
  test_name = test_name or name
  nondiff_argnums = tuple(sorted(set(nondiff_argnums)))
  return OpRecord(name, nargs, dtypes, rng_factory, test_grad, nondiff_argnums, test_name)

JAX_SPECIAL_FUNCTION_RECORDS = [
    op_record("betaln", 2, float_dtypes, jtu.rand_positive, False),
    op_record("betainc", 3, float_dtypes, jtu.rand_positive, False),
    op_record("digamma", 1, float_dtypes, jtu.rand_positive, True),
    op_record("gammainc", 2, float_dtypes, jtu.rand_positive, True),
    op_record("gammaincc", 2, float_dtypes, jtu.rand_positive, True),
    op_record("erf", 1, float_dtypes, jtu.rand_small_positive, True),
    op_record("erfc", 1, float_dtypes, jtu.rand_small_positive, True),
    op_record("erfinv", 1, float_dtypes, jtu.rand_small_positive, True),
    op_record("expit", 1, float_dtypes, jtu.rand_small_positive, True),
    # TODO: gammaln has slightly high error.
    op_record("gammaln", 1, float_dtypes, jtu.rand_positive, False),
    op_record("logit", 1, float_dtypes, jtu.rand_uniform, True),
    op_record("log_ndtr", 1, float_dtypes, jtu.rand_default, True),
    op_record("ndtri", 1, float_dtypes, partial(jtu.rand_uniform, low=0.05,
                                                high=0.95),
              True),
    op_record("ndtr", 1, float_dtypes, jtu.rand_default, True),
    # TODO(phawkins): gradient of entr yields NaNs.
    op_record("entr", 1, float_dtypes, jtu.rand_default, False),
    op_record("polygamma", 2, (int_dtypes, float_dtypes), jtu.rand_positive, True, (0,)),
    op_record("xlogy", 2, float_dtypes, jtu.rand_default, True),
    op_record("xlog1py", 2, float_dtypes, jtu.rand_default, True),
    # TODO: enable gradient test for zeta by restricting the domain of
    # of inputs to some reasonable intervals
    op_record("zeta", 2, float_dtypes, jtu.rand_positive, False),
]

CombosWithReplacement = itertools.combinations_with_replacement


class LaxBackedScipyTests(jtu.JaxTestCase):
  """Tests for LAX-backed Scipy implementation."""

  def _GetArgsMaker(self, rng, shapes, dtypes):
    return lambda: [rng(shape, dtype) for shape, dtype in zip(shapes, dtypes)]

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
          "_inshape={}_axis={}_keepdims={}_return_sign={}_use_b_{}".format(
             jtu.format_shape_dtype_string(shape, dtype),
             axis, keepdims,return_sign, use_b),
       # TODO(b/133842870): re-enable when exp(nan) returns NaN on CPU.
       "rng_factory": jtu.rand_some_inf_and_nan
                      if jtu.device_under_test() != "cpu"
                      else jtu.rand_default,
       "shape": shape, "dtype": dtype,
       "axis": axis, "keepdims": keepdims,
       "return_sign": return_sign, "use_b": use_b}
      for shape in all_shapes for dtype in float_dtypes
      for axis in range(-len(shape), len(shape))
      for keepdims in [False, True]
      for return_sign in [False, True]
      for use_b in [False, True]))
  @jtu.skip_on_flag("jax_xla_backend", "xrt")
  def testLogSumExp(self, rng_factory, shape, dtype, axis, keepdims,
                    return_sign, use_b):
    rng = rng_factory(self.rng())
    # TODO(mattjj): test autodiff
    if use_b:
      def scipy_fun(array_to_reduce, scale_array):
        return osp_special.logsumexp(array_to_reduce, axis, keepdims=keepdims,
                                     return_sign=return_sign, b=scale_array)

      def lax_fun(array_to_reduce, scale_array):
        return lsp_special.logsumexp(array_to_reduce, axis, keepdims=keepdims,
                                     return_sign=return_sign, b=scale_array)

      args_maker = lambda: [rng(shape, dtype), rng(shape, dtype)]
    else:
      def scipy_fun(array_to_reduce):
        return osp_special.logsumexp(array_to_reduce, axis, keepdims=keepdims,
                                     return_sign=return_sign)

      def lax_fun(array_to_reduce):
        return lsp_special.logsumexp(array_to_reduce, axis, keepdims=keepdims,
                                     return_sign=return_sign)

      args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker)
    self._CompileAndCheck(lax_fun, args_maker)

  @parameterized.named_parameters(itertools.chain.from_iterable(
    jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(
            rec.test_name, shapes, dtypes),
         "rng_factory": rec.rng_factory, "shapes": shapes, "dtypes": dtypes,
         "test_autodiff": rec.test_autodiff,
         "nondiff_argnums": rec.nondiff_argnums,
         "scipy_op": getattr(osp_special, rec.name),
         "lax_op": getattr(lsp_special, rec.name)}
        for shapes in CombosWithReplacement(all_shapes, rec.nargs)
        for dtypes in (CombosWithReplacement(rec.dtypes, rec.nargs)
          if isinstance(rec.dtypes, list) else itertools.product(*rec.dtypes)))
      for rec in JAX_SPECIAL_FUNCTION_RECORDS))
  def testScipySpecialFun(self, scipy_op, lax_op, rng_factory, shapes, dtypes,
                          test_autodiff, nondiff_argnums):
    rng = rng_factory(self.rng())
    args_maker = self._GetArgsMaker(rng, shapes, dtypes)
    args = args_maker()
    self.assertAllClose(scipy_op(*args), lax_op(*args), atol=1e-3, rtol=1e-3,
                        check_dtypes=False)
    self._CompileAndCheck(lax_op, args_maker, rtol=1e-4)

    if test_autodiff:
      def partial_lax_op(*vals):
        list_args = list(vals)
        for i in nondiff_argnums:
          list_args.insert(i, args[i])
        return lax_op(*list_args)

      assert list(nondiff_argnums) == sorted(set(nondiff_argnums))
      diff_args = [x for i, x in enumerate(args) if i not in nondiff_argnums]
      jtu.check_grads(partial_lax_op, diff_args, order=1,
                      atol=jtu.if_device_under_test("tpu", .1, 1e-3),
                      rtol=.1, eps=1e-3)

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

    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype) + (d - 1) / 2.]
    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker,
                            tol={onp.float32: 1e-3, onp.float64: 1e-14})
    self._CompileAndCheck(lax_fun, args_maker)

  def testIssue980(self):
    x = onp.full((4,), -1e20, dtype=onp.float32)
    self.assertAllClose(onp.zeros((4,), dtype=onp.float32),
                        lsp_special.expit(x))

  def testXlogyShouldReturnZero(self):
    self.assertAllClose(lsp_special.xlogy(0., 0.), 0., check_dtypes=False)

  def testGradOfXlogyAtZero(self):
    partial_xlogy = functools.partial(lsp_special.xlogy, 0.)
    self.assertAllClose(api.grad(partial_xlogy)(0.), 0., check_dtypes=False)

  def testXlog1pyShouldReturnZero(self):
    self.assertAllClose(lsp_special.xlog1py(0., -1.), 0., check_dtypes=False)

  def testGradOfXlog1pyAtZero(self):
    partial_xlog1py = functools.partial(lsp_special.xlog1py, 0.)
    self.assertAllClose(api.grad(partial_xlog1py)(-1.), 0., check_dtypes=False)


if __name__ == "__main__":
  absltest.main()
