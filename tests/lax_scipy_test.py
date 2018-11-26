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

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp
import scipy.misc as osp_misc
import scipy.special as osp_special
import scipy.stats as osp_stats

from jax import api
from jax import test_util as jtu
from jax.scipy import misc as lsp_misc
from jax.scipy import special as lsp_special
from jax.scipy import stats as lsp_stats

FLAGS = flags.FLAGS

all_shapes = [(), (4,), (3, 4), (3, 1), (1, 4), (2, 1, 4)]

float_dtypes = [onp.float32, onp.float64]
complex_dtypes = [onp.complex64]
int_dtypes = [onp.int32, onp.int64]
bool_dtypes = [onp.bool_]
default_dtypes = float_dtypes + int_dtypes
numeric_dtypes = float_dtypes + complex_dtypes + int_dtypes


OpRecord = collections.namedtuple("OpRecord", ["name", "nargs", "dtypes", "rng",
                                               "diff_modes", "test_name"])


def op_record(name, nargs, dtypes, rng, diff_modes, test_name=None):
  test_name = test_name or name
  return OpRecord(name, nargs, dtypes, rng, diff_modes, test_name)

JAX_SPECIAL_FUNCTION_RECORDS = [
    op_record("gammaln", 1, float_dtypes, jtu.rand_positive(), ["rev"]),
    op_record("digamma", 1, float_dtypes, jtu.rand_positive(), []),
    op_record("erf", 1, float_dtypes, jtu.rand_small_positive(), ["rev"]),
    op_record("erfc", 1, float_dtypes, jtu.rand_small_positive(), ["rev"]),
    op_record("erfinv", 1, float_dtypes, jtu.rand_small_positive(), ["rev"]),
]

CombosWithReplacement = itertools.combinations_with_replacement


class LaxBackedScipyTests(jtu.JaxTestCase):
  """Tests for LAX-backed Scipy implementation."""

  def _GetArgsMaker(self, rng, shapes, dtypes):
    return lambda: [rng(shape, dtype) for shape, dtype in zip(shapes, dtypes)]

  @parameterized.named_parameters(
      {"testcase_name": "_inshape={}_axis={}_keepdims={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, keepdims),
       "rng": jtu.rand_default(), "shape": shape, "dtype": dtype,
       "axis": axis, "keepdims": keepdims}
      for shape in all_shapes for dtype in float_dtypes
      for axis in range(-len(shape), len(shape))
      for keepdims in [False, True])
  @jtu.skip_on_flag("jax_xla_backend", "xrt")
  def testLogSumExp(self, rng, shape, dtype, axis, keepdims):
    # TODO(mattjj): test autodiff
    def scipy_fun(array_to_reduce):
      return osp_misc.logsumexp(array_to_reduce, axis, keepdims=keepdims)

    def lax_fun(array_to_reduce):
      return lsp_misc.logsumexp(array_to_reduce, axis, keepdims=keepdims)

    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(
      {"testcase_name": jtu.format_test_name_suffix(
          rec.test_name, shapes, dtypes),
       "rng": rec.rng, "shapes": shapes, "dtypes": dtypes,
       "modes": rec.diff_modes,
       "scipy_op": getattr(osp_special, rec.name),
       "lax_op": getattr(lsp_special, rec.name)}
      for rec in JAX_SPECIAL_FUNCTION_RECORDS
      for shapes in CombosWithReplacement(all_shapes, rec.nargs)
      for dtypes in CombosWithReplacement(rec.dtypes, rec.nargs))
  def testScipySpecialFun(self, scipy_op, lax_op, rng, shapes, dtypes, modes):
    # TODO(mattjj): unskip this test combination when real() on tpu is improved
    # TODO(mattjj): test autodiff
    if (FLAGS.jax_test_dut and FLAGS.jax_test_dut.startswith("tpu")
        and not shapes[0]):
      return absltest.unittest.skip("real() on scalar not supported on tpu")

    args_maker = self._GetArgsMaker(rng, shapes, dtypes)
    args = args_maker()
    self.assertAllClose(scipy_op(*args), lax_op(*args), atol=1e-3, rtol=1e-3,
                        check_dtypes=False)
    self._CompileAndCheck(lax_op, args_maker, check_dtypes=True)

  @parameterized.named_parameters(
      {"testcase_name": jtu.format_test_name_suffix(
          "", shapes, dtypes),
       "rng": rng, "shapes": shapes, "dtypes": dtypes}
      for shapes in CombosWithReplacement(all_shapes, 3)
      for dtypes in CombosWithReplacement(default_dtypes, 3)
      for rng in [jtu.rand_default()])
  @jtu.skip_on_flag("jax_xla_backend", "xrt")
  def testNormLogPdfThreeArgs(self, rng, shapes, dtypes):
    # TODO(mattjj): test autodiff
    scipy_fun = osp_stats.norm.logpdf
    lax_fun = lsp_stats.norm.logpdf
    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      scale = 0.5 + onp.abs(scale)
      return [x, loc, scale]
    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(
      {"testcase_name": jtu.format_test_name_suffix(
          "", shapes, dtypes),
       "rng": rng, "shapes": shapes, "dtypes": dtypes}
      for shapes in CombosWithReplacement(all_shapes, 2)
      for dtypes in CombosWithReplacement(default_dtypes, 2)
      for rng in [jtu.rand_default()])
  def testNormLogPdfTwoArgs(self, rng, shapes, dtypes):
    # TODO(mattjj): test autodiff
    scale = 0.5
    scipy_fun = functools.partial(osp_stats.norm.logpdf, scale=scale)
    lax_fun = functools.partial(lsp_stats.norm.logpdf, scale=scale)
    def args_maker():
      return list(map(rng, shapes, dtypes))
    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)


if __name__ == "__main__":
  absltest.main()
