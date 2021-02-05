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


import itertools

from absl.testing import absltest, parameterized

import numpy as np
import scipy.stats as osp_stats

from jax import api
from jax import test_util as jtu
from jax.scipy import stats as lsp_stats
from jax.scipy.special import expit

from jax.config import config
config.parse_flags_with_absl()

all_shapes = [(), (4,), (3, 4), (3, 1), (1, 4), (2, 1, 4)]


def genNamedParametersNArgs(n):
    return parameterized.named_parameters(
        jtu.cases_from_list(
          {"testcase_name": jtu.format_test_name_suffix("", shapes, dtypes),
            "shapes": shapes, "dtypes": dtypes}
          for shapes in itertools.combinations_with_replacement(all_shapes, n)
          for dtypes in itertools.combinations_with_replacement(jtu.dtypes.floating, n)))


class LaxBackedScipyStatsTests(jtu.JaxTestCase):
  """Tests for LAX-backed scipy.stats implementations"""

  @genNamedParametersNArgs(3)
  def testPoissonLogPmf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.poisson.logpmf
    lax_fun = lsp_stats.poisson.logpmf

    def args_maker():
      k, mu, loc = map(rng, shapes, dtypes)
      k = np.floor(k)
      # clipping to ensure that rate parameter is strictly positive
      mu = np.clip(np.abs(mu), a_min=0.1, a_max=None)
      loc = np.floor(loc)
      return [k, mu, loc]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-3)
    self._CompileAndCheck(lax_fun, args_maker, rtol={np.float64: 1e-14})

  @genNamedParametersNArgs(3)
  def testPoissonPmf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.poisson.pmf
    lax_fun = lsp_stats.poisson.pmf

    def args_maker():
      k, mu, loc = map(rng, shapes, dtypes)
      k = np.floor(k)
      # clipping to ensure that rate parameter is strictly positive
      mu = np.clip(np.abs(mu), a_min=0.1, a_max=None)
      loc = np.floor(loc)
      return [k, mu, loc]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-3)
    self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(3)
  def testPoissonCdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.poisson.cdf
    lax_fun = lsp_stats.poisson.cdf

    def args_maker():
      k, mu, loc = map(rng, shapes, dtypes)
      # clipping to ensure that rate parameter is strictly positive
      mu = np.clip(np.abs(mu), a_min=0.1, a_max=None)
      return [k, mu, loc]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-3)
    self._CompileAndCheck(lax_fun, args_maker)


  @genNamedParametersNArgs(3)
  def testBernoulliLogPmf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.bernoulli.logpmf
    lax_fun = lsp_stats.bernoulli.logpmf

    def args_maker():
      x, logit, loc = map(rng, shapes, dtypes)
      x = np.floor(x)
      p = expit(logit)
      loc = np.floor(loc)
      return [x, p, loc]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(3)
  def testGeomLogPmf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.geom.logpmf
    lax_fun = lsp_stats.geom.logpmf

    def args_maker():
      x, logit, loc = map(rng, shapes, dtypes)
      x = np.floor(x)
      p = expit(logit)
      loc = np.floor(loc)
      return [x, p, loc]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(5)
  def testBetaLogPdf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.beta.logpdf
    lax_fun = lsp_stats.beta.logpdf

    def args_maker():
      x, a, b, loc, scale = map(rng, shapes, dtypes)
      return [x, a, b, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-3)
    self._CompileAndCheck(lax_fun, args_maker,
                          rtol={np.float32: 2e-3, np.float64: 1e-4})

  @genNamedParametersNArgs(3)
  def testCauchyLogPdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.cauchy.logpdf
    lax_fun = lsp_stats.cauchy.logpdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None)
      return [x, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(2)
  def testDirichletLogPdf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.cauchy.logpdf
    lax_fun = lsp_stats.cauchy.logpdf
    dim = 4
    shapes = (shapes[0] + (dim,), shapes[1] + (dim,))

    def args_maker():
      x, alpha = map(rng, shapes, dtypes)
      x = x / np.sum(x, axis=-1, keepdims=True)
      return [x, alpha]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(3)
  def testExponLogPdf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.expon.logpdf
    lax_fun = lsp_stats.expon.logpdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      return [x, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(4)
  def testGammaLogPdf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.gamma.logpdf
    lax_fun = lsp_stats.gamma.logpdf

    def args_maker():
      x, a, loc, scale = map(rng, shapes, dtypes)
      return [x, a, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=5e-4)
    self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(3)
  def testLaplaceLogPdf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.laplace.logpdf
    lax_fun = lsp_stats.laplace.logpdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = np.clip(scale, a_min=0.1, a_max=None)
      return [x, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(3)
  def testLaplaceCdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.laplace.cdf
    lax_fun = lsp_stats.laplace.cdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # ensure that scale is not too low
      scale = np.clip(scale, a_min=0.1, a_max=None)
      return [x, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol={np.float32: 1e-5, np.float64: 1e-6})
    self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(1)
  def testLogisticCdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.logistic.cdf
    lax_fun = lsp_stats.logistic.cdf

    def args_maker():
      return list(map(rng, shapes, dtypes))

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-6)
    self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(1)
  def testLogisticLogpdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.logistic.logpdf
    lax_fun = lsp_stats.logistic.logpdf

    def args_maker():
      return list(map(rng, shapes, dtypes))

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-3)
    self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(1)
  def testLogisticPpf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.logistic.ppf
    lax_fun = lsp_stats.logistic.ppf

    def args_maker():
      return list(map(rng, shapes, dtypes))

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(1)
  def testLogisticSf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.logistic.sf
    lax_fun = lsp_stats.logistic.sf

    def args_maker():
      return list(map(rng, shapes, dtypes))

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-6)
    self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(3)
  def testNormLogPdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.norm.logpdf
    lax_fun = lsp_stats.norm.logpdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None)
      return [x, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-3)
    self._CompileAndCheck(lax_fun, args_maker)


  @genNamedParametersNArgs(3)
  def testNormLogCdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.norm.logcdf
    lax_fun = lsp_stats.norm.logcdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None)
      return [x, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(lax_fun, args_maker)


  @genNamedParametersNArgs(3)
  def testNormCdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.norm.cdf
    lax_fun = lsp_stats.norm.cdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None)
      return [x, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-6)
    self._CompileAndCheck(lax_fun, args_maker)


  @genNamedParametersNArgs(3)
  def testNormPpf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.norm.ppf
    lax_fun = lsp_stats.norm.ppf

    def args_maker():
      q, loc, scale = map(rng, shapes, dtypes)
      # ensure probability is between 0 and 1:
      q = np.clip(np.abs(q / 3), a_min=None, a_max=1)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None)
      return [q, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, tol=1e-4)
    self._CompileAndCheck(lax_fun, args_maker, rtol=3e-4)


  @genNamedParametersNArgs(4)
  def testParetoLogPdf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.pareto.logpdf
    lax_fun = lsp_stats.pareto.logpdf

    def args_maker():
      x, b, loc, scale = map(rng, shapes, dtypes)
      return [x, b, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-3)
    self._CompileAndCheck(lax_fun, args_maker)


  @genNamedParametersNArgs(4)
  def testTLogPdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.t.logpdf
    lax_fun = lsp_stats.t.logpdf

    def args_maker():
      x, df, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = np.clip(np.abs(scale), a_min=0.1, a_max=None)
      return [x, df, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-3)
    self._CompileAndCheck(lax_fun, args_maker,
                          rtol={np.float64: 1e-14}, atol={np.float64: 1e-14})


  @genNamedParametersNArgs(3)
  def testUniformLogPdf(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    scipy_fun = osp_stats.uniform.logpdf
    lax_fun = lsp_stats.uniform.logpdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      return [x, loc, np.abs(scale)]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(4)
  def testChi2LogPdf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.chi2.logpdf
    lax_fun = lsp_stats.chi2.logpdf

    def args_maker():
      x, df, loc, scale = map(rng, shapes, dtypes)
      return [x, df, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=5e-4)
    self._CompileAndCheck(lax_fun, args_maker)

  @genNamedParametersNArgs(5)
  def testBetaBinomLogPmf(self, shapes, dtypes):
    rng = jtu.rand_positive(self.rng())
    scipy_fun = osp_stats.betabinom.logpmf
    lax_fun = lsp_stats.betabinom.logpmf

    def args_maker():
      k, n, a, b, loc = map(rng, shapes, dtypes)
      k = np.floor(k)
      n = np.ceil(n)
      a = np.clip(a, a_min = 0.1, a_max = None)
      b = np.clip(a, a_min = 0.1, a_max = None)
      loc = np.floor(loc)
      return [k, n, a, b, loc]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=5e-4)
    self._CompileAndCheck(lax_fun, args_maker, rtol=1e-5, atol=1e-5)

  def testIssue972(self):
    self.assertAllClose(
      np.ones((4,), np.float32),
      lsp_stats.norm.cdf(np.full((4,), np.inf, np.float32)),
      check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_x={}_mean={}_cov={}".format(
          jtu.format_shape_dtype_string(x_shape, x_dtype),
          jtu.format_shape_dtype_string(mean_shape, mean_dtype)
          if mean_shape is not None else None,
          jtu.format_shape_dtype_string(cov_shape, cov_dtype)
          if cov_shape is not None else None),
       "x_shape": x_shape, "x_dtype": x_dtype,
       "mean_shape": mean_shape, "mean_dtype": mean_dtype,
       "cov_shape": cov_shape, "cov_dtype": cov_dtype}
      for x_shape, mean_shape, cov_shape in [
          # # These test cases cover default values for mean/cov, but we don't
          # # support those yet (and they seem not very valuable).
          # [(), None, None],
          # [(), (), None],
          # [(2,), None, None],
          # [(2,), (), None],
          # [(2,), (2,), None],
          # [(3, 2), (3, 2,), None],
          # [(5, 3, 2), (5, 3, 2,), None],

          [(), (), ()],
          [(3,), (), ()],
          [(3,), (3,), ()],
          [(3,), (3,), (3, 3)],
          [(3, 4), (4,), (4, 4)],

          # # These test cases are where scipy flattens things, which has
          # # different batch semantics than some might expect
          # [(5, 3, 2), (5, 3, 2,), ()],
          # [(5, 3, 2), (5, 3, 2,), (5, 3, 2, 2)],
          # [(5, 3, 2), (3, 2,), (5, 3, 2, 2)],
          # [(5, 3, 2), (3, 2,), (2, 2)],
      ]
      for x_dtype, mean_dtype, cov_dtype in itertools.combinations_with_replacement(jtu.dtypes.floating, 3)
      if (mean_shape is not None or mean_dtype == np.float32)
      and (cov_shape is not None or cov_dtype == np.float32)))
  def testMultivariateNormalLogpdf(self, x_shape, x_dtype, mean_shape,
                                   mean_dtype, cov_shape, cov_dtype):
    rng = jtu.rand_default(self.rng())
    def args_maker():
      args = [rng(x_shape, x_dtype)]
      if mean_shape is not None:
        args.append(5 * rng(mean_shape, mean_dtype))
      if cov_shape is not None:
        if cov_shape == ():
          args.append(0.1 + rng(cov_shape, cov_dtype) ** 2)
        else:
          factor_shape = (*cov_shape[:-1], 2 * cov_shape[-1])
          factor = rng(factor_shape, cov_dtype)
          args.append(np.matmul(factor, np.swapaxes(factor, -1, -2)))
      return args

    self._CheckAgainstNumpy(osp_stats.multivariate_normal.logpdf,
                            lsp_stats.multivariate_normal.logpdf,
                            args_maker, tol=1e-3)
    self._CompileAndCheck(lsp_stats.multivariate_normal.logpdf, args_maker,
                          rtol=1e-4, atol=1e-4)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_ndim={}_nbatch={}_dtype={}".format(ndim, nbatch, dtype.__name__),
       "ndim": ndim, "nbatch": nbatch, "dtype": dtype}
      for ndim in [2, 3]
      for nbatch in [1, 3, 5]
      for dtype in jtu.dtypes.floating))
  def testMultivariateNormalLogpdfBatch(self, ndim, nbatch, dtype):
    # Regression test for #5570
    rng = jtu.rand_default(self.rng())
    x = rng((nbatch, ndim), dtype)
    mean = 5 * rng((nbatch, ndim), dtype)
    factor = rng((nbatch, ndim, 2 * ndim), dtype)
    cov = factor @ factor.transpose(0, 2, 1)

    result1 = lsp_stats.multivariate_normal.logpdf(x, mean, cov)
    result2 = api.vmap(lsp_stats.multivariate_normal.logpdf)(x, mean, cov)
    self.assertArraysEqual(result1, result2)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
