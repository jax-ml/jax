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
import itertools
import unittest

from absl.testing import absltest, parameterized

import numpy as onp
import scipy.stats as osp_stats
from scipy.stats import random_correlation

from jax import test_util as jtu
from jax.scipy import stats as lsp_stats
from jax.scipy.special import expit

from jax.config import config
config.parse_flags_with_absl()

all_shapes = [(), (4,), (3, 4), (3, 1), (1, 4), (2, 1, 4)]

float_dtypes = [onp.float32, onp.float64]

CombosWithReplacement = itertools.combinations_with_replacement

def genNamedParametersNArgs(n, rng_factory):
    return parameterized.named_parameters(
        jtu.cases_from_list(
          {"testcase_name": jtu.format_test_name_suffix("", shapes, dtypes),
            "rng_factory": rng_factory, "shapes": shapes, "dtypes": dtypes}
          for shapes in CombosWithReplacement(all_shapes, n)
          for dtypes in CombosWithReplacement(float_dtypes, n)))


class LaxBackedScipyStatsTests(jtu.JaxTestCase):
  """Tests for LAX-backed scipy.stats implementations"""

  @genNamedParametersNArgs(3, jtu.rand_default)
  def testPoissonLogPmf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.poisson.logpmf
    lax_fun = lsp_stats.poisson.logpmf

    def args_maker():
      k, mu, loc = map(rng, shapes, dtypes)
      k = onp.floor(k)
      # clipping to ensure that rate parameter is strictly positive
      mu = onp.clip(onp.abs(mu), a_min=0.1, a_max=None)
      loc = onp.floor(loc)
      return [k, mu, loc]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-3)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)

  @genNamedParametersNArgs(3, jtu.rand_default)
  def testPoissonPmf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.poisson.pmf
    lax_fun = lsp_stats.poisson.pmf

    def args_maker():
      k, mu, loc = map(rng, shapes, dtypes)
      k = onp.floor(k)
      # clipping to ensure that rate parameter is strictly positive
      mu = onp.clip(onp.abs(mu), a_min=0.1, a_max=None)
      loc = onp.floor(loc)
      return [k, mu, loc]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-3)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)

  @genNamedParametersNArgs(3, jtu.rand_default)
  def testBernoulliLogPmf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.bernoulli.logpmf
    lax_fun = lsp_stats.bernoulli.logpmf

    def args_maker():
      x, logit, loc = map(rng, shapes, dtypes)
      x = onp.floor(x)
      p = expit(logit)
      loc = onp.floor(loc)
      return [x, p, loc]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)

  @genNamedParametersNArgs(5, jtu.rand_positive)
  def testBetaLogPdf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.beta.logpdf
    lax_fun = lsp_stats.beta.logpdf

    def args_maker():
      x, a, b, loc, scale = map(rng, shapes, dtypes)
      return [x, a, b, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-3)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True, rtol=1e-4)

  @genNamedParametersNArgs(3, jtu.rand_default)
  def testCauchyLogPdf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.cauchy.logpdf
    lax_fun = lsp_stats.cauchy.logpdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = onp.clip(onp.abs(scale), a_min=0.1, a_max=None)
      return [x, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)

  @genNamedParametersNArgs(2, jtu.rand_positive)
  def testDirichletLogPdf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.cauchy.logpdf
    lax_fun = lsp_stats.cauchy.logpdf
    dim = 4
    shapes = (shapes[0] + (dim,), shapes[1] + (dim,))

    def args_maker():
      x, alpha = map(rng, shapes, dtypes)
      x = x / onp.sum(x, axis=-1, keepdims=True)
      return [x, alpha]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)

  @genNamedParametersNArgs(3, jtu.rand_positive)
  def testExponLogPdf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.expon.logpdf
    lax_fun = lsp_stats.expon.logpdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      return [x, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)

  @genNamedParametersNArgs(4, jtu.rand_positive)
  def testGammaLogPdf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.gamma.logpdf
    lax_fun = lsp_stats.gamma.logpdf

    def args_maker():
      x, a, loc, scale = map(rng, shapes, dtypes)
      return [x, a, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=5e-4)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)

  @genNamedParametersNArgs(3, jtu.rand_positive)
  def testLaplaceLogPdf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.laplace.logpdf
    lax_fun = lsp_stats.laplace.logpdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = onp.clip(scale, a_min=0.1, a_max=None)
      return [x, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)

  @genNamedParametersNArgs(3, jtu.rand_default)
  def testLaplaceCdf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.laplace.cdf
    lax_fun = lsp_stats.laplace.cdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # ensure that scale is not too low
      scale = onp.clip(scale, a_min=0.1, a_max=None)
      return [x, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-6)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)

  @genNamedParametersNArgs(1, jtu.rand_default)
  def testLogisticCdf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.logistic.cdf
    lax_fun = lsp_stats.logistic.cdf

    def args_maker():
      return list(map(rng, shapes, dtypes))

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-6)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)

  @genNamedParametersNArgs(1, jtu.rand_default)
  def testLogisticLogpdf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.logistic.logpdf
    lax_fun = lsp_stats.logistic.logpdf

    def args_maker():
      return list(map(rng, shapes, dtypes))

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-6)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)

  @genNamedParametersNArgs(1, jtu.rand_default)
  def testLogisticPpf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.logistic.ppf
    lax_fun = lsp_stats.logistic.ppf

    def args_maker():
      return list(map(rng, shapes, dtypes))

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-6)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)

  @genNamedParametersNArgs(1, jtu.rand_default)
  def testLogisticSf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.logistic.sf
    lax_fun = lsp_stats.logistic.sf

    def args_maker():
      return list(map(rng, shapes, dtypes))

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-6)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)

  # TODO: currently it ignores the argument "shapes" and only tests dim=4
  @genNamedParametersNArgs(3, jtu.rand_default)
  def testMultivariateNormalLogPdf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.multivariate_normal.logpdf
    lax_fun = lsp_stats.multivariate_normal.logpdf
    dim = 4
    shapex = (dim,)

    def args_maker():
      x, mean, cov = map(rng, (shapex, shapex, (dim, dim)), dtypes)
      cov = random_correlation.rvs(onp.arange(1, 1+dim) * 2 / (dim + 1))
      return [x, mean, cov]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
      tol=1e-4)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)

  @genNamedParametersNArgs(3, jtu.rand_default)
  def testNormLogPdf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.norm.logpdf
    lax_fun = lsp_stats.norm.logpdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = onp.clip(onp.abs(scale), a_min=0.1, a_max=None)
      return [x, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-3)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)


  @genNamedParametersNArgs(3, jtu.rand_default)
  def testNormLogCdf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.norm.logcdf
    lax_fun = lsp_stats.norm.logcdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = onp.clip(onp.abs(scale), a_min=0.1, a_max=None)
      return [x, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)


  @genNamedParametersNArgs(3, jtu.rand_default)
  def testNormCdf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.norm.cdf
    lax_fun = lsp_stats.norm.cdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = onp.clip(onp.abs(scale), a_min=0.1, a_max=None)
      return [x, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-6)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)


  @genNamedParametersNArgs(3, jtu.rand_default)
  def testNormPpf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.norm.ppf
    lax_fun = lsp_stats.norm.ppf

    def args_maker():
      q, loc, scale = map(rng, shapes, dtypes)
      # ensure probability is between 0 and 1:
      q = onp.clip(onp.abs(q / 3), a_min=None, a_max=1)
      # clipping to ensure that scale is not too low
      scale = onp.clip(onp.abs(scale), a_min=0.1, a_max=None)
      return [q, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=True,
                            tol=1e-4)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True, rtol=1e-5)


  @genNamedParametersNArgs(4, jtu.rand_positive)
  def testParetoLogPdf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.pareto.logpdf
    lax_fun = lsp_stats.pareto.logpdf

    def args_maker():
      x, b, loc, scale = map(rng, shapes, dtypes)
      return [x, b, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-3)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)


  @genNamedParametersNArgs(4, jtu.rand_default)
  def testTLogPdf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.t.logpdf
    lax_fun = lsp_stats.t.logpdf

    def args_maker():
      x, df, loc, scale = map(rng, shapes, dtypes)
      # clipping to ensure that scale is not too low
      scale = onp.clip(onp.abs(scale), a_min=0.1, a_max=None)
      return [x, df, loc, scale]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-3)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)


  @genNamedParametersNArgs(3, jtu.rand_default)
  def testUniformLogPdf(self, rng_factory, shapes, dtypes):
    rng = rng_factory()
    scipy_fun = osp_stats.uniform.logpdf
    lax_fun = lsp_stats.uniform.logpdf

    def args_maker():
      x, loc, scale = map(rng, shapes, dtypes)
      return [x, loc, onp.abs(scale)]

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(lax_fun, args_maker, check_dtypes=True)

  def testIssue972(self):
    self.assertAllClose(
      onp.ones((4,), onp.float32),
      lsp_stats.norm.cdf(onp.full((4,), onp.inf, onp.float32)),
      check_dtypes=False)


if __name__ == "__main__":
    absltest.main()
