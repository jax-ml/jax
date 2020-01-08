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

from functools import partial
from unittest import SkipTest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp
import scipy.special
import scipy.stats

from jax import api
from jax import grad
from jax import lax
from jax import numpy as np
from jax import random
from jax import test_util as jtu
from jax import vmap
from jax.interpreters import xla

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS


class LaxRandomTest(jtu.JaxTestCase):

  def _CheckCollisions(self, samples, nbits):
    fail_prob = 0.01  # conservative bound on statistical fail prob by Chebyshev
    nitems = len(samples)
    nbins = 2 ** nbits
    nexpected = nbins * (1 - ((nbins - 1) / nbins) ** nitems)
    ncollisions = len(onp.unique(samples))
    sq_percent_deviation = ((ncollisions - nexpected) / nexpected) ** 2
    self.assertLess(sq_percent_deviation, 1 / onp.sqrt(nexpected * fail_prob))

  def _CheckKolmogorovSmirnovCDF(self, samples, cdf):
    fail_prob = 0.01  # conservative bound on statistical fail prob by Kolmo CDF
    self.assertGreater(scipy.stats.kstest(samples, cdf).pvalue, fail_prob)

  def _CheckChiSquared(self, samples, pmf):
    alpha = 0.01  # significance level, threshold for p-value
    values, actual_freq = onp.unique(samples, return_counts=True)
    expected_freq = pmf(values) * len(values)
    _, p_value = scipy.stats.chisquare(actual_freq, expected_freq)
    self.assertLess(p_value, alpha)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": onp.dtype(dtype).name}
      for dtype in [onp.float32, onp.float64]))
  def testNumpyAndXLAAgreeOnFloatEndianness(self, dtype):
    if not FLAGS.jax_enable_x64 and np.issubdtype(dtype, onp.float64):
      raise SkipTest("can't test float64 agreement")

    bits_dtype = onp.uint32 if np.finfo(dtype).bits == 32 else onp.uint64
    numpy_bits = onp.array(1., dtype).view(bits_dtype)
    xla_bits = api.jit(
        lambda: lax.bitcast_convert_type(onp.array(1., dtype), bits_dtype))()
    self.assertEqual(numpy_bits, xla_bits)

  def testThreefry2x32(self):
    # We test the hash by comparing to known values provided in the test code of
    # the original reference implementation of Threefry. For the values, see
    # https://github.com/DEShawResearch/Random123-Boost/blob/65e3d874b67aa7b3e02d5ad8306462f52d2079c0/libs/random/test/test_threefry.cpp#L30-L32
    def result_to_hex(result):
      return tuple([hex(x.copy()).rstrip("L") for x in result])

    expected = ("0x6b200159", "0x99ba4efe")
    result = random.threefry_2x32(onp.uint32([0, 0]), onp.uint32([0, 0]))

    self.assertEqual(expected, result_to_hex(result))

    expected = ("0x1cb996fc", "0xbb002be7")
    result = random.threefry_2x32(onp.uint32([-1, -1]), onp.uint32([-1, -1]))
    self.assertEqual(expected, result_to_hex(result))

    expected = ("0xc4923a9c", "0x483df7a0")
    result = random.threefry_2x32(
        onp.uint32([0x13198a2e, 0x03707344]),
        onp.uint32([0x243f6a88, 0x85a308d3]))
    self.assertEqual(expected, result_to_hex(result))

  def testThreefry2x32Large(self):
    n = 10000000
    result = random.threefry_2x32(
      (onp.uint32(0x13198a2e), onp.uint32(0x03707344)),
      np.concatenate([
        np.full((n,), 0x243f6a88, np.uint32),
        np.full((n,), 0x85a308d3, np.uint32)
      ]))
    onp.testing.assert_equal(result[:n], onp.full((n,), 0xc4923a9c, dtype=onp.uint32))
    onp.testing.assert_equal(result[n:], onp.full((n,), 0x483df7a0, dtype=onp.uint32))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": onp.dtype(dtype).name}
      for dtype in [onp.float32, onp.float64]))
  def testRngUniform(self, dtype):
    key = random.PRNGKey(0)
    rand = lambda key: random.uniform(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckCollisions(samples, np.finfo(dtype).nmant)
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.uniform().cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": onp.dtype(dtype).name}
      for dtype in [onp.int32, onp.int64]))
  def testRngRandint(self, dtype):
    lo = 5
    hi = 10

    key = random.PRNGKey(0)
    rand = lambda key: random.randint(key, (10000,), lo, hi, dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self.assertTrue(onp.all(lo <= samples))
      self.assertTrue(onp.all(samples < hi))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": onp.dtype(dtype).name}
      for dtype in [onp.float32, onp.float64]))
  def testNormal(self, dtype):
    key = random.PRNGKey(0)
    rand = lambda key: random.normal(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.norm().cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": onp.dtype(dtype).name}
      for dtype in [onp.float32, onp.float64, onp.int32, onp.int64]))
  def testShuffle(self, dtype):
    key = random.PRNGKey(0)
    x = onp.arange(100).astype(dtype)
    rand = lambda key: random.shuffle(key, x)
    crand = api.jit(rand)

    perm1 = rand(key)
    perm2 = crand(key)

    self.assertTrue(onp.all(perm1 == perm2))
    self.assertTrue(onp.all(perm1.dtype == perm2.dtype))
    self.assertFalse(onp.all(perm1 == x))  # seems unlikely!
    self.assertTrue(onp.all(onp.sort(perm1) == x))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_p={}_{}".format(p, dtype),
       "p": p, "dtype": onp.dtype(dtype).name}
      for p in [0.1, 0.5, 0.9]
      for dtype in [onp.float32, onp.float64]))
  def testBernoulli(self, p, dtype):
    key = random.PRNGKey(0)
    p = onp.array(p, dtype=dtype)
    rand = lambda key, p: random.bernoulli(key, p, (10000,))
    crand = api.jit(rand)

    uncompiled_samples = rand(key, p)
    compiled_samples = crand(key, p)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckChiSquared(samples, scipy.stats.bernoulli(p).pmf)

  def testBernoulliShape(self):
    key = random.PRNGKey(0)
    x = random.bernoulli(key, onp.array([0.2, 0.3]), shape=(3, 2))
    assert x.shape == (3, 2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_a={}_b={}_{}".format(a, b, dtype),
       "a": a, "b": b, "dtype": onp.dtype(dtype).name}
      for a in [0.2, 5.]
      for b in [0.2, 5.]
      for dtype in [onp.float32, onp.float64]))
  # TODO(phawkins): slow compilation times on cpu and tpu.
  # TODO(mattjj): test fails after https://github.com/google/jax/pull/1123
  @jtu.skip_on_devices("cpu", "gpu", "tpu")
  def testBeta(self, a, b, dtype):
    key = random.PRNGKey(0)
    rand = lambda key, a, b: random.beta(key, a, b, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key, a, b)
    compiled_samples = crand(key, a, b)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.beta(a, b).cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": onp.dtype(dtype).name}
      for dtype in [onp.float32, onp.float64]))
  def testCauchy(self, dtype):
    key = random.PRNGKey(0)
    rand = lambda key: random.cauchy(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.cauchy().cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_alpha={}_{}".format(alpha, dtype),
       "alpha": alpha, "dtype": onp.dtype(dtype).name}
      for alpha in [
          onp.array([0.2, 1., 5.]),
      ]
      for dtype in [onp.float32, onp.float64]))
  def testDirichlet(self, alpha, dtype):
    key = random.PRNGKey(0)
    rand = lambda key, alpha: random.dirichlet(key, alpha, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key, alpha)
    compiled_samples = crand(key, alpha)

    for samples in [uncompiled_samples, compiled_samples]:
      self.assertAllClose(samples.sum(-1), onp.ones(10000, dtype=dtype), check_dtypes=True)
      alpha_sum = sum(alpha)
      for i, a in enumerate(alpha):
        self._CheckKolmogorovSmirnovCDF(samples[..., i], scipy.stats.beta(a, alpha_sum - a).cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": onp.dtype(dtype).name}
      for dtype in [onp.float32, onp.float64]))
  def testExponential(self, dtype):
    key = random.PRNGKey(0)
    rand = lambda key: random.exponential(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.expon().cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_a={}_{}".format(a, dtype),
       "a": a, "dtype": onp.dtype(dtype).name}
      for a in [0.1, 1., 10.]
      for dtype in [onp.float32, onp.float64]))
  def testGamma(self, a, dtype):
    key = random.PRNGKey(0)
    rand = lambda key, a: random.gamma(key, a, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key, a)
    compiled_samples = crand(key, a)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.gamma(a).cdf)

  def testGammaShape(self):
    key = random.PRNGKey(0)
    x = random.gamma(key, onp.array([0.2, 0.3]), shape=(3, 2))
    assert x.shape == (3, 2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_a={}".format(alpha), "alpha": alpha}
      for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]))
  def testGammaGrad(self, alpha):
    rng = random.PRNGKey(0)
    alphas = onp.full((100,), alpha)
    z = random.gamma(rng, alphas)
    actual_grad = api.grad(lambda x: random.gamma(rng, x).sum())(alphas)

    eps = 0.01 * alpha / (1.0 + onp.sqrt(alpha))
    cdf_dot = (scipy.stats.gamma.cdf(z, alpha + eps)
               - scipy.stats.gamma.cdf(z, alpha - eps)) / (2 * eps)
    pdf = scipy.stats.gamma.pdf(z, alpha)
    expected_grad = -cdf_dot / pdf

    self.assertAllClose(actual_grad, expected_grad, check_dtypes=True,
                        rtol=2e-2 if jtu.device_under_test() == "tpu" else 5e-4)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": onp.dtype(dtype).name}
      for dtype in [onp.float32, onp.float64]))
  def testGumbel(self, dtype):
    key = random.PRNGKey(0)
    rand = lambda key: random.gumbel(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.gumbel_r().cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": onp.dtype(dtype).name}
      for dtype in [onp.float32, onp.float64]))
  def testLaplace(self, dtype):
    key = random.PRNGKey(0)
    rand = lambda key: random.laplace(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.laplace().cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": onp.dtype(dtype).name}
      for dtype in [onp.float32, onp.float64]))
  def testLogistic(self, dtype):
    key = random.PRNGKey(0)
    rand = lambda key: random.logistic(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.logistic().cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_b={}_{}".format(b, dtype),
       "b": b, "dtype": onp.dtype(dtype).name}
      for b in [0.1, 1., 10.]
      for dtype in [onp.float32, onp.float64]))
  def testPareto(self, b, dtype):
    key = random.PRNGKey(0)
    rand = lambda key, b: random.pareto(key, b, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key, b)
    compiled_samples = crand(key, b)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.pareto(b).cdf)

  def testParetoShape(self):
    key = random.PRNGKey(0)
    x = random.pareto(key, onp.array([0.2, 0.3]), shape=(3, 2))
    assert x.shape == (3, 2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_df={}_{}".format(df, dtype),
       "df": df, "dtype": onp.dtype(dtype).name}
      for df in [0.1, 1., 10.]
      for dtype in [onp.float32, onp.float64]))
  @jtu.skip_on_devices("cpu", "tpu")  # TODO(phawkins): slow compilation times
  def testT(self, df, dtype):
    key = random.PRNGKey(0)
    rand = lambda key, df: random.t(key, df, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key, df)
    compiled_samples = crand(key, df)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.t(df).cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}D_{}".format(dim, onp.dtype(dtype).name),
       "dim": dim, "dtype": dtype}
      for dim in [1, 3, 5]
      for dtype in [onp.float32, onp.float64]))
  def testMultivariateNormal(self, dim, dtype):
    r = onp.random.RandomState(dim)
    mean = r.randn(dim)
    cov_factor = r.randn(dim, dim)
    cov = onp.dot(cov_factor, cov_factor.T) + dim * onp.eye(dim)

    key = random.PRNGKey(0)
    rand = partial(random.multivariate_normal, mean=mean, cov=cov,
                   shape=(10000,))
    crand = api.jit(rand)

    uncompiled_samples = onp.asarray(rand(key), onp.float64)
    compiled_samples = onp.asarray(crand(key), onp.float64)

    inv_scale = scipy.linalg.lapack.dtrtri(onp.linalg.cholesky(cov), lower=True)[0]
    for samples in [uncompiled_samples, compiled_samples]:
      centered = samples - mean
      whitened = onp.einsum('nj,ij->ni', centered, inv_scale)

      # This is a quick-and-dirty multivariate normality check that tests that a
      # uniform mixture of the marginals along the covariance matrix's
      # eigenvectors follow a standard normal distribution.
      self._CheckKolmogorovSmirnovCDF(whitened.ravel(), scipy.stats.norm().cdf)

  def testMultivariateNormalCovariance(self):
    # test code based on https://github.com/google/jax/issues/1869
    N = 100000
    cov = np.array([[ 0.19,  0.00, -0.13,  0.00],
                   [  0.00,  0.29,  0.00, -0.23],
                   [ -0.13,  0.00,  0.39,  0.00],
                   [  0.00, -0.23,  0.00,  0.49]])
    mean = np.zeros(4)

    out_onp = onp.random.RandomState(0).multivariate_normal(mean, cov, N)

    key = random.PRNGKey(0)
    out_jnp = random.multivariate_normal(key, mean=mean, cov=cov, shape=(N,))

    var_onp = out_onp.var(axis=0)
    var_jnp = out_jnp.var(axis=0)
    self.assertAllClose(var_onp, var_jnp, rtol=1e-2, atol=1e-2,
                        check_dtypes=False)

    var_onp = onp.cov(out_onp, rowvar=False)
    var_jnp = onp.cov(out_jnp, rowvar=False)
    self.assertAllClose(var_onp, var_jnp, rtol=1e-2, atol=1e-2,
                        check_dtypes=False)

  def testIssue222(self):
    x = random.randint(random.PRNGKey(10003), (), 0, 0)
    assert x == 0

  def testFoldIn(self):
    key = random.PRNGKey(0)
    keys = [random.fold_in(key, i) for i in range(10)]
    assert onp.unique(onp.ravel(keys)).shape == (20,)

  def testStaticShapeErrors(self):
    if config.read("jax_disable_jit"):
      raise SkipTest("test only relevant when jit enabled")

    @api.jit
    def feature_map(n, d, sigma=1.0, seed=123):
      key = random.PRNGKey(seed)
      W = random.normal(key, (d, n)) / sigma
      w = random.normal(key, (d, )) / sigma
      b = 2 * np.pi * random.uniform(key, (d, ))

      phi = lambda x, t: np.sqrt(2.0 / d) * np.cos(np.matmul(W, x) + w*t + b)
      return phi

    self.assertRaisesRegex(ValueError, '.*requires a concrete.*',
                           lambda: feature_map(5, 3))

  def testIssue756(self):
    key = random.PRNGKey(0)
    w = random.normal(key, ())
    if FLAGS.jax_enable_x64:
      self.assertEqual(onp.result_type(w), onp.float64)
    else:
      self.assertEqual(onp.result_type(w), onp.float32)

  def testIssue1789(self):
    def f(x):
      return random.gamma(random.PRNGKey(0), x)

    grad(lambda x: np.sum(vmap(f)(x)))(np.ones(2))

  def testNoOpByOpUnderHash(self):
    def fail(*args, **kwargs): assert False
    apply_primitive, xla.apply_primitive = xla.apply_primitive, fail
    try:
      out = random.threefry_2x32(onp.zeros(2, onp.uint32), onp.arange(10, dtype=onp.uint32))
    finally:
      xla.apply_primitive = apply_primitive


if __name__ == "__main__":
  absltest.main()
