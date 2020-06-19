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


from functools import partial
from unittest import SkipTest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import scipy.linalg
import scipy.special
import scipy.stats

from jax import api
from jax import core
from jax import grad
from jax import lax
from jax import numpy as jnp
from jax import random
from jax import test_util as jtu
from jax import vmap
from jax.interpreters import xla

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS

def supported_dtypes(dtypes):
  return [t for t in dtypes if t in jtu.supported_dtypes()]

float_dtypes = supported_dtypes([jnp.bfloat16, np.float16, np.float32, np.float64])
int_dtypes = supported_dtypes([np.int8, np.int16, np.int32, np.int64])
uint_dtypes = supported_dtypes([np.uint8, np.uint16, np.uint32, np.uint64])

class LaxRandomTest(jtu.JaxTestCase):

  def _CheckCollisions(self, samples, nbits):
    fail_prob = 0.01  # conservative bound on statistical fail prob by Chebyshev
    nitems = len(samples)
    nbins = 2 ** nbits
    nexpected = nbins * (1 - ((nbins - 1) / nbins) ** nitems)
    ncollisions = len(np.unique(samples))
    sq_percent_deviation = ((ncollisions - nexpected) / nexpected) ** 2
    self.assertLess(sq_percent_deviation, 1 / np.sqrt(nexpected * fail_prob))

  def _CheckKolmogorovSmirnovCDF(self, samples, cdf):
    fail_prob = 0.01  # conservative bound on statistical fail prob by Kolmo CDF
    self.assertGreater(scipy.stats.kstest(samples, cdf).pvalue, fail_prob)

  def _CheckChiSquared(self, samples, pmf):
    alpha = 0.01  # significance level, threshold for p-value
    values, actual_freq = np.unique(samples, return_counts=True)
    expected_freq = pmf(values) * samples.size
    # per scipy: "A typical rule is that all of the observed and expected
    # frequencies should be at least 5."
    valid = (actual_freq > 5) & (expected_freq > 5)
    self.assertGreater(valid.sum(), 1,
                       msg='not enough valid frequencies for chi-squared test')
    _, p_value = scipy.stats.chisquare(
        actual_freq[valid], expected_freq[valid])
    self.assertGreater(
        p_value, alpha,
        msg=f'Failed chi-squared test with p={p_value}.\n'
            'Expected vs. actual frequencies:\n'
            f'{expected_freq[valid]}\n{actual_freq[valid]}')

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": np.dtype(dtype).name}
      for dtype in [np.float32, np.float64]))
  def testNumpyAndXLAAgreeOnFloatEndianness(self, dtype):
    if not FLAGS.jax_enable_x64 and jnp.issubdtype(dtype, np.float64):
      raise SkipTest("can't test float64 agreement")

    bits_dtype = np.uint32 if jnp.finfo(dtype).bits == 32 else np.uint64
    numpy_bits = np.array(1., dtype).view(bits_dtype)
    xla_bits = api.jit(
        lambda: lax.bitcast_convert_type(np.array(1., dtype), bits_dtype))()
    self.assertEqual(numpy_bits, xla_bits)

  def testThreefry2x32(self):
    # We test the hash by comparing to known values provided in the test code of
    # the original reference implementation of Threefry. For the values, see
    # https://github.com/DEShawResearch/Random123-Boost/blob/65e3d874b67aa7b3e02d5ad8306462f52d2079c0/libs/random/test/test_threefry.cpp#L30-L32
    def result_to_hex(result):
      return tuple([hex(x.copy()).rstrip("L") for x in result])

    expected = ("0x6b200159", "0x99ba4efe")
    result = random.threefry_2x32(np.uint32([0, 0]), np.uint32([0, 0]))

    self.assertEqual(expected, result_to_hex(result))

    expected = ("0x1cb996fc", "0xbb002be7")
    result = random.threefry_2x32(np.uint32([-1, -1]), np.uint32([-1, -1]))
    self.assertEqual(expected, result_to_hex(result))

    expected = ("0xc4923a9c", "0x483df7a0")
    result = random.threefry_2x32(
        np.uint32([0x13198a2e, 0x03707344]),
        np.uint32([0x243f6a88, 0x85a308d3]))
    self.assertEqual(expected, result_to_hex(result))

  def testThreefry2x32Large(self):
    n = 10000000
    result = random.threefry_2x32(
      (np.uint32(0x13198a2e), np.uint32(0x03707344)),
      jnp.concatenate([
        jnp.full((n,), 0x243f6a88, jnp.uint32),
        jnp.full((n,), 0x85a308d3, jnp.uint32)
      ]))
    np.testing.assert_equal(result[:n], np.full((n,), 0xc4923a9c, dtype=np.uint32))
    np.testing.assert_equal(result[n:], np.full((n,), 0x483df7a0, dtype=np.uint32))

  def testRngRandomBitsViewProperty(self):
    # TODO: add 64-bit if it ever supports this property.
    # TODO: will this property hold across endian-ness?
    N = 10
    key = random.PRNGKey(1701)
    nbits = [8, 16, 32]
    if jtu.device_under_test() == "tpu":
      # U8 and U16 are not supported on TPU.
      nbits = [32]
    rand_bits = [random._random_bits(key, n, (N * 64 // n,)) for n in nbits]
    rand_bits_32 = np.array([np.array(r).view(np.uint32) for r in rand_bits])
    assert np.all(rand_bits_32 == rand_bits_32[0])

  def testRngRandomBits(self):
    # Test specific outputs to ensure consistent random values between JAX versions.
    key = random.PRNGKey(1701)

    # U8 and U16 are not supported on TPU.
    if jtu.device_under_test() != "tpu":
      bits8 = random._random_bits(key, 8, (3,))
      expected8 = np.array([216, 115,  43], dtype=np.uint8)
      self.assertArraysEqual(bits8, expected8)

      bits16 = random._random_bits(key, 16, (3,))
      expected16 = np.array([41682,  1300, 55017], dtype=np.uint16)
      self.assertArraysEqual(bits16, expected16)

    bits32 = random._random_bits(key, 32, (3,))
    expected32 = np.array([56197195, 4200222568, 961309823], dtype=np.uint32)
    self.assertArraysEqual(bits32, expected32)

    bits64 = random._random_bits(key, 64, (3,))
    if FLAGS.jax_enable_x64:
      expected64 = np.array([3982329540505020460, 16822122385914693683,
                             7882654074788531506], dtype=np.uint64)
    else:
      expected64 = np.array([676898860, 3164047411, 4010691890], dtype=np.uint32)
    self.assertArraysEqual(bits64, expected64)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": np.dtype(dtype)}
      for dtype in float_dtypes))
  def testRngUniform(self, dtype):
    if jtu.device_under_test() == "tpu" and jnp.dtype(dtype).itemsize < 3:
      raise SkipTest("random.uniform() not supported on TPU for 16-bit types.")
    key = random.PRNGKey(0)
    rand = lambda key: random.uniform(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckCollisions(samples, jnp.finfo(dtype).nmant)
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.uniform().cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": np.dtype(dtype).name}
      for dtype in int_dtypes + uint_dtypes))
  def testRngRandint(self, dtype):
    if jtu.device_under_test() == "tpu" and jnp.dtype(dtype).itemsize < 3:
      raise SkipTest("random.randint() not supported on TPU for 8- or 16-bit types.")
    lo = 5
    hi = 10

    key = random.PRNGKey(0)
    rand = lambda key: random.randint(key, (10000,), lo, hi, dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self.assertTrue(np.all(lo <= samples))
      self.assertTrue(np.all(samples < hi))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": np.dtype(dtype)}
      for dtype in [np.float16, np.float32, np.float64]))
  def testNormal(self, dtype):
    if jtu.device_under_test() == "tpu" and jnp.dtype(dtype).itemsize < 3:
      raise SkipTest("random.normal() not supported on TPU for 16-bit types.")
    key = random.PRNGKey(0)
    rand = lambda key: random.normal(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.norm().cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": np.dtype(dtype).name}
      for dtype in [np.float32, np.float64, np.int32, np.int64]))
  def testShuffle(self, dtype):
    key = random.PRNGKey(0)
    x = np.arange(100).astype(dtype)
    rand = lambda key: random.shuffle(key, x)
    crand = api.jit(rand)

    with self.assertWarns(FutureWarning):
      perm1 = rand(key)
    with self.assertWarns(FutureWarning):
      perm2 = crand(key)

    self.assertAllClose(perm1, perm2)
    self.assertFalse(np.all(perm1 == x))  # seems unlikely!
    self.assertAllClose(np.sort(perm1), x, check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "dtype": np.dtype(dtype).name, "shape": shape}
      for dtype in [np.float32, np.float64, np.int32, np.int64]
      for shape in [100, (10, 10), (10, 5, 2)]))
  def testPermutationArray(self, dtype, shape):
    key = random.PRNGKey(0)
    x = jnp.arange(np.prod(shape)).reshape(shape).astype(dtype)
    rand = lambda key: random.permutation(key, x)
    crand = api.jit(rand)

    perm1 = rand(key)
    perm2 = crand(key)

    self.assertAllClose(perm1, perm2)
    self.assertFalse(np.all(perm1 == x))  # seems unlikely!
    self.assertAllClose(np.sort(perm1.ravel()), x.ravel(), check_dtypes=False)
    self.assertArraysAllClose(
      x, jnp.arange(np.prod(shape)).reshape(shape).astype(dtype))

  def testPermutationInteger(self):
    key = random.PRNGKey(0)
    x = 100
    rand = lambda key: random.permutation(key, x)
    crand = api.jit(rand)

    perm1 = rand(key)
    perm2 = crand(key)

    self.assertAllClose(perm1, perm2)
    self.assertEqual(perm1.dtype, perm2.dtype)
    self.assertFalse(np.all(perm1 == np.arange(100)))  # seems unlikely!
    self.assertAllClose(np.sort(perm1), np.arange(100), check_dtypes=False)

  def testPermutationErrors(self):
    key = random.PRNGKey(0)
    with self.assertRaises(TypeError):
      random.permutation(key, 10.)
    with self.assertRaises(core.ConcretizationTypeError):
      api.jit(random.permutation)(key, 10)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_p={}_{}".format(p, dtype),
       "p": p, "dtype": np.dtype(dtype).name}
      for p in [0.1, 0.5, 0.9]
      for dtype in [np.float32, np.float64]))
  def testBernoulli(self, p, dtype):
    key = random.PRNGKey(0)
    p = np.array(p, dtype=dtype)
    rand = lambda key, p: random.bernoulli(key, p, (10000,))
    crand = api.jit(rand)

    uncompiled_samples = rand(key, p)
    compiled_samples = crand(key, p)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckChiSquared(samples, scipy.stats.bernoulli(p).pmf)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_p={}_{}_{}".format(p, dtype, sample_shape),
     "p": p, "axis": axis, "dtype": np.dtype(dtype).name, 'sample_shape': sample_shape}
    for (p, axis) in [
        ([.25] * 4, -1),
        ([.1, .2, .3, .4], -1),
        ([[.5, .5], [.1, .9]], 1),
        ([[.5, .1], [.5, .9]], 0),
    ]
    for sample_shape in [(10000,), (5000, 2)]
    for dtype in [np.float32, np.float64]))
  def testCategorical(self, p, axis, dtype, sample_shape):
    key = random.PRNGKey(0)
    p = np.array(p, dtype=dtype)
    logits = np.log(p) - 42 # test unnormalized
    out_shape = tuple(np.delete(logits.shape, axis))
    shape = sample_shape + out_shape
    rand = lambda key, p: random.categorical(key, logits, shape=shape, axis=axis)
    crand = api.jit(rand)

    uncompiled_samples = rand(key, p)
    compiled_samples = crand(key, p)

    if axis < 0:
      axis += len(logits.shape)

    for samples in [uncompiled_samples, compiled_samples]:
      assert samples.shape == shape
      samples = jnp.reshape(samples, (10000,) + out_shape)
      if len(p.shape[:-1]) > 0:
        ps = np.transpose(p, (1, 0)) if axis == 0 else p
        for cat_samples, cat_p in zip(samples.transpose(), ps):
          self._CheckChiSquared(cat_samples, pmf=lambda x: cat_p[x])
      else:
        self._CheckChiSquared(samples, pmf=lambda x: p[x])

  def testBernoulliShape(self):
    key = random.PRNGKey(0)
    x = random.bernoulli(key, np.array([0.2, 0.3]), shape=(3, 2))
    assert x.shape == (3, 2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_a={}_b={}_{}".format(a, b, dtype),
       "a": a, "b": b, "dtype": np.dtype(dtype).name}
      for a in [0.2, 5.]
      for b in [0.2, 5.]
      for dtype in [np.float64]))  # NOTE: KS test fails with float32
  def testBeta(self, a, b, dtype):
    if not FLAGS.jax_enable_x64:
      raise SkipTest("skip test except on X64")
    key = random.PRNGKey(0)
    rand = lambda key, a, b: random.beta(key, a, b, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key, a, b)
    compiled_samples = crand(key, a, b)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.beta(a, b).cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": np.dtype(dtype)}
      for dtype in [np.float16, np.float32, np.float64]))
  def testCauchy(self, dtype):
    if jtu.device_under_test() == "tpu" and jnp.dtype(dtype).itemsize < 3:
      raise SkipTest("random.cauchy() not supported on TPU for 16-bit types.")
    key = random.PRNGKey(0)
    rand = lambda key: random.cauchy(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.cauchy().cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_alpha={}_{}".format(alpha, dtype),
       "alpha": alpha, "dtype": np.dtype(dtype).name}
      for alpha in [
          np.array([0.2, 1., 5.]),
      ]
      for dtype in [np.float32, np.float64]))
  def testDirichlet(self, alpha, dtype):
    key = random.PRNGKey(0)
    rand = lambda key, alpha: random.dirichlet(key, alpha, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key, alpha)
    compiled_samples = crand(key, alpha)

    for samples in [uncompiled_samples, compiled_samples]:
      self.assertAllClose(samples.sum(-1), np.ones(10000, dtype=dtype))
      alpha_sum = sum(alpha)
      for i, a in enumerate(alpha):
        self._CheckKolmogorovSmirnovCDF(samples[..., i], scipy.stats.beta(a, alpha_sum - a).cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": np.dtype(dtype)}
      for dtype in float_dtypes))
  def testExponential(self, dtype):
    if jtu.device_under_test() == "tpu" and jnp.dtype(dtype).itemsize < 3:
      raise SkipTest("random.exponential() not supported on TPU for 16-bit types.")
    key = random.PRNGKey(0)
    rand = lambda key: random.exponential(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.expon().cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_a={}_{}".format(a, dtype),
       "a": a, "dtype": np.dtype(dtype).name}
      for a in [0.1, 1., 10.]
      for dtype in [np.float32, np.float64]))
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
    x = random.gamma(key, np.array([0.2, 0.3]), shape=(3, 2))
    assert x.shape == (3, 2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_a={}".format(alpha), "alpha": alpha}
      for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]))
  def testGammaGrad(self, alpha):
    rng = random.PRNGKey(0)
    alphas = np.full((100,), alpha)
    z = random.gamma(rng, alphas)
    actual_grad = api.grad(lambda x: random.gamma(rng, x).sum())(alphas)

    eps = 0.01 * alpha / (1.0 + np.sqrt(alpha))
    cdf_dot = (scipy.stats.gamma.cdf(z, alpha + eps)
               - scipy.stats.gamma.cdf(z, alpha - eps)) / (2 * eps)
    pdf = scipy.stats.gamma.pdf(z, alpha)
    expected_grad = -cdf_dot / pdf

    self.assertAllClose(actual_grad, expected_grad, check_dtypes=True,
                        rtol=2e-2 if jtu.device_under_test() == "tpu" else 7e-4)

  def testGammaGradType(self):
    # Regression test for https://github.com/google/jax/issues/2130
    key = random.PRNGKey(0)
    a = jnp.array(1., dtype=jnp.float32)
    b = jnp.array(3., dtype=jnp.float32)
    f = lambda x, y: random.gamma(key=key, a=x, dtype=jnp.float32) / y
    # Should not crash with a type error.
    api.vjp(f, a, b)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_lam={}_{}".format(lam, dtype),
       "lam": lam, "dtype": np.dtype(dtype)}
      for lam in [0.5, 3, 9, 11, 50, 500]
      for dtype in [np.int16, np.int32, np.int64]))
  def testPoisson(self, lam, dtype):
    if jtu.device_under_test() == "tpu" and jnp.dtype(dtype).itemsize < 3:
      raise SkipTest("random.poisson() not supported on TPU for 16-bit types.")
    key = random.PRNGKey(0)
    rand = lambda key, lam: random.poisson(key, lam, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key, lam)
    compiled_samples = crand(key, lam)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckChiSquared(samples, scipy.stats.poisson(lam).pmf)
      # TODO(shoyer): determine error bounds for moments more rigorously (e.g.,
      # based on the central limit theorem).
      self.assertAllClose(samples.mean(), lam, rtol=0.01, check_dtypes=False)
      self.assertAllClose(samples.var(), lam, rtol=0.03, check_dtypes=False)

  def testPoissonBatched(self):
    key = random.PRNGKey(0)
    lam = jnp.concatenate([2 * jnp.ones(10000), 20 * jnp.ones(10000)])
    samples = random.poisson(key, lam, shape=(20000,))
    self._CheckChiSquared(samples[:10000], scipy.stats.poisson(2.0).pmf)
    self._CheckChiSquared(samples[10000:], scipy.stats.poisson(20.0).pmf)

  def testPoissonShape(self):
    key = random.PRNGKey(0)
    x = random.poisson(key, np.array([2.0, 20.0]), shape=(3, 2))
    assert x.shape == (3, 2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": np.dtype(dtype).name}
      for dtype in [np.float32, np.float64]))
  def testGumbel(self, dtype):
    key = random.PRNGKey(0)
    rand = lambda key: random.gumbel(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.gumbel_r().cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": np.dtype(dtype)}
      for dtype in float_dtypes))
  def testLaplace(self, dtype):
    if jtu.device_under_test() == "tpu" and jnp.dtype(dtype).itemsize < 3:
      raise SkipTest("random.laplace() not supported on TPU for 16-bit types.")
    key = random.PRNGKey(0)
    rand = lambda key: random.laplace(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.laplace().cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": np.dtype(dtype)}
      for dtype in float_dtypes))
  def testLogistic(self, dtype):
    if jtu.device_under_test() == "tpu" and jnp.dtype(dtype).itemsize < 3:
      raise SkipTest("random.logistic() not supported on TPU for 16-bit types.")
    key = random.PRNGKey(0)
    rand = lambda key: random.logistic(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.logistic().cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_b={}_{}".format(b, dtype),
       "b": b, "dtype": np.dtype(dtype).name}
      for b in [0.1, 1., 10.]
      for dtype in [np.float32, np.float64]))
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
    x = random.pareto(key, np.array([0.2, 0.3]), shape=(3, 2))
    assert x.shape == (3, 2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_df={}_{}".format(df, dtype),
       "df": df, "dtype": np.dtype(dtype).name}
      for df in [0.1, 1., 10.]
      for dtype in [np.float32, np.float64]))
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
      {"testcase_name": "_{}D_{}".format(dim, np.dtype(dtype)),
       "dim": dim, "dtype": dtype}
      for dim in [1, 3, 5]
      for dtype in float_dtypes))
  def testMultivariateNormal(self, dim, dtype):
    if jtu.device_under_test() == "tpu" and jnp.dtype(dtype).itemsize < 3:
      raise SkipTest("random.multivariate_normal() not supported on TPU for 16-bit types.")
    r = np.random.RandomState(dim)
    mean = r.randn(dim)
    cov_factor = r.randn(dim, dim)
    cov = np.dot(cov_factor, cov_factor.T) + dim * np.eye(dim)

    key = random.PRNGKey(0)
    rand = partial(random.multivariate_normal, mean=mean, cov=cov,
                   shape=(10000,))
    crand = api.jit(rand)

    uncompiled_samples = np.asarray(rand(key), np.float64)
    compiled_samples = np.asarray(crand(key), np.float64)

    inv_scale = scipy.linalg.lapack.dtrtri(np.linalg.cholesky(cov), lower=True)[0]
    for samples in [uncompiled_samples, compiled_samples]:
      centered = samples - mean
      whitened = np.einsum('nj,ij->ni', centered, inv_scale)

      # This is a quick-and-dirty multivariate normality check that tests that a
      # uniform mixture of the marginals along the covariance matrix's
      # eigenvectors follow a standard normal distribution.
      self._CheckKolmogorovSmirnovCDF(whitened.ravel(), scipy.stats.norm().cdf)

  def testMultivariateNormalCovariance(self):
    # test code based on https://github.com/google/jax/issues/1869
    N = 100000
    cov = jnp.array([[ 0.19,  0.00, -0.13,  0.00],
                   [  0.00,  0.29,  0.00, -0.23],
                   [ -0.13,  0.00,  0.39,  0.00],
                   [  0.00, -0.23,  0.00,  0.49]])
    mean = jnp.zeros(4)

    out_np = np.random.RandomState(0).multivariate_normal(mean, cov, N)

    key = random.PRNGKey(0)
    out_jnp = random.multivariate_normal(key, mean=mean, cov=cov, shape=(N,))

    var_np = out_np.var(axis=0)
    var_jnp = out_jnp.var(axis=0)
    self.assertAllClose(var_np, var_jnp, rtol=1e-2, atol=1e-2,
                        check_dtypes=False)

    var_np = np.cov(out_np, rowvar=False)
    var_jnp = np.cov(out_jnp, rowvar=False)
    self.assertAllClose(var_np, var_jnp, rtol=1e-2, atol=1e-2,
                        check_dtypes=False)

  def testIssue222(self):
    x = random.randint(random.PRNGKey(10003), (), 0, 0)
    assert x == 0

  def testFoldIn(self):
    key = random.PRNGKey(0)
    keys = [random.fold_in(key, i) for i in range(10)]
    assert np.unique(np.ravel(keys)).shape == (20,)

  def testStaticShapeErrors(self):
    if config.read("jax_disable_jit"):
      raise SkipTest("test only relevant when jit enabled")

    @api.jit
    def feature_map(n, d, sigma=1.0, seed=123):
      key = random.PRNGKey(seed)
      W = random.normal(key, (d, n)) / sigma
      w = random.normal(key, (d, )) / sigma
      b = 2 * jnp.pi * random.uniform(key, (d, ))

      phi = lambda x, t: jnp.sqrt(2.0 / d) * jnp.cos(jnp.matmul(W, x) + w*t + b)
      return phi

    self.assertRaisesRegex(TypeError, 'Shapes must be 1D.*',
                           lambda: feature_map(5, 3))

  def testIssue756(self):
    key = random.PRNGKey(0)
    w = random.normal(key, ())
    if FLAGS.jax_enable_x64:
      self.assertEqual(np.result_type(w), np.float64)
    else:
      self.assertEqual(np.result_type(w), np.float32)

  def testIssue1789(self):
    def f(x):
      return random.gamma(random.PRNGKey(0), x)

    grad(lambda x: jnp.sum(vmap(f)(x)))(jnp.ones(2))

  def testNoOpByOpUnderHash(self):
    def fail(*args, **kwargs): assert False
    apply_primitive, xla.apply_primitive = xla.apply_primitive, fail
    try:
      _ = random.threefry_2x32(np.zeros(2, np.uint32), np.arange(10, dtype=np.uint32))
    finally:
      xla.apply_primitive = apply_primitive

  def testPRNGValues(self):
    # Test to ensure consistent random values between JAX versions
    k = random.PRNGKey(0)

    if FLAGS.jax_enable_x64:
        self.assertAllClose(
            random.randint(k, (3, 3), 0, 8),
            np.array([[7, 2, 6],
                       [2, 1, 0],
                       [6, 7, 7]], dtype='int64'))
    else:
        self.assertAllClose(
            random.randint(k, (3, 3), 0, 8),
            np.array([[2, 1, 3],
                       [6, 1, 5],
                       [6, 3, 4]], dtype='int32'))

    self.assertAllClose(
        random.split(k, 4),
        np.array([[2285895361, 1501764800],
                   [1518642379, 4090693311],
                   [ 433833334, 4221794875],
                   [ 839183663, 3740430601]], dtype='uint32'))

    self.assertAllClose(
        random.fold_in(k, 4),
        np.array([2285895361,  433833334], dtype='uint32'))

  def testDtypeErrorMessage(self):
    with self.assertRaisesRegex(ValueError, r"dtype argument to.*"):
      random.normal(random.PRNGKey(0), (), dtype=jnp.int32)


if __name__ == "__main__":
  absltest.main()
