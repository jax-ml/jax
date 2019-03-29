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

from unittest import SkipTest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp
import scipy.special
import scipy.stats

from jax import api
from jax import lax
from jax import random
from jax import test_util as jtu

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
    statistic = scipy.stats.kstest(samples, cdf).statistic
    self.assertLess(1. - scipy.special.kolmogorov(statistic), fail_prob)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(dtype), "dtype": onp.dtype(dtype).name}
      for dtype in [onp.float32, onp.float64]))
  def testNumpyAndXLAAgreeOnFloatEndianness(self, dtype):
    if not FLAGS.jax_enable_x64 and onp.issubdtype(dtype, onp.float64):
      raise SkipTest("can't test float64 agreement")

    bits_dtype = onp.uint32 if onp.finfo(dtype).bits == 32 else onp.uint64
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
      self._CheckCollisions(samples, onp.finfo(dtype).nmant)
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
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.randint(lo, hi).cdf)

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

  def testBernoulli(self):
    key = random.PRNGKey(0)
    x = random.bernoulli(key, onp.array([0.2, 0.3]), shape=(3, 2))
    assert x.shape == (3, 2)

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

  def testIssue222(self):
    x = random.randint(random.PRNGKey(10003), (), 0, 0)
    assert x == 0

  def testFoldIn(self):
    key = random.PRNGKey(0)
    keys = [random.fold_in(key, i) for i in range(10)]
    assert onp.unique(onp.ravel(keys)).shape == (20,)


if __name__ == "__main__":
  absltest.main()
