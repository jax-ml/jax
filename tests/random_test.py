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
from unittest import SkipTest, skipIf

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import scipy.linalg
import scipy.special
import scipy.stats

from jax._src import api
from jax import core
from jax import dtypes
from jax import grad
from jax import lax
from jax import numpy as jnp
from jax import prng
from jax import random
from jax import test_util as jtu
from jax import vmap
from jax.interpreters import xla
import jax._src.random

from jax.config import config
config.parse_flags_with_absl()

float_dtypes = jtu.dtypes.all_floating
complex_dtypes = jtu.dtypes.complex
int_dtypes = jtu.dtypes.all_integer
uint_dtypes = jtu.dtypes.all_unsigned


def _prng_key_as_array(key):
  # TODO(frostig): remove once we upgrade to always enable_custom_prng
  return key.keys if config.jax_enable_custom_prng else key


@jtu.with_config(jax_numpy_rank_promotion="raise")
class PrngTest(jtu.JaxTestCase):

  def testThreefry2x32(self):
    # We test the hash by comparing to known values provided in the test code of
    # the original reference implementation of Threefry. For the values, see
    # https://github.com/DEShawResearch/Random123-Boost/blob/65e3d874b67aa7b3e02d5ad8306462f52d2079c0/libs/random/test/test_threefry.cpp#L30-L32
    def result_to_hex(result):
      return tuple([hex(x.copy()).rstrip("L") for x in result])

    expected = ("0x6b200159", "0x99ba4efe")
    result = prng.threefry_2x32(np.uint32([0, 0]), np.uint32([0, 0]))

    self.assertEqual(expected, result_to_hex(result))

    expected = ("0x1cb996fc", "0xbb002be7")
    result = prng.threefry_2x32(np.uint32([-1, -1]), np.uint32([-1, -1]))
    self.assertEqual(expected, result_to_hex(result))

    expected = ("0xc4923a9c", "0x483df7a0")
    result = prng.threefry_2x32(
        np.uint32([0x13198a2e, 0x03707344]),
        np.uint32([0x243f6a88, 0x85a308d3]))
    self.assertEqual(expected, result_to_hex(result))

  def testThreefry2x32Large(self):
    n = 10000000
    result = prng.threefry_2x32(
      (np.uint32(0x13198a2e), np.uint32(0x03707344)),
      jnp.concatenate([
        jnp.full((n,), 0x243f6a88, jnp.uint32),
        jnp.full((n,), 0x85a308d3, jnp.uint32)
      ]))
    np.testing.assert_equal(result[:n], np.full((n,), 0xc4923a9c, dtype=np.uint32))
    np.testing.assert_equal(result[n:], np.full((n,), 0x483df7a0, dtype=np.uint32))

  def testThreefry2x32Empty(self):
    # Regression test for an op-by-op crash for empty arrays in CUDA mode.
    with api.disable_jit():
      result = prng.threefry_2x32(
        (np.uint32(0x13198a2e), np.uint32(0x03707344)),
        jnp.ones((10, 0,), jnp.uint32))
    np.testing.assert_equal(result, np.zeros((10, 0,), dtype=np.uint32))

  def testNoOpByOpUnderHash(self):
    def fail(*args, **kwargs): assert False
    apply_primitive, xla.apply_primitive = xla.apply_primitive, fail
    try:
      _ = prng.threefry_2x32(np.zeros(2, np.uint32), np.arange(10, dtype=np.uint32))
    finally:
      xla.apply_primitive = apply_primitive

  def testRngRandomBits(self):
    # Test specific outputs to ensure consistent random values between JAX versions.
    key = random.PRNGKey(1701)

    bits8 = jax._src.random._random_bits(key, 8, (3,))
    expected8 = np.array([216, 115,  43], dtype=np.uint8)
    self.assertArraysEqual(bits8, expected8)

    bits16 = jax._src.random._random_bits(key, 16, (3,))
    expected16 = np.array([41682,  1300, 55017], dtype=np.uint16)
    self.assertArraysEqual(bits16, expected16)

    bits32 = jax._src.random._random_bits(key, 32, (3,))
    expected32 = np.array([56197195, 4200222568, 961309823], dtype=np.uint32)
    self.assertArraysEqual(bits32, expected32)

    with jtu.ignore_warning(category=UserWarning, message="Explicitly requested dtype.*"):
      bits64 = jax._src.random._random_bits(key, 64, (3,))
    if config.x64_enabled:
      expected64 = np.array([3982329540505020460, 16822122385914693683,
                             7882654074788531506], dtype=np.uint64)
    else:
      expected64 = np.array([676898860, 3164047411, 4010691890], dtype=np.uint32)
    self.assertArraysEqual(bits64, expected64)

  def testRngRandomBitsViewProperty(self):
    # TODO: add 64-bit if it ever supports this property.
    # TODO: will this property hold across endian-ness?
    N = 10
    key = random.PRNGKey(1701)
    nbits = [8, 16, 32]
    rand_bits = [jax._src.random._random_bits(key, n, (N * 64 // n,))
                 for n in nbits]
    rand_bits_32 = np.array([np.array(r).view(np.uint32) for r in rand_bits])
    assert np.all(rand_bits_32 == rand_bits_32[0])

  def testPRNGValues(self):
    # Test to ensure consistent random values between JAX versions
    k = random.PRNGKey(0)

    if config.x64_enabled:
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
        _prng_key_as_array(random.split(k, 4)),
        np.array([[2285895361, 1501764800],
                   [1518642379, 4090693311],
                   [ 433833334, 4221794875],
                   [ 839183663, 3740430601]], dtype='uint32'))

    self.assertAllClose(
        _prng_key_as_array(random.fold_in(k, 4)),
        np.array([2285895361,  433833334], dtype='uint32'))

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "seed={seed}_type={type}_jit={jit}".format(**dct), **dct} for dct in [
      {"seed": 0, "type": int, "jit": True, "key": [0, 0]},
      {"seed": 0, "type": int, "jit": False, "key": [0, 0]},
      {"seed": 1, "type": np.int32, "jit": True, "key": [0, 1]},
      {"seed": 1, "type": np.int32, "jit": False, "key": [0, 1]},
      {"seed": 2, "type": np.uint32, "jit": True, "key": [0, 2]},
      {"seed": 2, "type": np.uint32, "jit": False, "key": [0, 2]},
      {"seed": 3, "type": np.int64, "jit": True, "key": [0, 3]},
      {"seed": 3, "type": np.int64, "jit": False, "key": [0, 3]},
      {"seed": -1, "type": int, "jit": True, "key": [4294967295, 4294967295] if config.x64_enabled else [0, 4294967295]},
      {"seed": -1, "type": int, "jit": False, "key": [4294967295, 4294967295] if config.x64_enabled else [0, 4294967295]},
      {"seed": -2, "type": np.int32, "jit": True, "key": [0, 4294967294]},
      {"seed": -2, "type": np.int32, "jit": False, "key": [0, 4294967294]},
      {"seed": -3, "type": np.int64, "jit": True, "key": [4294967295, 4294967293] if config.x64_enabled else [0, 4294967293]},
      {"seed": -3, "type": np.int64, "jit": False, "key": [4294967295, 4294967293] if config.x64_enabled else [0, 4294967293]},
      {"seed": np.iinfo(np.int32).max + 100, "type": int, "jit": True, "key": [0, 2147483747]},
      {"seed": np.iinfo(np.int32).max + 100, "type": int, "jit": False, "key": [0, 2147483747]},
      {"seed": np.iinfo(np.int32).max + 101, "type": np.uint32, "jit": True, "key": [0, 2147483748]},
      {"seed": np.iinfo(np.int32).max + 101, "type": np.uint32, "jit": False, "key": [0, 2147483748]},
      {"seed": np.iinfo(np.int32).min - 100, "type": int, "jit": True, "key": [4294967295, 2147483548] if config.x64_enabled else [0, 2147483548]},
      {"seed": np.iinfo(np.int32).min - 100, "type": int, "jit": False, "key": [4294967295, 2147483548] if config.x64_enabled else [0, 2147483548]},
      {"seed": np.iinfo(np.int32).min - 101, "type": np.int64, "jit": True, "key": [4294967295, 2147483547] if config.x64_enabled else [0, 2147483547]},
      {"seed": np.iinfo(np.int32).min - 101, "type": np.int64, "jit": False, "key": [4294967295, 2147483547] if config.x64_enabled else [0, 2147483547]},
    ]
  ))
  def test_prng_seeds_and_keys(self, seed, type, jit, key):
    if (jit and type is int and not config.x64_enabled and
        (seed < np.iinfo('int32').min or seed > np.iinfo('int32').max)):
      self.skipTest("Expected failure: integer out of range for jit.")
    seed = type(seed)
    if jit:
      actual = _prng_key_as_array(api.jit(random.PRNGKey)(seed))
    else:
      actual = _prng_key_as_array(random.PRNGKey(seed))
    expected = jnp.array(key, dtype=jnp.uint32)
    self.assertArraysEqual(actual, expected)


@jtu.with_config(jax_numpy_rank_promotion="raise")
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

    # scipy.stats.chisquare requires the sum of expected and actual to
    # match; this is only the case if we compute the expected frequency
    # at *all* nonzero values of the pmf. We don't know this a priori,
    # so we add extra values past the largest observed value. The number
    # below is empirically enough to get full coverage for the current set
    # of tests. If a new test is added where this is not enough, chisquare()
    # below will error due to the sums of the inputs not matching.
    extra_values = 100
    actual_freq = np.bincount(samples, minlength=samples.max() + extra_values)
    values = np.arange(len(actual_freq))

    expected_freq = pmf(values) * samples.size

    valid = expected_freq > 0
    actual_freq = actual_freq[valid]
    expected_freq = expected_freq[valid]

    _, p_value = scipy.stats.chisquare(actual_freq, expected_freq)
    self.assertGreater(
        p_value, alpha,
        msg=f'Failed chi-squared test with p={p_value}.\n'
            'Expected vs. actual frequencies:\n'
            f'{expected_freq}\n{actual_freq}')

  def seed_prng(self, seed):
    return random.PRNGKey(seed)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dtype={}".format(np.dtype(dtype).name), "dtype": dtype}
      for dtype in jtu.dtypes.floating))
  def testNumpyAndXLAAgreeOnFloatEndianness(self, dtype):
    bits_dtype = np.uint32 if jnp.finfo(dtype).bits == 32 else np.uint64
    numpy_bits = np.array(1., dtype).view(bits_dtype)
    xla_bits = api.jit(
        lambda: lax.bitcast_convert_type(np.array(1., dtype), bits_dtype))()
    self.assertEqual(numpy_bits, xla_bits)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dtype={}".format(np.dtype(dtype).name), "dtype": dtype}
      for dtype in float_dtypes))
  def testRngUniform(self, dtype):
    key = self.seed_prng(0)
    rand = lambda key: random.uniform(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckCollisions(samples, jnp.finfo(dtype).nmant)
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.uniform().cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dtype={}".format(np.dtype(dtype).name), "dtype": dtype}
      for dtype in int_dtypes + uint_dtypes))
  def testRngRandint(self, dtype):
    lo = 5
    hi = 10

    key = self.seed_prng(0)
    rand = lambda key: random.randint(key, (10000,), lo, hi, dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self.assertTrue(np.all(lo <= samples))
      self.assertTrue(np.all(samples < hi))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dtype={}".format(np.dtype(dtype).name), "dtype": dtype}
      for dtype in float_dtypes))
  def testNormal(self, dtype):
    key = self.seed_prng(0)
    rand = lambda key: random.normal(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.norm().cdf)

  def testNormalBfloat16(self):
    # Passing bfloat16 as dtype string.
    # https://github.com/google/jax/issues/6813
    res_bfloat16_str = random.normal(self.seed_prng(0), dtype='bfloat16')
    res_bfloat16 = random.normal(self.seed_prng(0), dtype=jnp.bfloat16)
    self.assertAllClose(res_bfloat16, res_bfloat16_str)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "dtype={}".format(np.dtype(dtype).name), "dtype": dtype}
      for dtype in complex_dtypes))
  def testNormalComplex(self, dtype):
    key = self.seed_prng(0)
    rand = lambda key: random.normal(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(jnp.real(samples), scipy.stats.norm(scale=1/np.sqrt(2)).cdf)
      self._CheckKolmogorovSmirnovCDF(jnp.imag(samples), scipy.stats.norm(scale=1/np.sqrt(2)).cdf)
      self.assertEqual(dtype, samples.dtype)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dtype={}".format(np.dtype(dtype).name), "dtype": dtype}
      for dtype in float_dtypes))
  def testTruncatedNormal(self, dtype):
    key = self.seed_prng(0)
    rand = lambda key: random.truncated_normal(key, -0.3, 0.3, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    min_val = np.min(uncompiled_samples)
    max_val = np.max(uncompiled_samples)
    self.assertTrue(min_val > -0.3)
    self.assertTrue(max_val < 0.3)
    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.truncnorm(-0.3, 0.3).cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dtype={}".format(np.dtype(dtype).name), "dtype": dtype}
      for dtype in jtu.dtypes.floating + jtu.dtypes.integer))
  def testShuffle(self, dtype):
    key = self.seed_prng(0)
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
      {"testcase_name": "_{}_shape={}_replace={}_weighted={}_array_input={}".format(
          np.dtype(dtype).name, shape, replace, weighted, array_input),
        "dtype": dtype, "shape": shape, "replace": replace,
        "weighted": weighted, "array_input": array_input}
      for dtype in jtu.dtypes.floating + jtu.dtypes.integer
      for shape in [(), (5,), (4, 5)]
      for replace in [True, False]
      for weighted in [True, False]
      for array_input in [False, 'jnp', 'np']))
  def testChoice(self, dtype, shape, replace, weighted, array_input):
    N = 100
    key = self.seed_prng(0)
    x = (N if not array_input else
         jnp.arange(N, dtype=dtype) if array_input == 'jnp' else
         np.arange(N, dtype=dtype))
    p = None if not weighted else jnp.arange(N)
    rand = lambda key: random.choice(key, x, shape, p=p, replace=replace)
    crand = api.jit(rand)

    sample1 = rand(key)
    sample2 = crand(key)

    self.assertEqual(shape, sample1.shape)
    if array_input == 'jnp':
      self.assertEqual(x.dtype, sample1.dtype)
    if not replace:
      assert len(np.unique(sample1)) == len(np.ravel(sample1))
    self.assertAllClose(sample1, sample2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "dtype": dtype, "shape": shape}
      for dtype in jtu.dtypes.floating + jtu.dtypes.integer
      for shape in [100, (10, 10), (10, 5, 2), 0, 1, (0, 5), (1, 5)]))
  def testPermutationArray(self, dtype, shape):
    key = self.seed_prng(0)
    x = jnp.arange(np.prod(shape)).reshape(shape).astype(dtype)
    rand = lambda key: random.permutation(key, x)
    crand = api.jit(rand)

    perm1 = rand(key)
    perm2 = crand(key)

    self.assertAllClose(perm1, perm2)
    if x.shape[0] > 1:
      self.assertFalse(np.all(perm1 == x))  # seems unlikely!
    self.assertAllClose(np.sort(perm1.ravel()), x.ravel(), check_dtypes=False)
    self.assertArraysAllClose(
      x, jnp.arange(np.prod(shape)).reshape(shape).astype(dtype))

  def testPermutationInteger(self):
    key = self.seed_prng(0)
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
    key = self.seed_prng(0)
    with self.assertRaises(TypeError):
      random.permutation(key, 10.)
    with self.assertRaises(core.ConcretizationTypeError):
      api.jit(random.permutation)(key, 10)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_p={}_dtype={}".format(p, np.dtype(dtype).name),
       "p": p, "dtype": dtype}
      for p in [0.1, 0.5, 0.9]
      for dtype in jtu.dtypes.floating))
  def testBernoulli(self, p, dtype):
    key = self.seed_prng(0)
    p = np.array(p, dtype=dtype)
    rand = lambda key, p: random.bernoulli(key, p, (10000,))
    crand = api.jit(rand)

    uncompiled_samples = rand(key, p)
    compiled_samples = crand(key, p)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckChiSquared(samples, scipy.stats.bernoulli(p).pmf)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_p={}_{}_{}".format(p, np.dtype(dtype).name, sample_shape),
     "p": p, "axis": axis, "dtype": dtype, 'sample_shape': sample_shape}
    for (p, axis) in [
        ([.25] * 4, -1),
        ([.1, .2, .3, .4], -1),
        ([[.5, .5], [.1, .9]], 1),
        ([[.5, .1], [.5, .9]], 0),
    ]
    for sample_shape in [(10000,), (5000, 2)]
    for dtype in jtu.dtypes.floating))
  def testCategorical(self, p, axis, dtype, sample_shape):
    key = self.seed_prng(0)
    p = np.array(p, dtype=dtype)
    logits = np.log(p) - 42 # test unnormalized
    out_shape = tuple(np.delete(logits.shape, axis))
    shape = sample_shape + out_shape
    rand = partial(random.categorical, shape=shape, axis=axis)
    crand = api.jit(rand)

    uncompiled_samples = rand(key, logits)
    compiled_samples = crand(key, logits)

    if axis < 0:
      axis += len(logits.shape)

    for samples in [uncompiled_samples, compiled_samples]:
      assert samples.shape == shape
      samples = jnp.reshape(samples, (10000,) + out_shape)
      if len(p.shape[:-1]) > 0:
        ps = np.transpose(p, (1, 0)) if axis == 0 else p
        for cat_samples, cat_p in zip(samples.transpose(), ps):
          pmf = lambda x: np.where(x < len(cat_p), cat_p[np.minimum(len(cat_p) - 1, x)], 0.0)
          self._CheckChiSquared(cat_samples, pmf=pmf)
      else:
        pmf = lambda x: np.where(x < len(p), p[np.minimum(len(p) - 1, x)], 0.0)
        self._CheckChiSquared(samples, pmf=pmf)

  def testBernoulliShape(self):
    key = self.seed_prng(0)
    with jax.numpy_rank_promotion('allow'):
      x = random.bernoulli(key, np.array([0.2, 0.3]), shape=(3, 2))
    assert x.shape == (3, 2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_a={}_b={}_dtype={}".format(a, b, np.dtype(dtype).name),
       "a": a, "b": b, "dtype": dtype}
      for a in [0.2, 5.]
      for b in [0.2, 5.]
      for dtype in [np.float64]))  # NOTE: KS test fails with float32
  def testBeta(self, a, b, dtype):
    if not config.x64_enabled:
      raise SkipTest("skip test except on X64")
    key = self.seed_prng(0)
    rand = lambda key, a, b: random.beta(key, a, b, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key, a, b)
    compiled_samples = crand(key, a, b)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.beta(a, b).cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dtype={}".format(np.dtype(dtype).name), "dtype": dtype}
      for dtype in float_dtypes))
  def testCauchy(self, dtype):
    key = self.seed_prng(0)
    rand = lambda key: random.cauchy(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.cauchy().cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_alpha={}_dtype={}".format(alpha, np.dtype(dtype).name),
       "alpha": alpha, "dtype": dtype}
      for alpha in [
          np.array([0.2, 1., 5.]),
      ]
      for dtype in jtu.dtypes.floating))
  @jtu.skip_on_devices("tpu")  # TODO(mattjj): slow compilation times
  def testDirichlet(self, alpha, dtype):
    key = self.seed_prng(0)
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
      {"testcase_name": "_dtype={}".format(np.dtype(dtype).name), "dtype": dtype}
      for dtype in float_dtypes))
  def testExponential(self, dtype):
    key = self.seed_prng(0)
    rand = lambda key: random.exponential(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.expon().cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_a={}_dtype={}".format(a, np.dtype(dtype).name),
       "a": a, "dtype": dtype}
      for a in [0.1, 1., 10.]
      for dtype in jtu.dtypes.floating))
  def testGamma(self, a, dtype):
    key = self.seed_prng(0)
    rand = lambda key, a: random.gamma(key, a, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key, a)
    compiled_samples = crand(key, a)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.gamma(a).cdf)

  def testGammaShape(self):
    key = self.seed_prng(0)
    x = random.gamma(key, np.array([0.2, 0.3]), shape=(3, 2))
    assert x.shape == (3, 2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_a={}".format(alpha), "alpha": alpha}
      for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]))
  def testGammaGrad(self, alpha):
    rng = self.seed_prng(0)
    alphas = np.full((100,), alpha)
    z = random.gamma(rng, alphas)
    actual_grad = api.grad(lambda x: random.gamma(rng, x).sum())(alphas)

    eps = 0.01 * alpha / (1.0 + np.sqrt(alpha))
    cdf_dot = (scipy.stats.gamma.cdf(z, alpha + eps)
               - scipy.stats.gamma.cdf(z, alpha - eps)) / (2 * eps)
    with np.errstate(over='ignore'):
      pdf = scipy.stats.gamma.pdf(z, alpha)
    expected_grad = -cdf_dot / pdf

    self.assertAllClose(actual_grad, expected_grad, check_dtypes=True,
                        rtol=2e-2 if jtu.device_under_test() == "tpu" else 7e-4)

  def testGammaGradType(self):
    # Regression test for https://github.com/google/jax/issues/2130
    key = self.seed_prng(0)
    a = jnp.array(1., dtype=jnp.float32)
    b = jnp.array(3., dtype=jnp.float32)
    f = lambda x, y: random.gamma(key=key, a=x, dtype=jnp.float32) / y
    # Should not crash with a type error.
    api.vjp(f, a, b)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_lam={}_dtype={}".format(lam, np.dtype(dtype).name),
       "lam": lam, "dtype": np.dtype(dtype)}
      for lam in [0.5, 3, 9, 11, 50, 500]
      for dtype in [np.int16, np.int32, np.int64]))
  def testPoisson(self, lam, dtype):
    key = self.seed_prng(0)
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
    key = self.seed_prng(1)
    lam = jnp.concatenate([2 * jnp.ones(10000), 20 * jnp.ones(10000)])
    samples = random.poisson(key, lam, shape=(20000,))
    self._CheckChiSquared(samples[:10000], scipy.stats.poisson(2.0).pmf)
    self._CheckChiSquared(samples[10000:], scipy.stats.poisson(20.0).pmf)

  def testPoissonShape(self):
    key = self.seed_prng(0)
    x = random.poisson(key, np.array([2.0, 20.0]), shape=(3, 2))
    assert x.shape == (3, 2)

  def testPoissonZeros(self):
    key = self.seed_prng(0)
    lam = jnp.concatenate([jnp.zeros(10), 20 * jnp.ones(10)])
    samples = random.poisson(key, lam, shape=(2, 20))
    self.assertArraysEqual(samples[:, :10], jnp.zeros_like(samples[:, :10]))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dtype={}".format(np.dtype(dtype).name), "dtype": dtype}
      for dtype in jtu.dtypes.floating))
  def testGumbel(self, dtype):
    key = self.seed_prng(0)
    rand = lambda key: random.gumbel(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.gumbel_r().cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dtype={}".format(np.dtype(dtype).name), "dtype": dtype}
      for dtype in float_dtypes))
  def testLaplace(self, dtype):
    key = self.seed_prng(0)
    rand = lambda key: random.laplace(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.laplace().cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dtype={}".format(np.dtype(dtype).name), "dtype": dtype}
      for dtype in float_dtypes))
  def testLogistic(self, dtype):
    key = self.seed_prng(0)
    rand = lambda key: random.logistic(key, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.logistic().cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_b={}_dtype={}".format(b, np.dtype(dtype).name),
       "b": b, "dtype": dtype}
      for b in [0.1, 1., 10.]
      for dtype in jtu.dtypes.floating))
  def testPareto(self, b, dtype):
    key = self.seed_prng(0)
    rand = lambda key, b: random.pareto(key, b, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key, b)
    compiled_samples = crand(key, b)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.pareto(b).cdf)

  def testParetoShape(self):
    key = self.seed_prng(0)
    with jax.numpy_rank_promotion('allow'):
      x = random.pareto(key, np.array([0.2, 0.3]), shape=(3, 2))
    assert x.shape == (3, 2)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_df={}_dtype={}".format(df, np.dtype(dtype).name),
       "df": df, "dtype": dtype}
      for df in [0.1, 1., 10.]
      for dtype in jtu.dtypes.floating))
  @jtu.skip_on_devices("cpu", "tpu")  # TODO(phawkins): slow compilation times
  def testT(self, df, dtype):
    key = self.seed_prng(0)
    rand = lambda key, df: random.t(key, df, (10000,), dtype)
    crand = api.jit(rand)

    uncompiled_samples = rand(key, df)
    compiled_samples = crand(key, df)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.t(df).cdf)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dim={}_dtype={}_method={}".format(
          dim, np.dtype(dtype), method),
       "dim": dim, "dtype": dtype, "method": method}
      for dim in [1, 3, 5]
      for dtype in float_dtypes
      for method in ['svd', 'eigh', 'cholesky']))
  def testMultivariateNormal(self, dim, dtype, method):
    r = np.random.RandomState(dim)
    mean = r.randn(dim)
    cov_factor = r.randn(dim, dim)
    cov = np.dot(cov_factor, cov_factor.T) + dim * np.eye(dim)

    key = self.seed_prng(0)
    rand = partial(random.multivariate_normal, mean=mean, cov=cov,
                   shape=(10000,), method=method)
    crand = api.jit(rand)

    with jax.numpy_rank_promotion('allow'):
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

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dim={}_mean_batch_size={}_cov_batch_size={}_shape={}"\
       .format(dim, mean_batch_size, cov_batch_size, shape),
       "dim": dim,
       "mean_batch_size": mean_batch_size,
       "cov_batch_size": cov_batch_size,
       "shape": shape}
      for dim in [1, 2, 4]
      for mean_batch_size in [(), (3,), (2, 3)]
      for cov_batch_size in [(), (3,), (2, 3)]
      for shape in [(), (1,), (5,)]))
  def testMultivariateNormalShapes(self, dim, mean_batch_size, cov_batch_size,
                                   shape):
    r = np.random.RandomState(0)
    key = self.seed_prng(0)
    eff_batch_size = mean_batch_size \
      if len(mean_batch_size) > len(cov_batch_size) else cov_batch_size
    mean = r.randn(*(mean_batch_size + (dim,)))
    cov_factor = r.randn(*(cov_batch_size + (dim, dim)))
    cov = np.einsum('...ij,...kj->...ik', cov_factor, cov_factor)
    cov += 1e-3 * np.eye(dim)
    shape = shape + eff_batch_size
    with jax.numpy_rank_promotion('allow'):
      samples = random.multivariate_normal(key, mean, cov, shape=shape)
    assert samples.shape == shape + (dim,)

  def testMultivariateNormalCovariance(self):
    # test code based on https://github.com/google/jax/issues/1869
    N = 100000
    cov = jnp.array([[ 0.19,  0.00, -0.13,  0.00],
                   [  0.00,  0.29,  0.00, -0.23],
                   [ -0.13,  0.00,  0.39,  0.00],
                   [  0.00, -0.23,  0.00,  0.49]])
    mean = jnp.zeros(4)

    out_np = np.random.RandomState(0).multivariate_normal(mean, cov, N)

    key = self.seed_prng(0)
    with jax.numpy_rank_promotion('allow'):
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
    x = random.randint(self.seed_prng(10003), (), 0, 0)
    assert x == 0

  def testFoldIn(self):
    key = self.seed_prng(0)
    keys = [_prng_key_as_array(random.fold_in(key, i)) for i in range(10)]
    assert np.unique(keys, axis=0).shape[0] == 10

  def testFoldInBig(self):
    key = self.seed_prng(0)
    seeds = [2 ** 32 - 2, 2 ** 32 - 1]
    keys = [_prng_key_as_array(random.fold_in(key, seed)) for seed in seeds]
    assert np.unique(keys, axis=0).shape[0] == 2

  def testStaticShapeErrors(self):
    if config.jax_disable_jit:
      raise SkipTest("test only relevant when jit enabled")

    @api.jit
    def feature_map(n, d, sigma=1.0, seed=123):
      key = self.seed_prng(seed)
      W = random.normal(key, (d, n)) / sigma
      w = random.normal(key, (d, )) / sigma
      b = 2 * jnp.pi * random.uniform(key, (d, ))

      phi = lambda x, t: jnp.sqrt(2.0 / d) * jnp.cos(jnp.matmul(W, x) + w*t + b)
      return phi

    self.assertRaisesRegex(TypeError, 'Shapes must be 1D.*',
                           lambda: feature_map(5, 3))

  def testIssue756(self):
    key = self.seed_prng(0)
    w = random.normal(key, ())
    if config.x64_enabled:
      self.assertEqual(np.result_type(w), np.float64)
    else:
      self.assertEqual(np.result_type(w), np.float32)

  def testIssue1789(self):
    def f(x):
      return random.gamma(self.seed_prng(0), x)

    grad(lambda x: jnp.sum(vmap(f)(x)))(jnp.ones(2))

  def testDtypeErrorMessage(self):
    with self.assertRaisesRegex(ValueError, r"dtype argument to.*"):
      random.normal(self.seed_prng(0), (), dtype=jnp.int32)

  def testRandomBroadcast(self):
    """Issue 4033"""
    # test for broadcast issue in https://github.com/google/jax/issues/4033
    key = self.seed_prng(0)
    shape = (10, 2)
    with jax.numpy_rank_promotion('allow'):
      x1 = random.uniform(key, shape, minval=jnp.zeros(2), maxval=jnp.ones(2))
      x2 = random.randint(key, shape, jnp.array([0, 1]), jnp.array([1, 2]))
    assert x1.shape == shape
    assert x2.shape == shape

  def testMaxwellSample(self):
    num_samples = 10**5
    rng = self.seed_prng(0)

    rand = lambda x: random.maxwell(x, (num_samples, ))
    crand = api.jit(rand)

    loc = scipy.stats.maxwell.mean()
    std = scipy.stats.maxwell.std()

    uncompiled_samples = rand(rng)
    compiled_samples = crand(rng)

    for samples in [uncompiled_samples, compiled_samples]:
      # Check first and second moments.
      self.assertEqual((num_samples,), samples.shape)
      self.assertAllClose(np.mean(samples), loc, atol=0., rtol=0.1)
      self.assertAllClose(np.std(samples), std, atol=0., rtol=0.1)
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.maxwell().cdf)

  @parameterized.named_parameters(
      ('test1', 4.0, 1.0),
      ('test2', 2.0, 3.0))
  def testWeibullSample(self, concentration, scale):
    num_samples = 10**5
    rng = self.seed_prng(0)

    rand = lambda x: random.weibull_min(x, scale, concentration, (num_samples,))
    crand = api.jit(rand)

    loc = scipy.stats.weibull_min.mean(c=concentration, scale=scale)
    std = scipy.stats.weibull_min.std(c=concentration, scale=scale)

    uncompiled_samples = rand(rng)
    compiled_samples = crand(rng)

    for samples in [uncompiled_samples, compiled_samples]:
      # Check first and second moments.
      self.assertEqual((num_samples,), samples.shape)
      self.assertAllClose(np.mean(samples), loc, atol=0., rtol=0.1)
      self.assertAllClose(np.std(samples), std, atol=0., rtol=0.1)
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.weibull_min(
          c=concentration, scale=scale).cdf)

  @parameterized.named_parameters(
      ('test1', 4.0, 1.0),
      ('test2', 2.0, 3.0))
  def testDoublesidedMaxwellSample(self, loc, scale):
    num_samples = 10**5
    rng = self.seed_prng(0)

    rand = lambda key: random.double_sided_maxwell(
        rng, loc, scale, (num_samples,))
    crand = api.jit(rand)

    mean = loc
    std = np.sqrt(3.) * scale

    uncompiled_samples = rand(rng)
    compiled_samples = crand(rng)

    # Compute the double sided maxwell CDF through the one sided maxwell cdf.
    # This is done as follows:
    # P(DSM <= x) = P (loc + scale * radamacher_sample * one_sided_sample <=x) =
    # P (radamacher_sample * one_sided_sample <= (x - loc) / scale) =
    # 1/2 P(one_sided_sample <= (x - loc) / scale)
    #    + 1/2 P( - one_sided_sample <= (x - loc) / scale) =
    #  1/2 P(one_sided_sample <= (x - loc) / scale)
    #    + 1/2 P(one_sided_sample >= - (x - loc) / scale) =
    # 1/2 CDF_one_maxwell((x - loc) / scale))
    #   + 1/2 (1 - CDF_one_maxwell(- (x - loc) / scale)))
    def double_sided_maxwell_cdf(x, loc, scale):
      pos = scipy.stats.maxwell().cdf((x - loc) / scale)
      neg = (1 - scipy.stats.maxwell().cdf((-x + loc) / scale))
      return (pos + neg) / 2

    for samples in [uncompiled_samples, compiled_samples]:
      # Check first and second moments.
      self.assertEqual((num_samples,), samples.shape)
      self.assertAllClose(np.mean(samples), mean, atol=0., rtol=0.1)
      self.assertAllClose(np.std(samples), std, atol=0., rtol=0.1)

      self._CheckKolmogorovSmirnovCDF(
          samples, lambda x: double_sided_maxwell_cdf(x, loc, scale))

  def testRadamacher(self):
    rng = self.seed_prng(0)
    num_samples = 10**5

    rand = lambda x: random.rademacher(x, (num_samples,))
    crand = api.jit(rand)

    uncompiled_samples = rand(rng)
    compiled_samples = crand(rng)

    for samples in [uncompiled_samples, compiled_samples]:
      unique_values, counts = np.unique(samples, return_counts=True)
      assert len(unique_values) == 2
      assert len(counts) == 2

      self.assertAllClose(
          counts[0] / num_samples, 0.5, rtol=1e-02, atol=1e-02)
      self.assertAllClose(
          counts[1] / num_samples, 0.5, rtol=1e-02, atol=1e-02)

  def testChoiceShapeIsNotSequenceError(self):
    key = self.seed_prng(0)
    with self.assertRaises(TypeError):
      random.choice(key, 5, 2, replace=False)
    with self.assertRaises(TypeError):
      random.choice(key, 5, 2, replace=True)

  def test_eval_shape_big_random_array(self):
    def f(x):
      return random.normal(self.seed_prng(x), (int(1e12),))
    with jax.enable_checks(False):  # check_jaxpr will materialize array
      api.eval_shape(f, 0)  # doesn't error

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_seed={seed}_type={type}", "seed": seed, "type": type}
      for type in ["int", "np.array", "jnp.array"]
      for seed in [-1, 0, 1, (1 << 32) - 1, (1 << 63) - 1, np.uint64((1 << 64) - 1)]))
  def test_prng_jit_invariance(self, seed, type):
    if type == "int" and seed == (1 << 64) - 1:
      self.skipTest("Expected failure: Python int too large.")
    if not config.x64_enabled and seed > np.iinfo(np.int32).max:
      self.skipTest("Expected failure: Python int too large.")
    type = {"int": int, "np.array": np.array, "jnp.array": jnp.array}[type]
    args_maker = lambda: [type(seed)]
    make_prng = lambda seed: _prng_key_as_array(self.seed_prng(seed))
    self._CompileAndCheck(make_prng, args_maker)

  def test_prng_errors(self):
    seed = np.iinfo(np.int64).max + 1
    with self.assertRaises(OverflowError):
      self.seed_prng(seed)
    with self.assertRaises(OverflowError):
      api.jit(self.seed_prng)(seed)

  def test_random_split_doesnt_device_put_during_tracing(self):
    key = _prng_key_as_array(self.seed_prng(1)).block_until_ready()
    with jtu.count_device_put() as count:
      api.jit(random.split)(key)
    self.assertEqual(count[0], 1)  # 1 for the argument device_put

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_dtype={dtype}", "dtype": dtype}
      for dtype in int_dtypes + uint_dtypes))
  def test_randint_bounds(self, dtype):
    min = np.iinfo(dtype).min
    max = np.iinfo(dtype).max
    key = self.seed_prng(1701)
    shape = (10,)
    if np.iinfo(dtype).bits < np.iinfo(dtypes.canonicalize_dtype(int)).bits:
      expected = random.randint(key, shape, min, max, dtype)
      self.assertArraysEqual(expected, random.randint(key, shape, min - 12345, max + 12345, dtype))
    else:
      self.assertRaises(OverflowError, random.randint, key, shape, min - 12345, max + 12345, dtype)

  def test_randint_out_of_range(self):
    key = self.seed_prng(0)

    r = random.randint(key, (10,), 255, 256, np.uint8)
    self.assertAllClose(r, jnp.full_like(r, 255))

    r = random.randint(key, (1000,), -128, 128, np.int8)
    self.assertGreater((r == -128).sum(), 0)
    self.assertGreater((r == 127).sum(), 0)

    r = random.randint(key, (1000,), -1000, 1000, np.uint8)
    self.assertGreater((r == 0).sum(), 0)
    self.assertGreater((r == 255).sum(), 0)


threefry_seed = jax._src.prng.threefry_seed
threefry_split = jax._src.prng.threefry_split
threefry_random_bits = jax._src.prng.threefry_random_bits
threefry_fold_in = jax._src.prng.threefry_fold_in

def _double_threefry_seed(seed):
  return jnp.vstack([threefry_seed(seed + 1),
                     threefry_seed(seed + 2)])

def _double_threefry_split(key, num):
  split0 = threefry_split(key[0], num)
  split1 = threefry_split(key[1], num)
  merge = jnp.vstack([jnp.expand_dims(split0.T, axis=0),
                      jnp.expand_dims(split1.T, axis=0)])
  return merge.transpose((2, 0, 1))

def _double_threefry_random_bits(key, bit_width, shape):
  bits0 = threefry_random_bits(key[0], bit_width, shape)
  bits1 = threefry_random_bits(key[1], bit_width, shape)
  return bits0 * bits1

def _double_threefry_fold_in(key, data):
  return jnp.vstack([threefry_fold_in(key[0], data),
                     threefry_fold_in(key[1], data)])

double_threefry_prng_impl = prng.PRNGImpl(
    key_shape=(2, 2),
    seed=_double_threefry_seed,
    split=_double_threefry_split,
    random_bits=_double_threefry_random_bits,
    fold_in=_double_threefry_fold_in)

@skipIf(not config.jax_enable_custom_prng,
        'custom PRNG tests require config.jax_enable_custom_prng')
@jtu.with_config(jax_numpy_rank_promotion="raise")
class LaxRandomWithCustomPRNGTest(LaxRandomTest):
  def seed_prng(self, seed):
    return prng.seed_with_impl(double_threefry_prng_impl, seed)

  def test_split_shape(self):
    key = self.seed_prng(73)
    keys = random.split(key, 10)
    self.assertEqual(keys.shape, (10,))

  def test_vmap_fold_in_shape(self):
    key = self.seed_prng(73)
    keys = vmap(lambda i: random.fold_in(key, i))(jnp.arange(3))
    self.assertEqual(keys.shape, (3,))

  def test_cannot_add(self):
    key = self.seed_prng(73)
    self.assertRaisesRegex(
        TypeError, r'unsupported operand type\(s\) for \+*',
        lambda: key + 47)

def _sampler_unimplemented_with_custom_prng(*args, **kwargs):
  raise SkipTest('sampler only implemented for default RNG')

for test_prefix in [
    'testBeta',
    'testDirichlet',
    'testGamma',
    'testGammaGrad',
    'testGammaGradType',
    'testGammaShape',
    'testIssue1789',
    'testPoisson',
    'testPoissonBatched',
    'testPoissonShape',
    'testPoissonZeros',
]:
  for attr in dir(LaxRandomTest):
    if attr.startswith(test_prefix):
      setattr(LaxRandomWithCustomPRNGTest, attr,
              _sampler_unimplemented_with_custom_prng)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
