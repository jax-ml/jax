# Copyright 2019 Google LLC
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

"""Tests for nn module."""

import collections
import itertools

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from jax import core
from jax import test_util as jtu
from jax.test_util import check_grads
from jax import nn
from jax import random
import jax
import jax.numpy as jnp

from jax.config import config
config.parse_flags_with_absl()


class NNFunctionsTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    config.update("jax_numpy_rank_promotion", "raise")

  def tearDown(self):
    super().tearDown()
    config.update("jax_numpy_rank_promotion", "allow")

  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def testSoftplusGrad(self):
    check_grads(nn.softplus, (1e-8,), order=4,
                rtol=1e-2 if jtu.device_under_test() == "tpu" else None)

  def testSoftplusGradZero(self):
    check_grads(nn.softplus, (0.,), order=1,
                rtol=1e-2 if jtu.device_under_test() == "tpu" else None)

  def testSoftplusGradInf(self):
    self.assertAllClose(
        1., jax.grad(nn.softplus)(float('inf')))

  def testSoftplusGradNegInf(self):
    check_grads(nn.softplus, (-float('inf'),), order=1,
                rtol=1e-2 if jtu.device_under_test() == "tpu" else None)

  def testSoftplusGradNan(self):
    check_grads(nn.softplus, (float('nan'),), order=1,
                rtol=1e-2 if jtu.device_under_test() == "tpu" else None)

  @parameterized.parameters([
      int, jnp.int32, float, jnp.float64, jnp.float32, jnp.float64,])
  def testSoftplusZero(self, dtype):
    self.assertEqual(jnp.log(dtype(2)), nn.softplus(dtype(0)))

  def testReluGrad(self):
    rtol = 1e-2 if jtu.device_under_test() == "tpu" else None
    check_grads(nn.relu, (1.,), order=3, rtol=rtol)
    check_grads(nn.relu, (-1.,), order=3, rtol=rtol)
    jaxpr = jax.make_jaxpr(jax.grad(nn.relu))(0.)
    self.assertGreaterEqual(len(jaxpr.jaxpr.eqns), 2)

  def testSoftplusValue(self):
    val = nn.softplus(89.)
    self.assertAllClose(val, 89., check_dtypes=False)

  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def testEluGrad(self):
    check_grads(nn.elu, (1e4,), order=4, eps=1.)

  def testEluValue(self):
    val = nn.elu(1e4)
    self.assertAllClose(val, 1e4, check_dtypes=False)

  def testGluValue(self):
    val = nn.glu(jnp.array([1.0, 0.0]))
    self.assertAllClose(val, jnp.array([0.5]))

  @parameterized.parameters(*itertools.product(
      (jnp.float32, jnp.bfloat16, jnp.float16),
      (nn.gelu, nn.relu, nn.softplus, nn.sigmoid)))
  def testDtypeMatchesInput(self, dtype, fn):
    if dtype is jnp.float16 and jtu.device_under_test() == "tpu":
      self.skipTest("float16 not supported on TPU")
    x = jnp.zeros((), dtype=dtype)
    out = fn(x)
    self.assertEqual(out.dtype, dtype)

  @jtu.skip_on_devices("gpu", "tpu")
  def testEluMemory(self):
    # see https://github.com/google/jax/pull/1640
    with core.skipping_checks():  # With checks we materialize the array
      jax.make_jaxpr(nn.elu)(jnp.ones((10 ** 12,)))  # don't oom

  @jtu.skip_on_devices("gpu", "tpu")
  def testHardTanhMemory(self):
    # see https://github.com/google/jax/pull/1640
    with core.skipping_checks():  # With checks we materialize the array
      jax.make_jaxpr(nn.hard_tanh)(jnp.ones((10 ** 12,)))  # don't oom

  def testOneHot(self):
    actual = nn.one_hot(jnp.array([0, 1, 2]), 3)
    expected = jnp.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]])
    self.assertAllClose(actual, expected)

    actual = nn.one_hot(jnp.array([1, 2, 0]), 3)
    expected = jnp.array([[0., 1., 0.],
                         [0., 0., 1.],
                         [1., 0., 0.]])
    self.assertAllClose(actual, expected)

  def testOneHotOutOfBound(self):
    actual = nn.one_hot(jnp.array([-1, 3]), 3)
    expected = jnp.array([[0., 0., 0.],
                         [0., 0., 0.]])
    self.assertAllClose(actual, expected)

  def testOneHotNonArrayInput(self):
    actual = nn.one_hot([0, 1, 2], 3)
    expected = jnp.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]])
    self.assertAllClose(actual, expected)

  def testOneHotCustomDtype(self):
    actual = nn.one_hot(jnp.array([0, 1, 2]), 3, dtype=jnp.bool_)
    expected = jnp.array([[True, False, False],
                         [False, True, False],
                         [False, False, True]])
    self.assertAllClose(actual, expected)

InitializerRecord = collections.namedtuple(
  "InitializerRecord",
  ["name", "initializer", "shapes"])

ALL_SHAPES = [(2,), (2, 2), (2, 3), (3, 2), (2, 3, 4), (4, 3, 2), (2, 3, 4, 5)]

def initializer_record(name, initializer, min_dims=2, max_dims=4):
  shapes = [shape for shape in ALL_SHAPES
            if min_dims <= len(shape) <= max_dims]
  return InitializerRecord(name, initializer, shapes)

INITIALIZER_RECS = [
    initializer_record("uniform", nn.initializers.uniform, 1),
    initializer_record("normal", nn.initializers.normal, 1),
    initializer_record("he_normal", nn.initializers.he_normal),
    initializer_record("he_uniform", nn.initializers.he_uniform),
    initializer_record("glorot_normal", nn.initializers.glorot_normal),
    initializer_record("glorot_uniform", nn.initializers.glorot_uniform),
    initializer_record("lecun_normal", nn.initializers.lecun_normal),
    initializer_record("lecun_uniform", nn.initializers.lecun_uniform),
    initializer_record("orthogonal", nn.initializers.orthogonal, 2, 2),
    initializer_record("delta_orthogonal", nn.initializers.delta_orthogonal, 4, 4)
]

class NNInitializersTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    config.update("jax_numpy_rank_promotion", "raise")

  def tearDown(self):
    super().tearDown()
    config.update("jax_numpy_rank_promotion", "allow")

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_{}_{}".format(
           rec.name,
           jtu.format_shape_dtype_string(shape, dtype)),
       "initializer": rec.initializer(),
       "shape": shape, "dtype": dtype}
      for rec in INITIALIZER_RECS
      for shape in rec.shapes
      for dtype in [np.float32, np.float64]))
  def testInitializer(self, initializer, shape, dtype):
    rng = random.PRNGKey(0)
    val = initializer(rng, shape, dtype)
    self.assertEqual(shape, jnp.shape(val))
    self.assertEqual(jax.dtypes.canonicalize_dtype(dtype), jnp.dtype(val))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_{}_{}".format(
           rec.name,
           jtu.format_shape_dtype_string(shape, dtype)),
       "initializer_provider": rec.initializer,
       "shape": shape, "dtype": dtype}
      for rec in INITIALIZER_RECS
      for shape in rec.shapes
      for dtype in [np.float32, np.float64]))
  def testInitializerProvider(self, initializer_provider, shape, dtype):
    rng = random.PRNGKey(0)
    initializer = initializer_provider(dtype=dtype)
    val = initializer(rng, shape)

    self.assertEqual(shape, jnp.shape(val))
    self.assertEqual(jax.dtypes.canonicalize_dtype(dtype), jnp.dtype(val))


if __name__ == "__main__":
  absltest.main()
