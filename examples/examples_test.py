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


import os
import sys

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from jax import test_util as jtu
from jax import random
import jax.numpy as jnp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples import kernel_lsq
from examples import resnet50
sys.path.pop()

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS


def _CheckShapeAgreement(test_case, init_fun, apply_fun, input_shape):
  jax_rng = random.PRNGKey(0)
  result_shape, params = init_fun(jax_rng, input_shape)
  rng = np.random.RandomState(0)
  result = apply_fun(params, rng.randn(*input_shape).astype(dtype="float32"))
  test_case.assertEqual(result.shape, result_shape)


class ExamplesTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      {"testcase_name": "_input_shape={}".format(input_shape),
       "input_shape": input_shape}
      for input_shape in [(2, 20, 25, 2)])
  @jtu.skip_on_flag('jax_enable_x64', True)
  def testIdentityBlockShape(self, input_shape):
    init_fun, apply_fun = resnet50.IdentityBlock(2, (4, 3))
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shape)

  @parameterized.named_parameters(
      {"testcase_name": "_input_shape={}".format(input_shape),
       "input_shape": input_shape}
      for input_shape in [(2, 20, 25, 3)])
  @jtu.skip_on_flag('jax_enable_x64', True)
  def testConvBlockShape(self, input_shape):
    init_fun, apply_fun = resnet50.ConvBlock(3, (2, 3, 4))
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shape)

  @parameterized.named_parameters(
      {"testcase_name": "_num_classes={}_input_shape={}"
                        .format(num_classes, input_shape),
       "num_classes": num_classes, "input_shape": input_shape}
      for num_classes in [5, 10]
      for input_shape in [(224, 224, 3, 2)])
  @jtu.skip_on_flag('jax_enable_x64', True)
  def testResNet50Shape(self, num_classes, input_shape):
    init_fun, apply_fun = resnet50.ResNet50(num_classes)
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shape)

  def testKernelRegressionGram(self):
    n, d = 100, 20
    rng = np.random.RandomState(0)
    truth = rng.randn(d)
    xs = rng.randn(n, d)
    ys = jnp.dot(xs, truth)
    kernel = lambda x, y: jnp.dot(x, y)
    self.assertAllClose(kernel_lsq.gram(kernel, xs), jnp.dot(xs, xs.T),
                        check_dtypes=False)

  def testKernelRegressionTrainAndPredict(self):
    n, d = 100, 20
    rng = np.random.RandomState(0)
    truth = rng.randn(d)
    xs = rng.randn(n, d)
    ys = jnp.dot(xs, truth)
    kernel = lambda x, y: jnp.dot(x, y)
    predict = kernel_lsq.train(kernel, xs, ys)
    self.assertAllClose(predict(xs), ys, atol=1e-3, rtol=1e-3,
                        check_dtypes=False)


if __name__ == "__main__":
  absltest.main()
