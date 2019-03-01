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

"""Tests for Stax library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp

from jax import test_util as jtu
from jax import random
from jax.experimental import stax

from jax.config import config
config.parse_flags_with_absl()


def random_inputs(rng, input_shape):
  if type(input_shape) is tuple:
    return rng.randn(*input_shape).astype(onp.float32)
  elif type(input_shape) is list:
    return [random_inputs(rng, shape) for shape in input_shape]
  else:
    raise TypeError(type(input_shape))


def _CheckShapeAgreement(test_case, init_fun, apply_fun, input_shape):
  result_shape, params = init_fun(input_shape)
  inputs = random_inputs(onp.random.RandomState(0), input_shape)
  rng_key = random.PRNGKey(0)
  result = apply_fun(params, inputs, rng=rng_key)
  test_case.assertEqual(result.shape, result_shape)


class StaxTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(shape), "shape": shape}
      for shape in [(2, 3), (5,)]))
  def testRandnInitShape(self, shape):
    out = stax.randn()(shape)
    self.assertEqual(out.shape, shape)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(shape), "shape": shape}
      for shape in [(2, 3), (2, 3, 4)]))
  def testGlorotInitShape(self, shape):
    out = stax.glorot()(shape)
    self.assertEqual(out.shape, shape)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_channels={}_filter_shape={}_padding={}_strides={}_input_shape={}"
       .format(channels, filter_shape, padding, strides, input_shape),
       "channels": channels, "filter_shape": filter_shape, "padding": padding,
       "strides": strides, "input_shape": input_shape}
      for channels in [2, 3]
      for filter_shape in [(1, 1), (2, 3)]
      for padding in ["SAME", "VALID"]
      for strides in [None, (2, 1)]
      for input_shape in [(2, 10, 11, 1)]))
  def testConvShape(self, channels, filter_shape, padding, strides,
                    input_shape):
    init_fun, apply_fun = stax.Conv(channels, filter_shape, strides=strides,
                                    padding=padding)
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shape)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_out_dim={}_input_shape={}"
                        .format(out_dim, input_shape),
       "out_dim": out_dim, "input_shape": input_shape}
      for out_dim in [3, 4]
      for input_shape in [(2, 3), (3, 4)]))
  def testDenseShape(self, out_dim, input_shape):
    init_fun, apply_fun = stax.Dense(out_dim)
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shape)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_input_shape={}".format(input_shape),
       "input_shape": input_shape}
      for input_shape in [(2, 3), (2, 3, 4)]))
  def testReluShape(self, input_shape):
    init_fun, apply_fun = stax.Relu
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shape)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_window_shape={}_padding={}_strides={}_input_shape={}"
                        "_maxpool={}"
                        .format(window_shape, padding, strides, input_shape,
                                max_pool),
       "window_shape": window_shape, "padding": padding, "strides": strides,
       "input_shape": input_shape, "max_pool": max_pool}
      for window_shape in [(1, 1), (2, 3)]
      for padding in ["VALID"]
      for strides in [None, (2, 1)]
      for input_shape in [(2, 5, 6, 1)]
      for max_pool in [False, True]))
  def testPoolingShape(self, window_shape, padding, strides, input_shape,
                       max_pool):
    layer = stax.MaxPool if max_pool else stax.AvgPool
    init_fun, apply_fun = layer(window_shape, padding=padding, strides=strides)
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shape)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(input_shape),
       "input_shape": input_shape}
      for input_shape in [(2, 3), (2, 3, 4)]))
  def testFlattenShape(self, input_shape):
    init_fun, apply_fun = stax.Flatten
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shape)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_input_shape={}_spec={}".format(input_shape, i),
       "input_shape": input_shape, "spec": spec}
      for input_shape in [(2, 5, 6, 1)]
      for i, spec in enumerate([
          [stax.Conv(3, (2, 2))],
          [stax.Conv(3, (2, 2)), stax.Flatten, stax.Dense(4)]])))
  def testSerialComposeLayersShape(self, input_shape, spec):
    init_fun, apply_fun = stax.serial(*spec)
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shape)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_input_shape={}".format(input_shape),
       "input_shape": input_shape}
      for input_shape in [(3, 4), (2, 5, 6, 1)]))
  def testDropoutShape(self, input_shape):
    init_fun, apply_fun = stax.Dropout(0.9)
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shape)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_input_shape={}".format(input_shape),
       "input_shape": input_shape}
      for input_shape in [(3, 4), (2, 5, 6, 1)]))
  def testFanInSum(self, input_shape):
    init_fun, apply_fun = stax.FanInSum
    _CheckShapeAgreement(self, init_fun, apply_fun, [input_shape, input_shape])

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshapes={}_axis={}".format(input_shapes, axis),
       "input_shapes": input_shapes, "axis": axis}
      for input_shapes, axis in [
          ([(2, 3), (2, 1)], 1),
          ([(2, 3), (2, 1)], -1),
          ([(1, 2, 4), (1, 1, 4)], 1),
      ]))
  def testFanInConcat(self, input_shapes, axis):
    init_fun, apply_fun = stax.FanInConcat(axis)
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shapes)

  def testIsuse182(self):
    init_fun, apply_fun = stax.Softmax
    input_shape = (10, 3)
    inputs = onp.arange(30.).astype("float32").reshape(input_shape)

    out_shape, params = init_fun(input_shape)
    out = apply_fun(params, inputs)

    assert out_shape == out.shape
    assert onp.allclose(onp.sum(onp.asarray(out), -1), 1.)


if __name__ == "__main__":
  absltest.main()
