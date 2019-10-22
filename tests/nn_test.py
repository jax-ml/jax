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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp

from jax import test_util as jtu
from jax.test_util import check_grads
from jax import nn
from jax import random
import jax

from jax.config import config
config.parse_flags_with_absl()

class NNFunctionsTest(jtu.JaxTestCase):

  def testSoftplusGrad(self):
    check_grads(nn.softplus, (1e-8,), 4)

  def testSoftplusValue(self):
    val = nn.softplus(89.)
    self.assertAllClose(val, 89., check_dtypes=False)

  def testEluGrad(self):
    check_grads(nn.elu, (1e4,), 4, eps=1.)

  def testEluValue(self):
    val = nn.elu(1e4)
    self.assertAllClose(val, 1e4, check_dtypes=False)

InitializerRecord = collections.namedtuple(
  "InitializerRecord",
  ["name", "initializer", "shapes"])

ALL_SHAPES = [(2,), (2, 2), (2, 3), (3, 2), (2, 3, 4), (4, 3, 2), (2, 3, 4, 5)]

def initializer_record(name, initializer, min_dims=2, max_dims=4):
  shapes = [shape for shape in ALL_SHAPES
            if min_dims <= len(shape) <= max_dims]
  
  return InitializerRecord(name, initializer, shapes)

INITIALIZER_RECS = [
    initializer_record("uniform", nn.initializers.uniform(), 1),
    initializer_record("normal", nn.initializers.normal(), 1),
    initializer_record("he_normal", nn.initializers.he_normal()),
    initializer_record("he_uniform", nn.initializers.he_uniform()),
    initializer_record("glorot_normal", nn.initializers.glorot_normal()),
    initializer_record("glorot_uniform", nn.initializers.glorot_uniform()),
    initializer_record("lecun_normal", nn.initializers.lecun_normal()),
    initializer_record("lecun_uniform", nn.initializers.lecun_uniform()),
    initializer_record("orthogonal", nn.initializers.orthogonal(), 2, 2)
]

class NNInitializersTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_{}_{}".format(
           rec.name,
           jtu.format_shape_dtype_string(shape, dtype)),
       "initializer": rec.initializer, "rng": random.PRNGKey(0),
       "shape": shape, "dtype": dtype}
      for rec in INITIALIZER_RECS
      for shape in rec.shapes
      for dtype in [onp.float32, onp.float64]))
  def testInitializer(self, initializer, rng, shape, dtype):
    val = initializer(rng, shape, dtype)

if __name__ == "__main__":
  absltest.main()
