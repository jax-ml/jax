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

"""Tests for --debug_nans."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp

import jax
from jax import test_util as jtu
from jax.test_util import check_grads
from jax import numpy as np
from jax import random

from jax.config import config
config.parse_flags_with_absl()

class DebugNaNsTest(jtu.JaxTestCase):

  def setUp(self):
    self.cfg = config.read("jax_debug_nans")
    config.update("jax_debug_nans", True)

  def tearDown(self):
    config.update("jax_debug_nans", self.cfg)

  def testSingleResultPrimitiveNoNaN(self):
    A = np.array([[1., 2.], [2., 3.]])
    B = np.tanh(A)

  def testMultipleResultPrimitiveNoNaN(self):
    A = np.array([[1., 2.], [2., 3.]])
    D, V = np.linalg.eig(A)

  def testJitComputationNoNaN(self):
    A = np.array([[1., 2.], [2., 3.]])
    B = jax.jit(np.tanh)(A)

  def testSingleResultPrimitiveNaN(self):
    A = np.array(0.)
    with self.assertRaises(FloatingPointError):
      B = 0. / A
