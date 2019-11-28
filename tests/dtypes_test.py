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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import dtypes
from jax import numpy as np
from jax import test_util as jtu

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS


class DtypesTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
    {"testcase_name": "_type={}".format(type.__name__), "type": type,
     "dtype": dtype}
    for type, dtype in [(bool, np.bool_), (int, np.int_), (float, np.float_),
                        (complex, np.complex_)])
  def testDefaultTypes(self, type, dtype):
    for f in [np.array, jax.jit(np.array), jax.jit(lambda x: x)]:
      y = f(type(0))
      self.assertTrue(isinstance(y, np.ndarray), msg=(f, y))
      self.assertEqual(y.dtype, dtypes.canonicalize_dtype(dtype), msg=(f, y))

  @parameterized.named_parameters(
    {"testcase_name": "_swap={}_jit={}".format(swap, jit),
     "swap": swap, "jit": jit} 
    for swap in [False, True] for jit in [False, True])
  @jtu.skip_on_devices("tpu")  # F16 not supported on TPU
  def testBinaryPromotion(self, swap, jit):
    testcases = [
      (np.array(1.), 0., np.float_),
      (np.array(1.), np.array(0.), np.float_),
      (np.array(1.), np.array(0., dtype=np.float16), np.float_),
      (np.array(1.), np.array(0., dtype=np.float32), np.float_),
      (np.array(1.), np.array(0., dtype=np.float64), np.float64),
      (np.array(1., dtype=np.float16), 0., np.float16),
      (np.array(1., dtype=np.float32), 0., np.float32),
      (np.array(1., dtype=np.float64), 0., np.float64),
      (np.array(1., dtype=np.float16), np.array(0., dtype=np.float16), np.float16),
      (np.array(1., dtype=np.float16), np.array(0., dtype=np.float32), np.float32),
      (np.array(1., dtype=np.float16), np.array(0., dtype=np.float64), np.float64),
      (np.array(1., dtype=np.float32), np.array(0., dtype=np.float32), np.float32),
      (np.array(1., dtype=np.float32), np.array(0., dtype=np.float64), np.float64),
      (np.array(1., dtype=np.float64), np.array(0., dtype=np.float64), np.float64),
      (np.array([1.]), 0., np.float_),
      (np.array([1.]), np.array(0.), np.float_),
      (np.array([1.]), np.array(0., dtype=np.float16), np.float_),
      (np.array([1.]), np.array(0., dtype=np.float32), np.float_),
      (np.array([1.]), np.array(0., dtype=np.float64), np.float64),
      (np.array([1.], dtype=np.float32), np.array(0., dtype=np.float16), np.float32),
      (np.array([1.], dtype=np.float16), np.array(0., dtype=np.float32), np.float32),
      (np.array([1.], dtype=np.float16), 0., np.float16),
    ]
    op = jax.jit(operator.add) if jit else operator.add
    for x, y, dtype in testcases:
      x, y = (y, x) if swap else (x, y)
      z = x + y
      self.assertTrue(isinstance(z, np.ndarray), msg=(x, y, z))
      self.assertEqual(z.dtype, dtypes.canonicalize_dtype(dtype), msg=(x, y, z))

if __name__ == "__main__":
  absltest.main()
