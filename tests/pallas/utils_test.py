# Copyright 2026 The JAX Authors.
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

import math
from absl.testing import absltest
from absl.testing import parameterized
import hypothesis
from hypothesis import strategies as st
import jax
from jax._src import hypothesis_test_util as htu
from jax._src import test_util as jtu
from jax._src.pallas import utils as pallas_utils
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()
htu.setup_hypothesis(max_examples=100)


class CdivTest(jtu.JaxTestCase):

  @parameterized.parameters(
      # Negative values of a
      (-10, 3, -2),
      (-9, 3, -2),
      (-8, 3, -2),
      (-7, 3, -1),
      (-6, 3, -1),
      (-5, 3, -1),
      (-4, 3, 0),
      (-3, 3, 0),
      (-2, 3, 0),
      (-1, 3, 0),
      (-9, 2, -4),
      (-8, 2, -3),
      (-5, 2, -2),
      (-4, 2, -1),
      (-3, 2, -1),
      (-2, 2, 0),
      (-1, 2, 0),
      # Non-negative values of a
      (0, 1, 0),
      (0, 5, 0),
      (1, 1, 1),
      (1, 2, 1),
      (2, 2, 1),
      (3, 2, 2),
      (4, 2, 2),
      (5, 2, 3),
      (8, 2, 4),
      (9, 2, 5),
      (10, 3, 4),
      (11, 3, 4),
      (12, 3, 4),
      (13, 3, 5),
      (1, 100, 1),
      (100, 100, 1),
      (101, 100, 2),
      (1000, 10, 100),
      (1001, 10, 101),
  )
  def test_cdiv_integers(self, a, b, expected):
    self.assertEqual(pallas_utils.cdiv(a, b), expected)
    self.assertEqual(pl.cdiv(a, b), expected)
    if a >= 0:
      self.assertEqual(pallas_utils.cdiv(a, b), math.ceil(a / b))
    else:
      self.assertEqual(pallas_utils.cdiv(a, b), int((a + b - 1) / b))
    # Verify integer scalar evaluation matches JAX array evaluation
    a_arr = jnp.array(a, dtype=jnp.int32)
    b_arr = jnp.array(b, dtype=jnp.int32)
    self.assertEqual(int(pallas_utils.cdiv(a_arr, b_arr)), expected)

  def test_cdiv_array(self):
    a = jnp.array(
        [
            -10,
            -9,
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
        ],
        dtype=jnp.int32,
    )
    b = jnp.array(3, dtype=jnp.int32)
    expected = np.array(
        [-2, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        dtype=np.int32,
    )
    result = pallas_utils.cdiv(a, b)
    self.assertArraysEqual(result, expected)
    self.assertEqual(result.dtype, a.dtype)

  def test_cdiv_signed_integers(self):
    dtype = jnp.int32
    a = jnp.array([-9, -5, -2, 0, 1, 5, 8, 9, 10], dtype=dtype)
    b = jnp.array(2, dtype=dtype)
    expected = np.array([-4, -2, 0, 0, 1, 3, 4, 5, 5], dtype=dtype)
    result = pallas_utils.cdiv(a, b)
    self.assertArraysEqual(result, expected)
    self.assertEqual(result.dtype, dtype)

  def test_cdiv_unsigned_integers(self):
    dtype = jnp.uint32
    a = jnp.array([0, 1, 5, 8, 9, 10], dtype=dtype)
    b = jnp.array(2, dtype=dtype)
    expected = np.array([0, 1, 3, 4, 5, 5], dtype=dtype)
    result = pallas_utils.cdiv(a, b)
    self.assertArraysEqual(result, expected)
    self.assertEqual(result.dtype, dtype)

  def test_cdiv_jit(self):
    @jax.jit
    def f(a, b):
      return pallas_utils.cdiv(a, b)

    a = jnp.array([-9, -5, 0, 1, 5, 8, 9, 10], dtype=jnp.int32)
    b = jnp.array(2, dtype=jnp.int32)
    expected = np.array([-4, -2, 0, 1, 3, 4, 5, 5], dtype=np.int32)
    self.assertArraysEqual(f(a, b), expected)

    # JIT with scalar array second argument
    @jax.jit
    def g(a):
      return pallas_utils.cdiv(a, jnp.array(3, dtype=a.dtype))

    self.assertArraysEqual(
        g(jnp.array([-6, -3, -1, 0, 1, 2, 3, 4, 5], dtype=jnp.int32)),
        np.array([-1, 0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int32),
    )

  @hypothesis.given(
      st.integers(min_value=-1000, max_value=1000),
      st.integers(min_value=1, max_value=100),
  )
  def test_cdiv_hypothesis(self, a, b):
    expected = pallas_utils.cdiv(a, b)
    a_arr = jnp.array(a, dtype=jnp.int32)
    b_arr = jnp.array(b, dtype=jnp.int32)
    self.assertEqual(jax.jit(pallas_utils.cdiv)(a_arr, b_arr), expected)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
