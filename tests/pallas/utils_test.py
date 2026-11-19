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

from absl.testing import absltest
from absl.testing import parameterized
import hypothesis
from hypothesis import strategies as st
import jax
from jax._src import hypothesis_test_util as htu
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()
htu.setup_hypothesis(max_examples=100)


class CdivTest(jtu.JaxTestCase):

  @parameterized.product(
      a=[0, 8, 9, 1, 7, 10],
      b=[2, 1, 3, 4],
      factor_a=[1, -1],
      factor_b=[1, -1],
  )
  def test_cdiv_integers(self, a, b, factor_a, factor_b):
    a = a * factor_a
    b = b * factor_b
    expected = int((a + b - 1) / b)
    self.assertEqual(pl.cdiv(a, b), expected)

  def test_cdiv_array(self):
    a = [0, 0, -1, -1, 1, 1, -7, -7, 7, 7, -8, -8, 8, 8, -9, -9, 9, 9]
    b = [-2, 2] * 9
    expected = np.array(
        [int((x + y - 1) / y) for x, y in zip(a, b)],
        dtype=np.int32,
    )
    result = pl.cdiv(
        jnp.array(a, dtype=jnp.int32), jnp.array(b, dtype=jnp.int32)
    )
    self.assertArraysEqual(result, expected)
    self.assertEqual(result.dtype, jnp.int32)

  def test_cdiv_unsigned_integers(self):
    a = jnp.array([0, 1, 5, 8, 9, 10], dtype=jnp.uint32)
    b = jnp.array(2, dtype=jnp.uint32)
    expected = np.array([int((x + 1) / 2) for x in a], dtype=jnp.uint32)
    result = pl.cdiv(a, b)
    self.assertArraysEqual(result, expected)
    self.assertEqual(result.dtype, jnp.uint32)

  def test_cdiv_jit(self):
    a = jnp.array([-9, -5, 0, 1, 5, 8, 9, 10], dtype=jnp.int32)
    b = jnp.array(2, dtype=jnp.int32)
    expected = np.array([int((x + 1) / 2) for x in a], dtype=np.int32)
    f = jax.jit(pl.cdiv)
    self.assertArraysEqual(f(a, b), expected)

  @hypothesis.given(
      st.integers(min_value=-1000, max_value=1000),
      st.integers(min_value=-100, max_value=100),
  )
  def test_cdiv_hypothesis(self, a, b):
    hypothesis.assume(b != 0)
    expected = int((a + b - 1) / b)
    a_arr = jnp.array(a, dtype=jnp.int32)
    b_arr = jnp.array(b, dtype=jnp.int32)
    self.assertEqual(jax.jit(pl.cdiv)(a_arr, b_arr), expected)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
