# Copyright 2023 The JAX Authors.
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

import numpy as np
from absl.testing import absltest

import jax
from jax import config
from jax import lax
import jax.numpy as jnp
import jax._src.test_util as jtu
from jax.experimental import custom_impl

config.parse_flags_with_absl()


class CustomImplTest(jtu.JaxTestCase):
  @jax.default_matmul_precision("float32")
  def test_custom_dot_general(self):
    def custom_dot_general(x, y, **kwargs):
      return lax.dot_general(x / 2, y / 2, **kwargs)

    @custom_impl(lax.dot_general_p, custom_dot_general)
    def f(x, y):
      return x @ y
    def f_direct(x, y):
      return (x / 2) @ (y / 2)

    rng = jtu.rand_default(self.rng())
    x = rng((3, 4), np.float32)
    y = rng((4, 5), np.float32)

    actual = f(x, y)
    expected = f_direct(x, y)
    self.assertAllClose(actual, expected)

    actual_with_jit = jax.jit(f)(x, y)
    self.assertAllClose(actual_with_jit, expected)

  @jax.default_matmul_precision("float32")
  def test_composition(self):
    def dot_general_1(x, y, **kwargs):
      dot_general_1.count += 1
      return lax.dot_general(x - 1, y, **kwargs)
    dot_general_1.count = 0

    def dot_general_2(x, y, **kwargs):
      dot_general_2.count += 1
      return lax.dot_general(x, y + 1, **kwargs)
    dot_general_2.count = 0

    @custom_impl(lax.dot_general_p, dot_general_1)
    @custom_impl(lax.dot_general_p, dot_general_2)
    def f(x, y):
      return x @ y
    def f_direct(x, y):
      return (x - 1) @ (y + 1)

    rng = jtu.rand_default(self.rng())
    x = rng((3, 4), np.float32)
    y = rng((4, 5), np.float32)

    expected = f_direct(x, y)
    actual = f(x, y)
    self.assertEqual(dot_general_1.count, 1)
    self.assertEqual(dot_general_2.count, 1)
    self.assertAllClose(actual, expected)

    actual_with_jit = jax.jit(f)(x, y)
    self.assertEqual(dot_general_1.count, 2)
    self.assertEqual(dot_general_2.count, 2)
    self.assertAllClose(actual_with_jit, expected)

  @jax.default_matmul_precision("float32")
  def test_scan(self):
    def custom_dot_general(x, y, **kwargs):
      return lax.dot_general(x + 1, y, **kwargs)

    @custom_impl(lax.dot_general_p, custom_dot_general)
    def matrix_power_scan(M, N):
      return lax.fori_loop(0, N, lambda i, val: M @ val,
                           jnp.eye(*M.shape, dtype=M.dtype))

    rng = jtu.rand_default(self.rng())
    M = rng((3, 3), np.float32)

    actual = matrix_power_scan(M, 5)
    expected = jnp.linalg.matrix_power(M + 1, 5)

    self.assertAllClose(actual, expected)

  def test_while(self):
    def custom_add(x, y):
      custom_add.count += 1
      return x + y
    custom_add.count = 0

    def custom_lt(x, y):
      custom_lt.count += 1
      return x < y
    custom_lt.count = 0

    @custom_impl(lax.add_p, custom_add)
    @custom_impl(lax.lt_p, custom_lt)
    def f():
      return lax.while_loop(
        cond_fun = lambda x: x < 10,
        body_fun = lambda x: x + 2,
        init_val = 1)

    _ = f()
    self.assertEqual(custom_add.count, 1)
    self.assertEqual(custom_lt.count, 1)

  def test_incorrect_return_type(self):
    def custom_dot_general(x, y, **kwargs):
      return lax.dot_general(x, y, **kwargs).astype('bfloat16')

    @custom_impl(lax.dot_general_p, custom_dot_general)
    def f(x, y):
      return x @ y

    rng = jtu.rand_default(self.rng())
    x = rng((3, 4), np.float32)
    y = rng((4, 5), np.float32)

    msg = "custom impl for dot_general returned the wrong output types."
    with self.assertRaisesRegex(ValueError, msg):
      f(x, y)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
