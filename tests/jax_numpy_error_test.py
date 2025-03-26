# Copyright 2025 The JAX Authors.
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
import jax
from jax._src import config
from jax._src import error_check
from jax._src import test_util as jtu
from jax._src.numpy import error as jnp_error
import jax.numpy as jnp

config.parse_flags_with_absl()


JaxValueError = error_check.JaxValueError


class JaxNumpyErrorTests(jtu.JaxTestCase):
  @parameterized.product(jit=[True, False])
  def test_set_error_if_nan(self, jit):
    def f(x):
      jnp_error._set_error_if_nan(x)
      return x

    if jit:
      f = jax.jit(f)

    x = jnp.full((4,), jnp.nan, dtype=jnp.float32)

    with jnp_error.error_checking_behavior(nan="ignore"):
      _ = f(x)
      error_check.raise_if_error()  # should not raise error

    with jnp_error.error_checking_behavior(nan="raise"):
      _ = f(x)
      with self.assertRaisesRegex(JaxValueError, "NaN"):
        error_check.raise_if_error()

  @parameterized.product(jit=[True, False])
  def test_error_category_divide_check(self, jit):
    def f(x, y):
      jnp_error._set_error_if_with_category(
          y == 0.0, "division by zero", category="divide"
      )
      return x / y

    if jit:
      f = jax.jit(f)

    x = jnp.arange(4, dtype=jnp.float32) + 1
    y = jnp.arange(4, dtype=jnp.float32)

    with jnp_error.error_checking_behavior(divide="ignore"):
      _ = f(x, y)
      error_check.raise_if_error()  # should not raise error

    with jnp_error.error_checking_behavior(divide="raise"):
      _ = f(x, y)
      with self.assertRaisesRegex(JaxValueError, "division by zero"):
        error_check.raise_if_error()

  @parameterized.product(jit=[True, False])
  def test_error_category_oob_check(self, jit):
    def f(x, start_indices, slice_sizes):
      jnp_error._set_error_if_with_category(
          jnp.logical_or(
              start_indices < 0,
              start_indices + jnp.array(slice_sizes, dtype=jnp.int32)
              >= jnp.array(x.shape, dtype=jnp.int32),
          ),
          "Out of bounds in dynamic_slice",
          category="oob",
      )
      y = jax.lax.dynamic_slice(
          x, start_indices, slice_sizes, allow_negative_indices=False
      )
      return y

    if jit:
      f = jax.jit(f, static_argnums=(2,))

    x = jnp.arange(12).reshape(3, 4)
    start_indices = jnp.array([0, -1], dtype=jnp.int32)
    slice_sizes = (3, 4)

    with jnp_error.error_checking_behavior(oob="ignore"):
      _ = f(x, start_indices, slice_sizes)
      error_check.raise_if_error()  # should not raise error

    with jnp_error.error_checking_behavior(oob="raise"):
      _ = f(x, start_indices, slice_sizes)
      with self.assertRaisesRegex(
          JaxValueError, "Out of bounds in dynamic_slice",
      ):
        error_check.raise_if_error()

  def test_error_category_invalid_category(self):
    with self.assertRaisesRegex(ValueError, "Invalid category"):
      jnp_error._set_error_if_with_category(
          jnp.isnan(jnp.float32(1.0)), "x is NaN", category="invalid"
      )

  @parameterized.product(jit=[True, False])
  def test_can_raise_nan_error(self, jit):
    x = jnp.arange(4, dtype=jnp.float32) - 1

    f = jnp.log
    if jit:
      f = jax.jit(f)

    with jnp_error.error_checking_behavior(nan="raise"):
      f(x)
      with self.assertRaisesRegex(JaxValueError, "NaN"):
        error_check.raise_if_error()


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
