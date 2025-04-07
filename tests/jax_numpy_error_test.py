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

import operator

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
  def setUp(self):
    # TODO(b/408148001): Fix thread safety issue.
    if jtu.TEST_NUM_THREADS.value > 1:
      self.skipTest("Test does not work with multiple threads")
    super().setUp()

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
  def test_set_error_if_divide_by_zero(self, jit):
    def f(x, y):
      jnp_error._set_error_if_divide_by_zero(y)
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
      with self.assertRaisesRegex(JaxValueError, "Division by zero"):
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

  @staticmethod
  def nan_cases(cases):
    for jit in (True, False):
      for func, args_error, args_no_err in cases:
        if not isinstance(args_error, tuple):
          args_error = (args_error,)
        if not isinstance(args_no_err, tuple):
          args_no_err = (args_no_err,)

        jit_str = "jit" if jit else "nojit"
        func_str = f"{func.__module__}.{func.__name__}"
        name = f"_{jit_str}_{func_str}"

        yield name, jit, func, args_error, args_no_err

  @parameterized.named_parameters(
      nan_cases((
          # List of all NaN-producing jax.numpy functions.
          # The first group of numbers is the input that will produce a NaN, and
          # the second group is the input that will not produce a NaN.
          # go/keep-sorted start
          (jnp.acos, 2.0, 0.5),
          (jnp.acosh, 0.5, 2.0),
          (jnp.add, (jnp.inf, -jnp.inf), (0.0, 0.0)),
          (jnp.arccos, 2.0, 0.5),
          (jnp.arccosh, 0.5, 2.0),
          (jnp.arcsin, -2.0, 0.5),
          (jnp.arctanh, -2.0, 0.5),
          (jnp.asin, -2.0, 0.5),
          (jnp.atanh, -2.0, 0.5),
          (jnp.cos, jnp.inf, 1.0),
          (jnp.divide, (0.0, 0.0), (1.0, 1.0)),
          (jnp.divmod, (1.0, 0.0), (1.0, 1.0)),
          (jnp.float_power, (-1.0, 0.5), (1.0, 1.0)),
          (jnp.fmod, (1.0, 0.0), (1.0, 1.0)),
          (jnp.log, -1.0, 1.0),
          (jnp.log10, -1.0, 1.0),
          (jnp.log1p, -1.5, 1.0),
          (jnp.log2, -1.0, 1.0),
          (jnp.mod, (1.0, 0.0), (1.0, 1.0)),
          (jnp.pow, (-1.0, 0.5), (1.0, 1.0)),
          (jnp.power, (-1.0, 0.5), (1.0, 1.0)),
          (jnp.remainder, (1.0, 0.0), (1.0, 1.0)),
          (jnp.sin, jnp.inf, 1.0),
          # TODO(https://github.com/jax-ml/jax/issues/27470): Not yet supported.
          # (jnp.sinc, jnp.inf, 1.0),
          (jnp.sqrt, -4.0, 4.0),
          (jnp.subtract, (jnp.inf, jnp.inf), (0.0, 0.0)),
          (jnp.tan, jnp.inf, 1.0),
          (jnp.true_divide, (0.0, 0.0), (1.0, 1.0)),
          (operator.add, (jnp.inf, -jnp.inf), (0.0, 0.0)),
          (operator.mod, (1.0, 0.0), (1.0, 1.0)),
          (operator.pow, (-1.0, 0.5), (1.0, 1.0)),
          (operator.sub, (jnp.inf, jnp.inf), (0.0, 0.0)),
          (operator.truediv, (0.0, 0.0), (1.0, 1.0)),
          # go/keep-sorted end
      ))
  )
  def test_can_raise_nan_error(self, jit, f, args_err, args_no_err):
    args_err = [jnp.float32(x) for x in args_err]
    args_no_err = [jnp.float32(x) for x in args_no_err]

    if jit:
      f = jax.jit(f)

    with jnp_error.error_checking_behavior(nan="raise"):
      f(*args_no_err)
      error_check.raise_if_error()  # should not raise error

      f(*args_err)
      with self.assertRaisesRegex(JaxValueError, "NaN"):
        error_check.raise_if_error()

  INT_TYPES = (jnp.int32, jnp.uint32, jnp.int64, jnp.uint64, jnp.int16,
                  jnp.uint16, jnp.int8, jnp.uint8)
  FLOAT_TYPES = (jnp.float32, jnp.float64, jnp.float16, jnp.bfloat16)

  @staticmethod
  def divide_cases(cases):
    for jit in (True, False):
      for func, dtypes in cases:
        for dtype in dtypes:
          jit_str = "jit" if jit else "nojit"
          func_str = f"{func.__module__}.{func.__name__}"
          dtype_str = dtype.__name__
          name = f"_{jit_str}_{func_str}_{dtype_str}"
          yield name, jit, func, dtype

  @parameterized.named_parameters(
      divide_cases((
          # go/keep-sorted start
          (jnp.divmod, FLOAT_TYPES + INT_TYPES),
          (jnp.floor_divide, INT_TYPES),
          (jnp.mod, FLOAT_TYPES + INT_TYPES),
          (jnp.remainder, FLOAT_TYPES + INT_TYPES),
          (jnp.true_divide, FLOAT_TYPES),
          (operator.mod, FLOAT_TYPES + INT_TYPES),
          (operator.truediv, FLOAT_TYPES),
          # go/keep-sorted end
      ))
  )
  def test_can_raise_divide_by_zero_error(self, jit, div_func, dtype):
    if not jax.config.x64_enabled and jnp.dtype(dtype).itemsize == 8:
      self.skipTest("64-bit types require x64_enabled")

    args_err = (dtype(1), dtype(0))
    args_no_err = (dtype(1), dtype(1))

    if jit:
      div_func = jax.jit(div_func)

    with jnp_error.error_checking_behavior(divide="raise"):
      div_func(*args_no_err)
      error_check.raise_if_error()  # should not raise error

      div_func(*args_err)
      with self.assertRaisesRegex(JaxValueError, "Division by zero"):
        error_check.raise_if_error()

  @parameterized.product(jit=[True, False])
  def test_can_raise_oob_error_take(self, jit):
    def f(x, a):
      return x[a]

    if jit:
      f = jax.jit(f)

    x = jnp.arange(10)
    a = jnp.int32(10)

    with jnp_error.error_checking_behavior(oob="ignore"):
      f(x, a)
      error_check.raise_if_error()  # should not raise error

    with jnp_error.error_checking_behavior(oob="raise"):
      f(x, a)
      with self.assertRaisesRegex(JaxValueError, "Out of bounds"):
        error_check.raise_if_error()

  def test_can_raise_oob_error_dynamic_slice(self):
    def f(x, a):
      return x[:, a:a+4]  # dynamic indices are non-jittable

    x = jnp.arange(10).reshape(2, 5)
    a = jnp.array(3, dtype=jnp.int32)

    with jnp_error.error_checking_behavior(oob="ignore"):
      f(x, a)
      error_check.raise_if_error()  # should not raise error

    with jnp_error.error_checking_behavior(oob="raise"):
      f(x, a)
      with self.assertRaisesRegex(JaxValueError, "Out of bounds"):
        error_check.raise_if_error()


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
