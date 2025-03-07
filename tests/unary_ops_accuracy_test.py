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

"""Unit test for result accuracy for unary ops."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lax import lax
from jax._src.lib.mlir import ir
import jax.numpy as jnp
from mlir import ir
from mlir.dialects import stablehlo
import numpy as np

config.parse_flags_with_absl()


def make_unary_test_cases(
    testcase_name, ops, high_tol, low_tol, x, min_error_val=0.0
):
  """Creates a single test case."""
  suffix_list = ["_lax", "_jnp"] if len(ops) > 1 else ["_lax"]
  return [
      {
          "testcase_name": testcase_name + suffix,
          "op": op,
          "high_tol": high_tol,
          "low_tol": low_tol,
          "x": x,
          "min_error_val": min_error_val,
      }
      for op, suffix in zip(ops, suffix_list)
  ]


UNARY_OPS = {
    "exp": make_unary_test_cases(
        "exp",
        (lax.exp, jnp.exp),
        lax.Tolerance(atol=2**-5, rtol=2**-5, ulps=2),
        lax.Tolerance(atol=1.5 * 2**-8, rtol=2**-18, ulps=2),
        np.arange(84.0, 88.0, dtype=np.float32),
    ),
    "exp2": make_unary_test_cases(
        "exp2",
        (lax.exp2, jnp.exp2),
        lax.Tolerance(atol=2**-5, rtol=2**-5, ulps=2),
        lax.Tolerance(atol=1.5 * 2**-8, rtol=2**-18, ulps=2),
        np.arange(84.0, 88.0, dtype=np.float32),
    ),
    "expm1": make_unary_test_cases(
        "expm1",
        (lax.expm1, jnp.expm1),
        lax.Tolerance(atol=2**-5, rtol=2**-5, ulps=2),
        lax.Tolerance(atol=1.5 * 2**-8, rtol=2**-18, ulps=2),
        np.arange(84.0, 88.0, dtype=np.float32),
    ),
    "log": make_unary_test_cases(
        "log",
        (lax.log, jnp.log),
        lax.Tolerance(atol=0, rtol=2**-10, ulps=0),
        lax.Tolerance(atol=2**-16, rtol=2**-20, ulps=0),
        np.linspace(1e28, 2e28, 10, dtype=np.float32),
        1.0,
    ),
    "log1p": make_unary_test_cases(
        "log1p",
        (lax.log1p, jnp.log1p),
        lax.Tolerance(atol=0, rtol=2**-11, ulps=0),
        lax.Tolerance(atol=0, rtol=2**-14, ulps=0),
        np.linspace(-9e-8, -8e-8, 10, dtype=np.float32),
        1.0,
    ),
    # logistic doesn't use LogisticOp in JAX.
    "logistic": make_unary_test_cases(
        "logistic_lax",
        (lax.logistic,),
        lax.Tolerance(atol=2**-12, rtol=0, ulps=0),
        lax.Tolerance(atol=2**-16, rtol=0, ulps=0),
        np.linspace(8.24, 8.3, 10, dtype=np.float32),
        1.0,
    ),
    "tanh": make_unary_test_cases(
        "tanh",
        (lax.tanh, jnp.tanh),
        lax.Tolerance(atol=2**-12, rtol=0, ulps=0),
        lax.Tolerance(atol=2**-16, rtol=0, ulps=0),
        np.linspace(5.83, 5.86, 10, dtype=np.float32),
        0.0,
    ),
    "cos": make_unary_test_cases(
        "cos",
        (lax.cos, jnp.cos),
        lax.Tolerance(atol=0, rtol=2**-10, ulps=0),
        lax.Tolerance(atol=0, rtol=2**-30, ulps=0),
        np.linspace(9.7e22, 9.8e22, 10, dtype=np.float32),
        0.0,
    ),
    "sin": make_unary_test_cases(
        "sin",
        (lax.sin, jnp.sin),
        lax.Tolerance(atol=0, rtol=2**-10, ulps=0),
        lax.Tolerance(atol=0, rtol=2**-30, ulps=0),
        np.linspace(9.7e22, 9.8e22, 10, dtype=np.float32),
        0.0,
    ),
    "tan": make_unary_test_cases(
        "tan",
        (lax.tan, jnp.tan),
        lax.Tolerance(atol=0, rtol=2**-10, ulps=0),
        lax.Tolerance(atol=0, rtol=2**-30, ulps=0),
        np.linspace(250.0, 252.0, 10, dtype=np.float32),
        0.0,
    ),
    "cbrt": make_unary_test_cases(
        "cbrt",
        (lax.cbrt, jnp.cbrt),
        lax.Tolerance(atol=0, rtol=2**-10, ulps=0),
        lax.Tolerance(atol=0, rtol=2**-30, ulps=0),
        np.linspace(250.0, 252.0, 10, dtype=np.float32),
        0.0,
    ),
    "sqrt": make_unary_test_cases(
        "sqrt",
        (lax.sqrt, jnp.sqrt),
        lax.Tolerance(atol=0, rtol=2**-10, ulps=0),
        lax.Tolerance(atol=0, rtol=2**-30, ulps=0),
        np.linspace(250.0, 252.0, 10, dtype=np.float32),
        0.0,
    ),
    "rsqrt": make_unary_test_cases(
        "rsqrt",
        (lax.rsqrt,),
        lax.Tolerance(atol=0, rtol=2**-10, ulps=0),
        lax.Tolerance(atol=0, rtol=2**-30, ulps=0),
        np.linspace(250.0, 252.0, 10, dtype=np.float32),
        0.0,
    ),
}


def generate_test_cases(op_names):
  test_cases = []
  for op in op_names:
    op_group = UNARY_OPS[op]
    if op_group is None:
      raise ValueError(f"No test cases found for op: {op}")
    test_cases.extend(op_group)
  return test_cases


class UnaryOpsAccuracyTest(jtu.JaxTestCase):

  def test_result_accuracy_mode_attr(self):
    with ir.Context() as context:
      stablehlo.register_dialect(context)
      attr = stablehlo.ResultAccuracyModeAttr.get("DEFAULT")
      assert attr is not None
      assert attr.value == "DEFAULT"

  def test_result_accuracy_attr(self):
    with ir.Context() as context:
      stablehlo.register_dialect(context)
      attr = stablehlo.ResultAccuracyAttr.get(
          atol=1e-5, rtol=0.0, ulps=1, mode="TOLERANCE"
      )
      assert attr is not None
      assert attr.mode == "TOLERANCE"
      assert attr.atol == 1e-5
      assert attr.rtol == 0.0
      assert attr.ulps == 1

  @parameterized.named_parameters(
      *generate_test_cases(["exp", "expm1", "exp2", "log", "log1p", "tanh"])
  )
  def test_unary_ops_choose_impl(self, op, high_tol, low_tol, x, min_error_val):
    @jax.jit
    def f_default(x):
      y = op(x, accuracy=high_tol)
      return y

    @jax.jit
    def f_accurate(x):
      y = op(x, accuracy=low_tol)
      return y

    # Input values that would cause large differences between the two
    # implementations.
    diff = abs(f_default(x) - f_accurate(x))
    if (jtu.get_tpu_version() >= 5 and op in [lax.tanh, jnp.tanh, lax.log, jnp.log]):
      # From tpu version 5 and onwards, even with tighter tolerance, the high performant
      # implementation for tanh  is chosen because the the chip implementation has improved accuracy.
      self.assertTrue(jnp.all(diff == 0))
    else:
      self.assertTrue(jnp.any(diff > 0))

  @parameterized.named_parameters(
      *generate_test_cases(["exp", "expm1", "exp2", "log", "log1p", "tanh"])
  )
  def test_unary_vmap(self, op, high_tol, low_tol, x, min_error_val):
    @jax.jit
    def f(x, y):
      diff = lambda val: abs(
          op(val, accuracy=high_tol) - op(val, accuracy=low_tol)
      )
      return diff(x), diff(y)

    diff_x, diff_y = jax.vmap(f, in_axes=(None, 0), out_axes=0)(
        min_error_val, x
    )
    # diff(min_error_val) should be 0
    self.assertTrue(jnp.all(diff_x == 0))
    # diff(x) should be > 0
    if (jtu.get_tpu_version() >= 5 and op in [lax.tanh, jnp.tanh, lax.log, jnp.log]):
      # From tpu version 5 and onwards, even with tighter tolerance, the high performant
      # implementation for tanh and log is chosen because the the chip implementation has improved accuracy.
      self.assertTrue(jnp.all(diff_y == 0))
    else:
      self.assertTrue(jnp.any(diff_y > 0))

  @parameterized.named_parameters(
      *generate_test_cases(["exp", "expm1", "exp2"])
  )
  def test_diff_grad(self, op, high_tol, low_tol, x, min_error_val):
    @jax.jit
    def f_default(x):
      default_op = op(x, accuracy=low_tol)
      return jnp.sum(default_op)

    f_default_grad = jax.grad(f_default)

    @jax.jit
    def f_accurate(x):
      high_op = op(x, accuracy=high_tol)
      return jnp.sum(high_op)

    f_accurate_grad = jax.grad(f_accurate)
    # Accuracy should be carried through to the gradient causing
    # a large diff.
    diff = abs(f_default_grad(x) - f_accurate_grad(x))
    self.assertTrue(jnp.any(diff > 0))

  @parameterized.named_parameters(
      *generate_test_cases(["log", "log1p", "logistic", "tanh"])
  )
  def test_grad_unchanged(self, op, high_tol, low_tol, x, min_error_val):
    @jax.jit
    def f(x):
      return jnp.sum(op(x))

    f_grad = jax.grad(f)

    @jax.jit
    def f_default(x):
      default_op = op(x, accuracy=low_tol)
      return jnp.sum(default_op)

    f_default_grad = jax.grad(f_default)

    @jax.jit
    def f_accurate(x):
      high_op = op(x)
      return jnp.sum(high_op)

    f_accurate_grad = jax.grad(f_accurate)
    # Accuracy should be carried through to the gradient causing a large diff.
    # Diff between f_default and f_accurate should follow diff(f_grad,f_default_grad).
    expected_diff = abs(f_grad(x) - f_default_grad(x))
    if jnp.all(expected_diff > 0):
      # Don't expect f_accurate_grad and f_default_grad to be equal.
      self.assertFalse(
          jnp.all(abs(f_default_grad(x) - f_accurate_grad(x)) == 0)
      )
    elif jnp.all(expected_diff == 0):
      # f_accurate_grad and f_default_grad should be equal.
      diff = abs(f_default_grad(x) - f_accurate_grad(x))
      self.assertTrue(jnp.all(diff == 0))
    else:
      raise ValueError("Unexpected diff: ", expected_diff)

  @parameterized.named_parameters(*generate_test_cases(["cos", "sin", "tan", "logistic", "cbrt", "sqrt", "rsqrt"]))
  def test_single_impl(self, op, high_tol, low_tol, x, min_error_val):
    @jax.jit
    def f_tol(x):
      return op(x, accuracy=high_tol)

    @jax.jit
    def f(x):
      return op(x)

    diff = abs(f_tol(x) - f(x))
    self.assertTrue(jnp.all(diff == 0))

  @parameterized.named_parameters(
      *generate_test_cases(["cos", "sin", "tan", "logistic", "cbrt", "sqrt", "rsqrt"]))
  def test_default_grad(self, op, high_tol, low_tol, x, min_error_val):
    @jax.jit
    def f_tol(x):
      return jnp.sum(op(x, accuracy=high_tol))

    @jax.jit
    def f(x):
      return jnp.sum(op(x))

    self.assertTrue(jnp.all(abs(jax.grad(f_tol)(x) - jax.grad(f)(x)) == 0))

  def test_invalid_accuracy(self):
    with self.assertRaisesRegex(
        ValueError, "At least one of atol, rtol, or ulps must be set."
    ):
      lax.exp(1.0, accuracy=lax.Tolerance(atol=0.0, rtol=0.0, ulps=0))
    with self.assertRaisesRegex(
        ValueError, "Tolerances must be non-negative."
    ):
      lax.exp(1.0, accuracy=lax.Tolerance(atol=-4e-10, rtol=0.0, ulps=0))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
