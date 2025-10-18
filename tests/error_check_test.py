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


import traceback

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import error_check
from jax._src import mesh as mesh_lib
from jax._src import test_util as jtu
import jax.export
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P


JaxValueError = error_check.JaxValueError


config.parse_flags_with_absl()
jtu.request_cpu_devices(4)


# TODO: AOT tests fails with the tracer leak checker.
# Re-enable once https://github.com/jax-ml/jax/issues/27315 is fixed.
# @jtu.with_config(jax_check_tracer_leaks=True)
class ErrorCheckTests(jtu.JaxTestCase):

  @parameterized.product(jit=[True, False])
  def test_error_check(self, jit):
    def f(x):
      error_check.set_error_if(x <= 0, "x must be greater than 0")
      return x + 1

    if jit:
      f = jax.jit(f)

    x = jnp.full((4,), -1, dtype=jnp.int32)
    f(x)
    with self.assertRaisesRegex(JaxValueError, "x must be greater than 0"):
      error_check.raise_if_error()

  @parameterized.product(jit=[True, False])
  def test_error_check_no_error(self, jit):
    def f(x):
      error_check.set_error_if(x <= 0, "x must be greater than 0")
      return x + 1

    if jit:
      f = jax.jit(f)

    x = jnp.full((4,), 1, dtype=jnp.int32)
    f(x)
    error_check.raise_if_error()  # should not raise error

  @parameterized.product(jit=[True, False])
  def test_error_check_should_report_the_first_error(self, jit):
    def f(x):
      error_check.set_error_if(x >= 1, "x must be less than 1 in f")
      return x + 1

    def g(x):
      error_check.set_error_if(x >= 1, "x must be less than 1 in g")
      return x + 1

    if jit:
      f = jax.jit(f)
      g = jax.jit(g)

    x = jnp.full((4,), 0, dtype=jnp.int32)

    x = f(x)  # check passes, so it should not set error
    x = g(x)  # check fails. so it should set error
    _ = f(x)  # check fails, but should not override the error
    with self.assertRaisesRegex(JaxValueError, "x must be less than 1 in g"):
      error_check.raise_if_error()

  @parameterized.product(jit=[True, False])
  def test_raise_if_error_clears_error(self, jit):
    def f(x):
      error_check.set_error_if(x <= 0, "x must be greater than 0 in f")
      return x + 1

    def g(x):
      error_check.set_error_if(x <= 0, "x must be greater than 0 in g")
      return x + 1

    if jit:
      f = jax.jit(f)
      g = jax.jit(g)

    x = jnp.full((4,), -1, dtype=jnp.int32)
    f(x)
    with self.assertRaisesRegex(JaxValueError, "x must be greater than 0 in f"):
      error_check.raise_if_error()

    error_check.raise_if_error()  # should not raise error

    g(x)
    with self.assertRaisesRegex(JaxValueError, "x must be greater than 0 in g"):
      error_check.raise_if_error()

  @parameterized.product(jit=[True, False])
  def test_error_includes_traceback(self, jit):
    def function_that_triggers_error_for_traceback_test(x):
      error_check.set_error_if(  # This line must be included in the traceback.
          x <= 0, "x must be greater than 0"
      )
      return x + 1

    if jit:
      function_that_triggers_error_for_traceback_test = jax.jit(
          function_that_triggers_error_for_traceback_test
      )

    x = jnp.zeros((4,), dtype=jnp.int32)
    function_that_triggers_error_for_traceback_test(x)

    tb_string = ""
    try:
      error_check.raise_if_error()
    except JaxValueError as e:
      tb_string = traceback.format_tb(e.__traceback__)
      tb_string = "".join(tb_string)

    self.assertIn("function_that_triggers_error_for_traceback_test", tb_string)
    self.assertIn("This line must be included in the traceback", tb_string)

  @parameterized.product(jit=[True, False])
  def test_error_check_works_with_cond(self, jit):
    def f(x):
      error_check.set_error_if(x == 0, "x must be non-zero in f")
      return x + 1

    def g(x):
      error_check.set_error_if(x == 0, "x must be non-zero in g")
      return x + 1

    def body(pred, x):
      return jax.lax.cond(pred, f, g, x)

    if jit:
      body = jax.jit(body)

    x = jnp.zeros((4,), dtype=jnp.int32)

    _ = body(jnp.bool_(True), x)
    with self.assertRaisesRegex(JaxValueError, "x must be non-zero in f"):
      error_check.raise_if_error()

    _ = body(jnp.bool_(False), x)
    with self.assertRaisesRegex(JaxValueError, "x must be non-zero in g"):
      error_check.raise_if_error()

  @parameterized.product(jit=[True, False])
  def test_error_check_works_with_while_loop(self, jit):
    def f(x):
      error_check.set_error_if(x >= 10, "x must be less than 10")
      return x + 1

    def body(x):
      return jax.lax.while_loop(lambda x: (x < 10).any(), f, x)

    if jit:
      body = jax.jit(body)

    x = jnp.arange(4, dtype=jnp.int32)
    _ = body(x)
    with self.assertRaisesRegex(JaxValueError, "x must be less than 10"):
      error_check.raise_if_error()

  @parameterized.product(jit=[True, False])
  def test_error_check_works_with_scan(self, jit):
    def f(carry, x):
      error_check.set_error_if(x >= 4, "x must be less than 4")
      return carry + x, x + 1

    def body(init, xs):
      return jax.lax.scan(f, init=init, xs=xs)

    if jit:
      body = jax.jit(body)

    init = jnp.int32(0)
    xs = jnp.arange(5, dtype=jnp.int32)
    _ = body(init, xs)
    with self.assertRaisesRegex(JaxValueError, "x must be less than 4"):
      error_check.raise_if_error()

    xs = jnp.arange(4, dtype=jnp.int32)
    _ = body(init, xs)
    error_check.raise_if_error()  # should not raise error

  @parameterized.product(jit=[True, False])
  def test_raise_if_error_fails_in_traced_context(self, jit):
    def f(x):
      error_check.set_error_if(x <= 0, "x must be greater than 0")
      return x + 1

    if jit:
      f = jax.jit(f)

    x = jnp.full((4,), 1, dtype=jnp.int32)
    f(x)
    with self.assertRaises(
        ValueError,
        msg=(
            "raise_if_error() should not be called within a traced context,"
            " such as within a jitted function."
        ),
    ):
      jax.jit(error_check.raise_if_error)()

  @parameterized.product(jit=[True, False])
  @jtu.with_explicit_mesh((2, 2), ("x", "y"))
  def test_error_check_explicit_mode(self, mesh, jit):
    def f(x):
      error_check.set_error_if(x <= 0, "x must be greater than 0")
      return x + 1

    if jit:
      f = jax.jit(f)

    with error_check.error_checking_context():
      x = jnp.full((4, 4), -1, dtype=jnp.int32)
      f(x)
      with self.assertRaisesRegex(JaxValueError, "x must be greater than 0"):
        error_check.raise_if_error()

      sharding = NamedSharding(mesh, P("x", "y"))
      with error_check.error_checking_context():
        y = jnp.full((4, 4), -1, dtype=jnp.int32, device=sharding)
        f(y)
        with self.assertRaisesRegex(JaxValueError, "x must be greater than 0"):
          error_check.raise_if_error()

      # The unsharded version of `f` should still be able to check errors after
      # exiting the error checking context.
      f(x)
      with self.assertRaisesRegex(JaxValueError, "x must be greater than 0"):
        error_check.raise_if_error()

  @parameterized.product(jit=[True, False])
  @jtu.with_explicit_mesh(
      (2, 2),
      ("x", "y"),
      axis_types=(mesh_lib.AxisType.Auto, mesh_lib.AxisType.Auto),
  )
  @jtu.ignore_warning(
      message=(
          "When at least one mesh axis of `pred` is in auto mode, calling"
          " `set_error_if` will cause implicit communication between devices."
          " To avoid this, consider converting the mesh axis in auto mode to"
          " explicit mode."
      ),
      category=RuntimeWarning,
  )
  def test_error_check_auto_mode(self, jit, mesh):
    def f(x):
      error_check.set_error_if(x <= 0, "x must be greater than 0")
      return x + 1

    if jit:
      f = jax.jit(f)

    with error_check.error_checking_context():
      sharding = NamedSharding(mesh, P("x", "y"))
      x = jnp.full((4, 4), -1, dtype=jnp.int32, device=sharding)
      f(x)
      with self.assertRaisesRegex(JaxValueError, "x must be greater than 0"):
        error_check.raise_if_error()

  def test_error_check_aot(self):
    def run_export():
      def f(x):
        error_check.set_error_if(x <= 0, "x must be greater than 0")
        return x + 1

      f = jax.jit(error_check.wrap_for_export(jax.jit(f)))
      x = jax.ShapeDtypeStruct((), jnp.float32)
      serialized = jax.export.export(f)(x).serialize()
      return serialized

    def run_import(serialized):
      f = jax.export.deserialize(serialized).call
      f = jax.jit(error_check.unwrap_from_import(jax.jit(f)))
      x = jnp.float32(-3.)
      _ = f(x)
      with self.assertRaisesRegex(JaxValueError, "x must be greater than 0"):
        error_check.raise_if_error()

    serialized = run_export()
    run_import(serialized)

  def test_error_check_aot_includes_traceback(self):
    def run_export():
      def function_that_triggers_error_for_traceback_test(x):
        error_check.set_error_if(  # This line must be included in the traceback
            x <= 0, "x must be greater than 0"
        )
        return x + 1

      f = jax.jit(
          error_check.wrap_for_export(
              jax.jit(function_that_triggers_error_for_traceback_test)
          )
      )
      x = jax.ShapeDtypeStruct((), jnp.float32)
      serialized = jax.export.export(f)(x).serialize()
      return serialized

    def run_import(serialized):
      f = jax.export.deserialize(serialized).call
      f = jax.jit(error_check.unwrap_from_import(jax.jit(f)))
      x = jnp.float32(-3.0)
      _ = f(x)

      msg = ""
      try:
        error_check.raise_if_error()
      except JaxValueError as e:
        msg = str(e)

      self.assertIn("function_that_triggers_error_for_traceback_test", msg)
      self.assertIn("This line must be included in the traceback", msg)

    serialized = run_export()
    run_import(serialized)

  def test_error_check_aot_should_not_override_existing_error(self):
    def f1(x):
      error_check.set_error_if(x <= 0, "x must be greater than 0 in f1")
      return x + 1

    def run_export():
      def f2(x):
        error_check.set_error_if(x <= 0, "x must be greater than 0 in f2")
        return x + 1

      f2 = jax.jit(error_check.wrap_for_export(jax.jit(f2)))
      x = jax.ShapeDtypeStruct((), jnp.float32)
      serialized = jax.export.export(f2)(x).serialize()
      return serialized

    def run_import(serialized):
      f2 = jax.export.deserialize(serialized).call
      f2 = jax.jit(error_check.unwrap_from_import(jax.jit(f2)))
      return f2

    x = jnp.float32(-3.)
    _ = f1(x)  # check fails. so it should set error

    serialized = run_export()
    f2 = run_import(serialized)
    _ = f2(x)  # check fails, but should not override the error

    with self.assertRaisesRegex(
        JaxValueError, "x must be greater than 0 in f1"
    ):
      error_check.raise_if_error()


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
