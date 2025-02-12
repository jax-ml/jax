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
from jax._src.error_check import JaxValueError
import jax.numpy as jnp


config.parse_flags_with_absl()


@jtu.with_config(jax_check_tracer_leaks=True)
class ErrorCheckTests(jtu.JaxTestCase):

  @parameterized.named_parameters(("jit", True), ("no_jit", False))
  def test_error_check(self, jit=False):
    def body():
      def f(x):
        error_check.set_error_if((x <= 0).any(), "x must be greater than 0")
        return x + 1

      if jit:
        f = jax.jit(f)

      x = jnp.full((4,), -1, dtype=jnp.int32)
      f(x)
      error_check.raise_if_error()

    self.assertRaisesRegex(JaxValueError, "x must be greater than 0", body)

  @parameterized.named_parameters(("jit", True), ("no_jit", False))
  def test_error_check_no_error(self, jit=False):
    def body():
      def f(x):
        error_check.set_error_if((x <= 0).any(), "x must be greater than 0")
        return x + 1

      if jit:
        f = jax.jit(f)

      x = jnp.full((4,), 1, dtype=jnp.int32)
      f(x)
      error_check.raise_if_error()

    body()  # should not raise error

  @parameterized.named_parameters(("jit", True), ("no_jit", False))
  def test_error_check_should_report_the_first_error(self, jit=False):
    def body():
      def f(x):
        error_check.set_error_if((x >= 1).any(), "x must be less than 1 in f")
        return x + 1

      def g(x):
        error_check.set_error_if((x >= 1).any(), "x must be less than 1 in g")
        return x + 1

      if jit:
        f = jax.jit(f)
        g = jax.jit(g)

      x = jnp.full((4,), 0, dtype=jnp.int32)
      x = f(x)  # no error
      x = g(x)  # error
      _ = f(x)  # error
      error_check.raise_if_error()

    self.assertRaisesRegex(JaxValueError, "x must be less than 1 in g", body)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
