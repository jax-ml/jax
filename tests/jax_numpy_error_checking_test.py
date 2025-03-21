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
import jax.numpy as jnp

config.parse_flags_with_absl()


JaxValueError = error_check.JaxValueError


@jtu.with_config(jax_check_tracer_leaks=True)
class JaxNumpyErrorCheckingTests(jtu.JaxTestCase):

  @parameterized.product(jit=[True, False])
  def test_can_raise_nan_error(self, jit):
    x = jnp.arange(4, dtype=jnp.float32) - 1

    f = jnp.log
    if jit:
      f = jax.jit(f)

    with self.assertRaisesRegex(JaxValueError, "NaN"):
      f(x)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
