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

import jax
from jax import numpy as jnp
from jax._src import test_util as jtu
from jax._src.public_test_util import _default_eps, numerical_jvp

import numpy as np

jax.config.parse_flags_with_absl()


def _wrong_gradient_fn():
  @jax.custom_jvp
  def bad(x):
    return jnp.sin(x)

  @bad.defjvp
  def bad_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return jnp.sin(x), 2.0 * jnp.cos(x) * dx

  return bad


class PublicTestUtilTest(jtu.JaxTestCase):

  @jtu.sample_product(dtype=[np.float16, jnp.bfloat16])
  def test_check_grads_low_precision(self, dtype):
    # Regression test for https://github.com/jax-ml/jax/issues/7742: the fixed
    # 1e-4 step underflowed to zero for sub-float32 dtypes, so correct
    # gradients failed the check.
    x = jnp.asarray(1.0, dtype)
    jtu.check_grads(jnp.sin, (x,), order=1, modes=("fwd", "rev"))

  @jtu.sample_product(dtype=[np.float16, jnp.bfloat16])
  def test_check_grads_low_precision_catches_wrong_gradient(self, dtype):
    bad = _wrong_gradient_fn()
    x = jnp.asarray(1.0, dtype)
    with self.assertRaises(AssertionError):
      jtu.check_grads(bad, (x,), order=1, modes=("fwd",))

  def test_default_eps_by_dtype(self):
    # The float32-and-wider default must stay exactly 1e-4: downstream test
    # tolerances are tuned to it (see the rollback in 78599e65d1).
    for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
      self.assertEqual(_default_eps((np.asarray(1.0, dtype),)), 1e-4)
    self.assertEqual(_default_eps((np.asarray(1.0, np.float16),)), 0.0625)
    self.assertEqual(_default_eps((np.asarray(1.0, jnp.bfloat16),)), 0.125)
    # Mixed trees resolve to the step of the lowest-precision leaf.
    self.assertEqual(
        _default_eps(
            (np.asarray(1.0, np.float32), np.asarray(1.0, jnp.bfloat16))
        ),
        0.125,
    )
    # Non-float leaves and Python scalars fall back to the default.
    self.assertEqual(_default_eps((np.asarray(1, np.int32),)), 1e-4)
    self.assertEqual(_default_eps((1.0,)), 1e-4)

  def test_numerical_jvp_step_is_nonzero_in_low_precision(self):
    seen = []

    def f(x):
      seen.append(x)
      return x

    x = jnp.bfloat16(1.0)
    numerical_jvp(f, (x,), (jnp.bfloat16(1.0),))
    self.assertNotEqual(seen[0], seen[1])

  def test_explicit_eps_still_honored(self):
    # An explicit eps must win over the dtype-based default; 1e-4 underflows
    # the bfloat16 step, so the numerical side is zero and the check fails.
    x = jnp.bfloat16(1.0)
    with self.assertRaises(AssertionError):
      jtu.check_grads(jnp.sin, (x,), order=1, modes=("fwd",), eps=1e-4)

  def test_failure_message_labels_sides(self):
    bad = _wrong_gradient_fn()
    x = jnp.float32(1.0)
    with self.assertRaisesRegex(
        AssertionError, "DESIRED is the numerical estimate from finite"
    ):
      jtu.check_grads(bad, (x,), order=1, modes=("fwd",))
    with self.assertRaisesRegex(
        AssertionError, "DESIRED with the numerical JVP from finite"
    ):
      jtu.check_grads(bad, (x,), order=1, modes=("rev",))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
