# Copyright 2020 Google LLC
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

from functools import partial

from absl.testing import absltest
import numpy as onp

from jax import jit
from jax import dtypes
from jax import test_util as jtu
import jax.numpy as np
from jax.experimental.ode import odeint

from jax.config import config
config.parse_flags_with_absl()


def num_float_bits(dtype):
  return dtypes.finfo(dtypes.canonicalize_dtype(dtype)).bits


class JetTest(jtu.JaxTestCase):

  @jtu.skip_on_devices("tpu")
  def test_pend_grads(self):
    def pend(y, _, m, g):
      theta, omega = y
      return [omega, -m * omega - g * np.sin(theta)]

    def f(y0, ts, *args):
      return odeint(pend, y0, ts, *args)

    y0 = [np.pi - 0.1, 0.0]
    ts = np.linspace(0., 1., 11)
    args = (0.25, 9.8)

    tol = 1e-1 if num_float_bits(onp.float64) == 32 else 1e-3
    jtu.check_grads(f, (y0, ts, *args), modes=["rev"], order=2,
                    atol=tol, rtol=tol)

  @jtu.skip_on_devices("tpu")
  def test_weird_time_pendulum_grads(self):
    """Test that gradients are correct when the dynamics depend on t."""
    def dynamics(y, t):
      return np.array([y[1] * -t, -1 * y[1] - 9.8 * np.sin(y[0])])

    integrate = partial(odeint, dynamics)

    y0 = [np.pi - 0.1, 0.0]
    ts = np.linspace(0., 1., 11)

    tol = 1e-1 if num_float_bits(onp.float64) == 32 else 1e-3
    jtu.check_grads(integrate, (y0, ts), modes=["rev"], order=2,
                    rtol=tol, atol=tol)


if __name__ == '__main__':
  absltest.main()
