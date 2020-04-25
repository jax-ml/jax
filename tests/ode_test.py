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

  @jtu.skip_on_devices("tpu")
  def test_decay(self):
    def decay(y, t, arg1, arg2):
        return -np.sqrt(t) - y + arg1 - np.mean((y + arg2)**2)

    rng = onp.random.RandomState(0)
    arg1 = rng.randn(3)
    arg2 = rng.randn(3)

    def integrate(y0, ts):
      return odeint(decay, y0, ts, arg1, arg2)

    y0 = rng.randn(3)
    ts = np.linspace(0.1, 0.2, 4)

    tol = 1e-1 if num_float_bits(onp.float64) == 32 else 1e-3
    jtu.check_grads(integrate, (y0, ts), modes=["rev"], order=2,
                    rtol=tol, atol=tol)

  @jtu.skip_on_devices("tpu")
  def test_swoop(self):
    def swoop(y, t, arg1, arg2):
      return np.array(y - np.sin(t) - np.cos(t) * arg1 + arg2)

    ts = np.array([0.1, 0.2])
    tol = 1e-1 if num_float_bits(onp.float64) == 32 else 1e-3

    y0 = np.linspace(0.1, 0.9, 10)
    integrate = lambda y0, ts: odeint(swoop, y0, ts, 0.1, 0.2)
    jtu.check_grads(integrate, (y0, ts), modes=["rev"], order=2,
                    rtol=tol, atol=tol)

    big_y0 = np.linspace(1.1, 10.9, 10)
    integrate = lambda y0, ts: odeint(swoop, big_y0, ts, 0.1, 0.3)
    jtu.check_grads(integrate, (y0, ts), modes=["rev"], order=2,
                    rtol=tol, atol=tol)


if __name__ == '__main__':
  absltest.main()
