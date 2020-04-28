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
import unittest

from absl.testing import absltest
import numpy as onp

import jax
from jax import dtypes
from jax import test_util as jtu
import jax.numpy as np
from jax.experimental.ode import odeint

import scipy.integrate as osp_integrate

from jax.config import config
config.parse_flags_with_absl()


def num_float_bits(dtype):
  return dtypes.finfo(dtypes.canonicalize_dtype(dtype)).bits


class ODETest(jtu.JaxTestCase):

  def check_against_scipy(self, fun, y0, tspace, *args, tol=1e-1):
    y0, tspace = onp.array(y0), onp.array(tspace)
    onp_fun = partial(fun, onp)
    scipy_result = np.asarray(osp_integrate.odeint(onp_fun, y0, tspace, args))

    y0, tspace = np.array(y0), np.array(tspace)
    jax_fun = partial(fun, np)
    jax_result = odeint(jax_fun, y0, tspace, *args)

    self.assertAllClose(jax_result, scipy_result, check_dtypes=False, atol=tol, rtol=tol)

  @jtu.skip_on_devices("tpu")
  def test_pend_grads(self):
    def pend(_np, y, _, m, g):
      theta, omega = y
      return [omega, -m * omega - g * _np.sin(theta)]

    integrate = partial(odeint, partial(pend, np))

    y0 = [np.pi - 0.1, 0.0]
    ts = np.linspace(0., 1., 11)
    args = (0.25, 9.8)

    tol = 1e-1 if num_float_bits(onp.float64) == 32 else 1e-3

    self.check_against_scipy(pend, y0, ts, *args, tol=tol)

    jtu.check_grads(integrate, (y0, ts, *args), modes=["rev"], order=2,
                    atol=tol, rtol=tol)

  @jtu.skip_on_devices("tpu")
  def test_weird_time_pendulum_grads(self):
    """Test that gradients are correct when the dynamics depend on t."""
    def dynamics(_np, y, t):
      return _np.array([y[1] * -t, -1 * y[1] - 9.8 * _np.sin(y[0])])

    integrate = partial(odeint, partial(dynamics, np))

    y0 = [np.pi - 0.1, 0.0]
    ts = np.linspace(0., 1., 11)

    tol = 1e-1 if num_float_bits(onp.float64) == 32 else 1e-3

    self.check_against_scipy(dynamics, y0, ts, tol=tol)

    jtu.check_grads(integrate, (y0, ts), modes=["rev"], order=2,
                    rtol=tol, atol=tol)

  @jtu.skip_on_devices("tpu")
  def test_decay(self):
    def decay(_np, y, t, arg1, arg2):
        return -_np.sqrt(t) - y + arg1 - _np.mean((y + arg2)**2)

    integrate = partial(odeint, partial(decay, np))

    rng = onp.random.RandomState(0)
    args = (rng.randn(3), rng.randn(3))

    y0 = rng.randn(3)
    ts = np.linspace(0.1, 0.2, 4)

    tol = 1e-1 if num_float_bits(onp.float64) == 32 else 1e-3

    self.check_against_scipy(decay, y0, ts, *args, tol=tol)

    jtu.check_grads(integrate, (y0, ts, *args), modes=["rev"], order=2,
                    rtol=tol, atol=tol)

  @jtu.skip_on_devices("tpu")
  def test_swoop(self):
    def swoop(_np, y, t, arg1, arg2):
      return _np.array(y - _np.sin(t) - _np.cos(t) * arg1 + arg2)

    integrate = partial(odeint, partial(swoop, np))

    ts = np.array([0.1, 0.2])
    tol = 1e-1 if num_float_bits(onp.float64) == 32 else 1e-3

    y0 = np.linspace(0.1, 0.9, 10)
    args = (0.1, 0.2)
    self.check_against_scipy(swoop, y0, ts, *args, tol=tol)
    jtu.check_grads(integrate, (y0, ts, *args), modes=["rev"], order=2,
                    rtol=tol, atol=tol)

    big_y0 = np.linspace(1.1, 10.9, 10)
    args = (0.1, 0.3)
    self.check_against_scipy(swoop, y0, ts, *args, tol=tol)
    jtu.check_grads(integrate, (big_y0, ts, *args), modes=["rev"], order=2,
                    rtol=tol, atol=tol)

  def test_odeint_vmap_grad(self):
    # https://github.com/google/jax/issues/2531

    def dx_dt(x, *args):
      return 0.1 * x

    def f(x, y):
      y0 = np.array([x, y])
      t = np.array([0., 5.])
      y = odeint(dx_dt, y0, t)
      return y[-1].sum()

    def g(x):
      # Two initial values for the ODE
      y0_arr = np.array([[x, 0.1],
                         [x, 0.2]])

      # Run ODE twice
      t = np.array([0., 5.])
      y = jax.vmap(lambda y0: odeint(dx_dt, y0, t))(y0_arr)
      return y[:,-1].sum()

    ans = jax.grad(g)(2.)  # don't crash
    expected = jax.grad(f, 0)(2., 0.1) + jax.grad(f, 0)(2., 0.2)

    atol = {onp.float64: 5e-15}
    rtol = {onp.float64: 2e-15}
    self.assertAllClose(ans, expected, check_dtypes=False, atol=atol, rtol=rtol)


if __name__ == '__main__':
  absltest.main()
