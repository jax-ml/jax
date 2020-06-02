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

import jax
from jax import dtypes
from jax import test_util as jtu
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax.tree_util import tree_map

import scipy.integrate as osp_integrate

from jax.config import config
config.parse_flags_with_absl()


def num_float_bits(dtype):
  return dtypes.finfo(dtypes.canonicalize_dtype(dtype)).bits


class ODETest(jtu.JaxTestCase):

  def check_against_scipy(self, fun, y0, tspace, *args, tol=1e-1):
    y0, tspace = onp.array(y0), onp.array(tspace)
    np_fun = partial(fun, onp)
    scipy_result = jnp.asarray(osp_integrate.odeint(np_fun, y0, tspace, args))

    y0, tspace = jnp.array(y0), jnp.array(tspace)
    jax_fun = partial(fun, jnp)
    jax_result = odeint(jax_fun, y0, tspace, *args)

    self.assertAllClose(jax_result, scipy_result, check_dtypes=False, atol=tol, rtol=tol)

  @jtu.skip_on_devices("tpu")
  def test_pend_grads(self):
    def pend(_np, y, _, m, g):
      theta, omega = y
      return [omega, -m * omega - g * jnp.sin(theta)]

    integrate = partial(odeint, partial(pend, onp))

    y0 = [jnp.pi - 0.1, 0.0]
    ts = jnp.linspace(0., 1., 11)
    args = (0.25, 9.8)

    tol = 1e-1 if num_float_bits(onp.float64) == 32 else 1e-3

    self.check_against_scipy(pend, y0, ts, *args, tol=tol)

    jtu.check_grads(integrate, (y0, ts, *args), modes=["rev"], order=2,
                    atol=tol, rtol=tol)

  @jtu.skip_on_devices("tpu")
  def test_pytree_state(self):
    """Test calling odeint with y(t) values that are pytrees."""
    def dynamics(y, _t):
      return tree_map(jnp.negative, y)

    y0 = (jnp.array(-0.1), jnp.array([[[0.1]]]))
    integrate = partial(odeint, dynamics)
    ts = jnp.linspace(0., 1., 11)
    tol = 1e-1 if num_float_bits(onp.float64) == 32 else 1e-3
    jtu.check_grads(integrate, (y0, ts), modes=["rev"], order=2,
                    atol=tol, rtol=tol)

  @jtu.skip_on_devices("tpu")
  def test_weird_time_pendulum_grads(self):
    """Test that gradients are correct when the dynamics depend on t."""
    def dynamics(_np, y, t):
      return jnp.array([y[1] * -t, -1 * y[1] - 9.8 * jnp.sin(y[0])])

    integrate = partial(odeint, partial(dynamics, onp))

    y0 = [jnp.pi - 0.1, 0.0]
    ts = jnp.linspace(0., 1., 11)

    tol = 1e-1 if num_float_bits(onp.float64) == 32 else 1e-3

    self.check_against_scipy(dynamics, y0, ts, tol=tol)

    jtu.check_grads(integrate, (y0, ts), modes=["rev"], order=2,
                    rtol=tol, atol=tol)

  @jtu.skip_on_devices("tpu")
  def test_decay(self):
    def decay(_np, y, t, arg1, arg2):
        return -jnp.sqrt(t) - y + arg1 - jnp.mean((y + arg2)**2)

    integrate = partial(odeint, partial(decay, onp))

    rng = onp.random.RandomState(0)
    args = (rng.randn(3), rng.randn(3))

    y0 = rng.randn(3)
    ts = jnp.linspace(0.1, 0.2, 4)

    tol = 1e-1 if num_float_bits(onp.float64) == 32 else 1e-3

    self.check_against_scipy(decay, y0, ts, *args, tol=tol)

    jtu.check_grads(integrate, (y0, ts, *args), modes=["rev"], order=2,
                    rtol=tol, atol=tol)

  @jtu.skip_on_devices("tpu")
  def test_swoop(self):
    def swoop(_np, y, t, arg1, arg2):
      return jnp.array(y - jnp.sin(t) - jnp.cos(t) * arg1 + arg2)

    integrate = partial(odeint, partial(swoop, onp))

    ts = jnp.array([0.1, 0.2])
    tol = 1e-1 if num_float_bits(onp.float64) == 32 else 1e-3

    y0 = jnp.linspace(0.1, 0.9, 10)
    args = (0.1, 0.2)
    self.check_against_scipy(swoop, y0, ts, *args, tol=tol)
    jtu.check_grads(integrate, (y0, ts, *args), modes=["rev"], order=2,
                    rtol=tol, atol=tol)

    big_y0 = jnp.linspace(1.1, 10.9, 10)
    args = (0.1, 0.3)
    self.check_against_scipy(swoop, y0, ts, *args, tol=tol)
    jtu.check_grads(integrate, (big_y0, ts, *args), modes=["rev"], order=2,
                    rtol=tol, atol=tol)

  def test_odeint_vmap_grad(self):
    # https://github.com/google/jax/issues/2531

    def dx_dt(x, *args):
      return 0.1 * x

    def f(x, y):
      y0 = jnp.array([x, y])
      t = jnp.array([0., 5.])
      y = odeint(dx_dt, y0, t)
      return y[-1].sum()

    def g(x):
      # Two initial values for the ODE
      y0_arr = jnp.array([[x, 0.1],
                         [x, 0.2]])

      # Run ODE twice
      t = jnp.array([0., 5.])
      y = jax.vmap(lambda y0: odeint(dx_dt, y0, t))(y0_arr)
      return y[:,-1].sum()

    ans = jax.grad(g)(2.)  # don't crash
    expected = jax.grad(f, 0)(2., 0.1) + jax.grad(f, 0)(2., 0.2)

    atol = {onp.float64: 5e-15}
    rtol = {onp.float64: 2e-15}
    self.assertAllClose(ans, expected, check_dtypes=False, atol=atol, rtol=rtol)

  def test_disable_jit_odeint_with_vmap(self):
    # https://github.com/google/jax/issues/2598
    with jax.disable_jit():
      t = jax.numpy.array([0.0, 1.0])
      x0_eval = jax.numpy.zeros((5, 2))
      f = lambda x0: odeint(lambda x, _t: x, x0, t)
      jax.vmap(f)(x0_eval)  # doesn't crash


if __name__ == '__main__':
  absltest.main()
