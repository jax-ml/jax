# Copyright 2019 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from unittest import SkipTest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp

import jax
import jax.numpy as np
from jax import test_util as jtu
from jax.test_util import check_vjp
from jax.flatten_util import ravel_pytree
from jax.experimental import ode

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS


def nd(f, x, eps=0.0001):
  flat_x, unravel = ravel_pytree(x)
  dim = len(flat_x)
  g = onp.zeros_like(flat_x)
  for i in range(dim):
    d = onp.zeros_like(flat_x)
    d[i] = eps
    g[i] = (f(unravel(flat_x + d)) - f(unravel(flat_x - d))) / (2.0 * eps)
  return g


@jax.jit
def pend(y, t, arg1, arg2):
  """Simple pendulum system for odeint testing."""
  del t
  theta, omega = y
  dydt = np.array([omega, -arg1*omega - arg2*np.sin(theta)])
  return dydt


@jax.jit
def swoop(y, t, arg1, arg2):
  return np.array(y - np.sin(t) - np.cos(t) * arg1 + arg2)


@jax.jit
def decay(y, t, arg1, arg2):
  return -np.sqrt(t) - y + arg1 - np.mean((y + arg2)**2)


def my_odeint_grad(fun):
  """Calculate the Jacobian of an odeint."""
  @jax.jit
  def _gradfun(*args, **kwargs):
    ys, pullback = ode.vjp_odeint(fun, *args, **kwargs)
    my_grad = pullback(np.ones_like(ys))
    return my_grad
  return _gradfun


def my_odeint_jacrev(fun):
  """Calculate the Jacobian of an odeint."""
  @jax.jit
  def _jacfun(*args, **kwargs):
    ys, pullback = ode.vjp_odeint(fun, *args, **kwargs)
    my_jac = jax.vmap(pullback)(jax.api._std_basis(ys))
    my_jac = jax.api.tree_map(
        functools.partial(jax.api._unravel_array_into_pytree, ys, 0), my_jac)
    my_jac = jax.api.tree_transpose(
        jax.api.tree_structure(args), jax.api.tree_structure(ys), my_jac)
    return my_jac
  return _jacfun


class ODETest(jtu.JaxTestCase):

  def test_grad_vjp_odeint(self):
    """Compare numerical and exact differentiation of a simple odeint."""

    def f(y, t, arg1, arg2):
      return -np.sqrt(t) - y + arg1 - np.mean((y + arg2)**2)

    def onearg_odeint(args):
      return np.sum(
          ode.odeint(f, *args, atol=1e-8, rtol=1e-8))

    dim = 10
    t0 = 0.1
    t1 = 0.2
    y0 = np.linspace(0.1, 0.9, dim)
    arg1 = 0.1
    arg2 = 0.2
    wrap_args = (y0, np.array([t0, t1]), arg1, arg2)

    numerical_grad = nd(onearg_odeint, wrap_args)
    exact_grad, _ = ravel_pytree(my_odeint_grad(f)(*wrap_args))

    self.assertAllClose(numerical_grad, exact_grad, check_dtypes=False)

  def test_odeint_grad(self):
    """Test the gradient behavior of various ODE integrations."""
    if not FLAGS.jax_enable_x64:
      raise SkipTest("test only runs with x64 enabled")

    def _test_odeint_grad(func, *args):
      def onearg_odeint(fargs):
        return np.sum(ode.odeint(func, *fargs))

      numerical_grad = nd(onearg_odeint, args)
      exact_grad, _ = ravel_pytree(my_odeint_grad(func)(*args))
      self.assertAllClose(numerical_grad, exact_grad, check_dtypes=False)

    ts = np.array((0.1, 0.2))
    y0 = np.linspace(0.1, 0.9, 10)
    big_y0 = np.linspace(1.1, 10.9, 10)

    # check pend()
    for cond in (
        (np.array((onp.pi - 0.1, 0.0)), ts, 0.25, 0.98),
        (np.array((onp.pi * 0.1, 0.0)), ts, 0.1, 0.4),
        ):
      _test_odeint_grad(pend, *cond)

    # check swoop
    for cond in (
        (y0, ts, 0.1, 0.2),
        (big_y0, ts, 0.1, 0.3),
        ):
      _test_odeint_grad(swoop, *cond)

    # check decay
    for cond in (
        (y0, ts, 0.1, 0.2),
        (big_y0, ts, 0.1, 0.3),
        ):
      _test_odeint_grad(decay, *cond)


  def test_odeint_vjp(self):
    """Use check_vjp to check odeint VJP calculations."""

    # check pend()
    y = np.array([np.pi - 0.1, 0.0])
    t = np.linspace(0., 10., 11)
    b = 0.25
    c = 9.8
    wrap_args = (y, t, b, c)
    pend_odeint_wrap = lambda y, t, *args: ode.odeint(pend, y, t, *args)
    pend_vjp_wrap = lambda y, t, *args: ode.vjp_odeint(pend, y, t, *args)
    check_vjp(pend_odeint_wrap, pend_vjp_wrap, wrap_args)

    # check swoop()
    y = np.array([0.1])
    t = np.linspace(0., 10., 11)
    arg1 = 0.1
    arg2 = 0.2
    wrap_args = (y, t, arg1, arg2)
    swoop_odeint_wrap = lambda y, t, *args: ode.odeint(swoop, y, t, *args)
    swoop_vjp_wrap = lambda y, t, *args: ode.vjp_odeint(swoop, y, t, *args)
    check_vjp(swoop_odeint_wrap, swoop_vjp_wrap, wrap_args)

    # decay() check_vjp hangs!

  def test_defvjp_all(self):
    """Use build_odeint to check odeint VJP calculations."""
    n_trials = 5
    swoop_build = ode.build_odeint(swoop)
    jacswoop = jax.jit(jax.jacrev(swoop_build))
    y = np.array([0.1])
    t = np.linspace(0., 2., 11)
    arg1 = 0.1
    arg2 = 0.2
    wrap_args = (y, t, arg1, arg2)
    for k in range(n_trials):
      start = time.time()
      rslt = jacswoop(*wrap_args)
      rslt.block_until_ready()
      end = time.time()
      print('JAX jacrev elapsed time ({} of {}): {}'.format(
          k+1, n_trials, end-start))


if __name__ == "__main__":
  absltest.main()
