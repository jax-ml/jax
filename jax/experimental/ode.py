# Copyright 2018 Google LLC
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

"""JAX-based Dormand-Prince ODE integration with adaptive stepsize.

Integrate systems of ordinary differential equations (ODEs) using the JAX
autograd/diff library and the Dormand-Prince method for adaptive integration
stepsize calculation. Provides improved integration accuracy over fixed
stepsize integration methods.

Adjoint algorithm based on Appendix C of https://arxiv.org/pdf/1806.07366.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import time

import jax
from jax.flatten_util import ravel_pytree
import jax.lax
import jax.numpy as np
import jax.ops
from jax.test_util import check_vjp
import numpy as onp
import scipy.integrate as osp_integrate


@jax.jit
def interp_fit_dopri(y0, y1, k, dt):
  # Fit a polynomial to the results of a Runge-Kutta step.
  dps_c_mid = np.array([
      6025192743 / 30085553152 / 2, 0, 51252292925 / 65400821598 / 2,
      -2691868925 / 45128329728 / 2, 187940372067 / 1594534317056 / 2,
      -1776094331 / 19743644256 / 2, 11237099 / 235043384 / 2])
  y_mid = y0 + dt * np.dot(dps_c_mid, k)
  return np.array(fit_4th_order_polynomial(y0, y1, y_mid, k[0], k[-1], dt))


@jax.jit
def fit_4th_order_polynomial(y0, y1, y_mid, dy0, dy1, dt):
  """Fit fourth order polynomial over function interval.

  Args:
      y0: function value at the start of the interval.
      y1: function value at the end of the interval.
      y_mid: function value at the mid-point of the interval.
      dy0: derivative value at the start of the interval.
      dy1: derivative value at the end of the interval.
      dt: width of the interval.
  Returns:
      Coefficients `[a, b, c, d, e]` for the polynomial
      p = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e
  """
  v = np.stack([dy0, dy1, y0, y1, y_mid])
  a = np.dot(np.hstack([-2. * dt, 2. * dt, np.array([-8., -8., 16.])]), v)
  b = np.dot(np.hstack([5. * dt, -3. * dt, np.array([18., 14., -32.])]), v)
  c = np.dot(np.hstack([-4. * dt, dt, np.array([-11., -5., 16.])]), v)
  d = dt * dy0
  e = y0
  return a, b, c, d, e


@functools.partial(jax.jit, static_argnums=(0,))
def initial_step_size(fun, t0, y0, order, rtol, atol, f0):
  """Empirically choose initial step size.

  Args:
    fun: Function to evaluate like `func(y, t)` to compute the time
      derivative of `y`.
    t0: initial time.
    y0: initial value for the state.
    order: order of interpolation
    rtol: relative local error tolerance for solver.
    atol: absolute local error tolerance for solver.
    f0: initial value for the derivative, computed from `func(t0, y0)`.
  Returns:
    Initial step size for odeint algorithm.

  Algorithm from:
  E. Hairer, S. P. Norsett G. Wanner,
  Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
  """
  scale = atol + np.abs(y0) * rtol
  d0 = np.linalg.norm(y0 / scale)
  d1 = np.linalg.norm(f0 / scale)
  order_pow = (1. / (order + 1.))

  h0 = np.where(np.any(np.asarray([d0 < 1e-5, d1 < 1e-5])),
                1e-6,
                0.01 * d0 / d1)

  y1 = y0 + h0 * f0
  f1 = fun(y1, t0 + h0)
  d2 = np.linalg.norm((f1 - f0) / scale) / h0

  h1 = np.where(np.all(np.asarray([d1 <= 1e-15, d2 <= 1e-15])),
                np.maximum(1e-6, h0 * 1e-3),
                (0.01 / np.max(d1 + d2))**order_pow)

  return np.minimum(100. * h0, h1)


@functools.partial(jax.jit, static_argnums=(0,))
def runge_kutta_step(func, y0, f0, t0, dt):
  """Take an arbitrary Runge-Kutta step and estimate error.

  Args:
      func: Function to evaluate like `func(y, t)` to compute the time
        derivative of `y`.
      y0: initial value for the state.
      f0: initial value for the derivative, computed from `func(t0, y0)`.
      t0: initial time.
      dt: time step.
      alpha, beta, c: Butcher tableau describing how to take the Runge-Kutta
        step.

  Returns:
      y1: estimated function at t1 = t0 + dt
      f1: derivative of the state at t1
      y1_error: estimated error at t1
      k: list of Runge-Kutta coefficients `k` used for calculating these terms.
  """
  # Dopri5 Butcher tableaux
  alpha = np.array([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1., 0])
  beta = np.array(
      [[1 / 5, 0, 0, 0, 0, 0, 0],
       [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
       [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
       [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
       [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
       [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]])
  c_sol = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84,
                    0])
  c_error = np.array([35 / 384 - 1951 / 21600, 0, 500 / 1113 - 22642 / 50085,
                      125 / 192 - 451 / 720, -2187 / 6784 - -12231 / 42400,
                      11 / 84 - 649 / 6300, -1. / 60.])

  def _fori_body_fun(i, val):
    ti = t0 + dt * alpha[i-1]
    yi = y0 + dt * np.dot(beta[i-1, :], val)
    ft = func(yi, ti)
    return jax.ops.index_update(val, jax.ops.index[i, :], ft)

  k = jax.lax.fori_loop(
      1,
      7,
      _fori_body_fun,
      jax.ops.index_update(np.zeros((7, f0.shape[0])), jax.ops.index[0, :], f0))

  y1 = dt * np.dot(c_sol, k) + y0
  y1_error = dt * np.dot(c_error, k)
  f1 = k[-1]
  return y1, f1, y1_error, k


@jax.jit
def error_ratio(error_estimate, rtol, atol, y0, y1):
  err_tol = atol + rtol * np.maximum(np.abs(y0), np.abs(y1))
  err_ratio = error_estimate / err_tol
  return np.mean(err_ratio**2)


@jax.jit
def optimal_step_size(last_step,
                      mean_error_ratio,
                      safety=0.9,
                      ifactor=10.0,
                      dfactor=0.2,
                      order=5.0):
  """Compute optimal Runge-Kutta stepsize."""
  mean_error_ratio = np.max(mean_error_ratio)
  dfactor = np.where(mean_error_ratio < 1,
                     1.0,
                     dfactor)

  err_ratio = np.sqrt(mean_error_ratio)
  factor = np.maximum(1.0 / ifactor,
                      np.minimum(err_ratio**(1.0 / order) / safety,
                                 1.0 / dfactor))
  return np.where(mean_error_ratio == 0,
                  last_step * ifactor,
                  last_step / factor,)


@functools.partial(jax.jit, static_argnums=(0,))
def odeint(ofunc, y0, t, *args, **kwargs):
  """Adaptive stepsize (Dormand-Prince) Runge-Kutta odeint implementation.

  Args:
    ofunc: Function to evaluate `yt = ofunc(y, t, *args)` that
      returns the time derivative of `y`.
    y0: initial value for the state.
    t: Timespan for `ofunc` evaluation like `np.linspace(0., 10., 101)`.
    *args: Additional arguments to `ofunc` beyond y0 and t.
    **kwargs: Two relevant keyword arguments:
      'rtol': Relative local error tolerance for solver.
      'atol': Absolute local error tolerance for solver.

  Returns:
    Integrated system values at each timepoint.
  """
  rtol = kwargs.get('rtol', 1.4e-8)
  atol = kwargs.get('atol', 1.4e-8)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _fori_body_fun(func, i, val):
    """Internal fori_loop body to interpolate an integral at each timestep."""
    t, cur_y, cur_f, cur_t, dt, last_t, interp_coeff, solution = val
    cur_y, cur_f, cur_t, dt, last_t, interp_coeff = jax.lax.while_loop(
        lambda x: x[2] < t[i],
        functools.partial(_while_body_fun, func),
        (cur_y, cur_f, cur_t, dt, last_t, interp_coeff))

    relative_output_time = (t[i] - last_t) / (cur_t - last_t)
    out_x = np.polyval(interp_coeff, relative_output_time)

    return (t, cur_y, cur_f, cur_t, dt, last_t, interp_coeff,
            jax.ops.index_update(solution,
                                 jax.ops.index[i, :],
                                 out_x))

  @functools.partial(jax.jit, static_argnums=(0,))
  def _while_body_fun(func, x):
    """Internal while_loop body to determine interpolation coefficients."""
    cur_y, cur_f, cur_t, dt, last_t, interp_coeff = x
    next_t = cur_t + dt
    next_y, next_f, next_y_error, k = runge_kutta_step(
        func, cur_y, cur_f, cur_t, dt)
    error_ratios = error_ratio(next_y_error, rtol, atol, cur_y, next_y)
    new_interp_coeff = interp_fit_dopri(cur_y, next_y, k, dt)
    dt = optimal_step_size(dt, error_ratios)

    new_rav, unravel = ravel_pytree(
        (next_y, next_f, next_t, dt, cur_t, new_interp_coeff))
    old_rav, _ = ravel_pytree(
        (cur_y, cur_f, cur_t, dt, last_t, interp_coeff))

    return unravel(np.where(np.all(error_ratios <= 1.),
                            new_rav,
                            old_rav))

  func = lambda y, t: ofunc(y, t, *args)
  f0 = func(y0, t[0])
  dt = initial_step_size(func, t[0], y0, 4, rtol, atol, f0)
  interp_coeff = np.array([y0] * 5)

  return jax.lax.fori_loop(1,
                           t.shape[0],
                           functools.partial(_fori_body_fun, func),
                           (t, y0, f0, t[0], dt, t[0], interp_coeff,
                            jax.ops.index_update(
                                np.zeros((t.shape[0], y0.shape[0])),
                                jax.ops.index[0, :],
                                y0)))[-1]


def vjp_odeint(ofunc, y0, t, *args, **kwargs):
  """Return a function that calculates `vjp(odeint(func(y, t, *args))`.

  Args:
    ofunc: Function `ydot = ofunc(y, t, *args)` to compute the time
      derivative of `y`.
    y0: initial value for the state.
    t: Timespan for `ofunc` evaluation like `np.linspace(0., 10., 101)`.
    *args: Additional arguments to `ofunc` beyond y0 and t.
    **kwargs: Two relevant keyword arguments:
      'rtol': Relative local error tolerance for solver.
      'atol': Absolute local error tolerance for solver.

  Returns:
    VJP function `vjp = vjp_all(g)` where `yt = ofunc(y, t, *args)`
    and g is used for VJP calculation. To evaluate the gradient w/ the VJP,
    supply `g = np.ones_like(yt)`. To evaluate the reverse Jacobian do a vmap
    over the standard basis of yt.
  """
  rtol = kwargs.get('rtol', 1.4e-8)
  atol = kwargs.get('atol', 1.4e-8)

  flat_args, unravel_args = ravel_pytree(args)
  flat_func = lambda y, t, flat_args: ofunc(y, t, *unravel_args(flat_args))

  @jax.jit
  def aug_dynamics(augmented_state, t, flat_args):
    """Original system augmented with vjp_y, vjp_t and vjp_args."""
    state_len = int(np.floor_divide(
        augmented_state.shape[0] - flat_args.shape[0] - 1, 2))
    y = augmented_state[:state_len]
    adjoint = augmented_state[state_len:2*state_len]
    dy_dt, vjpfun = jax.vjp(flat_func, y, t, flat_args)
    return np.hstack([np.ravel(dy_dt), np.hstack(vjpfun(-adjoint))])

  rev_aug_dynamics = lambda y, t, flat_args: -aug_dynamics(y, -t, flat_args)

  @jax.jit
  def _fori_body_fun(i, val):
    """fori_loop function for VJP calculation."""
    rev_yt, rev_t, rev_tarray, rev_gi, vjp_y, vjp_t0, vjp_args, time_vjp_list = val
    this_yt = rev_yt[i, :]
    this_t = rev_t[i]
    this_tarray = rev_tarray[i, :]
    this_gi = rev_gi[i, :]
    # this is g[i-1, :] when g has been reversed
    this_gim1 = rev_gi[i+1, :]
    state_len = this_yt.shape[0]
    vjp_cur_t = np.dot(flat_func(this_yt, this_t, flat_args), this_gi)
    vjp_t0 = vjp_t0 - vjp_cur_t
    # Run augmented system backwards to the previous observation.
    aug_y0 = np.hstack((this_yt, vjp_y, vjp_t0, vjp_args))
    aug_ans = odeint(rev_aug_dynamics,
                     aug_y0,
                     this_tarray,
                     flat_args,
                     rtol=rtol,
                     atol=atol)
    vjp_y = aug_ans[1][state_len:2*state_len] + this_gim1
    vjp_t0 = aug_ans[1][2*state_len]
    vjp_args = aug_ans[1][2*state_len+1:]
    time_vjp_list = jax.ops.index_update(time_vjp_list, i, vjp_cur_t)
    return rev_yt, rev_t, rev_tarray, rev_gi, vjp_y, vjp_t0, vjp_args, time_vjp_list

  @jax.jit
  def vjp_all(g, yt, t):
    """Calculate the VJP g * Jac(odeint(ofunc, y0, t, *args))."""
    rev_yt = yt[-1::-1, :]
    rev_t = t[-1::-1]
    rev_tarray = -np.array([t[-1:0:-1], t[-2::-1]]).T
    rev_gi = g[-1::-1, :]

    vjp_y = g[-1, :]
    vjp_t0 = 0.
    vjp_args = np.zeros_like(flat_args)
    time_vjp_list = np.zeros_like(t)

    result = jax.lax.fori_loop(0,
                               rev_t.shape[0]-1,
                               _fori_body_fun,
                               (rev_yt,
                                rev_t,
                                rev_tarray,
                                rev_gi,
                                vjp_y,
                                vjp_t0,
                                vjp_args,
                                time_vjp_list))

    time_vjp_list = jax.ops.index_update(result[-1], -1, result[-3])
    vjp_times = np.hstack(time_vjp_list)[::-1]

    return tuple([result[-4], vjp_times] + list(result[-2]))

  primals_out = odeint(flat_func, y0, t, flat_args)
  vjp_fun = lambda g: vjp_all(g, primals_out, t)

  return primals_out, vjp_fun


def build_odeint(ofunc, rtol=1.4e-8, atol=1.4e-8):
  """Return `f(y0, t, args) = odeint(ofunc(y, t, *args), y0, t, args)`.

  Given the function ofunc(y, t, *args), return the jitted function
  `f(y0, t, args) = odeint(ofunc(y, t, *args), y0, t, args)` with
  the VJP of `f` defined using `vjp_odeint`, where:

    `y0` is the initial condition of the ODE integration,
    `t` is the time course of the integration, and
    `*args` are all other arguments to `ofunc`.

  Args:
    ofunc: The function to be wrapped into an ODE integration.
    rtol: relative local error tolerance for solver.
    atol: absolute local error tolerance for solver.

  Returns:
    `f(y0, t, args) = odeint(ofunc(y, t, *args), y0, t, args)`
  """
  ct_odeint = jax.custom_transforms(
      lambda y0, t, *args: odeint(ofunc, y0, t, *args, rtol=rtol, atol=atol))

  v = lambda y0, t, *args: vjp_odeint(ofunc, y0, t, *args, rtol=rtol, atol=atol)
  jax.defvjp_all(ct_odeint, v)

  return jax.jit(ct_odeint)


def my_odeint_grad(fun):
  """Calculate the Jacobian of an odeint."""
  @jax.jit
  def _gradfun(*args, **kwargs):
    ys, pullback = vjp_odeint(fun, *args, **kwargs)
    my_grad = pullback(np.ones_like(ys))
    return my_grad
  return _gradfun


def my_odeint_jacrev(fun):
  """Calculate the Jacobian of an odeint."""
  @jax.jit
  def _jacfun(*args, **kwargs):
    ys, pullback = vjp_odeint(fun, *args, **kwargs)
    my_jac = jax.vmap(pullback)(jax.api._std_basis(ys))
    my_jac = jax.api.tree_map(
        functools.partial(jax.api._unravel_array_into_pytree, ys, 0), my_jac)
    my_jac = jax.api.tree_transpose(
        jax.api.tree_structure(args), jax.api.tree_structure(ys), my_jac)
    return my_jac
  return _jacfun


def nd(f, x, eps=0.0001):
  flat_x, unravel = ravel_pytree(x)
  dim = len(flat_x)
  g = onp.zeros_like(flat_x)
  for i in range(dim):
    d = onp.zeros_like(flat_x)
    d[i] = eps
    g[i] = (f(unravel(flat_x + d)) - f(unravel(flat_x - d))) / (2.0 * eps)
  return g


def test_grad_vjp_odeint():
  """Compare numerical and exact differentiation of a simple odeint."""

  def f(y, t, arg1, arg2):
    return -np.sqrt(t) - y + arg1 - np.mean((y + arg2)**2)

  def onearg_odeint(args):
    return np.sum(
        odeint(f, *args, atol=1e-8, rtol=1e-8))

  dim = 10
  t0 = 0.1
  t1 = 0.2
  y0 = np.linspace(0.1, 0.9, dim)
  arg1 = 0.1
  arg2 = 0.2
  wrap_args = (y0, np.array([t0, t1]), arg1, arg2)

  numerical_grad = nd(onearg_odeint, wrap_args)
  exact_grad, _ = ravel_pytree(my_odeint_grad(f)(*wrap_args))

  assert np.allclose(numerical_grad, exact_grad)


def plot_gradient_field(ax, func, xlimits, ylimits, numticks=30):
  """Plot the gradient field of `func` on `ax`."""
  x = np.linspace(*xlimits, num=numticks)
  y = np.linspace(*ylimits, num=numticks)
  x_mesh, y_mesh = np.meshgrid(x, y)
  zs = jax.vmap(func)(y_mesh.ravel(), x_mesh.ravel())
  z_mesh = zs.reshape(x_mesh.shape)
  ax.quiver(x_mesh, y_mesh, np.ones(z_mesh.shape), z_mesh)
  ax.set_xlim(xlimits)
  ax.set_ylim(ylimits)


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


def benchmark_odeint(fun, y0, tspace, *args):
  """Time performance of JAX odeint method against scipy.integrate.odeint."""
  n_trials = 5
  for k in range(n_trials):
    start = time.time()
    scipy_result = osp_integrate.odeint(fun, y0, tspace, args)
    end = time.time()
    print('scipy odeint elapsed time ({} of {}): {}'.format(
        k+1, n_trials, end-start))
  for k in range(n_trials):
    start = time.time()
    jax_result = odeint(fun, np.array(y0), np.array(tspace), *args)
    jax_result.block_until_ready()
    end = time.time()
    print('JAX odeint elapsed time ({} of {}): {}'.format(
        k+1, n_trials, end-start))
  print('norm(scipy result-jax result): {}'.format(
      np.linalg.norm(np.asarray(scipy_result) - jax_result)))

  return scipy_result, jax_result


def pend_benchmark_odeint():
  _, _ = benchmark_odeint(pend,
                          (onp.pi - 0.1, 0.0),
                          onp.linspace(0., 10., 101),
                          0.25,
                          9.8)


def test_odeint_grad():
  """Test the gradient behavior of various ODE integrations."""
  def _test_odeint_grad(func, *args):
    def onearg_odeint(fargs):
      return np.sum(odeint(func, *fargs))

    numerical_grad = nd(onearg_odeint, args)
    exact_grad, _ = ravel_pytree(my_odeint_grad(func)(*args))
    assert np.allclose(numerical_grad, exact_grad)

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


def test_odeint_vjp():
  """Use check_vjp to check odeint VJP calculations."""

  # check pend()
  y = np.array([np.pi - 0.1, 0.0])
  t = np.linspace(0., 10., 11)
  b = 0.25
  c = 9.8
  wrap_args = (y, t, b, c)
  pend_odeint_wrap = lambda y, t, *args: odeint(pend, y, t, *args)
  pend_vjp_wrap = lambda y, t, *args: vjp_odeint(pend, y, t, *args)
  check_vjp(pend_odeint_wrap, pend_vjp_wrap, wrap_args)

  # check swoop()
  y = np.array([0.1])
  t = np.linspace(0., 10., 11)
  arg1 = 0.1
  arg2 = 0.2
  wrap_args = (y, t, arg1, arg2)
  swoop_odeint_wrap = lambda y, t, *args: odeint(swoop, y, t, *args)
  swoop_vjp_wrap = lambda y, t, *args: vjp_odeint(swoop, y, t, *args)
  check_vjp(swoop_odeint_wrap, swoop_vjp_wrap, wrap_args)

  # decay() check_vjp hangs!


def test_defvjp_all():
  """Use build_odeint to check odeint VJP calculations."""
  n_trials = 5
  swoop_build = build_odeint(swoop)
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


if __name__ == '__main__':

  test_odeint_grad()
  test_odeint_vjp()
