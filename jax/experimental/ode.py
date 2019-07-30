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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import numpy as onp

import matplotlib.pyplot as plt

from jax import jit, vmap, vjp
from jax.flatten_util import ravel_pytree
import jax.numpy as np
from jax.config import config
config.update("jax_enable_x64", True)


"""This demo contains an adaptive-step Runge Kutta ODE solver (Dopri5)
   as well as the adjoint sensitivities algorithm for computing
   memory-efficient vector-Jacobian products wrt the solution."""

# Dopri5 Butcher tableaux
alpha=[1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1.]
beta=[np.array([1 / 5]),
      np.array([3 / 40, 9 / 40]),
      np.array([44 / 45, -56 / 15, 32 / 9]),
      np.array([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]),
      np.array([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]),
      np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]),]
c_sol=np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
c_error=np.array([35 / 384 - 1951 / 21600, 0, 500 / 1113 - 22642 / 50085,
    125 / 192 - 451 / 720, -2187 / 6784 - -12231 / 42400, 11 / 84 - 649 / 6300,
    -1. / 60.,])
dps_c_mid = np.array([
    6025192743 / 30085553152 / 2, 0, 51252292925 / 65400821598 / 2, -2691868925 / 45128329728 / 2,
    187940372067 / 1594534317056 / 2, -1776094331 / 19743644256 / 2, 11237099 / 235043384 / 2
])

@jit
def L2_norm(x):
    return np.sqrt(np.sum(x**2))

@jit
def interp_fit_dopri(y0, y1, k, dt):
    # Fit a polynomial to the results of a Runge-Kutta step.
    y_mid = y0 + dt * np.dot(dps_c_mid, k)
    return fit_4th_order_polynomial(y0, y1, y_mid, k[0], k[-1], dt)

@jit
def fit_4th_order_polynomial(y0, y1, y_mid, dy0, dy1, dt):
    """ y0: function value at the start of the interval.
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
    a = np.dot(np.hstack([-2 * dt,  2 * dt, np.array([ -8., -8.,  16.])]), v)
    b = np.dot(np.hstack([ 5 * dt, -3 * dt, np.array([ 18., 14., -32.])]), v)
    c = np.dot(np.hstack([-4 * dt,      dt, np.array([-11., -5.,  16.])]), v)
    d = dt * dy0
    e = y0
    return a, b, c, d, e

def initial_step_size(fun, t0, y0, order, rtol, atol, f0):
    """Empirically choose initial step size.  Algorithm from:
    E. Hairer, S. P. Norsett G. Wanner,
    Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4."""
    scale = atol + np.abs(y0) * rtol
    d0 = L2_norm(y0 / scale)
    d1 = L2_norm(f0 / scale)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    y1 = y0 + h0 * f0
    f1 = fun(y1, t0 + h0)
    d2 = (L2_norm(f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = np.maximum(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / np.max(d1 + d2))**(1. / float(order + 1))

    return np.minimum(100 * h0, h1)


@partial(jit, static_argnums=(0,))
def runge_kutta_step(func, y0, f0, t0, dt):
    """Take an arbitrary Runge-Kutta step and estimate error.
    Args:
        func: Function to evaluate like `func(t, y)` to compute the time derivative
            of `y`.
        y0: initial value for the state.
        f0: initial value for the derivative, computed from `func(t0, y0)`.
        t0: initial time.
        dt: time step.
        alpha, beta, c: Butcher tableau describing how to take the Runge-Kutta step.
    Returns:
        y1: estimated function at t1 = t0 + dt
        f1: derivative of the state at t1
        y1_error: estimated error at t1
        k: list of Runge-Kutta coefficients `k` used for calculating these terms.
    """
    k = np.array([f0])
    for alpha_i, beta_i in zip(alpha, beta):
        ti = t0 + dt * alpha_i
        yi = y0 + dt * np.dot(k.T, beta_i)
        ft = func(yi, ti)
        k = np.append(k, np.array([ft]), axis=0)

    y1       = dt * np.dot(c_sol, k) + y0
    y1_error = dt * np.dot(c_error, k)
    f1 = k[-1]
    return y1, f1, y1_error, k


@jit
def error_ratio(error_estimate, rtol, atol, y0, y1):
    error_tol = atol + rtol * np.maximum(np.abs(y0), np.abs(y1))
    error_ratio = error_estimate / error_tol
    return np.mean(error_ratio**2)

def optimal_step_size(last_step, mean_error_ratio, safety=0.9, ifactor=10.0, dfactor=0.2, order=5):
    mean_error_ratio = np.max(mean_error_ratio)
    if mean_error_ratio == 0:
        return last_step * ifactor
    if mean_error_ratio < 1:
        dfactor = 1.0
    error_ratio = np.sqrt(mean_error_ratio)
    factor = np.maximum(1.0 / ifactor,
                    np.minimum(error_ratio**(1.0 / order) / safety, 1.0 / dfactor))
    return last_step / factor


def odeint(ofunc, y0, t, args=(), rtol=1e-7, atol=1e-9, return_evals=False):

    if len(args) > 0:
        func = lambda y, t: ofunc(y, t, *args)
    else:
        func = ofunc

    # Reverse time if necessary.
    t = np.array(t)
    if t[-1] < t[0]:
        t = -t
        reversed_func = func
        func = lambda y, t: -reversed_func(y, -t)
    assert np.all(t[1:] > t[:-1]), 't must be strictly increasing or decreasing'

    f0 = func(y0, t[0])
    dt = initial_step_size(func, t[0], y0, 4, rtol, atol, f0)
    interp_coeff = np.array([y0] * 5)

    solution = [y0]
    cur_t = t[0]
    cur_y = y0
    cur_f = f0

    if return_evals:
        evals = [(t0, y0, f0)]

    for output_t in t[1:]:
        # Interpolate through to the next time point, integrating as necessary.
        while cur_t < output_t:
            next_t = cur_t + dt
            assert next_t > cur_t, 'underflow in dt {}'.format(dt)

            next_y, next_f, next_y_error, k =\
                runge_kutta_step(func, cur_y, cur_f, cur_t, dt)
            error_ratios = error_ratio(next_y_error, atol, rtol, cur_y, next_y)

            if np.all(error_ratios <= 1):  # Accept the step?
                interp_coeff = interp_fit_dopri(cur_y, next_y, k, dt)
                cur_y = next_y
                cur_f = next_f
                last_t = cur_t
                cur_t = next_t

                if return_evals:
                    evals.append((cur_t, cur_y, cur_f))

            dt = optimal_step_size(dt, error_ratios)

        relative_output_time = (output_t - last_t) / (cur_t - last_t)
        output_y = np.polyval(interp_coeff, relative_output_time)
        solution.append(output_y)
    if return_evals:
        return np.stack(solution), zip(*evals)
    return np.stack(solution)


def plot_gradient_field(ax, func, xlimits, ylimits, numticks=30):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    zs = vmap(func)(Y.ravel(), X.ravel())
    Z = zs.reshape(X.shape)
    ax.quiver(X, Y, np.ones(Z.shape), Z)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)


def test_fwd_back():
    # Run a system forwards then backwards,
    # and check that we end up in the same place.
    D = 10
    t0 = 0.1
    t1 = 2.2
    y0 = np.linspace(0.1, 0.9, D)

    def f(y, t):
        return -np.sqrt(t) - y + 0.1 - np.mean((y + 0.2)**2)

    ys  = odeint(f, y0,     np.array([t0, t1]), atol=1e-8, rtol=1e-8)
    rys = odeint(f, ys[-1], np.array([t1, t0]), atol=1e-8, rtol=1e-8)

    assert np.allclose(y0, rys[-1])



def grad_odeint(yt, func, y0, t, func_args):
    # Func should have signature func(y, t, args)

    T, D = np.shape(yt)
    flat_args, unravel = ravel_pytree(func_args)

    def flat_func(y, t, flat_args):
        return func(y, t, *unravel(flat_args))

    def unpack(x):
        #      y,      vjp_y,      vjp_t,    vjp_args
        return x[0:D], x[D:2 * D], x[2 * D], x[2 * D + 1:]

    @jit
    def augmented_dynamics(augmented_state, t, flat_args):
        # Orginal system augmented with vjp_y, vjp_t and vjp_args.
        y, adjoint, _, _ = unpack(augmented_state)
        dy_dt, vjp_all = vjp(flat_func, y, t, flat_args)
        vjp_a, vjp_t, vjp_args = vjp_all(-adjoint)
        return np.concatenate([dy_dt, vjp_a, vjp_t.reshape(1), vjp_args])

    def vjp_all(g):

        vjp_y = g[-1, :]
        vjp_t0 = 0
        time_vjp_list = []
        vjp_args = np.zeros(np.size(flat_args))

        for i in range(T - 1, 0, -1):

            # Compute effect of moving measurement time.
            vjp_cur_t = np.dot(func(yt[i, :], t[i], *func_args), g[i, :])
            time_vjp_list.append(vjp_cur_t)
            vjp_t0 = vjp_t0 - vjp_cur_t

            # Run augmented system backwards to the previous observation.
            aug_y0 = np.hstack((yt[i, :], vjp_y, vjp_t0, vjp_args))
            aug_ans = odeint(augmented_dynamics, aug_y0, np.stack([t[i], t[i - 1]]), (flat_args,))
            _, vjp_y, vjp_t0, vjp_args = unpack(aug_ans[1])

            # Add gradient from current output.
            vjp_y = vjp_y + g[i - 1, :]

        time_vjp_list.append(vjp_t0)
        vjp_times = np.hstack(time_vjp_list)[::-1]

        return None, vjp_y, vjp_times, unravel(vjp_args)
    return vjp_all


def nd(f, x, eps=0.0001):
    flat_x, unravel = ravel_pytree(x)
    D = len(flat_x)
    g = onp.zeros_like(flat_x)
    for i in range(D):
        d = onp.zeros_like(flat_x)
        d[i] = eps
        g[i] = (f(unravel(flat_x + d)) - f(unravel(flat_x - d))) / (2.0 * eps)
    return g


def test_odeint_vjp():
    D = 10
    t0 = 0.1
    t1 = 0.2
    y0 = np.linspace(0.1, 0.9, D)
    fargs = (0.1, 0.2)
    def f(y, t, arg1, arg2):
        return -np.sqrt(t) - y + arg1 - np.mean((y + arg2)**2)

    def onearg_odeint(args):
        return np.sum(odeint(f, *args, atol=1e-8, rtol=1e-8))
    numerical_grad = nd(onearg_odeint, (y0, np.array([t0, t1]), fargs))

    ys = odeint(f, y0, np.array([t0, t1]), fargs, atol=1e-8, rtol=1e-8)
    ode_vjp = grad_odeint(ys, f, y0, np.array([t0, t1]), fargs)
    g = np.ones_like(ys)
    exact_grad, _ = ravel_pytree(ode_vjp(g))

    assert np.allclose(numerical_grad, exact_grad)


if __name__ == "__main__":

    def f(y, t, arg1, arg2):
        return y - np.sin(t) - np.cos(t) * arg1 + arg2

    t0 = 0.
    t1 = 5.0
    ts = np.linspace(t0, t1, 100)
    y0 = np.array([1.])
    fargs = (1.0, 0.0)

    ys, (t_evals, y_evals, f_evals) =\
        odeint(f, y0, ts, fargs, atol=0.001, rtol=0.001, return_evals=True)

    # Set up figure.
    fig = plt.figure(figsize=(8,6), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)
    plt.cla()
    f_no_args = lambda y, t: f(y, t, *fargs)
    plot_gradient_field(ax, f_no_args, xlimits=[t0, t1], ylimits=[-1.1, 1.1])
    ax.plot(ts, ys, 'g-')
    ax.plot(t_evals, y_evals, 'bo')
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    plt.draw()
    plt.pause(10)

    # Tests
    test_fwd_back()
    test_odeint_vjp()
