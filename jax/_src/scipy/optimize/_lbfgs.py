# Copyright 2020 The JAX Authors.
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
"""The Limited-Memory Broyden-Fletcher-Goldfarb-Shanno minimization algorithm."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import numpy as np

from jax._src import api
from jax._src import dtypes
from jax._src import lax
from jax._src import numpy as jnp
from jax._src.numpy import linalg as jnp_linalg
from jax._src.scipy.optimize.dcsrch import line_search_dcsrch as line_search
from jax._src.typing import Array


_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)


class LBFGSResults(NamedTuple):
  """Results from L-BFGS optimization

  Parameters:
    converged: True if minimization converged
    failed: True if non-zero status and not converged
    k: integer number of iterations of the main loop (optimisation steps)
    nfev: integer total number of objective evaluations performed.
    ngev: integer total number of jacobian evaluations
    x_k: array containing the last argument value found during the search. If
      the search converged, then this value is the argmin of the objective
      function.
    f_k: array containing the value of the objective function at `x_k`. If the
      search converged, then this is the (local) minimum of the objective
      function.
    g_k: array containing the gradient of the objective function at `x_k`. If
      the search converged the l2-norm of this tensor should be below the
      tolerance.
    status: integer describing the status:
      0 = nominal  ,  1 = max iters reached  ,  2 = max fun evals reached
      3 = max grad evals reached  ,  4 = insufficient progress (ftol)
      5 = line search failed
    ls_status: integer describing the end status of the last line search
  """
  converged: Array
  failed: Array
  k: int | Array
  nfev: int | Array
  ngev: int | Array
  x_k: Array
  f_k: Array
  g_k: Array
  s_history: Array
  y_history: Array
  rho_history: Array
  gamma: float | Array
  status: int | Array
  ls_status: int | Array
  old_old_fval: float | Array


def _minimize_lbfgs(
    fun: Callable,
    x0: Array,
    maxiter: float | None = None,
    norm=np.inf,
    maxcor: int = 10,
    ftol: float = 2.220446049250313e-09,
    gtol: float = 1e-05,
    maxfun: float | None = None,
    maxgrad: float | None = None,
    maxls: int = 20,
):
  """
  Minimize a function using L-BFGS

  Implements the L-BFGS algorithm from
    Algorithm 7.5 from Wright and Nocedal, 'Numerical Optimization', 1999, pg. 176-185
  And generalizes to complex variables from
     Sorber, L., Barel, M.V. and Lathauwer, L.D., 2012.
     "Unconstrained optimization of real functions in complex variables"
     SIAM Journal on Optimization, 22(3), pp.879-898.

  Args:
    fun: function of the form f(x) where x is a flat ndarray and returns a real scalar.
      The function should be composed of operations with vjp defined.
    x0: initial guess
    maxiter: maximum number of iterations
    norm: order of norm for convergence check. Default inf.
    maxcor: maximum number of metric corrections ("history size")
    ftol: terminates the minimization when `(f_k - f_{k+1}) < ftol`
    gtol: terminates the minimization when `|g_k|_norm < gtol`
    maxfun: maximum number of function evaluations
    maxgrad: maximum number of gradient evaluations
    maxls: maximum number of line search steps (per iteration)

  Returns:
    Optimization results.
  """
  d = len(x0)
  dtype = dtypes.dtype(x0)

  # ensure there is at least one termination condition
  if (maxiter is None) and (maxfun is None) and (maxgrad is None):
    maxiter = d * 200

  # set others to inf, such that >= is supported
  if maxiter is None:
    maxiter = np.inf
  if maxfun is None:
    maxfun = np.inf
  if maxgrad is None:
    maxgrad = np.inf

  # initial evaluation
  f_0, g_0 = api.value_and_grad(fun)(x0)
  state_initial = LBFGSResults(
    converged=jnp.array(False, dtype=bool),
    failed=jnp.array(False, dtype=bool),
    k=0,
    nfev=1,
    ngev=1,
    x_k=x0,
    f_k=f_0,
    g_k=g_0,
    s_history=jnp.zeros((maxcor, d), dtype=dtype),
    y_history=jnp.zeros((maxcor, d), dtype=dtype),
    rho_history=jnp.zeros((maxcor,), dtype=dtype),
    gamma=1.,
    status=0,
    ls_status=0,
    old_old_fval=f_0 + jnp_linalg.norm(g_0) / 2,
  )

  def cond_fun(state: LBFGSResults):
    return (~state.converged) & (~state.failed)

  def body_fun(state: LBFGSResults):
    # find search direction
    p_k = _two_loop_recursion(state)

    # compute initial step size from old_old_fval (same as BFGS)
    dphi0 = jnp.real(_dot(state.g_k, p_k))
    candidate = 1.01 * 2 * (state.f_k - state.old_old_fval) / (dphi0 + 1e-30)
    alpha0 = jnp.where(
      (dphi0 != 0) & (state.f_k < state.old_old_fval),
      jnp.clip(candidate, 1e-10, 1.0),
      1.0,
    )
    # line search
    ls_results = line_search(
      f=fun,
      xk=state.x_k,
      pk=p_k,
      old_fval=state.f_k,
      gfk=state.g_k,
      maxiter=maxls,
      alpha0=alpha0,
    )

    # If line search failed, fall back to a small gradient descent step
    # instead of stopping. This mirrors scipy's fallback behavior.
    gnorm = jnp_linalg.norm(state.g_k)
    fallback_alpha = jnp.where(gnorm > 0, 1e-4 / gnorm, 1e-4)
    safe_alpha = jnp.where(
      ls_results.failed | ~jnp.isfinite(ls_results.a_k) | (ls_results.a_k <= 0),
      fallback_alpha,
      ls_results.a_k,
    )

    # evaluate at next iterate
    ls_ok = (
      ~ls_results.failed & jnp.isfinite(ls_results.a_k) & (ls_results.a_k > 0)
    )
    s_k = jnp.asarray(safe_alpha).astype(p_k.dtype) * p_k
    x_kp1 = state.x_k + s_k
    # reuse line search results when available, only re-evaluate on fallback
    f_kp1, g_kp1 = lax.cond(
      ls_ok,
      lambda _: (ls_results.f_k, ls_results.g_k),
      lambda _: api.value_and_grad(fun)(x_kp1),
      None,
    )
    y_k = g_kp1 - state.g_k
    rho_k_inv = jnp.real(_dot(y_k, s_k))
    rho_k = jnp.where(
      rho_k_inv == 0., jnp.array(1000., dtype=y_k.dtype),
      jnp.reciprocal(rho_k_inv).astype(y_k.dtype),
    )
    y_dot_y = jnp.real(_dot(jnp.conj(y_k), y_k))
    gamma = jnp.where(y_dot_y > 0, rho_k_inv / y_dot_y, state.gamma)

    # replacements for next iteration
    status = jnp.array(0)
    status = jnp.where(state.f_k - f_kp1 < ftol, 4, status)
    status = jnp.where(state.ngev >= maxgrad, 3, status)
    status = jnp.where(state.nfev >= maxfun, 2, status)
    status = jnp.where(state.k >= maxiter, 1, status)

    converged = jnp_linalg.norm(g_kp1, ord=norm) < gtol

    # TODO(jakevdp): use a fixed-point procedure rather than type-casting?
    state = state._replace(
      converged=converged,
      failed=(status > 0) & (~converged),
      k=state.k + 1,
      nfev=state.nfev + ls_results.nfev + jnp.where(ls_ok, 0, 1),
      ngev=state.ngev + ls_results.ngev + jnp.where(ls_ok, 0, 1),
      x_k=x_kp1.astype(state.x_k.dtype),
      f_k=f_kp1.astype(state.f_k.dtype),
      g_k=g_kp1.astype(state.g_k.dtype),
      s_history=_update_history_vectors(history=state.s_history, new=s_k),
      y_history=_update_history_vectors(history=state.y_history, new=y_k),
      rho_history=_update_history_scalars(history=state.rho_history, new=rho_k),
      gamma=gamma.astype(state.g_k.dtype),
      status=jnp.where(converged, 0, status),
      ls_status=ls_results.status,
      old_old_fval=state.f_k,
    )

    return state

  return lax.while_loop(cond_fun, body_fun, state_initial)


def _two_loop_recursion(state: LBFGSResults):
  dtype = state.rho_history.dtype
  his_size = len(state.rho_history)
  curr_size = jnp.where(state.k < his_size, state.k, his_size)
  q = -jnp.conj(state.g_k)
  a_his = jnp.zeros_like(state.rho_history)

  def body_fun1(j, carry):
    i = his_size - 1 - j
    _q, _a_his = carry
    a_i = state.rho_history[i] * _dot(jnp.conj(state.s_history[i]), _q).real.astype(dtype)
    _a_his = _a_his.at[i].set(a_i)
    _q = _q - a_i * jnp.conj(state.y_history[i])
    return _q, _a_his

  q, a_his = lax.fori_loop(0, curr_size, body_fun1, (q, a_his))
  q = state.gamma * q

  def body_fun2(j, _q):
    i = his_size - curr_size + j
    b_i = state.rho_history[i] * _dot(state.y_history[i], _q).real.astype(dtype)
    _q = _q + (a_his[i] - b_i) * state.s_history[i]
    return _q

  q = lax.fori_loop(0, curr_size, body_fun2, q)
  return q


def _update_history_vectors(history, new):
  # TODO(Jakob-Unfried) use rolling buffer instead? See #6053
  return jnp.roll(history, -1, axis=0).at[-1, :].set(new)


def _update_history_scalars(history, new):
  # TODO(Jakob-Unfried) use rolling buffer instead? See #6053
  return jnp.roll(history, -1, axis=0).at[-1].set(new)
