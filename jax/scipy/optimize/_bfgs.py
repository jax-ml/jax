"""The Broyden-Fletcher-Goldfarb-Shanno minimization algorithm."""
from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import lax
from ._line_search import line_search


class _BFGSResults(NamedTuple):
  """Results from BFGS optimization.

  Parameters:
    converged: True if minimization converged.
    failed: True if line search failed.
    k: the number of iterations of the BFGS update.
    nfev: total number of objective evaluations performed.
    ngev: total number of jacobian evaluations
    nhev: total number of hessian evaluations
    x_k: array containing the last argument value found during the search. If
      the search converged, then this value is the argmin of the objective
      function.
    f_k: array containing the value of the objective function at `x_k`. If the
      search converged, then this is the (local) minimum of the objective
      function.
    g_k: array containing the gradient of the objective function at `x_k`. If
      the search converged the l2-norm of this tensor should be below the
      tolerance.
    H_k: array containing the inverse of the estimated Hessian.
    status: int describing end state.
    line_search_status: int describing line search end state (only means
      something if line search fails).
  """
  converged: bool
  failed: bool
  k: int
  nfev: int
  ngev: int
  nhev: int
  x_k: jnp.ndarray
  f_k: jnp.ndarray
  g_k: jnp.ndarray
  H_k: jnp.ndarray
  status: int
  line_search_status: int


_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=lax.Precision.HIGHEST)


def minimize_bfgs(
    func,
    x0,
    args=(),
    maxiter=None,
    norm=jnp.inf,
    gtol=1e-5,
    line_search_maxiter=10,
) -> _BFGSResults:
  """Minimize a function using BFGS.

  Implements the BFGS algorithm from
    Algorithm 6.1 from Wright and Nocedal, 'Numerical Optimization', 1999, pg.
    136-143.

  Args:
    func: callable
      Function of the form f(x) where x is a flat ndarray and returns a real
      scalar. The function should be composed of operations with vjp defined.
      If func is jittable then fmin_bfgs is jittable. If func is not jittable,
      then _nojit should be set to True.
    x0: ndarray
      Initial variable.
    args: tuple, optional
      Extra arguments to pass to func as func(x,*args).
    maxiter: maximum number of iterations.
    norm: order of norm for convergence check. Default inf.
    gtol: terminates minimization when |grad|_norm < g_tol.
    line_search_maxiter: maximum number of linesearch iterations.
  """

  # Note: we utilise boolean arithmetic to avoid jax.cond calls which don't
  # work on accelerators. A side effect is that we perform more gradient
  # evaluations than scipy's BFGS
  state = _BFGSResults(
      converged=False,
      failed=False,
      k=0,
      nfev=0,
      ngev=0,
      nhev=0,
      x_k=x0,
      f_k=None,
      g_k=None,
      H_k=None,
      status=None,
      line_search_status=jnp.array(0),
  )

  if maxiter is None:
    maxiter = jnp.size(x0) * 200

  d = x0.shape[0]

  initial_H = jnp.eye(d)

  func_with_args = partial(func, *args)
  value_and_grad = jax.value_and_grad(func_with_args)

  f_0, g_0 = value_and_grad(x0)
  state = state._replace(
      f_k=f_0,
      g_k=g_0,
      H_k=initial_H,
      nfev=state.nfev + 1,
      ngev=state.ngev + 1,
      converged=jnp.linalg.norm(g_0, ord=norm) < gtol,
  )

  def body(state):
    p_k = -_dot(state.H_k, state.g_k)
    line_search_results = line_search(
        func_with_args,
        state.x_k,
        p_k,
        old_fval=state.f_k,
        gfk=state.g_k,
        maxiter=line_search_maxiter,
    )
    state = state._replace(
        nfev=state.nfev + line_search_results.nfev,
        ngev=state.ngev + line_search_results.ngev,
        failed=line_search_results.failed,
        line_search_status=line_search_results.status,
    )
    s_k = line_search_results.a_k * p_k
    x_kp1 = state.x_k + s_k
    f_kp1 = line_search_results.f_k
    g_kp1 = line_search_results.g_k
    y_k = g_kp1 - state.g_k
    rho_k = jnp.reciprocal(_dot(y_k, s_k))

    sy_k = s_k[:, jnp.newaxis] * y_k[jnp.newaxis, :]
    w = jnp.eye(d) - rho_k * sy_k
    H_kp1 = (_einsum('ij,jk,lk', w, state.H_k, w)
             + rho_k * s_k[:, jnp.newaxis] * s_k[jnp.newaxis, :])
    H_kp1 = jnp.where(jnp.isfinite(rho_k), H_kp1, state.H_k)
    converged = jnp.linalg.norm(g_kp1, ord=norm) < gtol

    state = state._replace(
        converged=converged,
        k=state.k + 1,
        x_k=x_kp1,
        f_k=f_kp1,
        g_k=g_kp1,
        H_k=H_kp1
    )
    return state

  state = lax.while_loop(
      lambda state: (~state.converged) & (~state.failed) & (state.k < maxiter),
      body,
      state,
  )
  status = jnp.where(
      state.converged,
      jnp.array(0),  # converged
      jnp.where(
          state.k == maxiter,
          jnp.array(1),  # max iters reached
          jnp.where(
              state.failed,
              jnp.array(2) + state.line_search_status, # ls failed (+ reason)
              jnp.array(-1)  # undefined
          )
      )
  )
  state = state._replace(status=status)
  return state
