"""The Broyden-Fletcher-Goldfarb-Shanno minimization algorithm.
https://pages.mtu.edu/~struther/Courses/OLD/Sp2013/5630/Jorge_Nocedal_Numerical_optimization_267490.pdf
"""

import jax
import jax.numpy as jnp
from jax.lax import while_loop
from .line_search import line_search
from typing import NamedTuple, Optional


class BFGSResults(NamedTuple):
  converged: bool  # bool, True if minimization converges
  failed: bool  # bool, True if line search fails
  k: int  # The number of iterations of the BFGS update.
  nfev: int  # The total number of objective evaluations performed.
  ngev: int  # total number of jacobian evaluations
  nhev: int  # total number of hessian evaluations
  x_k: jnp.ndarray  # A tensor containing the last argument value found during the search. If the search converged,
  # then this value is the argmin of the objective function.
  f_k: jnp.ndarray  # A tensor containing the value of the objective
  # function at the `position`. If the search
  # converged, then this is the (local) minimum of
  # the objective function.
  g_k: jnp.ndarray  # A tensor containing the gradient of the objective function at the `final_position`.
  # If the search converged the l2-norm of this tensor should be below the tolerance.
  H_k: jnp.ndarray  # A tensor containing the inverse of the estimated Hessian.


def bfgs_minimize(func, x0, options=None, _nojit=False):
  """
  The BFGS algorithm from
      Algorithm 6.1 from Wright and Nocedal, 'Numerical Optimization', 1999, pg. 136-143

      Notes:
          We utilise boolean arithmetic to avoid jax.cond calls which don't work on accelerators.
          A side effect is that we perform more gradient evaluations than scipy's BFGS
      func: callable
          Function of the form f(x) where x is a flat ndarray and returns a real scalar. The function should be
          composed of operations with vjp defined. If func is jittable then bfgs_minimize is jittable. If func is
          not jittable, then _nojit should be set to True.

      x0: ndarray
          initial variable
      options: Optional dict of parameters
          maxiter: int
              Maximum number of evaluations
          g_tol: flat
              Terminates minimization when |grad|_2 < g_tol
          ls_maxiter: int
              Maximum number of linesearch iterations
      _nojit: bool
          Whether to use pythonic control flow so that func without XLA ops can be used. It is also very useful to
          set _nojit=True in order to perform debugging.

  Returns: BFGSResults

  """
  if options is None:
    options = dict()
  maxiter: Optional[int] = options.get('maxiter', None)
  analytic_initial_hessian: bool = options.get('analytic_initial_hessian', False)
  g_tol: float = options.get('g_tol', 1e-5)
  ls_maxiter: int = options.get('ls_maxiter', 10)

  state = BFGSResults(converged=False,
                      failed=False,
                      k=0,
                      nfev=0,
                      ngev=0,
                      nhev=0,
                      x_k=x0,
                      f_k=None,
                      g_k=None,
                      H_k=None)

  if maxiter is None:
    maxiter = jnp.inf

  D = x0.shape[0]

  if analytic_initial_hessian:
    hess = jax.hessian(func, argnums=0)
    initial_B = hess(x0)
    initial_H = jnp.linalg.pinv(initial_B)
    # TODO: experimental, should remove
    if jnp.any(jnp.linalg.eigvals(initial_B) <= 0):
      initial_H = jnp.eye(D)
    state = state._replace(nhev=state.nhev + 1)

  else:
    initial_H = jnp.eye(D)

  value_and_grad = jax.value_and_grad(func)

  f_0, g_0 = value_and_grad(x0)
  state = state._replace(f_k=f_0, g_k=g_0, H_k=initial_H, nfev=state.nfev + 1, ngev=state.ngev + 1,
                         converged=jnp.linalg.norm(g_0) < g_tol)

  def body(state):
    p_k = -jnp.dot(state.H_k, state.g_k)
    line_search_results = line_search(value_and_grad, state.x_k, p_k, f_0=state.f_k, g_0=state.g_k,
                                      max_iterations=ls_maxiter, _nojit=_nojit)
    state = state._replace(nfev=state.nfev + line_search_results.nfev,
                           ngev=state.ngev + line_search_results.ngev,
                           failed=line_search_results.failed)
    s_k = line_search_results.a_k * p_k
    x_kp1 = state.x_k + s_k
    f_kp1 = line_search_results.f_k
    g_kp1 = line_search_results.g_k
    y_k = g_kp1 - state.g_k
    rho_k = jnp.reciprocal(jnp.dot(y_k, s_k))

    sy_k = s_k[:, None] * y_k[None, :]
    w = jnp.eye(D) - rho_k * sy_k
    H_kp1 = jnp.where(jnp.isfinite(rho_k),
                      jnp.dot(jnp.dot(w, state.H_k), w.T) + rho_k * s_k[:, None] * s_k[None, :], state.H_k)

    converged = jnp.linalg.norm(g_kp1) < g_tol

    state = state._replace(converged=converged,
                           k=state.k + 1,
                           x_k=x_kp1,
                           f_k=f_kp1,
                           g_k=g_kp1,
                           H_k=H_kp1
                           )

    return state

  if _nojit:
    while (~ state.converged) & (~state.failed) & (state.k < maxiter):
      state = body(state)
  else:
    state = while_loop(
      lambda state: (~ state.converged) & (~state.failed) & (state.k < maxiter),
      body,
      state)

  return state
