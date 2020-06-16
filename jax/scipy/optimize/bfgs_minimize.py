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
  # function at the `xk`. If the search
  # converged, then this is the (local) minimum of
  # the objective function.
  g_k: jnp.ndarray  # A tensor containing the gradient of the objective function at the `final_position`.
  # If the search converged the l2-norm of this tensor should be below the tolerance.
  H_k: jnp.ndarray  # A tensor containing the inverse of the estimated Hessian.


def fmin_bfgs(func, x0, args=(), options=None):
  """
  The BFGS algorithm from
      Algorithm 6.1 from Wright and Nocedal, 'Numerical Optimization', 1999, pg. 136-143

      Notes:
          We utilise boolean arithmetic to avoid jax.cond calls which don't work on accelerators.
          A side effect is that we perform more gradient evaluations than scipy's BFGS
      func: callable
          Function of the form f(x) where x is a flat ndarray and returns a real scalar. The function should be
          composed of operations with vjp defined. If func is jittable then fmin_bfgs is jittable. If func is
          not jittable, then _nojit should be set to True.

      x0: ndarray
          initial variable
      args: tuple, optional
          Extra arguments to pass to func as func(x,*args)
      options: Optional dict of parameters
          maxiter: int
              Maximum number of evaluations
          norm: float
              Order of norm for convergence check. Default inf.
          gtol: flat
              Terminates minimization when |grad|_norm < g_tol
          ls_maxiter: int
              Maximum number of linesearch iterations

  Returns: BFGSResults

  """

  if options is None:
    options = dict()
  maxiter: Optional[int] = options.get('maxiter', None)
  norm: float = options.get('norm', jnp.inf)
  gtol: float = options.get('gtol', 1e-5)
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
    maxiter = jnp.size(x0) * 200

  d = x0.shape[0]

  initial_H = jnp.eye(d)

  def func_with_args(x):
    return func(x, *args)

  value_and_grad = jax.value_and_grad(func_with_args)

  f_0, g_0 = value_and_grad(x0)
  state = state._replace(f_k=f_0, g_k=g_0, H_k=initial_H, nfev=state.nfev + 1, ngev=state.ngev + 1,
                         converged=jnp.linalg.norm(g_0) < gtol)

  def body(state):
    p_k = -(state.H_k @ state.g_k)
    line_search_results = line_search(func_with_args, state.x_k, p_k, old_fval=state.f_k, gfk=state.g_k,
                                      maxiter=ls_maxiter)
    state = state._replace(nfev=state.nfev + line_search_results.nfev,
                           ngev=state.ngev + line_search_results.ngev,
                           failed=line_search_results.failed)
    s_k = line_search_results.a_k * p_k
    x_kp1 = state.x_k + s_k
    f_kp1 = line_search_results.f_k
    g_kp1 = line_search_results.g_k
    y_k = g_kp1 - state.g_k
    rho_k = jnp.reciprocal(y_k @ s_k)

    sy_k = s_k[:, None] * y_k[None, :]
    w = jnp.eye(d) - rho_k * sy_k
    H_kp1 = jnp.where(jnp.isfinite(rho_k),
                      jnp.linalg.multi_dot([w, state.H_k, w.T]) + rho_k * s_k[:, None] * s_k[None, :], state.H_k)

    converged = jnp.linalg.norm(g_kp1, ord=norm) < gtol

    state = state._replace(converged=converged,
                           k=state.k + 1,
                           x_k=x_kp1,
                           f_k=f_kp1,
                           g_k=g_kp1,
                           H_k=H_kp1
                           )

    return state

  state = while_loop(
    lambda state: (~ state.converged) & (~state.failed) & (state.k < maxiter),
    body,
    state)

  return state
