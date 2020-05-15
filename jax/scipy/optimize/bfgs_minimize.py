"""The Broyden-Fletcher-Goldfarb-Shanno minimization algorithm.
https://pages.mtu.edu/~struther/Courses/OLD/Sp2013/5630/Jorge_Nocedal_Numerical_optimization_267490.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
import jax.numpy as np
from jax.lax import while_loop
from .line_search import line_search_nojit, line_search
from functools import partial
import collections

BFGSResults = collections.namedtuple(
    'BFGSResults', [
        'converged',  # bool, True if minimization converges
        'failed',  # bool, True if line search fails
        'k',  # The number of iterations of the BFGS update.
        'nfev',  # The total number of objective evaluations performed.
        'ngev',  # total number of jacobian evaluations
        'nhev',  # total number of hessian evaluations
        'x_k',  # A tensor containing the last argument value found during the search. If the search converged, then
        # this value is the argmin of the objective function.
        'f_k',  # A tensor containing the value of the objective
        # function at the `position`. If the search
        # converged, then this is the (local) minimum of
        # the objective function.
        'g_k',  # A tensor containing the gradient of the
        # objective function at the
        # `final_position`. If the search converged
        # the max-norm of this tensor should be
        # below the tolerance.
        'H_k'  # A tensor containing the inverse of the estimated Hessian.
    ])


def hessian(f, argnums=0, holomorphic=False):
    """
    Hessian for optional initialisation.
    Args:
        f: func
        argnums:
        holomorphic:

    Returns:
        Hessian function
    """
    return jax.jacfwd(jax.jacrev(f, argnums=argnums, holomorphic=holomorphic), argnums=argnums, holomorphic=holomorphic)


def minimize_nojit(func, x0, max_iterations: int = None, analytic_initial_hessian: bool = True, g_tol: float = 1e-8):
    """
    Minimisation with BFGS. Inexact line search is performed satisfying both Wolfe conditions
    (strong version of second one).
    Args:
        func: function to minimise, should take a single flat parameter, similar to numpy
        x0: initial guess of parameter
        max_iterations: int or None which means inf
        analytic_initial_hessian: whether to use JAX to get hessian for the initialisation, otherwise it's eye.
        g_tol: float, stopping criteria is |grad(func)|_2 \lt g_tol

    Returns: BFGSResults

    """
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

    if max_iterations is None:
        max_iterations = np.inf

    D = x0.shape[0]

    if analytic_initial_hessian:
        hess = hessian(func, argnums=0)
        initial_B = hess(x0)
        # TODO: pinc may give pathological behaviour if function not C^2 and hess is all zeros
        initial_H = np.linalg.pinv(initial_B)
        state = state._replace(nhev=state.nhev + 1)
    else:
        initial_H = np.eye(D)

    value_and_grad = jax.value_and_grad(func)

    f_0, g_0 = value_and_grad(x0)
    state = state._replace(f_k=f_0, g_k=g_0, H_k=initial_H, nfev=state.nfev + 1, ngev=state.ngev + 1,
                           converged=np.linalg.norm(g_0) < g_tol)

    while not state.converged and not state.failed and state.k < max_iterations:
        p_k = -np.dot(state.H_k, state.g_k)
        line_search_results = line_search_nojit(value_and_grad, state.x_k, p_k, f_0=state.f_k, g_0=state.g_k)
        state = state._replace(nfev=state.nfev + line_search_results.nfev,
                               ngev=state.ngev + line_search_results.ngev,
                               failed=line_search_results.failed)
        s_k = line_search_results.a_k * p_k
        x_kp1 = state.x_k + s_k
        f_kp1 = line_search_results.f_k
        g_kp1 = line_search_results.g_k
        # print(g_kp1)
        # f_kp1, g_kp1 = value_and_grad(x_kp1)
        # print(g_kp1)
        y_k = g_kp1 - state.g_k
        rho_k = np.reciprocal(np.dot(y_k, s_k))

        sy_k = s_k[:, None] * y_k[None, :]
        w = np.eye(D) - rho_k * sy_k
        H_kp1 = np.dot(np.dot(w, state.H_k), w.T) + rho_k * s_k[:, None] * s_k[None, :]

        converged = np.linalg.norm(g_kp1) < g_tol

        state = state._replace(converged=converged,
                               k=state.k + 1,
                               x_k=x_kp1,
                               f_k=f_kp1,
                               g_k=g_kp1,
                               H_k=H_kp1
                               )
        print(state)
    return state


def minimize(func, x0, max_iterations: int = None, analytic_initial_hessian: bool = True, g_tol: float = 1e-8):
    """
        Minimisation with BFGS. Inexact line search is performed satisfying both Wolfe conditions
        (strong version of second one).
        Args:
            func: function to minimise, should take a single flat parameter, similar to numpy
            x0: initial guess of parameter
            max_iterations: int or None which means inf
            analytic_initial_hessian: whether to use JAX to get hessian for the initialisation, otherwise it's eye.
            g_tol: float, stopping criteria is |grad(func)|_2 \lt g_tol

        Returns: BFGSResults

        """
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

    if max_iterations is None:
        max_iterations = np.inf

    D = x0.shape[0]

    if analytic_initial_hessian:
        hess = hessian(func, argnums=0)
        initial_B = hess(x0)
        initial_H = np.linalg.pinv(initial_B)
        state = state._replace(nhev=state.nhev + 1)
    else:
        initial_H = np.eye(D)

    value_and_grad = jax.value_and_grad(func)

    f_0, g_0 = value_and_grad(x0)
    state = state._replace(f_k=f_0, g_k=g_0, H_k=initial_H, nfev=state.nfev + 1, ngev=state.ngev + 1,
                           converged=np.linalg.norm(g_0) < g_tol)

    def body(state):
        p_k = -np.dot(state.H_k, state.g_k)
        line_search_results = line_search(value_and_grad, state.x_k, p_k, f_0=state.f_k, g_0=state.g_k)
        state = state._replace(nfev=state.nfev + line_search_results.nfev,
                               ngev=state.ngev + line_search_results.ngev,
                               failed=line_search_results.failed)
        s_k = line_search_results.a_k * p_k
        x_kp1 = state.x_k + s_k
        f_kp1 = line_search_results.f_k
        g_kp1 = line_search_results.g_k
        y_k = g_kp1 - state.g_k
        rho_k = np.reciprocal(np.dot(y_k, s_k))

        sy_k = s_k[:, None] * y_k[None, :]
        w = np.eye(D) - rho_k * sy_k
        H_kp1 = np.dot(np.dot(w, state.H_k), w.T) + rho_k * s_k[:, None] * s_k[None, :]

        converged = np.linalg.norm(g_kp1) < g_tol

        state = state._replace(converged=converged,
                               k=state.k + 1,
                               x_k=x_kp1,
                               f_k=f_kp1,
                               g_k=g_kp1,
                               H_k=H_kp1
                               )

        return state

    state = while_loop(
        lambda state: np.logical_and(np.logical_and(~ state.converged, ~state.failed), state.k < max_iterations),
        body,
        state)

    return state


def test_minimize():
    def rosenbrock(x):
        return np.sum(100. * np.diff(x) ** 2 + (1. - x[:-1]) ** 2)

    x0 = np.zeros(2)

    @jax.jit
    def min_op(x0):
        result = minimize(rosenbrock, x0, analytic_initial_hessian=True)
        return result

    jax_res1 = min_op(x0)

    print("Final with analytic initialisation:\n", jax_res1)

    @jax.jit
    def min_op(x0):
        result = minimize(rosenbrock, x0, analytic_initial_hessian=False)
        return result

    jax_res2 = min_op(x0)

    print("Final eye initialisation (like scipy):\n", jax_res2)

    from scipy.optimize import minimize as smin
    import numpy as onp

    def rosenbrock(x):
        return onp.sum(100. * onp.diff(x) ** 2 + (1. - x[:-1]) ** 2)

    scipy_res = smin(rosenbrock, x0, method='BFGS')
    print("Scipy:\n", scipy_res)

    assert np.all(np.isclose(scipy_res.x, jax_res1.x_k))
    assert np.all(np.isclose(scipy_res.x, jax_res2.x_k))
