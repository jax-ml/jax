import jax.numpy as jnp
import jax
from jax.lax import while_loop, cond
from typing import NamedTuple

LineSearchResults = NamedTuple('LineSearchResults',
                               [('failed', bool),  # Were both Wolfe criteria satisfied
                                ('nfev', int),  # Number of functions evaluations
                                ('ngev', int),  # Number of gradients evaluations
                                ('k', int),  # Number of iterations
                                ('a_k', float),  # Step size
                                ('f_k', jnp.ndarray),  # Final function value
                                ('g_k', jnp.ndarray)  # Final gradient value
                                ])


def line_search(value_and_gradient, position, direction, f_0=None, g_0=None, max_iterations=50, c1=1e-4, c2=0.9):
    """
    Performs an inexact line-search. It is a modified backtracking line search. Instead of reducing step size by a
    number < 1 if Wolfe conditions are not met, we check the sign of  u = del_t restricted_func(t).
    If u > 0 then we do normal backtrack, otherwise we search forward. Normal backtracking can fail to satisfy strong
    Wolfe conditions. This extra step costs one extra gradient evaluation. For explaination see figures 3.1--3.4 in
    https://pages.mtu.edu/~struther/Courses/OLD/Sp2013/5630/Jorge_Nocedal_Numerical_optimization_267490.pdf

    The original backtracking algorithim is p. 37.

    Args:
        value_and_gradient: function and gradient
        position: position to search from
        direction: descent direction to search along
        f_0: optionally give starting function value at position
        g_0: optionally give starting gradient at position
        max_iterations: maximum number of searches
        c1, c2: Wolfe criteria numbers from above reference

    Returns: LineSearchResults

    """

    def restricted_func(t):
        return value_and_gradient(position + t * direction)

    grad_restricted = jax.grad(lambda t: restricted_func(t)[0])

    state = LineSearchResults(failed=jnp.array(True), nfev=0, ngev=0, k=0, a_k=1., f_k=None, g_k=None)
    rho_neg = 0.8
    rho_pos = 1.2

    if f_0 is None or g_0 is None:
        f_0, g_0 = value_and_gradient(position)
        state = state._replace(nfev=state.nfev + 1, ngev=state.ngev + 1)
    state = state._replace(f_k=f_0, g_k=g_0)

    def body(state):
        f_kp1, g_kp1 = restricted_func(state.a_k)
        state = state._replace(nfev=state.nfev + 1, ngev=state.ngev + 1)
        # Wolfe 1 (3.6a)
        wolfe_1 = f_kp1 <= state.f_k + c1 * state.a_k * jnp.dot(state.g_k, direction)
        # Wolfe 2 (3.7b)
        wolfe_2 = jnp.abs(jnp.dot(g_kp1, direction)) <= c2 * jnp.abs(jnp.dot(state.g_k, direction))

        state = state._replace(failed=~(wolfe_1 & wolfe_2), k=state.k + 1)

        def backtrack(state):
            # TODO: it may make sense to only do this once on the first iteration.
            # Moreover, can this be taken out of cond?
            u = grad_restricted(state.a_k)
            state = state._replace(ngev=state.ngev + 1)
            # state = state._replace(a_k=cond(u > 0, None, lambda *x: state.a_k * rho_neg,
            # None, lambda *x: state.a_k * rho_pos))
            a_k = state.a_k * jnp.where(u > 0, rho_neg, rho_pos)
            state = state._replace(a_k=a_k)
            return state

        def finish(args):
            state, f_kp1, g_kp1 = args
            state = state._replace(f_k=f_kp1, g_k=g_kp1)
            return state

        state = cond(state.failed, state, backtrack, (state, f_kp1, g_kp1), finish)

        return state

    state = while_loop(lambda state: state.failed & (state.k < max_iterations),
                       body,
                       state
                       )

    def maybe_update(state):
        f_kp1, g_kp1 = restricted_func(state.a_k)
        state = state._replace(f_k=f_kp1, g_k=g_kp1, nfev=state.nfev + 1, ngev=state.ngev + 1)
        return state

    state = cond(state.failed, state, maybe_update, state, lambda state: state)

    return state
