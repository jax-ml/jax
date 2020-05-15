import jax.numpy as np
import jax
from jax.lax import while_loop, cond
from collections import namedtuple

LineSearchResults = namedtuple('LineSearchResults', ['failed', 'nfev', 'ngev', 'k', 'a_k', 'f_k', 'g_k'])


def line_search_nojit(value_and_gradient, position, direction, f_0=None, g_0=None, max_iterations=50, c1=1e-4, c2=0.9):
    def restricted_func(t):
        return value_and_gradient(position + t * direction)

    grad_restricted = jax.grad(lambda t: restricted_func(t)[0])

    state = LineSearchResults(failed=np.array(True), nfev=0, ngev=0, k=0, a_k=1., f_k=None, g_k=None)
    rho_neg = 0.8
    rho_pos = 1.2

    if f_0 is None or g_0 is None:
        f_0, g_0 = value_and_gradient(position)
        state = state._replace(nfev=state.nfev + 1, ngev=state.ngev + 1)
    state = state._replace(f_k=f_0, g_k=g_0)

    while state.failed and state.k < max_iterations:
        f_kp1, g_kp1 = restricted_func(state.a_k)
        # print(f_kp1, g_kp1)
        state = state._replace(nfev=state.nfev + 1, ngev=state.ngev + 1)
        # Wolfe 1 (3.6a)
        wolfe_1 = f_kp1 <= state.f_k + c1 * state.a_k * np.dot(state.g_k, direction)
        # Wolfe 2 (3.7b)
        wolfe_2 = np.abs(np.dot(g_kp1, direction)) <= c2 * np.abs(np.dot(state.g_k, direction))

        # print(state.a_k)
        # print('W1', f_kp1, state.f_k + c1 * state.a_k * np.dot(state.g_k, direction))
        # print('W2', np.abs(np.dot(g_kp1, direction)), c2 * np.abs(np.dot(state.g_k, direction)))

        state = state._replace(failed=~np.logical_and(wolfe_1, wolfe_2), k=state.k + 1)
        if not state.failed:
            # print('break')
            state = state._replace(f_k=f_kp1, g_k=g_kp1)
            break

        u = grad_restricted(state.a_k)
        state = state._replace(ngev=state.ngev + 1)
        if u > 0:
            state = state._replace(a_k=state.a_k * rho_neg)
        else:
            state = state._replace(a_k=state.a_k * rho_pos)

    if state.failed:
        f_kp1, g_kp1 = restricted_func(state.a_k)
        state = state._replace(f_k=f_kp1, g_k=g_kp1, nfev=state.nfev + 1, ngev=state.ngev + 1)
    return state


def line_search(value_and_gradient, position, direction, f_0=None, g_0=None, max_iterations=50, c1=1e-4, c2=0.9):
    def restricted_func(t):
        return value_and_gradient(position + t * direction)

    grad_restricted = jax.grad(lambda t: restricted_func(t)[0])

    state = LineSearchResults(failed=np.array(True), nfev=0, ngev=0, k=0, a_k=1., f_k=None, g_k=None)
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
        wolfe_1 = f_kp1 <= state.f_k + c1 * state.a_k * np.dot(state.g_k, direction)
        # Wolfe 2 (3.7b)
        wolfe_2 = np.abs(np.dot(g_kp1, direction)) <= c2 * np.abs(np.dot(state.g_k, direction))

        state = state._replace(failed=~np.logical_and(wolfe_1, wolfe_2), k=state.k + 1)

        def backtrack(state):
            u = grad_restricted(state.a_k)
            state = state._replace(ngev=state.ngev + 1)
            state = state._replace(
                a_k=cond(u > 0, None, lambda *x: state.a_k * rho_neg, None, lambda *x: state.a_k * rho_pos))
            return state

        def finish(args):
            state, f_kp1, g_kp1 = args
            state = state._replace(f_k=f_kp1, g_k=g_kp1)
            return state

        state = cond(state.failed, state, backtrack, (state, f_kp1, g_kp1), finish)

        return state

    state = while_loop(lambda state: np.logical_and(state.failed, state.k < max_iterations),
                       body,
                       state
                       )

    def maybe_update(state):
        f_kp1, g_kp1 = restricted_func(state.a_k)
        state = state._replace(f_k=f_kp1, g_k=g_kp1, nfev=state.nfev + 1, ngev=state.ngev + 1)
        return state

    state = cond(state.failed, state, maybe_update, state, lambda state: state)

    return state


def test_line_search():
    def f(x):
        return np.sum(x) ** 3

    assert line_search(jax.value_and_grad(f), np.ones(2), np.array([0.5, -0.25])).failed

    def f(x):
        return np.sum(x) ** 3

    assert not line_search(jax.value_and_grad(f), np.ones(2), np.array([-0.5, -0.25])).failed
