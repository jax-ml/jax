# Copyright 2020 Google LLC
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

from functools import partial
import textwrap

import scipy.optimize as spoptimize

from jax._src import api
from jax._src import tree_util
from jax import jit, vjp
import jax.numpy as jnp
from jax.lax import while_loop, cond
from jax._src.numpy.util import _wraps, _promote_args_inexact
from jax.tree_util import tree_flatten, tree_leaves, tree_unflatten, tree_structure
from jax.util import safe_zip


from typing import Callable, Optional, Any, Tuple, TypeVar, cast

X0Type = TypeVar("X0Type")


FIXED_POINT_LAX_DESC = textwrap.dedent(
    """\
    This function implements a fixed point iteration function with similar API
    to the implementation of `scipy`. Additionally, it implements as additional
    feature the possibility to supply user-specific functions to check the
    convergence of the iteration and to handle a non-converged result. These
    function have to be jax-transformable since they will be jitted in the
    call of :obj:`jax.lax.while_loop`.

    Caution: The jax implementation supports in contrast to the scipy one the
    usage of pytrees as input. If you define own functions for the check of
    convergence or error handling, then these function should support pytrees
    as well.
    """
)


FIXED_POINT_EXTRA_PARAMS = textwrap.dedent(
    """\
    check_converged_func : function, optional
        Jax-transformable function which accepts the previous and the new value
        of one iteration and the tolerance criterion as parameters to determine
        if the fixed point routine is converged. Have to return a boolean array.
        If unset, the default relative error function as in scipy will be used
    not_converged_func : function, optional
        Jax-transformable function which is applied to the result if the fixed
        point routine did not converge within `maxiter` steps. This function can
        e.g. be used to set the result to NaN to indicate this.
    """
)


def _relerr(x_prev, x, eps):
    x_prev = tree_leaves(x_prev)
    x = tree_leaves(x)

    result = jnp.array(True)

    for x_prev_i, x_i in safe_zip(x_prev, x):
        diff = jnp.where(x_prev_i != 0, (x_i - x_prev_i) / x_prev_i, x_i)
        result = result & jnp.all(jnp.abs(diff) < eps)

    return result


def _iteration_body(carry):
    f, conv_func, _, x_prev, count, eps, maxsteps = carry
    return f, conv_func, x_prev, f(x_prev), count + 1, eps, maxsteps


def _del2_body(carry):
    f, conv_func, _, x_prev, count, eps, maxsteps = carry

    x_next_1 = f(x_prev)
    x_next_2 = f(x_next_1)

    x_prev = tree_leaves(x_prev)
    x_next_1 = tree_leaves(x_next_1)
    x_next_2, x_def = tree_flatten(x_next_2)

    def new_x(xp, xn1, xn2):
        d = (xn2 - xn1) - (xn1 - xp)
        x = jnp.where(d != 0, xp - jnp.square(xn1 - xp) / d, xn2)
        return x

    x = [new_x(xp, xn1, xn2) for xp, xn1, xn2 in safe_zip(x_prev, x_next_1, x_next_2)]

    return (
        f,
        conv_func,
        tree_unflatten(x_def, x_next_2),
        tree_unflatten(x_def, x),
        count + 1,
        eps,
        maxsteps,
    )


@_wraps(
    spoptimize.fixed_point,
    module="scipy.optimize",
    lax_description=FIXED_POINT_LAX_DESC,
    extra_params=FIXED_POINT_EXTRA_PARAMS,
)
@partial(api.custom_vjp, nondiff_argnums=(0, 5, 6, 7))
def fixed_point(
    func: Callable,
    x0: X0Type,
    args: Tuple[Any, ...] = (),
    xtol: float = 1e-8,
    maxiter: int = 500,
    method: str = "del2",
    check_converged_func: Optional[Callable] = None,
    not_converged_func: Optional[Callable] = None,
) -> X0Type:
    xtol, maxiter = _promote_args_inexact("fixed_point", xtol, maxiter)

    if check_converged_func is None:
        check_converged_func = tree_util.Partial(_relerr)
    else:
        check_converged_func = tree_util.Partial(check_converged_func)

    func_with_args = tree_util.Partial(lambda x: func(x, *args))

    x0_def = tree_structure(x0)
    func_out_def = tree_structure(func_with_args(x0))
    if x0_def != func_out_def:
        raise TypeError(
            f"Tree structure of output of the function ({func_out_def}) "
            "does not match the structure of the initial guess x0 "
            f"({x0_def})."
        )

    def cond_func(carry):
        _, conv_func, x_prev, x, count, eps, maxsteps = carry

        return jnp.logical_not(conv_func(x_prev, x, eps)) & (count < maxsteps)

    if method == "iteration":
        body_func = _iteration_body
    elif method == "del2":
        body_func = _del2_body
    else:
        raise ValueError(f"Unknown method '{method}'.")

    _, _, _, x_star, end_count, _, _ = while_loop(
        cond_func,
        body_func,
        (
            func_with_args,
            check_converged_func,
            x0,
            func_with_args(x0),
            0,
            xtol,
            maxiter,
        ),
    )

    if not_converged_func is not None:
        x_star = cond(end_count == maxiter, not_converged_func, lambda x: x, x_star)

    return x_star


def fixed_point_fwd(
    func: Callable,
    x0: X0Type,
    args: Tuple[Any, ...] = (),
    xtol: float = 1e-8,
    maxiter: int = 500,
    method: str = "del2",
    check_converged_func: Optional[Callable] = None,
    not_converged_func: Optional[Callable] = None,
) -> Tuple[X0Type, Tuple[X0Type, X0Type, Tuple[Any, ...], float, int]]:
    x_star = cast(X0Type, fixed_point(
        func,
        x0,
        args,
        xtol,
        maxiter,
        method,
        check_converged_func,
        not_converged_func,
    ))

    return x_star, (x_star, x0, args, xtol, maxiter)


def fixed_point_rev(
    func: Callable,
    method: str,
    check_converged_func: Optional[Callable],
    not_converged_func: Optional[Callable],
    res,
    result_bar,
):
    x_star, x0, args, xtol, maxiter = res

    _, vjp_args = vjp(lambda a: func(x_star, *a), args)
    _, vjp_x_star = vjp(lambda x: func(x, *args), x_star)

    def f_x_star_bar(x, result_bar):
        result_bar, result_bar_def = tree_flatten(result_bar)
        vjp_leaves = tree_leaves(vjp_x_star(x)[0])
        out_leaves = [i + j for i, j in safe_zip(result_bar, vjp_leaves)]
        return tree_unflatten(result_bar_def, out_leaves)

    x_star_bar = fixed_point(
        f_x_star_bar,
        result_bar,
        args=(result_bar,),
        xtol=xtol,
        maxiter=maxiter,
        method=method,
        check_converged_func=check_converged_func,
        not_converged_func=not_converged_func,
    )

    args_bar = vjp_args(x_star_bar)[0]

    x0_leaves, x0_def = tree_flatten(x0)
    x0_zeros = [jnp.zeros_like(i) for i in x0_leaves]

    return (
        tree_unflatten(x0_def, x0_zeros),
        args_bar,
        jnp.zeros_like(xtol),
        jnp.zeros_like(maxiter),
    )


fixed_point.defvjp(fixed_point_fwd, fixed_point_rev)
