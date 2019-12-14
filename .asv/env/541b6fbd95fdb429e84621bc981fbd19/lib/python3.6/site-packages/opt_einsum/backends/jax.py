"""
Required functions for optimized contractions of numpy arrays using jax.
"""

import numpy as np

from ..sharing import to_backend_cache_wrap

__all__ = ["build_expression", "evaluate_constants"]

try:
    import jax

    @to_backend_cache_wrap
    @jax.jit
    def to_jax(x):
        return x

except ImportError:
    pass


def build_expression(_, expr):  # pragma: no cover
    """Build a jax function based on ``arrays`` and ``expr``.
    """
    import jax

    jax_expr = jax.jit(expr._contract)

    def jax_contract(*arrays):
        return np.asarray(jax_expr(arrays))

    return jax_contract


def evaluate_constants(const_arrays, expr):  # pragma: no cover
    """Convert constant arguments to jax arrays, and perform any possible
    constant contractions.
    """
    return expr(*[to_jax(x) for x in const_arrays], backend='jax', evaluate_constants=True)
