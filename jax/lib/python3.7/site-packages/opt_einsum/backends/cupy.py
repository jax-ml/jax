"""
Required functions for optimized contractions of numpy arrays using cupy.
"""

import numpy as np

from ..sharing import to_backend_cache_wrap

__all__ = ["to_cupy", "build_expression", "evaluate_constants"]


@to_backend_cache_wrap
def to_cupy(array):  # pragma: no cover
    import cupy

    if isinstance(array, np.ndarray):
        return cupy.asarray(array)

    return array


def build_expression(_, expr):  # pragma: no cover
    """Build a cupy function based on ``arrays`` and ``expr``.
    """
    def cupy_contract(*arrays):
        return expr._contract([to_cupy(x) for x in arrays], backend='cupy').get()

    return cupy_contract


def evaluate_constants(const_arrays, expr):  # pragma: no cover
    """Convert constant arguments to cupy arrays, and perform any possible
    constant contractions.
    """
    return expr(*[to_cupy(x) for x in const_arrays], backend='cupy', evaluate_constants=True)
