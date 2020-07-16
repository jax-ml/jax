"""
Handles dispatching array operations to the correct backend library, as well
as converting arrays to backend formats and then potentially storing them as
constants.
"""

import importlib

import numpy

from . import cupy as _cupy
from . import jax as _jax
from . import tensorflow as _tensorflow
from . import theano as _theano
from . import torch as _torch

__all__ = ["get_func", "has_einsum", "has_tensordot", "build_expression", "evaluate_constants", "has_backend"]

# known non top-level imports
_aliases = {
    'dask': 'dask.array',
    'theano': 'theano.tensor',
    'torch': 'opt_einsum.backends.torch',
    'jax': 'jax.numpy',
    'autograd': 'autograd.numpy',
    'mars': 'mars.tensor',
}


def _import_func(func, backend, default=None):
    """Try and import ``{backend}.{func}``.
    If library is installed and func is found, return the func;
    otherwise if default is provided, return default;
    otherwise raise an error.
    """
    try:
        lib = importlib.import_module(_aliases.get(backend, backend))
        return getattr(lib, func) if default is None else getattr(lib, func, default)
    except AttributeError:
        raise AttributeError("{} doesn't seem to provide the function {}".format(backend, func))


# manually cache functions as python2 doesn't support functools.lru_cache
#     other libs will be added to this if needed, but pre-populate with numpy
_cached_funcs = {
    ('tensordot', 'numpy'): numpy.tensordot,
    ('transpose', 'numpy'): numpy.transpose,
    ('einsum', 'numpy'): numpy.einsum,
}


def get_func(func, backend='numpy', default=None):
    """Return ``{backend}.{func}``, e.g. ``numpy.einsum``,
    or a default func if provided. Cache result.
    """
    try:
        return _cached_funcs[func, backend]
    except KeyError:
        fn = _import_func(func, backend, default)
        _cached_funcs[func, backend] = fn
        return fn


# mark libs with einsum, else try to use tensordot/tranpose as much as possible
_has_einsum = {}


def has_einsum(backend):
    """Check if ``{backend}.einsum`` exists, cache result for performance.
    """
    try:
        return _has_einsum[backend]
    except KeyError:
        try:
            get_func('einsum', backend)
            _has_einsum[backend] = True
        except AttributeError:
            _has_einsum[backend] = False

        return _has_einsum[backend]


_has_tensordot = {}


def has_tensordot(backend):
    """Check if ``{backend}.tensordot`` exists, cache result for performance.
    """
    try:
        return _has_tensordot[backend]
    except KeyError:
        try:
            get_func('tensordot', backend)
            _has_tensordot[backend] = True
        except AttributeError:
            _has_tensordot[backend] = False

        return _has_tensordot[backend]


# Dispatch to correct expression backend
#    these are the backends which support explicit to-and-from numpy conversion
CONVERT_BACKENDS = {
    'tensorflow': _tensorflow.build_expression,
    'theano': _theano.build_expression,
    'cupy': _cupy.build_expression,
    'torch': _torch.build_expression,
    'jax': _jax.build_expression,
}

EVAL_CONSTS_BACKENDS = {
    'tensorflow': _tensorflow.evaluate_constants,
    'theano': _theano.evaluate_constants,
    'cupy': _cupy.evaluate_constants,
    'torch': _torch.evaluate_constants,
    'jax': _jax.evaluate_constants,
}


def build_expression(backend, arrays, expr):
    """Build an expression, based on ``expr`` and initial arrays ``arrays``,
    that evaluates using backend ``backend``.
    """
    return CONVERT_BACKENDS[backend](arrays, expr)


def evaluate_constants(backend, arrays, expr):
    """Convert constant arrays to the correct backend, and perform as much of
    the contraction of ``expr`` with these as possible.
    """
    return EVAL_CONSTS_BACKENDS[backend](arrays, expr)


def has_backend(backend):
    """Checks if the backend is known.
    """
    return backend.lower() in CONVERT_BACKENDS
