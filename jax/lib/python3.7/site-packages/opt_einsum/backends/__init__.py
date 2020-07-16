"""
Compute backends for opt_einsum.
"""

# Backends
from .cupy import to_cupy
from .dispatch import (get_func, has_einsum, has_tensordot, build_expression, evaluate_constants, has_backend)
from .tensorflow import to_tensorflow
from .theano import to_theano
from .torch import to_torch

__all__ = [
    'get_func',
    'has_einsum',
    'has_tensordot',
    'build_expression',
    'evaluate_constants',
    'has_backend',
    'to_tensorflow',
    'to_theano',
    'to_cupy',
    'to_torch',
]
