"""
A module for sharing intermediates between contractions.

Copyright (c) 2018 Uber Technologies
"""

import contextlib
import functools
import numbers
import threading
from collections import Counter, defaultdict

from .parser import alpha_canonicalize, parse_einsum_input

__all__ = [
    "currently_sharing", "get_sharing_cache", "shared_intermediates", "count_cached_ops", "transpose_cache_wrap",
    "einsum_cache_wrap", "to_backend_cache_wrap"
]

_SHARING_STACK = defaultdict(list)

try:
    get_thread_id = threading.get_ident
except AttributeError:

    def get_thread_id():
        return threading.current_thread().ident


def currently_sharing():
    """Check if we are currently sharing a cache -- thread specific.
    """
    return get_thread_id() in _SHARING_STACK


def get_sharing_cache():
    """Return the most recent sharing cache -- thread specific.
    """
    return _SHARING_STACK[get_thread_id()][-1]


def _add_sharing_cache(cache):
    _SHARING_STACK[get_thread_id()].append(cache)


def _remove_sharing_cache():
    tid = get_thread_id()
    _SHARING_STACK[tid].pop()
    if not _SHARING_STACK[tid]:
        del _SHARING_STACK[tid]


@contextlib.contextmanager
def shared_intermediates(cache=None):
    """Context in which contract intermediate results are shared.

    Note that intermediate computations will not be garbage collected until
    1. this context exits, and
    2. the yielded cache is garbage collected (if it was captured).

    Parameters
    ----------
    cache : dict
        If specified, a user-stored dict in which intermediate results will
        be stored. This can be used to interleave sharing contexts.

    Returns
    -------
    cache : dict
        A dictionary in which sharing results are stored. If ignored,
        sharing results will be garbage collected when this context is
        exited. This dict can be passed to another context to resume
        sharing.
    """
    if cache is None:
        cache = {}
    _add_sharing_cache(cache)
    try:
        yield cache
    finally:
        _remove_sharing_cache()


def count_cached_ops(cache):
    """Returns a counter of the types of each op in the cache.
    This is useful for profiling to increase sharing.
    """
    return Counter(key[0] for key in cache.keys())


def _save_tensors(*tensors):
    """Save tensors in the cache to prevent their ids from being recycled.
    This is needed to prevent false cache lookups.
    """
    cache = get_sharing_cache()
    for tensor in tensors:
        cache['tensor', id(tensor)] = tensor


def _memoize(key, fn, *args, **kwargs):
    """Memoize ``fn(*args, **kwargs)`` using the given ``key``.
    Results will be stored in the innermost ``cache`` yielded by
    :func:`shared_intermediates`.
    """
    cache = get_sharing_cache()
    if key in cache:
        return cache[key]
    result = fn(*args, **kwargs)
    cache[key] = result
    return result


def transpose_cache_wrap(transpose):
    """Decorates a ``transpose()`` implementation to be memoized inside a
    :func:`shared_intermediates` context.
    """

    @functools.wraps(transpose)
    def cached_transpose(a, axes, backend='numpy'):
        if not currently_sharing():
            return transpose(a, axes, backend=backend)

        # hash by axes
        _save_tensors(a)
        axes = tuple(axes)
        key = 'transpose', backend, id(a), axes
        return _memoize(key, transpose, a, axes, backend=backend)

    return cached_transpose


def tensordot_cache_wrap(tensordot):
    """Decorates a ``tensordot()`` implementation to be memoized inside a
    :func:`shared_intermediates` context.
    """

    @functools.wraps(tensordot)
    def cached_tensordot(x, y, axes=2, backend='numpy'):
        if not currently_sharing():
            return tensordot(x, y, axes, backend=backend)

        # hash based on the (axes_x,axes_y) form of axes
        _save_tensors(x, y)
        if isinstance(axes, numbers.Number):
            axes = list(range(len(x.shape)))[len(x.shape) - axes:], list(range(len(y.shape)))[:axes]
        axes = tuple(axes[0]), tuple(axes[1])
        key = 'tensordot', backend, id(x), id(y), axes
        return _memoize(key, tensordot, x, y, axes, backend=backend)

    return cached_tensordot


def einsum_cache_wrap(einsum):
    """Decorates an ``einsum()`` implementation to be memoized inside a
    :func:`shared_intermediates` context.
    """

    @functools.wraps(einsum)
    def cached_einsum(*args, **kwargs):
        if not currently_sharing():
            return einsum(*args, **kwargs)

        # hash modulo commutativity by computing a canonical ordering and names
        backend = kwargs.pop('backend', 'numpy')
        equation = args[0]
        inputs, output, operands = parse_einsum_input(args)
        inputs = inputs.split(',')

        _save_tensors(*operands)

        # Build canonical key
        canonical = sorted(zip(inputs, map(id, operands)), key=lambda x: x[1])
        canonical_ids = tuple(id_ for _, id_ in canonical)
        canonical_inputs = ','.join(input_ for input_, _ in canonical)
        canonical_equation = alpha_canonicalize(canonical_inputs + "->" + output)

        key = 'einsum', backend, canonical_equation, canonical_ids
        return _memoize(key, einsum, equation, *operands, backend=backend)

    return cached_einsum


def to_backend_cache_wrap(to_backend=None, constants=False):
    """Decorates an ``to_backend()`` implementation to be memoized inside a
    :func:`shared_intermediates` context (e.g. ``to_cupy``, ``to_torch``).
    """
    # manage the case that decorator is called with args
    if to_backend is None:
        return functools.partial(to_backend_cache_wrap, constants=constants)

    if constants:

        @functools.wraps(to_backend)
        def cached_to_backend(array, constant=False):
            if not currently_sharing():
                return to_backend(array, constant=constant)

            # hash by id
            key = to_backend.__name__, id(array), constant
            return _memoize(key, to_backend, array, constant=constant)

    else:

        @functools.wraps(to_backend)
        def cached_to_backend(array):
            if not currently_sharing():
                return to_backend(array)

            # hash by id
            key = to_backend.__name__, id(array)
            return _memoize(key, to_backend, array)

    return cached_to_backend
