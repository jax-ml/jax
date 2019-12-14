""" C implementation of LRU caching.

Provides 2 LRU caching function decorators:

clru_cache - built-in (faster)
           >>> from fastcache import clru_cache
           >>> @clru_cache(maxsize=128,typed=False)
           ... def f(a, b):
           ...     return (a, ) + (b, )
           ...
           >>> type(f)
           >>> <class 'fastcache.clru_cache'>

lru_cache  - python wrapper around clru_cache (slower)
           >>> from fastcache import lru_cache
           >>> @lru_cache(maxsize=128,typed=False)
           ... def f(a, b):
           ...     return (a, ) + (b, )
           ...
           >>> type(f)
           >>> <class 'function'>
"""

__version__ = "1.1.0"


from ._lrucache import clru_cache
from functools import update_wrapper

def lru_cache(maxsize=128, typed=False, state=None, unhashable='error'):
    """Least-recently-used cache decorator.

    If *maxsize* is set to None, the LRU features are disabled and
    the cache can grow without bound.

    If *typed* is True, arguments of different types will be cached
    separately. For example, f(3.0) and f(3) will be treated as distinct
    calls with distinct results.

    If *state* is a list or dict, the items will be incorporated into
    argument hash.

    The result of calling the cached function with unhashable (mutable)
    arguments depends on the value of *unhashable*:

        If *unhashable* is 'error', a TypeError will be raised.

        If *unhashable* is 'warning', a UserWarning will be raised, and
        the wrapped function will be called with the supplied arguments.
        A miss will be recorded in the cache statistics.

        If *unhashable* is 'ignore', the wrapped function will be called
        with the supplied arguments. A miss will will be recorded in
        the cache statistics.

    View the cache statistics named tuple (hits, misses, maxsize, currsize)
    with f.cache_info().  Clear the cache and statistics with
    f.cache_clear(). Access the underlying function with f.__wrapped__.

    See:  http://en.wikipedia.org/wiki/Cache_algorithms#Least_Recently_Used

    """
    def func_wrapper(func):
        _cached_func = clru_cache(maxsize, typed, state, unhashable)(func)

        def wrapper(*args, **kwargs):
            return _cached_func(*args, **kwargs)

        wrapper.__wrapped__ = func
        wrapper.cache_info = _cached_func.cache_info
        wrapper.cache_clear = _cached_func.cache_clear

        return update_wrapper(wrapper,func)

    return func_wrapper

def test(*args):
    import pytest, os
    return not pytest.main([os.path.dirname(os.path.abspath(__file__))] +
                           list(args))
