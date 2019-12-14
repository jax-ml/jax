import pytest
import fastcache
import itertools
import warnings

try:
    itertools.count(start=0, step=-1)
    count = itertools.count
except TypeError:
    def count(start=0, step=1):
        i = step-1
        for j, c in enumerate(itertools.count(start)):
            yield c + i*j

def arg_gen(min=1, max=100, repeat=3):
    for i in range(min, max):
        for r in range(repeat):
            for j, k in zip(range(i), count(i, -1)):
                yield j, k

@pytest.fixture(scope='module', params=[fastcache.clru_cache,
                                        fastcache.lru_cache])
def cache(request):
    param = request.param
    return param


def test_function_attributes(cache):
    """ Simple tests for attribute preservation. """

    def tfunc(a, b):
        """test function docstring."""
        return a + b
    cfunc = cache()(tfunc)
    assert cfunc.__doc__ == tfunc.__doc__
    assert hasattr(cfunc, 'cache_info')
    assert hasattr(cfunc, 'cache_clear')
    assert hasattr(cfunc, '__wrapped__')


def test_function_cache(cache):
    """ Test that cache returns appropriate values. """

    cat_tuples = [True]

    def tfunc(a, b, c=None):
        if (cat_tuples[0] == True):
            return (a, b, c) + (c, a)
        else:
            return 2*a-10*b

    cfunc = cache(maxsize=100, state=cat_tuples)(tfunc)

    for i, j in arg_gen(max=75, repeat=5):
        assert cfunc(i, j) == tfunc(i, j)

    # change extra state
    cat_tuples[0] = False

    for i, j in arg_gen(max=75, repeat=5):
        assert cfunc(i, j) == tfunc(i, j)

    # test dict state
    d = {}
    cfunc = cache(maxsize=100, state=d)(tfunc)
    cfunc(1, 2)
    assert cfunc.cache_info().misses == 1
    d['a'] = 42
    cfunc(1, 2)
    assert cfunc.cache_info().misses == 2
    cfunc(1, 2)
    assert cfunc.cache_info().misses == 2
    assert cfunc.cache_info().hits == 1
    d.clear()
    cfunc(1, 2)
    assert cfunc.cache_info().misses == 2
    assert cfunc.cache_info().hits == 2
    d['a'] = 44
    cfunc(1, 2)
    assert cfunc.cache_info().misses == 3

def test_memory_leaks(cache):
    """ Longer running test to check for memory leaks. """

    def tfunc(a, b, c):
        return (a-1, 2*c) + (10*b-1, a*b, a*b+c)

    cfunc = cache(maxsize=2000)(tfunc)

    for i, j in arg_gen(max=1500, repeat=5):
        assert cfunc(i, j, c=i-j) == tfunc(i, j, c=i-j)

def test_warn_unhashable_args(cache, recwarn):
    """ Function arguments must be hashable. """

    @cache(unhashable='warning')
    def f(a, b):
        return (a, ) + (b, )

    with warnings.catch_warnings() :
        warnings.simplefilter("always")
        assert f([1], 2) == f.__wrapped__([1], 2)
        w = recwarn.pop(UserWarning)
        assert issubclass(w.category, UserWarning)
        assert "Unhashable arguments cannot be cached" in str(w.message)
        assert w.filename
        assert w.lineno


def test_ignore_unhashable_args(cache):
    """ Function arguments must be hashable. """

    @cache(unhashable='ignore')
    def f(a, b):
        return (a, ) + (b, )

    assert f([1], 2) == f.__wrapped__([1], 2)

def test_default_unhashable_args(cache):
    @cache()
    def f(a, b):
        return (a, ) + (b, )

    with pytest.raises(TypeError):
        f([1], 2)

    @cache(unhashable='error')
    def f(a, b):
        pass
    with pytest.raises(TypeError):
        f([1], 2)

def test_state_type(cache):
    """ State must be a list or dict. """
    f = lambda x : x
    with pytest.raises(TypeError):
        cache(state=(1, ))(f)
    with pytest.raises(TypeError):
        cache(state=-1)(f)

def test_typed_False(cache):
    """ Verify typed==False. """

    @cache(typed=False)
    def cfunc(a, b):
        return a+b

    # initialize cache with integer args
    cfunc(1, 2)
    assert cfunc(1, 2) is cfunc(1.0, 2)
    assert cfunc(1, 2) is cfunc(1, 2.0)
    # test keywords
    cfunc(1, b=2)
    assert cfunc(1,b=2) is cfunc(1.0,b=2)
    assert cfunc(1,b=2) is cfunc(1,b=2.0)

def test_typed_True(cache):
    """ Verify typed==True. """

    @cache(typed=True)
    def cfunc(a, b):
        return a+b

    assert cfunc(1, 2) is not cfunc(1.0, 2)
    assert cfunc(1, 2) is not cfunc(1, 2.0)
    # test keywords
    assert cfunc(1,b=2) is not cfunc(1.0,b=2)
    assert cfunc(1,b=2) is not cfunc(1,b=2.0)

def test_dynamic_attribute(cache):
    f = lambda x : x
    cfunc = cache()(f)
    cfunc.new_attr = 5
    assert cfunc.new_attr == 5
