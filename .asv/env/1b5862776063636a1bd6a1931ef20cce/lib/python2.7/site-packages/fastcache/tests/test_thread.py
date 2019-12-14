"""The Python interpreter may switch between threads inbetween bytecode
execution.  Bytecode execution in fastcache may occur during:
(1) Calls to make_key which will call the __hash__ methods of the args and
(2) `PyDict_Get(Set)Item` calls rely on Python comparisons (i.e, __eq__)
    to determine if a match has been found

A good test for threadsafety is then to cache a function which takes user
defined Python objects that have __hash__ and __eq__ methods which live in
Python land rather built-in land.

The test should not only ensure that the correct result is acheived (and no
segfaults) but also assess memory leaks.

The thread switching interval can be altered using sys.setswitchinterval.
"""

class PythonInt:
    """ Wrapper for an integer with python versions of __eq__ and __hash__."""

    def __init__(self, val):
        self.value = val

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        # only compare with other instances of PythonInt
        if not isinstance(other, PythonInt):
            raise TypeError("PythonInt cannot be compared to %s" % type(other))
        return self.value == other.value

from random import randint
import unittest
from fastcache import clru_cache as lru_cache
from threading import Thread
try:
    from sys import setswitchinterval as setinterval
except ImportError:
    from sys import setcheckinterval
    def setinterval(i):
        return setcheckinterval(int(i))


def run_threads(threads):
    for t in threads:
        t.start()
    for t in threads:
        t.join()

CACHE_SIZE=301
FIB=CACHE_SIZE-1
RAND_MIN, RAND_MAX = 1, 10

@lru_cache(maxsize=CACHE_SIZE, typed=False)
def fib(n):
    """Terrible Fibonacci number generator."""
    v = n.value
    return v if v < 2 else fib(PythonInt(v-1)) + fib(PythonInt(v-2))

# establish correct result from single threaded exectution
RESULT = fib(PythonInt(FIB))

def run_fib_with_clear(r):
    """ Run Fibonacci generator r times. """
    for i in range(r):
        if randint(RAND_MIN, RAND_MAX) == RAND_MIN:
            fib.cache_clear()
        res = fib(PythonInt(FIB))
        if RESULT != res:
            raise ValueError("Expected %d, Got %d" % (RESULT, res))

def run_fib_with_stats(r):
    """ Run Fibonacci generator r times. """
    for i in range(r):
        res = fib(PythonInt(FIB))
        if RESULT != res:
            raise ValueError("Expected %d, Got %d" % (RESULT, res))


class Test_Threading(unittest.TestCase):
    """ Threadsafety Tests for lru_cache. """

    def setUp(self):
        setinterval(1e-6)
        self.numthreads = 4
        self.repeat = 1000

    def test_thread_random_cache_clears(self):
        """ randomly clear the cache during calls to fib. """

        threads = [Thread(target=run_fib_with_clear, args=(self.repeat, ))
                   for _ in range(self.numthreads)]
        run_threads(threads)
        # if we have gotten this far no exceptions have been raised
        self.assertEqual(0, 0)

    def test_thread_cache_info(self):
        """ Run thread safety test to make sure the cache statistics
        are correct."""
        fib.cache_clear()
        threads = [Thread(target=run_fib_with_stats, args=(self.repeat, ))
                   for _ in range(self.numthreads)]
        run_threads(threads)

        hits, misses, maxsize, currsize = fib.cache_info()
        self.assertEqual(misses, CACHE_SIZE)
        self.assertEqual(currsize, CACHE_SIZE)
