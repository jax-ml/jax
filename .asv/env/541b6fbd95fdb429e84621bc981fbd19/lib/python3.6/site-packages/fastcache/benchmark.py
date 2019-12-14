""" Benchmark against functools.lru_cache.

    Benchmark script from http://bugs.python.org/file28400/lru_cache_bench.py
    with a few modifications.

    Not available for Py < 3.3.
"""
from __future__ import print_function

import sys

if sys.version_info[:2] >= (3, 3):

    import functools
    import fastcache
    import timeit
    from itertools import count

    def _untyped(*args, **kwargs):
        pass

    def _typed(*args, **kwargs):
        pass

    _py_untyped = functools.lru_cache(maxsize=100)(_untyped)
    _c_untyped =  fastcache.clru_cache(maxsize=100)(_untyped)

    _py_typed = functools.lru_cache(maxsize=100, typed=True)(_typed)
    _c_typed =  fastcache.clru_cache(maxsize=100, typed=True)(_typed)

    def _arg_gen(min=1, max=100, repeat=3):
        for i in range(min, max):
            for r in range(repeat):
                for j, k in zip(range(i), count(i, -1)):
                    yield j, k

    def _print_speedup(results):
        print('')
        print('{:9s} {:>6s} {:>6s} {:>6s}'.format('','min', 'mean', 'max'))
        def print_stats(name,off0, off1):
            arr = [py[0]/c[0] for py, c in zip(results[off0::4],
                                              results[off1::4])]
            print('{:9s} {:6.3f} {:6.3f} {:6.3f}'.format(name,
                                                         min(arr),
                                                         sum(arr)/len(arr),
                                                         max(arr)))
        print_stats('untyped', 0, 1)
        print_stats('typed', 2, 3)

    def _print_single_speedup(res=None, init=False):
        if init:
            print('{:29s} {:>8s}'.format('function call', 'speed up'))
        else:
            print('{:32s} {:5.2f}'.format(res[0][1].split('_')[-1],
                                          res[0][0]/res[1][0]), end = ', ')
            print('{:32s} {:5.2f}'.format(res[2][1].split('_')[-1],
                                          res[2][0]/res[3][0]))
    def run():

        print("Test Suite 1 : ", end='\n\n')
        print("Primarily tests cost of function call, hashing and cache hits.")
        print("Benchmark script based on")
        print("    http://bugs.python.org/file28400/lru_cache_bench.py",
              end = '\n\n')

        _print_single_speedup(init=True)

        results = []
        args = ['i', '"spam", i', '"spam", "spam", i',
                'a=i', 'a="spam", b=i', 'a="spam", b="spam", c=i']
        for a in args:
            for f in ['_py_untyped', '_c_untyped', '_py_typed', '_c_typed']:
                s = '%s(%s)' % (f, a)
                t = min(timeit.repeat('''
                for i in range(100):
                    {}
                '''.format(s),
                        setup='from fastcache.benchmark import %s' % f,
                        repeat=10, number=1000))
                results.append([t, s])
            _print_single_speedup(results[-4:])

        _print_speedup(results)

        print("\n\nTest Suite 2 :", end='\n\n')
        print("Tests millions of misses and millions of hits to quantify")
        print("cache behavior when cache is full.", end='\n\n')
        setup = "from fastcache.benchmark import {}\n" + \
                "from fastcache.benchmark import _arg_gen"

        results = []
        for f in ['_py_untyped', '_c_untyped', '_py_typed', '_c_typed']:
            s = '%s(i, j, a="spammy")' % f
            t = min(timeit.repeat('''
            for i, j in _arg_gen():
                %s
            ''' % s, setup=setup.format(f),
                                  repeat=3, number=100))
            results.append([t, s])

        _print_single_speedup(init=True)
        _print_single_speedup(results)
