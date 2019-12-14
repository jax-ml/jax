"""
Tests the accuracy of the opt_einsum paths in addition to unit tests for
the various path helper functions.
"""

import itertools

import numpy as np
import pytest

import opt_einsum as oe

explicit_path_tests = {
    'GEMM1': ([set('abd'), set('ac'), set('bdc')], set(''), {
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4
    }),
    'Inner1': ([set('abcd'), set('abc'), set('bc')], set(''), {
        'a': 5,
        'b': 2,
        'c': 3,
        'd': 4
    }),
}

path_edge_tests = [
    ['greedy', 'eb,cb,fb->cef', ((0, 2), (0, 1))],
    ['branch-all', 'eb,cb,fb->cef', ((0, 2), (0, 1))],
    ['branch-2', 'eb,cb,fb->cef', ((0, 2), (0, 1))],
    ['optimal', 'eb,cb,fb->cef', ((0, 2), (0, 1))],
    ['greedy', 'dd,fb,be,cdb->cef', ((0, 3), (0, 1), (0, 1))],
    ['branch-all', 'dd,fb,be,cdb->cef', ((0, 3), (0, 1), (0, 1))],
    ['branch-2', 'dd,fb,be,cdb->cef', ((0, 3), (0, 1), (0, 1))],
    ['optimal', 'dd,fb,be,cdb->cef', ((0, 3), (0, 1), (0, 1))],
    ['greedy', 'bca,cdb,dbf,afc->', ((1, 2), (0, 2), (0, 1))],
    ['branch-all', 'bca,cdb,dbf,afc->', ((1, 2), (0, 2), (0, 1))],
    ['branch-2', 'bca,cdb,dbf,afc->', ((1, 2), (0, 2), (0, 1))],
    ['optimal', 'bca,cdb,dbf,afc->', ((1, 2), (0, 2), (0, 1))],
    ['greedy', 'dcc,fce,ea,dbf->ab', ((1, 2), (0, 1), (0, 1))],
    ['branch-all', 'dcc,fce,ea,dbf->ab', ((1, 2), (0, 2), (0, 1))],
    ['branch-2', 'dcc,fce,ea,dbf->ab', ((1, 2), (0, 2), (0, 1))],
    ['optimal', 'dcc,fce,ea,dbf->ab', ((1, 2), (0, 2), (0, 1))],
]


def check_path(test_output, benchmark, bypass=False):
    if not isinstance(test_output, list):
        return False

    if len(test_output) != len(benchmark):
        return False

    ret = True
    for pos in range(len(test_output)):
        ret &= isinstance(test_output[pos], tuple)
        ret &= test_output[pos] == benchmark[pos]
    return ret


def assert_contract_order(func, test_data, max_size, benchmark):

    test_output = func(test_data[0], test_data[1], test_data[2], max_size)
    assert check_path(test_output, benchmark)


def test_size_by_dict():

    sizes_dict = {}
    for ind, val in zip('abcdez', [2, 5, 9, 11, 13, 0]):
        sizes_dict[ind] = val

    path_func = oe.helpers.compute_size_by_dict

    assert 1 == path_func('', sizes_dict)
    assert 2 == path_func('a', sizes_dict)
    assert 5 == path_func('b', sizes_dict)

    assert 0 == path_func('z', sizes_dict)
    assert 0 == path_func('az', sizes_dict)
    assert 0 == path_func('zbc', sizes_dict)

    assert 104 == path_func('aaae', sizes_dict)
    assert 12870 == path_func('abcde', sizes_dict)


def test_flop_cost():

    size_dict = {v: 10 for v in "abcdef"}

    # Loop over an array
    assert 10 == oe.helpers.flop_count("a", False, 1, size_dict)

    # Hadamard product (*)
    assert 10 == oe.helpers.flop_count("a", False, 2, size_dict)
    assert 100 == oe.helpers.flop_count("ab", False, 2, size_dict)

    # Inner product (+, *)
    assert 20 == oe.helpers.flop_count("a", True, 2, size_dict)
    assert 200 == oe.helpers.flop_count("ab", True, 2, size_dict)

    # Inner product x3 (+, *, *)
    assert 30 == oe.helpers.flop_count("a", True, 3, size_dict)

    # GEMM
    assert 2000 == oe.helpers.flop_count("abc", True, 2, size_dict)


def test_path_optimal():

    test_func = oe.paths.optimal

    test_data = explicit_path_tests['GEMM1']
    assert_contract_order(test_func, test_data, 5000, [(0, 2), (0, 1)])
    assert_contract_order(test_func, test_data, 0, [(0, 1, 2)])


def test_path_greedy():

    test_func = oe.paths.greedy

    test_data = explicit_path_tests['GEMM1']
    assert_contract_order(test_func, test_data, 5000, [(0, 2), (0, 1)])
    assert_contract_order(test_func, test_data, 0, [(0, 1, 2)])


def test_memory_paths():

    expression = "abc,bdef,fghj,cem,mhk,ljk->adgl"

    views = oe.helpers.build_views(expression)

    # Test tiny memory limit
    path_ret = oe.contract_path(expression, *views, optimize="optimal", memory_limit=5)
    assert check_path(path_ret[0], [(0, 1, 2, 3, 4, 5)])

    path_ret = oe.contract_path(expression, *views, optimize="greedy", memory_limit=5)
    assert check_path(path_ret[0], [(0, 1, 2, 3, 4, 5)])

    # Check the possibilities, greedy is capped
    path_ret = oe.contract_path(expression, *views, optimize="optimal", memory_limit=-1)
    assert check_path(path_ret[0], [(0, 3), (0, 4), (0, 2), (0, 2), (0, 1)])

    path_ret = oe.contract_path(expression, *views, optimize="greedy", memory_limit=-1)
    assert check_path(path_ret[0], [(0, 3), (0, 4), (0, 2), (0, 2), (0, 1)])


@pytest.mark.parametrize("alg,expression,order", path_edge_tests)
def test_path_edge_cases(alg, expression, order):
    views = oe.helpers.build_views(expression)

    # Test tiny memory limit
    path_ret = oe.contract_path(expression, *views, optimize=alg)
    assert check_path(path_ret[0], order)


def test_optimal_edge_cases():

    # Edge test5
    expression = 'a,ac,ab,ad,cd,bd,bc->'
    edge_test4 = oe.helpers.build_views(expression, dimension_dict={"a": 20, "b": 20, "c": 20, "d": 20})
    path, path_str = oe.contract_path(expression, *edge_test4, optimize='greedy', memory_limit='max_input')
    assert check_path(path, [(0, 1), (0, 1, 2, 3, 4, 5)])

    path, path_str = oe.contract_path(expression, *edge_test4, optimize='optimal', memory_limit='max_input')
    assert check_path(path, [(0, 1), (0, 1, 2, 3, 4, 5)])


def test_greedy_edge_cases():

    expression = "abc,cfd,dbe,efa"
    dim_dict = {k: 20 for k in expression.replace(",", "")}
    tensors = oe.helpers.build_views(expression, dimension_dict=dim_dict)

    path, path_str = oe.contract_path(expression, *tensors, optimize='greedy', memory_limit='max_input')
    assert check_path(path, [(0, 1, 2, 3)])

    path, path_str = oe.contract_path(expression, *tensors, optimize='greedy', memory_limit=-1)
    assert check_path(path, [(0, 1), (0, 2), (0, 1)])


@pytest.mark.parametrize("optimize", ['greedy', 'branch-2', 'branch-all', 'optimal'])
def test_can_optimize_outer_products(optimize):
    a, b, c = [np.random.randn(10, 10) for _ in range(3)]
    d = np.random.randn(10, 2)
    assert oe.contract_path("ab,cd,ef,fg", a, b, c, d, optimize=optimize)[0] == [(2, 3), (0, 2), (0, 1)]


@pytest.mark.parametrize('num_symbols', [2, 3, 26, 26 + 26, 256 - 140, 300])
def test_large_path(num_symbols):
    symbols = ''.join(oe.get_symbol(i) for i in range(num_symbols))
    dimension_dict = dict(zip(symbols, itertools.cycle([2, 3, 4])))
    expression = ','.join(symbols[t:t + 2] for t in range(num_symbols - 1))
    tensors = oe.helpers.build_views(expression, dimension_dict=dimension_dict)

    # Check that path construction does not crash
    oe.contract_path(expression, *tensors, optimize='greedy')
