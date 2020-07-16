import itertools
import weakref
from collections import Counter

import numpy as np
import pytest

from opt_einsum import (contract, contract_expression, contract_path, get_symbol, helpers, shared_intermediates)
from opt_einsum.backends import to_cupy, to_torch
from opt_einsum.contract import _einsum
from opt_einsum.parser import parse_einsum_input
from opt_einsum.sharing import (count_cached_ops, currently_sharing, get_sharing_cache)

try:
    import cupy
    cupy_if_found = 'cupy'
except ImportError:
    cupy_if_found = pytest.param('cupy', marks=[pytest.mark.skip(reason="CuPy not installed.")])

try:
    import torch
    torch_if_found = 'torch'
except ImportError:
    torch_if_found = pytest.param('torch', marks=[pytest.mark.skip(reason="PyTorch not installed.")])

backends = ['numpy', torch_if_found, cupy_if_found]
equations = [
    'ab,bc->ca',
    'abc,bcd,dea',
    'abc,def->fedcba',
    'abc,bcd,df->fa',
    # test 'prefer einsum' ops
    'ijk,ikj',
    'i,j->ij',
    'ijk,k->ij',
    'AB,BC->CA',
]
to_backend = {
    'numpy': lambda x: x,
    'torch': to_torch,
    'cupy': to_cupy,
}


@pytest.mark.parametrize('eq', equations)
@pytest.mark.parametrize('backend', backends)
def test_sharing_value(eq, backend):
    views = helpers.build_views(eq)
    shapes = [v.shape for v in views]
    expr = contract_expression(eq, *shapes)

    expected = expr(*views, backend=backend)
    with shared_intermediates():
        actual = expr(*views, backend=backend)

    assert (actual == expected).all()


@pytest.mark.parametrize('backend', backends)
def test_complete_sharing(backend):
    eq = 'ab,bc,cd->'
    views = helpers.build_views(eq)
    expr = contract_expression(eq, *(v.shape for v in views))

    print('-' * 40)
    print('Without sharing:')
    with shared_intermediates() as cache:
        expr(*views, backend=backend)
        expected = count_cached_ops(cache)

    print('-' * 40)
    print('With sharing:')
    with shared_intermediates() as cache:
        expr(*views, backend=backend)
        expr(*views, backend=backend)
        actual = count_cached_ops(cache)

    print('-' * 40)
    print('Without sharing: {} expressions'.format(expected))
    print('With sharing: {} expressions'.format(actual))
    assert actual == expected


@pytest.mark.parametrize('backend', backends)
def test_sharing_reused_cache(backend):
    eq = 'ab,bc,cd->'
    views = helpers.build_views(eq)
    expr = contract_expression(eq, *(v.shape for v in views))

    print('-' * 40)
    print('Without sharing:')
    with shared_intermediates() as cache:
        expr(*views, backend=backend)
        expected = count_cached_ops(cache)

    print('-' * 40)
    print('With sharing:')
    with shared_intermediates() as cache:
        expr(*views, backend=backend)
    with shared_intermediates(cache):
        expr(*views, backend=backend)
        actual = count_cached_ops(cache)

    print('-' * 40)
    print('Without sharing: {} expressions'.format(expected))
    print('With sharing: {} expressions'.format(actual))
    assert actual == expected


@pytest.mark.parametrize('backend', backends)
def test_no_sharing_separate_cache(backend):
    eq = 'ab,bc,cd->'
    views = helpers.build_views(eq)
    expr = contract_expression(eq, *(v.shape for v in views))

    print('-' * 40)
    print('Without sharing:')
    with shared_intermediates() as cache:
        expr(*views, backend=backend)
        expected = count_cached_ops(cache)
        expected.update(count_cached_ops(cache))  # we expect double

    print('-' * 40)
    print('With sharing:')
    with shared_intermediates() as cache1:
        expr(*views, backend=backend)
        actual = count_cached_ops(cache1)
    with shared_intermediates() as cache2:
        expr(*views, backend=backend)
        actual.update(count_cached_ops(cache2))

    print('-' * 40)
    print('Without sharing: {} expressions'.format(expected))
    print('With sharing: {} expressions'.format(actual))
    assert actual == expected


@pytest.mark.parametrize('backend', backends)
def test_sharing_nesting(backend):
    eqs = ['ab,bc,cd->a', 'ab,bc,cd->b', 'ab,bc,cd->c', 'ab,bc,cd->c']
    views = helpers.build_views(eqs[0])
    shapes = [v.shape for v in views]
    refs = weakref.WeakValueDictionary()

    def method1(views):
        with shared_intermediates():
            w = contract_expression(eqs[0], *shapes)(*views, backend=backend)
            x = contract_expression(eqs[2], *shapes)(*views, backend=backend)
            result = contract_expression('a,b->', w.shape, x.shape)(w, x, backend=backend)
            refs['w'] = w
            refs['x'] = x
            del w, x
            assert 'w' in refs
            assert 'x' in refs
        assert 'w' not in refs, 'cache leakage'
        assert 'x' not in refs, 'cache leakage'
        return result

    def method2(views):
        with shared_intermediates():
            y = contract_expression(eqs[2], *shapes)(*views, backend=backend)
            z = contract_expression(eqs[3], *shapes)(*views, backend=backend)
            refs['y'] = y
            refs['z'] = z
            result = contract_expression('c,d->', y.shape, z.shape)(y, z, backend=backend)
            result = result + method1(views)  # nest method1 in method2
            del y, z
            assert 'y' in refs
            assert 'z' in refs
        assert 'y' not in refs
        assert 'z' not in refs

    method1(views)
    method2(views)


@pytest.mark.parametrize('eq', equations)
@pytest.mark.parametrize('backend', backends)
def test_sharing_modulo_commutativity(eq, backend):
    ops = helpers.build_views(eq)
    ops = [to_backend[backend](x) for x in ops]
    inputs, output, _ = parse_einsum_input([eq] + ops)
    inputs = inputs.split(',')

    print('-' * 40)
    print('Without sharing:')
    with shared_intermediates() as cache:
        _einsum(eq, *ops, backend=backend)
        expected = count_cached_ops(cache)

    print('-' * 40)
    print('With sharing:')
    with shared_intermediates() as cache:
        for permuted in itertools.permutations(zip(inputs, ops)):
            permuted_inputs = [p[0] for p in permuted]
            permuted_ops = [p[1] for p in permuted]
            permuted_eq = '{}->{}'.format(','.join(permuted_inputs), output)
            _einsum(permuted_eq, *permuted_ops, backend=backend)
        actual = count_cached_ops(cache)

    print('-' * 40)
    print('Without sharing: {} expressions'.format(expected))
    print('With sharing: {} expressions'.format(actual))
    assert actual == expected


@pytest.mark.parametrize('backend', backends)
def test_partial_sharing(backend):
    eq = 'ab,bc,de->'
    x, y, z1 = helpers.build_views(eq)
    z2 = 2.0 * z1 - 1.0
    expr = contract_expression(eq, x.shape, y.shape, z1.shape)

    print('-' * 40)
    print('Without sharing:')
    num_exprs_nosharing = Counter()
    with shared_intermediates() as cache:
        expr(x, y, z1, backend=backend)
        num_exprs_nosharing.update(count_cached_ops(cache))
    with shared_intermediates() as cache:
        expr(x, y, z2, backend=backend)
        num_exprs_nosharing.update(count_cached_ops(cache))

    print('-' * 40)
    print('With sharing:')
    with shared_intermediates() as cache:
        expr(x, y, z1, backend=backend)
        expr(x, y, z2, backend=backend)
        num_exprs_sharing = count_cached_ops(cache)

    print('-' * 40)
    print('Without sharing: {} expressions'.format(num_exprs_nosharing))
    print('With sharing: {} expressions'.format(num_exprs_sharing))
    assert num_exprs_nosharing['einsum'] > num_exprs_sharing['einsum']


@pytest.mark.parametrize('backend', backends)
def test_sharing_with_constants(backend):
    inputs = 'ij,jk,kl'
    outputs = 'ijkl'
    equations = ['{}->{}'.format(inputs, output) for output in outputs]
    shapes = (2, 3), (3, 4), (4, 5)
    constants = {0, 2}
    ops = [np.random.rand(*shp) if i in constants else shp for i, shp in enumerate(shapes)]
    var = np.random.rand(*shapes[1])

    expected = [contract_expression(eq, *shapes)(ops[0], var, ops[2]) for eq in equations]

    with shared_intermediates():
        actual = [contract_expression(eq, *ops, constants=constants)(var) for eq in equations]

    for dim, expected_dim, actual_dim in zip(outputs, expected, actual):
        assert np.allclose(expected_dim, actual_dim), 'error at {}'.format(dim)


@pytest.mark.parametrize('size', [3, 4, 5])
@pytest.mark.parametrize('backend', backends)
def test_chain(size, backend):
    xs = [np.random.rand(2, 2) for _ in range(size)]
    shapes = [x.shape for x in xs]
    alphabet = ''.join(get_symbol(i) for i in range(size + 1))
    names = [alphabet[i:i + 2] for i in range(size)]
    inputs = ','.join(names)

    with shared_intermediates():
        print(inputs)
        for i in range(size + 1):
            target = alphabet[i]
            eq = '{}->{}'.format(inputs, target)
            path_info = contract_path(eq, *xs)
            print(path_info[1])
            expr = contract_expression(eq, *shapes)
            expr(*xs, backend=backend)
        print('-' * 40)


@pytest.mark.parametrize('size', [3, 4, 5, 10])
@pytest.mark.parametrize('backend', backends)
def test_chain_2(size, backend):
    xs = [np.random.rand(2, 2) for _ in range(size)]
    shapes = [x.shape for x in xs]
    alphabet = ''.join(get_symbol(i) for i in range(size + 1))
    names = [alphabet[i:i + 2] for i in range(size)]
    inputs = ','.join(names)

    with shared_intermediates():
        print(inputs)
        for i in range(size):
            target = alphabet[i:i + 2]
            eq = '{}->{}'.format(inputs, target)
            path_info = contract_path(eq, *xs)
            print(path_info[1])
            expr = contract_expression(eq, *shapes)
            expr(*xs, backend=backend)
        print('-' * 40)


def _compute_cost(cache):
    counts = count_cached_ops(cache)
    return counts['einsum'] + counts['tensordot']


@pytest.mark.parametrize('backend', backends)
def test_chain_2_growth(backend):
    sizes = list(range(1, 21))
    costs = []
    for size in sizes:
        xs = [np.random.rand(2, 2) for _ in range(size)]
        alphabet = ''.join(get_symbol(i) for i in range(size + 1))
        names = [alphabet[i:i + 2] for i in range(size)]
        inputs = ','.join(names)

        with shared_intermediates() as cache:
            for i in range(size):
                target = alphabet[i:i + 2]
                eq = '{}->{}'.format(inputs, target)
                expr = contract_expression(eq, *(x.shape for x in xs))
                expr(*xs, backend=backend)
            costs.append(_compute_cost(cache))

    print('sizes = {}'.format(repr(sizes)))
    print('costs = {}'.format(repr(costs)))
    for size, cost in zip(sizes, costs):
        print('{}\t{}'.format(size, cost))


@pytest.mark.parametrize('size', [3, 4, 5])
@pytest.mark.parametrize('backend', backends)
def test_chain_sharing(size, backend):
    xs = [np.random.rand(2, 2) for _ in range(size)]
    alphabet = ''.join(get_symbol(i) for i in range(size + 1))
    names = [alphabet[i:i + 2] for i in range(size)]
    inputs = ','.join(names)

    num_exprs_nosharing = 0
    for i in range(size + 1):
        with shared_intermediates() as cache:
            target = alphabet[i]
            eq = '{}->{}'.format(inputs, target)
            expr = contract_expression(eq, *(x.shape for x in xs))
            expr(*xs, backend=backend)
            num_exprs_nosharing += _compute_cost(cache)

    with shared_intermediates() as cache:
        print(inputs)
        for i in range(size + 1):
            target = alphabet[i]
            eq = '{}->{}'.format(inputs, target)
            path_info = contract_path(eq, *xs)
            print(path_info[1])
            expr = contract_expression(eq, *(x.shape for x in xs))
            expr(*xs, backend=backend)
        num_exprs_sharing = _compute_cost(cache)

    print('-' * 40)
    print('Without sharing: {} expressions'.format(num_exprs_nosharing))
    print('With sharing: {} expressions'.format(num_exprs_sharing))
    assert num_exprs_nosharing > num_exprs_sharing


def test_multithreaded_sharing():
    from multiprocessing.pool import ThreadPool

    def fn():
        X, Y, Z = helpers.build_views('ab,bc,cd')

        with shared_intermediates():
            contract('ab,bc,cd->a', X, Y, Z)
            contract('ab,bc,cd->b', X, Y, Z)

            return len(get_sharing_cache())

    expected = fn()
    pool = ThreadPool(8)
    fs = [pool.apply_async(fn) for _ in range(16)]
    assert not currently_sharing()
    assert [f.get() for f in fs] == [expected] * 16
    pool.close()
