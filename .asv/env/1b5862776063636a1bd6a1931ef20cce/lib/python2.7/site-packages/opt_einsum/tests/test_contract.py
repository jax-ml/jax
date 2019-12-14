"""
Tets a series of opt_einsum contraction paths to ensure the results are the same for different paths
"""

import numpy as np
import pytest

from opt_einsum import compat, contract, contract_path, helpers, contract_expression
from opt_einsum.paths import linear_to_ssa, ssa_to_linear

tests = [
    # Test hadamard-like products
    'a,ab,abc->abc',
    'a,b,ab->ab',

    # Test index-transformations
    'ea,fb,gc,hd,abcd->efgh',
    'ea,fb,abcd,gc,hd->efgh',
    'abcd,ea,fb,gc,hd->efgh',

    # Test complex contractions
    'acdf,jbje,gihb,hfac,gfac,gifabc,hfac',
    'acdf,jbje,gihb,hfac,gfac,gifabc,hfac',
    'cd,bdhe,aidb,hgca,gc,hgibcd,hgac',
    'abhe,hidj,jgba,hiab,gab',
    'bde,cdh,agdb,hica,ibd,hgicd,hiac',
    'chd,bde,agbc,hiad,hgc,hgi,hiad',
    'chd,bde,agbc,hiad,bdi,cgh,agdb',
    'bdhe,acad,hiab,agac,hibd',

    # Test collapse
    'ab,ab,c->',
    'ab,ab,c->c',
    'ab,ab,cd,cd->',
    'ab,ab,cd,cd->ac',
    'ab,ab,cd,cd->cd',
    'ab,ab,cd,cd,ef,ef->',

    # Test outer prodcuts
    'ab,cd,ef->abcdef',
    'ab,cd,ef->acdf',
    'ab,cd,de->abcde',
    'ab,cd,de->be',
    'ab,bcd,cd->abcd',
    'ab,bcd,cd->abd',

    # Random test cases that have previously failed
    'eb,cb,fb->cef',
    'dd,fb,be,cdb->cef',
    'bca,cdb,dbf,afc->',
    'dcc,fce,ea,dbf->ab',
    'fdf,cdd,ccd,afe->ae',
    'abcd,ad',
    'ed,fcd,ff,bcf->be',
    'baa,dcf,af,cde->be',
    'bd,db,eac->ace',
    'fff,fae,bef,def->abd',
    'efc,dbc,acf,fd->abe',

    # Inner products
    'ab,ab',
    'ab,ba',
    'abc,abc',
    'abc,bac',
    'abc,cba',

    # GEMM test cases
    'ab,bc',
    'ab,cb',
    'ba,bc',
    'ba,cb',
    'abcd,cd',
    'abcd,ab',
    'abcd,cdef',
    'abcd,cdef->feba',
    'abcd,efdc',

    # Inner than dot
    'aab,bc->ac',
    'ab,bcc->ac',
    'aab,bcc->ac',
    'baa,bcc->ac',
    'aab,ccb->ac',

    # Randomly build test caes
    'aab,fa,df,ecc->bde',
    'ecb,fef,bad,ed->ac',
    'bcf,bbb,fbf,fc->',
    'bb,ff,be->e',
    'bcb,bb,fc,fff->',
    'fbb,dfd,fc,fc->',
    'afd,ba,cc,dc->bf',
    'adb,bc,fa,cfc->d',
    'bbd,bda,fc,db->acf',
    'dba,ead,cad->bce',
    'aef,fbc,dca->bde',
]

all_optimizers = ['optimal', 'branch-all', 'branch-2', 'branch-1', 'greedy']


@pytest.mark.parametrize("string", tests)
@pytest.mark.parametrize("optimize", all_optimizers)
def test_compare(optimize, string):
    views = helpers.build_views(string)

    ein = contract(string, *views, optimize=False, use_blas=False)
    opt = contract(string, *views, optimize=optimize, use_blas=False)
    assert np.allclose(ein, opt)


@pytest.mark.parametrize("string", tests)
def test_drop_in_replacement(string):
    views = helpers.build_views(string)
    opt = contract(string, *views)
    assert np.allclose(opt, np.einsum(string, *views))


@pytest.mark.parametrize("string", tests)
@pytest.mark.parametrize("optimize", all_optimizers)
def test_compare_greek(optimize, string):
    views = helpers.build_views(string)

    ein = contract(string, *views, optimize=False, use_blas=False)

    # convert to greek
    string = ''.join(compat.get_char(ord(c) + 848) if c not in ',->.' else c for c in string)

    opt = contract(string, *views, optimize=optimize, use_blas=False)
    assert np.allclose(ein, opt)


@pytest.mark.parametrize("string", tests)
@pytest.mark.parametrize("optimize", all_optimizers)
def test_compare_blas(optimize, string):
    views = helpers.build_views(string)

    ein = contract(string, *views, optimize=False)
    opt = contract(string, *views, optimize=optimize)
    assert np.allclose(ein, opt)


@pytest.mark.parametrize("string", tests)
@pytest.mark.parametrize("optimize", all_optimizers)
def test_compare_blas_greek(optimize, string):
    views = helpers.build_views(string)

    ein = contract(string, *views, optimize=False)

    # convert to greek
    string = ''.join(compat.get_char(ord(c) + 848) if c not in ',->.' else c for c in string)

    opt = contract(string, *views, optimize=optimize)
    assert np.allclose(ein, opt)


def test_some_non_alphabet_maintains_order():
    # 'c beta a' should automatically go to -> 'a c beta'
    string = 'c' + compat.get_char(ord('b') + 848) + 'a'
    # but beta will be temporarily replaced with 'b' for which 'cba->abc'
    # so check manual output kicks in:
    x = np.random.rand(2, 3, 4)
    assert np.allclose(contract(string, x), contract('cxa', x))


def test_printing():
    string = "bbd,bda,fc,db->acf"
    views = helpers.build_views(string)

    ein = contract_path(string, *views)
    assert len(str(ein[1])) == 726


@pytest.mark.parametrize("string", tests)
@pytest.mark.parametrize("optimize", all_optimizers)
@pytest.mark.parametrize("use_blas", [False, True])
@pytest.mark.parametrize("out_spec", [False, True])
def test_contract_expressions(string, optimize, use_blas, out_spec):
    views = helpers.build_views(string)
    shapes = [view.shape for view in views]
    expected = contract(string, *views, optimize=False, use_blas=False)

    expr = contract_expression(string, *shapes, optimize=optimize, use_blas=use_blas)

    if out_spec and ("->" in string) and (string[-2:] != "->"):
        out, = helpers.build_views(string.split('->')[1])
        expr(*views, out=out)
    else:
        out = expr(*views)

    assert np.allclose(out, expected)

    # check representations
    assert string in expr.__repr__()
    assert string in expr.__str__()


def test_contract_expression_interleaved_input():
    x, y, z = (np.random.randn(2, 2) for _ in 'xyz')
    expected = np.einsum(x, [0, 1], y, [1, 2], z, [2, 3], [3, 0])
    xshp, yshp, zshp = ((2, 2) for _ in 'xyz')
    expr = contract_expression(xshp, [0, 1], yshp, [1, 2], zshp, [2, 3], [3, 0])
    out = expr(x, y, z)
    assert np.allclose(out, expected)


@pytest.mark.parametrize("string,constants", [
    ('hbc,bdef,cdkj,ji,ikeh,lfo', [1, 2, 3, 4]),
    ('bdef,cdkj,ji,ikeh,hbc,lfo', [0, 1, 2, 3]),
    ('hbc,bdef,cdkj,ji,ikeh,lfo', [1, 2, 3, 4]),
    ('hbc,bdef,cdkj,ji,ikeh,lfo', [1, 2, 3, 4]),
    ('ijab,acd,bce,df,ef->ji', [1, 2, 3, 4]),
    ('ab,cd,ad,cb', [1, 3]),
    ('ab,bc,cd', [0, 1]),
])
def test_contract_expression_with_constants(string, constants):
    views = helpers.build_views(string)
    expected = contract(string, *views, optimize=False, use_blas=False)

    shapes = [view.shape for view in views]

    expr_args = []
    ctrc_args = []
    for i, (shape, view) in enumerate(zip(shapes, views)):
        if i in constants:
            expr_args.append(view)
        else:
            expr_args.append(shape)
            ctrc_args.append(view)

    expr = contract_expression(string, *expr_args, constants=constants)
    print(expr)
    out = expr(*ctrc_args)
    assert np.allclose(expected, out)


@pytest.mark.parametrize("optimize", ['greedy', 'optimal'])
@pytest.mark.parametrize("n", [4, 5])
@pytest.mark.parametrize("reg", [2, 3])
@pytest.mark.parametrize("n_out", [0, 2, 4])
@pytest.mark.parametrize("global_dim", [False, True])
def test_rand_equation(optimize, n, reg, n_out, global_dim):
    eq, _, size_dict = helpers.rand_equation(n, reg, n_out, d_min=2, d_max=5, seed=42, return_size_dict=True)
    views = helpers.build_views(eq, size_dict)

    expected = contract(eq, *views, optimize=False)
    actual = contract(eq, *views, optimize=optimize)

    assert np.allclose(expected, actual)


@pytest.mark.parametrize('equation', tests)
def test_linear_vs_ssa(equation):
    views = helpers.build_views(equation)
    linear_path, _ = contract_path(equation, *views)
    ssa_path = linear_to_ssa(linear_path)
    linear_path2 = ssa_to_linear(ssa_path)
    assert linear_path2 == linear_path
