"""
Tests thte BLAS capability for the opt_einsum module.
"""

import numpy as np
import pytest

from opt_einsum import blas, contract, helpers

blas_tests = [
    # DOT
    ((['k', 'k'], '', set('k')),            'DOT'),  # DDOT
    ((['ijk', 'ijk'], '', set('ijk')),      'DOT'),  # DDOT

    # GEMV?

    # GEMM
    ((['ij', 'jk'], 'ik', set('j')),        'GEMM'), # GEMM N N
    ((['ijl', 'jlk'], 'ik', set('jl')),     'GEMM'), # GEMM N N Tensor
    ((['ij', 'kj'], 'ik', set('j')),        'GEMM'), # GEMM N T
    ((['ijl', 'kjl'], 'ik', set('jl')),     'GEMM'), # GEMM N T Tensor
    ((['ji', 'jk'], 'ik', set('j')),        'GEMM'), # GEMM T N
    ((['jli', 'jlk'], 'ik', set('jl')),     'GEMM'), # GEMM T N Tensor
    ((['ji', 'kj'], 'ik', set('j')),        'GEMM'), # GEMM T T
    ((['jli', 'kjl'], 'ik', set('jl')),     'GEMM'), # GEMM T T Tensor

    # GEMM with final transpose
    ((['ij', 'jk'], 'ki', set('j')),        'GEMM'), # GEMM N N
    ((['ijl', 'jlk'], 'ki', set('jl')),     'GEMM'), # GEMM N N Tensor
    ((['ij', 'kj'], 'ki', set('j')),        'GEMM'), # GEMM N T
    ((['ijl', 'kjl'], 'ki', set('jl')),     'GEMM'), # GEMM N T Tensor
    ((['ji', 'jk'], 'ki', set('j')),        'GEMM'), # GEMM T N
    ((['jli', 'jlk'], 'ki', set('jl')),     'GEMM'), # GEMM T N Tensor
    ((['ji', 'kj'], 'ki', set('j')),        'GEMM'), # GEMM T T
    ((['jli', 'kjl'], 'ki', set('jl')),     'GEMM'), # GEMM T T Tensor

   # Tensor Dot (requires copy), lets not deal with this for now
   ((['ilj', 'jlk'], 'ik', set('jl')),     'TDOT'), # FT GEMM N N Tensor
   ((['ijl', 'ljk'], 'ik', set('jl')),     'TDOT'), # ST GEMM N N Tensor
   ((['ilj', 'kjl'], 'ik', set('jl')),     'TDOT'), # FT GEMM N T Tensor
   ((['ijl', 'klj'], 'ik', set('jl')),     'TDOT'), # ST GEMM N T Tensor
   ((['lji', 'jlk'], 'ik', set('jl')),     'TDOT'), # FT GEMM T N Tensor
   ((['jli', 'ljk'], 'ik', set('jl')),     'TDOT'), # ST GEMM T N Tensor
   ((['lji', 'jlk'], 'ik', set('jl')),     'TDOT'), # FT GEMM T N Tensor
   ((['jli', 'ljk'], 'ik', set('jl')),     'TDOT'), # ST GEMM T N Tensor

   # Tensor Dot (requires copy), lets not deal with this for now with transpose
   ((['ilj', 'jlk'], 'ik', set('lj')),     'TDOT'), # FT GEMM N N Tensor
   ((['ijl', 'ljk'], 'ik', set('lj')),     'TDOT'), # ST GEMM N N Tensor
   ((['ilj', 'kjl'], 'ik', set('lj')),     'TDOT'), # FT GEMM N T Tensor
   ((['ijl', 'klj'], 'ik', set('lj')),     'TDOT'), # ST GEMM N T Tensor
   ((['lji', 'jlk'], 'ik', set('lj')),     'TDOT'), # FT GEMM T N Tensor
   ((['jli', 'ljk'], 'ik', set('lj')),     'TDOT'), # ST GEMM T N Tensor
   ((['lji', 'jlk'], 'ik', set('lj')),     'TDOT'), # FT GEMM T N Tensor
   ((['jli', 'ljk'], 'ik', set('lj')),     'TDOT'), # ST GEMM T N Tensor

   # Other
   ((['ijk', 'ikj'], '', set('ijk')),       'DOT/EINSUM'  ),  # Transpose DOT
   ((['i', 'j'], 'ij', set()),              'OUTER/EINSUM'),  # Outer
   ((['ijk', 'ik'], 'j', set('ik')),        'GEMV/EINSUM' ),  # Matrix-vector
   ((['ijj', 'jk'], 'ik', set('j')),        False         ),  # Double index
   ((['ijk', 'j'], 'ij', set()),            False         ),  # Index sum 1
   ((['ij', 'ij'], 'ij', set()),            False         ),  # Index sum 2
]

@pytest.mark.parametrize("inp,benchmark", blas_tests)
def test_can_blas(inp, benchmark):
    result = blas.can_blas(*inp)
    assert result == benchmark


@pytest.mark.parametrize("inp,benchmark", blas_tests)
def test_tensor_blas(inp, benchmark):

    # Weed out non-blas cases
    if benchmark is False:
        return

    tensor_strs, output, reduced_idx = inp
    einsum_str = ','.join(tensor_strs) + '->' + output

    # Only binary operations should be here
    if len(tensor_strs) != 2:
        assert False

    view_left, view_right = helpers.build_views(einsum_str)

    einsum_result = np.einsum(einsum_str, view_left, view_right)
    blas_result = blas.tensor_blas(view_left, tensor_strs[0], view_right, tensor_strs[1], output, reduced_idx)

    assert np.allclose(einsum_result, blas_result)


def test_blas_out():
    a = np.random.rand(4, 4)
    b = np.random.rand(4, 4)
    c = np.random.rand(4, 4)
    d = np.empty((4, 4))

    contract('ij,jk->ik', a, b, out=d)
    assert np.allclose(d, np.dot(a, b))

    contract('ij,jk,kl->il', a, b, c, out=d)
    assert np.allclose(d, np.dot(a, b).dot(c))
