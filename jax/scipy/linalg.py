# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import partial

import scipy.linalg
import textwrap

from jax import jit, vmap
from .. import api
from .. import lax
from .. import lax_linalg
from ..numpy._util import _wraps
from ..numpy import lax_numpy as jnp
from ..numpy import linalg as np_linalg

_T = lambda x: jnp.swapaxes(x, -1, -2)

@partial(jit, static_argnums=(1,))
def _cholesky(a, lower):
  a = np_linalg._promote_arg_dtypes(jnp.asarray(a))
  l = lax_linalg.cholesky(a if lower else jnp.conj(_T(a)), symmetrize_input=False)
  return l if lower else jnp.conj(_T(l))

@_wraps(scipy.linalg.cholesky)
def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
  del overwrite_a, check_finite
  return _cholesky(a, lower)

@_wraps(scipy.linalg.cho_factor)
def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
  return (cholesky(a, lower=lower), lower)

@partial(jit, static_argnums=(2,))
def _cho_solve(c, b, lower):
  c, b = np_linalg._promote_arg_dtypes(jnp.asarray(c), jnp.asarray(b))
  np_linalg._check_solve_shapes(c, b)
  b = lax_linalg.triangular_solve(c, b, left_side=True, lower=lower,
                                  transpose_a=not lower, conjugate_a=not lower)
  b = lax_linalg.triangular_solve(c, b, left_side=True, lower=lower,
                                  transpose_a=lower, conjugate_a=lower)
  return b

@_wraps(scipy.linalg.cho_solve, update_doc=False)
def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
  del overwrite_b, check_finite
  c, lower = c_and_lower
  return _cho_solve(c, b, lower)

@_wraps(scipy.linalg.svd)
def svd(a, full_matrices=True, compute_uv=True, overwrite_a=False,
        check_finite=True, lapack_driver='gesdd'):
  del overwrite_a, check_finite, lapack_driver
  a = np_linalg._promote_arg_dtypes(jnp.asarray(a))
  return lax_linalg.svd(a, full_matrices, compute_uv)


@_wraps(scipy.linalg.det)
def det(a, overwrite_a=False, check_finite=True):
  del overwrite_a, check_finite
  return np_linalg.det(a)


@_wraps(scipy.linalg.eigh)
def eigh(a, b=None, lower=True, eigvals_only=False, overwrite_a=False,
         overwrite_b=False, turbo=True, eigvals=None, type=1,
         check_finite=True):
  del overwrite_a, overwrite_b, turbo, check_finite
  if b is not None:
    raise NotImplementedError("Only the b=None case of eigh is implemented")
  if type != 1:
    raise NotImplementedError("Only the type=1 case of eigh is implemented.")
  if eigvals is not None:
    raise NotImplementedError(
        "Only the eigvals=None case of eigh is implemented.")

  a = np_linalg._promote_arg_dtypes(jnp.asarray(a))
  v, w = lax_linalg.eigh(a, lower=lower)

  if eigvals_only:
    return w
  else:
    return w, v


@_wraps(scipy.linalg.inv)
def inv(a, overwrite_a=False, check_finite=True):
  del overwrite_a, check_finite
  return np_linalg.inv(a)


@_wraps(scipy.linalg.lu_factor)
def lu_factor(a, overwrite_a=False, check_finite=True):
  del overwrite_a, check_finite
  a = np_linalg._promote_arg_dtypes(jnp.asarray(a))
  return lax_linalg.lu(a)


@_wraps(scipy.linalg.lu_solve)
def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
  del overwrite_b, check_finite
  lu, pivots = lu_and_piv
  return lax_linalg.lu_solve(lu, pivots, b, trans)


@partial(jit, static_argnums=(1,))
def _lu(a, permute_l):
  a = np_linalg._promote_arg_dtypes(jnp.asarray(a))
  lu, pivots = lax_linalg.lu(a)
  dtype = lax.dtype(a)
  m, n = jnp.shape(a)
  permutation = lax_linalg.lu_pivots_to_permutation(pivots, m)
  p = jnp.real(jnp.array(permutation == jnp.arange(m)[:, None], dtype=dtype))
  k = min(m, n)
  l = jnp.tril(lu, -1)[:, :k] + jnp.eye(m, k, dtype=dtype)
  u = jnp.triu(lu)[:k, :]
  if permute_l:
    return jnp.matmul(p, l), u
  else:
    return p, l, u

@_wraps(scipy.linalg.lu, update_doc=False)
def lu(a, permute_l=False, overwrite_a=False, check_finite=True):
  del overwrite_a, check_finite
  return _lu(a, permute_l)

@partial(jit, static_argnums=(1, 2))
def _qr(a, mode, pivoting):
  if pivoting:
    raise NotImplementedError(
        "The pivoting=True case of qr is not implemented.")
  if mode in ("full", "r"):
    full_matrices = True
  elif mode == "economic":
    full_matrices = False
  else:
    raise ValueError("Unsupported QR decomposition mode '{}'".format(mode))
  a = np_linalg._promote_arg_dtypes(jnp.asarray(a))
  q, r = lax_linalg.qr(a, full_matrices)
  if mode == "r":
    return r
  return q, r

@_wraps(scipy.linalg.qr)
def qr(a, overwrite_a=False, lwork=None, mode="full", pivoting=False,
       check_finite=True):
  del overwrite_a, lwork, check_finite
  return _qr(a, mode, pivoting)


@partial(jit, static_argnums=(2, 3))
def _solve(a, b, sym_pos, lower):
  if not sym_pos:
    return np_linalg.solve(a, b)

  a, b = np_linalg._promote_arg_dtypes(jnp.asarray(a), jnp.asarray(b))
  np_linalg._check_solve_shapes(a, b)

  # With custom_linear_solve, we can reuse the same factorization when
  # computing sensitivities. This is considerably faster.
  factors = cho_factor(lax.stop_gradient(a), lower=lower)
  custom_solve = partial(
      lax.custom_linear_solve,
      lambda x: np_linalg._matvec_multiply(a, x),
      solve=lambda _, x: cho_solve(factors, x),
      symmetric=True)
  if a.ndim == b.ndim + 1:
    # b.shape == [..., m]
    return custom_solve(b)
  else:
    # b.shape == [..., m, k]
    return vmap(custom_solve, b.ndim - 1, max(a.ndim, b.ndim) - 1)(b)


@_wraps(scipy.linalg.solve)
def solve(a, b, sym_pos=False, lower=False, overwrite_a=False, overwrite_b=False,
          debug=False, check_finite=True):
  del overwrite_a, overwrite_b, debug, check_finite
  return _solve(a, b, sym_pos, lower)

@partial(jit, static_argnums=(2, 3, 4))
def _solve_triangular(a, b, trans, lower, unit_diagonal):
  if trans == 0 or trans == "N":
    transpose_a, conjugate_a = False, False
  elif trans == 1 or trans == "T":
    transpose_a, conjugate_a = True, False
  elif trans == 2 or trans == "C":
    transpose_a, conjugate_a = True, True
  else:
    raise ValueError("Invalid 'trans' value {}".format(trans))

  a, b = np_linalg._promote_arg_dtypes(jnp.asarray(a), jnp.asarray(b))

  # lax_linalg.triangular_solve only supports matrix 'b's at the moment.
  b_is_vector = jnp.ndim(a) == jnp.ndim(b) + 1
  if b_is_vector:
    b = b[..., None]
  out = lax_linalg.triangular_solve(a, b, left_side=True, lower=lower,
                                    transpose_a=transpose_a,
                                    conjugate_a=conjugate_a,
                                    unit_diagonal=unit_diagonal)
  if b_is_vector:
    return out[..., 0]
  else:
    return out

@_wraps(scipy.linalg.solve_triangular)
def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False,
                     overwrite_b=False, debug=None, check_finite=True):
  del overwrite_b, debug, check_finite
  return _solve_triangular(a, b, trans, lower, unit_diagonal)



@_wraps(scipy.linalg.tril)
def tril(m, k=0):
  return jnp.tril(m, k)


@_wraps(scipy.linalg.triu)
def triu(m, k=0):
  return jnp.triu(m, k)

@_wraps(scipy.linalg.expm, lax_description=textwrap.dedent("""\
    In addition to the original NumPy argument(s) listed below,
    also supports the optional boolean argument ``upper_triangular``
    to specify whether the ``A`` matrix is upper triangular.
    """))
@api.custom_transforms
def expm(A, *, upper_triangular=False):
    return _expm(A, upper_triangular)

def _expm(A, upper_triangular):
    P,Q,n_squarings = _calc_P_Q(A)
    R = _solve_P_Q(P, Q, upper_triangular)
    R = _squaring(R, n_squarings)
    return R

@jit
def _calc_P_Q(A):
    A = jnp.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be a square matrix')
    A_L1 = np_linalg.norm(A,1)
    n_squarings = 0
    if A.dtype == 'float64' or A.dtype == 'complex128':
       U3,V3 = _pade3(A)
       U5,V5 = _pade5(A)
       U7,V7 = _pade7(A)
       U9,V9 = _pade9(A)
       maxnorm = 5.371920351148152
       n_squarings = jnp.maximum(0, jnp.floor(jnp.log2(A_L1 / maxnorm)))
       A = A / 2**n_squarings
       U13,V13 = _pade13(A)
       conds=jnp.array([1.495585217958292e-002, 2.539398330063230e-001, 9.504178996162932e-001, 2.097847961257068e+000])
       U = jnp.select((maxnorm<conds),(U3,U5,U7,U9),U13)
       V = jnp.select((maxnorm<conds),(V3,V5,V7,V9),V13)
    elif A.dtype == 'float32' or A.dtype == 'complex64':
        U3,V3 = _pade3(A)
        U5,V5 = _pade5(A)
        maxnorm = 3.925724783138660
        n_squarings = jnp.maximum(0, jnp.floor(jnp.log2(A_L1 / maxnorm)))
        A = A / 2**n_squarings
        U7,V7 = _pade7(A)
        conds=jnp.array([4.258730016922831e-001, 1.880152677804762e+000])
        U = jnp.select((maxnorm<conds),(U3,U5),U7)
        V = jnp.select((maxnorm<conds),(V3,V5),V7)
    else:
        raise TypeError("A.dtype={} is not supported.".format(A.dtype))
    P = U + V  # p_m(A) : numerator
    Q = -U + V # q_m(A) : denominator
    return P,Q,n_squarings

def _solve_P_Q(P, Q, upper_triangular=False):
    if upper_triangular:
        return solve_triangular(Q, P)
    else:
        return np_linalg.solve(Q,P)

@jit
def _squaring(R, n_squarings):
    # squaring step to undo scaling
    def my_body_fun(i,R):
      return jnp.dot(R,R)
    lower = jnp.zeros(1, dtype=n_squarings.dtype)
    R = lax.fori_loop(lower[0],n_squarings,my_body_fun,R)
    return R

def _pade3(A):
    b = (120., 60., 12., 1.)
    ident = jnp.eye(*A.shape, dtype=A.dtype)
    A2 = jnp.dot(A,A)
    U = jnp.dot(A , (b[3]*A2 + b[1]*ident))
    V = b[2]*A2 + b[0]*ident
    return U,V

def _pade5(A):
    b = (30240., 15120., 3360., 420., 30., 1.)
    ident = jnp.eye(*A.shape, dtype=A.dtype)
    A2 = jnp.dot(A,A)
    A4 = jnp.dot(A2,A2)
    U = jnp.dot(A, b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[4]*A4 + b[2]*A2 + b[0]*ident
    return U,V

def _pade7(A):
    b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
    ident = jnp.eye(*A.shape, dtype=A.dtype)
    A2 = jnp.dot(A,A)
    A4 = jnp.dot(A2,A2)
    A6 = jnp.dot(A4,A2)
    U = jnp.dot(A, b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U,V

def _pade9(A):
    b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
                2162160., 110880., 3960., 90., 1.)
    ident = jnp.eye(*A.shape, dtype=A.dtype)
    A2 = jnp.dot(A,A)
    A4 = jnp.dot(A2,A2)
    A6 = jnp.dot(A4,A2)
    A8 = jnp.dot(A6,A2)
    U = jnp.dot(A, b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[8]*A8 + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U,V

def _pade13(A):
    b = (64764752532480000., 32382376266240000., 7771770303897600.,
    1187353796428800., 129060195264000., 10559470521600., 670442572800.,
    33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.)
    ident = jnp.eye(*A.shape, dtype=A.dtype)
    A2 = jnp.dot(A,A)
    A4 = jnp.dot(A2,A2)
    A6 = jnp.dot(A4,A2)
    U = jnp.dot(A,jnp.dot(A6, b[13]*A6 + b[11]*A4 + b[9]*A2) + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = jnp.dot(A6, b[12]*A6 + b[10]*A4 + b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U,V

@_wraps(scipy.linalg.expm_frechet,lax_description=textwrap.dedent("""\
    Does not currently support the Scipy argument ``jax.numpy.asarray_chkfinite``, because `jax.numpy.asarray_chkfinite` does not exist at the moment. Does not support the ``method='blockEnlarge'`` argument.
    """))
def expm_frechet(A, E, *, method=None, compute_expm=True):
    return _expm_frechet(A, E, method, compute_expm)

def _expm_frechet(A, E, method=None, compute_expm=True):
    A = jnp.asarray(A)
    E = jnp.asarray(E)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be a square matrix')
    if E.ndim != 2 or E.shape[0] != E.shape[1]:
        raise ValueError('expected E to be a square matrix')
    if A.shape != E.shape:
        raise ValueError('expected A and E to be the same shape')
    if method is None:
        method = 'SPS'
    if method == 'SPS':
        expm_A, expm_frechet_AE = expm_frechet_algo_64(A, E)
    else:
        raise ValueError('only method=\'SPS\' is supported')
    expm_A, expm_frechet_AE = expm_frechet_algo_64(A, E)
    if compute_expm:
        return expm_A, expm_frechet_AE
    else:
        return expm_frechet_AE

"""
Maximal values ell_m of ||2**-s A|| such that the backward error bound
does not exceed 2**-53.
"""
ell_table_61 = (
        None,
        # 1
        2.11e-8,
        3.56e-4,
        1.08e-2,
        6.49e-2,
        2.00e-1,
        4.37e-1,
        7.83e-1,
        1.23e0,
        1.78e0,
        2.42e0,
        # 11
        3.13e0,
        3.90e0,
        4.74e0,
        5.63e0,
        6.56e0,
        7.52e0,
        8.53e0,
        9.56e0,
        1.06e1,
        1.17e1,
        )
        
@jit
def expm_frechet_algo_64(A, E):
    ident = jnp.eye(*A.shape, dtype=A.dtype)
    A_norm_1 = np_linalg.norm(A, 1)
    """
    Subset of the Maximal values ell_m of ||2**-s A||
    such that the backward error bound does not exceed 2**-53.
    """
    ell_table_61_local = jnp.array([ell_table_61[3],ell_table_61[5],ell_table_61[7],ell_table_61[9]])
    
    args = (A, E, ident, A_norm_1)
    U3579, V3579, Lu3579, Lv3579, s3579 = lax.cond(A_norm_1<=ell_table_61[3],
       args, lambda args: _diff_pade3(args),
       args, lambda args: lax.cond(A_norm_1<=ell_table_61[5],
           args, lambda args: _diff_pade5(args),
           args, lambda args: lax.cond(A_norm_1<=ell_table_61[7],
                args, lambda args: _diff_pade7(args),
                args, lambda args: _diff_pade9(args))))
    U13, V13, Lu13, Lv13, s13 = _diff_pade13(args)
    
    # Must be of minimum length 2 for np.select to be used
    ell_table_61_local99 = jnp.array([ell_table_61[9],ell_table_61[9]])
    U = jnp.select((A_norm_1<=ell_table_61_local99),(U3579,U3579),U13)
    V = jnp.select((A_norm_1<=ell_table_61_local99),(V3579,V3579),V13)
    Lu = jnp.select((A_norm_1<=ell_table_61_local99),(Lu3579,Lu3579),Lu13)
    Lv = jnp.select((A_norm_1<=ell_table_61_local99),(Lv3579,Lv3579),Lv13)
    s = jnp.select((A_norm_1<=ell_table_61_local99),(s3579,s3579),s13)

    lu_piv = lu_factor(-U + V)
    R = lu_solve(lu_piv, U + V)
    L = lu_solve(lu_piv, Lu + Lv + jnp.dot((Lu - Lv), R))
    # squaring
    def my_body_fun(i,my_arg):
        R,L = my_arg
        L = jnp.dot(R, L) + jnp.dot(L, R)
        R = jnp.dot(R,R)
        return R,L
    lower = jnp.zeros(1, dtype=s.dtype)
    R,L = lax.fori_loop(lower[0], s, my_body_fun, (R, L))
    return R,L

"""
# The b vectors and U and V are copypasted
# from scipy.sparse.linalg.matfuncs.py.
# M, Lu, Lv follow (6.11), (6.12), (6.13), (3.3)
"""
@jit
def _diff_pade3(args):
    A,E,ident,_ = args
    s = 0
    b = (120., 60., 12., 1.)
    A2 = A.dot(A)
    M2 = jnp.dot(A, E) + jnp.dot(E, A)
    U = A.dot(b[3]*A2 + b[1]*ident)
    V = b[2]*A2 + b[0]*ident
    Lu = A.dot(b[3]*M2) + E.dot(b[3]*A2 + b[1]*ident)
    Lv = b[2]*M2
    return U, V, Lu, Lv, s

@jit
def _diff_pade5(args):
    A,E,ident,_ = args
    s = 0
    b = (30240., 15120., 3360., 420., 30., 1.)
    A2 = A.dot(A)
    M2 = jnp.dot(A, E) + jnp.dot(E, A)
    A4 = jnp.dot(A2, A2)
    M4 = jnp.dot(A2, M2) + jnp.dot(M2, A2)
    U = A.dot(b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[4]*A4 + b[2]*A2 + b[0]*ident
    Lu = (A.dot(b[5]*M4 + b[3]*M2) +
            E.dot(b[5]*A4 + b[3]*A2 + b[1]*ident))
    Lv = b[4]*M4 + b[2]*M2
    return U, V, Lu, Lv, s

@jit
def _diff_pade7(args):
    A,E,ident,_ = args
    s = 0
    b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
    A2 = A.dot(A)
    M2 = jnp.dot(A, E) + jnp.dot(E, A)
    A4 = jnp.dot(A2, A2)
    M4 = jnp.dot(A2, M2) + jnp.dot(M2, A2)
    A6 = jnp.dot(A2, A4)
    M6 = jnp.dot(A4, M2) + jnp.dot(M4, A2)
    U = A.dot(b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    Lu = (A.dot(b[7]*M6 + b[5]*M4 + b[3]*M2) +
            E.dot(b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident))
    Lv = b[6]*M6 + b[4]*M4 + b[2]*M2
    return U, V, Lu, Lv, s

@jit
def _diff_pade9(args):
    A,E,ident,_ = args
    s = 0
    b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
            2162160., 110880., 3960., 90., 1.)
    A2 = A.dot(A)
    M2 = jnp.dot(A, E) + jnp.dot(E, A)
    A4 = jnp.dot(A2, A2)
    M4 = jnp.dot(A2, M2) + jnp.dot(M2, A2)
    A6 = jnp.dot(A2, A4)
    M6 = jnp.dot(A4, M2) + jnp.dot(M4, A2)
    A8 = jnp.dot(A4, A4)
    M8 = jnp.dot(A4, M4) + jnp.dot(M4, A4)
    U = A.dot(b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[8]*A8 + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    Lu = (A.dot(b[9]*M8 + b[7]*M6 + b[5]*M4 + b[3]*M2) +
            E.dot(b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident))
    Lv = b[8]*M8 + b[6]*M6 + b[4]*M4 + b[2]*M2
    return U, V, Lu, Lv, s

@jit
def _diff_pade13(args):
    A,E,ident,A_norm_1 = args
    s = jnp.maximum(0, jnp.floor_divide(lax.ceil(jnp.log2(A_norm_1 / ell_table_61[13])),1))
    two = jnp.array([2.0],A.dtype)
    A = A * two[0]**-s
    E = E * two[0]**-s
    # pade order 13
    A2 = jnp.dot(A, A)
    M2 = jnp.dot(A, E) + jnp.dot(E, A)
    A4 = jnp.dot(A2, A2)
    M4 = jnp.dot(A2, M2) + jnp.dot(M2, A2)
    A6 = jnp.dot(A2, A4)
    M6 = jnp.dot(A4, M2) + jnp.dot(M4, A2)
    b = (64764752532480000., 32382376266240000., 7771770303897600.,
            1187353796428800., 129060195264000., 10559470521600.,
            670442572800., 33522128640., 1323241920., 40840800., 960960.,
            16380., 182., 1.)
    W1 = b[13]*A6 + b[11]*A4 + b[9]*A2
    W2 = b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident
    Z1 = b[12]*A6 + b[10]*A4 + b[8]*A2
    Z2 = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    W = jnp.dot(A6, W1) + W2
    U = jnp.dot(A, W)
    V = jnp.dot(A6, Z1) + Z2
    Lw1 = b[13]*M6 + b[11]*M4 + b[9]*M2
    Lw2 = b[7]*M6 + b[5]*M4 + b[3]*M2
    Lz1 = b[12]*M6 + b[10]*M4 + b[8]*M2
    Lz2 = b[6]*M6 + b[4]*M4 + b[2]*M2
    Lw = jnp.dot(A6, Lw1) + jnp.dot(M6, W1) + Lw2
    Lu = jnp.dot(A, Lw) + jnp.dot(E, W)
    Lv = jnp.dot(A6, Lz1) + jnp.dot(M6, Z1) + Lz2
    return U, V, Lu, Lv, s

api.defjvp(expm, lambda g, ans, matrix:
                    expm_frechet(matrix, g, compute_expm=False))

@_wraps(scipy.linalg.block_diag)
@jit
def block_diag(*arrs):
  if len(arrs) == 0:
    arrs = [jnp.zeros((1, 0))]
  arrs = jnp._promote_dtypes(*arrs)
  bad_shapes = [i for i, a in enumerate(arrs) if jnp.ndim(a) > 2]
  if bad_shapes:
    raise ValueError("Arguments to jax.scipy.linalg.block_diag must have at "
                     "most 2 dimensions, got {} at argument {}."
                     .format(arrs[bad_shapes[0]], bad_shapes[0]))
  arrs = [jnp.atleast_2d(a) for a in arrs]
  acc = arrs[0]
  dtype = lax.dtype(acc)
  for a in arrs[1:]:
    _, c = a.shape
    a = lax.pad(a, dtype.type(0), ((0, 0, 0), (acc.shape[-1], 0, 0)))
    acc = lax.pad(acc, dtype.type(0), ((0, 0, 0), (0, c, 0)))
    acc = lax.concatenate([acc, a], dimension=0)
  return acc
