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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import scipy.linalg

from jax import jit
from .. import lax
from .. import lax_linalg
from ..numpy.lax_numpy import _wraps
from ..numpy import lax_numpy as np
from ..numpy import linalg as np_linalg

_T = lambda x: np.swapaxes(x, -1, -2)

@partial(jit, static_argnums=(1,))
def _cholesky(a, lower):
  a = np_linalg._promote_arg_dtypes(np.asarray(a))
  l = lax_linalg.cholesky(a if lower else np.conj(_T(a)), symmetrize_input=False)
  return l if lower else np.conj(_T(l))

@_wraps(scipy.linalg.cholesky)
def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
  del overwrite_a, check_finite
  return _cholesky(a, lower)

@_wraps(scipy.linalg.cho_factor)
def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
  return (cholesky(a, lower=lower), lower)

@partial(jit, static_argnums=(2,))
def _cho_solve(c, b, lower):
  c, b = np_linalg._promote_arg_dtypes(np.asarray(c), np.asarray(b))
  c_shape = np.shape(c)
  b_shape = np.shape(b)
  c_ndims = len(c_shape)
  b_ndims = len(b_shape)
  if not (c_ndims >= 2 and c_shape[-1] == c_shape[-2] and
          (c_ndims == b_ndims or c_ndims == b_ndims + 1)):
    msg = ("The arguments to solve must have shapes a=[..., m, m] and "
           "b=[..., m, k] or b=[..., m]; got a={} and b={}")
    raise ValueError(msg.format(c_shape, b_shape))

  # TODO(phawkins): triangular_solve only supports matrices on the RHS, so we
  # add a dummy dimension. Extend it to support vectors and simplify this.
  b = b if c_ndims == b_ndims else b[..., None]
  b = lax_linalg.triangular_solve(c, b, left_side=True, lower=lower,
                                  transpose_a=not lower, conjugate_a=not lower)
  b = lax_linalg.triangular_solve(c, b, left_side=True, lower=lower,
                                  transpose_a=lower, conjugate_a=lower)
  return b[..., 0] if c_ndims != b_ndims else b

@_wraps(scipy.linalg.cho_solve, update_doc=False)
def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
  del overwrite_b, check_finite
  c, lower = c_and_lower
  return _cho_solve(c, b, lower)

@_wraps(scipy.linalg.svd)
def svd(a, full_matrices=True, compute_uv=True, overwrite_a=False,
        check_finite=True, lapack_driver='gesdd'):
  del overwrite_a, check_finite, lapack_driver
  a = np_linalg._promote_arg_dtypes(np.asarray(a))
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

  a = np_linalg._promote_arg_dtypes(np.asarray(a))
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
  a = np_linalg._promote_arg_dtypes(np.asarray(a))
  return lax_linalg.lu(a)

@partial(jit, static_argnums=(3,))
def _lu_solve(lu, pivots, b, trans):
  lu_shape = np.shape(lu)
  b_shape = np.shape(b)
  if len(lu_shape) != 2 or lu_shape[0] != lu_shape[1]:
    raise ValueError("LU decomposition must be a square matrix, got shape {}"
                     .format(lu_shape))
  if len(b_shape) < 1:
    raise ValueError("b matrix must have rank >= 1, got shape {}"
                     .format(b_shape))

  if b_shape[0] != lu_shape[0]:
    raise ValueError("Dimension of LU decomposition matrix (shape {}) must "
                     "match leading axis of b array (shape {})"
                     .format(lu_shape, b_shape))
  m = lu_shape[0]
  permutation = lax_linalg.lu_pivots_to_permutation(np.array(pivots), m)
  x = np.reshape(b, (m, -1))
  if trans == 0:
    x = x[permutation, :]
    x = lax_linalg.triangular_solve(lu, x, left_side=True, lower=True,
                                    unit_diagonal=True)
    x = lax_linalg.triangular_solve(lu, x, left_side=True, lower=False)
  elif trans == 1 or trans == 2:
    conj = trans == 2
    x = lax_linalg.triangular_solve(lu, x, left_side=True, lower=False,
                                    transpose_a=True, conjugate_a=conj)
    x = lax_linalg.triangular_solve(lu, x, left_side=True, lower=True,
                                    unit_diagonal=True, transpose_a=True,
                                    conjugate_a=conj)
    x = x[np.argsort(permutation), :]
  else:
    raise ValueError("'trans' value must be 0, 1, or 2, got {}".format(trans))
  return lax.reshape(x, b_shape)

@_wraps(scipy.linalg.lu_solve)
def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
  del overwrite_b, check_finite
  lu, pivots = lu_and_piv
  return _lu_solve(lu, pivots, b, trans)


@partial(jit, static_argnums=(1,))
def _lu(a, permute_l):
  a = np_linalg._promote_arg_dtypes(np.asarray(a))
  lu, pivots = lax_linalg.lu(a)
  dtype = lax.dtype(a)
  m, n = np.shape(a)
  permutation = lax_linalg.lu_pivots_to_permutation(pivots, m)
  p = np.real(np.array(permutation == np.arange(m)[:, None], dtype=dtype))
  k = min(m, n)
  l = np.tril(lu, -1)[:, :k] + np.eye(m, k, dtype=dtype)
  u = np.triu(lu)[:k, :]
  if permute_l:
    return np.matmul(p, l), u
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
  a = np_linalg._promote_arg_dtypes(np.asarray(a))
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

  a, b = np_linalg._promote_arg_dtypes(np.asarray(a), np.asarray(b))
  return cho_solve(cho_factor(a, lower=lower), b)

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

  a, b = np_linalg._promote_arg_dtypes(np.asarray(a), np.asarray(b))

  # lax_linalg.triangular_solve only supports matrix 'b's at the moment.
  b_is_vector = np.ndim(a) == np.ndim(b) + 1
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
  return np.tril(m, k)


@_wraps(scipy.linalg.triu)
def triu(m, k=0):
  return np.triu(m, k)

@jit
def expm(A):
    """
    Compute the matrix exponential using Pade approximation.
    Based on TensorFlow's tf.linalg.expm

    Args:
    A : square matrix of type float32, float64, complex64, complex128
    
    Returns:
    Matrix exponential of A , square matrix of same dimensions as A
        

    See: "The Scaling and Squaring Method for the Matrix
    Exponential Revisited", Nicholas Higham, 2005.
    
    TODO: Write a version to exploit the sparsity of A
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be a square matrix')
    A_L1 = np_linalg.norm(A,1)
    n_squarings = 0
    def _nest_where(vals, cases):
      assert len(vals) == len(cases) - 1
      if len(vals) == 1:
        return np.where(
            np.less(A_L1, vals[0]), cases[0], cases[1])
      else:
        return np.where(
            np.less(A_L1, vals[0]), cases[0],
            _nest_where(vals[1:], cases[1:]))
    if A.dtype == 'float64' or A.dtype == 'complex128':
       U3,V3 = _pade3(A)
       U5,V5 = _pade5(A)
       U7,V7 = _pade7(A)
       U9,V9 = _pade9(A)
       maxnorm = 5.371920351148152
       n_squarings = np.maximum(0, np.floor_divide(np.log2(A_L1 / maxnorm),1))
       A = A / 2**n_squarings
       U13,V13 = _pade13(A)
       conds=np.array([1.495585217958292e-002, 2.539398330063230e-001, 9.504178996162932e-001, 2.097847961257068e+000])
       U=_nest_where(conds,(U3,U5,U7,U9,U13))
       V=_nest_where(conds,(V3,V5,V7,V9,V13))
    elif A.dtype == 'float32' or A.dtype == 'complex64':
        U3,V3 = _pade3(A)
        U5,V5 = _pade5(A)
        maxnorm = 3.925724783138660
        n_squarings = np.maximum(0, np.floor_divide(np.log2(A_L1 / maxnorm),1))
        A = A / 2**n_squarings
        U7,V7 = _pade7(A)
        conds=np.array([4.258730016922831e-001, 1.880152677804762e+000])
        U=_nest_where(conds,(U3,U5,U7))
        V=_nest_where(conds,(V3,V5,V7))
    else:
        raise TypeError("A.dtype={} is not supported.".format(A.dtype))
    P = U + V  # p_m(A) : numerator
    Q = -U + V # q_m(A) : denominator
    R = np_linalg.solve(Q,P)
    # squaring step to undo scaling
    def my_body_fun(i,R):
      return np.dot(R,R)
    lower = np.zeros(1, dtype=n_squarings.dtype)
    R = lax.fori_loop(lower[0],n_squarings,my_body_fun,R)
    return R

def _pade3(A):
    b = (120., 60., 12., 1.)
    ident = np.eye(*A.shape, dtype=A.dtype)
    A2 = np.dot(A,A)
    U = np.dot(A , (b[3]*A2 + b[1]*ident))
    V = b[2]*A2 + b[0]*ident
    return U,V

def _pade5(A):
    b = (30240., 15120., 3360., 420., 30., 1.)
    ident = np.eye(*A.shape, dtype=A.dtype)
    A2 = np.dot(A,A)
    A4 = np.dot(A2,A2)
    U = np.dot(A, b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[4]*A4 + b[2]*A2 + b[0]*ident
    return U,V

def _pade7(A):
    b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
    ident = np.eye(*A.shape, dtype=A.dtype)
    A2 = np.dot(A,A)
    A4 = np.dot(A2,A2)
    A6 = np.dot(A4,A2)
    U = np.dot(A, b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U,V

def _pade9(A):
    b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
                2162160., 110880., 3960., 90., 1.)
    ident = np.eye(*A.shape, dtype=A.dtype)
    A2 = np.dot(A,A)
    A4 = np.dot(A2,A2)
    A6 = np.dot(A4,A2)
    A8 = np.dot(A6,A2)
    U = np.dot(A, b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[8]*A8 + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U,V

def _pade13(A):
    b = (64764752532480000., 32382376266240000., 7771770303897600.,
    1187353796428800., 129060195264000., 10559470521600., 670442572800.,
    33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.)
    ident = np.eye(*A.shape, dtype=A.dtype)
    A2 = np.dot(A,A)
    A4 = np.dot(A2,A2)
    A6 = np.dot(A4,A2)
    U = np.dot(A,np.dot(A6, b[13]*A6 + b[11]*A4 + b[9]*A2) + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = np.dot(A6, b[12]*A6 + b[10]*A4 + b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U,V
