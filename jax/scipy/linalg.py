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

import warnings

import scipy.linalg

from .. import lax
from .. import lax_linalg
from ..numpy.lax_numpy import _wraps
from ..numpy import lax_numpy as np
from ..numpy import linalg as np_linalg


_EXPERIMENTAL_WARNING = "scipy.linalg support is experimental and may cause silent failures or wrong outputs"

_T = lambda x: np.swapaxes(x, -1, -2)

@_wraps(scipy.linalg.cholesky)
def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
  warnings.warn(_EXPERIMENTAL_WARNING)
  del overwrite_a, check_finite
  l = lax_linalg.cholesky(a if lower else np.conj(a.T))
  return l if lower else np.conj(l.T)


@_wraps(scipy.linalg.det)
def det(a, overwrite_a=False, check_finite=True):
  warnings.warn(_EXPERIMENTAL_WARNING)
  del overwrite_a, check_finite
  return np_linalg.det(a)


@_wraps(scipy.linalg.inv)
def inv(a, overwrite_a=False, check_finite=True):
  warnings.warn(_EXPERIMENTAL_WARNING)
  del overwrite_a, check_finite
  return np_linalg.inv(a)


@_wraps(scipy.linalg.lu_factor)
def lu_factor(a, overwrite_a=False, check_finite=True):
  del overwrite_a, check_finite
  return lax_linalg.lu(a)


@_wraps(scipy.linalg.lu)
def lu(a, permute_l=False, overwrite_a=False, check_finite=True):
  del overwrite_a, check_finite
  lu, pivots = lax_linalg.lu(a)
  dtype = lax._dtype(a)
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


@_wraps(scipy.linalg.qr)
def qr(a, overwrite_a=False, lwork=None, mode="full", pivoting=False,
       check_finite=True):
  warnings.warn(_EXPERIMENTAL_WARNING)
  del overwrite_a, lwork, check_finite
  if pivoting:
    raise NotImplementedError(
        "The pivoting=True case of qr is not implemented.")
  if mode in ("full", "r"):
    full_matrices = True
  elif mode == "economic":
    full_matrices = False
  else:
    raise ValueError("Unsupported QR decomposition mode '{}'".format(mode))
  q, r = lax_linalg.qr(a, full_matrices)
  if mode == "r":
    return r
  return q, r

@_wraps(scipy.linalg.solve)
def solve(a, b, sym_pos=False, lower=False, overwrite_a=False, overwrite_b=False,
          debug=False, check_finite=True):
  del overwrite_a, overwrite_b, debug, check_finite
  if not sym_pos:
    return np_linalg.solve(a, b)

  a_shape = np.shape(a)
  b_shape = np.shape(b)
  a_ndims = len(a_shape)
  b_ndims = len(b_shape)
  if not (a_ndims >= 2 and a_shape[-1] == a_shape[-2] and
          (a_ndims == b_ndims or a_ndims == b_ndims + 1)):
    msg = ("The arguments to solve must have shapes a=[..., m, m] and "
           "b=[..., m, k] or b=[..., m]; got a={} and b={}")
    raise ValueError(msg.format(a_shape, b_shape))

  # TODO(phawkins): triangular_solve only supports matrices on the RHS, so we
  # add a dummy dimension. Extend it to support vectors and simplify this.
  b = b if a_ndims == b_ndims else b[..., None]
  lu = lax_linalg.cholesky(a if lower else np.conj(_T(a)))
  b = lax_linalg.triangular_solve(lu, b, left_side=True, lower=True)
  b = lax_linalg.triangular_solve(lu, b, left_side=True, lower=True,
                                  transpose_a=True, conjugate_a=True)
  return b[..., 0] if a_ndims != b_ndims else b



@_wraps(scipy.linalg.solve_triangular)
def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False,
                     overwrite_b=False, debug=None, check_finite=True):
  warnings.warn(_EXPERIMENTAL_WARNING)
  del overwrite_b, debug, check_finite

  if unit_diagonal:
    raise NotImplementedError("unit_diagonal=True is not implemented.")

  if trans == 0 or trans == "N":
    transpose_a, conjugate_a = False, False
  elif trans == 1 or trans == "T":
    transpose_a, conjugate_a = True, False
  elif trans == 2 or trans == "C":
    transpose_a, conjugate_a = True, True
  else:
    raise ValueError("Invalid 'trans' value {}".format(trans))

  a = np.tril(a) if lower else np.triu(a)

  # lax_linalg.triangular_solve only supports matrix 'b's at the moment.
  b_is_vector = np.ndim(a) == np.ndim(b) + 1
  if b_is_vector:
    b = b[..., None]
  out = lax_linalg.triangular_solve(a, b, left_side=True, lower=lower,
                                    transpose_a=transpose_a,
                                    conjugate_a=conjugate_a)
  if b_is_vector:
    return out[..., 0]
  else:
    return out


@_wraps(scipy.linalg.tril)
def tril(m, k=0):
  return np.tril(m, k)


@_wraps(scipy.linalg.triu)
def triu(m, k=0):
  return np.triu(m, k)
