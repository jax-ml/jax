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

import scipy.sparse.linalg
import jax.numpy as np
from jax.numpy.lax_numpy import _wraps
from .. import lax

_T = lambda x: np.swapaxes(np.conj(x), -1, -2)

def body_fun(matvec, x):
  p, x_current, r, r_norm, k = x
  # inspired by https://en.wikipedia.org/wiki/Conjugate_gradient_method#Example_code_in_MATLAB_/_GNU_Octave
  Ap = matvec(p)
  alpha = r_norm / np.matmul(_T(p), Ap)
  x_new = x_current + alpha * p
  r_new = r - alpha * Ap
  r_norm_new = np.linalg.norm(r_new)
  p_new = r_new + (r_norm_new / r_norm) * p
  return p_new, x_new, r_new, r_norm_new, k+1

def _cg_solve(matvec, b, x0, tol, maxiter):
  N = np.shape(b)[0]
  if x0 is None:
    x_k = np.zeros_like(b)
  else:
    x_k = x0
    if not np.shape(x0) == (N,):
      raise ValueError('A and x have incompatible dimensions')
    if maxiter is None:
      maxiter = N*10
  r_k = b - matvec(x_k)
  r_k_norm = np.linalg.norm(r_k)
  p_k = r_k
  k = 0
  full_tol = tol * np.linalg.norm(b)
  p_k, x_k, r_k, r_k_norm, k = lax.while_loop(
      lambda x: (x[-2] < full_tol) & (x[-1] < maxiter),
      partial(body_fun, matvec),
      (p_k, x_k, r_k, r_k_norm, k)
  )
  return x_k

@_wraps(scipy.sparse.linalg.cg)
def cg(matvec, b, x0=None, tol=1e-05, maxiter=None):
  # exposed as scipy.sparse.cg
  info = 0
  cg_solve = lambda matvec, b: _cg_solve(matvec, b, x0, tol, maxiter)
  x = lax.custom_linear_solve(matvec, b, cg_solve, symmetric=True)
  return x, info
