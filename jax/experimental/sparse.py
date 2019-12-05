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

import scipy.sparse.linalg
import jax.numpy as np
from jax.numpy.lax_numpy import _wraps
from .. import lax

_T = lambda x: np.swapaxes(x, -1, -2)

@_wraps(scipy.sparse.linalg.cg)
def cg(A, b, x0=None, tol=1e-05, maxiter=None):
  # solves for x where Ax = b  
  N = np.shape(A)[0]
  if x0 is None:
    x_k = np.zeros(N)
  else:
    x_k = x0
    if not (np.shape(x0) == (N, 1) or np.shape(x0) == (N,)):
      raise ValueError('A and x have incompatible dimensions')
  if maxiter is None:
    maxiter = np.inf
  r_k = b - np.matmul(A, x_k)
  p_k = r_k
  k = 0
  A, p_k, x_k, r_k, k = lax.while_loop(
    lambda r_k, k: (r_k < tol) & (k < maxiter), body_fun, (A, p_k, x_k, r_k, k)
    )
  return x_k

def body_fun(A, p, x, r, k):
  # inspired by https://en.wikipedia.org/wiki/Conjugate_gradient_method#Example_code_in_MATLAB_/_GNU_Octave
  r_sq = np.matmul(_T(r), r)
  Ap = np.matmul(A, p)
  alpha = r_sq / np.matmul(_T(p), Ap)
  x_new = x + alpha * p
  r_new = r - alpha * Ap
  r_sq_new = np.matmul(_T(r_new), r_new)
  p_new = r_new + (r_sq_new / r_sq) * p
  return A, p_new, x_new, r_new, k+1
