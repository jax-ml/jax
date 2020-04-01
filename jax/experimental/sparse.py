# Copyright 2020 Google LLC
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
import textwrap

import scipy.sparse.linalg
import jax.numpy as jnp
import numpy as np
from jax.numpy.lax_numpy import _wraps
from .. import lax


def _vdot(x, y):
  return jnp.vdot(x, y, precision=lax.Precision.HIGHEST)


def _cg_solve(matvec, b, x0=None, *, tol=1e-5, atol=0.0, maxiter=None):
  # inspired by https://en.wikipedia.org/wiki/Conjugate_gradient_method#Example_code_in_MATLAB_/_GNU_Octave

  # copied from the non-legacy behavior of scipy.sparse.linalg.cg
  bs = _vdot(b, b)
  atol2 = jnp.maximum(tol ** 2 * bs, atol ** 2)

  def cond_fun(value):
    x, r, rs, p, k = value
    return (rs > atol2) & (maxiter is None or k < maxiter)

  def body_fun(value):
    x, r, rs, p, k = value
    Ap = matvec(p)
    alpha = rs / _vdot(p, Ap)
    x_new = x + alpha * p
    r_new = r - alpha * Ap
    rs_new = _vdot(r_new, r_new)
    beta = rs_new / rs
    p_new = r_new + beta * p
    return x_new, r_new, rs_new, p_new, k + 1

  r0 = b - matvec(x0)
  rs0 = _vdot(r0, r0)
  initial_value = (x0, r0, rs0, r0, 0)

  x_final, *_ = lax.while_loop(cond_fun, body_fun, initial_value)

  return x_final


@_wraps(scipy.sparse.linalg.cg,
    lax_description=textwrap.dedent("""\
        Unlike scipy.sparse.linalg.cg, the first argument should be a function
        that returns the matrix-vector product, not a LinearOperator. Also, the
        return code ``info`` is currently always fixed at 0.
        """))
def cg(matvec, b, x0=None, *, tol=1e-5, atol=0.0, maxiter=None):
  if x0 is None:
    x0 = jnp.zeros_like(b)
  if x0.shape != b.shape:
    raise ValueError('x0 and b must have matching shape')
  cg_solve = partial(_cg_solve, x0=x0, tol=tol, atol=atol, maxiter=maxiter)
  x = lax.custom_linear_solve(matvec, b, cg_solve, symmetric=True)
  info = 0  # TODO(shoyer): return the real iteration count here
  return x, info
