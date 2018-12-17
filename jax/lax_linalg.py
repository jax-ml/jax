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

import numpy as onp

from jax.numpy import lax_numpy as np
from jax import core
from jax import lax
from jax.interpreters import xla
from jax.interpreters import ad
from jax.util import partial
from jax.abstract_arrays import ShapedArray
from jax.core import Primitive
from jax.lax import (standard_primitive, standard_unop, binop_dtype_rule,
                     _float, _complex, _input_dtype)
import jaxlib

# traceables

def cholesky(x): return cholesky_p.bind(x)

def qr(x, full_matrices=True):
  q, r = qr_p.bind(x, full_matrices=full_matrices)
  return q, r

def triangular_solve(a, b, left_side=False, lower=False, transpose_a=False,
                     conjugate_a=False):
  return triangular_solve_p.bind(
      a, b, left_side=left_side, lower=lower, transpose_a=transpose_a,
      conjugate_a=conjugate_a)


# utilities

def _T(x):
  return np.swapaxes(x, -1, -2)


# primitives


def cholesky_jvp_rule(primals, tangents):
  x, = primals
  sigma_dot, = tangents
  L = cholesky_p.bind(x)

  # Forward-mode rule from https://arxiv.org/pdf/1602.07527.pdf
  sigma_dot = (sigma_dot + _T(sigma_dot)) / 2
  phi = lambda X: np.tril(X) / (1 + np.eye(x.shape[-1]))
  tmp = triangular_solve(L, sigma_dot,
                         left_side=False, transpose_a=True, lower=True)
  L_dot = lax.dot(L, phi(triangular_solve(
      L, tmp, left_side=True, transpose_a=False, lower=True)))
  return L, L_dot

cholesky_p = standard_unop(_float, 'cholesky')
ad.primitive_jvps[cholesky_p] = cholesky_jvp_rule


def cholesky_cpu_translation_rule(c, operand):
  shape = c.GetShape(operand)
  if len(shape.dimensions()) == 2 and (
    shape.element_type() == np.float32 or shape.element_type() == np.float64):
    return c.GetTupleElement(jaxlib.lapack.jax_potrf(c, operand, lower=True), 0)
  else:
    # Fall back to the HLO implementation for batched Cholesky decomposition or
    # unsupported types.
    # TODO(phawkins): support LAPACK primitives in batched mode.
    return c.Cholesky(operand)

if hasattr(jaxlib, "lapack"):
  xla.backend_specific_translations['Host'][cholesky_p] = cholesky_cpu_translation_rule


triangular_solve_dtype_rule = partial(
    binop_dtype_rule, _input_dtype, (_float | _complex, _float | _complex),
    'triangular_solve')

def triangular_solve_shape_rule(a, b, left_side=False, **unused_kwargs):
  if a.ndim < 2:
    msg = "triangular_solve requires a.ndim to be at least 2, got {}."
    raise TypeError(msg.format(a.ndim))
  if a.shape[-1] != a.shape[-2]:
    msg = ("triangular_solve requires the last two dimensions of a to be equal "
           "in size, got a.shape of {}.")
    raise TypeError(msg.format(a.shape))
  if a.shape[:-2] != b.shape[:-2]:
    msg = ("triangular_solve requires both arguments to have the same number "
           "of dimensions and equal batch dimensions, got {} and {}.")
    raise TypeError(msg.format(a.shape, b.shape))
  common_dim = -2 if left_side else -1
  if a.shape[-1] != b.shape[common_dim]:
    msg = "Incompatible shapes for arguments to triangular_solve: {} and {}."
    raise TypeError(msg.format(a.shape, b.shape))
  return b.shape

def triangular_solve_transpose_rule(
    cotangent, a, b, left_side, lower, transpose_a, conjugate_a):
  assert a is not None and b is None
  cotangent_b = triangular_solve(a, cotangent, left_side, lower,
                                 not transpose_a, conjugate_a)
  return [None, cotangent_b]

triangular_solve_p = standard_primitive(
    triangular_solve_shape_rule, triangular_solve_dtype_rule,
    'triangular_solve')
ad.defjvp(triangular_solve_p,
          None,
          lambda g_b, a, b, **kwargs: triangular_solve(a, g_b, **kwargs))
ad.primitive_transposes[triangular_solve_p] = triangular_solve_transpose_rule


def triangular_solve_cpu_translation_rule(
    c, a, b, left_side, lower, transpose_a, conjugate_a):
  shape = c.GetShape(a)
  if len(shape.dimensions()) == 2 and shape.element_type() == np.float32:
    return jaxlib.lapack.jax_trsm(
      c, c.ConstantF32Scalar(1.0), a, b, left_side, lower, transpose_a,
      conjugate_a)
  elif len(shape.dimensions()) == 2 and shape.element_type() == np.float64:
    return jaxlib.lapack.jax_trsm(
      c, c.ConstantF64Scalar(1.0), a, b, left_side, lower, transpose_a,
      conjugate_a)
  else:
    # Fall back to the HLO implementation for batched triangular_solve or
    # unsupported types.
    # TODO(phawkins): support BLAS primitives in batched mode.
    return c.TriangularSolve(a, b, left_side, lower, transpose_a, conjugate_a)

if hasattr(jaxlib, "lapack"):
  xla.backend_specific_translations['Host'][triangular_solve_p] = triangular_solve_cpu_translation_rule


def qr_impl(operand, full_matrices):
  q, r = xla.apply_primitive(qr_p, operand, full_matrices=full_matrices)
  return core.pack((q, r))

def qr_translation_rule(c, operand, full_matrices):
  return c.QR(operand, full_matrices=full_matrices)

def qr_abstract_eval(operand, full_matrices):
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2:
      raise ValueError("Argument to QR decomposition must have ndims >= 2")
    batch_dims = operand.shape[:-2]
    m = operand.shape[-2]
    n = operand.shape[-1]
    k = m if full_matrices else min(m, n)
    q = ShapedArray(batch_dims + (m, k), operand.dtype)
    r = ShapedArray(batch_dims + (k, n), operand.dtype)
  else:
    q = operand
    r = operand
  return core.AbstractTuple((q, r))

def qr_dtype_rule(operand, full_matrices=True):
  return operand.dtype

def qr_jvp_rule(primals, tangents, full_matrices):
  # See j-towns.github.io/papers/qr-derivative.pdf for a terse derivation.
  x, = primals
  if full_matrices or np.shape(x)[-2] < np.shape(x)[-1]:
    raise NotImplementedError
  dx, = tangents
  q, r = qr_p.bind(x, full_matrices=False)
  dx_rinv = triangular_solve(r, dx)  # Right side solve by default
  qt_dx_rinv = np.matmul(_T(q), dx_rinv)
  qt_dx_rinv_lower = np.tril(qt_dx_rinv, -1)
  domega = qt_dx_rinv_lower - _T(qt_dx_rinv_lower)  # This is skew-symmetric
  dq = np.matmul(q, domega - qt_dx_rinv) + dx_rinv
  dr = np.matmul(qt_dx_rinv - domega, r)
  return core.pack((q, r)), core.pack((dq, dr))

qr_p = Primitive('qr')
qr_p.def_impl(qr_impl)
qr_p.def_abstract_eval(qr_abstract_eval)
xla.translations[qr_p] = qr_translation_rule
ad.primitive_jvps[qr_p] = qr_jvp_rule
