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
from jax import ad_util
from jax.interpreters import xla
from jax.interpreters import ad
from jax.interpreters import batching
from jax.util import partial
from jax.abstract_arrays import ShapedArray
from jax.core import Primitive
from jax.lax import (standard_primitive, standard_unop, binop_dtype_rule,
                     _float, _complex, _input_dtype)
from jaxlib import lapack

# traceables

def cholesky(x, symmetrize_input=True):
  if symmetrize_input:
    x = symmetrize(x)
  return np.tril(cholesky_p.bind(x))

def eig(x):
  w, vl, vr = eig_p.bind(x)
  return w, vl, vr

def eigh(x, lower=True, symmetrize_input=True):
  if symmetrize_input:
    x = symmetrize(x)
  v, w = eigh_p.bind(x, lower=lower)
  return v, w

def lu(x):
  lu, pivots = lu_p.bind(x)
  return lu, pivots

def qr(x, full_matrices=True):
  q, r = qr_p.bind(x, full_matrices=full_matrices)
  return q, r

def svd(x, full_matrices=True, compute_uv=True):
  s, u, v = svd_p.bind(x, full_matrices=full_matrices, compute_uv=compute_uv)
  if compute_uv:
    return u, s, v
  else:
    return s

def triangular_solve(a, b, left_side=False, lower=False, transpose_a=False,
                     conjugate_a=False):
  return triangular_solve_p.bind(
      a, b, left_side=left_side, lower=lower, transpose_a=transpose_a,
      conjugate_a=conjugate_a)


# utilities

def _T(x): return np.swapaxes(x, -1, -2)
def _H(x): return np.conj(_T(x))
def symmetrize(x): return (x + _H(x)) / 2


# primitives

_cpu_lapack_types = {np.float32, np.float64, np.complex64, np.complex128}

# Cholesky decomposition

def cholesky_jvp_rule(primals, tangents):
  x, = primals
  sigma_dot, = tangents
  L = np.tril(cholesky_p.bind(x))

  # Forward-mode rule from https://arxiv.org/pdf/1602.07527.pdf
  phi = lambda X: np.tril(X) / (1 + np.eye(X.shape[-1], dtype=X.dtype))
  tmp = triangular_solve(L, sigma_dot,
                         left_side=False, transpose_a=True, lower=True)
  L_dot = lax.batch_matmul(L, phi(triangular_solve(
      L, tmp, left_side=True, transpose_a=False, lower=True)))
  return L, L_dot

def cholesky_batching_rule(batched_args, batch_dims):
  x, = batched_args
  bd, = batch_dims
  x = batching.bdim_at_front(x, bd)
  return cholesky(x), 0

cholesky_p = standard_unop(_float | _complex, 'cholesky')
ad.primitive_jvps[cholesky_p] = cholesky_jvp_rule
batching.primitive_batchers[cholesky_p] = cholesky_batching_rule


def cholesky_cpu_translation_rule(c, operand):
  shape = c.GetShape(operand)
  dtype = shape.element_type().type
  if len(shape.dimensions()) == 2 and dtype in _cpu_lapack_types:
    return c.GetTupleElement(lapack.jax_potrf(c, operand, lower=True), 0)
  else:
    # Fall back to the HLO implementation for batched Cholesky decomposition or
    # unsupported types.
    # TODO(phawkins): support LAPACK primitives in batched mode.
    return c.Cholesky(operand)

xla.backend_specific_translations['cpu'][cholesky_p] = cholesky_cpu_translation_rule

# Asymmetric eigendecomposition

def eig_impl(operand):
  return xla.apply_primitive(eig_p, operand)

def eig_translation_rule(c, operand):
  raise NotImplementedError(
    "Nonsymmetric eigendecomposition is only implemented on the CPU backend")

def eig_abstract_eval(operand):
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2 or operand.shape[-2] != operand.shape[-1]:
      raise ValueError("Argument to nonsymmetric eigendecomposition must have "
                       "shape [..., n, n], got shape {}".format(operand.shape))

    batch_dims = operand.shape[:-2]
    n = operand.shape[-1]
    vl = vr = ShapedArray(batch_dims + (n, n), operand.dtype)
    w = ShapedArray(batch_dims + (n,), lax.lax._complex_basetype(operand.dtype))
  else:
    w = vl = vr = operand
  return core.AbstractTuple((w, vl, vr))

def eig_cpu_translation_rule(c, operand):
  out = lapack.jax_geev(c, operand)
  return c.Tuple(c.GetTupleElement(out, 0), c.GetTupleElement(out, 1),
                 c.GetTupleElement(out, 2))

def eig_batching_rule(batched_args, batch_dims):
  x, = batched_args
  bd, = batch_dims
  x = batching.bdim_at_front(x, bd)
  return eig_p.bind(x), 0

eig_p = Primitive('eig')
eig_p.def_impl(eig_impl)
eig_p.def_abstract_eval(eig_abstract_eval)
xla.translations[eig_p] = eig_translation_rule
xla.backend_specific_translations['cpu'][eig_p] = eig_cpu_translation_rule
batching.primitive_batchers[eig_p] = eig_batching_rule


# Symmetric/Hermitian eigendecomposition

def eigh_impl(operand, lower):
  v, w = xla.apply_primitive(eigh_p, operand, lower=lower)
  return core.pack((v, w))

def eigh_translation_rule(c, operand, lower):
  raise NotImplementedError(
    "Symmetric eigendecomposition is only implemented on the CPU backend")

def eigh_abstract_eval(operand, lower):
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2 or operand.shape[-2] != operand.shape[-1]:
      raise ValueError(
        "Argument to symmetric eigendecomposition must have shape [..., n, n],"
        "got shape {}".format(operand.shape))

    batch_dims = operand.shape[:-2]
    n = operand.shape[-1]
    v = ShapedArray(batch_dims + (n, n), operand.dtype)
    w = ShapedArray(batch_dims + (n,), lax.lax._complex_basetype(operand.dtype))
  else:
    v, w = operand, operand
  return core.AbstractTuple((v, w))

def eigh_cpu_translation_rule(c, operand, lower):
  out = lapack.jax_syevd(c, operand, lower=lower)
  return c.Tuple(c.GetTupleElement(out, 0), c.GetTupleElement(out, 1))

def eigh_jvp_rule(primals, tangents, lower):
  # Derivative for eigh in the simplest case of distinct eigenvalues.
  # This is classic nondegenerate perurbation theory, but also see
  # https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
  # The general solution treating the case of degenerate eigenvalues is
  # considerably more complicated. Ambitious readers may refer to the general
  # methods below or refer to degenerate perturbation theory in physics.
  # https://www.win.tue.nl/analysis/reports/rana06-33.pdf and
  # https://people.orie.cornell.edu/aslewis/publications/99-clarke.pdf
  a, = primals
  a_dot, = tangents
  v, w = eigh_p.bind(symmetrize(a), lower=lower)
  # for complex numbers we need eigenvalues to be full dtype of v, a:
  w = w.astype(a.dtype)
  eye_n = np.eye(a.shape[-1], dtype=a.dtype)
  # carefully build reciprocal delta-eigenvalue matrix, avoiding NaNs.
  Fmat = np.reciprocal(eye_n + w - w[..., np.newaxis]) - eye_n
  # eigh impl doesn't support batch dims, but future-proof the grad.
  dot = lax.dot if a.ndim == 2 else lax.batch_matmul
  vdag_adot_v = dot(dot(_H(v), a_dot), v)
  dv = dot(v, np.multiply(Fmat, vdag_adot_v))
  dw = np.diagonal(vdag_adot_v)
  return core.pack((v, w)), core.pack((dv, dw))

def eigh_batching_rule(batched_args, batch_dims, lower):
  x, = batched_args
  bd, = batch_dims
  x = batching.bdim_at_front(x, bd)
  return eigh_p.bind(x, lower=lower), 0

eigh_p = Primitive('eigh')
eigh_p.def_impl(eigh_impl)
eigh_p.def_abstract_eval(eigh_abstract_eval)
xla.translations[eigh_p] = eigh_translation_rule
ad.primitive_jvps[eigh_p] = eigh_jvp_rule
xla.backend_specific_translations['cpu'][eigh_p] = eigh_cpu_translation_rule
batching.primitive_batchers[eigh_p] = eigh_batching_rule



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

def triangular_solve_jvp_rule_a(
    g_a, ans, a, b, left_side, lower, transpose_a, conjugate_a):
  g_a = lax.neg(g_a)
  g_a = np.swapaxes(g_a, -1, -2) if transpose_a else g_a
  tmp = triangular_solve(a, g_a, left_side, lower, transpose_a, conjugate_a)
  dot = lax.dot if g_a.ndim == 2 else lax.batch_matmul
  if left_side:
    return dot(tmp, ans)
  else:
    return dot(ans, tmp)

def triangular_solve_transpose_rule(
    cotangent, a, b, left_side, lower, transpose_a, conjugate_a):
  # Triangular solve is linear in its first argument and nonlinear in its second
  # argument, similar to `div`. We need both a JVP rule and a transpose rule
  # for the first argument.
  assert a is not None and b is None
  cotangent_b = triangular_solve(a, cotangent, left_side, lower,
                                 not transpose_a, conjugate_a)
  return [None, cotangent_b]


def triangular_solve_batching_rule(batched_args, batch_dims, left_side,
                                   lower, transpose_a, conjugate_a):
  x, y = batched_args
  bx, by = batch_dims
  size = next(t.shape[i] for t, i in zip(batched_args, batch_dims)
              if i is not None)
  x = batching.bdim_at_front(x, bx, size, force_broadcast=True)
  y = batching.bdim_at_front(y, by, size, force_broadcast=True)
  return triangular_solve(x, y, left_side=left_side, lower=lower,
                          transpose_a=transpose_a, conjugate_a=conjugate_a), 0

triangular_solve_p = standard_primitive(
    triangular_solve_shape_rule, triangular_solve_dtype_rule,
    'triangular_solve')
ad.defjvp2(triangular_solve_p,
           triangular_solve_jvp_rule_a,
           lambda g_b, _, a, b, **kws: triangular_solve(a, g_b, **kws))
ad.primitive_transposes[triangular_solve_p] = triangular_solve_transpose_rule
batching.primitive_batchers[triangular_solve_p] = triangular_solve_batching_rule


def triangular_solve_cpu_translation_rule(
    c, a, b, left_side, lower, transpose_a, conjugate_a):
  shape = c.GetShape(a)
  dtype = shape.element_type().type
  if len(shape.dimensions()) == 2 and dtype in _cpu_lapack_types:
    return lapack.jax_trsm(
      c, c.Constant(onp.array(1, dtype=dtype)), a, b, left_side, lower,
                    transpose_a, conjugate_a)
  else:
    # Fall back to the HLO implementation for batched triangular_solve or
    # unsupported types.
    # TODO(phawkins): support BLAS primitives in batched mode.
    return c.TriangularSolve(a, b, left_side, lower, transpose_a, conjugate_a)

xla.backend_specific_translations['cpu'][triangular_solve_p] = triangular_solve_cpu_translation_rule


# LU decomposition

# Computes a pivoted LU decomposition such that
# PA = LU
# In the style of LAPACK, LU are stored in the same matrix.
# TODO(phawkins): add a mechanism to report errors for singular matrices.

def lu_impl(operand):
  lu, pivot = xla.apply_primitive(lu_p, operand)
  return core.pack((lu, pivot))

def lu_translation_rule(c, operand):
  raise NotImplementedError(
    "LU decomposition is only implemented on the CPU backend")

def lu_abstract_eval(operand):
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2:
      raise ValueError("Argument to LU decomposition must have ndims >= 2")

    batch_dims = operand.shape[:-2]
    m = operand.shape[-2]
    n = operand.shape[-1]
    pivot = ShapedArray(batch_dims + (min(m, n),), np.int32)
  else:
    pivot = operand
  return core.AbstractTuple((operand, pivot))

def lu_jvp_rule(primals, tangents):
  a, = primals
  a_dot, = tangents
  lu, pivots = lu_p.bind(a)

  a_shape = np.shape(a)
  m, n = a_shape[-2:]
  dtype = lax.dtype(a)
  k = min(m, n)

  permutation = lu_pivots_to_permutation(pivots, m)
  x = a_dot[..., permutation, :]

  # Differentiation of Matrix Functionals Using Triangular Factorization
  # F. R. De Hoog, R. S. Anderssen, and M. A. Lukas
  #
  #     LU = A
  # ==> L'U + LU' = A'
  # ==> inv(L) . L' + U' . inv(U) = inv(L) A' inv(U)
  # ==> L' = L . tril(inv(L) . A' . inv(U), -1)
  #     U' = triu(inv(L) . A' . inv(U)) . U

  ndims = len(a_shape)
  l_padding = [(0, 0, 0)] * ndims
  l_padding[-1] = (0, m - k, 0)
  zero = np._constant_like(lu, 0)
  l = lax.pad(np.tril(lu[..., :, :k], -1), zero, l_padding)
  l = l + np.eye(m, m, dtype=dtype)

  u_eye = lax.pad(np.eye(n - k, n - k, dtype=dtype), zero,
                  ((k, 0, 0), (k, 0, 0)))
  u_padding = [(0, 0, 0)] * ndims
  u_padding[-2] = (0, n - k, 0)
  u = lax.pad(np.triu(lu[..., :k, :]), zero, u_padding) + u_eye


  la = triangular_solve(l, x, left_side=True, transpose_a=False, lower=True)
  lau = triangular_solve(u, la, left_side=False, transpose_a=False,
                         lower=False)

  l_dot = np.matmul(l, np.tril(lau, -1))
  u_dot = np.matmul(np.triu(lau), u)
  lu_dot = l_dot + u_dot
  return core.pack((lu, pivots)), ad.TangentTuple((lu_dot, ad_util.zero))


def lu_batching_rule(batched_args, batch_dims):
  x, = batched_args
  bd, = batch_dims
  x = batching.bdim_at_front(x, bd)
  return lu_p.bind(x), 0


lu_p = Primitive('lu')
lu_p.def_impl(lu_impl)
lu_p.def_abstract_eval(lu_abstract_eval)
xla.translations[lu_p] = lu_translation_rule
ad.primitive_jvps[lu_p] = lu_jvp_rule
batching.primitive_batchers[lu_p] = lu_batching_rule

def lu_cpu_translation_rule(c, operand):
  shape = c.GetShape(operand)
  dtype = shape.element_type().type
  out = lapack.jax_getrf(c, operand)
  lu = c.GetTupleElement(out, 0)
  # Subtract 1 from the pivot to get 0-based indices.
  pivot = c.Sub(c.GetTupleElement(out, 1), c.ConstantS32Scalar(1))
  # Throw away the `info` value, because we have no way to report errors.
  return c.Tuple(lu, pivot)

xla.backend_specific_translations['cpu'][lu_p] = lu_cpu_translation_rule


def lu_pivots_to_permutation(swaps, k):
  """Converts the pivots (row swaps) returned by LU to a permutation."""

  def body_fn(i, loop_carry):
    swaps, permutation = loop_carry
    j = swaps[i]
    x, y = np.ravel(permutation[i]), np.ravel(permutation[j])
    permutation = lax.dynamic_update_index_in_dim(permutation, y, i, axis=0)
    permutation = lax.dynamic_update_index_in_dim(permutation, x, j, axis=0)
    return swaps, permutation

  n, = np.shape(swaps)
  permutation = np.arange(k)
  _, permutation = lax.fori_loop(
      onp.array(0, onp.int32), onp.array(n, onp.int32), body_fn, (swaps, permutation))
  return permutation



# QR decomposition

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

def qr_batching_rule(batched_args, batch_dims, full_matrices):
  x, = batched_args
  bd, = batch_dims
  x = batching.bdim_at_front(x, bd)
  return qr_p.bind(x, full_matrices=full_matrices), 0

qr_p = Primitive('qr')
qr_p.def_impl(qr_impl)
qr_p.def_abstract_eval(qr_abstract_eval)
xla.translations[qr_p] = qr_translation_rule
ad.primitive_jvps[qr_p] = qr_jvp_rule
batching.primitive_batchers[qr_p] = qr_batching_rule


# Singular value decomposition

def svd_impl(operand, full_matrices, compute_uv):
  s, u, vt = xla.apply_primitive(svd_p, operand, full_matrices=full_matrices, compute_uv=compute_uv)
  return core.pack((s, u, vt))

def svd_translation_rule(c, operand, full_matrices, compute_uv):
  raise NotImplementedError(
    "Singular value decomposition is only implemented on the CPU backend")

def svd_abstract_eval(operand, full_matrices, compute_uv):
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2:
      raise ValueError("Argument to singular value decomposition must have ndims >= 2")

    batch_dims = operand.shape[:-2]
    m = operand.shape[-2]
    n = operand.shape[-1]
    s = ShapedArray(batch_dims + (min(m, n),), operand.dtype)
    u = ShapedArray(batch_dims + (m, m if full_matrices else min(m, n)), operand.dtype)
    vt = ShapedArray(batch_dims + (n if full_matrices else min(m, n), n), operand.dtype)
  else:
    s = operand
    u = operand
    vt = operand
  return core.AbstractTuple((s, u, vt))

def svd_jvp_rule(primals, tangents, full_matrices, compute_uv):
  if full_matrices:
    #TODO: implement full matrices case, documented here: https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
    raise NotImplementedError("Singular value decomposition JVP not implemented for full matrices")

  A, = primals
  dA, = tangents
  s, U, Vt = svd_p.bind(A, full_matrices=False, compute_uv=True)

  k = s.shape[-1]
  Ut, V = np.conj(U).T, np.conj(Vt).T
  s_dim = s[..., None, :]
  dS = np.dot(np.dot(Ut, dA), V)
  ds = np.real(np.diag(dS))
  F = 1 / (np.square(s_dim) - np.square(s_dim.T) + np.eye(k)) - np.eye(k)
  dSS = s_dim * dS
  SdS = s_dim.T * dS
  dU = np.dot(U, F * (dSS + dSS.T))
  dV = np.dot(V, F * (SdS + SdS.T))

  m, n = A.shape[-2], A.shape[-1]
  if m > n:
    dU = dU + np.dot(np.eye(m) - np.dot(U, Ut), np.dot(dA, V)) / s_dim
  if n > m:
    dV = dV + np.dot(np.eye(n) - np.dot(V, Vt), np.dot(np.conj(dA).T, U)) / s_dim
  return core.pack((s, U, Vt)), core.pack((ds, dU, dV.T))

def svd_cpu_translation_rule(c, operand, full_matrices, compute_uv):
  shape = c.GetShape(operand)
  dtype = shape.element_type().type
  if len(shape.dimensions()) == 2 and dtype in _cpu_lapack_types:
    out = lapack.jax_gesdd(c, operand, full_matrices=full_matrices, compute_uv=compute_uv)
    return c.Tuple(c.GetTupleElement(out, 0),
                   c.GetTupleElement(out, 1),
                   c.GetTupleElement(out, 2))
  else:
    raise NotImplementedError(
        "Only unbatched singular value decomposition is implemented on CPU")

def svd_batching_rule(batched_args, batch_dims, full_matrices, compute_uv):
  x, = batched_args
  bd, = batch_dims
  x = batching.bdim_at_front(x, bd)
  return svd_p.bind(x, full_matrices=full_matrices, compute_uv=compute_uv), 0

svd_p = Primitive('svd')
svd_p.def_impl(svd_impl)
svd_p.def_abstract_eval(svd_abstract_eval)
xla.translations[svd_p] = svd_translation_rule
xla.backend_specific_translations['cpu'][svd_p] = svd_cpu_translation_rule
ad.primitive_jvps[svd_p] = svd_jvp_rule
batching.primitive_batchers[svd_p] = svd_batching_rule
