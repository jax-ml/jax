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

import numpy as onp
import textwrap
import operator
from typing import Tuple, Union, cast

from jax import jit, vmap, custom_jvp
from .. import lax
from .. import ops
from .. import lax_linalg
from .. import dtypes
from .lax_numpy import _not_implemented
from .lax_numpy import _wraps
from .vectorize import vectorize
from . import lax_numpy as np
from ..util import get_module_functions
from ..third_party.numpy.linalg import cond, tensorinv, tensorsolve

_T = lambda x: np.swapaxes(x, -1, -2)


def _promote_arg_dtypes(*args):
  """Promotes `args` to a common inexact type."""
  def _to_inexact_type(type):
    return type if np.issubdtype(type, np.inexact) else np.float_
  inexact_types = [_to_inexact_type(np._dtype(arg)) for arg in args]
  dtype = dtypes.canonicalize_dtype(np.result_type(*inexact_types))
  args = [lax.convert_element_type(arg, dtype) for arg in args]
  if len(args) == 1:
    return args[0]
  else:
    return args


@_wraps(onp.linalg.cholesky)
def cholesky(a):
  a = _promote_arg_dtypes(np.asarray(a))
  return lax_linalg.cholesky(a)


@_wraps(onp.linalg.svd)
def svd(a, full_matrices=True, compute_uv=True):
  a = _promote_arg_dtypes(np.asarray(a))
  return lax_linalg.svd(a, full_matrices, compute_uv)


@_wraps(onp.linalg.matrix_power)
def matrix_power(a, n):
  a = _promote_arg_dtypes(np.asarray(a))

  if a.ndim < 2:
    raise TypeError("{}-dimensional array given. Array must be at least "
                    "two-dimensional".format(a.ndim))
  if a.shape[-2] != a.shape[-1]:
    raise TypeError("Last 2 dimensions of the array must be square")
  try:
    n = operator.index(n)
  except TypeError:
    raise TypeError("exponent must be an integer, got {}".format(n))

  if n == 0:
    return np.broadcast_to(np.eye(a.shape[-2], dtype=a.dtype), a.shape)
  elif n < 0:
    a = inv(a)
    n = np.abs(n)

  if n == 1:
    return a
  elif n == 2:
    return a @ a
  elif n == 3:
    return (a @ a) @ a

  z = result = None
  while n > 0:
    z = a if z is None else (z @ z)
    n, bit = divmod(n, 2)
    if bit:
      result = z if result is None else (result @ z)

  return result


@_wraps(onp.linalg.matrix_rank)
def matrix_rank(M, tol=None):
  M = _promote_arg_dtypes(np.asarray(M))
  if M.ndim > 2:
    raise TypeError("array should have 2 or fewer dimensions")
  if M.ndim < 2:
    return np.any(M != 0).astype(np.int32)
  S = svd(M, full_matrices=False, compute_uv=False)
  if tol is None:
    tol = S.max() * np.max(M.shape) * np.finfo(S.dtype).eps
  return np.sum(S > tol)


@custom_jvp
@_wraps(onp.linalg.slogdet)
@jit
def slogdet(a):
  a = _promote_arg_dtypes(np.asarray(a))
  dtype = lax.dtype(a)
  a_shape = np.shape(a)
  if len(a_shape) < 2 or a_shape[-1] != a_shape[-2]:
    msg = "Argument to slogdet() must have shape [..., n, n], got {}"
    raise ValueError(msg.format(a_shape))
  lu, pivot = lax_linalg.lu(a)
  diag = np.diagonal(lu, axis1=-2, axis2=-1)
  is_zero = np.any(diag == np.array(0, dtype=dtype), axis=-1)
  parity = np.count_nonzero(pivot != np.arange(a_shape[-1]), axis=-1)
  if np.iscomplexobj(a):
    sign = np.prod(diag / np.abs(diag), axis=-1)
  else:
    sign = np.array(1, dtype=dtype)
    parity = parity + np.count_nonzero(diag < 0, axis=-1)
  sign = np.where(is_zero,
                  np.array(0, dtype=dtype),
                  sign * np.array(-2 * (parity % 2) + 1, dtype=dtype))
  logdet = np.where(
      is_zero, np.array(-np.inf, dtype=dtype),
      np.sum(np.log(np.abs(diag)), axis=-1))
  return sign, np.real(logdet)

@slogdet.defjvp
def _slogdet_jvp(primals, tangents):
  x, = primals
  g, = tangents
  if np.issubdtype(np._dtype(x), np.complexfloating):
    raise NotImplementedError  # TODO(pfau): make this work for complex types
  sign, ans = slogdet(x)
  sign_dot, ans_dot = np.zeros_like(sign), np.trace(solve(x, g), axis1=-1, axis2=-2)
  return (sign, ans), (sign_dot, ans_dot)


def _cofactor_solve(a, b):
  """Equivalent to det(a)*solve(a, b) for nonsingular mat.

  Intermediate function used for jvp and vjp of det.
  This function borrows heavily from jax.numpy.linalg.solve and
  jax.numpy.linalg.slogdet to compute the gradient of the determinant
  in a way that is well defined even for low rank matrices.
  * assumes a is at least rank n-1
  * assumes u_{nn} is the element set to 0 in rank n-1 cases.

  Args:
    a: A square matrix or batch of matrices, possibly singular.
    b: A matrix, or batch of matrices of the same dimension as a.

  Returns:
    det(a) and cofactor(a)^T*b, aka adjugate(a)*b
  """
  a = _promote_arg_dtypes(np.asarray(a))
  b = _promote_arg_dtypes(np.asarray(b))
  a_shape = np.shape(a)
  b_shape = np.shape(b)
  a_ndims = len(a_shape)
  b_ndims = len(b_shape)
  if not (a_ndims >= 2 and a_shape[-1] == a_shape[-2]
    and b_shape[-2:] == a_shape[-2:]):
    msg = ("The arguments to _cofactor_solve must have shapes "
           "a=[..., m, m] and b=[..., m, m]; got a={} and b={}")
    raise ValueError(msg.format(a_shape, b_shape))
  if a_shape[-1] == 1:
    return a[0, 0], b
  # lu contains u in the upper triangular matrix and l in the strict lower
  # triangular matrix.
  # The diagonal of l is set to ones without loss of generality.
  lu, pivots = lax_linalg.lu(a)
  dtype = lax.dtype(a)
  batch_dims = lax.broadcast_shapes(lu.shape[:-2], b.shape[:-2])
  x = np.broadcast_to(b, batch_dims + b.shape[-2:])
  lu = np.broadcast_to(lu, batch_dims + lu.shape[-2:])
  # Compute (partial) determinant, ignoring last diagonal of LU
  diag = np.diagonal(lu, axis1=-2, axis2=-1)
  parity = np.count_nonzero(pivots != np.arange(a_shape[-1]), axis=-1)
  sign = np.array(-2 * (parity % 2) + 1, dtype=dtype)
  # partial_det[:, -1] contains the full determinant and
  # partial_det[:, -2] contains U_{nn} / det{U}.
  # TODO(pfau): make sure this is as numerically stable as using slogdet
  partial_det = np.cumprod(diag, axis=-1) * sign[..., None]
  lu = ops.index_update(lu, ops.index[..., -1, -1], 1.0 / partial_det[..., -2])
  permutation = lax_linalg.lu_pivots_to_permutation(pivots, a_shape[-1])
  permutation = np.broadcast_to(permutation, batch_dims + (a_shape[-1],))
  iotas = np.ix_(*(lax.iota(np.int32, b) for b in batch_dims + (1,)))
  x = x[iotas[:-1] + (permutation, slice(None))]
  x = lax_linalg.triangular_solve(lu, x, left_side=True, lower=True,
                                  unit_diagonal=True)
  x = np.concatenate((x[..., :-1, :] * partial_det[..., -1, None, None],
                      x[..., -1:, :]), axis=-2)
  x = lax_linalg.triangular_solve(lu, x, left_side=True, lower=False)
  return partial_det[..., -1], np.where(np.isnan(x), np.zeros_like(x), x)


# TODO(pfau): get it so you can do both VJP and JVP
@custom_vjp
@custom_jvp
@_wraps(onp.linalg.det)
def det(a):
  sign, logdet = slogdet(a)
  return sign * np.exp(logdet)
def _det_jvp(primals, tangents):
  x, = primals
  g, = tangents
  y, z = _cofactor_solve(x, g)
  return y, np.trace(z, axis1=-1, axis2=-2)
def _det_vjp_fwd(a):
  m = np.shape(a)[-1]
  ndim = len(np.shape(a))
  eye = np.broadcast_to(np.eye(m), (ndim-2)*[1] + [m, m])
  det, cof = _cofactor_solve(a, eye)
  return det, (cof,)
def _det_vjp_rev(res, g):
  cof, = res
  gs = np.expand_dims(np.expand_dims(g, -1), -1)
  gs = np.broadcast_to(gs, cof.shape)
  return (gs * _T(cof),)
det.defjvp(_det_jvp)
det.defvjp(_det_vjp_fwd, _det_vjp_rev)


@_wraps(onp.linalg.eig)
def eig(a):
  a = _promote_arg_dtypes(np.asarray(a))
  w, vl, vr = lax_linalg.eig(a)
  return w, vr


@_wraps(onp.linalg.eigvals)
def eigvals(a):
  w, _ = eig(a)
  return w


@_wraps(onp.linalg.eigh)
def eigh(a, UPLO=None, symmetrize_input=True):
  if UPLO is None or UPLO == "L":
    lower = True
  elif UPLO == "U":
    lower = False
  else:
    msg = "UPLO must be one of None, 'L', or 'U', got {}".format(UPLO)
    raise ValueError(msg)

  a = _promote_arg_dtypes(np.asarray(a))
  v, w = lax_linalg.eigh(a, lower=lower, symmetrize_input=symmetrize_input)
  return w, v


@_wraps(onp.linalg.eigvalsh)
def eigvalsh(a, UPLO='L'):
  w, _ = eigh(a, UPLO)
  return w


@_wraps(onp.linalg.pinv, lax_description=textwrap.dedent("""\
    It differs only in default value of `rcond`. In `numpy.linalg.pinv`, the
    default `rcond` is `1e-15`. Here the default is
    `10. * max(num_rows, num_cols) * np.finfo(dtype).eps`.
    """))
def pinv(a, rcond=None):
  # ported from https://github.com/numpy/numpy/blob/v1.17.0/numpy/linalg/linalg.py#L1890-L1979
  a = np.conj(a)
  # copied from https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/linalg.py#L442
  if rcond is None:
      max_rows_cols = max(a.shape[-2:])
      rcond = 10. * max_rows_cols * np.finfo(a.dtype).eps
  rcond = np.asarray(rcond)
  u, s, v = svd(a, full_matrices=False)
  # Singular values less than or equal to ``rcond * largest_singular_value``
  # are set to zero.
  cutoff = rcond[..., np.newaxis] * np.amax(s, axis=-1, keepdims=True)
  large = s > cutoff
  s = np.divide(1, s)
  s = np.where(large, s, 0)
  vT = np.swapaxes(v, -1, -2)
  uT = np.swapaxes(u, -1, -2)
  res = np.matmul(vT, np.multiply(s[..., np.newaxis], uT))
  return lax.convert_element_type(res, a.dtype)


@_wraps(onp.linalg.inv)
def inv(a):
  if np.ndim(a) < 2 or a.shape[-1] != a.shape[-2]:
    raise ValueError("Argument to inv must have shape [..., n, n], got {}."
      .format(np.shape(a)))
  return solve(
    a, lax.broadcast(np.eye(a.shape[-1], dtype=lax.dtype(a)), a.shape[:-2]))


@partial(jit, static_argnums=(1, 2, 3))
def _norm(x, ord, axis: Union[None, Tuple[int, ...], int], keepdims):
  x = _promote_arg_dtypes(np.asarray(x))
  x_shape = np.shape(x)
  ndim = len(x_shape)

  if axis is None:
    # NumPy has an undocumented behavior that admits arbitrary rank inputs if
    # `ord` is None: https://github.com/numpy/numpy/issues/14215
    if ord is None:
      return np.sqrt(np.sum(np.real(x * np.conj(x)), keepdims=keepdims))
    axis = tuple(range(ndim))
  elif isinstance(axis, tuple):
    axis = tuple(np._canonicalize_axis(x, ndim) for x in axis)
  else:
    axis = (np._canonicalize_axis(axis, ndim),)

  num_axes = len(axis)
  if num_axes == 1:
    if ord is None or ord == 2:
      return np.sqrt(np.sum(np.real(x * np.conj(x)), axis=axis,
                            keepdims=keepdims))
    elif ord == np.inf:
      return np.amax(np.abs(x), axis=axis, keepdims=keepdims)
    elif ord == -np.inf:
      return np.amin(np.abs(x), axis=axis, keepdims=keepdims)
    elif ord == 0:
      return np.sum(x != 0, dtype=np.finfo(lax.dtype(x)).dtype,
                    axis=axis, keepdims=keepdims)
    elif ord == 1:
      # Numpy has a special case for ord == 1 as an optimization. We don't
      # really need the optimization (XLA could do it for us), but the Numpy
      # code has slightly different type promotion semantics, so we need a
      # special case too.
      return np.sum(np.abs(x), axis=axis, keepdims=keepdims)
    else:
      abs_x = np.abs(x)
      ord = lax._const(abs_x, ord)
      out = np.sum(abs_x ** ord, axis=axis, keepdims=keepdims)
      return np.power(out, 1. / ord)

  elif num_axes == 2:
    row_axis, col_axis = cast(Tuple[int, ...], axis)
    if ord is None or ord in ('f', 'fro'):
      return np.sqrt(np.sum(np.real(x * np.conj(x)), axis=axis,
                            keepdims=keepdims))
    elif ord == 1:
      if not keepdims and col_axis > row_axis:
        col_axis -= 1
      return np.amax(np.sum(np.abs(x), axis=row_axis, keepdims=keepdims),
                     axis=col_axis, keepdims=keepdims)
    elif ord == -1:
      if not keepdims and col_axis > row_axis:
        col_axis -= 1
      return np.amin(np.sum(np.abs(x), axis=row_axis, keepdims=keepdims),
                     axis=col_axis, keepdims=keepdims)
    elif ord == np.inf:
      if not keepdims and row_axis > col_axis:
        row_axis -= 1
      return np.amax(np.sum(np.abs(x), axis=col_axis, keepdims=keepdims),
                     axis=row_axis, keepdims=keepdims)
    elif ord == -np.inf:
      if not keepdims and row_axis > col_axis:
        row_axis -= 1
      return np.amin(np.sum(np.abs(x), axis=col_axis, keepdims=keepdims),
                     axis=row_axis, keepdims=keepdims)
    elif ord in ('nuc', 2, -2):
      x = np.moveaxis(x, axis, (-2, -1))
      if ord == 2:
        reducer = np.amax
      elif ord == -2:
        reducer = np.amin
      else:
        reducer = np.sum
      y = reducer(svd(x, compute_uv=False), axis=-1)
      if keepdims:
        result_shape = list(x_shape)
        result_shape[axis[0]] = 1
        result_shape[axis[1]] = 1
        y = np.reshape(y, result_shape)
      return y
    else:
      raise ValueError("Invalid order '{}' for matrix norm.".format(ord))
  else:
    raise ValueError(
        "Invalid axis values ({}) for np.linalg.norm.".format(axis))

@_wraps(onp.linalg.norm)
def norm(x, ord=None, axis=None, keepdims=False):
  return _norm(x, ord, axis, keepdims)


@_wraps(onp.linalg.qr)
def qr(a, mode="reduced"):
  if mode in ("reduced", "r", "full"):
    full_matrices = False
  elif mode == "complete":
    full_matrices = True
  else:
    raise ValueError("Unsupported QR decomposition mode '{}'".format(mode))
  a = _promote_arg_dtypes(np.asarray(a))
  q, r = lax_linalg.qr(a, full_matrices)
  if mode == "r":
    return r
  return q, r


def _check_solve_shapes(a, b):
  if not (a.ndim >= 2 and a.shape[-1] == a.shape[-2] and b.ndim >= 1):
    msg = ("The arguments to solve must have shapes a=[..., m, m] and "
           "b=[..., m, k] or b=[..., m]; got a={} and b={}")
    raise ValueError(msg.format(a.shape, b.shape))


@partial(vectorize, signature='(n,m),(m)->(n)')
def _matvec_multiply(a, b):
  return np.dot(a, b, precision=lax.Precision.HIGHEST)


@_wraps(onp.linalg.solve)
@jit
def solve(a, b):
  a, b = _promote_arg_dtypes(np.asarray(a), np.asarray(b))
  _check_solve_shapes(a, b)

  # With custom_linear_solve, we can reuse the same factorization when
  # computing sensitivities. This is considerably faster.
  lu, pivots = lax.stop_gradient(lax_linalg.lu)(a)
  custom_solve = partial(
      lax.custom_linear_solve,
      lambda x: _matvec_multiply(a, x),
      solve=lambda _, x: lax_linalg.lu_solve(lu, pivots, x, trans=0),
      transpose_solve=lambda _, x: lax_linalg.lu_solve(lu, pivots, x, trans=1))
  if a.ndim == b.ndim + 1:
    # b.shape == [..., m]
    return custom_solve(b)
  else:
    # b.shape == [..., m, k]
    return vmap(custom_solve, b.ndim - 1, max(a.ndim, b.ndim) - 1)(b)


for func in get_module_functions(onp.linalg):
  if func.__name__ not in globals():
    globals()[func.__name__] = _not_implemented(func)
