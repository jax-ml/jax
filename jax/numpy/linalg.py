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

import numpy as onp
import warnings

from jax import jit
from .. import lax
from .. import lax_linalg
from .lax_numpy import _not_implemented
from .lax_numpy import _wraps
from . import lax_numpy as np
from ..api import custom_transforms, defjvp
from ..util import get_module_functions
from ..lib import xla_bridge


_T = lambda x: np.swapaxes(x, -1, -2)


def _promote_arg_dtypes(*args):
  """Promotes `args` to a common inexact type."""
  def _to_inexact_type(type):
    return type if np.issubdtype(type, np.inexact) else np.float64
  inexact_types = [_to_inexact_type(np._dtype(arg)) for arg in args]
  dtype = xla_bridge.canonicalize_dtype(np.result_type(*inexact_types))
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


# TODO(pfau): make this work for complex types
def _jvp_slogdet(g, ans, x):
  jvp_sign = np.zeros(x.shape[:-2])
  jvp_logdet = np.trace(solve(x, g), axis1=-1, axis2=-2)
  return jvp_sign, jvp_logdet


@_wraps(onp.linalg.slogdet)
@custom_transforms
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
defjvp(slogdet, _jvp_slogdet)


@_wraps(onp.linalg.det)
def det(a):
  sign, logdet = slogdet(a)
  return sign * np.exp(logdet)


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


@_wraps(onp.linalg.pinv)
def pinv(a, rcond=1e-15):
  a = np.conj(a)
  rcond = np.asarray(rcond)
  u, s, v = svd(a, full_matrices=False)
  # Singular values less than or equal to ``rcond * largest_singular_value`` 
  # are set to zero.
  cutoff = rcond[..., np.newaxis] * np.amax(s, axis=-1, keepdims=True)
  large = s > cutoff
  s = np.divide(1, s)
  s = np.where(large, s, 0)
  res = np.matmul(np.transpose(v), np.multiply(s[..., np.newaxis], np.transpose(u)))
  return res


@_wraps(onp.linalg.inv)
def inv(a):
  if np.ndim(a) < 2 or a.shape[-1] != a.shape[-2]:
    raise ValueError("Argument to inv must have shape [..., n, n], got {}."
      .format(np.shape(a)))
  return solve(
    a, lax.broadcast(np.eye(a.shape[-1], dtype=lax.dtype(a)), a.shape[:-2]))


@partial(jit, static_argnums=(1, 2, 3))
def _norm(x, ord, axis, keepdims):
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
      return np.power(np.sum(np.abs(x) ** ord, axis=axis, keepdims=keepdims),
                      1. / ord)

  elif num_axes == 2:
    row_axis, col_axis = axis
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


@_wraps(onp.linalg.solve)
@jit
def solve(a, b):
  a, b = _promote_arg_dtypes(np.asarray(a), np.asarray(b))
  a_shape = np.shape(a)
  b_shape = np.shape(b)
  a_ndims = len(a_shape)
  b_ndims = len(b_shape)
  if not (a_ndims >= 2 and a_shape[-1] == a_shape[-2] and b_ndims >= 1):
    msg = ("The arguments to solve must have shapes a=[..., m, m] and "
           "b=[..., m, k] or b=[..., m]; got a={} and b={}")
    raise ValueError(msg.format(a_shape, b_shape))
  lu, pivots = lax_linalg.lu(a)
  dtype = lax.dtype(a)

  m = a_shape[-1]

  # Numpy treats the RHS as a (batched) vector if the number of dimensions
  # differ by 1. Otherwise, broadcasting rules apply.
  x = b[..., None] if a_ndims == b_ndims + 1 else b

  batch_dims = lax.broadcast_shapes(lu.shape[:-2], x.shape[:-2])
  x = np.broadcast_to(x, batch_dims + x.shape[-2:])
  lu = np.broadcast_to(lu, batch_dims + lu.shape[-2:])

  permutation = lax_linalg.lu_pivots_to_permutation(pivots, m)
  permutation = np.broadcast_to(permutation, batch_dims + (m,))
  iotas = np.ix_(*(lax.iota(np.int32, b) for b in batch_dims + (1,)))
  x = x[iotas[:-1] + (permutation, slice(None))]

  x = lax_linalg.triangular_solve(lu, x, left_side=True, lower=True,
                                  unit_diagonal=True)
  x = lax_linalg.triangular_solve(lu, x, left_side=True, lower=False)

  return x[..., 0] if a_ndims == b_ndims + 1 else x


for func in get_module_functions(onp.linalg):
  if func.__name__ not in globals():
    globals()[func.__name__] = _not_implemented(func)
