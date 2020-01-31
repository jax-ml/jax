from __future__ import absolute_import, division, print_function

import numpy as onp

from jax.numpy import lax_numpy as np
from jax.numpy import linalg as la
from jax.numpy.lax_numpy import _wraps


def _isEmpty2d(arr):
  # check size first for efficiency
  return arr.size == 0 and np.product(arr.shape[-2:]) == 0


def _assertNoEmpty2d(*arrays):
  for a in arrays:
    if _isEmpty2d(a):
      raise onp.linalg.LinAlgError("Arrays cannot be empty")


def _assertRankAtLeast2(*arrays):
  for a in arrays:
    if a.ndim < 2:
      raise onp.linalg.LinAlgError(
          '%d-dimensional array given. Array must be '
          'at least two-dimensional' % a.ndim)


def _assertNdSquareness(*arrays):
  for a in arrays:
    m, n = a.shape[-2:]
    if m != n:
      raise onp.linalg.LinAlgError(
          'Last 2 dimensions of the array must be square')


@_wraps(onp.linalg.cond)
def cond(a, p=None):
  x = np.asarray(a)  # in case we have a matrix
  _assertNoEmpty2d(x)
  if p is None or p == 2 or p == -2:
    s = la.svd(x, compute_uv=False)
    if p == -2:
      r = s[..., -1] / s[..., 0]
    else:
      r = s[..., 0] / s[..., -1]
  else:
    # Call inv(x) ignoring errors. The result array will
    # contain nans in the entries where inversion failed.
    _assertRankAtLeast2(x)
    _assertNdSquareness(x)
    invx = la.inv(x)
    r = la.norm(x, p, axis=(-2, -1)) * la.norm(invx, p, axis=(-2, -1))

  # Convert nans to infs unless the original array had nan entries
  r = np.asarray(r)
  nan_mask = np.isnan(r)
  if nan_mask.any():
    nan_mask &= ~np.isnan(x).any(axis=(-2, -1))
    if r.ndim > 0:
      r[nan_mask] = np.inf
    elif nan_mask:
      r[()] = np.inf

  # Convention is to return scalars instead of 0d arrays
  if r.ndim == 0:
    r = r[()]

  return r

@_wraps(onp.linalg.tensorinv)
def tensorinv(a, ind=2):
  a = np.asarray(a)
  oldshape = a.shape
  prod = 1
  if ind > 0:
    invshape = oldshape[ind:] + oldshape[:ind]
    for k in oldshape[ind:]:
      prod *= k
  else:
    raise ValueError("Invalid ind argument.")
  a = a.reshape(prod, -1)
  ia = la.inv(a)
  return ia.reshape(*invshape)
