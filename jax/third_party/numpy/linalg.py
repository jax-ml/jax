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


def __pnorm2Calc(x):
  s = la.svd(x, compute_uv=False)
  return s[..., 0] / s[..., -1]


def __pnormNeg2Calc(x):
  s = la.svd(x, compute_uv=False)
  return s[..., -1] / s[..., 0]


def __pnormDefaultCalc(x, p):
  # Call inv(x) ignoring errors. The result array will
  # contain nans in the entries where inversion failed.
  _assertRankAtLeast2(x)
  _assertNdSquareness(x)
  invx = la.inv(x)
  r = la.norm(x, ord=p, axis=(-2, -1)) * la.norm(invx, ord=p, axis=(-2, -1))
  return r


def _nanMaskUpdate(args):
  nan_mask, x, r = args
  nan_mask = np.logical_and(~nan_mask, ~np.isnan(x).any(axis=(-2, -1)))
  r = np.where(nan_mask, np.inf, r)
  return r


@_wraps(onp.linalg.cond)
def cond(x, p=None):
  _assertNoEmpty2d(x)
  if p in (None, 2):
    r = __pnorm2Calc(x)
  if p == -2:
    r = __pnormNeg2Calc(x)
  if p not in (None, 2, -2):
    r = __pnormDefaultCalc(x, p)

  # Convert nans to infs unless the original array had nan entries
  r = np.asarray(r)
  orig_nan_check = np.full_like(r, ~np.isnan(r).any())
  nan_mask = np.logical_and(np.isnan(r), ~np.isnan(x).any(axis=(-2, -1)))
  r = np.where(orig_nan_check, np.where(nan_mask, np.inf, r), r)
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
