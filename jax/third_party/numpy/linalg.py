import numpy as onp

from jax.numpy import lax_numpy as np
from jax.numpy import linalg as la
from jax.numpy.lax_numpy import _wraps
from jax.lax import cond as lax_cond
from functools import partial


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
  res = s[..., 0] / s[..., -1]
  return res


def __pnormNeg2Calc(x):
  s = la.svd(x, compute_uv=False)
  return s[..., -1] / s[..., 0]


def __pnormOtherCalc(args):
  x, p = args
  # Call inv(x) ignoring errors. The result array will
  # contain nans in the entries where inversion failed.
  _assertRankAtLeast2(x)
  _assertNdSquareness(x)
  norm_fn = partial(la.norm, ord=p, axis=(-2, -1), keepdims=False)
  invx = la.inv(x)
  r = norm_fn(x) * norm_fn(invx)
  return r


def _nanMaskUpdate(args):
  nan_mask, x, r = args
  nan_mask = np.logical_and(~nan_mask, ~np.isnan(x).any(axis=(-2, -1)))
  r = np.where(nan_mask, np.inf, r)
  return r


@_wraps(onp.linalg.cond)
def cond(a, p=None):
  x = np.asarray(a)  # in case we have a matrix
  _assertNoEmpty2d(x)
  r = lax_cond(p in (None, 2), x, __pnorm2Calc, x, lambda x: np.empty([]))
  r = lax_cond(p == -2, x, __pnormNeg2Calc, r, lambda r: r)
  r = lax_cond(p not in (None, 2, -2), [x, p], __pnormOtherCalc, r, lambda r: r)

  # Convert nans to infs unless the original array had nan entries
  r = np.asarray(r)
  nan_mask = ~np.isnan(r)
  r = lax_cond(nan_mask.all(), r, lambda r: r, [nan_mask, x, r], _nanMaskUpdate)
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
