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
def cond(x, p=None):
  _assertNoEmpty2d(x)
  if p in (None, 2):
    s = la.svd(x, compute_uv=False)
    return s[..., 0] / s[..., -1]
  elif p == -2:
    s = la.svd(x, compute_uv=False)
    r = s[..., -1] / s[..., 0]
  else:
    _assertRankAtLeast2(x)
    _assertNdSquareness(x)
    invx = la.inv(x)
    r = la.norm(x, ord=p, axis=(-2, -1)) * la.norm(invx, ord=p, axis=(-2, -1))

  # Convert nans to infs unless the original array had nan entries
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


@_wraps(onp.linalg.tensorsolve)
def tensorsolve(a, b, axes=None):
  a = np.asarray(a)
  b = np.asarray(b)
  an = a.ndim
  if axes is not None:
    allaxes = list(range(0, an))
    for k in axes:
      allaxes.remove(k)
      allaxes.insert(an, k)

    a = a.transpose(allaxes)
  
  Q = a.shape[-(an - b.ndim):]

  prod = 1
  for k in Q:
    prod *= k

  a = a.reshape(-1, prod)
  b = b.ravel()
  
  res = np.asarray(la.solve(a, b))
  res = res.reshape(Q)
  
  return res
