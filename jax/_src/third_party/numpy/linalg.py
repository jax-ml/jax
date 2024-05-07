import numpy as np

import jax.numpy as jnp
import jax.numpy.linalg as la
from jax._src.numpy.util import check_arraylike, implements


def _isEmpty2d(arr):
  # check size first for efficiency
  return arr.size == 0 and np.prod(arr.shape[-2:]) == 0


def _assertNoEmpty2d(*arrays):
  for a in arrays:
    if _isEmpty2d(a):
      raise np.linalg.LinAlgError("Arrays cannot be empty")


def _assertRankAtLeast2(*arrays):
  for a in arrays:
    if a.ndim < 2:
      raise np.linalg.LinAlgError(
          '%d-dimensional array given. Array must be '
          'at least two-dimensional' % a.ndim)


def _assertNdSquareness(*arrays):
  for a in arrays:
    m, n = a.shape[-2:]
    if m != n:
      raise np.linalg.LinAlgError(
          'Last 2 dimensions of the array must be square')


@implements(np.linalg.cond)
def cond(x, p=None):
  check_arraylike('jnp.linalg.cond', x)
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
  orig_nan_check = jnp.full_like(r, ~jnp.isnan(r).any())
  nan_mask = jnp.logical_and(jnp.isnan(r), ~jnp.isnan(x).any(axis=(-2, -1)))
  r = jnp.where(orig_nan_check, jnp.where(nan_mask, jnp.inf, r), r)
  return r
