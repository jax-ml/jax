
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


def _assert2d(*arrays):
    for a in arrays:
        if a.ndim != 2:
            raise ValueError(f'{a.ndim}-dimensional array given. '
                             'Array must be two-dimensional')


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


@_wraps(onp.linalg.multi_dot)
def multi_dot(arrays, *, precision=None):
    n = len(arrays)
    # optimization only makes sense for len(arrays) > 2
    if n < 2:
        raise ValueError("Expecting at least two arrays.")
    elif n == 2:
        return np.dot(arrays[0], arrays[1], precision=precision)

    arrays = [np.asarray(a) for a in arrays]

    # save original ndim to reshape the result array into the proper form later
    ndim_first, ndim_last = arrays[0].ndim, arrays[-1].ndim
    # Explicitly convert vectors to 2D arrays to keep the logic of the internal
    # _multi_dot_* functions as simple as possible.
    if arrays[0].ndim == 1:
        arrays[0] = np.atleast_2d(arrays[0])
    if arrays[-1].ndim == 1:
        arrays[-1] = np.atleast_2d(arrays[-1]).T
    _assert2d(*arrays)

    # _multi_dot_three is much faster than _multi_dot_matrix_chain_order
    if n == 3:
        result = _multi_dot_three(*arrays, precision)
    else:
        order = _multi_dot_matrix_chain_order(arrays)
        result = _multi_dot(arrays, order, 0, n - 1, precision)

    # return proper shape
    if ndim_first == 1 and ndim_last == 1:
        return result[0, 0]  # scalar
    elif ndim_first == 1 or ndim_last == 1:
        return result.ravel()  # 1-D
    else:
        return result


def _multi_dot_three(A, B, C, precision):
    """
    Find the best order for three arrays and do the multiplication.
    For three arguments `_multi_dot_three` is approximately 15 times faster
    than `_multi_dot_matrix_chain_order`
    """
    a0, a1b0 = A.shape
    b1c0, c1 = C.shape
    # cost1 = cost((AB)C) = a0*a1b0*b1c0 + a0*b1c0*c1
    cost1 = a0 * b1c0 * (a1b0 + c1)
    # cost2 = cost(A(BC)) = a1b0*b1c0*c1 + a0*a1b0*c1
    cost2 = a1b0 * c1 * (a0 + b1c0)

    if cost1 < cost2:
        return np.dot(np.dot(A, B, precision=precision), C, precision=precision)
    else:
        return np.dot(A, np.dot(B, C, precision=precision), precision=precision)


def _multi_dot_matrix_chain_order(arrays, return_costs=False):
    """
    Return a np.array that encodes the optimal order of mutiplications.
    The optimal order array is then used by `_multi_dot()` to do the
    multiplication.
    Also return the cost matrix if `return_costs` is `True`
    The implementation CLOSELY follows Cormen, "Introduction to Algorithms",
    Chapter 15.2, p. 370-378.  Note that Cormen uses 1-based indices.
        cost[i, j] = min([
            cost[prefix] + cost[suffix] + cost_mult(prefix, suffix)
            for k in range(i, j)])
    """
    n = len(arrays)
    # p stores the dimensions of the matrices
    # Example for p: A_{10x100}, B_{100x5}, C_{5x50} --> p = [10, 100, 5, 50]
    p = [a.shape[0] for a in arrays] + [arrays[-1].shape[1]]
    # m is a matrix of costs of the subproblems
    # m[i,j]: min number of scalar multiplications needed to compute A_{i..j}
    m = onp.zeros((n, n), dtype=onp.double)
    # s is the actual ordering
    # s[i, j] is the value of k at which we split the product A_i..A_j
    s = onp.empty((n, n), dtype=onp.intp)

    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            m[i, j] = np.inf
            for k in range(i, j):
                q = m[i, k] + m[k+1, j] + p[i]*p[k+1]*p[j+1]
                if q < m[i, j]:
                    m[i, j] = q
                    s[i, j] = k  # Note that Cormen uses 1-based index

    return (s, m) if return_costs else s


def _multi_dot(arrays, order, i, j, precision):
    """Actually do the multiplication with the given order."""
    if i == j:
        return arrays[i]
    else:
        return np.dot(_multi_dot(arrays, order, i, order[i, j], precision),
                      _multi_dot(arrays, order, order[i, j] + 1, j, precision),
                      precision=precision)