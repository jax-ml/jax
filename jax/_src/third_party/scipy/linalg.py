from __future__ import annotations

from typing import Callable

import scipy.linalg

from jax import jit, lax
import jax.numpy as jnp
from jax._src.numpy.linalg import norm
from jax._src.numpy.util import _wraps
from jax._src.scipy.linalg import rsf2csf, schur
from jax._src.typing import ArrayLike, Array


@jit
def _algorithm_11_1_1(F: Array, T: Array) -> tuple[Array, Array]:
  # Algorithm 11.1.1 from Golub and Van Loan "Matrix Computations"
  N = T.shape[0]
  minden = jnp.abs(T[0, 0])

  def _outer_loop(p, F_minden):
    _, F, minden = lax.fori_loop(1, N-p+1, _inner_loop, (p, *F_minden))
    return F, minden

  def _inner_loop(i, p_F_minden):
    p, F, minden = p_F_minden
    j = i+p
    s = T[i-1, j-1] * (F[j-1, j-1] - F[i-1, i-1])
    T_row, T_col = T[i-1], T[:, j-1]
    F_row, F_col = F[i-1], F[:, j-1]
    ind = (jnp.arange(N) >= i) & (jnp.arange(N) < j-1)
    val = (jnp.where(ind, T_row, 0) @ jnp.where(ind, F_col, 0) -
            jnp.where(ind, F_row, 0) @ jnp.where(ind, T_col, 0))
    s = s + val
    den = T[j-1, j-1] - T[i-1, i-1]
    s = jnp.where(den != 0, s / den, s)
    F = F.at[i-1, j-1].set(s)
    minden = jnp.minimum(minden, jnp.abs(den))
    return p, F, minden

  return lax.fori_loop(1, N, _outer_loop, (F, minden))

_FUNM_LAX_DESCRIPTION = """\
The array returned by :py:func:`jax.scipy.linalg.funm` may differ in dtype
from the array returned by py:func:`scipy.linalg.funm`. Specifically, in cases
where all imaginary parts of the array values are close to zero, the SciPy
function may return a real-valued array, whereas the JAX implementation will
return a complex-valued array.

Additionally, unlike the SciPy implementation, when ``disp=True`` no warning
will be printed if the error in the array output is estimated to be large.
"""

@_wraps(scipy.linalg.funm, lax_description=_FUNM_LAX_DESCRIPTION)
def funm(A: ArrayLike, func: Callable[[Array], Array],
         disp: bool = True) -> Array | tuple[Array, Array]:
  A_arr = jnp.asarray(A)
  if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
    raise ValueError('expected square array_like input')

  T, Z = schur(A_arr)
  T, Z = rsf2csf(T, Z)

  F = jnp.diag(func(jnp.diag(T)))
  F = F.astype(T.dtype.char)

  F, minden = _algorithm_11_1_1(F, T)
  F = Z @ F @ Z.conj().T

  if disp:
    return F

  if F.dtype.char.lower() == 'e':
    tol = jnp.finfo(jnp.float16).eps
  if F.dtype.char.lower() == 'f':
    tol = jnp.finfo(jnp.float32).eps
  else:
    tol = jnp.finfo(jnp.float64).eps

  minden = jnp.where(minden == 0.0, tol, minden)
  err = jnp.where(jnp.any(jnp.isinf(F)), jnp.inf, jnp.minimum(1, jnp.maximum(
          tol, (tol / minden) * norm(jnp.triu(T, 1), 1))))

  return F, err
