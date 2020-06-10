import numpy as np

from jax.numpy import lax_numpy as jnp
from jax.numpy._util import _wraps

@_wraps(np.setxor1d)
def setxor1d(ar1, ar2, assume_unique=False):
  if not assume_unique:
    ar1 = jnp.unique(ar1)
    ar2 = jnp.unique(ar2)
  
  aux = jnp.concatenate((ar1, ar2))
  if aux.size == 0:
    return aux
  
  aux = jnp.sort(aux)
  ar_true = jnp.array([True])
  flag = jnp.concatenate((ar_true, aux[1:] != aux[:-1], ar_true))
  return aux[flag[1:] & flag[:-1]]
