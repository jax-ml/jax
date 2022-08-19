import scipy.stats as osp_stats
from jax import lax
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy import lax_numpy as jnp
from jax._src.numpy.lax_numpy import _promote_args_inexact
from jax._src.numpy.util import _wraps
from jax._src.scipy.special import gammaln, xlogy

@_wraps(osp_stats.multinomial.logpmf, update_doc=False)
def logpmf(x, n, p):
  """JAX implementation of scipy.stats.multinomial.logpmf."""
  x, p = _promote_args_inexact("multinomial.logpmf", x, p)
  one = _lax_const(x, 1)
  nplusone, xplusone = jnp.add(n, one), jnp.add(x, one)
  logprobs = gammaln(nplusone) + jnp.sum(xlogy(x, p) - gammaln(xplusone), axis=-1)
  return jnp.where(jnp.equal(jnp.sum(x), n), logprobs, -jnp.inf)

@_wraps(osp_stats.multinomial.pmf, update_doc=False)
def pmf(x, n, p):
  """JAX implementation of scipy.stats.multinomial.pmf."""
  x, p = _promote_args_inexact("multinomial.pmf", x, p)
  return jnp.exp(logpmf(x, n, p))
