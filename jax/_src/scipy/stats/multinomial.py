import scipy.stats as osp_stats
from jax import lax
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy import lax_numpy as jnp
from jax._src.numpy.lax_numpy import _promote_args_inexact
from jax._src.numpy.util import _wraps
from jax._src.scipy.special import gammaln, xlogy

def _check_args(x, n, p):
    if jnp.sum(x) != n:
        raise ValueError("The sum of x should equal n.")
    if jnp.sum(p) != 1:
        raise ValueError("The values of p must sum to 1.")

@_wraps(osp_stats.multinomial.logpmf, update_doc=False)
def logpmf(x, n, p):
    """JAX implementation of scipy.stats.multinomial.logpmf."""
    x, n, p = _promote_args_inexact("multinomial.logpmf", x, n, p)
    _check_args(x, n, p)
    one = _lax_const(x, 1)
    nplusone, xplusone = jnp.add(n, 1), jnp.add(x, 1)
    return gammaln(nplusone) + jnp.sum(xlogy(x, p) - gammaln(xplusone), axis=-1)

@_wraps(osp_stats.multinomial.pmf, update_doc=False)
def pmf(x, n, p):
    """JAX implementation of scipy.stats.multinomial.pmf."""
    x, n, p = _promote_args_inexact("multinomial.pmf", x, n, p)
    return jnp.exp(logpmf(x, n, p))