from jax import custom_jvp
from jax._src import lax
from jax._src import numpy as jnp
from jax._src.typing import Array, ArrayLike
from jax._src.numpy.util import promote_args_inexact

# Note: for mysterious reasons, annotating this leads to very slow mypy runs.
# def algdiv(a: ArrayLike, b: ArrayLike) -> Array:

def algdiv(a, b):
    """
    Compute ``log(gamma(a))/log(gamma(a + b))`` when ``b >= 8``.

    Derived from scipy's implementation of `algdiv`_.

    This differs from the scipy implementation in that it assumes a <= b
    because recomputing ``a, b = jnp.minimum(a, b), jnp.maximum(a, b)`` might
    be expensive and this is only called by ``betaln``.

    .. _algdiv:
        https://github.com/scipy/scipy/blob/c89dfc2b90d993f2a8174e57e0cbc8fbe6f3ee19/scipy/special/cdflib/algdiv.f
    """
    c0 = 0.833333333333333e-01
    c1 = -0.277777777760991e-02
    c2 = 0.793650666825390e-03
    c3 = -0.595202931351870e-03
    c4 = 0.837308034031215e-03
    c5 = -0.165322962780713e-02
    h = a / b
    c = h / (1 + h)
    x = h / (1 + h)
    d = b + (a - 0.5)
    # Set sN = (1 - x**n)/(1 - x)
    x2 = x * x
    s3 = 1.0 + (x + x2)
    s5 = 1.0 + (x + x2 * s3)
    s7 = 1.0 + (x + x2 * s5)
    s9 = 1.0 + (x + x2 * s7)
    s11 = 1.0 + (x + x2 * s9)
    # Set w = del(b) - del(a + b)
    # where del(x) is defined by ln(gamma(x)) = (x - 0.5)*ln(x) - x + 0.5*ln(2*pi) + del(x)
    t = (1.0 / b) ** 2
    w = ((((c5 * s11 * t + c4 * s9) * t + c3 * s7) * t + c2 * s5) * t + c1 * s3) * t + c0
    w = w * (c / b)
    # Combine the results
    u = d * lax.log1p(a / b)
    v = a * (lax.log(b) - 1.0)
    return jnp.where(u <= v, (w - v) - u, (w - u) - v)


def _betaln_impl(a, b):
    """Compute betaln with numerical stability for large inputs.
    
    Uses the standard lgamma formula for small inputs and a more stable
    algorithm (algdiv) for large inputs where catastrophic cancellation
    would otherwise occur.
    """
    # Swap so that a <= b for the algdiv algorithm
    a_orig, b_orig = a, b
    a = jnp.minimum(a_orig, b_orig)
    b = jnp.maximum(a_orig, b_orig)
    
    small_b = lax.lgamma(a) + (lax.lgamma(b) - lax.lgamma(a + b))
    large_b = lax.lgamma(a) + algdiv(a, b)
    return jnp.where(b < 8, small_b, large_b)


@custom_jvp
def betaln(a: ArrayLike, b: ArrayLike) -> Array:
    """Compute the log of the beta function.

    Uses a numerically stable algorithm for both small and large inputs.
    The derivative is computed analytically using the digamma function,
    which ensures correct first and second-order derivatives even at
    boundary cases where a == b (fixes #34353).

    .. _betaln:
        https://github.com/scipy/scipy/blob/ef2dee592ba8fb900ff2308b9d1c79e4d6a0ad8b/scipy/special/cdflib/betaln.f
    """
    a, b = promote_args_inexact("betaln", a, b)
    return _betaln_impl(a, b)


@betaln.defjvp
def betaln_jvp(primals, tangents):
    """Custom JVP for betaln with mathematically correct derivatives.
    
    d/da betaln(a, b) = digamma(a) - digamma(a + b)
    d/db betaln(a, b) = digamma(b) - digamma(a + b)
    
    This avoids the derivative discontinuity from jnp.minimum/jnp.maximum
    that caused incorrect Hessians in the original implementation.
    """
    a, b = primals
    a_dot, b_dot = tangents
    
    # Forward pass uses the numerically stable implementation
    result = betaln(a, b)
    
    # Backward pass uses the mathematically correct formula
    # d/da betaln(a, b) = digamma(a) - digamma(a + b)
    # d/db betaln(a, b) = digamma(b) - digamma(a + b)
    digamma_a = lax.digamma(a)
    digamma_b = lax.digamma(b)
    digamma_ab = lax.digamma(a + b)
    
    da = digamma_a - digamma_ab
    db = digamma_b - digamma_ab
    
    tangent_out = a_dot * da + b_dot * db
    
    return result, tangent_out
