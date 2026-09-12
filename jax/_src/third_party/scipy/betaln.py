from jax._src import lax
from jax._src import numpy as jnp
from jax._src.typing import Array, ArrayLike
from jax._src.numpy.util import promote_args_inexact


def algdiv(a: ArrayLike, b: ArrayLike) -> Array:
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


def betaln(a: ArrayLike, b: ArrayLike) -> Array:
    """Compute the log of the beta function.

    Derived from scipy's implementation of `betaln`_.

    This implementation does not handle all branches of the scipy implementation, but is still much more accurate
    than just doing lgamma(a) + lgamma(b) - lgamma(a + b) when inputs are large (> 1M or so).

    .. _betaln:
        https://github.com/scipy/scipy/blob/ef2dee592ba8fb900ff2308b9d1c79e4d6a0ad8b/scipy/special/cdflib/betaln.f
    """
    a, b = promote_args_inexact("betaln", a, b)
    a, b = jnp.minimum(a, b), jnp.maximum(a, b)
    small_b = lax.lgamma(a) + (lax.lgamma(b) - lax.lgamma(a + b))
    large_b = lax.lgamma(a) + algdiv(a, b)
    result = jnp.where(b < 8, small_b, large_b)
    # When both ``a`` and ``a + b`` are non-positive integers, ``gamma`` has a
    # pole at each, so ``lgamma(a)`` and ``lgamma(a + b)`` are both ``+inf`` and
    # the expression above evaluates to ``nan``. The beta function is finite
    # there, however, because the poles cancel: taking the limit ``a -> -n`` via
    # the residues of ``gamma`` gives
    #   B(a, b) = (-1)**b * gamma(b) * gamma(1 - a - b) / gamma(1 - a),
    # so ``betaln`` is ``lgamma(b) + lgamma(1 - a - b) - lgamma(1 - a)``. This
    # also yields the correct ``+inf`` when ``b`` is a non-positive integer.
    both_poles = _is_nonpos_int(a) & _is_nonpos_int(a + b)
    # Evaluate the reflection formula on safe arguments where it is unused, so
    # that the final ``where`` does not introduce nan gradients for ordinary
    # inputs (``lgamma`` has infinite derivatives at its own poles).
    safe_a = jnp.where(both_poles, a, -1.0)
    safe_b = jnp.where(both_poles, b, 1.0)
    reflected = (lax.lgamma(safe_b) + lax.lgamma(1 - safe_a - safe_b)
                 - lax.lgamma(1 - safe_a))
    return jnp.where(both_poles, reflected, result)


def _is_nonpos_int(x: Array) -> Array:
    return (x <= 0) & (x == jnp.floor(x)) & jnp.isfinite(x)
