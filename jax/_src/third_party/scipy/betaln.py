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
    x = 1.0 / (1.0 + h)
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
    b_is_inf = jnp.isinf(b)
    d_safe = jnp.where(b_is_inf, 0.0, d)
    a_safe = jnp.where(b_is_inf & (a == 0.0), 1.0, a)
    u = jnp.where(b_is_inf, a, d_safe * lax.log1p(a / b))
    v = jnp.where(b_is_inf & (a == 0.0), 0.0, a_safe * (lax.log(b) - 1.0))
    return jnp.where(u <= v, (w - v) - u, (w - u) - v)


def _is_integer(x: Array) -> Array:
    # Mask non-finite inputs so lax.rem(inf, 1.0) does not produce NaN.
    x_safe = jnp.where(jnp.isfinite(x), x, 0.5)
    return lax.rem(x_safe, lax.full_like(x_safe, 1.0, shape=())) == 0


def betaln(a: ArrayLike, b: ArrayLike) -> Array:
    """Compute the log of the beta function.

    Derived from scipy's implementation of `betaln`_.

    This implementation does not handle all branches of the scipy implementation, but is still much more accurate
    than just doing lgamma(a) + lgamma(b) - lgamma(a + b) when inputs are large (> 1M or so).

    .. _betaln:
        https://github.com/scipy/scipy/blob/ef2dee592ba8fb900ff2308b9d1c79e4d6a0ad8b/scipy/special/cdflib/betaln.f
    """
    a, b = promote_args_inexact("betaln", a, b)
    # Use jnp.where instead of jnp.minimum/maximum to ensure correct gradients
    # at the boundary where a == b. With jnp.where, the gradient flows through
    # only one branch, avoiding the 0.5/0.5 gradient split that causes issues
    # with higher-order derivatives.
    new_a = jnp.where(a <= b, a, b)
    new_b = jnp.where(a <= b, b, a)
    # Apply reflection B(a, b) = (-1)**b B(1 - a - b, b) when new_a is a
    # negative integer, new_b is a positive integer, and new_a + new_b <= 0.
    is_neg_int_a = (new_a < 0) & _is_integer(new_a)
    is_pos_int_b = (new_b > 0) & _is_integer(new_b)
    reflect = is_neg_int_a & is_pos_int_b & (new_a <= -new_b)
    reflected_a = (
        1.0 - jnp.where(reflect, new_a, 0.0) - jnp.where(reflect, new_b, 1.0)
    )
    new_a = jnp.where(reflect, reflected_a, new_a)
    swap = reflect & (new_a > new_b)
    new_a, new_b = jnp.where(swap, new_b, new_a), jnp.where(swap, new_a, new_b)
    is_small_b = (new_b < 8) | (new_a < 8 - new_b)
    is_nonpos_int_a = (new_a <= 0) & _is_integer(new_a)
    is_nonpos_int_b = (new_b <= 0) & _is_integer(new_b)
    use_small = is_small_b & ~(is_nonpos_int_a & is_nonpos_int_b)
    a_small = jnp.where(use_small, new_a, 1.0)
    b_small = jnp.where(use_small, new_b, 1.0)
    a_large = jnp.where(is_small_b, 1.0, new_a)
    b_large = jnp.where(is_small_b, 8.0, new_b)
    lgamma_a = lax.lgamma(new_a)
    small_b = lgamma_a + (lax.lgamma(b_small) - lax.lgamma(a_small + b_small))
    large_b = lgamma_a + algdiv(a_large, b_large)
    return jnp.where(is_small_b, small_b, large_b)
