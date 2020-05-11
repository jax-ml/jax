import scipy.special as osp_special

from jax.numpy import lax_numpy as jnp
from jax.numpy._util import _wraps

#TODO
def hyp2f1_neg_c_equal_bc(a, b, x):
    raise NotImplementedError

#TODO
def hyt2f1(a, b, c, x):
    raise NotImplementedError

@_wraps(osp_special.hyp2f1)
def hyp2f1(a, b, c, x):
    neg_int_a, neg_int_b = False, False
    neg_int_ca_or_cb = 0

    err = 0.0
    ax = jnp.abs(x)
    s = 1.0 - x
    ia = jnp.round(a)
    ib = jnp.round(b)

    if x == 0.0:
        return 1.0


    d = c - a - b
    id = jnp.round(d)

    if (a == 0 or b == 0) and c != 0:
        return 1.0

    if a <= 0 and jnp.abs(a - ia) < EPS: # a is a negative integer
        neg_int_a = True

    if b <= 0 and jnp.abs(b - ib) < EPS:
        neg_int_b = True

    if d <= -1 and not(jnp.abs(d - id) > EPS and s < 0) and not(neg_int_a or neg_int_b):
        return jnp.power(s, d) * hyp2f1(c - a, c - b, c, x)

    if d <= 0 and x == 1 and not(neg_int_a or neg_int_b):
        raise Exception

    if ax < 1.0 or x == -1.0:
        if jnp.abs(b - c) < EPS:
            if neg_int_b:
                y = hyp2f1_neg_c_equal_bc(a, b, x)
            else:
                y = jnp.power(s, -a)
            return y
        elif jnp.abs(a - c) < EPS:
            return jnp.power(s, -b)

    if c <= 0.0:
        ic = jnp.round(c)
        if jnp.abs(c - ic) < EPS:
            if neg_int_a and (ia > ic):
                y = hyt2f1(a, b, c, x)
                return y
            elif neg_int_b and (ib > ic):
                y = hyt2f1(a, b, c, x)
                return y
            else:
                return jnp.inf
    if neg_int_a or neg_int_b:
        y = hyt2f1(a, b, c, x)

    t1 = jnp.abs(b - a)
    if x < -2.0 and jnp.abs(t1 - jnp.round(t1)) > EPS:
        p = hyp2f1(a, 1. - c + a, 1. - b + a, 1.0 / x)
        q = hyp2f1(b, 1. - c + b, 1. - a + b, 1.0 / x)
        p *= jnp.power(-x, -a)
        q *= jnp.power(-x, -b)
        t1 = gamma(c) #TODO find where is gamma
        s = t1 * gamma(b - a) / (gamma(b) * gamma(c - a))
        y = t1 * gamma(b - a) / (gamma(a) * gamma(c - b))
        return s * p + y * q
    elif x < -1.0:
        if jnp.abs(a) < jnp.abs(b):
            return jnp.power(s, -a) * hyp2f1(a, c - b, c, x / (x - 1))
        else:
            return jnp.power(s, -b) * hyp2f1(b, c - a, c, x / (x - 1))
    if ax > 1.0:
        raise ValueError

    p = c - a
    ia = jnp.round(p)
    if ia <= 0.0 and jnp.abs(p - ia) < EPS:
        neg_int_ca_or_cb = True

    r = c - b
    ib = jnp.round(r)
    if ib <= 0.0 and jnp.abs(p - ib) < EPS:
        neg_int_ca_or_cb = True

    id = jnp.round(d)
    q = jnp.abs(d - id)

    if jnp.abs(ax - 1.) < EPS:
        if x > 0.:
           if d >= 0.0:
               return jnp.power(s, d) * hys2f1(c - a, c - b, c, x)
           else:
               raise ValueError
