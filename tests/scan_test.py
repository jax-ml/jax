from functools import partial

import numpy as onp

from jax.core import pack
from jax.lax import scan
import jax.numpy as np
from jax import jvp, linearize, grad

###

def scan_reference(f, init, xs):
  carry = init
  ys = []
  for x in xs:
    (carry, y) = f(carry, x)
    ys.append(y)
  ys = np.stack(ys)
  return pack((np.array(carry), ys))

d = np.zeros(2)
def f(c, a):
  assert a.shape == (3,)
  assert c.shape == (4,)
  b = np.sum(np.sin(a)) + np.sum(np.sin(c)) + np.sum(np.sin(d))
  c = np.sin(c * b)
  assert b.shape == ()
  return pack((c, b))

as_ = np.ones((5, 3))
c = np.ones(4)

print scan_reference(f, c, as_)
print scan(f, c, as_)
print

print jvp(lambda c, as_: scan_reference(f, c, as_), (c, as_), (c, as_))[1]
print jvp(lambda c, as_:   scan(f, c, as_), (c, as_), (c, as_))[1]
print

print linearize(lambda c, as_: scan_reference(f, c, as_), c, as_)[1](c, as_)
print linearize(lambda c, as_:   scan(f, c, as_), c, as_)[1](c, as_)
print

print grad(lambda c, as_: list(scan_reference(f, c, as_))[0].sum())(c, as_)
print grad(lambda c, as_:   list(scan(f, c, as_))[0].sum())(c, as_)
print

# ###


# def f(x, carry):
#   carry = carry + np.sin(x)
#   y = pack((carry**2, -carry))
#   return pack((y, carry))

# ys, z = scan_initial(f, 0.0, np.arange(4.))
# ys_ref, z_ref = scan_reference(f, 0.0, np.arange(4.))
# print onp.allclose(z, z_ref)

# print ys
# print ys_ref
# print z
# print z_ref
# print
# print jvp(partial(scan_initial, f), (0.0, np.arange(4.)), (1., np.array([0.3, 0.2, 0.1, 0.1])))
# print jvp(partial(scan_reference, f), (0.0, np.arange(4.)), (1., np.array([0.3, 0.2, 0.1, 0.1])))
# print
