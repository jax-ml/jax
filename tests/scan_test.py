from functools import partial

import numpy as onp

from jax.scan import scan_reference
from jax.initial_style import scan_initial
from jax.core import pack
import jax.core as core
import jax.numpy as np
from jax import jvp, linearize

# scan :: (a -> c -> (b,c)) -> c -> [a] -> ([b],c)

###

def cumsum(xs):
  def f(x, carry):
    carry = carry + x
    return pack((carry, carry))

  ys, _ = scan_initial(f, 0.0, xs)
  return ys

x = np.linspace(0, 3, 4)

print np.cumsum(x)
print cumsum(x)
print

# print jvp(np.cumsum, (x,), (x*0.1,))
# print jvp(cumsum, (x,), (x*0.1,))
# print
# print linearize(np.cumsum, x)[1](x*0.1)
# print linearize(cumsum, x)[1](x*0.1)


###


def f(x, carry):
  carry = carry + np.sin(x)
  y = pack((carry**2, -carry))
  return pack((y, carry))

ys, z = scan_initial(f, 0.0, np.arange(4.))
ys_ref, z_ref = scan_reference(f, 0.0, np.arange(4.))
print onp.allclose(z, z_ref)

print ys
print ys_ref
print z
print z_ref
# print jvp(partial(scan_initial, f), (0.0, np.arange(4.)), (1., np.array([0.3, 0.2, 0.1, 0.1])))
# print jvp(partial(scan_reference, f), (0.0, np.arange(4.)), (1., np.array([0.3, 0.2, 0.1, 0.1])))
print
