from functools import partial

from jax.core import pack
import jax.core as core
import jax.numpy as np
from jax import jvp, linearize
from jax.initial_style import call_initial


def f1(x, y, z):
  return core.pack((np.sin(x * y), y, 1.0))

# def f1(x, y, z):
#   return core.pack((np.sin(x * y), y))

f2 = partial(call_initial, f1)

xs = (1., 2., 3.)
xst = (4., 5., 6.)

print "\neval"
print f1(*xs)
print f2(*xs)


print "\njvp"
print jvp(f1, xs, xst)
print jvp(f2, xs, xst)

print "\nlinearize"
print linearize(f1, *xs)[1](*xst)
print linearize(f2, *xs)[1](*xst)


