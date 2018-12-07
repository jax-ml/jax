import os

from jax import rayjit
from jax import grad

@rayjit
def f(x):
  print 'hi from {}'.format(os.getpid())
  return x**2

print f(3.)
print grad(f)(3.)
