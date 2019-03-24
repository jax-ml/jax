from jax.interpreters import cse
import jax.linear_util as lu
import jax.numpy as np
from jax import make_jaxpr, grad

###

def f(x):
  return np.sin(x) + np.sin(x)

print f(3.)
print cse.cse(lu.wrap_init(f)).call_wrapped(3.)
print make_jaxpr(cse.cse(lu.wrap_init(f)).call_wrapped)(3.)

print grad(f)(3.)
print grad(cse.cse(lu.wrap_init(f)).call_wrapped)(3.)

print grad(f)(3.)
print cse.cse(lu.wrap_init(grad(f))).call_wrapped(3.)

###

print
print

###

def f(x):
  return (np.sin(x) + np.cos(x)) + (np.cos(x) + np.sin(x))

print f(3.)
print cse.cse(lu.wrap_init(f)).call_wrapped(3.)
print make_jaxpr(cse.cse(lu.wrap_init(f)).call_wrapped)(3.)
