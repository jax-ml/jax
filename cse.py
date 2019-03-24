from jax.interpreters import cse
import jax.linear_util as lu
from jax.core import pack
import jax.numpy as np
from jax import make_jaxpr, grad, jvp

###

# def f(x):
#   return np.sin(x) + np.sin(x)

# print f(3.)
# print cse.cse(lu.wrap_init(f)).call_wrapped(3.)
# print make_jaxpr(cse.cse(lu.wrap_init(f)).call_wrapped)(3.)

# print grad(f)(3.)
# print grad(cse.cse(lu.wrap_init(f)).call_wrapped)(3.)

# print grad(f)(3.)
# print cse.cse(lu.wrap_init(grad(f))).call_wrapped(3.)

# print
# print

###

# def f(x):
#   return (np.sin(x) * np.cos(x)) * (np.cos(x) * np.sin(x))

# print f(3.)
# print cse.cse(lu.wrap_init(f)).call_wrapped(3.)
# print make_jaxpr(cse.cse(lu.wrap_init(f)).call_wrapped)(3.)

# print
# print

###

def deriv(f):
  return lambda x: jvp(f, (x,), (1.,))

from collections import Counter

g = np.sin
for i in range(10):
  jaxpr = make_jaxpr(g)(3.)
  print(Counter(eqn.primitive for eqn in jaxpr.eqns).most_common())
  print()

  g = deriv(g)
