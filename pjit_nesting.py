import numpy as onp

from jax import pjit, pmap, make_jaxpr
from jax.interpreters.parallel import psum


# def f(x):
#   return psum(psum(x, 'i'), 'j')
# f = pjit(f, 'i')
# f = pjit(f, 'j')

# x = onp.zeros((3, 4), onp.float32)
# print make_jaxpr(f)(x)


def f(x, y):
  return psum(psum(x, 'i'), 'j')
f = pjit(f, 'i')
f = pjit(f, 'j')

x = onp.zeros((3, 4), onp.float32)
print make_jaxpr(f)(x, x)
print f(x, x)
