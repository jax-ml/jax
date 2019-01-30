import numpy as onp

import jax.numpy as np
from jax import grad, pjit, pmap, make_jaxpr
from jax.interpreters.parallel import psum


# def f(x, y):
#   return psum(psum(x, 'i'), 'j')
# f = pjit(f, 'i')
# f = pjit(f, 'j', out_axes=1)
# x = onp.ones((3, 4), onp.float32)
# print make_jaxpr(f)(x, x)
# print f(x, x)


def f(x):
  return x - psum(x, 'i')

x = np.zeros(4)
print grad(lambda x: np.sum(pmap(f, 'i')(x)))(x)
print grad(lambda x: np.sum(x - np.sum(x)))(x)

g = pjit(f, axis_name='i')
print grad(lambda x: np.sum(g(x)))(x)
