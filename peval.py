from functools import partial
import numpy as onp
import jax.numpy as np
from jax import jit, pjit, grad, linearize, jvp, make_jaxpr
from jax.lax import psum

@partial(pjit, axis_name='i')
def f(x):
  return np.sin(np.sin(x))

x = onp.arange(2).reshape(1, 2).astype(onp.float32)
print f(x)

def splitjvp(x):
  _, jvp = linearize(f, x)
  return jvp(np.ones_like(x))

print splitjvp(x)
print make_jaxpr(splitjvp)(x)
print grad(lambda x: np.sum(np.sin(x)))(x)
print grad(lambda x: np.sum(f(x)))(x)

print grad(lambda x: np.sum(splitjvp(x)))(x)
print grad(lambda x: np.sum(jvp(np.sin, (x,), (np.ones_like(x),))[1]))(x)


###

@partial(pjit, axis_name='i')
@partial(pjit, axis_name='j')
def f(x):
  return psum(psum(x, 'i'), 'j')

print f(x.reshape((1, 1, -1)))
