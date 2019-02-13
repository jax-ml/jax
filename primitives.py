import jax.numpy as np
from jax import jit, grad, jvp, make_jaxpr
from jax import primitive

@primitive  # try commenting me out
def library_fun(x):
  return np.sin(np.cos(x))

def user_fun(x):
  y = library_fun(x)
  z = np.tanh(x)
  return y + z


print user_fun(3.)
print jit(user_fun)(3.)
print grad(user_fun)(3.)
print grad(jit(user_fun))(3.)

print make_jaxpr(user_fun)(3.)
