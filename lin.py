import jax.linear_util as lu
import jax.interpreters.linearize as lin
import jax.numpy as np

def f(x, y):
  return -np.sin(x) * y

### reference

from jax import make_jaxpr, jvp

def lin2(f, *args):
  jaxpr = make_jaxpr(lambda *args_dot: jvp(f, args, args_dot)[1])(*args)
  print(jaxpr)

jaxpr = lin2(f, 3., 4.)

### test

lin.lin(lu.wrap_init(f), 3., 4.)
