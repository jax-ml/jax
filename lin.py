import jax.linear_util as lu
import jax.interpreters.linearize as lin
import jax.numpy as np

def f(x, y):
  return -np.sin(x) * y

def f(x, y):
  a = np.sin(x)
  # print(a)
  b = -a
  # print(b)
  c = b * y
  # print(c)
  return c

### reference

from jax import make_jaxpr, jvp

def lin2(f, *args):
  jaxpr = make_jaxpr(lambda *args_dot: jvp(f, args, args_dot)[1])(*args)
  print(jaxpr)

x, y = np.array([3.]), np.array([4.])
# x, y = 3., 4.
jaxpr = lin2(f, x, y)

### test

print(lin.lin(lu.wrap_init(f), x, y))
print(lin.rematlin(lu.wrap_init(f), x, y))
