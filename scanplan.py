from functools import partial
import numpy as onp

import jax.numpy as np
from jax import core
from jax import make_jaxpr, vjp
from jax import lax


def f(x, y):
  x_, = x
  y_, = y
  return core.pack((x_ * y_,))

g = partial(lax.scan, f)
a = onp.array(7, onp.float32)
bs = onp.array([2, 4, -2, 6], onp.float32)
out = g((a,), (bs,))

jaxpr = make_jaxpr(g)((a,), (bs,))


###

# scan :: (a -> b -> a) -> a -> [b] -> [a]


# first scan
as_ = lax.scan(f, a, bs)

# second scan
f_vjp = lambda args, ct: vjp(f, *args)[1](ct)
lax.scan(f_vjp, cts[-1], (as_, bs, cts))  # off by one stuff... draw boxes
