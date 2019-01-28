import jax.numpy as np

from jax.interpreters import pxla
from jax.interpreters.parallel import psum
from jax import pjit

import numpy as onp

def f(x):
  return x - psum(x, 'i')

x = onp.arange(8., dtype=onp.float32).reshape(4, 2)
f = pjit(f, axis_name='i', in_axes=0, out_axes=0, mesh_axis=0)

print f(x)
print x - x.sum(0)
