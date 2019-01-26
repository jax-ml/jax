import jax.numpy as np

from jax.interpreters import pxla
from jax.interpreters.parallel import psum
from jax import pjit

import numpy as onp
pxla.mesh_spec = (1,)

def f(x):
  return x - np.log(psum(np.exp(x), 'i'))

x = onp.ones((4, 2), onp.float32)
f = pjit(f, axis_name='i', in_axes=0, out_axes=0, mesh_axis=0)
print f(x)
