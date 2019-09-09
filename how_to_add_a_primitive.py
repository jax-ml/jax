import jax.numpy as np
import numpy as onp  # raw numpy

from jax.core import Primitive

def sin(x):
  return sin_p.bind(x)

sin_p = Primitive('sin')


# add an evaluation rule
sin_p.def_impl(onp.sin)
print sin(3.)


# add a jvp rule
from jax.interpreters import ad
def sin_jvp(primals, tangents):
  x, = primals
  t, = tangents
  primal_out = sin(x)
  tangent_out = np.cos(x) * t
  return primal_out, tangent_out
ad.primitive_jvps[sin_p] = sin_jvp


from jax import jvp
print jvp(sin, (3.,), (1.,))
print jvp(lambda x: sin(sin(x)), (3.,), (1.,))

def deriv(f):
  return lambda x: jvp(f, (x,), (1.,))[1]

print deriv(deriv(sin))(3.)
