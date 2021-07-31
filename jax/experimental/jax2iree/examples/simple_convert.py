from jax import numpy as jnp
from jax.experimental import jax2iree


def f(x, y):
  return jnp.add(x, y)

def fabs(x, y):
  z = jnp.add(x, y)
  return jnp.abs(z)


builder = jax2iree.Builder()
jax2iree.trace_function(fabs, builder=builder,
    shapes_and_dtypes=[(None, (1, 4)), (None, (4, 1))])
# jax2iree.trace_function(fabs, builder=builder,
#     shapes_and_dtypes=[(None, (4,))])

print(builder.module)
