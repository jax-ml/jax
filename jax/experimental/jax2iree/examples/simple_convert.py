from jax import numpy as jnp
from jax.experimental import jax2iree


def f(x, y):
  return jnp.add(x, y)


builder = jax2iree.Builder()
jax2iree.trace_function(f, builder=builder,
    shapes_and_dtypes=[(None, (4, 4)), (None, (4, 4))])

print(builder.module)
