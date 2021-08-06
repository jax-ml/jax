from jax import core
from jax import numpy as jnp
from jax.experimental import jax2iree


def f(x, y):
  return jnp.add(x, y)


def fabs(x, y):
  z = jnp.add(x, y)
  return (jnp.abs(z),)


builder = jax2iree.Builder()
out_avals = jax2iree.trace_flat_function(
    # Trace function.
    fabs,
    builder=builder,
    in_avals=[
        core.ShapedArray([1, 4], jnp.float32),
        core.ShapedArray([4, 1], jnp.float32)
    ])
print(f"OUTPUT avals: {out_avals}")

print(builder.module)
