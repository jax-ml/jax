import numpy as np
import jax
from jax.sharding import Mesh, PartitionSpec as P
import jax.numpy as jnp
import timeline
from functools import partial

# How many GPUs we are using
num_devices = 8

# This defines our communication pattern
# Each GPU will pass their data around in a simple ring.
perm = [(i, (i - 1) % num_devices) for i in range(num_devices)]

mesh = Mesh(np.array(jax.devices()), ('data',))


def ring(a):
    return jax.lax.ppermute(a, 'data', perm)

def gemm(a, b):
    return a @ b

@jax.jit
@partial(jax.shard_map, mesh=mesh,
         in_specs=(P('data', None), P(None, 'data')),
         out_specs=P(None, 'data'))
def collective_matmul(a, b):

  c = jnp.zeros((num_devices, a.shape[0], b.shape[1]))

  idx = jax.lax.axis_index('data')
  c_part = None
  with timeline.Timeline() as t:
    for i in range(num_devices):
      if i != num_devices - 1:
        a_tmp = ring(a)
      c_part_tmp = t.async_call(gemm)(a, b)
      if c_part:
        c_part = t.ready(c_part)
        c = c.at[(idx + i - 1) % num_devices].set(c_part)
      c_part = c_part_tmp
      a = a_tmp
    c = c.at[num_devices - 1].set(t.ready(c_part))
  # Final reshape turns it back into a matrix
  return c.reshape((-1, c.shape[-1]))


a = jnp.ones((2048 * 8, 2048))
b = jnp.ones((2048, 2048 * 8))


print(collective_matmul.lower(a, b).as_text())
print(collective_matmul.lower(a, b).compile().as_text())
print(collective_matmul(a, b))
