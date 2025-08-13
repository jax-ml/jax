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


@jax.jit
@partial(jax.shard_map, mesh=mesh,
         in_specs=(P('data', None), P(None, 'data')),
         out_specs=P(None, 'data'))
def collective_matmul(a, b):

  # Initialize our placeholder c with zeros
  # We are making it a 3D tensor to make indexing math simpler later.
  c = jnp.zeros((num_devices, a.shape[0], b.shape[1]))

  # Mesh id of the GPU
  idx = jax.lax.axis_index('data')
  for i in range(num_devices):

    # Calculate the partial slice of c that we have
    c_part = a @ b

    # Store that result at the correct place in the full c matrix
    c = c.at[(idx + i) % num_devices].set(c_part)

    # Pass `a` around in a ring (except for the last cycle)
    if i != num_devices - 1:
      a = jax.lax.ppermute(a, 'data', perm)

  # Final reshape turns it back into a matrix
  return c.reshape((-1, c.shape[-1]))


a = jnp.ones((2048 * 8, 2048))
b = jnp.ones((2048, 2048 * 8))


print(collective_matmul.lower(a, b).as_text())
print(collective_matmul.lower(a, b).compile().as_text())
print(collective_matmul(a, b))
