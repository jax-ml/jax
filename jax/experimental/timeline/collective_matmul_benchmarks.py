import numpy as np
import jax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental.compute_on import compute_on
from jax._src.xla_metadata import set_xla_metadata
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
def update_slice(c, c_part, idx):
    return c.at[idx].set(c_part)

def ring(a):
    return jax.lax.ppermute(a, 'data', perm)

@jax.jit
def gemm(a, b):
    return a @ b


@jax.jit
@partial(jax.shard_map, mesh=mesh,
         in_specs=(P('data', None), P('data', None)),
         out_specs=P('data', None))
def collective_matmul_normal(a, b):
  c = jnp.zeros((num_devices, a.shape[0], b.shape[1]))
  idx = jax.lax.axis_index('data')
  for i in range(num_devices):
    c_part = compute_on(f"gpu_stream:{(i%2) + 1}")(gemm)(a, b)
    if i != num_devices - 1:
      b = ring(b)
    c = update_slice(c, c_part, (idx + i - 1) % num_devices)
  # Final reshape turns it back into a matrix
  return c.reshape((-1, c.shape[-1]))



@jax.jit
@partial(jax.shard_map, mesh=mesh,
         in_specs=(P('data', None), P('data', None)),
         out_specs=P('data', None))
def collective_matmul_groups(a, b):
  c = jnp.zeros((num_devices, a.shape[0], b.shape[1]))
  idx = jax.lax.axis_index('data')
  for i in range(num_devices):
    with set_xla_metadata(_scheduling_group_id=str(i)):
      c_part = compute_on(f"gpu_stream:{(i%2) + 1}")(gemm)(a, b)
      if i != num_devices - 1:
        b = ring(b)
      c = update_slice(c, c_part, (idx + i - 1) % num_devices)
  # Final reshape turns it back into a matrix
  return c.reshape((-1, c.shape[-1]))




@jax.jit
@partial(jax.shard_map, mesh=mesh,
         in_specs=(P('data', None), P('data', None)),
         out_specs=P('data', None))
def collective_matmul(a, b):
  c_fut = None
  c_part_fut = None
  c = jnp.zeros((num_devices, a.shape[0], b.shape[1]))
  idx = jax.lax.axis_index('data')
  
  # Setup a control dependency timeline.
  with timeline.Timeline() as t:
      
    for i in range(num_devices):
      # Launch the communication first
      if i != num_devices - 1:
        b_fut = t.launch(ring(b))
      
      # Launch the gemm.
      c_part_fut_tmp = t.launch(
          compute_on(f"gpu_stream:{(i%2) + 1}")(gemm)(a, b))

      # If we have a part form the last iteration, 
      # use that and update the final c.
      if c_part_fut:
        c_part = t.finish(c_part_fut)
        c = c.at[(idx + i - 1) % num_devices].set(c_part)

      # Use the tmp c_part from earlier in the ext iteration.
      c_part_fut = c_part_fut_tmp
      if i != num_devices - 1:
        b = t.finish(b_fut)

    # Use the last piece of c before exiting the timeline.
    c = c.at[(idx - 1) % num_devices].set(t.finish(c_part_fut))
  
  # Final reshape turns it back into a matrix
  return c.reshape((-1, c.shape[-1]))

s = NamedSharding(mesh, P("data"))
a = jnp.ones((2048 * 8 * 4, 2048 * 4), dtype=jnp.bfloat16)
a = jax.device_put(a, s)
b = jnp.ones((2048 * 8 * 4, 2048 * 4), dtype=jnp.bfloat16)
b = jax.device_put(b, s)

print("Running scheduling group model")
for _ in range(10):
  collective_matmul_normal(a, b).block_until_ready()

print("Running scheduling group model")
for _ in range(10):
  collective_matmul_groups(a, b).block_until_ready()


print("Running timeline model")
for _ in range(10):
  collective_matmul(a, b).block_until_ready()


