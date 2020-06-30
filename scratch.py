import jax
import jax.numpy as np
import jax.numpy as jnp
import jax.linear_util as lu
from functools import partial

from jax.gmap import LoopType, gmap_impl

import os
from jax.lib import xla_bridge
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
# Clear any cached backends so new CPU backend will pick up the env var.
xla_bridge.get_backend.cache_clear()
print(jax.devices())

def f(x, y):
    return x.dot(jnp.sin(y))

lim_vmap = [(LoopType.sequential, None), (LoopType.vectorized, 2)]
vmap = [(LoopType.vectorized, None)]
soft_pmap = [(LoopType.parallel, 4), (LoopType.vectorized, None)]

x = jnp.ones((8, 64, 64))

def check(fun, sched, *args):
  def h(*args):
    return gmap_impl(lu.wrap_init(lambda *args: (fun(*args),)),
                     *args, axis_name='i', axis_size=8,
                     schedule=tuple(sched), mapped_invars=(True,) * len(args))
  print(jax.make_jaxpr(h)(*args))

check(f, lim_vmap, x, x)
check(f, soft_pmap, x, x)

