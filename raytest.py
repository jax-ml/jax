import os

import ray

import jax.numpy as np
from jax import rayjit
from jax import grad

from jax import config
config.config.update('jax_device_values', False)

ray.init()

@rayjit
def f(x):
  return np.sin(x)

print os.getpid()
print f(3.)
# print grad(f)(3.)
