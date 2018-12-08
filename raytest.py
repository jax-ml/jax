import os
import time

import ray

import jax.numpy as np
from jax import rayjit
from jax import grad

from jax import config
config.config.update('jax_device_values', False)

ray.init()
time.sleep(1)

@rayjit
def f(x):
  return np.sin(x)

print "master pid", os.getpid()
# print "eval", f(3.)
print "grad", grad(f)(3.)
