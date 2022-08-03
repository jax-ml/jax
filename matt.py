import pdb, sys, traceback
def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    pdb.pm()
sys.excepthook = info

import os
import functools
import re

import jax
import jax.numpy as jnp
from jax.config import config
import numpy as np

config.update("jax_new_checkpoint", True)
config.update("jax_traceback_filtering", "off")
config.update("jax_platform_name", "cpu")

def set_host_device_count(n):
    xla_flags = os.getenv("XLA_FLAGS", "")
    xla_flags = re.sub(
        r"--xla_force_host_platform_device_count=\S+", "", xla_flags
    ).split()
    os.environ["XLA_FLAGS"] = " ".join(
        ["--xla_force_host_platform_device_count={}".format(n)] + xla_flags
    )

set_host_device_count(8)

# @functools.partial(jax.pmap, axis_name="foo")
# def f(x):
#   def cond(i):
#     return i < 10
#   def body(i):
#     return i + 1
#   return jax.lax.while_loop(cond, body, x)

# @functools.partial(jax.pmap, axis_name="foo")
# def f(x):
#   def cond(i):
#     return i < 10
#   def body(i):
#     return i + 1
#   return jax.lax.while_loop(cond, body, x)

# print(f(jnp.arange(4.)))

# with jax.disable_jit():
#   print(f(jnp.arange(4.)))


# @jax.pmap
# def g(x):
#   with jax._src.config.disable_jit(False):
#     return jax.jit(jnp.sin)(x)

# print(g(jnp.arange(4.)))

# with jax.disable_jit():
#   print(g(jnp.arange(4.)))

# @functools.partial(jax.pmap, in_axes=(0, None, 0, None), axis_name='i')
# @functools.partial(jax.pmap, in_axes=(None, 0, 0, None), axis_name='j')
# def f(x, y, z, w):
#   return jax.lax.axis_index(['i', 'j'])  + x * y + z + w

# print(f(jnp.arange(4.), jnp.arange(2.), jnp.arange(8.).reshape((4, 2)), 100.))

# with jax.disable_jit():
#   print(f(jnp.arange(4.), jnp.arange(2.), jnp.arange(8.).reshape((4, 2)), 100.))

device_count = jax.device_count()

# @functools.partial(jax.pmap, axis_name='i')
# def f(x):
#   @functools.partial(jax.pmap, axis_name='j')
#   def g(y):
#     a = jax.lax.psum(1, 'i')
#     b = jax.lax.psum(1, 'j')
#     c = jax.lax.psum(1, ('i', 'j'))
#     return a, b, c
#   return g(x)

# import numpy as np
# shape = (device_count, 1, 4)
# x = jnp.arange(np.prod(shape)).reshape(shape)
# a, b, c = f(x)
# print(a)
# print(b)
# print(c)

# with jax.disable_jit():
#   a, b, c = f(x)
#   print(a)
#   print(b)
#   print(c)

# f = lambda axis: jax.pmap(jax.pmap(lambda x: x + jax.lax.axis_index(axis), 'j'), 'i')
# x = jnp.ones((2, 2), dtype='int32')
# print(f('i')(x))
# print(f('j')(x))

# with jax.disable_jit():
#   print(f('i')(x))
  # print(f('j')(x))


# def f(key):
#   key = jax.random.fold_in(key, jax.lax.axis_index('i'))
#   return jax.random.bernoulli(key, p=0.5)

# keys = jax.random.split(jax.random.PRNGKey(0), len(jax.devices()))

# print(jax.pmap(jax.remat(f), axis_name='i')(keys))

# with jax.disable_jit():
#   print(jax.pmap(jax.remat(f), axis_name='i')(keys))


# jax.pmap(lambda x: x)(jnp.zeros(jax.device_count() + 1))
# with jax.disable_jit():
#   jax.pmap(lambda x: x)(jnp.zeros(jax.device_count() + 1))

# jax.pmap(lambda x: x)(jnp.zeros(jax.device_count() + 1))
# with jax.disable_jit():
#   jax.pmap(jax.pmap(jnp.square))(jnp.arange(16).reshape((4, 4)))

# f = jax.pmap(lambda x: jax.pmap(lambda x: x)(x))
# x = jnp.ones((jax.device_count(), 2, 10))
# f(x)
# with jax.disable_jit():
#   print(f(x))

f = jax.pmap(jax.pmap(lambda x: 3))
shape = (2, jax.device_count() // 2, 3)
x = jnp.arange(np.prod(shape)).reshape(shape)
print(f(x))
with jax.disable_jit():
  print(f(x))
# TODO:
# * [x] process_call
#   * jit-of-emap = pmap (already, b/c we're only changing the pmap impl)
#   * emap-of-jit = our processs_call rule should act same as initial style HOPs
#   * emap-of-core.call = do a subtrace like thing where we turn around and stay
#     in python
# * [ ] collectives
# * [ ] testing
# * [ ] nesting (process_map, sublift, etc)
#   * [ ] shadowing of names

# * delete process_call and core.call, just have an xla_call rule
#   * no call updaters!
#   * blocked on delete old remat
#   * first just make xla_call have its own rule
#   * then make it INITIAL STYLE
#     * make it take closed jaxprs, so we can delete core.closed_call
# * delete all updaters






# brainstorming process_map
    # assert map_primitive is xla_pmap_p
    # backend, axis_size, axis_name = (
    #     params['backend'], params['axis_size'], params['axis_name'])
    # if config.jax_disable_jit:
    #   shape = [f.size for f in self._get_frames()]
    #   devices = xb.devices(backend=backend)[:prod(shape) * axis_size]
    #   breakpoint()
    #   # def reshard(x: jnp.ndarray, devices: Array[devices]):
    #   #   assert x.ndim == devices.ndim
    #   #   e.g  . x.shape = (4, 3, 2)
    #   #    devices.shape = (4, 1, 2)
    #   # reshard(x, devices.reshape(4, 1, 2))
    #   sharded_args = [jax.device_put_sharded(list(x), devices) for x in args]
    #   with core.new_sublevel(), core.extend_axis_env(axis_name, axis_size, main):
    #     t = main.with_cur_sublevel()
    #     shard_axes = {axis_name: 0}
    #     tracers = [MapTracer(t, arg, shard_axes) for arg in sharded_args]
    #     ans = fun.call_wrapped(*tracers)
    #     out_tracers = map(t.full_raise, ans)
    #     outvals = [t.val for t in out_tracers]
    #   return outvals
    # else:
    #   breakpoint()
