# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Microbenchmarks for JAX `api` functions."""
import functools
import operator

import google_benchmark
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


partial = functools.partial

def required_devices(num_devices_required):
  """Helper to skip benchmarks that require more devices."""
  def helper1(f):
    @functools.wraps(f)
    def helper2(state):
      if jax.device_count() < num_devices_required:
        state.skip_with_error(f"requires {num_devices_required} devices")
        return
      return f(state)
    return helper2
  return helper1


@google_benchmark.register
def eager_unary_dispatch(state):
  a = jax.device_put(1)
  lax.neg(a)
  while state:
    lax.neg(a)


@google_benchmark.register
def eager_unary(state):
  a = jax.device_put(1)
  lax.neg(a).block_until_ready()
  while state:
    lax.neg(a).block_until_ready()


@google_benchmark.register
def eager_binary_dispatch(state):
  a = jax.device_put(1)
  b = jax.device_put(2)
  lax.add(a, b)
  while state:
    lax.add(a, b)


@google_benchmark.register
def eager_binary(state):
  a = jax.device_put(1)
  b = jax.device_put(2)
  lax.add(a, b).block_until_ready()
  while state:
    lax.add(a, b).block_until_ready()


@google_benchmark.register
def jit_trivial_dispatch(state):
  """Benchmarks only the duration for jitted_f to return the future."""
  f = jax.jit(swap)
  a, b = f(1, 2)
  x = f(a, b)
  while state:
    x = f(a, b)
  x[0].block_until_ready()


@google_benchmark.register
def jit_trivial(state):
  f = jax.jit(swap)
  a, b = f(1, 2)
  f(a, b)

  while state:
    c, d = f(a, b)
    c.block_until_ready()
    d.block_until_ready()


@google_benchmark.register
def jit_simple_dispatch(state):
  a = jax.device_put(1)
  b = jax.device_put(2)
  f = jax.jit(operator.add)
  f(a, b)

  while state:
    f(a, b)


@google_benchmark.register
def jit_simple(state):
  a = jax.device_put(1)
  b = jax.device_put(2)
  f = jax.jit(operator.add)
  f(a, b)

  while state:
    f(a, b).block_until_ready()


def jit_simple_many_args_dispatch(n, state):
  args = [jax.device_put(i) for i in range(n)]
  f = jax.jit(lambda xs: functools.reduce(operator.add, xs))
  x = f(args)
  x.block_until_ready()

  while state:
    x = f(args)
  x.block_until_ready()

def jit_simple_many_args(n, state):
  args = [jax.device_put(i) for i in range(n)]
  f = jax.jit(lambda xs: functools.reduce(operator.add, xs))
  f(args).block_until_ready()

  while state:
    f(args).block_until_ready()

def jit_simple_pruned_args_dispatch(n, state):
  args = [jax.device_put(i) for i in range(n)]
  f = jax.jit(lambda *xs: xs[0] + 1)
  x = f(*args)
  x.block_until_ready()

  while state:
    x = f(*args)
  x.block_until_ready()


def jit_simple_pruned_args(n, state):
  args = [jax.device_put(i) for i in range(n)]
  f = jax.jit(lambda *xs: xs[0] + 1)
  x = f(*args)
  x.block_until_ready()

  while state:
    f(*args).block_until_ready()

benchmarks = []
for n in [10, 100, 1000, 2000]:
  benchmarks += [
      google_benchmark.register(partial(jit_simple_many_args_dispatch, n),
                                name=f"jit_simple_many_args_dispatch_{n}"),
      google_benchmark.register(partial(jit_simple_many_args, n),
                                name=f"jit_simple_many_args_{n}"),
      google_benchmark.register(partial(jit_simple_pruned_args_dispatch, n),
                                name=f"jit_simple_pruned_args_dispatch_{n}"),
      google_benchmark.register(partial(jit_simple_pruned_args, n),
                                name=f"jit_simple_pruned_args_{n}")
  ]


@google_benchmark.register
def jit_dispatch_without_transfer(state):
  # We pick up a realistic input. 224 is usual for classification and 128 a
  # TPU-friendly batch-size.
  imgs = np.ones((128, 224, 224), np.float32)
  imgs = jax.device_put(imgs)

  f = jax.api.jit(lambda x: x+1)
  f(imgs)

  while state:
    f(imgs)


@google_benchmark.register
def jit_dispatch_with_transfer(state):
  imgs = np.ones((128, 224, 224), np.float32)

  f = jax.api.jit(lambda x: x+1)
  f(imgs).block_until_ready()

  while state:
    x = f(imgs)
  x.block_until_ready()


@google_benchmark.register
@required_devices(2)
def pmap_trivial_2_devices(state):
  f = jax.pmap(swap)
  a, b = f(jnp.array([1, 2]), jnp.array([3, 4]))

  while state:
    c, d = f(a, b)
    c.block_until_ready()
    d.block_until_ready()


@google_benchmark.register
@required_devices(8)
def pmap_trivial_8_devices(state):
  f = jax.pmap(swap)
  a, b = f(jnp.array([1, 2, 3, 4, 5, 6, 7, 8]),
           jnp.array([2, 3, 4, 5, 6, 7, 8, 9]))

  while state:
    c, d = f(a, b)
    c.block_until_ready()
    d.block_until_ready()


@google_benchmark.register
@required_devices(2)
def pmap_simple_2_devices(state):
  f = jax.pmap(lambda a, b: (a + b, a - b))
  a, b = f(jnp.array([1, 2]), jnp.array([3, 4]))

  while state:
    c, d = f(a, b)
    c.block_until_ready()
    d.block_until_ready()


@google_benchmark.register
@required_devices(8)
def pmap_simple_8_devices(state):
  f = jax.pmap(lambda a, b: (a + b, a - b))
  a, b = f(jnp.array([1, 2, 3, 4, 5, 6, 7, 8]),
           jnp.array([2, 3, 4, 5, 6, 7, 8, 9]))

  while state:
    c, d = f(a, b)
    c.block_until_ready()
    d.block_until_ready()


def _run_sda_index_bench(state, num_devices):
  x = jax.pmap(jnp.sin)(jnp.arange(num_devices))
  jax.device_get(x)
  while state:
    for i in range(num_devices):
      _ = x[i]


@google_benchmark.register
@required_devices(1)
def sda_index_1(state):
  _run_sda_index_bench(state, 1)


@google_benchmark.register
@required_devices(2)
def sda_index_2(state):
  _run_sda_index_bench(state, 2)


@google_benchmark.register
@required_devices(8)
def sda_index_8(state):
  _run_sda_index_bench(state, 8)


def swap(a, b):
  return b, a


if __name__ == "__main__":
  google_benchmark.main()
