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
from jax import lax
from jax.experimental import sparse
import jax.numpy as jnp
import numpy as np


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


@google_benchmark.register
def jit_small_matmul(state):
  x = np.random.uniform(size=(2, 2)).astype(np.float32)
  x = jax.device_put(x)

  f = jax.jit(lambda x: jnp.dot(x, x))
  f(x).block_until_ready()

  while state:
    f(x).block_until_ready()


@google_benchmark.register
def jit_big_matmul(state):
  x = np.random.uniform(size=(100, 100)).astype(np.float32)
  x = jax.device_put(x)

  f = jax.jit(lambda x: jnp.dot(x, x))
  f(x).block_until_ready()

  while state:
    f(x).block_until_ready()


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

  f = jax.jit(lambda x: x+1)
  f(imgs)

  while state:
    f(imgs)


@google_benchmark.register
def jit_dispatch_with_transfer(state):
  imgs = np.ones((128, 224, 224), np.float32)

  f = jax.jit(lambda x: x+1)
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
def pmap_trivial_dispatch_8_devices(state):
  f = jax.pmap(swap)
  a, b = f(jnp.array([1, 2, 3, 4, 5, 6, 7, 8]),
           jnp.array([2, 3, 4, 5, 6, 7, 8, 9]))

  while state:
    a, b = f(a, b)


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
def pmap_simple_dispatch_8_devices(state):
  f = jax.pmap(lambda a, b: (a + b, a - b))
  a, b = f(jnp.array([1, 2, 3, 4, 5, 6, 7, 8]),
           jnp.array([2, 3, 4, 5, 6, 7, 8, 9]))

  while state:
    a, b = f(a, b)


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


@google_benchmark.register
@required_devices(8)
def pmap_simple_dispatch_8_devices_100_args(state):
  f = jax.pmap(lambda *args: args[1:] + (args[0] + 1,))
  args = []
  for i in range(100):
    args.append(jnp.array(list(range(i, i+8))))

  args = f(*args)

  while state:
    args = f(*args)


@google_benchmark.register
@required_devices(8)
def pmap_simple_8_devices_100_args(state):
  f = jax.pmap(lambda *args: args[1:] + (args[0] + 1,))
  args = []
  for i in range(100):
    args.append(jnp.array(list(range(i, i+8))))

  # Warmup loop.
  out = f(*args)

  while state:
    out = f(*args)
    jax.tree_map(lambda x: x.block_until_ready(), out)


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


def _sparse_bcoo_fromdense(state, jit: bool = False, compile: bool = False):
  shape = (2000, 2000)
  nse = 10000
  size = np.prod(shape)
  rng = np.random.RandomState(1701)
  data = rng.randn(nse)
  indices = np.unravel_index(
      rng.choice(size, size=nse, replace=False), shape=shape)
  mat = jnp.zeros(shape).at[indices].set(data)

  f = sparse.BCOO.fromdense
  if compile or jit:
    # Note: nse must be specified for JIT.
    f = jax.jit(partial(f, nse=nse))

  if compile:
    while state:
      f.lower(mat).compile()
  else:
    f(mat).block_until_ready()
    while state:
      f(mat).block_until_ready()


@google_benchmark.register
def sparse_bcoo_fromdense(state):
  return _sparse_bcoo_fromdense(state)


@google_benchmark.register
def sparse_bcoo_fromdense_jit(state):
  return _sparse_bcoo_fromdense(state, jit=True)


@google_benchmark.register
def sparse_bcoo_fromdense_compile(state):
  return _sparse_bcoo_fromdense(state, compile=True)


def _sparse_bcoo_todense(state, jit: bool = False, compile: bool = False):
  shape = (2000, 2000)
  nse = 10000
  size = np.prod(shape)
  rng = np.random.RandomState(1701)
  data = rng.randn(nse)
  indices = np.unravel_index(
      rng.choice(size, size=nse, replace=False), shape=shape)
  mat = sparse.BCOO((jnp.array(data), jnp.column_stack(indices)), shape=shape)

  f = lambda mat: mat.todense()
  if jit or compile:
    f = jax.jit(f)

  if compile:
    while state:
      f.lower(mat).compile()
  else:
    f(mat).block_until_ready()
    while state:
      f(mat).block_until_ready()


@google_benchmark.register
def sparse_bcoo_todense(state):
  return _sparse_bcoo_todense(state)


@google_benchmark.register
def sparse_bcoo_todense_jit(state):
  return _sparse_bcoo_todense(state, jit=True)


@google_benchmark.register
def sparse_bcoo_todense_compile(state):
  return _sparse_bcoo_todense(state, compile=True)


def _sparse_bcoo_matvec(state, jit: bool = False, compile: bool = False):
  shape = (2000, 2000)
  nse = 10000
  key = jax.random.PRNGKey(1701)
  mat = sparse.random_bcoo(key, nse=nse, shape=shape, dtype=jnp.float32,
                           indices_dtype=jnp.int32, sorted_indices=True)
  vec = jax.random.uniform(key, shape=(shape[1],), dtype=jnp.float32)

  f = lambda mat, vec: mat @ vec
  if jit or compile:
    f = jax.jit(f)

  if compile:
    while state:
      f.lower(mat, vec).compile()
  else:
    f(mat, vec).block_until_ready()
    while state:
      f(mat, vec).block_until_ready()


@google_benchmark.register
def sparse_bcoo_matvec(state):
  return _sparse_bcoo_matvec(state)


@google_benchmark.register
def sparse_bcoo_matvec_jit(state):
  return _sparse_bcoo_matvec(state, jit=True)


@google_benchmark.register
def sparse_bcoo_matvec_compile(state):
  return _sparse_bcoo_matvec(state, compile=True)


def swap(a, b):
  return b, a


if __name__ == "__main__":
  google_benchmark.main()
