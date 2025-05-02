# Copyright 2023 The JAX Authors.
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
"""Microbenchmarks for sparse JAX."""

from functools import partial
import jax.numpy as jnp
import numpy as np
import math
import google_benchmark
import jax
from jax.experimental import sparse


def _sparse_fromdense(
    state,
    bcsr: bool = False,
    jit: bool = False,
    compile: bool = False,
):
  shape = (2000, 2000)
  nse = 10000
  size = math.prod(shape)
  rng = np.random.RandomState(1701)
  data = rng.randn(nse)
  indices = np.unravel_index(
      rng.choice(size, size=nse, replace=False), shape=shape
  )
  mat = jnp.zeros(shape).at[indices].set(data)

  f = sparse.BCSR.fromdense if bcsr else sparse.BCOO.fromdense
  if compile or jit:
    # Note: nse must be specified for JIT.
    f = jax.jit(partial(f, nse=nse))

  if compile:
    while state:
      state.pause_timing()
      jax.clear_caches()
      state.resume_timing()
      f.lower(mat).compile()
  else:
    f(mat).block_until_ready()
    while state:
      f(mat).block_until_ready()


def _sparse_todense(
    state,
    bcsr: bool = False,
    jit: bool = False,
    compile: bool = False,
):
  shape = (2000, 2000)
  nse = 10000
  size = math.prod(shape)
  rng = np.random.RandomState(1701)
  data = rng.randn(nse)
  indices = np.unravel_index(
      rng.choice(size, size=nse, replace=False), shape=shape
  )
  mat = sparse.BCOO((jnp.array(data), jnp.column_stack(indices)), shape=shape)
  if bcsr:
    mat = sparse.BCSR.from_bcoo(mat)

  f = lambda mat: mat.todense()
  if jit or compile:
    f = jax.jit(f)

  if compile:
    while state:
      state.pause_timing()
      jax.clear_caches()
      state.resume_timing()
      f.lower(mat).compile()
  else:
    f(mat).block_until_ready()
    while state:
      f(mat).block_until_ready()


def _sparse_matvec(
    state,
    bcsr: bool = False,
    jit: bool = False,
    compile: bool = False,
):
  shape = (2000, 2000)
  nse = 10000
  key = jax.random.key(1701)
  mat = sparse.random_bcoo(
      key,
      nse=nse,
      shape=shape,
      dtype=jnp.float32,
      indices_dtype=jnp.int32,
      sorted_indices=True,
  )
  if bcsr:
    mat = sparse.BCSR.from_bcoo(mat)

  vec = jax.random.uniform(key, shape=(shape[1],), dtype=jnp.float32)

  f = lambda mat, vec: mat @ vec
  if jit or compile:
    f = jax.jit(f)

  if compile:
    while state:
      state.pause_timing()
      jax.clear_caches()
      state.resume_timing()
      f.lower(mat, vec).compile()
  else:
    f(mat, vec).block_until_ready()
    while state:
      f(mat, vec).block_until_ready()


@google_benchmark.register
def sparse_bcoo_fromdense(state):
  return _sparse_fromdense(state)


@google_benchmark.register
def sparse_bcoo_fromdense_jit(state):
  return _sparse_fromdense(state, jit=True)


@google_benchmark.register
def sparse_bcoo_fromdense_compile(state):
  return _sparse_fromdense(state, compile=True)


@google_benchmark.register
def sparse_bcoo_todense(state):
  return _sparse_todense(state)


@google_benchmark.register
def sparse_bcoo_todense_jit(state):
  return _sparse_todense(state, jit=True)


@google_benchmark.register
def sparse_bcoo_todense_compile(state):
  return _sparse_todense(state, compile=True)


@google_benchmark.register
def sparse_bcoo_matvec(state):
  return _sparse_matvec(state)


@google_benchmark.register
def sparse_bcoo_matvec_jit(state):
  return _sparse_matvec(state, jit=True)


@google_benchmark.register
def sparse_bcoo_matvec_compile(state):
  return _sparse_matvec(state, compile=True)


@google_benchmark.register
def sparse_bscr_fromdense(state):
  return _sparse_fromdense(state, bcsr=True)


@google_benchmark.register
def sparse_bscr_fromdense_jit(state):
  return _sparse_fromdense(state, bcsr=True, jit=True)


@google_benchmark.register
def sparse_bscr_fromdense_compile(state):
  return _sparse_fromdense(state, bcsr=True, compile=True)


@google_benchmark.register
def sparse_bscr_todense(state):
  return _sparse_todense(state, bcsr=True)


@google_benchmark.register
def sparse_bscr_todense_jit(state):
  return _sparse_todense(state, bcsr=True, jit=True)


@google_benchmark.register
def sparse_bscr_todense_compile(state):
  return _sparse_todense(state, bcsr=True, compile=True)


@google_benchmark.register
def sparse_bcsr_matvec(state):
  return _sparse_matvec(state, bcsr=True)


@google_benchmark.register
def sparse_bcsr_matvec_jit(state):
  return _sparse_matvec(state, bcsr=True, jit=True)


@google_benchmark.register
def sparse_bcsr_matvec_compile(state):
  return _sparse_matvec(state, bcsr=True, compile=True)


if __name__ == "__main__":
  google_benchmark.main()
