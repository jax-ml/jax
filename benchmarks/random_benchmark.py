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
"""Microbenchmarks for JAX random."""

import google_benchmark
import jax
from jax import dtypes


def _assert_raw_key(key):
  assert key.dtype == "uint32"


def _assert_typed_key(key):
  assert dtypes.issubdtype(key.dtype, dtypes.prng_key)


def _bench_trivial_dispatch(state, key):
  f = jax.jit(lambda key: key)
  _ = f(key)
  while state:
    f(key)
  f(key).block_until_ready()


@google_benchmark.register
def trivial_dispatch_raw_key(state):
  key = jax.random.PRNGKey(0)
  _assert_raw_key(key)
  _bench_trivial_dispatch(state, key)


@google_benchmark.register
def trivial_dispatch_typed_key(state):
  key = jax.random.key(0)
  _assert_typed_key(key)
  _bench_trivial_dispatch(state, key)


def _bench_nontrivial_dispatch(state, key, do_split=False):
  key_op = jax.random.split if do_split else jax.random.normal
  f = jax.jit(lambda key: key_op(key))
  _ = f(key)
  while state:
    f(key)
  f(key).block_until_ready()


@google_benchmark.register
def nontrivial_dispatch_raw_key(state):
  key = jax.random.PRNGKey(0)
  _assert_raw_key(key)
  _bench_nontrivial_dispatch(state, key, do_split=False)


@google_benchmark.register
def nontrivial_dispatch_typed_key(state):
  key = jax.random.key(0)
  _assert_typed_key(key)
  _bench_nontrivial_dispatch(state, key, do_split=False)


@google_benchmark.register
def nontrivial_dispatch_raw_key_split(state):
  key = jax.random.PRNGKey(0)
  _assert_raw_key(key)
  _bench_nontrivial_dispatch(state, key, do_split=True)


@google_benchmark.register
def nontrivial_dispatch_typed_key_split(state):
  key = jax.random.key(0)
  _assert_typed_key(key)
  _bench_nontrivial_dispatch(state, key, do_split=True)



def _bench_custom_container(state, key):
  @jax.tree_util.register_pytree_node_class
  class A:
    def __init__(self, x):
      self.x = x

    def tree_flatten(self):
      return (self.x,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
      x, = children
      return cls(x)

  f = jax.jit(
      lambda key, a: jax.random.normal(key) + a.x)
  a = A(5.)
  _ = f(key, a)
  while state:
    f(key, a)
  f(key, a).block_until_ready()


@google_benchmark.register
def custom_container_raw_key(state):
  key = jax.random.PRNGKey(0)
  _assert_raw_key(key)
  _bench_custom_container(state, key)


@google_benchmark.register
def custom_container_typed_key(state):
  key = jax.random.key(0)
  _assert_typed_key(key)
  _bench_custom_container(state, key)


if __name__ == "__main__":
  google_benchmark.main()
