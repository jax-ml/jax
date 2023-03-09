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
"""Microbenchmarks for floating point operations."""

import functools

import google_benchmark as benchmark
import jax
import jax.numpy as jnp
import numpy as np

from google_benchmark import Counter


def math_benchmark(*args):
  def decorator(func):
    for test_case in args[0]:

      @benchmark.register(name=f"{func.__name__}_{test_case['name']}")
      @functools.wraps(func)
      def wrapper(state, test_case=test_case):
        return func(state, **test_case)

    return wrapper

  return decorator


@math_benchmark(
    [
        {
            'name': f'{op.__name__}_{shape}_{dtype}',
            'shape': shape,
            'dtype': dtype,
            'op': op,
        }
        for op in [
            jnp.exp,
            jnp.exp2,
            jnp.expm1,
            jnp.log,
            jnp.log2,
            jnp.log1p,
            jnp.tanh,
        ]
        for shape in [2**i for i in range(10, 15, 2)]
        for dtype in ['float32']
    ]
)
def jax_unary(state, **kwargs):
  shape = kwargs['shape']
  dtype = kwargs['dtype']
  op = kwargs['op']
  input0 = np.random.random(shape).astype(dtype)
  f = op
  f_jitted = jax.jit(f)
  f_jitted(input0).block_until_ready()
  while state:
    f_jitted(input0).block_until_ready()
  state.counters['items_per_second'] = Counter(
      input0.size * state.iterations, Counter.kIsRate
  )

@math_benchmark(
    [
        {
            'name': f'{op.__name__}_{mkn[0]}x{mkn[1]}x{mkn[2]}_{dtype}',
            'mkn': mkn,
            'dtype': dtype,
            'op': op,
        }
        for op in [
            jnp.dot,
        ]
        for mkn in [[2**i, 2**i, 2**i] for i in range(4, 11, 1)] +
        [
            [1, 2, 256],
            [1, 8, 256],
            [1, 18, 300],
            [1, 37, 256],
            [1, 91, 256],
            [1, 111, 256],
            [1, 192, 192],
            [1, 226, 256],
            [1, 256, 192],
            [1, 256, 256],
            [1, 512, 512],
            [1, 300, 18],
            [21, 24, 1],
            [21, 120, 1],
            [10, 10, 10],
            [100, 100, 100],
            [18, 1, 300],
            [18, 300, 1],
            [300, 1, 18],
            [300, 18, 1],
        ]
        for dtype in ['float32']
    ]
)
def jax_binary_op(state, **kwargs):
  mkn = kwargs['mkn']
  m = mkn[0]
  k = mkn[1]
  n = mkn[2]
  dtype = kwargs['dtype']
  op = kwargs['op']
  a = np.random.random([m, k]).astype(dtype)
  b = np.random.random([k, n]).astype(dtype)
  f = op
  f_jitted = jax.jit(f)
  f_jitted(a, b).block_until_ready()
  while state:
    f_jitted(a, b).block_until_ready()
  state.counters['items_per_second'] = Counter(
      state.iterations, Counter.kIsRate
  )


if __name__ == '__main__':
  benchmark.main()
