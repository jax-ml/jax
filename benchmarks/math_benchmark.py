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


if __name__ == '__main__':
  benchmark.main()
