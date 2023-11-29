# Copyright 2020 The JAX Authors.
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

"""Benchmarks for JAX linear algebra functions."""

import google_benchmark
import jax
import jax.numpy as jnp
import numpy as np


@google_benchmark.register
@google_benchmark.option.arg_names(['m', 'n'])
@google_benchmark.option.args_product(
    [[1, 2, 5, 10, 100, 500, 800, 1000], [1, 2, 5, 10, 100, 500, 800, 1000]]
)
def svd(state):
  np.random.seed(1234)
  m, n = state.range(0), state.range(1)
  x = np.random.randn(m, n).astype(np.float32)
  jax.block_until_ready(jnp.linalg.svd(x)[0])
  while state:
    jax.block_until_ready(jnp.linalg.svd(x)[0])


if __name__ == '__main__':
  google_benchmark.main()
