# Copyright 2026 The JAX Authors.
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
"""Microbenchmarks for closed-over constants and embedded_constants_max_bytes.

Evaluates execution latency, host dispatch overhead, and compiled executable
binary size bloat when embedding constants vs hoisting them as arguments for
1KB, 4KB, 16KB, and 32KB constant arrays.
"""

import functools
import os

# Ensure simplified constant hoisting is enabled before JAX imports and classes define.
os.environ["JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS"] = "True"

import google_benchmark as benchmark
import jax
from jax._src import config
import jax.numpy as jnp
import numpy as np

SIZES_BYTES = [1024, 4096, 16384, 32768]


def _make_fn(num_bytes: int, embed: bool):
  """Creates a jitted function closing over a constant of num_bytes."""
  num_elements = max(1, num_bytes // 4)
  const = np.sin(np.arange(num_elements, dtype=np.float32))
  threshold = num_bytes if embed else 0
  with config.embedded_constants_max_bytes(threshold):

    @jax.jit
    def f(x):
      return x + const

    x_init = jnp.zeros_like(const)
    compiled = f.lower(x_init).compile()
  return f, compiled, x_init


# =============================================================================
# 1. Execution Latency & Binary Size Benchmarks
# =============================================================================


def _bench_execution(state, num_bytes: int, embed: bool):
  f, compiled, x = _make_fn(num_bytes, embed)
  threshold = num_bytes if embed else 0

  try:
    state.counters["executable_bytes"] = len(
        compiled.runtime_executable().serialize()
    )
  except Exception:
    pass

  with config.embedded_constants_max_bytes(threshold):
    f(x).block_until_ready()
    while state:
      f(x).block_until_ready()


for size in SIZES_BYTES:
  for embed, embed_str in [(True, "embedded"), (False, "hoisted")]:
    name = f"BM_Exec_{size}B_{embed_str}"
    benchmark.register(
        functools.partial(_bench_execution, num_bytes=size, embed=embed),
        name=name,
    )


# =============================================================================
# 2. Host Dispatch Latency Benchmarks (pure async host dispatch)
# =============================================================================


def _bench_dispatch(state, num_bytes: int, embed: bool):
  f, _, x = _make_fn(num_bytes, embed)
  threshold = num_bytes if embed else 0
  with config.embedded_constants_max_bytes(threshold):
    f(x).block_until_ready()
    while state:
      out = f(x)
    out.block_until_ready()


for size in SIZES_BYTES:
  for embed, embed_str in [(True, "embedded"), (False, "hoisted")]:
    name = f"BM_Dispatch_{size}B_{embed_str}"
    benchmark.register(
        functools.partial(_bench_dispatch, num_bytes=size, embed=embed),
        name=name,
    )


if __name__ == "__main__":
  benchmark.main()
