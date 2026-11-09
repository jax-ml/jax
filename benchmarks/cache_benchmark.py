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

import itertools
import random
import google_benchmark
from jax._src import util

def setup_util_cache(maxsize):
  @util.cache(max_size=maxsize)
  def f(x, y):
    return x + y
  return f

def run_cache_benchmark(state, cache_factory):
  hit_ratio = state.range(0)
  maxsize = 500
  f = cache_factory(maxsize)

  # Pre-generate queries with a stable hit ratio
  rng = random.Random(42)
  num_queries = 100000
  hot_keys = [(i, i) for i in range(100)]

  # Warm up the cache with hot keys so they start as hits
  for k in hot_keys:
    f(*k)

  queries = []
  cold_key_counter = 1000
  for _ in range(num_queries):
    if rng.randint(0, 99) < hit_ratio:
      # Hit: access a hot key
      queries.append(rng.choice(hot_keys))
    else:
      # Miss: access a new unique cold key
      queries.append((cold_key_counter, cold_key_counter))
      cold_key_counter += 1

  # Cycle through pre-generated queries to avoid Python overhead in the loop
  query_cycle = itertools.cycle(queries)

  while state:
    q = next(query_cycle)
    f(*q)

@google_benchmark.register
@google_benchmark.option.arg_names(['hit_ratio'])
@google_benchmark.option.arg(100)
@google_benchmark.option.arg(90)
@google_benchmark.option.arg(50)
@google_benchmark.option.arg(0)
def benchmark_util_cache(state):
  run_cache_benchmark(state, setup_util_cache)

if __name__ == "__main__":
  google_benchmark.main()
