/* Copyright 2026 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Benchmarks comparing ReentrantHashMap/Set against standard containers
// (absl::flat_hash_map and std::unordered_map).

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "jaxlib/reentrant_hash_map.h"
#include "jaxlib/reentrant_hash_set.h"
#include "xla/tsl/platform/test_benchmark.h"

namespace jax {
namespace {

template <typename Set>
void BM_SetInsertErase(benchmark::State& state) {
  const int n = state.range(0);
  absl::BitGen gen;
  std::vector<int> keys(n);
  for (int i = 0; i < n; ++i) {
    keys[i] = absl::Uniform<int>(gen, 0, 1 << 30);
  }

  for (auto _ : state) {
    Set s;
    for (int k : keys) {
      s.insert(k);
    }
    for (int k : keys) {
      auto it = s.find(k);
      if (it != s.end()) {
        s.erase(it);
      }
    }
  }
}

template <typename Set>
void BM_SetFind(benchmark::State& state) {
  const int num_elements = state.range(0);
  const int hit_percent = state.range(1);

  absl::BitGen gen;
  std::vector<int> keys(num_elements);
  Set s;
  for (int i = 0; i < num_elements; ++i) {
    keys[i] = absl::Uniform<int>(gen, 0, 1 << 30);
    s.insert(keys[i]);
  }

  std::vector<int> queries(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    if (absl::Bernoulli(gen, hit_percent / 100.0)) {
      queries[i] = keys[absl::Uniform<size_t>(gen, 0, keys.size())];  // Hit
    } else {
      queries[i] = absl::Uniform<int>(gen, -(1 << 30), -1);  // Miss (negative)
    }
  }

  int hits = 0;
  for (auto _ : state) {
    for (int q : queries) {
      auto it = s.find(q);
      if (it != s.end()) {
        ++hits;
      }
    }
  }
  benchmark::DoNotOptimize(hits);
}

BENCHMARK(BM_SetInsertErase<absl::flat_hash_set<int>>)->Range(8, 8 << 10);
BENCHMARK(BM_SetInsertErase<jax::ReentrantHashSet<int>>)->Range(8, 8 << 10);
BENCHMARK(BM_SetInsertErase<std::unordered_set<int>>)->Range(8, 8 << 10);

BENCHMARK(BM_SetFind<absl::flat_hash_set<int>>)
    ->Args({8, 100})
    ->Args({8, 50})
    ->Args({8, 0})
    ->Args({8 << 10, 100})
    ->Args({8 << 10, 50})
    ->Args({8 << 10, 0});
BENCHMARK(BM_SetFind<jax::ReentrantHashSet<int>>)
    ->Args({8, 100})
    ->Args({8, 50})
    ->Args({8, 0})
    ->Args({8 << 10, 100})
    ->Args({8 << 10, 50})
    ->Args({8 << 10, 0});
BENCHMARK(BM_SetFind<std::unordered_set<int>>)
    ->Args({8, 100})
    ->Args({8, 50})
    ->Args({8, 0})
    ->Args({8 << 10, 100})
    ->Args({8 << 10, 50})
    ->Args({8 << 10, 0});

template <typename Map>
void BM_MapInsertErase(benchmark::State& state) {
  const int n = state.range(0);
  absl::BitGen gen;
  std::vector<int> keys(n);
  for (int i = 0; i < n; ++i) {
    keys[i] = absl::Uniform<int>(gen, 0, 1 << 30);
  }

  for (auto _ : state) {
    Map m;
    for (int k : keys) {
      m.insert({k, k});
    }
    for (int k : keys) {
      auto it = m.find(k);
      if (it != m.end()) {
        m.erase(it);
      }
    }
  }
}

BENCHMARK(BM_MapInsertErase<absl::flat_hash_map<int, int>>)->Range(8, 8 << 10);
BENCHMARK(BM_MapInsertErase<jax::ReentrantHashMap<int, int>>)
    ->Range(8, 8 << 10);
BENCHMARK(BM_MapInsertErase<std::unordered_map<int, int>>)->Range(8, 8 << 10);

template <typename Map>
void BM_MapFind(benchmark::State& state) {
  const int num_elements = state.range(0);
  const int hit_percent = state.range(1);

  absl::BitGen gen;
  std::vector<int> keys(num_elements);
  Map m;
  for (int i = 0; i < num_elements; ++i) {
    keys[i] = absl::Uniform<int>(gen, 0, 1 << 30);
    m.insert({keys[i], keys[i]});
  }

  std::vector<int> queries(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    if (absl::Bernoulli(gen, hit_percent / 100.0)) {
      queries[i] = keys[absl::Uniform<size_t>(gen, 0, keys.size())];  // Hit
    } else {
      queries[i] = absl::Uniform<int>(gen, -(1 << 30), -1);  // Miss (negative)
    }
  }

  int hits = 0;
  for (auto _ : state) {
    for (int q : queries) {
      auto it = m.find(q);
      if (it != m.end()) {
        ++hits;
      }
    }
  }
  benchmark::DoNotOptimize(hits);
}

BENCHMARK(BM_MapFind<absl::flat_hash_map<int, int>>)
    ->Args({8, 100})
    ->Args({8, 50})
    ->Args({8, 0})
    ->Args({8 << 10, 100})
    ->Args({8 << 10, 50})
    ->Args({8 << 10, 0});
BENCHMARK(BM_MapFind<jax::ReentrantHashMap<int, int>>)
    ->Args({8, 100})
    ->Args({8, 50})
    ->Args({8, 0})
    ->Args({8 << 10, 100})
    ->Args({8 << 10, 50})
    ->Args({8 << 10, 0});
BENCHMARK(BM_MapFind<std::unordered_map<int, int>>)
    ->Args({8, 100})
    ->Args({8, 50})
    ->Args({8, 0})
    ->Args({8 << 10, 100})
    ->Args({8 << 10, 50})
    ->Args({8 << 10, 0});

}  // namespace
}  // namespace jax
