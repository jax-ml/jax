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

#ifndef JAXLIB_REENTRANT_HASH_MAP_H_
#define JAXLIB_REENTRANT_HASH_MAP_H_

#include <cstddef>
#include <functional>
#include <utility>

#include "absl/hash/hash.h"
#include "jaxlib/reentrant_hash_set.h"

namespace jax {

// A reentrancy-safe hash map, implemented as a wrapper around ReentrantHashSet.
// See reentrant_hash_set.h for details.
template <typename Key, typename Value, typename Hash = absl::Hash<Key>,
          typename Eq = std::equal_to<Key>>
class ReentrantHashMap {
 private:
  struct PairEq {
    Eq eq_;
    PairEq(Eq eq = Eq()) : eq_(eq) {}
    bool operator()(const std::pair<Key, Value>& a,
                    const std::pair<Key, Value>& b) const {
      return eq_(a.first, b.first);
    }
    template <typename K>
    bool operator()(const std::pair<Key, Value>& a, const K& b) const {
      return eq_(a.first, b);
    }
  };

  struct PairHash {
    Hash hasher_;
    PairHash(Hash hasher = Hash()) : hasher_(hasher) {}
    size_t operator()(const std::pair<Key, Value>& pair) const {
      return hasher_(pair.first);
    }
    template <typename K>
    size_t operator()(const K& key) const {
      return hasher_(key);
    }
  };

 public:
  ReentrantHashMap(size_t capacity = 0, const Hash& hasher = Hash(),
                   const Eq& eq = Eq())
      : set_(capacity, PairHash(hasher), PairEq(eq)) {}

  // This isn't exactly the usual type for iterators (the first part of the pair
  // should be const), but it's close enough for our use case.
  using iterator = typename ReentrantHashSet<std::pair<Key, Value>, PairHash,
                                             PairEq>::iterator;
  using const_iterator =
      typename ReentrantHashSet<std::pair<Key, Value>, PairHash,
                                PairEq>::const_iterator;

  template <typename K>
  iterator find(const K& key) {
    return set_.find(key);
  }

  template <typename K>
  const_iterator find(const K& key) const {
    return set_.find(key);
  }

  iterator begin() { return set_.begin(); }
  iterator end() { return set_.end(); }
  const_iterator begin() const { return set_.begin(); }
  const_iterator end() const { return set_.end(); }
  const_iterator cbegin() const { return set_.cbegin(); }
  const_iterator cend() const { return set_.cend(); }

  std::pair<std::pair<const Key, Value>*, bool> insert(const Key& key,
                                                       const Value& value) {
    auto res = set_.insert(std::pair<Key, Value>(key, value));
    return {reinterpret_cast<std::pair<const Key, Value>*>(res.first),
            res.second};
  }

  std::pair<std::pair<const Key, Value>*, bool> insert(
      const std::pair<Key, Value>& kv) {
    auto res = set_.insert(kv);
    return {reinterpret_cast<std::pair<const Key, Value>*>(res.first),
            res.second};
  }

  void erase(const_iterator it) { set_.erase(it); }

  void erase(iterator it) { set_.erase(it); }

  size_t size() const { return set_.size(); }
  bool empty() const { return set_.empty(); }
  void clear() { set_.clear(); }

 private:
  ReentrantHashSet<std::pair<Key, Value>, PairHash, PairEq> set_;
};

}  // namespace jax

#endif  // JAXLIB_REENTRANT_HASH_MAP_H_
