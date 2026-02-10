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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/random/random.h"
#include "absl/synchronization/mutex.h"
#include "jaxlib/reentrant_hash_map.h"
#include "jaxlib/reentrant_hash_set.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"

namespace jax {
namespace {

TEST(BitMaskTest, Basic) {
  BitMask mask(0x2C);  // 0b101100
  EXPECT_TRUE(mask);
  EXPECT_EQ(mask.LowestBitSet(), 2);

  std::vector<int> set_bits;
  for (int bit : mask) {
    set_bits.push_back(bit);
  }
  ASSERT_EQ(set_bits.size(), 3);
  EXPECT_EQ(set_bits[0], 2);
  EXPECT_EQ(set_bits[1], 3);
  EXPECT_EQ(set_bits[2], 5);

  BitMask empty(0);
  EXPECT_FALSE(empty);
}

TEST(GroupTest, Match) {
  int8_t ctrl[Group::kWidth];
  std::memset(ctrl, kEmpty, Group::kWidth);
  ctrl[0] = 42;
  ctrl[1] = 42;
  ctrl[Group::kWidth - 1] = 43;

  Group g(ctrl);
  BitMask m = g.Match(42);
  std::vector<int> matches;
  for (int bit : m) matches.push_back(bit);
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], 0);
  EXPECT_EQ(matches[1], 1);

  BitMask m2 = g.Match(43);
  EXPECT_EQ(m2.LowestBitSet(), Group::kWidth - 1);
}

TEST(GroupTest, MatchEmpty) {
  int8_t ctrl[Group::kWidth];
  std::memset(ctrl, 42, Group::kWidth);
  ctrl[2] = kEmpty;
  ctrl[5] = kEmpty;

  Group g(ctrl);
  BitMask m = g.MatchEmpty();
  std::vector<int> matches;
  for (int bit : m) matches.push_back(bit);
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], 2);
  EXPECT_EQ(matches[1], 5);
}

TEST(GroupTest, MatchEmptyOrDeleted) {
  int8_t ctrl[Group::kWidth];
  std::memset(ctrl, 42, Group::kWidth);
  ctrl[1] = kEmpty;
  ctrl[3] = kDeleted;
  ctrl[4] = kEmpty;

  Group g(ctrl);
  BitMask m = g.MatchEmptyOrDeleted();
  std::vector<int> matches;
  for (int bit : m) matches.push_back(bit);
  ASSERT_EQ(matches.size(), 3);
  EXPECT_EQ(matches[0], 1);
  EXPECT_EQ(matches[1], 3);
  EXPECT_EQ(matches[2], 4);
}

TEST(ReentrantHashSetTest, Basic) {
  ReentrantHashSet<int> set;
  EXPECT_EQ(set.find(42), set.end());
  EXPECT_TRUE(set.insert(42).second);
  EXPECT_FALSE(set.insert(42).second);
  ASSERT_NE(set.find(42), set.end());
  EXPECT_EQ(*set.find(42), 42);
  EXPECT_EQ(set.size(), 1);
  auto it1 = set.find(42);
  ASSERT_NE(it1, set.end());
  set.erase(it1);
  EXPECT_EQ(set.size(), 0);
  EXPECT_EQ(set.find(42), set.end());
}

TEST(ReentrantHashMapTest, Basic) {
  ReentrantHashMap<int, int> map;
  EXPECT_EQ(map.find(42), map.end());
  EXPECT_TRUE(map.insert(42, 100).second);
  EXPECT_FALSE(map.insert(42, 200).second);
  ASSERT_NE(map.find(42), map.end());
  EXPECT_EQ(map.find(42)->second, 100);
  EXPECT_EQ(map.size(), 1);
  auto it1 = map.find(42);
  ASSERT_NE(it1, map.end());
  map.erase(it1);
  EXPECT_EQ(map.size(), 0);
  EXPECT_EQ(map.find(42), map.end());
}

struct StringHash {
  using is_transparent = void;
  size_t operator()(const std::string& s) const {
    return absl::Hash<std::string>()(s);
  }
  size_t operator()(const char* s) const {
    return absl::Hash<std::string>()(s);
  }
};

struct StringEq {
  using is_transparent = void;
  bool operator()(const std::string& a, const std::string& b) const {
    return a == b;
  }
  bool operator()(const std::string& a, const char* b) const { return a == b; }
  bool operator()(const char* a, const std::string& b) const { return b == a; }
};

TEST(ReentrantHashMapTest, HeterogeneousLookup) {
  ReentrantHashMap<std::string, int, StringHash, StringEq> map;
  EXPECT_TRUE(map.insert("hello", 42).second);

  // Heterogeneous find
  ASSERT_NE(map.find("hello"), map.end());  // const char* lookup
  EXPECT_EQ(map.find("hello")->second, 42);

  // Heterogeneous erase
  auto it1 = map.find("hello");
  ASSERT_NE(it1, map.end());
  map.erase(it1);
  EXPECT_EQ(map.size(), 0);
}

TEST(ReentrantHashSetTest, ReentrantCallbackRehash) {
  struct ReentrantEq;
  using SetType = ReentrantHashSet<int, absl::Hash<int>, ReentrantEq>;

  struct ReentrantEq {
    SetType** set_ptr_ptr = nullptr;
    bool operator()(int a, int b) const {
      if (a == 99 && b == 99) {
        SetType* set = *set_ptr_ptr;
        // Trigger a rehash by inserting many elements while we are inside
        // `find(99)`
        for (int i = 1000; i < 2000; ++i) {
          set->insert(i);
        }
      }
      return a == b;
    }
  };

  SetType* set_ptr = nullptr;
  SetType set(0, absl::Hash<int>(), ReentrantEq{&set_ptr});
  set_ptr = &set;

  set.insert(99);

  // This find will call ReentrantEq(99, 99)
  // which will cause a rehash, moving all elements to a new array and deleting
  // the old one. The outer find loop MUST detect the table_version change and
  // restart, rather than returning a pointer into the deleted array.
  auto found = set.find(99);
  ASSERT_NE(found, set.end());
  EXPECT_EQ(*found, 99);
}

TEST(ReentrantHashSetTest, ConcurrentAccess) {
  absl::Mutex mu;

  // Simulates Python's free-threaded or GIL-releasing behavior where a
  // comparison temporarily drops a lock, allowing other threads to
  // interleave modifications (inserts/erases) before the comparison finishes.
  struct YieldingEq {
    absl::Mutex* mu;
    bool operator()(int a, int b) const ABSL_NO_THREAD_SAFETY_ANALYSIS {
      if ((a + b) % 10 == 0) {
        mu->Unlock();
        tsl::Env::Default()->SleepForMicroseconds(1);
        mu->Lock();
      }
      return a == b;
    }
  };

  ReentrantHashMap<int, int, absl::Hash<int>, YieldingEq> map(
      0, absl::Hash<int>(), YieldingEq{&mu});

  auto worker = [&](int offset) {
    for (int i = 0; i < 1000; ++i) {
      absl::MutexLock lock(&mu);
      int key = (i + offset) % 100;
      if (i % 3 == 0) {
        map.insert(key, i);
      } else if (i % 3 == 1) {
        auto it = map.find(key);
        if (it != map.end()) map.erase(it);
      } else {
        auto it = map.find(key);
        if (it != map.end()) {
          EXPECT_GE(it->second, 0);
        }
      }
    }
  };

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "concurrent_test", 8);
  for (int i = 0; i < 8; ++i) {
    pool.Schedule([worker, i]() { worker(i * 10); });
  }
}

struct BadHash {
  size_t operator()(int) const { return 0; }
};

struct ThrowingEq {
  bool operator()(int a, int b) const {
    if (a == 42 || b == 42) {
      throw std::runtime_error("Eq threw!");
    }
    return a == b;
  }
};

TEST(ReentrantHashSetTest, ExceptionSafety) {
  ReentrantHashSet<int, BadHash, ThrowingEq> set;
  set.insert(1);
  set.insert(2);
  EXPECT_EQ(set.size(), 2);

  EXPECT_THROW(set.insert(42), std::runtime_error);
  EXPECT_EQ(set.size(), 2);  // State unchanged

  EXPECT_THROW(set.find(42), std::runtime_error);
  EXPECT_EQ(set.size(), 2);  // State unchanged

  EXPECT_TRUE(set.insert(3).second);
  EXPECT_EQ(set.size(), 3);
}

TEST(ReentrantHashSetTest, RandomizedCorrectnessTest) {
  ReentrantHashSet<int> set;
  absl::flat_hash_set<int> ref_set;
  absl::BitGen gen;

  for (int i = 0; i < 10000; ++i) {
    int val = absl::Uniform(gen, 0, 1000);
    double action = absl::Uniform(gen, 0.0, 1.0);

    if (action < 0.4) {  // Insert
      auto res1 = set.insert(val);
      auto res2 = ref_set.insert(val);
      EXPECT_EQ(res1.second, res2.second);
    } else if (action < 0.7) {  // Find
      auto it1 = set.find(val);
      auto it2 = ref_set.find(val);
      if (it2 == ref_set.end()) {
        EXPECT_EQ(it1, set.end());
      } else {
        ASSERT_NE(it1, set.end());
        EXPECT_EQ(*it1, *it2);
      }
    } else {  // Erase
      auto it1 = set.find(val);
      auto it2 = ref_set.find(val);
      if (it2 == ref_set.end()) {
        EXPECT_EQ(it1, set.end());
      } else {
        ASSERT_NE(it1, set.end());
        set.erase(it1);
        ref_set.erase(val);
      }
    }
    EXPECT_EQ(set.size(), ref_set.size());
  }
}

}  // namespace
}  // namespace jax
