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

// A ABSL-link open-addressing hash set that is safe to use where
// the equality function may mutate the table or throw exceptions. This code is
// motivated by Python use cases where the hash and equality functions may be
// implemented in Python and may release the GIL, allowing other threads to
// mutate the table. For any other purpose you should use
// absl::flat_hash_set or absl::flat_hash_map.
//
// We assume that mutations from equality functions are improbable, and use
// version counters (`version_`) to detect such modifications and restart any
// in-progress lookups/insertions.
//
// We have the following requirements on the Eq functor:
//  * if Eq directly or indirectly mutates the set (including by releasing a
//    lock and allowing other threads to mutate the set), this invalidates the
//    key reference. Eq should copy any values it needs before any
//    mutations.
//  * Eq may throw exceptions.
//
// We have the following requirements on the Hash functor:
//  * Hash must not mutate the table or release any locks. However, this is
//    an easy requirement to satisfy since one can simply precompute the hash
//    as part of the Key.
//  * Hash must not throw exceptions. But again, this is easy to satisfy
//    since one can simply precompute the hash as part of the Key.
// These requirements are needed for when we resize the table, which requires
// rehashing everything.

#ifndef JAXLIB_REENTRANT_HASH_SET_H_
#define JAXLIB_REENTRANT_HASH_SET_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>  // SSE2
#define JAXLIB_HAVE_SSE2 1
#endif

namespace jax {

// Control byte states
constexpr int8_t kEmpty = -128;   // 0x80
constexpr int8_t kDeleted = -2;   // 0xFE
constexpr int8_t kSentinel = -1;  // 0xFF

inline bool IsEmpty(int8_t c) { return c == kEmpty; }
inline bool IsFull(int8_t c) { return c >= 0; }
inline bool IsDeleted(int8_t c) { return c == kDeleted; }
inline bool IsEmptyOrDeleted(int8_t c) { return c < kSentinel; }

// In the swiss table design, we split our hash into two hash functions. H1
// is used to determine the location of the initial probe in the table, and H2
// is a 7-bit value used for fast SIMD metadata scans.
inline size_t H1(size_t hash) { return hash >> 7; }
inline int8_t H2(size_t hash) { return hash & 0x7F; }

// Bitmask is a helper class used to iterate over the bitmask that results from
// a SIMD Group comparison.
class BitMask {
 public:
  class Iterator {
   public:
    explicit Iterator(uint32_t mask) : mask_(mask) {}
    int operator*() const { return __builtin_ctz(mask_); }
    Iterator& operator++() {
      // mask_ - 1 flips all bits up to and including the lowest set bit.
      // E.g. 0b101100 - 1 = 0b101011
      // ANDing them together clears the lowest set bit.
      // E.g. 0b101100 & 0b101011 = 0b101000
      mask_ &= (mask_ - 1);
      return *this;
    }
    bool operator!=(const Iterator& other) const {
      return mask_ != other.mask_;
    }
    bool operator==(const Iterator& other) const {
      return mask_ == other.mask_;
    }

   private:
    uint32_t mask_;
  };

  explicit BitMask(uint32_t mask) : mask_(mask) {}
  Iterator begin() const { return Iterator(mask_); }
  Iterator end() const { return Iterator(0); }
  bool operator!() const { return mask_ == 0; }
  explicit operator bool() const { return mask_ != 0; }
  int LowestBitSet() const { return __builtin_ctz(mask_); }

 private:
  uint32_t mask_;
};

#if JAXLIB_HAVE_SSE2
class GroupSse2 {
 public:
  static constexpr size_t kWidth = 16;
  explicit GroupSse2(const int8_t* pos) {
    ctrl_ = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pos));
  }

  // Returns a bitmask of the positions where the metadata has value of `hash`.
  BitMask Match(int8_t hash) const {
    auto match = _mm_set1_epi8(hash);
    return BitMask(_mm_movemask_epi8(_mm_cmpeq_epi8(match, ctrl_)));
  }

  // Returns a bitmask of the empty positions in the group.
  BitMask MatchEmpty() const {
    auto match = _mm_set1_epi8(kEmpty);
    return BitMask(_mm_movemask_epi8(_mm_cmpeq_epi8(match, ctrl_)));
  }

  // Returns a bitmask of the positions where the metadata is empty or deleted.
  BitMask MatchEmptyOrDeleted() const {
    auto match = _mm_set1_epi8(kSentinel);
    // x < -1  => Empty(-128) or Deleted(-2).
    // Note: _mm_cmpgt_epi8 does signed comparison.
    // -128 < -1, -2 < -1. So if match(-1) > ctrl, it works!
    return BitMask(_mm_movemask_epi8(_mm_cmpgt_epi8(match, ctrl_)));
  }

 private:
  __m128i ctrl_;
};
using Group = GroupSse2;
#elif defined(JAXLIB_HAVE_NEON)
class GroupNeon {
 public:
  static constexpr size_t kWidth = 8;
  explicit GroupNeon(const int8_t* pos) { ctrl_ = vld1_s8(pos); }

  BitMask Match(int8_t hash) const {
    uint8x8_t match = vceq_s8(ctrl_, vdup_n_s8(hash));
    return BitMask(GetMask(match));
  }

  BitMask MatchEmpty() const {
    uint8x8_t match = vceq_s8(ctrl_, vdup_n_s8(kEmpty));
    return BitMask(GetMask(match));
  }

  BitMask MatchEmptyOrDeleted() const {
    uint8x8_t match = vreinterpret_u8_s8(vclt_s8(ctrl_, vdup_n_s8(0)));
    return BitMask(GetMask(match));
  }

 private:
  uint32_t GetMask(uint8x8_t match) const {
    const uint8_t mask_data[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    uint8x8_t bit_mask = vld1_u8(mask_data);
    uint8x8_t masked = vand_u8(match, bit_mask);
    return vaddv_u8(masked);
  }

  int8x8_t ctrl_;
};
using Group = GroupNeon;
#else
class GroupScalar {
 public:
  static constexpr size_t kWidth = 8;
  explicit GroupScalar(const int8_t* pos) : pos_(pos) {}

  BitMask Match(int8_t hash) const {
    uint32_t mask = 0;
    for (int i = 0; i < 8; ++i) {
      if (pos_[i] == hash) {
        mask |= (1 << i);
      }
    }
    return BitMask(mask);
  }

  BitMask MatchEmpty() const {
    uint32_t mask = 0;
    for (int i = 0; i < 8; ++i) {
      if (pos_[i] == kEmpty) {
        mask |= (1 << i);
      }
    }
    return BitMask(mask);
  }

  BitMask MatchEmptyOrDeleted() const {
    uint32_t mask = 0;
    for (int i = 0; i < 8; ++i) {
      if (IsEmptyOrDeleted(pos_[i])) {
        mask |= (1 << i);
      }
    }
    return BitMask(mask);
  }

 private:
  const int8_t* pos_;
};
using Group = GroupScalar;
#endif

template <typename Key, typename Hash = absl::Hash<Key>,
          typename Eq = std::equal_to<Key>>
class ReentrantHashSet {
 public:
  ReentrantHashSet(size_t capacity = 0, const Hash& hasher = Hash(),
                   const Eq& eq = Eq())
      : hasher_(hasher), eq_(eq) {
    Initialize(capacity);
  }

  ~ReentrantHashSet() {
    clear();
    ::operator delete(static_cast<void*>(slots_));
    delete[] ctrl_;
  }

  ReentrantHashSet(const ReentrantHashSet&) = delete;
  ReentrantHashSet& operator=(const ReentrantHashSet&) = delete;

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  void clear() {
    if (capacity_ == 0) return;
    for (size_t i = 0; i < capacity_; ++i) {
      if (IsFull(ctrl_[i])) {
        slots_[i].~Key();
      }
    }
    size_ = 0;
    num_deleted_ = 0;
    memset(ctrl_, kEmpty, capacity_ + Group::kWidth);
    ctrl_[capacity_] = kSentinel;
    ++version_;
  }

  class const_iterator;

  class iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Key;
    using difference_type = std::ptrdiff_t;
    using pointer = Key*;
    using reference = Key&;

    iterator() : set_(nullptr), idx_(0) {}

    reference operator*() const { return set_->slots_[idx_]; }
    pointer operator->() const { return &set_->slots_[idx_]; }

    iterator& operator++() {
      ++idx_;
      SkipEmpty();
      return *this;
    }
    iterator operator++(int) {
      iterator tmp(*this);
      ++(*this);
      return tmp;
    }

    friend bool operator==(const iterator& a, const iterator& b) {
      return a.idx_ == b.idx_ && a.set_ == b.set_;
    }
    friend bool operator!=(const iterator& a, const iterator& b) {
      return !(a == b);
    }

   private:
    friend class ReentrantHashSet;
    friend class const_iterator;

    iterator(ReentrantHashSet* set, size_t idx) : set_(set), idx_(idx) {
      SkipEmpty();
    }
    void SkipEmpty() {
      if (!set_) return;
      while (idx_ < set_->capacity_ && !IsFull(set_->ctrl_[idx_])) {
        ++idx_;
      }
    }

    ReentrantHashSet* set_;
    size_t idx_;
  };

  class const_iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = const Key;
    using difference_type = std::ptrdiff_t;
    using pointer = const Key*;
    using reference = const Key&;

    const_iterator() : set_(nullptr), idx_(0) {}
    const_iterator(const iterator& it) : set_(it.set_), idx_(it.idx_) {}

    reference operator*() const { return set_->slots_[idx_]; }
    pointer operator->() const { return &set_->slots_[idx_]; }

    const_iterator& operator++() {
      ++idx_;
      SkipEmpty();
      return *this;
    }
    const_iterator operator++(int) {
      const_iterator tmp(*this);
      ++(*this);
      return tmp;
    }

    friend bool operator==(const const_iterator& a, const const_iterator& b) {
      return a.idx_ == b.idx_ && a.set_ == b.set_;
    }
    friend bool operator!=(const const_iterator& a, const const_iterator& b) {
      return !(a == b);
    }

   private:
    friend class ReentrantHashSet;

    const_iterator(const ReentrantHashSet* set, size_t idx)
        : set_(set), idx_(idx) {
      SkipEmpty();
    }
    void SkipEmpty() {
      if (!set_) return;
      while (idx_ < set_->capacity_ && !IsFull(set_->ctrl_[idx_])) {
        ++idx_;
      }
    }

    const ReentrantHashSet* set_;
    size_t idx_;
  };

  iterator begin() { return iterator(this, 0); }
  iterator end() { return iterator(this, capacity_); }
  const_iterator begin() const { return const_iterator(this, 0); }
  const_iterator end() const { return const_iterator(this, capacity_); }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }

  // Returns an iterator to the key if found, end() otherwise.
  template <typename K>
  const_iterator find(const K& key) const {
    size_t hash = hasher_(key);
    while (true) {
      if (capacity_ == 0) {
        return end();
      }

      FindResult res = FindInternal(key, hash);
      if (ABSL_PREDICT_FALSE(res.needs_restart)) {
        continue;
      }
      if (res.found) {
        return const_iterator(this, res.idx);
      }
      return end();
    }
  }

  // Returns an iterator to the key if found, end() otherwise.
  template <typename K>
  iterator find(const K& key) {
    size_t hash = hasher_(key);
    while (true) {
      if (capacity_ == 0) {
        return end();
      }

      FindResult res = FindInternal(key, hash);
      if (ABSL_PREDICT_FALSE(res.needs_restart)) {
        continue;
      }
      if (res.found) {
        return iterator(this, res.idx);
      }
      return end();
    }
  }

  // Returns a pair of the inserted (or existing) key pointer, and a boolean
  // which is true if inserted, false if already exists.
  std::pair<Key*, bool> insert(const Key& key) {
    size_t hash = hasher_(key);
    while (true) {
      if (size_ + num_deleted_ + 1 > capacity_ * max_load_factor_ ||
          capacity_ == 0) {
        size_t new_cap = capacity_ == 0 ? 8 : capacity_;
        if (size_ >= capacity_ / 2) {
          new_cap = capacity_ * 2;
        }
        Rehash(new_cap);
      }

      FindResult res = FindInternal(key, hash);
      if (ABSL_PREDICT_FALSE(res.needs_restart)) {
        continue;
      }
      if (res.found) {
        return {&slots_[res.idx], false};
      }

      if (res.idx == ~size_t{0}) {
        continue;
      }

      if (IsEmptyOrDeleted(ctrl_[res.idx])) {
        bool was_deleted = (ctrl_[res.idx] == kDeleted);
        int8_t h2 = H2(hash);
        ctrl_[res.idx] = h2;
        SetCtrlMirrored(res.idx, h2);
        new (&slots_[res.idx]) Key(key);

        ++size_;
        if (was_deleted) {
          --num_deleted_;
        }
        ++version_;
        return {&slots_[res.idx], true};
      }
    }
  }

  void erase(const_iterator it) {
    CHECK(it.set_ == this);
    CHECK(it.idx_ < capacity_);
    CHECK(IsFull(ctrl_[it.idx_]));
    ctrl_[it.idx_] = kDeleted;
    SetCtrlMirrored(it.idx_, kDeleted);
    slots_[it.idx_].~Key();
    --size_;
    ++num_deleted_;
    ++version_;
  }

  void erase(iterator it) { erase(const_iterator(it)); }

 private:
  struct FindResult {
    // True if the table was mutated during lookup and the operation must
    // restart.
    bool needs_restart;
    // True if the key was found in the table.
    bool found;
    // If 'found' is true, the index of the slot containing the key.
    // If 'found' is false, the index of an available slot for insertion
    // (or ~size_t{0} if no such slot was found).
    size_t idx;
  };

  template <typename K>
  FindResult FindInternal(const K& key, size_t hash) const {
    size_t current_version = version_;
    int8_t h2 = H2(hash);
    size_t curr = H1(hash) & capacity_mask_;
    size_t step = 0;
    size_t target_idx = ~size_t{0};

    while (true) {
      Group g(ctrl_ + curr);
      for (int i : g.Match(h2)) {
        size_t idx = (curr + i) & capacity_mask_;
        const Key& slot = slots_[idx];
        bool is_match = eq_(slot, key);

        if (ABSL_PREDICT_FALSE(version_ != current_version)) {
          return {true, false, ~size_t{0}};
        }

        if (ABSL_PREDICT_TRUE(is_match)) {
          return {false, true, idx};
        }
      }
      if (target_idx == ~size_t{0}) {
        auto match_eod = g.MatchEmptyOrDeleted();
        if (match_eod) {
          target_idx = (curr + match_eod.LowestBitSet()) & capacity_mask_;
        }
      }
      if (g.MatchEmpty()) {
        break;
      }
      // Triangular probing
      step += Group::kWidth;
      curr = (curr + step) & capacity_mask_;
    }
    return {false, false, target_idx};
  }

  void Initialize(size_t capacity) {
    size_ = 0;
    num_deleted_ = 0;
    capacity_ = capacity;
    if (capacity_ == 0) {
      capacity_mask_ = 0;
      ctrl_ = new int8_t[Group::kWidth + 1];
      std::memset(ctrl_, kEmpty, Group::kWidth + 1);
      ctrl_[Group::kWidth] = kSentinel;
      slots_ = nullptr;
    } else {
      capacity_mask_ = capacity_ - 1;
      ctrl_ = new int8_t[capacity_ + Group::kWidth + 1];
      std::memset(ctrl_, kEmpty, capacity_ + Group::kWidth + 1);
      ctrl_[capacity_ + Group::kWidth] = kSentinel;
      slots_ = static_cast<Key*>(::operator new(capacity_ * sizeof(Key)));
    }
    ++version_;
  }

  void SetCtrlMirrored(size_t idx, int8_t h) {
    ctrl_[idx] = h;
    if (idx < Group::kWidth) {
      ctrl_[idx + capacity_] = h;
    }
  }

  void Rehash(size_t new_capacity) {
    if (new_capacity < 8) {
      new_capacity = 8;
    }
    int8_t* old_ctrl = ctrl_;
    Key* old_slots = slots_;
    size_t old_capacity = capacity_;

    Initialize(new_capacity);

    if (old_capacity > 0) {
      for (size_t i = 0; i < old_capacity; ++i) {
        if (IsFull(old_ctrl[i])) {
          size_t hash = hasher_(old_slots[i]);
          int8_t h2 = old_ctrl[i];
          size_t curr = H1(hash) & capacity_mask_;
          size_t step = 0;
          while (true) {
            Group g(ctrl_ + curr);
            auto match_eod = g.MatchEmptyOrDeleted();
            if (match_eod) {
              size_t idx = (curr + match_eod.LowestBitSet()) & capacity_mask_;
              SetCtrlMirrored(idx, h2);
              new (&slots_[idx]) Key(std::move(old_slots[i]));

              size_++;
              break;
            }
            step += Group::kWidth;
            curr = (curr + step) & capacity_mask_;
          }
          old_slots[i].~Key();
        }
      }
    }
    delete[] old_ctrl;
    if (old_slots) {
      ::operator delete(static_cast<void*>(old_slots));
    }
  }

  // The design of this hash table follows the swiss table design. ctrl_
  // is a metadata array where there is one byte per slot, and where the lowest
  // group is mirrored at the end of the array. The sign bit of each entry in
  // ctrl_ is negative if the slot is empty or deleted. If the value is
  // positive, the entry is present and the metadata stores the lowest 7 bits
  // of the hash of the key in the slot (H2). This design allows for SIMD
  // accelerated lookups that scan, say, 16 elements at a time.
  int8_t* ctrl_ = nullptr;

  // The array of slots.
  Key* slots_ = nullptr;

  size_t size_ = 0;
  size_t num_deleted_ = 0;

  // The number of slots allocated in slots_. If 0, slots_ is null, but
  // ctrl_ points to a dummy control array so lookups can safely fail.
  // capacity_ is always a power of 2.
  size_t capacity_ = 0;
  // capacity_mask_ is capacity_ - 1, used for fast modulo operations.
  size_t capacity_mask_ = 0;

  const float max_load_factor_ = 0.875f;

  // Tracks modifications to the hash table (insertions, erasures, rehashes)
  // to detect concurrent mutations during callbacks and prevent ABA
  // and Use-After-Free problems.
  size_t version_ = 0;

  Hash hasher_;
  Eq eq_;
};

}  // namespace jax

#endif  // JAXLIB_REENTRANT_HASH_SET_H_
