/* Copyright 2022 The JAX Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/pytree.h"
#include "jaxlib/reentrant_hash_map.h"

namespace nb = nanobind;

namespace jax {

// A WeakrefLRUCache is implemented as two levels of nested maps, where the
// entries are linked by an intrusive LRU list.
// The outermost level of the cache is a map keyed by weak part of the key, and
// the inner level is a map keyed by strong part of the key. We separate these
// because we need to be able to efficiently evict all entries that relate to
// a particular weak key when the object that it refers to is destroyed.
//
// We use ReentrantHashMap, which is a hash map inspired by absl::flat_hash_map
// that supports two key properties that we need:
// a) it allows for exceptions during equality tests. ABSL does not promise
//    exception safety for its containers, although if this were the only
//    problem we could probably work around it by plumbing exceptions out via
//    the Python interpreter's error state.
// b) it allows for mutations of the table during equality tests. This is the
//    harder problem: it is certainly a violation of the contract of any
//    standard C++ hash map class to do this. Why does this happen? We use a
//    Python lock, such as the GIL, to protect the weakref_lru_cache. However
//    our key equality tests may be arbitrary Python code, which may release the
//    GIL (or the equivalent free-threading critical section) at any time. Hence
//    our data structure must be robust to mutations in the middle of equality
//    tests.
//
//    We achieve this by adding a version number to the hash map and having
//    find() restart if the version changes. This is not sufficient to
//    defend against equality tests that intentionally mutate the hash map: if
//    the equality always mutates the map we would have an endless loop, but
//    it is sufficient to defend against the case of the GIL being released,
//    in which case we don't actually expect a mutation with high probability.
class WeakrefLRUCacheBase;
class WeakrefLRUCache;
class MultiWeakrefLRUCache;

namespace {

struct CacheEntry;

// Returns a well-distributed hash of a Python object.
// Python hashes do not distribute entropy well across all bits of an integer.
// For example, the Python hash of a small integer is the integer itself.
// A good bit distribution is a property that ABSL-style swiss tables depend on,
// so we use absl::HashOf to upgrade a poorly distributed hash function into
// a better distributed one.
inline size_t StrongPythonHash(nb::handle h) {
  return absl::HashOf(nb::hash(h));
}

// PointerWeakKey is a WeakKey that compares by pointer identity, used to find
// a specific weakref object as part of a heterogeneous lookup.
struct PointerWeakKey {
  absl::InlinedVector<nb::weakref, 1> refs;
  size_t cached_hash;
};

// WeakKey is the key to the first level of the table.
struct WeakKey {
  static WeakKey Make(absl::Span<PyObject* const> weakref_args,
                      nb::object weakref_callback) {
    absl::InlinedVector<nb::weakref, 1> refs;
    refs.reserve(weakref_args.size());
    for (PyObject* arg : weakref_args) {
      refs.push_back(
          nb::weakref(nb::borrow<nb::object>(arg), weakref_callback));
    }
    return WeakKey(refs);
  }
  static WeakKey Make(absl::Span<nb::object const> weakref_args,
                      nb::object weakref_callback) {
    absl::InlinedVector<nb::weakref, 1> refs;
    refs.reserve(weakref_args.size());
    for (const nb::object& arg : weakref_args) {
      refs.push_back(nb::weakref(arg, weakref_callback));
    }
    return WeakKey(refs);
  }

  WeakKey(absl::InlinedVector<nb::weakref, 1> refs)
      : refs(std::move(refs)), cached_hash(absl::HashOf(*this)) {}

  absl::InlinedVector<nb::weakref, 1> refs;

  // The contract of ReentrantHashMap does not allow hash functions to release
  // locks, and hence we cannot call back into Python during our hash function.
  // We solve this by precomputing the hash value.
  size_t cached_hash;

  struct SafeEqual {
    // It is important that we take the keys by value not by reference because
    // equal() may release locks, and per the contract of our hash map this may
    // invalidate references.
    bool operator()(WeakKey a, WeakKey b) const {
      if (a.refs.size() != b.refs.size()) {
        return false;
      }
      for (size_t i = 0; i < a.refs.size(); ++i) {
        if (!a.refs[i].equal(b.refs[i])) {
          return false;
        }
      }
      return true;
    }
      bool operator()(WeakKey a, PointerWeakKey b) const {
      if (a.refs.size() != b.refs.size()) {
        return false;
      }
      for (size_t i = 0; i < a.refs.size(); ++i) {
        if (a.refs[i].ptr() != b.refs[i].ptr()) {
          return false;
        }
      }
      return true;
    }
  };

  template <typename H>
  friend H AbslHashValue(H h, const WeakKey& key) {
    for (const auto& ref : key.refs) {
      h = H::combine(std::move(h), nb::hash(ref));
    }
    return h;
  }

  struct CachedHash {
    size_t operator()(WeakKey key) const { return key.cached_hash; }
    size_t operator()(PointerWeakKey key) const { return key.cached_hash; }
  };
};

// StrongKey is the key to the strong part of the table. It has three parts:
// the user-provided context object, and the positional and keyword arguments.
// We store the arguments as the concatenation of the positional arguments and
// the keyword arguments, together with a vector of keyword names. This is
// efficient to construct from the Python vectorcall protocol; we need never
// build a dictionary.
struct PointerStrongKey;
class StrongKey {
 public:
  StrongKey(nb::object context, absl::InlinedVector<nb::object, 2> kwnames,
            absl::InlinedVector<nb::object, 4> args)
      : context_(std::move(context)),
        kwnames_(std::move(kwnames)),
        args_(std::move(args)) {
    cached_hash_ = absl::HashOf(*this);
  }

  bool operator==(const StrongKey& other) const;

  template <typename H>
  friend H AbslHashValue(H h, const StrongKey& key) {
    h = H::combine(std::move(h), nb::hash(key.context_));
    for (const auto& kwname : key.kwnames_) {
      h = H::combine(std::move(h), kwname.ptr());
    }
    for (const auto& arg : key.args_) {
      h = H::combine(std::move(h), nb::hash(arg));
    }
    return h;
  }

  struct SafeEqual {
    // It is important that we take the keys by value not by reference because
    // equal() may release locks, and per the contract of our hash map this may
    // invalidate references.
    bool operator()(StrongKey a, StrongKey b) const { return a == b; }
    bool operator()(StrongKey a, const PointerStrongKey& b) const;
  };

  struct CachedHash {
    size_t operator()(StrongKey key) const { return key.cached_hash_; }
    size_t operator()(const PointerStrongKey& key) const;
  };

  nb::object context() const { return context_; }
  absl::Span<const nb::object> kwnames() const { return kwnames_; }
  absl::Span<const nb::object> args_span() const { return args_; }
  size_t cached_hash() const { return cached_hash_; }

  nb::object args() const;
  nb::object kwargs() const;

  int tp_traverse(visitproc visit, void* arg) const;

 private:
  nb::object context_;

  // Keyword argument names, interned and sorted by pointer.
  absl::InlinedVector<nb::object, 2> kwnames_;

  // Positional arguments followed by keyword arguments. The keyword arguments
  // are stored in the order they appear in kwnames.
  absl::InlinedVector<nb::object, 4> args_;

  // The cached hash value. See the comment on WeakKey.
  size_t cached_hash_;
};

bool StrongKey::operator==(const StrongKey& other) const {
  if (!context_.equal(other.context_)) return false;

  if (kwnames_.size() != other.kwnames_.size()) return false;
  for (size_t i = 0; i < kwnames_.size(); ++i) {
    if (kwnames_[i].ptr() != other.kwnames_[i].ptr()) return false;
  }

  if (args_.size() != other.args_.size()) return false;
  for (size_t i = 0; i < args_.size(); ++i) {
    if (!args_[i].equal(other.args_[i])) return false;
  }
  return true;
}

nb::object StrongKey::args() const {
  size_t num_kwargs = kwnames_.size();
  size_t num_pos = args_.size() - num_kwargs;
  nb::tuple t = nb::steal<nb::tuple>(PyTuple_New(num_pos));
  for (size_t i = 0; i < num_pos; ++i) {
    PyTuple_SET_ITEM(t.ptr(), i, args_[i].inc_ref().ptr());
  }
  return t;
}

nb::object StrongKey::kwargs() const {
  nb::dict d;
  size_t num_kwargs = kwnames_.size();
  size_t num_pos = args_.size() - num_kwargs;
  for (size_t i = 0; i < num_kwargs; ++i) {
    d[kwnames_[i]] = args_[num_pos + i];
  }
  return d;
}

int StrongKey::tp_traverse(visitproc visit, void* arg) const {
  Py_VISIT(context_.ptr());
  for (const auto& kwname : kwnames_) {
    Py_VISIT(kwname.ptr());
  }
  for (const auto& a : args_) {
    Py_VISIT(a.ptr());
  }
  return 0;
}

struct PointerStrongKey {
  nb::object context;
  absl::Span<nb::object const> kwnames;
  absl::Span<nb::object const> args;
  size_t cached_hash;
};

bool StrongKey::SafeEqual::operator()(StrongKey a,
                                      const PointerStrongKey& b) const {
  if (a.context_.ptr() != b.context.ptr()) return false;
  if (a.kwnames_.size() != b.kwnames.size()) return false;
  for (size_t i = 0; i < a.kwnames_.size(); ++i) {
    if (a.kwnames_[i].ptr() != b.kwnames[i].ptr()) return false;
  }
  if (a.args_.size() != b.args.size()) return false;
  for (size_t i = 0; i < a.args_.size(); ++i) {
    if (a.args_[i].ptr() != b.args[i].ptr()) return false;
  }
  return true;
}

size_t StrongKey::CachedHash::operator()(const PointerStrongKey& key) const {
  return key.cached_hash;
}

// The LRU list has the following property:
// The next pointers are not circular, and point to CacheEntrys.
// The prev pointers are circular, and point to LRUNodes. This allows the head
// pointer of the list to be an LRUNode.
struct LRUNode {
  CacheEntry* next = nullptr;
  LRUNode* prev = nullptr;
};

// CacheEntry objects are the values of the weak -> strong -> CacheEntry map.
// They also participate in the LRU list intrusively.
struct CacheEntry {
  // Pointer to the owning cache. Used primarily so we can adjust the lru_size_
  // during Unlink().
  WeakrefLRUCacheBase* parent;

  // Has the thread that was computing this entry finished its work?
  absl::Notification completed;

  // The following two fields are only valid once completed is true.

  // Did we compute a result, or raise an exception?
  bool has_result = false;

  // The result, if we computed one.
  nb::object result;

  // The thread that is computing the entry. This is used to detect reentrant
  // calls to the same cache entry.
  // TODO(phawkins): we could probably just skip this: the user will notice
  // if they write an infinite recursion anyway.
  std::thread::id thread_id = std::this_thread::get_id();

  // The links in the LRU list. Protected by the cache's lock.
  LRUNode lru_node;

  // Identities for eviction mapping
  WeakKey wr_key;
  StrongKey key;

  CacheEntry(WeakrefLRUCacheBase* p, WeakKey wr, StrongKey k)
      : parent(p), wr_key(std::move(wr)), key(std::move(k)) {}

  ~CacheEntry();

  // Is this node linked into the LRU list?
  bool IsLinked() const { return lru_node.prev != nullptr; }

  // Remove the node from the LRU list. Requires holding the cache's lock.
  void Unlink();
};

}  // namespace

class WeakrefLRUCacheBase {
 public:
  WeakrefLRUCacheBase(nb::callable cache_context_fn, nb::callable fn,
                      int64_t maxsize, std::optional<nb::callable> explain);

  virtual ~WeakrefLRUCacheBase() { Clear(); }

  void EvictWeakKey(const WeakKey& search_key);

  // Returns a list of the keys in the cache.
  std::vector<nb::object> GetKeys();

  struct CacheInfo {
    int64_t hits;
    int64_t misses;
    int64_t maxsize;
    int64_t currsize;
  };
  CacheInfo GetCacheInfo() const;

  void Clear();

  nb::object weakref_callback() const { return weakref_callback_; }

 protected:
  friend struct CacheEntry;

  // Python callable, called each time the cache is invoked, whose return value
  // is used to augment the strong key with implicit context.
  nb::callable cache_context_fn_;

  // Function called on cache miss.
  nb::callable fn_;

  std::optional<nb::callable> explain_;

  using Cache = ReentrantHashMap<StrongKey, std::shared_ptr<CacheEntry>,
                                 StrongKey::CachedHash, StrongKey::SafeEqual>;
  using WeakrefCacheValue = std::shared_ptr<Cache>;

  // Map WeakKeys to a Cache, which contains a map of the strong key/value
  // pairs.
  ReentrantHashMap<WeakKey, WeakrefCacheValue, WeakKey::CachedHash,
                   WeakKey::SafeEqual>
      entries_;
  // Maps weakref objects to the strong CacheEntry objects that reference them.
  // Used to evict entries when a weak reference becomes dead.
  absl::flat_hash_map<PyObject*, absl::flat_hash_set<CacheEntry*>>
      reverse_index_;

  // LRU list used for eviction
  int64_t lru_maxsize_;  // Maximum size of the cache in entries.
  int64_t lru_size_{0};  // Current size of the cache in entries.
  LRUNode lru_head_;     // Root of the LRU list.

  // Cache statistics.
  int64_t misses_ = 0;
  int64_t total_queries_ = 0;

  // Callback invoked when a weak reference is cleared. Constructed by Create().
  nb::object weakref_callback_;

  // Helper used by VectorCall.
  PyObject* Call(PyObject* self_obj, absl::Span<PyObject* const> args,
                 Py_ssize_t nargsf, PyObject* kwnames, WeakKey wrcache_key,
                 const StrongKey& key);

  // Evict all references to `dying_weakref_ptr` from the cache.
  void EvictWeakref(PyObject* dying_weakref_ptr);

  // Moves 'node' to the front of the LRU list. Assumes `node` is already
  // linked in the LRU list.
  void MoveToFront(CacheEntry* node);

  // Adds 'node' to the front of the LRU list. Assumes `node` is not already
  // linked in the LRU list.
  void PushFront(CacheEntry* node);

  // Removes the least recently used entry from the cache.
  void EvictLeastRecentlyUsed();

  void RemoveEntryFromReverseIndex(CacheEntry* entry,
                                   PyObject* skip_weakref = nullptr);

  // Helpers that implement GC traversal and clearing for subclasses.
  int TpTraverse(visitproc visit, void* arg);
  void TpClear();
};

CacheEntry::~CacheEntry() {
  if (IsLinked()) {
    Unlink();
  }
  parent->RemoveEntryFromReverseIndex(this);
}

void CacheEntry::Unlink() {
  CHECK(IsLinked());
  lru_node.prev->next = lru_node.next;
  if (lru_node.next) {
    lru_node.next->lru_node.prev = lru_node.prev;
  } else {
    parent->lru_head_.prev = lru_node.prev;
  }
  lru_node.prev = nullptr;
  lru_node.next = nullptr;
  parent->lru_size_--;
}

WeakrefLRUCacheBase::WeakrefLRUCacheBase(nb::callable cache_context_fn,
                                         nb::callable fn, int64_t maxsize,
                                         std::optional<nb::callable> explain)
    : cache_context_fn_(cache_context_fn),
      fn_(fn),
      explain_(explain),
      lru_maxsize_(maxsize) {
  lru_head_.next = nullptr;
  lru_head_.prev = &lru_head_;
}

void WeakrefLRUCacheBase::RemoveEntryFromReverseIndex(CacheEntry* entry,
                                                      PyObject* skip_weakref) {
  for (const nb::weakref& wref : entry->wr_key.refs) {
    PyObject* weakref_ptr = wref.ptr();
    auto rev_it = reverse_index_.find(weakref_ptr);
    if (rev_it != reverse_index_.end()) {
      rev_it->second.erase(entry);
      if (rev_it->second.empty()) {
        reverse_index_.erase(rev_it);
      }
    }
  }
}

void WeakrefLRUCacheBase::MoveToFront(CacheEntry* node) {
  if (node->IsLinked()) {
    node->Unlink();
  }
  PushFront(node);
}

void WeakrefLRUCacheBase::PushFront(CacheEntry* node) {
  CHECK(!node->IsLinked());
  node->lru_node.next = lru_head_.next;
  node->lru_node.prev = &lru_head_;
  if (lru_head_.next) {
    lru_head_.next->lru_node.prev = &node->lru_node;
  } else {
    lru_head_.prev = &node->lru_node;
  }
  lru_head_.next = node;
  ++lru_size_;
}

void WeakrefLRUCacheBase::EvictLeastRecentlyUsed() {
  CacheEntry* tail = lru_head_.prev->prev->next;
  if (tail->IsLinked()) {
    tail->Unlink();
  }

  // Use heterogeneous lookups so we compare objects by pointer identity.
  // This avoids calling Python __eq__ methods which might release the lock.
  PointerWeakKey ptr_wr_key{tail->wr_key.refs, tail->wr_key.cached_hash};
  auto cache_it = entries_.find(ptr_wr_key);
  if (cache_it == entries_.end()) {
    return;
  }
  std::shared_ptr<Cache> cache_ptr = cache_it->second;

  PointerStrongKey ptr_strong_key{
      tail->key.context(), absl::MakeConstSpan(tail->key.kwnames()),
      absl::MakeConstSpan(tail->key.args_span()), tail->key.cached_hash()};
  auto inner_it = cache_ptr->find(ptr_strong_key);
  if (inner_it == cache_ptr->end()) {
    return;
  }

  // To prevent Python object destructors from running (and potentially
  // dropping the lock) *during* the erase operation, we grab an extra
  // reference to the keys and values here. They will be destroyed at the
  // end of this function block, after the map operations are complete.
  WeakKey wr_key_copy = tail->wr_key;
  StrongKey strong_key_copy = tail->key;
  std::shared_ptr<CacheEntry> value_copy = inner_it->second;

  // Now erase from the map. Because we hold references, no Python
  // destruction happens here.
  cache_ptr->erase(inner_it);
  if (cache_ptr->empty()) {
    entries_.erase(cache_it);
  }
}

static nb::object WeakrefKeyToPython(absl::Span<PyObject* const> weakref_args) {
  if (weakref_args.size() == 1) {
    return nb::borrow<nb::object>(weakref_args[0]);
  }
  nb::tuple keys = nb::steal<nb::tuple>(PyTuple_New(weakref_args.size()));
  for (size_t i = 0; i < weakref_args.size(); ++i) {
    PyTuple_SET_ITEM(keys.ptr(), i, weakref_args[i]);
    Py_INCREF(weakref_args[i]);
  }
  return keys;
}

static nb::object WeakrefKeyToPython(
    absl::Span<const nb::weakref> weakref_args) {
  if (weakref_args.size() == 1) {
    return weakref_args[0];
  }
  nb::tuple keys = nb::steal<nb::tuple>(PyTuple_New(weakref_args.size()));
  for (size_t i = 0; i < weakref_args.size(); ++i) {
    nb::object obj = nb::cast(weakref_args[i]);
    PyTuple_SET_ITEM(keys.ptr(), i, obj.inc_ref().ptr());
  }
  return keys;
}

void WeakrefLRUCacheBase::EvictWeakKey(const WeakKey& search_key) {
  auto it = entries_.find(search_key);
  if (it != entries_.end()) {
    auto& [wr_key, cache_ptr] = *it;
    std::vector<std::shared_ptr<CacheEntry>> deferred_deletes;
    deferred_deletes.reserve(cache_ptr->size());
    for (auto& [strong_key, entry_ptr] : *cache_ptr) {
      if (entry_ptr->IsLinked()) {
        entry_ptr->Unlink();
      }

      deferred_deletes.push_back(std::move(entry_ptr));
    }
    cache_ptr->clear();
    // Create temp-var to avoid re-entrant erase.
    auto tmp = std::move(*it);
    entries_.erase(it);
  }
}

void WeakrefLRUCacheBase::EvictWeakref(PyObject* dying_weakref_ptr) {
  auto rev_it = reverse_index_.find(dying_weakref_ptr);
  if (rev_it == reverse_index_.end()) {
    return;
  }

  // We need to move the set because Unlink and modifying reverse_index_
  // will change the collections.
  absl::flat_hash_set<CacheEntry*> entries_to_evict = std::move(rev_it->second);
  reverse_index_.erase(rev_it);

  std::vector<std::shared_ptr<CacheEntry>> deferred_deletes;
  deferred_deletes.reserve(entries_to_evict.size());

  for (CacheEntry* entry : entries_to_evict) {
    if (entry->IsLinked()) {
      entry->Unlink();
    }

    PointerWeakKey ptr_wr_key{entry->wr_key.refs, entry->wr_key.cached_hash};
    auto cache_it = entries_.find(ptr_wr_key);
    if (cache_it != entries_.end()) {
      auto& [wr_key, cache_ptr] = *cache_it;
      PointerStrongKey ptr_strong_key{
          entry->key.context(), absl::MakeConstSpan(entry->key.kwnames()),
          absl::MakeConstSpan(entry->key.args_span()),
          entry->key.cached_hash()};
      auto inner_it = cache_ptr->find(ptr_strong_key);
      if (inner_it != cache_ptr->end()) {
        auto& [strong_key, entry_ptr] = *inner_it;
        deferred_deletes.push_back(std::move(entry_ptr));
        cache_ptr->erase(inner_it);
      }
      if (cache_ptr->empty()) {
        auto tmp = std::move(*cache_it);
        entries_.erase(cache_it);
      }
    }
  }
}

PyObject* WeakrefLRUCacheBase::Call(PyObject* self_obj,
                                    absl::Span<PyObject* const> args,
                                    Py_ssize_t nargsf, PyObject* kwnames,
                                    WeakKey wrcache_key, const StrongKey& key) {
  Py_ssize_t num_pos_args = PyVectorcall_NARGS(nargsf);

  bool inserted = false;
  std::shared_ptr<CacheEntry> entry;
  std::shared_ptr<Cache> cache_ptr;
  std::vector<nb::object> miss_keys;
  nb::object explainer;

  {
    nb::ft_object_guard lock(self_obj);
    ++total_queries_;

    // NOTE: entries_.insert may release the lock and may throw exceptions.
    auto [it_weak, weak_inserted] =
        entries_.insert(wrcache_key, WeakrefCacheValue());

    if (weak_inserted) {
      it_weak->second = std::make_shared<Cache>();
    } else {
      wrcache_key = it_weak->first;
    }
    // We need to make sure the Cache remains alive as long as this code
    // block. Also, we must drop it safely under the lock because its
    // destruction destroys CacheEntries which call Unlink() on the LRU list.
    cache_ptr = it_weak->second;
    Cache& cache = *cache_ptr;

    // NOTE: cache.insert may release the lock and may throw exceptions.
    auto [it_strong, strong_inserted] = cache.insert(key, nullptr);

    if (strong_inserted) {
      inserted = true;
      ++misses_;
      entry = std::make_shared<CacheEntry>(this, wrcache_key, key);
      it_strong->second = entry;
      PushFront(entry.get());

      for (const nb::weakref& wref : wrcache_key.refs) {
        reverse_index_[wref.ptr()].insert(entry.get());
      }

      if (lru_maxsize_ > 0 && lru_size_ > lru_maxsize_) {
        // Note: EvictLeastRecentlyUsed may release the lock and may throw
        // exceptions.
        EvictLeastRecentlyUsed();
      }

      if (explain_.has_value()) {
        explainer = (*explain_)();
        if (!explainer.is_none()) {
          miss_keys = GetKeys();
        } else {
          explainer = nb::object();
        }
      }
    } else {
      entry = it_strong->second;
      if (entry->IsLinked()) {
        MoveToFront(entry.get());
      } else {
        PushFront(entry.get());
      }
    }
  }

  // We must ensure that entry is destroyed under the lock so that if it is
  // the last reference, its destructor ~CacheEntry runs under the lock.
  // There are various scenarios where our copy of entry ends up as the last
  // owner of the entry, e.g., consider if a thread races to insert an entry
  // at the same time another thread calls Clear().
  absl::Cleanup destroy_entry_under_lock = [self_obj, &entry, &cache_ptr]() {
    nb::ft_object_guard lock(self_obj);
    entry.reset();
    cache_ptr.reset();
  };

  if (!entry->completed.HasBeenNotified()) {
    if (inserted) {
      // explainer and fn_ may throw, so we use an absl::Cleanup to ensure
      // entry->completed is always notified.
      absl::Cleanup notify = [&] { entry->completed.Notify(); };

      if (explainer) {
        nb::object py_miss_keys = nb::cast(miss_keys);

        absl::InlinedVector<PyObject*, 4> explainer_call_args;
        explainer_call_args.reserve(args.size() + 1);
        explainer_call_args.push_back(py_miss_keys.ptr());
        for (size_t i = 0; i < args.size(); ++i) {
          explainer_call_args.push_back(args[i]);
        }

        PyObject* exp =
            PyObject_Vectorcall(explainer.ptr(), explainer_call_args.data(),
                                num_pos_args + 1, kwnames);
        if (!exp) return nullptr;
        Py_DECREF(exp);
      }

      // Call fn_ with original args (weakref_key, *args, **kwargs)
      nb::object result = nb::steal<nb::object>(
          PyObject_Vectorcall(fn_.ptr(), args.data(), nargsf, kwnames));

      if (!result) return nullptr;

      entry->result = result;
      entry->has_result = true;
    } else {
      if (entry->thread_id == std::this_thread::get_id()) {
        nb::object repr_obj = WeakrefKeyToPython(wrcache_key.refs);
        auto error_string = absl::StrCat(
            "Recursively calling ", nb::cast<std::string>(nb::repr(repr_obj)));
        PyErr_SetString(PyExc_RecursionError, error_string.c_str());
        return nullptr;
      }
      nb::gil_scoped_release release;
      entry->completed.WaitForNotification();
    }
  }

  if (entry->has_result) {
    return entry->result.inc_ref().ptr();
  } else {
    // There was an error when computing fn_, so give up on caching and rerun
    // it.
    return PyObject_Vectorcall(fn_.ptr(), args.data(), nargsf, kwnames);
  }
}

std::vector<nb::object> WeakrefLRUCacheBase::GetKeys() {
  std::vector<nb::object> results;
  for (const auto& kv : entries_) {
    const WeakKey& wr_key = kv.first;
    for (const auto& inner_kv : *kv.second) {
      const StrongKey& key = inner_kv.first;
      const std::shared_ptr<CacheEntry>& value = inner_kv.second;
      if (!value->completed.HasBeenNotified()) {
        continue;
      }

      nb::object wr_key_obj = WeakrefKeyToPython(wr_key.refs);

      nb::tuple result =
          nb::make_tuple(wr_key_obj, key.context(), key.args(), key.kwargs());
      results.push_back(std::move(result));
    }
  }
  return results;
}

WeakrefLRUCacheBase::CacheInfo WeakrefLRUCacheBase::GetCacheInfo() const {
  CacheInfo result;
  result.hits = total_queries_ - misses_;
  result.misses = misses_;
  result.maxsize = lru_maxsize_;
  result.currsize = lru_size_;
  return result;
}

void WeakrefLRUCacheBase::Clear() {
  std::vector<std::pair<StrongKey, std::shared_ptr<CacheEntry>>>
      deferred_deletes;
  deferred_deletes.reserve(entries_.size());

  for (auto& kv : entries_) {
    for (auto& inner_kv : *(kv.second)) {
      if (inner_kv.second->IsLinked()) {
        inner_kv.second->Unlink();
      }
      if (inner_kv.second->IsLinked()) {
        inner_kv.second->Unlink();
      }
      deferred_deletes.emplace_back(inner_kv.first, inner_kv.second);
    }
    kv.second->clear();
  }
  entries_.clear();
  deferred_deletes.clear();
  reverse_index_.clear();

  total_queries_ = misses_ = 0;
}

int WeakrefLRUCacheBase::TpTraverse(visitproc visit, void* arg) {
  Py_VISIT(cache_context_fn_.ptr());
  Py_VISIT(fn_.ptr());
  if (explain_) {
    Py_VISIT(explain_->ptr());
  }
  Py_VISIT(weakref_callback_.ptr());

  for (auto& [wr_key, wr_value] : entries_) {
    for (const nb::weakref& wref : wr_key.refs) {
      Py_VISIT(wref.ptr());
    }
    for (auto& [strong_key, value] : *wr_value) {
      int status = strong_key.tp_traverse(visit, arg);
      if (status != 0) return status;

      // We only visit the value and do not visit the backreferences to the
      // keys, since we must have visited them above.
      Py_VISIT(value->result.ptr());
    }
  }

  return 0;
}

void WeakrefLRUCacheBase::TpClear() {
  Clear();
  cache_context_fn_.reset();
  fn_.reset();
  explain_ = std::nullopt;
  weakref_callback_.reset();
}

// WeakrefLRUCache is a cache where the first `num_weak_keys` positional
// arguments should be treated as weak, and the remaining positional and
// keyword arguments should be treated as strong.
class WeakrefLRUCache : public WeakrefLRUCacheBase {
 public:
  static nb_class_ptr<WeakrefLRUCache> Create(
      nb::callable cache_context_fn, nb::callable fn, int64_t maxsize,
      std::optional<nb::callable> explain, int num_weak_keys);

  WeakrefLRUCache(nb::callable cache_context_fn, nb::callable fn,
                  int64_t maxsize, std::optional<nb::callable> explain,
                  int num_weak_keys)
      : WeakrefLRUCacheBase(cache_context_fn, fn, maxsize, explain),
        num_weak_keys_(num_weak_keys) {}

  static PyObject* VectorCall(PyObject* self_obj, PyObject* const* args,
                              Py_ssize_t nargsf, PyObject* kwnames);

  int num_weak_keys() const { return num_weak_keys_; }

  static PyType_Slot slots_[];

 private:
  // Number of positional arguments to treat as weak.
  const int num_weak_keys_;

  static int tp_traverse(PyObject* self, visitproc visit, void* arg);
  static int tp_clear(PyObject* self);
};

/*static*/ nb_class_ptr<WeakrefLRUCache> WeakrefLRUCache::Create(
    nb::callable cache_context_fn, nb::callable fn, int64_t maxsize,
    std::optional<nb::callable> explain, int num_weak_keys) {
  auto self = make_nb_class<WeakrefLRUCache>(cache_context_fn, fn, maxsize,
                                             explain, num_weak_keys);
  self->weakref_callback_ = nb::cpp_function(
      [this_weak = nb::weakref(self)](nb::handle dying_weakref) {
        nb::object py_cache = this_weak();
        if (py_cache.is_none()) {
          return;
        }
        nb::ft_object_guard lock(py_cache);
        nb::cast<WeakrefLRUCache*>(py_cache)->EvictWeakref(dying_weakref.ptr());
      });
  return self;
}

PyObject* WeakrefLRUCache::VectorCall(PyObject* self_obj, PyObject* const* args,
                                      Py_ssize_t nargsf, PyObject* kwnames) {
  try {
    WeakrefLRUCache* self = nb::inst_ptr<WeakrefLRUCache>(self_obj);
    if (!self) {
      PyErr_SetString(PyExc_TypeError, "Not a WeakrefLRUCache");
      return nullptr;
    }

    Py_ssize_t num_pos_args = PyVectorcall_NARGS(nargsf);
    if (num_pos_args < self->num_weak_keys()) {
      PyErr_SetString(PyExc_TypeError,
                      absl::StrCat("Missing weakref_key argument(s). Expected ",
                                   self->num_weak_keys())
                          .c_str());
      return nullptr;
    }

    nb::object context = self->cache_context_fn_();

    Py_ssize_t num_kwargs = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    size_t num_args = num_pos_args + num_kwargs;
    absl::Span<PyObject* const> args_span = absl::MakeConstSpan(args, num_args);

    WeakKey wrcache_key = WeakKey::Make(
        args_span.subspan(0, self->num_weak_keys()), self->weakref_callback_);

    absl::InlinedVector<nb::object, 2> sorted_kwnames;
    absl::InlinedVector<nb::object, 4> strong_args;

    strong_args.reserve(num_args);
    for (Py_ssize_t i = self->num_weak_keys(); i < num_pos_args; ++i) {
      strong_args.push_back(nb::borrow<nb::object>(args[i]));
    }

    if (num_kwargs > 0) {
      // Intern the keyword argument names, then sort them by interned pointer
      // order.
      absl::InlinedVector<std::pair<PyObject*, PyObject*>, 4> sorted;
      sorted.reserve(num_kwargs);
      for (Py_ssize_t i = 0; i < num_kwargs; ++i) {
        PyObject* p = PyTuple_GET_ITEM(kwnames, i);
        Py_INCREF(p);
        PyUnicode_InternInPlace(&p);
        sorted.push_back({p, args[num_pos_args + i]});
      }
      absl::c_sort(sorted, [](const std::pair<PyObject*, PyObject*>& a,
                              const std::pair<PyObject*, PyObject*>& b) {
        return a.first < b.first;
      });

      sorted_kwnames.reserve(num_kwargs);
      for (auto& info : sorted) {
        sorted_kwnames.push_back(nb::steal<nb::object>(info.first));
        strong_args.push_back(nb::borrow<nb::object>(info.second));
      }
    }

    StrongKey key(std::move(context), std::move(sorted_kwnames),
                  std::move(strong_args));
    return self->Call(self_obj, args_span, nargsf, kwnames,
                      std::move(wrcache_key), key);
  } catch (nb::python_error& e) {
    e.restore();
    return nullptr;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

/*static*/ int WeakrefLRUCache::tp_traverse(PyObject* self_obj, visitproc visit,
                                            void* arg) {
  Py_VISIT(Py_TYPE(self_obj));
  if (!nb::inst_ready(self_obj)) {
    return 0;
  }
  WeakrefLRUCache* self = nb::inst_ptr<WeakrefLRUCache>(self_obj);
  return self->TpTraverse(visit, arg);
}

/*static*/ int WeakrefLRUCache::tp_clear(PyObject* self_obj) {
  WeakrefLRUCache* self = nb::inst_ptr<WeakrefLRUCache>(self_obj);
  self->TpClear();
  return 0;
}

static PyMethodDef call_def = {
    "__call__", reinterpret_cast<PyCFunction>(WeakrefLRUCache::VectorCall),
    METH_FASTCALL | METH_KEYWORDS, "Calls the cache."};

/* static */ PyType_Slot WeakrefLRUCache::slots_[] = {
    {Py_tp_traverse, (void*)WeakrefLRUCache::tp_traverse},
    {Py_tp_clear, (void*)WeakrefLRUCache::tp_clear},
    {0, nullptr},
};

// A cache where arguments are treated as pytrees, and amongst the pytree
// leaves arguments of a set of specified types are treated as weak.
class MultiWeakrefLRUCache : public WeakrefLRUCacheBase {
 public:
  // `registry` is the pytree registry to use when flattening arguments.
  // `weak_types` is an iterable of types to treat as weak.
  static nb_class_ptr<MultiWeakrefLRUCache> Create(
      nb::callable cache_context_fn, nb::callable fn, int64_t maxsize,
      std::optional<nb::callable> explain, nb::object registry,
      nb::set weak_types);

  MultiWeakrefLRUCache(nb::callable cache_context_fn, nb::callable fn,
                       int64_t maxsize, std::optional<nb::callable> explain,
                       nb::object registry, nb::set weak_types)
      : WeakrefLRUCacheBase(cache_context_fn, fn, maxsize, explain),
        registry_(std::move(registry)),
        weak_types_(std::move(weak_types)) {}

  static PyObject* VectorCall(PyObject* self_obj, PyObject* const* args,
                              Py_ssize_t nargsf, PyObject* kwnames);

  static PyType_Slot slots_[];

 private:
  nb::object registry_;
  nb::set weak_types_;

  static int tp_traverse(PyObject* self_obj, visitproc visit, void* arg);
  static int tp_clear(PyObject* self_obj);
};

/*static*/ nb_class_ptr<MultiWeakrefLRUCache> MultiWeakrefLRUCache::Create(
    nb::callable cache_context_fn, nb::callable fn, int64_t maxsize,
    std::optional<nb::callable> explain, nb::object registry,
    nb::set weak_types) {
  auto self = make_nb_class<MultiWeakrefLRUCache>(
      std::move(cache_context_fn), std::move(fn), maxsize, std::move(explain),
      std::move(registry), std::move(weak_types));
  self->weakref_callback_ = nb::cpp_function(
      [this_weak = nb::weakref(self)](nb::handle dying_weakref) {
        nb::object py_cache = this_weak();
        if (py_cache.is_none()) {
          return;
        }
        nb::ft_object_guard lock(py_cache);
        nb::cast<MultiWeakrefLRUCache*>(py_cache)->EvictWeakref(
            dying_weakref.ptr());
      });
  return self;
}

PyObject* MultiWeakrefLRUCache::VectorCall(PyObject* self_obj,
                                           PyObject* const* args,
                                           Py_ssize_t nargsf,
                                           PyObject* kwnames) {
  try {
    MultiWeakrefLRUCache* self = nb::inst_ptr<MultiWeakrefLRUCache>(self_obj);
    if (!self) {
      PyErr_SetString(PyExc_TypeError, "Not a MultiWeakrefLRUCache");
      return nullptr;
    }

    Py_ssize_t num_pos_args = PyVectorcall_NARGS(nargsf);
    size_t num_kwargs = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    size_t num_args = num_pos_args + num_kwargs;
    absl::Span<PyObject* const> args_span = absl::MakeConstSpan(args, num_args);

    nb::object context = self->cache_context_fn_();

    absl::InlinedVector<nb::object, 4> weak_leaves;
    absl::InlinedVector<nb::object, 2> sorted_kwnames;
    absl::InlinedVector<nb::object, 4> strong_args;
    strong_args.reserve(num_args);

    nb_class_ptr<PyTreeRegistry> registry =
        nb::borrow<nb_class_ptr<PyTreeRegistry>>(self->registry_);

    auto process_leaf = [&](nb::object leaf) {
      if (self->weak_types_.contains(leaf.type()) ||
          PyCallable_Check(leaf.ptr())) {
        weak_leaves.push_back(std::move(leaf));
      } else {
        strong_args.push_back(std::move(leaf));
      }
    };

    for (size_t i = 0; i < num_pos_args; ++i) {
      auto [leaves, treedef] =
          PyTreeDef::Flatten(nb::handle(args[i]), registry);
      strong_args.push_back(std::move(treedef));
      for (auto& leaf : leaves) {
        process_leaf(leaf);
      }
    }

    if (num_kwargs > 0) {
      // Intern the keyword argument names, then sort them by interned pointer
      // order.
      absl::InlinedVector<std::pair<nb::object, PyObject*>, 4> sorted_kwargs;
      sorted_kwargs.reserve(num_kwargs);
      for (size_t i = 0; i < num_kwargs; ++i) {
        PyObject* name = PyTuple_GET_ITEM(kwnames, i);
        Py_INCREF(name);
        PyUnicode_InternInPlace(&name);
        sorted_kwargs.push_back(
            {nb::steal<nb::object>(name), args[num_pos_args + i]});
      }
      absl::c_sort(sorted_kwargs, [](const auto& a, const auto& b) {
        return a.first < b.first;
      });

      sorted_kwnames.reserve(num_kwargs);
      for (auto& [name, val] : sorted_kwargs) {
        sorted_kwnames.push_back(std::move(name));
        auto [leaves, treedef] = PyTreeDef::Flatten(nb::handle(val), registry);
        strong_args.push_back(std::move(treedef));
        for (auto& leaf : leaves) {
          process_leaf(leaf);
        }
      }
    }

    WeakKey wrcache_key = WeakKey::Make(weak_leaves, self->weakref_callback_);
    StrongKey key(context, std::move(sorted_kwnames), std::move(strong_args));

    return self->Call(self_obj, args_span, nargsf, kwnames,
                      std::move(wrcache_key), key);
  } catch (nb::python_error& e) {
    e.restore();
    return nullptr;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}
/*static*/ int MultiWeakrefLRUCache::tp_traverse(PyObject* self_obj,
                                                 visitproc visit, void* arg) {
  Py_VISIT(Py_TYPE(self_obj));
  if (!nb::inst_ready(self_obj)) {
    return 0;
  }
  MultiWeakrefLRUCache* self = nb::inst_ptr<MultiWeakrefLRUCache>(self_obj);
  Py_VISIT(self->registry_.ptr());
  Py_VISIT(self->weak_types_.ptr());
  return self->TpTraverse(visit, arg);
}

/*static*/ int MultiWeakrefLRUCache::tp_clear(PyObject* self_obj) {
  MultiWeakrefLRUCache* self = nb::inst_ptr<MultiWeakrefLRUCache>(self_obj);
  self->TpClear();
  self->registry_.reset();
  self->weak_types_.reset();
  return 0;
}

/* static */ PyType_Slot MultiWeakrefLRUCache::slots_[] = {
    {Py_tp_traverse, (void*)MultiWeakrefLRUCache::tp_traverse},
    {Py_tp_clear, (void*)MultiWeakrefLRUCache::tp_clear},
    {0, nullptr},
};

static PyMethodDef multi_call_def = {
    "__call__", reinterpret_cast<PyCFunction>(MultiWeakrefLRUCache::VectorCall),
    METH_FASTCALL | METH_KEYWORDS, "Calls the multi-weakref cache."};

NB_MODULE(weakref_lru_cache, m) {
  auto weakref_lru_cache =
      nb::class_<WeakrefLRUCache>(m, "WeakrefLRUCache",
                                  nb::is_weak_referenceable(),
                                  nb::type_slots(WeakrefLRUCache::slots_))
          .def(
              "evict_weakref",
              [](WeakrefLRUCache& cache, nb::object weakref_key) {
                if (cache.num_weak_keys() == 1) {
                  PyObject* ptr = weakref_key.ptr();
                  cache.EvictWeakKey(
                      WeakKey::Make({&ptr, 1}, cache.weakref_callback()));
                } else {
                  if (!nb::isinstance<nb::tuple>(weakref_key)) {
                    PyErr_SetString(PyExc_TypeError,
                                    "evict_weakref expects a tuple of weak "
                                    "keys for multi-weakref cache");
                    return;
                  }
                  nb::tuple t = nb::cast<nb::tuple>(weakref_key);
                  if (t.size() != cache.num_weak_keys()) {
                    PyErr_SetString(PyExc_ValueError,
                                    "Incorrect number of weak keys");
                    return;
                  }
                  absl::InlinedVector<PyObject*, 2> ptrs;
                  for (auto item : t) {
                    ptrs.push_back(item.ptr());
                  }
                  cache.EvictWeakKey(
                      WeakKey::Make(ptrs, cache.weakref_callback()));
                }
              },
              nb::lock_self())
          .def("cache_keys", &WeakrefLRUCache::GetKeys, nb::lock_self())
          .def("cache_info", &WeakrefLRUCache::GetCacheInfo, nb::lock_self())
          .def("cache_clear", &WeakrefLRUCache::Clear, nb::lock_self());

  weakref_lru_cache.attr("__call__") = nb::steal<nb::object>(PyDescr_NewMethod(
      reinterpret_cast<PyTypeObject*>(weakref_lru_cache.ptr()), &call_def));

  nb::class_<WeakrefLRUCache::CacheInfo>(weakref_lru_cache,
                                         "WeakrefLRUCacheInfo")
      .def_ro("hits", &WeakrefLRUCache::CacheInfo::hits)
      .def_ro("misses", &WeakrefLRUCache::CacheInfo::misses)
      .def_ro("maxsize", &WeakrefLRUCache::CacheInfo::maxsize)
      .def_ro("currsize", &WeakrefLRUCache::CacheInfo::currsize)
      .def("__repr__", [](WeakrefLRUCache::CacheInfo& info) {
        return absl::StrCat(
            "WeakrefLRUCache(hits=", info.hits, ", misses=", info.misses,
            ", maxsize=", info.maxsize, ", currsize=", info.currsize, ")");
      });

  m.def(
      "weakref_lru_cache",
      [](nb::callable cache_context_fn, nb::callable fn,
         std::optional<int64_t> maxsize, std::optional<nb::callable> explain,
         int num_weakrefs) {
        return WeakrefLRUCache::Create(std::move(cache_context_fn),
                                       std::move(fn), maxsize.value_or(-1),
                                       std::move(explain), num_weakrefs);
      },
      nb::arg("cache_context_fn"), nb::arg("fn"),
      nb::arg("maxsize").none() = 2048,
      nb::arg("explain") = std::optional<nb::callable>(),
      nb::arg("num_weakrefs") = 1);

  auto multi_weakref_lru_cache =
      nb::class_<MultiWeakrefLRUCache>(
          m, "MultiWeakrefLRUCache", nb::is_weak_referenceable(),
          nb::type_slots(MultiWeakrefLRUCache::slots_))
          .def("cache_keys", &MultiWeakrefLRUCache::GetKeys, nb::lock_self())
          .def("cache_info", &MultiWeakrefLRUCache::GetCacheInfo,
               nb::lock_self())
          .def("cache_clear", &MultiWeakrefLRUCache::Clear, nb::lock_self());

  multi_weakref_lru_cache.attr("__call__") =
      nb::steal<nb::object>(PyDescr_NewMethod(
          reinterpret_cast<PyTypeObject*>(multi_weakref_lru_cache.ptr()),
          &multi_call_def));

  m.def(
      "multi_weakref_lru_cache",
      [](nb::callable cache_context_fn, nb::callable fn,
         std::optional<int64_t> maxsize, std::optional<nb::callable> explain,
         nb::object registry, nb::set weak_types) {
        return MultiWeakrefLRUCache::Create(
            std::move(cache_context_fn), std::move(fn), maxsize.value_or(-1),
            std::move(explain), std::move(registry), std::move(weak_types));
      },
      nb::arg("cache_context_fn"), nb::arg("fn"),
      nb::arg("maxsize").none() = 2048,
      nb::arg("explain") = std::optional<nb::callable>(), nb::arg("registry"),
      nb::arg("weak_types"));
}

}  // namespace jax
