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
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/reentrant_hash_map.h"
#include "xla/tsl/platform/logging.h"

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
class WeakrefLRUCache;

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

// WeakKey is the key to the first level of the table.
struct WeakKey {
  WeakKey(nb::weakref ref, size_t cached_hash)
      : ref(std::move(ref)), cached_hash(cached_hash) {}

  nb::weakref ref;

  // The contract of ReentrantHashMap does not allow hash functions to release
  // locks, and hence we cannot call back into Python during our hash function.
  // We solve this by precomputing the hash value.
  size_t cached_hash;

  struct SafeEqual {
    // It is important that we take the keys by value not by reference because
    // equal() may release locks, and per the contract of our hash map this may
    // invalidate references.
    bool operator()(WeakKey a, WeakKey b) const { return a.ref.equal(b.ref); }
  };

  struct CachedHash {
    size_t operator()(WeakKey key) const { return key.cached_hash; }
  };
};

// StrongKey is the key to the strong part of the table. It has three parts:
// the user-provided context object, and the positional and keyword arguments.
// We store the arguments as the concatenation of the positional arguments and
// the keyword arguments, together with a vector of keyword names. This is
// efficient to construct from the Python vectorcall protocol; we need never
// build a dictionary.
class StrongKey {
 public:
  StrongKey(nb::object context, absl::Span<PyObject* const> args,
            nb::tuple kwnames);

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
  };

  struct CachedHash {
    size_t operator()(StrongKey key) const { return key.cached_hash_; }
  };

  nb::object context() const { return context_; }
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

StrongKey::StrongKey(nb::object context, absl::Span<PyObject* const> args,
                     nb::tuple kwnames)
    : context_(std::move(context)) {
  size_t num_kwargs = kwnames.is_valid() ? kwnames.size() : 0;
  CHECK_GE(args.size(), num_kwargs);
  size_t num_pos = args.size() - num_kwargs;
  args_.reserve(args.size());
  for (size_t i = 0; i < num_pos; ++i) {
    args_.push_back(nb::borrow<nb::object>(args[i]));
  }

  if (num_kwargs > 0) {
    // Intern the keyword argument names, then sort them by interned pointer
    // order.
    absl::InlinedVector<std::pair<PyObject*, PyObject*>, 4> sorted;
    sorted.reserve(num_kwargs);
    for (size_t i = 0; i < num_kwargs; ++i) {
      PyObject* p = PyTuple_GET_ITEM(kwnames.ptr(), i);
      Py_INCREF(p);
      PyUnicode_InternInPlace(&p);
      sorted.push_back({p, args[num_pos + i]});
    }
    absl::c_sort(sorted, [](const std::pair<PyObject*, PyObject*>& a,
                            const std::pair<PyObject*, PyObject*>& b) {
      return a.first < b.first;
    });

    kwnames_.reserve(num_kwargs);
    for (auto& info : sorted) {
      kwnames_.push_back(nb::steal<nb::object>(info.first));
      args_.push_back(nb::borrow<nb::object>(info.second));
    }
  }
  cached_hash_ = absl::HashOf(*this);
}

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
  WeakrefLRUCache* parent;

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

  CacheEntry(WeakrefLRUCache* p, WeakKey wr, StrongKey k)
      : parent(p), wr_key(std::move(wr)), key(std::move(k)) {}

  ~CacheEntry();

  // Is this node linked into the LRU list?
  bool IsLinked() const { return lru_node.prev != nullptr; }

  // Remove the node from the LRU list. Requires holding the cache's lock.
  void Unlink();
};

}  // namespace

class WeakrefLRUCache : public std::enable_shared_from_this<WeakrefLRUCache> {
 public:
  WeakrefLRUCache(nb::callable cache_context_fn, nb::callable fn,
                  int64_t maxsize, std::optional<nb::callable> explain)
      : cache_context_fn_(cache_context_fn),
        fn_(fn),
        explain_(explain),
        lru_maxsize_(maxsize) {
    lru_head_.next = nullptr;
    lru_head_.prev = &lru_head_;
  }

  ~WeakrefLRUCache() { Clear(); }

  static PyObject* VectorCall(PyObject* self_obj, PyObject* const* args,
                              Py_ssize_t nargsf, PyObject* kwnames);

  PyObject* Call(PyObject* self_obj, absl::Span<PyObject* const> args,
                 Py_ssize_t nargsf, PyObject* kwnames);

  void EvictWeakref(const WeakKey& search_key);

  std::vector<nb::object> GetKeys();

  struct CacheInfo {
    int64_t hits;
    int64_t misses;
    int64_t maxsize;
    int64_t currsize;
  };
  CacheInfo GetCacheInfo() const;

  void Clear();

  static PyType_Slot slots_[];

 private:
  friend struct CacheEntry;
  using Cache = ReentrantHashMap<StrongKey, std::shared_ptr<CacheEntry>,
                                 StrongKey::CachedHash, StrongKey::SafeEqual>;

  struct WeakrefCacheValue {
    std::shared_ptr<Cache> cache;
  };

  WeakKey MakeWeakrefKey(const nb::object& weakref_key);

  nb::callable cache_context_fn_;
  nb::callable fn_;
  std::optional<nb::callable> explain_;
  ReentrantHashMap<WeakKey, WeakrefCacheValue, WeakKey::CachedHash,
                   WeakKey::SafeEqual>
      entries_;
  int64_t misses_ = 0;
  int64_t total_queries_ = 0;

  int64_t lru_maxsize_;
  int64_t lru_size_{0};
  LRUNode lru_head_;

  void MoveToFront(CacheEntry* node) {
    node->Unlink();
    PushFront(node);
  }

  void PushFront(CacheEntry* node) {
    CHECK(!node->IsLinked());
    node->lru_node.next = lru_head_.next;
    node->lru_node.prev = &lru_head_;
    if (lru_head_.next) {
      lru_head_.next->lru_node.prev = &node->lru_node;
    }
    lru_head_.next = node;
    ++lru_size_;
  }

  void EvictLeastRecentlyUsed() {
    if (!lru_head_.next) return;
    CacheEntry* tail = lru_head_.prev->prev->next;
    auto cache_it = entries_.find(tail->wr_key);

    tail->Unlink();

    // Now erase from the map, which may trigger deletion if no other references
    // exist
    if (cache_it != entries_.end()) {
      auto inner_it = cache_it->second.cache->find(tail->key);
      if (inner_it != cache_it->second.cache->end()) {
        cache_it->second.cache->erase(inner_it);
      }
    }
  }

  static int tp_traverse(PyObject* self, visitproc visit, void* arg);
  static int tp_clear(PyObject* self);
};

CacheEntry::~CacheEntry() {
  if (IsLinked()) {
    Unlink();
  }
}

void CacheEntry::Unlink() {
  CHECK(IsLinked());
  lru_node.prev->next = lru_node.next;
  if (lru_node.next) {
    lru_node.next->lru_node.prev = lru_node.prev;
  }
  lru_node.prev = nullptr;
  lru_node.next = nullptr;
  parent->lru_size_--;
}

WeakKey WeakrefLRUCache::MakeWeakrefKey(const nb::object& weakref_key) {
  size_t wrcache_hash = StrongPythonHash(weakref_key);

  auto weakref_gc_callback = nb::cpp_function(
      [this_weak = weak_from_this(), wrcache_hash](nb::handle weakref) {
        // We are careful to use a weak reference to the cache object here to
        // avoid the following reference cycle: the cache holds weakref objects
        // as its keys, and weakrefs, despite having "weak" in their name,
        // hold a strong reference to their callbacks. This would be a strong
        // reference cycle.
        auto cache = this_weak.lock();
        if (cache == nullptr) {
          return;
        }
        auto py_cache = nb::find(cache);
        // This should never happen as python cache should always be found
        CHECK(py_cache.ptr() != nullptr);
        nb::ft_object_guard lock(py_cache);

        // The object the reference referred to is now in the process of being
        // destroyed, so we cannot refer to its contents. Python weakref
        // objects compare based on identity if the object they refer to is
        // gone, so the hash lookup will work fine.
        WeakKey search_key(nb::borrow<nb::weakref>(weakref), wrcache_hash);
        cache->EvictWeakref(search_key);
      });
  return WeakKey(nb::weakref(weakref_key, weakref_gc_callback), wrcache_hash);
}

void WeakrefLRUCache::EvictWeakref(const WeakKey& search_key) {
  auto it = entries_.find(search_key);
  if (it != entries_.end()) {
    for (auto& inner_kv : *(it->second.cache)) {
      inner_kv.second->Unlink();
    }
    // Create temp-var to avoid re-entrant erase.
    auto tmp = std::move(*it);
    entries_.erase(it);
  }
}

PyObject* WeakrefLRUCache::VectorCall(PyObject* self_obj, PyObject* const* args,
                                      Py_ssize_t nargsf, PyObject* kwnames) {
  WeakrefLRUCache* self = nb::inst_ptr<WeakrefLRUCache>(self_obj);
  if (!self) {
    PyErr_SetString(PyExc_TypeError, "Not a WeakrefLRUCache");
    return nullptr;
  }

  size_t num_args =
      PyVectorcall_NARGS(nargsf) + (kwnames ? PyTuple_GET_SIZE(kwnames) : 0);
  try {
    return self->Call(self_obj, absl::MakeConstSpan(args, num_args), nargsf,
                      kwnames);
  } catch (nb::python_error& e) {
    e.restore();
    return nullptr;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

PyObject* WeakrefLRUCache::Call(PyObject* self_obj,
                                absl::Span<PyObject* const> args,
                                Py_ssize_t nargsf, PyObject* kwnames) {
  Py_ssize_t nargs_positional = PyVectorcall_NARGS(nargsf);
  if (nargs_positional < 1) {
    PyErr_SetString(PyExc_TypeError, "Missing weakref_key argument");
    return nullptr;
  }

  nb::object context = cache_context_fn_();
  nb::object weakref_key = nb::borrow<nb::object>(args[0]);

  WeakKey wrcache_key = MakeWeakrefKey(weakref_key);
  StrongKey key(context, args.subspan(1),
                kwnames ? nb::borrow<nb::tuple>(kwnames) : nb::tuple());

  bool inserted = false;
  std::shared_ptr<CacheEntry> entry;
  std::vector<nb::object> miss_keys;
  nb::object explainer;

  {
    nb::ft_object_guard lock(self_obj);
    ++total_queries_;

    // NOTE: entries_.insert may release the lock and may throw exceptions.
    auto [it_weak, weak_inserted] =
        entries_.insert(wrcache_key, WeakrefCacheValue());

    if (weak_inserted) {
      it_weak->second.cache = std::make_shared<Cache>();
    }
    // We need to make sure the Cache remains alive as long as this code block.
    std::shared_ptr<Cache> cache_ptr = it_weak->second.cache;
    Cache& cache = *it_weak->second.cache;

    // NOTE: cache.insert may release the lock and may throw exceptions.
    auto [it_strong, strong_inserted] = cache.insert(key, nullptr);

    if (strong_inserted) {
      inserted = true;
      ++misses_;
      entry = std::make_shared<CacheEntry>(this, wrcache_key, key);
      it_strong->second = entry;
      PushFront(entry.get());

      if (lru_maxsize_ > 0 && lru_size_ > lru_maxsize_) {
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
      MoveToFront(entry.get());
    }
  }

  // We must ensure that entry is destroyed under the lock so that if it is
  // the last reference, its destructor ~CacheEntry runs under the lock.
  // There are various scenarios where our copy of entry ends up as the last
  // owner of the entry, e.g., consider if a thread races to insert an entry
  // at the same time another thread calls Clear().
  absl::Cleanup destroy_entry_under_lock = [self_obj, &entry]() {
    nb::ft_object_guard lock(self_obj);
    entry.reset();
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
                                nargs_positional + 1, kwnames);
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
        auto error_string =
            absl::StrCat("Recursively calling ",
                         nb::cast<std::string>(nb::repr(weakref_key)));
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

std::vector<nb::object> WeakrefLRUCache::GetKeys() {
  std::vector<nb::object> results;
  for (const auto& kv : entries_) {
    const WeakKey& wr_key = kv.first;
    for (const auto& inner_kv : *kv.second.cache) {
      const StrongKey& key = inner_kv.first;
      const std::shared_ptr<CacheEntry>& value = inner_kv.second;
      if (!value->completed.HasBeenNotified()) {
        continue;
      }
      nb::tuple result =
          nb::make_tuple(*wr_key.ref, key.context(), key.args(), key.kwargs());
      results.push_back(std::move(result));
    }
  }
  return results;
}

WeakrefLRUCache::CacheInfo WeakrefLRUCache::GetCacheInfo() const {
  CacheInfo result;
  result.hits = total_queries_ - misses_;
  result.misses = misses_;
  result.maxsize = lru_maxsize_;
  result.currsize = lru_size_;
  return result;
}

void WeakrefLRUCache::Clear() {
  std::vector<std::pair<StrongKey, std::shared_ptr<CacheEntry>>>
      deferred_deletes;
  deferred_deletes.reserve(entries_.size());

  for (auto& kv : entries_) {
    for (auto& inner_kv : *(kv.second.cache)) {
      inner_kv.second->Unlink();
      deferred_deletes.emplace_back(inner_kv.first, inner_kv.second);
    }
    kv.second.cache->clear();
  }
  entries_.clear();
  deferred_deletes.clear();

  total_queries_ = misses_ = 0;
}

/*static*/ int WeakrefLRUCache::tp_traverse(PyObject* self, visitproc visit,
                                            void* arg) {
  Py_VISIT(Py_TYPE(self));
  if (!nb::inst_ready(self)) {
    return 0;
  }
  WeakrefLRUCache* cache = nb::inst_ptr<WeakrefLRUCache>(self);
  Py_VISIT(cache->cache_context_fn_.ptr());
  Py_VISIT(cache->fn_.ptr());
  if (cache->explain_) {
    Py_VISIT(cache->explain_->ptr());
  }

  for (const auto& kv : cache->entries_) {
    const WeakKey& wr_key = kv.first;
    const WeakrefCacheValue& wr_value = kv.second;

    Py_VISIT(wr_key.ref.ptr());

    for (const auto& inner_kv : *wr_value.cache) {
      const StrongKey& key = inner_kv.first;
      const std::shared_ptr<CacheEntry>& value = inner_kv.second;

      int return_val = key.tp_traverse(visit, arg);
      if (return_val != 0) return return_val;

      // We only visit the value and do not visit the backreferences to the
      // keys, since we must have visited them above.
      Py_VISIT(value->result.ptr());
    }
  }

  return 0;
}

/*static*/ int WeakrefLRUCache::tp_clear(PyObject* self) {
  WeakrefLRUCache* cache = nb::inst_ptr<WeakrefLRUCache>(self);
  cache->Clear();
  cache->cache_context_fn_.reset();
  cache->fn_.reset();
  cache->explain_ = std::nullopt;
  return 0;
}

/* static */ PyType_Slot WeakrefLRUCache::slots_[] = {
    {Py_tp_traverse, (void*)WeakrefLRUCache::tp_traverse},
    {Py_tp_clear, (void*)WeakrefLRUCache::tp_clear},
    {0, nullptr},
};

static PyMethodDef call_def = {
    "__call__", reinterpret_cast<PyCFunction>(WeakrefLRUCache::VectorCall),
    METH_FASTCALL | METH_KEYWORDS, "Calls the cache."};

NB_MODULE(weakref_lru_cache, m) {
  auto weakref_lru_cache =
      nb::class_<WeakrefLRUCache>(m, "WeakrefLRUCache",
                                  nb::is_weak_referenceable(),
                                  nb::type_slots(WeakrefLRUCache::slots_))
          .def(
              "evict_weakref",
              [](WeakrefLRUCache& cache, nb::object weakref_key) {
                cache.EvictWeakref(WeakKey(nb::weakref(weakref_key),
                                           StrongPythonHash(weakref_key)));
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
         std::optional<int64_t> maxsize, std::optional<nb::callable> explain) {
        return std::make_shared<WeakrefLRUCache>(
            cache_context_fn, fn,
            maxsize.value_or(std::numeric_limits<int>::max()), explain);
      },
      nb::arg("cache_context_fn"), nb::arg("fn"),
      nb::arg("maxsize").none() = 2048,
      nb::arg("explain") = std::optional<nb::callable>());
}

}  // namespace jax
