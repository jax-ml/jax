/* Copyright 2026 The JAX Authors

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

#include "jaxlib/strong_lru_cache.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/pair.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/pytree.h"
#include "jaxlib/reentrant_hash_map.h"

namespace nb = nanobind;

namespace jax {

class StrongLRUCache;
struct StrongCacheEntry;

struct StrongLRUNode {
  StrongCacheEntry* next = nullptr;
  StrongLRUNode* prev = nullptr;
};

struct StrongCacheEntry {
  StrongLRUCache* parent;

  // Has the thread that was computing this entry finished its work?
  absl::Notification completed;

  // The result, if we computed one. A valid (non-null) object indicates
  // a successfully computed result. Only valid once completed is true.
  nb::object result;

  // The thread that is computing the entry. This is used to detect reentrant
  // calls to the same cache entry.
  // TODO(phawkins): we could probably just skip this: the user will notice
  // if they write an infinite recursion anyway.
  std::thread::id thread_id = std::this_thread::get_id();

  // The links in the LRU list. Protected by the shard's lock.
  StrongLRUNode lru_node;

  // The strong key for this entry.
  StrongKey key;

  StrongCacheEntry(StrongLRUCache* p, StrongKey k)
      : parent(p), key(std::move(k)) {}

  ~StrongCacheEntry();

  // Is this node linked into the LRU list?
  bool IsLinked() const { return lru_node.prev != nullptr; }

  // Remove the node from the LRU list. Requires holding the shard's lock.
  void Unlink();
};

class StrongLRUCache {
 public:
  StrongLRUCache(const StrongLRUCache&) = delete;
  StrongLRUCache& operator=(const StrongLRUCache&) = delete;
  StrongLRUCache(StrongLRUCache&&) = delete;
  StrongLRUCache& operator=(StrongLRUCache&&) = delete;

  static nb_class_ptr<StrongLRUCache> Create(
      std::optional<nb::callable> cache_context_fn, nb::callable fn,
      int64_t maxsize, std::optional<nb::callable> explain, int64_t num_shards);

  StrongLRUCache(std::optional<nb::callable> cache_context_fn, nb::callable fn,
                 int64_t maxsize, std::optional<nb::callable> explain,
                 int64_t num_shards);

  ~StrongLRUCache() { ClearUnlocked(); }

  std::vector<nb::object> GetKeys();
  nb::callable wrapped() const { return fn_; }

  struct CacheInfo {
    int64_t hits;
    int64_t misses;
    int64_t maxsize;
    int64_t currsize;
  };
  CacheInfo GetCacheInfo() const;

  void Clear();

  static PyObject* VectorCall(PyObject* self_obj, PyObject* const* args,
                              Py_ssize_t nargsf, PyObject* kwnames);

  static PyType_Slot slots_[];

 private:
  friend struct StrongCacheEntry;

  std::optional<nb::callable> cache_context_fn_;
  nb::callable fn_;
  std::optional<nb::callable> explain_;

  using Cache = ReentrantHashMap<StrongKey, std::shared_ptr<StrongCacheEntry>,
                                 StrongKey::CachedHash, StrongKey::SafeEqual>;

  struct Shard {
    nanobind::object lock;
    Cache entries;
    int64_t lru_size{0};
    StrongLRUNode lru_head;
    int64_t misses = 0;
    int64_t total_queries = 0;
  };

  int64_t lru_maxsize_;
  size_t num_shards_;
  std::vector<std::unique_ptr<Shard>> shards_;

  PyObject* Call(PyObject* self_obj, absl::Span<PyObject* const> args,
                 Py_ssize_t nargsf, PyObject* kwnames, const StrongKey& key);

  void MoveToFront(size_t shard_idx, StrongCacheEntry* node);
  void PushFront(size_t shard_idx, StrongCacheEntry* node);
  void EvictLeastRecentlyUsed(size_t shard_idx);
  void ClearShard(size_t i);
  void ClearUnlocked();

  int TpTraverse(visitproc visit, void* arg);
  void TpClear();

  static int tp_traverse(PyObject* self, visitproc visit, void* arg);
  static int tp_clear(PyObject* self);
};

// StrongKey implementation
StrongKey::StrongKey(const StrongKeyView& lkey)
    : context_(lkey.context),
      kwnames_(lkey.kwnames.begin(), lkey.kwnames.end()),
      args_(lkey.args.begin(), lkey.args.end()),
      treedef_(lkey.treedef ? std::optional<PyTreeDef>(*lkey.treedef)
                            : std::nullopt),
      cached_hash_(lkey.cached_hash) {}

bool StrongKey::operator==(const StrongKey& other) const {
  // ReentrantHashMap, like absl::flat_hash_map, does not store or compare all
  // 64 bits of the hash directly. Since we have to store the hash anyway,
  // benchmarks show it is profitable to compare the full hash first before
  // comparing the keys.
  if (cached_hash_ != other.cached_hash_) return false;
  if (treedef_ != other.treedef_) return false;
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
  if (treedef_) {
    int ret = treedef_->Traverse(visit, arg);
    if (ret) return ret;
  }
  return 0;
}

bool StrongKey::SafeEqual::operator()(StrongKey a,
                                      const StrongKeyView& b) const {
  if (a.cached_hash() != b.cached_hash) return false;
  if (a.treedef_.has_value() != (b.treedef != nullptr)) return false;
  if (a.treedef_ && !(*a.treedef_ == *b.treedef)) return false;
  if (!a.context_.equal(b.context)) return false;
  if (a.kwnames_.size() != b.kwnames.size()) return false;
  for (size_t i = 0; i < a.kwnames_.size(); ++i) {
    if (a.kwnames_[i].ptr() != b.kwnames[i].ptr()) return false;
  }
  if (a.args_.size() != b.args.size()) return false;
  for (size_t i = 0; i < a.args_.size(); ++i) {
    if (!a.args_[i].equal(b.args[i])) return false;
  }
  return true;
}

bool StrongKey::SafeEqual::operator()(StrongKey a,
                                      const PointerStrongKey& b) const {
  if (a.cached_hash() != b.cached_hash) return false;
  if (a.treedef_.has_value() != (b.treedef != nullptr)) return false;
  if (a.treedef_ && !(*a.treedef_ == *b.treedef)) return false;
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

// StrongCacheEntry implementation
void StrongCacheEntry::Unlink() {
  CHECK(IsLinked());
  lru_node.prev->next = lru_node.next;
  size_t shard_idx = key.cached_hash() % parent->num_shards_;
  auto& shard = parent->shards_[shard_idx];
  if (lru_node.next) {
    lru_node.next->lru_node.prev = lru_node.prev;
  } else {
    shard->lru_head.prev = lru_node.prev;
  }
  lru_node.prev = nullptr;
  lru_node.next = nullptr;
  shard->lru_size--;
}

StrongCacheEntry::~StrongCacheEntry() {
  if (IsLinked()) {
    Unlink();
  }
}

// StrongLRUCache implementation
StrongLRUCache::StrongLRUCache(std::optional<nb::callable> cache_context_fn,
                               nb::callable fn, int64_t maxsize,
                               std::optional<nb::callable> explain,
                               int64_t num_shards)
    : cache_context_fn_(std::move(cache_context_fn)),
      fn_(fn),
      explain_(explain),
      lru_maxsize_(maxsize),
      num_shards_(num_shards) {
  shards_.reserve(num_shards_);
  for (size_t i = 0; i < num_shards_; ++i) {
    auto shard = std::make_unique<Shard>();
    shard->lru_head.prev = &shard->lru_head;
#ifdef NB_FREE_THREADED
    shard->lock = nb::steal<nb::object>(
        PyObject_CallNoArgs(reinterpret_cast<PyObject*>(&PyBaseObject_Type)));
#else
    shard->lock = nb::none();
#endif
    shards_.push_back(std::move(shard));
  }
}

/*static*/ nb_class_ptr<StrongLRUCache> StrongLRUCache::Create(
    std::optional<nb::callable> cache_context_fn, nb::callable fn,
    int64_t maxsize, std::optional<nb::callable> explain, int64_t num_shards) {
  return make_nb_class<StrongLRUCache>(std::move(cache_context_fn),
                                       std::move(fn), maxsize,
                                       std::move(explain), num_shards);
}

void StrongLRUCache::PushFront(size_t shard_idx, StrongCacheEntry* node) {
  auto& shard = shards_[shard_idx];
  CHECK(!node->IsLinked());
  node->lru_node.next = shard->lru_head.next;
  node->lru_node.prev = &shard->lru_head;
  if (shard->lru_head.next) {
    shard->lru_head.next->lru_node.prev = &node->lru_node;
  } else {
    shard->lru_head.prev = &node->lru_node;
  }
  shard->lru_head.next = node;
  shard->lru_size++;
}

void StrongLRUCache::MoveToFront(size_t shard_idx, StrongCacheEntry* node) {
  auto& shard = shards_[shard_idx];
  CHECK(node->IsLinked());
  if (shard->lru_head.next == node) {
    return;
  }
  node->lru_node.prev->next = node->lru_node.next;
  if (node->lru_node.next) {
    node->lru_node.next->lru_node.prev = node->lru_node.prev;
  } else {
    shard->lru_head.prev = node->lru_node.prev;
  }
  node->lru_node.next = shard->lru_head.next;
  node->lru_node.prev = &shard->lru_head;
  shard->lru_head.next->lru_node.prev = &node->lru_node;
  shard->lru_head.next = node;
}

void StrongLRUCache::EvictLeastRecentlyUsed(size_t shard_idx) {
  auto& shard = shards_[shard_idx];
  StrongCacheEntry* last = shard->lru_head.prev->prev->next;
  CHECK(last != nullptr);
  CHECK(last->IsLinked());
  last->Unlink();

  PointerStrongKey ptr_strong_key{
      last->key.context(), absl::MakeConstSpan(last->key.kwnames()),
      last->key.args_span(), last->key.treedef(), last->key.cached_hash()};
  auto it = shard->entries.find(ptr_strong_key);
  if (it != shard->entries.end()) {
    shard->entries.erase(it);
  }
}

PyObject* StrongLRUCache::Call(PyObject* self_obj,
                               absl::Span<PyObject* const> args,
                               Py_ssize_t nargsf, PyObject* kwnames,
                               const StrongKey& key) {
  size_t shard_idx = key.cached_hash() % num_shards_;
  auto& shard = shards_[shard_idx];

  std::shared_ptr<StrongCacheEntry> entry;
  bool is_new = false;
  {
    nb::ft_object_guard lock(shard->lock);
    shard->total_queries++;

    auto [it, inserted] = shard->entries.try_emplace(key, nullptr);
    is_new = inserted;

    if (is_new) {
      entry = std::make_shared<StrongCacheEntry>(this, key);
      it->second = entry;
      shard->misses++;
      if (lru_maxsize_ > 0) {
        PushFront(shard_idx, entry.get());
      }
    } else {
      entry = it->second;
      if (entry->completed.HasBeenNotified()) {
        if (entry->result.is_valid()) {
          if (lru_maxsize_ > 0) {
            MoveToFront(shard_idx, entry.get());
          }
          return entry->result.inc_ref().ptr();
        }
      } else {
        if (entry->thread_id == std::this_thread::get_id()) {
          nb::tuple key_tuple = nb::make_tuple(
              entry->key.context(), entry->key.args(), entry->key.kwargs());
          auto error_string =
              absl::StrCat("Recursively calling ",
                           nb::cast<std::string>(nb::repr(key_tuple)));
          PyErr_SetString(PyExc_RecursionError, error_string.c_str());
          return nullptr;
        }
      }
    }
  }

  if (is_new) {
    absl::Cleanup cleanup = [&] {
      {
        nb::ft_object_guard lock(shard->lock);
        entry->completed.Notify();
        if (entry->IsLinked() && !entry->result.is_valid()) {
          entry->Unlink();
          PointerStrongKey ptr_strong_key{
              entry->key.context(), absl::MakeConstSpan(entry->key.kwnames()),
              entry->key.args_span(), entry->key.treedef(),
              entry->key.cached_hash()};
          auto it = shard->entries.find(ptr_strong_key);
          if (it != shard->entries.end()) {
            shard->entries.erase(it);
          }
        }
      }
    };

    // Explain cache miss before calling the function, if enabled.
    if (explain_) {
      try {
        nb::object explainer = (*explain_)();
        if (!explainer.is_none()) {
          std::vector<nb::object> miss_keys = GetKeys();
          nb::object py_miss_keys = nb::cast(std::move(miss_keys));

          Py_ssize_t num_pos_args = PyVectorcall_NARGS(nargsf);
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
      } catch (nb::python_error& e) {
        e.restore();
        return nullptr;
      }
    }

    PyObject* result =
        PyObject_Vectorcall(fn_.ptr(), args.data(), nargsf, kwnames);
    if (!result) {
      return nullptr;
    }

    {
      nb::ft_object_guard lock(shard->lock);
      entry->result = nb::steal<nb::object>(result);

      int64_t shard_maxsize = (lru_maxsize_ + num_shards_ - 1) / num_shards_;
      if (lru_maxsize_ > 0 && shard->lru_size > shard_maxsize) {
        EvictLeastRecentlyUsed(shard_idx);
      }
    }
    return entry->result.inc_ref().ptr();
  } else {
    nb::gil_scoped_release release;
    entry->completed.WaitForNotification();
  }

  if (entry->result.is_valid()) {
    return entry->result.inc_ref().ptr();
  } else {
    return PyObject_Vectorcall(fn_.ptr(), args.data(), nargsf, kwnames);
  }
}

std::vector<nb::object> StrongLRUCache::GetKeys() {
  // Snapshot entries without allocating any Python objects. It is not safe
  // to allocate Python objects during traversal of the map, because that may
  // trigger garbage collection and ultimately may mutate the map (e.g., because
  // of keys being garbage collected).
  std::vector<std::shared_ptr<StrongCacheEntry>> snapshot;

  for (size_t i = 0; i < num_shards_; ++i) {
    nb::ft_object_guard lock(shards_[i]->lock);
    for (const auto& kv : shards_[i]->entries) {
      if (kv.second->completed.HasBeenNotified()) {
        snapshot.push_back(kv.second);
      }
    }
  }

  // Produce the output, which may allocate Python objects.
  std::vector<nb::object> results;
  results.reserve(snapshot.size());
  for (const auto& entry : snapshot) {
    nb::tuple result = nb::make_tuple(entry->key.context(), entry->key.args(),
                                      entry->key.kwargs());
    results.push_back(std::move(result));
  }
  return results;
}

StrongLRUCache::CacheInfo StrongLRUCache::GetCacheInfo() const {
  CacheInfo result;
  result.hits = 0;
  result.misses = 0;
  result.maxsize = lru_maxsize_;
  result.currsize = 0;

  for (size_t i = 0; i < num_shards_; ++i) {
    nb::ft_object_guard lock(shards_[i]->lock);
    result.misses += shards_[i]->misses;
    result.currsize += shards_[i]->lru_size;
    result.hits += (shards_[i]->total_queries - shards_[i]->misses);
  }
  return result;
}

void StrongLRUCache::ClearShard(size_t i) {
  std::vector<std::pair<StrongKey, std::shared_ptr<StrongCacheEntry>>>
      deferred_deletes;
  deferred_deletes.reserve(shards_[i]->entries.size());

  // We must copy the Python references in the inner and map before calling
  // clear().
  for (auto& kv : shards_[i]->entries) {
    if (kv.second->IsLinked()) {
      kv.second->Unlink();
    }
    deferred_deletes.emplace_back(kv.first, kv.second);
  }
  shards_[i]->entries.clear();
  shards_[i]->total_queries = shards_[i]->misses = 0;
  // deferred_deletes is deleted here.
}

void StrongLRUCache::ClearUnlocked() {
  for (size_t i = 0; i < num_shards_; ++i) {
    ClearShard(i);
  }
}

void StrongLRUCache::Clear() {
  for (size_t i = 0; i < num_shards_; ++i) {
    nb::ft_object_guard lock(shards_[i]->lock);
    ClearShard(i);
  }
}

int StrongLRUCache::TpTraverse(visitproc visit, void* arg) {
  if (cache_context_fn_) {
    Py_VISIT(cache_context_fn_->ptr());
  }
  Py_VISIT(fn_.ptr());
  if (explain_) {
    Py_VISIT(explain_->ptr());
  }

  for (size_t i = 0; i < num_shards_; ++i) {
    Py_VISIT(shards_[i]->lock.ptr());
    for (const auto& [strong_key, value] : shards_[i]->entries) {
      int status = strong_key.tp_traverse(visit, arg);
      if (status != 0) return status;
      // We only visit the value and do not visit the backreferences to the
      // keys, since we must have visited them above.
      if (value->result.is_valid()) {
        Py_VISIT(value->result.ptr());
      }
    }
  }

  return 0;
}

void StrongLRUCache::TpClear() {
  ClearUnlocked();
  for (size_t i = 0; i < num_shards_; ++i) {
    shards_[i]->lock.reset();
  }
  cache_context_fn_.reset();
  fn_.reset();
  explain_ = std::nullopt;
}

/*static*/ PyObject* StrongLRUCache::VectorCall(PyObject* self_obj,
                                                PyObject* const* args,
                                                Py_ssize_t nargsf,
                                                PyObject* kwnames) {
  try {
    StrongLRUCache* self = nb::inst_ptr<StrongLRUCache>(self_obj);
    if (!self) {
      PyErr_SetString(PyExc_TypeError, "Not a StrongLRUCache");
      return nullptr;
    }

    Py_ssize_t num_pos_args = PyVectorcall_NARGS(nargsf);
    nb::object context =
        self->cache_context_fn_ ? (*self->cache_context_fn_)() : nb::none();

    Py_ssize_t num_kwargs = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    size_t num_args = num_pos_args + num_kwargs;
    absl::Span<PyObject* const> args_span = absl::MakeConstSpan(args, num_args);

    absl::InlinedVector<nb::object, 2> sorted_kwnames;
    absl::InlinedVector<nb::object, 4> strong_args;

    strong_args.reserve(num_args);
    for (Py_ssize_t i = 0; i < num_pos_args; ++i) {
      strong_args.push_back(nb::borrow<nb::object>(args[i]));
    }

    if (num_kwargs > 0) {
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
    return self->Call(self_obj, args_span, nargsf, kwnames, key);
  } catch (nb::python_error& e) {
    e.restore();
    return nullptr;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

/*static*/ int StrongLRUCache::tp_traverse(PyObject* self_obj, visitproc visit,
                                           void* arg) {
  Py_VISIT(Py_TYPE(self_obj));
  if (!nb::inst_ready(self_obj)) {
    return 0;
  }
  StrongLRUCache* self = nb::inst_ptr<StrongLRUCache>(self_obj);
  return self->TpTraverse(visit, arg);
}

/*static*/ int StrongLRUCache::tp_clear(PyObject* self_obj) {
  StrongLRUCache* self = nb::inst_ptr<StrongLRUCache>(self_obj);
  self->TpClear();
  return 0;
}

static PyMethodDef strong_call_def = {
    "__call__", reinterpret_cast<PyCFunction>(StrongLRUCache::VectorCall),
    METH_FASTCALL | METH_KEYWORDS, "Calls the strong cache."};

/* static */ PyType_Slot StrongLRUCache::slots_[] = {
    {Py_tp_traverse, (void*)StrongLRUCache::tp_traverse},
    {Py_tp_clear, (void*)StrongLRUCache::tp_clear},
    {0, nullptr},
};

void RegisterStrongLruCache(nb::module_& m) {
  auto strong_lru_cache =
      nb::class_<StrongLRUCache>(m, "StrongLRUCache",
                                 nb::is_weak_referenceable(),
                                 nb::type_slots(StrongLRUCache::slots_))
          .def("cache_keys", &StrongLRUCache::GetKeys)
          .def("cache_info", &StrongLRUCache::GetCacheInfo)
          .def("cache_clear", &StrongLRUCache::Clear)
          .def_prop_ro("__wrapped__", &StrongLRUCache::wrapped);

  strong_lru_cache.attr("__call__") = nb::steal<nb::object>(
      PyDescr_NewMethod(reinterpret_cast<PyTypeObject*>(strong_lru_cache.ptr()),
                        &strong_call_def));

  nb::class_<StrongLRUCache::CacheInfo>(strong_lru_cache, "StrongLRUCacheInfo")
      .def_ro("hits", &StrongLRUCache::CacheInfo::hits)
      .def_ro("misses", &StrongLRUCache::CacheInfo::misses)
      .def_ro("maxsize", &StrongLRUCache::CacheInfo::maxsize)
      .def_ro("currsize", &StrongLRUCache::CacheInfo::currsize)
      .def("__repr__", [](StrongLRUCache::CacheInfo& info) {
        return absl::StrCat(
            "StrongLRUCache(hits=", info.hits, ", misses=", info.misses,
            ", maxsize=", info.maxsize, ", currsize=", info.currsize, ")");
      });

  m.def(
      "strong_lru_cache",
      [](nb::callable fn, std::optional<nb::callable> cache_context_fn,
         std::optional<int64_t> maxsize, std::optional<nb::callable> explain,
         std::optional<int64_t> num_shards) {
        int64_t shards = num_shards.value_or(
#ifdef NB_FREE_THREADED
            16
#else
            1
#endif
        );
        return StrongLRUCache::Create(std::move(cache_context_fn),
                                      std::move(fn), maxsize.value_or(-1),
                                      std::move(explain), shards);
      },
      nb::arg("fn"), nb::arg("cache_context_fn").none() = nb::none(),
      nb::arg("maxsize").none() = 2048,
      nb::arg("explain") = std::optional<nb::callable>(),
      nb::arg("num_shards") = std::optional<int64_t>());
}

}  // namespace jax
