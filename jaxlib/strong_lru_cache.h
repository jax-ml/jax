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

#ifndef JAXLIB_STRONG_LRU_CACHE_H_
#define JAXLIB_STRONG_LRU_CACHE_H_

#include <cstddef>
#include <optional>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "jaxlib/pytree.h"

namespace jax {

// StrongKeyView is used for heterogeneous lookup in Call to avoid copies on
// hits. It uses value comparison for context and args, which may release the
// GIL.
struct StrongKeyView {
  nanobind::object context;
  absl::Span<nanobind::object const> kwnames;
  absl::Span<nanobind::object const> args;
  const PyTreeDef* treedef;
  size_t cached_hash;
};

struct PointerStrongKey {
  nanobind::object context;
  absl::Span<nanobind::object const> kwnames;
  absl::Span<nanobind::object const> args;
  const PyTreeDef* treedef;
  size_t cached_hash;
};

// StrongKey is the key to the strong part of the table. It has three parts:
// the user-provided context object, and the positional and keyword arguments.
// We store the arguments as the concatenation of the positional arguments and
// the keyword arguments, together with a vector of keyword names. This is
// efficient to construct from the Python vectorcall protocol; we need never
// build a dictionary.
class StrongKey {
 public:
  StrongKey(nanobind::object context,
            absl::InlinedVector<nanobind::object, 2> kwnames,
            absl::InlinedVector<nanobind::object, 4> args,
            std::optional<PyTreeDef> treedef = std::nullopt)
      : context_(std::move(context)),
        kwnames_(std::move(kwnames)),
        args_(std::move(args)),
        treedef_(std::move(treedef)) {
    cached_hash_ = absl::HashOf(*this);
  }
  explicit StrongKey(const StrongKeyView& lkey);

  bool operator==(const StrongKey& other) const;

  template <typename H>
  friend H AbslHashValue(H h, const StrongKey& key) {
    h = H::combine(std::move(h), nanobind::hash(key.context_));
    for (const auto& kwname : key.kwnames_) {
      h = H::combine(std::move(h), kwname.ptr());
    }
    for (const auto& arg : key.args_) {
      h = H::combine(std::move(h), nanobind::hash(arg));
    }
    if (key.treedef_) {
      h = H::combine(std::move(h), *key.treedef_);
    }
    return h;
  }

  struct SafeEqual {
    // It is important that we take the keys by value not by reference because
    // equal() may release locks, and per the contract of our hash map this may
    // invalidate references.
    bool operator()(StrongKey a, StrongKey b) const { return a == b; }
    bool operator()(StrongKey a, const PointerStrongKey& b) const;
    bool operator()(StrongKey a, const StrongKeyView& b) const;
  };

  struct CachedHash {
    size_t operator()(StrongKey key) const { return key.cached_hash_; }
    size_t operator()(const PointerStrongKey& key) const;
    size_t operator()(const StrongKeyView& key) const {
      return key.cached_hash;
    }
  };

  nanobind::object context() const { return context_; }
  absl::Span<const nanobind::object> kwnames() const { return kwnames_; }
  absl::Span<const nanobind::object> args_span() const { return args_; }
  const PyTreeDef* treedef() const { return treedef_ ? &*treedef_ : nullptr; }
  size_t cached_hash() const { return cached_hash_; }

  nanobind::object args() const;
  nanobind::object kwargs() const;

  int tp_traverse(visitproc visit, void* arg) const;

 private:
  nanobind::object context_;

  // Keyword argument names, interned and sorted by pointer.
  absl::InlinedVector<nanobind::object, 2> kwnames_;

  // Positional arguments followed by keyword arguments. The keyword arguments
  // are stored in the order they appear in kwnames.
  absl::InlinedVector<nanobind::object, 4> args_;

  // The pytree definition of the arguments, if applicable.
  std::optional<PyTreeDef> treedef_;

  // The cached hash value. See the comment on WeakKey.
  size_t cached_hash_;
};

void RegisterStrongLruCache(nanobind::module_& m);

}  // namespace jax

#endif  // JAXLIB_STRONG_LRU_CACHE_H_
