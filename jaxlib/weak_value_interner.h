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

#ifndef JAXLIB_WEAK_VALUE_INTERNER_H_
#define JAXLIB_WEAK_VALUE_INTERNER_H_

#include <Python.h>

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/reentrant_hash_map.h"

namespace jax {

// WeakValueInterner is a thread-safe, reentrancy-safe interner for Python
// objects. It maps strongly-referenced arguments (*args and **kwargs) to
// weakly-referenced results. When a result is garbage collected by Python, the
// entry is automatically removed from the interner via a weakref callback.
//
// Caution: The interner does not know about the *signature* of the cached
// function. In particular, if the same argument value can be passed as either
// an arg or a kwarg, then the interner may store multiple entries for the same
// logical call. If this troubles you canonicalize the arguments first, e.g.
// via a wrapper function.
class WeakValueInterner {
 public:
  static void Register(nanobind::module_& m);

 private:
  // Factory method for creating a new interner.
  static nb_class_ptr<WeakValueInterner> Create(nanobind::callable fn);

  // Do not use directly, call Create() instead.
  explicit WeakValueInterner(nanobind::callable fn) : fn_(std::move(fn)) {}

 public:
  // A non-owning view of a key, used for looking up a map entry by value
  // equality.
  struct KeyView {
    absl::Span<nanobind::object const> kwnames;
    absl::Span<nanobind::object const> args;
    size_t cached_hash;

    template <typename H>
    friend H AbslHashValue(H h, const KeyView& key);
  };

  // A non-owning view of a key, used for looking up a map entry by pointer
  // equality rather than by value equality.
  struct PointerKey {
    absl::Span<nanobind::object const> kwnames;
    absl::Span<nanobind::object const> args;
    size_t cached_hash;

    template <typename H>
    friend H AbslHashValue(H h, const PointerKey& key);
  };

  // An owning key used to store arguments in the interner.
  struct Key {
    absl::InlinedVector<nanobind::object, 2> kwnames;
    absl::InlinedVector<nanobind::object, 4> args;
    size_t cached_hash;

    template <typename H>
    friend H AbslHashValue(H h, const Key& key);
  };

  struct KeyEqual {
    // We take keys by value to avoid invalidating references if the lock is
    // released during equality checks and the map is mutated.
    bool operator()(Key a, Key b) const;
    bool operator()(Key a, KeyView b) const;
    bool operator()(Key a, PointerKey b) const;
  };

  struct KeyHash {
    size_t operator()(Key const& k) const { return k.cached_hash; }
    size_t operator()(KeyView const& k) const { return k.cached_hash; }
    size_t operator()(PointerKey const& k) const { return k.cached_hash; }
  };

 private:
  template <typename H>
  friend H AbslHashValue(H h, const PointerKey& key);

  template <typename H>
  friend H AbslHashValue(H h, const Key& key);

  template <typename U, class... Args>
  friend nb_class_ptr<U> make_nb_class(Args&&... args);

  // The __call__ implementation.
  static PyObject* VectorCall(PyObject* self_obj, PyObject* const* args,
                              Py_ssize_t nargsf, PyObject* kwnames);

  // Called when an object referenced by a value's weakref is garbage collected.
  void OnWeakrefDestroyed(PyObject* weakref_obj);

  static int tp_traverse(PyObject* self_obj, visitproc visit, void* arg);
  static int tp_clear(PyObject* self_obj);
  static PyType_Slot slots_[];

  nanobind::callable fn_;
  nanobind::callable weakref_callback_;

  // The forward map from keys to entries.
  struct Entry {
    Key key;
    nanobind::weakref value_weakref;

    Entry(Key k, nanobind::weakref w)
        : key(std::move(k)), value_weakref(std::move(w)) {}
  };

  using MapType =
      ReentrantHashMap<Key, std::shared_ptr<Entry>, KeyHash, KeyEqual>;
  MapType entries_;

  // Maps address of weakref object to Entry*, used to evict an entry when
  // a weak reference expires.
  absl::flat_hash_map<PyObject*, Entry*> reverse_index_;
};

}  // namespace jax

#endif  // JAXLIB_WEAK_VALUE_INTERNER_H_
