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

#ifndef JAXLIB_WEAK_KEY_WEAK_VALUE_CACHE_H_
#define JAXLIB_WEAK_KEY_WEAK_VALUE_CACHE_H_

#include <Python.h>

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "nanobind/nanobind.h"
#include "jaxlib/nb_class_ptr.h"

namespace jax {

// A cache that maps keys to values, but only keeps entries alive while both
// the key and value are alive. Keys must be a single Python object, and keys
// are compared and hashed based on their object ID alone. Both keys and values
// must be weak-referenceable.
// This class is implemented in C++ partially for efficiency, but also because
// it is much easier to be sure it is thread-safe if implemented this way.
class WeakKeyWeakValueCache {
 public:
  static void Register(nanobind::module_& m);

 private:
  // Factory function. Use this instead of the constructor since it populates
  // the weakref_callback_.
  static nb_class_ptr<WeakKeyWeakValueCache> Create(nanobind::callable fn);

  explicit WeakKeyWeakValueCache(nanobind::callable fn) : fn_(std::move(fn)) {}
  template <typename U, class... Args>
  friend nb_class_ptr<U> make_nb_class(Args&&... args);

  // __call__ method.
  static PyObject* VectorCall(PyObject* self_obj, PyObject* const* args,
                              Py_ssize_t nargs);

  // Callback when a weakref is destroyed.
  void OnWeakrefDestroyed(PyObject* weakref_obj);

  static int tp_traverse(PyObject* self_obj, visitproc visit, void* arg);
  static int tp_clear(PyObject* self_obj);
  static PyType_Slot slots_[];

  nanobind::callable fn_;
  nanobind::callable weakref_callback_;

  // Maps id(key) -> (weakref(key), weakref(value))
  absl::flat_hash_map<PyObject*,
                      std::pair<nanobind::weakref, nanobind::weakref>>
      entries_;

  // Reverse map for efficient eviction. Maps the address of a weakref object to
  // the address of a key object that is present in entries_.
  absl::flat_hash_map<PyObject*, PyObject*> weakref_to_key_;
};

}  // namespace jax

#endif  // JAXLIB_WEAK_KEY_WEAK_VALUE_CACHE_H_
