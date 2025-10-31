/* Copyright 2025 The JAX Authors.

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

#ifndef JAX_JAXLIB_CACHED_PY_OBJECT_H_
#define JAX_JAXLIB_CACHED_PY_OBJECT_H_

#include <atomic>

#include "absl/functional/function_ref.h"
#include "nanobind/nanobind.h"

namespace jax {

// A lock-free thread-safe cache for a single Python object.
// Example use case: caching a hash value in an object.
class CachedPyObject {
 public:
  CachedPyObject() = default;
  ~CachedPyObject() {
    PyObject* value = value_.load();
    Py_XDECREF(value);
  }

  // Returns the cached value of the object. If the object is not present,
  // factory() will be called to create it and the cache will be populated.
  // Note: factory() may be called multiple times if used concurrently. The
  // returned value will be one of the returned values of factory().
  // Thread-safe.
  nanobind::object Get(absl::FunctionRef<nanobind::object()> factory) {
    PyObject* v = value_.load();
    if (v) {
      return nanobind::borrow<nanobind::object>(v);
    }
    nanobind::object new_value = factory();
    if (value_.compare_exchange_strong(v, new_value.inc_ref().ptr())) {
      return new_value;
    } else {
      new_value.dec_ref();
      return nanobind::borrow<nanobind::object>(v);
    }
  }

 private:
  std::atomic<PyObject*> value_ = nullptr;
};

}  // namespace jax

#endif  // JAX_JAXLIB_CACHED_PY_OBJECT_H_
