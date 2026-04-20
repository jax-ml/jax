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

#ifndef JAXLIB_HASH_UTIL_H_
#define JAXLIB_HASH_UTIL_H_

#include <Python.h>

#include <cstddef>

#include "absl/base/casts.h"

namespace jax {

// Safely casts an unsigned hash value to a signed Python hash value.
// If we return an unsigned integer with the high bit set, then Python falls out
// of a fast path:
// https://github.com/python/cpython/blob/dd88e77fad887aaf00ead1a3ba655fc00fd1dc25/Objects/typeobject.c#L10799
inline Py_hash_t AbslHashToPythonHash(size_t h) {
  return absl::bit_cast<Py_hash_t>(h);
}

}  // namespace jax

#endif  // JAXLIB_HASH_UTIL_H_
