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

#ifndef JAXLIB_IFRT_RTTI_H_
#define JAXLIB_IFRT_RTTI_H_

// This is a temporary shim used for migrating IFRT's RTTI API from LLVM to
// IFRT's own API. Once JAX builds against IFRT >= 64, this file can be removed
// and all includes can directly use `ifrt/rtti.h`.
//
// always_keep and exports are used temporarily to suppress linter warnings.

#include "xla/python/version.h"

// IWYU pragma: always_keep

#if JAX_IFRT_VERSION_NUMBER >= 64

// IWYU pragma: begin_exports
#include "xla/python/ifrt/rtti.h"
// IWYU pragma: end_exports

#else  // JAX_IFRT_VERSION_NUMBER < 64

// IWYU pragma: begin_exports
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
// IWYU pragma: end_exports

namespace xla {
namespace ifrt {

using ::llvm::RTTIExtends;
using ::llvm::RTTIRoot;

using ::llvm::cast;
using ::llvm::cast_if_present;
using ::llvm::cast_or_null;
using ::llvm::dyn_cast;
using ::llvm::dyn_cast_if_present;
using ::llvm::dyn_cast_or_null;
using ::llvm::isa;
using ::llvm::isa_and_nonnull;
using ::llvm::isa_and_present;

}  // namespace ifrt
}  // namespace xla

#endif  // JAX_IFRT_VERSION_NUMBER >= 64

#endif  // JAXLIB_IFRT_RTTI_H_
