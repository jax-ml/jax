/* Copyright 2020 The JAX Authors

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

#ifndef JAXLIB_JAX_JIT_H_
#define JAXLIB_JAX_JIT_H_

#include <Python.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// placeholder for index annotation headers
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "jaxlib/py_values.h"
#include "jaxlib/pytree.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/tsl/platform/logging.h"

namespace jax {

void InitializeThreadLocalState();
bool GetDisableJit();
bool GetEnableX64();
std::optional<nanobind::callable> GetPostHook();

// An ArgumentSignature describes the static arguments to a function call, and
// how the dynamic arguments are related to the arguments. Together with the
// values of the dynamic arguments, this fully describes the arguments.
struct ArgumentSignature {
  // A PyTreeDef for each dynamic argument, positional arguments first
  // followed by keyword arguments. Keyword arguments are in the order given
  // by dynamic_arg_names.
  absl::InlinedVector<PyTreeDef, 2> dynamic_arg_treedefs;

  // Dynamic keyword argument names. Interned, and sorted by the keyword
  // name. Interned values are safe to compare by pointer.
  std::vector<nanobind::str> dynamic_arg_names;

  // Static arguments. Contains the positional arguments sorted in argument
  // order, followed by static keyword arguments in the order given by
  // `static_arg_names`.
  std::vector<nanobind::object> static_args;

  // Static keyword argument names. Interned, and sorted by keyword name.
  std::vector<nanobind::str> static_arg_names;

  bool operator==(const ArgumentSignature& other) const;
  bool operator!=(const ArgumentSignature& other) const {
    return !(*this == other);
  }

  std::string DebugString() const;
};

template <typename H>
H AbslHashValue(H h, const ArgumentSignature& s) {
  h = H::combine(std::move(h), s.dynamic_arg_treedefs,
                 s.dynamic_arg_names.size(), s.static_args.size(),
                 s.static_arg_names.size());

  for (const auto& name : s.dynamic_arg_names) {
    h = H::combine(std::move(h), name.ptr());
  }
  for (size_t i = 0; i < s.static_args.size(); ++i) {
    const auto& static_arg = s.static_args[i];
    Py_hash_t hash;
    try {
      hash = nanobind::hash(static_arg);
    } catch (const nanobind::python_error& e) {
      if (!e.matches(PyExc_TypeError)) throw;
      throw std::invalid_argument(absl::StrCat(
          "Non-hashable static arguments are not supported. An error occurred "
          "while trying to hash an object of type ",
          nanobind::cast<std::string_view>(nanobind::str(static_arg.type())),
          ", ", nanobind::cast<std::string_view>(nanobind::str(static_arg)),
          ". The error was:\n", e.what(), "\n"));
    }
    h = H::combine(std::move(h), hash);
  }
  for (const auto& name : s.static_arg_names) {
    h = H::combine(std::move(h), name.ptr());
  }
  return h;
}

// Filter out static arguments, flatten and concatenate other arguments (i.e.
// dynamic positional and keyword arguments), filling `arguments` in place.
// Args:
// positional_args: positional arguments
// keyword_args: the values of the keyword arguments
// kwnames: either None or a tuple containing the keyword argument names
// static_argnums: the indices of the static arguments in the positional
//   arguments
// static_argnames: the names of the static arguments, which must be interned.
// pytree_registry: the registry to use to convert the arguments to pytrees
// signature: output; describes the static arguments and the identities of the
//  dynamic arguments.
// flat_dynamic_args: output; the concatenation of the dynamic positional
//  arguments and sorted keyword arguments.
absl::Status ParseArguments(
    absl::Span<PyObject* const> positional_args,
    absl::Span<PyObject* const> keyword_args, nanobind::handle kwnames,
    absl::Span<int const> static_argnums,
    absl::Span<nanobind::str const> static_argnames,
    PyTreeRegistry* pytree_registry, ArgumentSignature& signature,
    absl::InlinedVector<nanobind::object, 2>& flat_dynamic_args);

// The signature of Python jitted function call, partitioned into:
// - dynamic positional arguments (i.e. positional args which are not static)
// - static positional arguments (i.e. the args associated to static_argnums)
// - keyword arguments
// The CallSignature should unambiguously identify a function call, thus,
// equality is based on:
// (a) Same PyTree for all dynamic positional arguments and keyword arguments
// (a) equality of the arguments and keyword arguments ArgSignature
// (a) equality (delegated to Python) of the static arguments.
struct CallSignature {
  // Not part of the signature, but we need it for error messages.
  std::string_view function_name;

  ArgumentSignature arg_signature;

  // Shape and dtype for both the dynamic positional arguments and the keyword
  // arguments (sorted by keyword name).
  absl::InlinedVector<PyArgSignature, 2> dynamic_arg_signatures;

  // The sharding of the jax.Array arguments.
  std::vector<nanobind::object> dynamic_arg_shardings;

  // The layout of the jax.Array arguments.
  std::vector<std::shared_ptr<const xla::PjRtLayout>> dynamic_arg_layouts;

  absl::InlinedVector<bool, 2> committed_args;

  // For JIT, we need this in the key because computation follows the data, so
  // we may have multiple executables depending on the devices the data is on.
  // This is not the case for PMAP, and is set to `nullptr`.
  xla::PjRtDevice* device = nullptr;

  std::vector<nanobind::object> configs;

  // Cached hash of the signature. Must be filled in using `absl::HashOf` as
  // part of CallSignature construction. The hash computation happens in a C++
  // exception-safe context, which simplifies using `CallSignature` as a key in
  // a non-exception-safe container because `Hash()` would never throw when used
  // inside the container implementation.
  size_t cached_hash;

  bool operator==(const CallSignature& other) const;
  bool operator!=(const CallSignature& other) const {
    return !(*this == other);
  }

  std::string DebugString() const;

  struct Hash {
    size_t operator()(const CallSignature& s) const noexcept {
      return s.cached_hash;
    }
  };
};

// A hash and equality for shardings that may sometimes return different hashes
// for equal values, and may sometimes return "not equal" for equal values.
// These are not correct implementations of `__hash__` and `__eq__` in python,
// but they are fine for jit/pjit dispatch since they only causes spurious cache
// misses.
size_t HashShardingForJit(nanobind::handle sharding);
bool EqualShardingsForJit(nanobind::handle a, nanobind::handle b);

template <typename H>
H AbslHashValue(H h, const CallSignature& s) {
  h = H::combine(std::move(h), s.arg_signature, s.dynamic_arg_signatures);

  DCHECK(s.dynamic_arg_shardings.empty() ||
         s.dynamic_arg_shardings.size() == s.dynamic_arg_signatures.size());

  DCHECK(s.dynamic_arg_layouts.empty() ||
         s.dynamic_arg_layouts.size() == s.dynamic_arg_signatures.size());

  // TODO(chky): For now, we are only hashing the pointer of shardings to avoid
  // slow python hashing function. Consider implementing hashing function and
  // equality checks in C++ in Sharding and use those here.
  for (const auto& sharding : s.dynamic_arg_shardings) {
    h = H::combine(std::move(h), HashShardingForJit(sharding));
  }

  for (const auto& layout : s.dynamic_arg_layouts) {
    if (layout != nullptr) {
      h = H::combine(std::move(h), *layout);
    }
  }

  h = H::combine(std::move(h), s.committed_args, s.device);

  // We do not hash the extra_jit_context fields since calling Python hash
  // functions is expensive (~300ns) and we don't expect a large number of
  // different contexts.
  return h;
}

// The function to call in `xla.cc` to add the bindings for this module.
void BuildJaxjitSubmodule(nanobind::module_& m);

}  // namespace jax

#endif  // JAXLIB_JAX_JIT_H_
