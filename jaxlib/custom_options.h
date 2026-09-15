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
==============================================================================*/

#ifndef JAXLIB_CUSTOM_OPTIONS_H_
#define JAXLIB_CUSTOM_OPTIONS_H_

#include <optional>

#include "nanobind/nanobind.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/executable.h"

namespace jax {

// Per-call custom options are set from Python as a thread-local (see
// `jax._src.custom_options`) and merged into
// `xla::ifrt::ExecuteOptions::custom_options` on every execution dispatched
// from the current thread.

void SetCustomOptionsThreadLocal(
    std::optional<xla::ifrt::AttributeMap> options);
const std::optional<xla::ifrt::AttributeMap>& GetCustomOptionsThreadLocal();

// Merges thread-local custom options (if any) into `options.custom_options`.
void PopulateCustomOptions(xla::ifrt::ExecuteOptions& options);

// Merges `options` into the thread-local custom options for its lifetime;
// `options` take precedence over the enclosing thread-local ones.
class ScopedCustomOptions {
 public:
  explicit ScopedCustomOptions(std::optional<xla::ifrt::AttributeMap> options);
  ~ScopedCustomOptions();

  ScopedCustomOptions(const ScopedCustomOptions&) = delete;
  ScopedCustomOptions& operator=(const ScopedCustomOptions&) = delete;

 private:
  bool active_ = false;
  std::optional<xla::ifrt::AttributeMap> previous_;
};

// Reserved keyword argument of jitted and AOT compiled functions that carries
// per-call custom options: `f(*args, custom_options={"foo": 42})`.
inline constexpr char kCustomOptionsKwarg[] = "custom_options";

// Converts a Python dictionary with `str` keys and `bool`, `int`, `float`,
// `str`, `bytes` or `Sequence[int]` values to an IFRT attribute map.
xla::ifrt::AttributeMap AttributeMapFromPyDict(nanobind::dict dict);
nanobind::dict AttributeMapToPyDict(const xla::ifrt::AttributeMap& map);

void BuildCustomOptionsSubmodule(nanobind::module_& m);

}  // namespace jax

#endif  // JAXLIB_CUSTOM_OPTIONS_H_
