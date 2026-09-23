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

#ifndef JAXLIB_EXECUTION_OPTIONS_H_
#define JAXLIB_EXECUTION_OPTIONS_H_

#include <optional>

#include "nanobind/nanobind.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/executable.h"

namespace jax {

// Per-execution options stored in config state (see
// `jax.execution_options`) and merged into `xla::ifrt::ExecuteOptions` on every
// execution dispatched from the current thread.
struct ExecutionOptions {
  std::optional<xla::ifrt::AttributeMap> custom_options;
};

// Merges the current config's execution options (if any) into `options`.
// Requires the GIL (or an attached thread state under free-threading).
void PopulateExecutionOptions(xla::ifrt::ExecuteOptions& options);

// Converts a Python dictionary with `str` keys and `bool`, `int`, `float`,
// `str` or `Sequence[int]` values to an IFRT attribute map.
xla::ifrt::AttributeMap AttributeMapFromPyDict(nanobind::dict dict);
nanobind::dict AttributeMapToPyDict(const xla::ifrt::AttributeMap& map);

void BuildExecutionOptionsSubmodule(nanobind::module_& m);

}  // namespace jax

#endif  // JAXLIB_EXECUTION_OPTIONS_H_
