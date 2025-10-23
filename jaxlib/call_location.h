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

#ifndef JAXLIB_CALL_LOCATION_H_
#define JAXLIB_CALL_LOCATION_H_

#include <string>
#include <optional>

#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/user_context.h"

namespace jax {

enum class RuntimeTracebackMode {
  kOff = 0,
  kOn = 1,
  kFull = 2,
};

void PopulateCallLocation(xla::ifrt::ExecuteOptions& options,
                          const xla::ifrt::UserContext* user_context);

void AddExcludePath(std::string path);
void SetSendTracebackToRuntimeGlobal(RuntimeTracebackMode mode);
void SetSendTracebackToRuntimeThreadLocal(
    std::optional<RuntimeTracebackMode> mode);


}  // namespace jax

#endif  // JAX_JAXLIB_CALL_LOCATION_H_
