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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "nanobind/nanobind.h"
#include "nanobind/nb_defs.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/status_casters.h"

namespace jax {

static absl::StatusOr<int> StatusTest() {
  return absl::FailedPreconditionError("we are testing a status");
}

NB_MODULE(test_exceptions, m) {
  m.def("returns_status", xla::ValueOrThrowWrapper(StatusTest));

  m.def("throws_string",
        []() { throw xla::XlaRuntimeError("we are testing a string"); });
}

}  // namespace jax
