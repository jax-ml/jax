/* Copyright 2025 The JAX Authors

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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_RESHARD_ARRAYS_LIB_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_RESHARD_ARRAYS_LIB_H_

#include "absl/status/statusor.h"
#include "nanobind/nanobind.h"
#include "xla/xla_data.pb.h"

namespace jax {

absl::StatusOr<nanobind::list> ExperimentalReshardArrays(
    nanobind::sequence py_arrays, nanobind::sequence out_shardings,
    bool donate_input);

}  // namespace jax

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_RESHARD_ARRAYS_LIB_H_
