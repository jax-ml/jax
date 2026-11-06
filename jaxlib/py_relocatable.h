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

#ifndef JAXLIB_PY_RELOCATABLE_H_
#define JAXLIB_PY_RELOCATABLE_H_

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "nanobind/nanobind.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/pjrt_relocatable.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace jax {

// Thin Python wrapper around std::shared_ptr<const xla::PjRtRelocatable>.
// Wrapping the pure C++ std::shared_ptr avoids nanobind's Python-refcounting
// shared_ptr deleter when passing relocatables across GIL-released compilation
// calls.
class PyRelocatable {
 public:
  explicit PyRelocatable(
      std::shared_ptr<const xla::PjRtRelocatable> relocatable);
  ~PyRelocatable() = default;

  const std::shared_ptr<const xla::PjRtRelocatable>& relocatable() const {
    return relocatable_;
  }

  std::string_view name() const { return relocatable_->name(); }
  std::string_view build_id() const { return relocatable_->build_id(); }
  nanobind::bytes call_backend_config() const;
  bool has_side_effects() const { return relocatable_->has_side_effects(); }

  absl::StatusOr<nanobind::bytes> Serialize() const;
  absl::StatusOr<std::vector<xla::PrimitiveType>> GetParameterElementTypes()
      const {
    return relocatable_->GetParameterElementTypes();
  }
  absl::StatusOr<std::vector<xla::PrimitiveType>> GetOutputElementTypes()
      const {
    return relocatable_->GetOutputElementTypes();
  }
  absl::StatusOr<std::vector<xla::DimensionVector>> GetParameterDimensions()
      const {
    return relocatable_->GetParameterDimensions();
  }
  absl::StatusOr<std::vector<xla::DimensionVector>> GetOutputDimensions()
      const {
    return relocatable_->GetOutputDimensions();
  }
  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetParameterLayouts() const {
    return relocatable_->GetParameterLayouts();
  }
  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetOutputLayouts() const {
    return relocatable_->GetOutputLayouts();
  }
  absl::StatusOr<std::vector<std::pair<int64_t, int64_t>>>
  GetParameterOutputAliases() const {
    return relocatable_->GetParameterOutputAliases();
  }

  static void Register(nanobind::module_& m);

 private:
  std::shared_ptr<const xla::PjRtRelocatable> relocatable_;
};

}  // namespace jax

#endif  // JAXLIB_PY_RELOCATABLE_H_
