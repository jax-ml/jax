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

#include "jaxlib/py_relocatable.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/pair.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/pjrt/pjrt_relocatable.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/nb_absl_inlined_vector.h"  // IWYU pragma: keep

namespace nb = nanobind;

namespace jax {

PyRelocatable::PyRelocatable(
    std::shared_ptr<const xla::PjRtRelocatable> relocatable)
    : relocatable_(std::move(relocatable)) {
  CHECK(relocatable_ != nullptr);
}

nb::bytes PyRelocatable::call_backend_config() const {
  absl::string_view config = relocatable_->call_backend_config();
  return nb::bytes(config.data(), config.size());
}

absl::StatusOr<nb::bytes> PyRelocatable::Serialize() const {
  std::string serialized;
  {
    nb::gil_scoped_release gil_release;
    ABSL_ASSIGN_OR_RETURN(serialized, relocatable_->Serialize());
  }
  return nb::bytes(serialized.data(), serialized.size());
}

void PyRelocatable::Register(nb::module_& m) {
  nb::class_<PyRelocatable>(m, "Relocatable")
      .def_prop_ro("name", &PyRelocatable::name)
      .def_prop_ro("build_id", &PyRelocatable::build_id)
      .def_prop_ro("call_backend_config", &PyRelocatable::call_backend_config)
      .def_prop_ro("has_side_effects", &PyRelocatable::has_side_effects)
      .def("serialize", xla::ValueOrThrowWrapper(&PyRelocatable::Serialize))
      .def("get_parameter_element_types",
           xla::ValueOrThrowWrapper(&PyRelocatable::GetParameterElementTypes))
      .def("get_output_element_types",
           xla::ValueOrThrowWrapper(&PyRelocatable::GetOutputElementTypes))
      .def("get_parameter_dimensions",
           xla::ValueOrThrowWrapper(&PyRelocatable::GetParameterDimensions))
      .def("get_output_dimensions",
           xla::ValueOrThrowWrapper(&PyRelocatable::GetOutputDimensions))
      .def("get_parameter_layouts",
           xla::ValueOrThrowWrapper(&PyRelocatable::GetParameterLayouts))
      .def("get_output_layouts",
           xla::ValueOrThrowWrapper(&PyRelocatable::GetOutputLayouts))
      .def("get_parameter_output_aliases",
           xla::ValueOrThrowWrapper(&PyRelocatable::GetParameterOutputAliases));
}

}  // namespace jax
