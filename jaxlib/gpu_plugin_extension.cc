/* Copyright 2024 The JAX Authors.

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

#include "jaxlib/gpu_plugin_extension.h"

#include <cstddef>
#include <utility>

#include "nanobind/nanobind.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "xla/ffi/api/c_api.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/py_client_gpu.h"
#include "xla/tsl/python/lib/core/numpy.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace nb = nanobind;

namespace xla {

namespace {

absl::Status RegisterCustomCallTarget(const PJRT_Api* c_api,
                                      const char* fn_name_c_str,
                                      size_t fn_name_size, nb::object fn,
                                      int api_version,
                                      XLA_FFI_Handler_Traits traits) {
  const PJRT_Gpu_Custom_Call* custom_call_ext =
      pjrt::FindExtension<PJRT_Gpu_Custom_Call>(
          c_api, PJRT_Extension_Type::PJRT_Extension_Type_Gpu_Custom_Call);
  if (custom_call_ext == nullptr) {
    return Unimplemented("The plugin does not have a custom call extension.");
  }
  PJRT_Gpu_Register_Custom_Call* register_custom_call =
      custom_call_ext->custom_call;

  if (traits != 0) {
    return Unimplemented("The plugin does not support custom call traits.");
  }

  PJRT_Gpu_Register_Custom_Call_Args args;
  args.struct_size = PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE;
  args.function_name = fn_name_c_str;
  args.function_name_size = fn_name_size;

#if PJRT_API_GPU_EXTENSION_VERSION >= 1
  args.api_version = api_version;
#endif

  auto as_capsule = [](nb::object obj) -> absl::StatusOr<nb::capsule> {
    nb::capsule capsule;
    if (!nb::try_cast<nb::capsule>(obj, capsule)) {
      return absl::InvalidArgumentError(
          "Custom call target registration requires handlers as PyCapsules");
    }
    return capsule;
  };

#if PJRT_API_GPU_EXTENSION_VERSION <= 1
  TF_ASSIGN_OR_RETURN(nb::capsule fn_execute, as_capsule(fn));
  args.custom_call_function = fn_execute.data();
  RETURN_STATUS_IF_PJRT_ERROR(register_custom_call(&args), c_api);
  return absl::OkStatus();
#else
  args.handler_instantiate = nullptr;
  args.handler_prepare = nullptr;
  args.handler_initialize = nullptr;
  args.handler_execute = nullptr;

  // Register legacy custom call target (untyped void* API).
  if (api_version == 0) {
    TF_ASSIGN_OR_RETURN(nb::capsule capsule_execute, as_capsule(fn));
    args.handler_execute = capsule_execute.data();
    RETURN_STATUS_IF_PJRT_ERROR(register_custom_call(&args), c_api);
    return absl::OkStatus();
  }

  // Register XLA FFI handler (typed API with explicit function signatures).
  if (api_version == 1) {
    auto capsule_execute = as_capsule(fn);
    if (capsule_execute.ok()) {
      args.handler_execute = capsule_execute->data();
      RETURN_STATUS_IF_PJRT_ERROR(register_custom_call(&args), c_api);
      return absl::OkStatus();
    }

    nb::dict bundle;
    if (nb::try_cast<nb::dict>(fn, bundle)) {
      auto handler = [&](const char* name) -> absl::StatusOr<void*> {
        if (!bundle.contains(name)) return nullptr;
        TF_ASSIGN_OR_RETURN(nb::capsule capsule, as_capsule(bundle[name]));
        return capsule.data();
      };

      TF_ASSIGN_OR_RETURN(args.handler_instantiate, handler("instantiate"));
      TF_ASSIGN_OR_RETURN(args.handler_prepare, handler("prepare"));
      TF_ASSIGN_OR_RETURN(args.handler_initialize, handler("initialize"));
      TF_ASSIGN_OR_RETURN(args.handler_execute, handler("execute"));
      RETURN_STATUS_IF_PJRT_ERROR(register_custom_call(&args), c_api);
      return absl::OkStatus();
    }

    return absl::InvalidArgumentError(
        "Unsupported custom call target type for api_version=1");
  }

  return absl::UnimplementedError(absl::StrFormat(
      "API version %d is not supported by RegisterCustomCallTarget. "
      "Supported versions are 0 and 1.",
      api_version));
#endif
}

absl::Status RegisterCustomTypeId(const PJRT_Api* c_api,
                                  const char* type_name_c_str,
                                  size_t type_name_size, nb::object type_id) {
  const PJRT_FFI_Extension* ffi_ext = pjrt::FindExtension<PJRT_FFI_Extension>(
      c_api, PJRT_Extension_Type::PJRT_Extension_Type_FFI);
  if (ffi_ext == nullptr) {
    return Unimplemented("The plugin does not have the FFI extension.");
  }

  PJRT_FFI_TypeID_Register_Args args;
  args.struct_size = PJRT_FFI_TypeID_Register_Args_STRUCT_SIZE;
  args.type_name = type_name_c_str;
  args.type_name_size = type_name_size;
  RETURN_STATUS_IF_PJRT_ERROR(ffi_ext->type_id_register(&args), c_api);

  nb::capsule capsule;
  if (!nb::try_cast<nb::capsule>(type_id, capsule)) {
    return absl::InvalidArgumentError(
        "The type_id argument to register_custom_call_type_id must be a "
        "PyCapsule object holding a pointer to a XLA_FFI_TypeId.");
  }
  XLA_FFI_TypeId* type_id_ptr =
      reinterpret_cast<XLA_FFI_TypeId*>(static_cast<void*>(capsule.data()));
  type_id_ptr->type_id = args.type_id;

  return absl::OkStatus();
}

nb::dict Registrations() {
  nb::dict dict;
  dict["xla_python_gpu_callback"] =
      jax::EncapsulateFunction(xla::XlaPythonGpuCallback);
  return dict;
}

}  //  namespace

void BuildGpuPluginExtension(nanobind::module_& m) {
  tsl::ImportNumpy();
  m.def(
      "register_custom_call_target",
      [](nb::capsule c_api, nb::object fn_name_py, nb::object fn,
         nb::str xla_platform_name, int api_version,
         XLA_FFI_Handler_Traits traits) {
        const char* fn_name_c_str;
        size_t fn_name_size;
        nb::str fn_name_bn_str;
        if (nb::try_cast<nb::str>(fn_name_py, fn_name_bn_str)) {
          fn_name_c_str = fn_name_bn_str.c_str();
          fn_name_size = nb::len(fn_name_bn_str);
        } else {
          nb::bytes bytes = nb::cast<nb::bytes>(fn_name_py);
          fn_name_c_str = bytes.c_str();
          fn_name_size = bytes.size();
        }
        xla::ThrowIfError(RegisterCustomCallTarget(
            static_cast<const PJRT_Api*>(c_api.data()), fn_name_c_str,
            fn_name_size, std::move(fn), api_version, traits));
      },
      nb::arg("c_api"), nb::arg("fn_name"), nb::arg("fn"),
      nb::arg("xla_platform_name"), nb::arg("api_version") = 0,
      nb::arg("traits") = 0);
  m.def(
      "register_custom_type_id",
      [](nb::capsule c_api, nb::str type_name_py, nb::object type_id) {
        const char* type_name_c_str = type_name_py.c_str();
        size_t type_name_size = nb::len(type_name_py);
        xla::ThrowIfError(RegisterCustomTypeId(
            static_cast<const PJRT_Api*>(c_api.data()), type_name_c_str,
            type_name_size, std::move(type_id)));
      },
      nb::arg("c_api"), nb::arg("type_name"), nb::arg("type_id"));
  m.def("registrations", &Registrations);
}

}  // namespace xla
