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

// Python bindings for XLA compiler pass registration.
//
// Registers user-defined HLO transforms via two mechanisms:
//  1. RegisterHloXlaTransform (direct, for in-process backends like CPU)
//  2. PJRT C API XlaTransform extension (for plugin backends like TPU, GPU)

#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "jaxlib/nb_class_ptr.h"
#include "jaxlib/py_client.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/pjrt/c/pjrt_c_api_xla_transform_extension.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/python/version.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/xla_transform.h"
#include "xla/tsl/lib/strings/proto_serialization.h"

namespace nb = nanobind;

namespace {

// An HloXlaTransform that delegates to a Python callback.
//
// The callback receives a serialized HloModuleProto (bytes) and returns either
// modified serialized HloModuleProto bytes if the module was changed, or None
// if no changes were made.
class PyHloXlaTransform : public xla::HloXlaTransform {
 public:
  PyHloXlaTransform(std::string name, nb::object callback)
      : HloXlaTransform(std::move(name)), callback_(std::move(callback)) {}

  absl::StatusOr<bool> Transform(xla::HloModule* module) override {
    // Serialize the HLO module to a proto.
    xla::HloModuleProto proto = module->ToProto();
    std::string serialized_proto;
    if (!tsl::SerializeToStringDeterministic(proto, &serialized_proto)) {
      return absl::InternalError("Failed to serialize HLO module");
    }

    std::string returned_proto_bytes;
    bool changed = false;

    {
      // Call the Python callback with the GIL held.
      nb::gil_scoped_acquire gil;
      try {
        nb::bytes input_bytes(serialized_proto.data(), serialized_proto.size());
        nb::object result = callback_(input_bytes);

        if (!result.is_none()) {
          nb::bytes result_bytes = nb::cast<nb::bytes>(result);
          returned_proto_bytes =
              std::string(result_bytes.c_str(), result_bytes.size());
          changed = true;
        }
      } catch (nb::python_error& e) {
        return absl::InternalError(e.what());
      }
    }

    if (!changed) {
      return false;
    }

    // Parse the returned bytes back into an HloModuleProto outside the GIL.
    xla::HloModuleProto transformed_proto;
    if (!transformed_proto.ParseFromString(returned_proto_bytes)) {
      return absl::InternalError("Failed to parse transformed HLO module");
    }

    // Rebuild the module from the transformed proto using the shared helper.
    auto status = xla::UpdateHloModuleFromProto(module, transformed_proto);
    if (!status.ok()) {
      return status;
    }

    return true;
  }

 private:
  nb::object callback_;
};

// State for a PJRT C API callback. Intentionally leaked since registered
// transforms persist for the lifetime of the process.
struct CApiCallbackState {
  nb::object py_callback;
  PJRT_XlaTransform_Callbacks callbacks;
};

struct CallbackCleanupData {
  char* transformed_data = nullptr;
  char* error_msg_data = nullptr;

  ~CallbackCleanupData() {
    delete[] transformed_data;
    delete[] error_msg_data;
  }
};

// The C callback that bridges between the PJRT C API and the Python callback.
void CApiTransformHloModuleCallback(PJRT_XlaTransform_Callbacks* callbacks,
                                    PJRT_XlaTransform_Args* args) {
  auto* state = reinterpret_cast<CApiCallbackState*>(
      reinterpret_cast<char*>(callbacks) -
      offsetof(CApiCallbackState, callbacks));

  auto* cleanup_data = new CallbackCleanupData();
  args->header.data = cleanup_data;
  args->header.cleanup_fn = [](void* data) {
    delete static_cast<CallbackCleanupData*>(data);
  };

  nb::gil_scoped_acquire gil;
  try {
    nb::bytes input_bytes(args->hlo_module.data, args->hlo_module.size);
    nb::object result = state->py_callback(input_bytes);

    if (result.is_none()) {
      args->changed = false;
    } else {
      nb::bytes result_bytes = nb::cast<nb::bytes>(result);
      size_t size = result_bytes.size();
      cleanup_data->transformed_data = new char[size];
      std::memcpy(cleanup_data->transformed_data, result_bytes.c_str(), size);
      args->transformed_hlo_module.data = cleanup_data->transformed_data;
      args->transformed_hlo_module.size = size;
      args->changed = true;
    }
  } catch (nb::python_error& e) {
    args->header.has_error = true;
    args->header.code = PJRT_Error_Code_INTERNAL;
    std::string error_msg = e.what();
    cleanup_data->error_msg_data = new char[error_msg.size()];
    std::memcpy(cleanup_data->error_msg_data, error_msg.data(),
                error_msg.size());
    args->header.error_msg.data = cleanup_data->error_msg_data;
    args->header.error_msg.size = error_msg.size();
  }
}

}  // namespace

NB_MODULE(_xla, m) {
  // Register a transform directly via RegisterHloXlaTransform.
  // This is used for in-process backends (e.g. CPU).
  m.def(
      "register_xla_transform",
      [](std::string name, int stage, nb::object callback) {
        xla::HloXlaTransform::PipelineStage pipeline_stage;
        switch (stage) {
          case 0:
            pipeline_stage = xla::HloXlaTransform::PipelineStage::kPreScheduler;
            break;
          case 1:
            pipeline_stage =
                xla::HloXlaTransform::PipelineStage::kPostScheduler;
            break;
          default:
            throw std::runtime_error("Invalid pipeline stage");
        }

        auto transform = std::make_shared<PyHloXlaTransform>(
            std::move(name), std::move(callback));
        xla::RegisterHloXlaTransform(pipeline_stage, std::move(transform));
      },
      nb::arg("name"), nb::arg("stage"), nb::arg("callback"));

  m.def(
      "clear_xla_transform",
      [](std::string name, int stage) {
        xla::HloXlaTransform::PipelineStage pipeline_stage;
        switch (stage) {
          case 0:
            pipeline_stage = xla::HloXlaTransform::PipelineStage::kPreScheduler;
            break;
          case 1:
            pipeline_stage =
                xla::HloXlaTransform::PipelineStage::kPostScheduler;
            break;
          default:
            throw std::runtime_error("Invalid pipeline stage");
        }
        return xla::ClearHloXlaTransform(pipeline_stage, name);
      },
      nb::arg("name"), nb::arg("stage"));

  // Register a transform via the PJRT C API XlaTransform extension.
  // This is used for plugin backends (e.g. TPU, GPU).
  m.def(
      "register_xla_transform_c_api",
      [](nb::object client_obj, std::string name, int stage,
         nb::object callback) {
        if (client_obj.is_none()) {
          throw std::runtime_error(
              "register_xla_transform_c_api: client cannot be None.");
        }
        auto client = nb::cast<jax::nb_class_ptr<jax::PyClient>>(client_obj);
        std::shared_ptr<xla::PjRtClient> pjrt_client =
            client->shared_ptr_pjrt_client();
        auto* c_api_client =
            dynamic_cast<xla::PjRtCApiClient*>(pjrt_client.get());
        if (c_api_client == nullptr) {
          // Client is not a PJRT C API client; skip silently.
          return;
        }
        const PJRT_Api* c_api_value = c_api_client->pjrt_c_api();

        PJRT_Xla_Transform_Extension* extension =
            pjrt::FindExtension<PJRT_Xla_Transform_Extension>(
                c_api_value,
                PJRT_Extension_Type::PJRT_Extension_Type_XlaTransform);
        if (extension == nullptr) {
          // Extension not available for this client; skip silently.
          return;
        }

        // Allocate callback state on the heap. Cleared via dtor if
        // clear_xla_transform is called.
        auto* state = new CApiCallbackState();
        state->py_callback = std::move(callback);
        state->callbacks.version = PJRT_API_XLA_TRANSFORM_EXTENSION_VERSION;
        state->callbacks.dtor = [](PJRT_XlaTransform_Callbacks* callbacks) {
          auto* state = reinterpret_cast<CApiCallbackState*>(
              reinterpret_cast<char*>(callbacks) -
              offsetof(CApiCallbackState, callbacks));
          delete state;
        };
        state->callbacks.transform_hlo_module = CApiTransformHloModuleCallback;

        PJRT_Register_Xla_Transform_Args args;
        args.struct_size = PJRT_Register_Xla_Transform_Args_STRUCT_SIZE;
        args.name = name.c_str();
        args.name_size = name.size();
        args.stage = static_cast<PJRT_XlaTransform_PipelineStage>(stage);
        args.callbacks = &state->callbacks;

        PJRT_Error* error = extension->register_xla_transform(&args);
        if (error != nullptr) {
          absl::Status status = pjrt::PjrtErrorToStatus(error);
          throw std::runtime_error(status.ToString());
        }
      },
      nb::arg("client"), nb::arg("name"), nb::arg("stage"),
      nb::arg("callback"));

  m.def(
      "clear_xla_transform_c_api",
      [](nb::object client_obj, std::string name, int stage) {
        if (client_obj.is_none()) {
          throw std::runtime_error(
              "clear_xla_transform_c_api: client cannot be None.");
        }
        auto client = nb::cast<jax::nb_class_ptr<jax::PyClient>>(client_obj);
        std::shared_ptr<xla::PjRtClient> pjrt_client =
            client->shared_ptr_pjrt_client();
        auto* c_api_client =
            dynamic_cast<xla::PjRtCApiClient*>(pjrt_client.get());
        if (c_api_client == nullptr) {
          return false;
        }
        const PJRT_Api* c_api_value = c_api_client->pjrt_c_api();

        PJRT_Xla_Transform_Extension* extension =
            pjrt::FindExtension<PJRT_Xla_Transform_Extension>(
                c_api_value,
                PJRT_Extension_Type::PJRT_Extension_Type_XlaTransform);
        if (extension == nullptr) {
          return false;
        }

        PJRT_Clear_Xla_Transform_Args args;
        args.struct_size = PJRT_Clear_Xla_Transform_Args_STRUCT_SIZE;
        args.name = name.c_str();
        args.name_size = name.size();
        args.stage = static_cast<PJRT_XlaTransform_PipelineStage>(stage);
        args.callbacks = nullptr;

        PJRT_Error* error = extension->clear_xla_transform(&args);
        if (error != nullptr) {
          absl::Status status = pjrt::PjrtErrorToStatus(error);
          throw std::runtime_error(status.ToString());
        }
        return args.cleared;
      },
      nb::arg("client"), nb::arg("name"), nb::arg("stage"));
}
