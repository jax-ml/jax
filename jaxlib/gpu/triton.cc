#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "absl/status/statusor.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/triton.pb.h"
#include "jaxlib/gpu/triton_kernels.h"
#include "jaxlib/gpu/triton_utils.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_pybind11_helpers.h"
#include "pybind11_abseil/status_casters.h"  // IWYU pragma: keep

#define CUDA_RETURN_IF_ERROR(expr) JAX_RETURN_IF_ERROR(JAX_AS_STATUS(expr))


namespace py = pybind11;

namespace jax::JAX_GPU_NAMESPACE {

PYBIND11_MODULE(_triton, m) {
  py::class_<Kernel>(m, "TritonKernel")
      .def(py::init<std::string, uint32_t, uint32_t, std::string, std::string,
                    int>());

  py::class_<KernelCall::Parameter>(m, "TritonParameter");

  m.def("create_array_parameter",
        [](size_t bytes_to_zero, size_t ptr_divisibility) {
          return KernelCall::Parameter{
              KernelCall::Parameter::Array{bytes_to_zero, ptr_divisibility}};
        });

  m.def("create_scalar_parameter",
        [](py::bool_ value,
           std::string_view dtype) -> absl::StatusOr<KernelCall::Parameter> {
          if ((dtype == "i1") || (dtype == "B")) {
            return KernelCall::Parameter{static_cast<bool>(value)};
          } else {
            return absl::InvalidArgumentError(std::string("unknown dtype: ") +
                                              dtype.data());
          }
        });

  m.def("create_scalar_parameter",
        [](py::int_ value,
           std::string_view dtype) -> absl::StatusOr<KernelCall::Parameter> {
          if (dtype == "i32") {
            return KernelCall::Parameter{static_cast<int32_t>(value)};
          } else if (dtype == "u32") {
            return KernelCall::Parameter{static_cast<uint32_t>(value)};
          } else if (dtype == "i64") {
            return KernelCall::Parameter{static_cast<int64_t>(value)};
          } else if (dtype == "u64") {
            return KernelCall::Parameter{static_cast<uint64_t>(value)};
          } else {
            return absl::InvalidArgumentError(std::string("unknown dtype: ") +
                                              dtype.data());
          }
        });

  m.def("create_scalar_parameter",
        [](py::float_ value,
           std::string_view dtype) -> absl::StatusOr<KernelCall::Parameter> {
          if (dtype == "fp32") {
            return KernelCall::Parameter{static_cast<float>(value)};
          } else if (dtype == "fp64") {
            return KernelCall::Parameter{static_cast<double>(value)};
          } else {
            return absl::InvalidArgumentError(std::string("unknown dtype: ") +
                                              dtype.data());
          }
        });

  py::class_<KernelCall>(m, "TritonKernelCall")
      .def(py::init<Kernel, uint32_t, uint32_t, uint32_t,
                    std::vector<KernelCall::Parameter>>())
      .def("to_proto", [](const KernelCall& kernel_call, std::string name,
                          std::string metadata) {
        jax_triton::TritonAnyKernelCall proto;
        *proto.mutable_kernel_call() = kernel_call.ToProto();
        proto.set_name(std::move(name));
        proto.set_metadata(std::move(metadata));
        return py::bytes(proto.SerializeAsString());
      });

  py::class_<AutotunedKernelCall>(m, "TritonAutotunedKernelCall")
      .def(py::init<>([](std::string name,
                         std::vector<std::pair<KernelCall, std::string>>
                             calls_and_descriptions,
                         std::vector<std::tuple<size_t, size_t, size_t>>
                             input_output_aliases) {
        std::vector<AutotunedKernelCall::Config> configs;
        configs.reserve(calls_and_descriptions.size());
        for (auto& [kernel_call, desc] : calls_and_descriptions) {
          configs.push_back({std::move(kernel_call), std::move(desc)});
        }
        return std::make_unique<AutotunedKernelCall>(
            std::move(name), std::move(configs),
            std::move(input_output_aliases));
      }))
      .def("to_proto", [](const AutotunedKernelCall& kernel_call,
                          std::string name, std::string metadata) {
        jax_triton::TritonAnyKernelCall proto;
        *proto.mutable_autotuned_kernel_call() = kernel_call.ToProto();
        proto.set_name(std::move(name));
        proto.set_metadata(std::move(metadata));
        return py::bytes(proto.SerializeAsString());
      });

  m.def("get_compute_capability", [](int device) -> absl::StatusOr<int> {
    int major, minor;
    CUDA_RETURN_IF_ERROR(cuInit(device));
    CUDA_RETURN_IF_ERROR(cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CUDA_RETURN_IF_ERROR(cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    return major * 10 + minor;
  });

  m.def("get_serialized_metadata",
        [](absl::string_view opaque) -> absl::StatusOr<py::bytes> {
          JAX_ASSIGN_OR_RETURN(std::string metadata,
                               GetTritonKernelCallSerializedMetadata(opaque));
          return py::bytes(metadata);
        });
}

}  // namespace jax::JAX_GPU_NAMESPACE
