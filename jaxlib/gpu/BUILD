# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Shared CUDA/ROCM GPU kernels.

load(
    "//jaxlib:jax.bzl",
    "cc_proto_library",
    "jax_visibility",
    "xla_py_proto_library",
)
# Placeholder: load proto_library

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:internal"],
)

exports_files(srcs = [
    "blas_handle_pool.cc",
    "blas_handle_pool.h",
    "ffi_wrapper.h",
    "gpu_kernel_helpers.cc",
    "gpu_kernel_helpers.h",
    "gpu_kernels.cc",
    "hybrid.cc",
    "hybrid_kernels.cc",
    "hybrid_kernels.h",
    "linalg.cc",
    "linalg_kernels.cc",
    "linalg_kernels.cu.cc",
    "linalg_kernels.h",
    "make_batch_pointers.cu.cc",
    "make_batch_pointers.h",
    "prng.cc",
    "prng_kernels.cc",
    "prng_kernels.cu.cc",
    "prng_kernels.h",
    "py_client_gpu.cc",
    "py_client_gpu.h",
    "rnn.cc",
    "rnn_kernels.cc",
    "rnn_kernels.h",
    "solver.cc",
    "solver_handle_pool.cc",
    "solver_handle_pool.h",
    "solver_interface.cc",
    "solver_interface.h",
    "solver_kernels_ffi.cc",
    "solver_kernels_ffi.h",
    "sparse.cc",
    "sparse_kernels.cc",
    "sparse_kernels.h",
    "triton.cc",
    "triton_kernels.cc",
    "triton_kernels.h",
    "triton_utils.cc",
    "triton_utils.h",
    "vendor.h",
])

proto_library(
    name = "triton_proto",
    srcs = ["triton.proto"],
)

cc_proto_library(
    name = "triton_cc_proto",
    compatible_with = None,
    deps = [":triton_proto"],
)

xla_py_proto_library(
    name = "triton_py_pb2",
    visibility = jax_visibility("triton_proto_py_users"),
    deps = [":triton_proto"],
)

cc_library(
    name = "handle_pool",
    hdrs = ["handle_pool.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "gpu_plugin_extension",
    srcs = ["gpu_plugin_extension.cc"],
    hdrs = ["gpu_plugin_extension.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//jaxlib:kernel_nanobind_helpers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@nanobind",
        "@xla//xla:util",
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/pjrt/c:pjrt_c_api_ffi_extension_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_gpu_extension_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_helpers",
        "@xla//xla/pjrt/c:pjrt_c_api_triton_extension_hdrs",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/python/lib/core:numpy",
    ],
)
