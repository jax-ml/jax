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

# NVIDIA CUDA kernels

load(
    "//jaxlib:jax.bzl",
    "cuda_library",
    "if_cuda_is_configured",
    "pybind_extension",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)

cc_library(
    name = "cuda_vendor",
    hdrs = [
        "//jaxlib/gpu:vendor.h",
    ],
    defines = ["JAX_GPU_CUDA=1"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudnn_header",
    ],
)

cc_library(
    name = "cuda_gpu_kernel_helpers",
    srcs = [
        "//jaxlib/gpu:gpu_kernel_helpers.cc",
    ],
    hdrs = [
        "//jaxlib/gpu:gpu_kernel_helpers.h",
    ],
    copts = [
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":cuda_vendor",
        "@xla//xla/stream_executor/cuda:cusolver_lib",
        "@xla//xla/stream_executor/cuda:cusparse_lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_cuda//cuda:cublas_headers",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "cublas_kernels",
    srcs = ["//jaxlib/gpu:blas_kernels.cc"],
    hdrs = ["//jaxlib/gpu:blas_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        "//jaxlib:handle_pool",
        "//jaxlib:kernel_helpers",
        "@xla//xla/service:custom_call_status",
        "@xla//xla/stream_executor/cuda:cublas_lib",
        "@xla//xla/stream_executor/cuda:cudart_stub",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cublas_headers",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_blas",
    srcs = ["//jaxlib/gpu:blas.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    linkopts = select({
        "@xla//xla/python:use_jax_cuda_pip_rpaths": [
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cuda_runtime/lib",
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cublas/lib",
        ],
        "//conditions:default": [],
    }),
    module_name = "_blas",
    deps = [
        ":cublas_kernels",
        ":cuda_vendor",
        "//jaxlib:kernel_pybind11_helpers",
        "@xla//xla/stream_executor/cuda:cublas_lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@pybind11",
    ],
)

cc_library(
    name = "cudnn_rnn_kernels",
    srcs = ["//jaxlib/gpu:rnn_kernels.cc"],
    hdrs = ["//jaxlib/gpu:rnn_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        "//jaxlib:handle_pool",
        "//jaxlib:kernel_helpers",
        "@xla//xla/service:custom_call_status",
        "@xla//xla/stream_executor/cuda:cudart_stub",
        "@xla//xla/stream_executor/cuda:cudnn_lib",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_rnn",
    srcs = ["//jaxlib/gpu:rnn.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_rnn",
    deps = [
        ":cuda_vendor",
        ":cudnn_rnn_kernels",
        "//jaxlib:kernel_pybind11_helpers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ],
)

cc_library(
    name = "cusolver_kernels",
    srcs = ["//jaxlib/gpu:solver_kernels.cc"],
    hdrs = ["//jaxlib/gpu:solver_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        "//jaxlib:handle_pool",
        "//jaxlib:kernel_helpers",
        "@xla//xla/service:custom_call_status",
        "@xla//xla/stream_executor/cuda:cusolver_lib",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_solver",
    srcs = ["//jaxlib/gpu:solver.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    linkopts = select({
        "@xla//xla/python:use_jax_cuda_pip_rpaths": [
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cuda_runtime/lib",
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cusolver/lib",
        ],
        "//conditions:default": [],
    }),
    module_name = "_solver",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        ":cusolver_kernels",
        "//jaxlib:kernel_pybind11_helpers",
        "@xla//xla/stream_executor/cuda:cudart_stub",
        "@xla//xla/stream_executor/cuda:cusolver_lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_cuda//cuda:cuda_headers",
        "@pybind11",
    ],
)

cc_library(
    name = "cusparse_kernels",
    srcs = ["//jaxlib/gpu:sparse_kernels.cc"],
    hdrs = ["//jaxlib/gpu:sparse_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        "//jaxlib:handle_pool",
        "//jaxlib:kernel_helpers",
        "@xla//xla/service:custom_call_status",
        "@xla//xla/stream_executor/cuda:cudart_stub",
        "@xla//xla/stream_executor/cuda:cusparse_lib",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_sparse",
    srcs = ["//jaxlib/gpu:sparse.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    linkopts = select({
        "@xla//xla/python:use_jax_cuda_pip_rpaths": [
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cuda_runtime/lib",
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cusparse/lib",
        ],
        "//conditions:default": [],
    }),
    module_name = "_sparse",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        ":cusparse_kernels",
        "//jaxlib:kernel_pybind11_helpers",
        "@xla//xla/stream_executor/cuda:cudart_stub",
        "@xla//xla/stream_executor/cuda:cusparse_lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
        "@pybind11",
    ],
)

cc_library(
    name = "cuda_lu_pivot_kernels",
    srcs = [
        "//jaxlib/gpu:lu_pivot_kernels.cc",
    ],
    hdrs = ["//jaxlib/gpu:lu_pivot_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_lu_pivot_kernels_impl",
        ":cuda_vendor",
        "//jaxlib:kernel_helpers",
        "@xla//xla/service:custom_call_status",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cuda_library(
    name = "cuda_lu_pivot_kernels_impl",
    srcs = [
        "//jaxlib/gpu:lu_pivot_kernels.cu.cc",
    ],
    hdrs = ["//jaxlib/gpu:lu_pivot_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        "//jaxlib:kernel_helpers",
        "@xla//xla/service:custom_call_status",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_linalg",
    srcs = ["//jaxlib/gpu:linalg.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_linalg",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_lu_pivot_kernels",
        ":cuda_lu_pivot_kernels_impl",
        ":cuda_vendor",
        "//jaxlib:kernel_pybind11_helpers",
        "@xla//xla/stream_executor/cuda:cudart_stub",
        "@local_config_cuda//cuda:cuda_headers",
        "@pybind11",
    ],
)

cc_library(
    name = "cuda_prng_kernels",
    srcs = [
        "//jaxlib/gpu:prng_kernels.cc",
    ],
    hdrs = ["//jaxlib/gpu:prng_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_prng_kernels_impl",
        ":cuda_vendor",
        "//jaxlib:kernel_helpers",
        "@xla//xla/service:custom_call_status",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cuda_library(
    name = "cuda_prng_kernels_impl",
    srcs = [
        "//jaxlib/gpu:prng_kernels.cu.cc",
    ],
    hdrs = ["//jaxlib/gpu:prng_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        "//jaxlib:kernel_helpers",
        "@xla//xla/service:custom_call_status",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_prng",
    srcs = ["//jaxlib/gpu:prng.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_prng",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_prng_kernels",
        "//jaxlib:kernel_pybind11_helpers",
        "@xla//xla/stream_executor/cuda:cudart_stub",
        "@local_config_cuda//cuda:cuda_headers",
        "@pybind11",
    ],
)

cc_library(
    name = "cuda_gpu_kernels",
    srcs = ["//jaxlib/gpu:gpu_kernels.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":cublas_kernels",
        ":cuda_lu_pivot_kernels",
        ":cuda_prng_kernels",
        ":cuda_vendor",
        ":cusolver_kernels",
        ":cusparse_kernels",
        "@xla//xla/service:custom_call_target_registry",
    ],
    alwayslink = 1,
)

pybind_extension(
    name = "_triton",
    srcs = ["//jaxlib/gpu:triton.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    linkopts = select({
        "@xla//xla/python:use_jax_cuda_pip_rpaths": [
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cuda_runtime/lib",
        ],
        "//conditions:default": [],
    }),
    module_name = "_triton",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        "@xla//xla/service:custom_call_status",
        "@xla//xla/stream_executor/cuda:cudart_stub",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ],
)

py_library(
    name = "cuda_gpu_support",
    deps = [
        ":_blas",
        ":_linalg",
        ":_prng",
        ":_rnn",
        ":_solver",
        ":_sparse",
        ":_triton",
    ],
)

# We cannot nest select and if_cuda_is_configured so we introduce
# a standalone py_library target.
py_library(
    name = "gpu_only_test_deps",
    # `if_cuda_is_configured` will default to `[]`.
    deps = if_cuda_is_configured([":cuda_gpu_support"]),
)
