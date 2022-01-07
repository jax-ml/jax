# Copyright 2018 Google LLC
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

# JAX is Autograd and XLA

load(
    "//jaxlib:jax.bzl",
    "cuda_library",
    "flatbuffer_cc_library",
    "flatbuffer_py_library",
    "if_rocm_is_configured",
    "pybind_extension",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "kernel_pybind11_helpers",
    hdrs = ["kernel_pybind11_helpers.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":kernel_helpers",
        "@com_google_absl//absl/base",
        "@pybind11",
    ],
)

cc_library(
    name = "kernel_helpers",
    hdrs = ["kernel_helpers.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
    ],
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
    name = "cuda_gpu_kernel_helpers",
    srcs = ["cuda_gpu_kernel_helpers.cc"],
    hdrs = ["cuda_gpu_kernel_helpers.h"],
    copts = [
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@org_tensorflow//tensorflow/stream_executor/cuda:cusolver_lib",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cusparse_lib",
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
    name = "rocm_gpu_kernel_helpers",
    srcs = ["rocm_gpu_kernel_helpers.cc"],
    hdrs = ["rocm_gpu_kernel_helpers.h"],
    copts = [
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

py_library(
    name = "jaxlib",
    srcs = [
        "init.py",
        "lapack.py",
        "pocketfft.py",
        "version.py",
    ] + if_rocm_is_configured([
        "rocsolver.py",
    ]),
    deps = [":pocketfft_flatbuffers_py"],
)

exports_files([
    "setup.py",
    "setup.cfg",
])

# CPU kernels

pybind_extension(
    name = "cpu_feature_guard",
    srcs = ["cpu_feature_guard.c"],
    module_name = "cpu_feature_guard",
    deps = [
        "@org_tensorflow//third_party/python_runtime:headers",
    ],
)

# LAPACK

cc_library(
    name = "lapack_kernels",
    srcs = ["lapack_kernels.cc"],
    hdrs = ["lapack_kernels.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:dynamic_annotations",
    ],
)

py_library(
    name = "lapack",
    srcs = ["lapack.py"],
    deps = ["//third_party/py/numpy"],
)

pybind_extension(
    name = "_lapack",
    srcs = ["lapack.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_lapack",
    deps = [
        ":kernel_pybind11_helpers",
        ":lapack_kernels",
        "@pybind11",
    ],
)

# PocketFFT

flatbuffer_cc_library(
    name = "pocketfft_flatbuffers_cc",
    srcs = ["pocketfft.fbs"],
)

flatbuffer_py_library(
    name = "pocketfft_flatbuffers_py",
    srcs = ["pocketfft.fbs"],
)

cc_library(
    name = "pocketfft_kernels",
    srcs = ["pocketfft_kernels.cc"],
    hdrs = ["pocketfft_kernels.h"],
    copts = ["-fexceptions"],  # PocketFFT may throw.
    features = ["-use_header_modules"],
    deps = [
        ":pocketfft_flatbuffers_cc",
        "@flatbuffers//:runtime_cc",
        "@pocketfft",
    ],
)

pybind_extension(
    name = "_pocketfft",
    srcs = ["pocketfft.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_pocketfft",
    deps = [
        ":kernel_pybind11_helpers",
        ":pocketfft_kernels",
        "@pybind11",
    ],
)

cc_library(
    name = "cpu_kernels",
    srcs = ["cpu_kernels.cc"],
    deps = [
        ":lapack_kernels",
        ":pocketfft_kernels",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_target_registry",
    ],
    alwayslink = 1,
)

# GPU kernels

py_library(
    name = "gpu_support",
    srcs = [
        "cuda_linalg.py",
        "cuda_prng.py",
        "cusolver.py",
        "cusparse.py",
    ],
    deps = [
        ":_cublas",
        ":_cuda_linalg",
        ":_cuda_prng",
        ":_cusolver",
        ":_cusparse",
    ],
)

cc_library(
    name = "cublas_kernels",
    srcs = ["cublas_kernels.cc"],
    hdrs = ["cublas_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":handle_pool",
        ":kernel_helpers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cublas_lib",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cublas_headers",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_cublas",
    srcs = ["cublas.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_cublas",
    deps = [
        ":cublas_kernels",
        ":kernel_pybind11_helpers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cublas_lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_cuda//cuda:cuda_headers",
        "@pybind11",
    ],
)

cc_library(
    name = "cusolver_kernels",
    srcs = ["cusolver_kernels.cc"],
    hdrs = ["cusolver_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":handle_pool",
        ":kernel_helpers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cusolver_lib",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_cusolver",
    srcs = ["cusolver.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_cusolver",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cusolver_kernels",
        ":kernel_pybind11_helpers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cusolver_lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_cuda//cuda:cuda_headers",
        "@pybind11",
    ],
)

cc_library(
    name = "cusparse_kernels",
    srcs = ["cusparse_kernels.cc"],
    hdrs = ["cusparse_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":handle_pool",
        ":kernel_helpers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cusparse_lib",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_cusparse",
    srcs = ["cusparse.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_cusparse",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cusparse_kernels",
        ":kernel_pybind11_helpers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cusparse_lib",
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
        "cuda_lu_pivot_kernels.cc",
    ],
    hdrs = ["cuda_lu_pivot_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_lu_pivot_kernels_impl",
        ":kernel_helpers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cuda_library(
    name = "cuda_lu_pivot_kernels_impl",
    srcs = [
        "cuda_lu_pivot_kernels.cu.cc",
    ],
    hdrs = ["cuda_lu_pivot_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":kernel_helpers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_cuda_linalg",
    srcs = ["cuda_linalg.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_cuda_linalg",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_lu_pivot_kernels",
        ":cuda_lu_pivot_kernels_impl",
        ":kernel_pybind11_helpers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@local_config_cuda//cuda:cuda_headers",
        "@pybind11",
    ],
)

cc_library(
    name = "cuda_prng_kernels",
    srcs = [
        "cuda_prng_kernels.cc",
    ],
    hdrs = ["cuda_prng_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_prng_kernels_impl",
        ":kernel_helpers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cuda_library(
    name = "cuda_prng_kernels_impl",
    srcs = [
        "cuda_prng_kernels.cu.cc",
    ],
    hdrs = ["cuda_prng_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":kernel_helpers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_cuda_prng",
    srcs = ["cuda_prng.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_cuda_prng",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_prng_kernels",
        ":kernel_pybind11_helpers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@local_config_cuda//cuda:cuda_headers",
        "@pybind11",
    ],
)

# AMD GPU support (ROCm)
pybind_extension(
    name = "rocblas_kernels",
    srcs = ["rocblas.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "rocblas_kernels",
    deps = [
        ":handle_pool",
        ":kernel_pybind11_helpers",
        ":rocm_gpu_kernel_helpers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_config_rocm//rocm:rocblas",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:rocsolver",
        "@pybind11",
    ],
)

cc_library(
    name = "gpu_kernels",
    srcs = ["gpu_kernels.cc"],
    deps = [
        ":cublas_kernels",
        ":cuda_lu_pivot_kernels",
        ":cuda_prng_kernels",
        ":cusolver_kernels",
        ":cusparse_kernels",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_target_registry",
    ],
    alwayslink = 1,
)
