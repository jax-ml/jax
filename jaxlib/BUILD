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

load("@org_tensorflow//tensorflow/core/platform/default:build_config.bzl", "pyx_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

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
    ],
)

cc_library(
    name = "gpu_kernel_helpers",
    srcs = ["gpu_kernel_helpers.cc"],
    hdrs = ["gpu_kernel_helpers.h"],
    copts = [
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pyx_library(
    name = "lapack",
    srcs = ["lapack.pyx"],
    py_deps = ["@org_tensorflow//third_party/py/numpy"],
)

py_library(
    name = "jaxlib",
    srcs = [
        "cuda_prng.py",
        "cusolver.py",
        "version.py",
    ],
)

py_library(
    name = "gpu_support",
    deps = [
        ":cublas_kernels",
        ":cuda_prng_kernels",
        ":cusolver_kernels",
    ],
)

cc_library(
    name = "pytree_lib",
    srcs = ["pytree.cc"],
    hdrs = ["pytree.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@pybind11",
    ],
)

pybind_extension(
    name = "pytree",
    srcs = ["pytree_extension.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "pytree",
    deps = [
        ":pytree_lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@pybind11",
    ],
)

pybind_extension(
    name = "cublas_kernels",
    srcs = ["cublas.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "cublas_kernels",
    deps = [
        ":gpu_kernel_helpers",
        ":kernel_pybind11_helpers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cublas_lib",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cublas_headers",
        "@local_config_cuda//cuda:cuda_headers",
        "@pybind11",
    ],
)

pybind_extension(
    name = "cusolver_kernels",
    srcs = ["cusolver.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "cusolver_kernels",
    deps = [
        ":gpu_kernel_helpers",
        ":kernel_pybind11_helpers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cusolver_lib",
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

cuda_library(
    name = "cuda_prng_kernels_lib",
    srcs = ["cuda_prng_kernels.cu.cc"],
    hdrs = ["cuda_prng_kernels.h"],
    deps = [
        ":gpu_kernel_helpers",
        ":kernel_helpers",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "cuda_prng_kernels",
    srcs = ["cuda_prng_kernels.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "cuda_prng_kernels",
    deps = [
        ":cuda_prng_kernels_lib",
        ":kernel_pybind11_helpers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@local_config_cuda//cuda:cuda_headers",
        "@pybind11",
    ],
)
