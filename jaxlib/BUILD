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
    "flatbuffer_cc_library",
    "pybind_extension",
)

licenses(["notice"])

package(default_visibility = ["//:__subpackages__"])

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

py_library(
    name = "jaxlib",
    srcs = [
        "gpu_linalg.py",
        "gpu_prng.py",
        "gpu_solver.py",
        "gpu_sparse.py",
        "init.py",
        "lapack.py",
        "mhlo_helpers.py",
        "pocketfft.py",
        "version.py",
    ],
    deps = [
        ":_lapack",
        ":_pocketfft",
        ":cpu_feature_guard",
    ],
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
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@com_google_absl//absl/base:dynamic_annotations",
    ],
)

cc_library(
    name = "lapack_kernels_using_lapack",
    srcs = ["lapack_kernels_using_lapack.cc"],
    deps = [":lapack_kernels"],
    alwayslink = 1,
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

cc_library(
    name = "pocketfft_kernels",
    srcs = ["pocketfft_kernels.cc"],
    hdrs = ["pocketfft_kernels.h"],
    copts = ["-fexceptions"],  # PocketFFT may throw.
    features = ["-use_header_modules"],
    deps = [
        ":pocketfft_flatbuffers_cc",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
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
        ":pocketfft_flatbuffers_cc",
        ":pocketfft_kernels",
        "@flatbuffers//:runtime_cc",
        "@pybind11",
    ],
)

cc_library(
    name = "cpu_kernels",
    srcs = ["cpu_kernels.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":lapack_kernels",
        ":lapack_kernels_using_lapack",
        ":pocketfft_kernels",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_target_registry",
    ],
    alwayslink = 1,
)

# TODO(phawkins): Remove this forwarding target.
cc_library(
    name = "gpu_kernels",
    visibility = ["//visibility:public"],
    deps = [
        "//jaxlib/cuda:cuda_gpu_kernels",
    ],
    alwayslink = 1,
)
