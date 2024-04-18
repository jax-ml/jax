# Copyright 2024 The JAX Authors.
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

load("@rules_python//python:defs.bzl", "py_library")
load("//jaxlib:jax.bzl", "pybind_extension")

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)

py_library(
    name = "mosaic_gpu",
    data = [":libmlir_cuda_runtime.so"],
    deps = [
        ":_mosaic_gpu_ext",
        "//jaxlib/mlir:execution_engine",
        "//jaxlib/mlir:gpu_dialect",
        "//jaxlib/mlir:llvm_dialect",
        "//jaxlib/mlir:nvgpu_dialect",
        "//jaxlib/mlir:nvvm_dialect",
    ],
)

pybind_extension(
    name = "_mosaic_gpu_ext",
    srcs = ["_mosaic_gpu_ext.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    linkopts = select({
        "@xla//xla/python:use_jax_cuda_pip_rpaths": [
            "-Wl,-rpath,$$ORIGIN/../../../nvidia/cuda_runtime/lib",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//jaxlib:kernel_nanobind_helpers",
        "@xla//xla/service:custom_call_status",
        "@nanobind",
    ],
)

cc_binary(
    name = "libmlir_cuda_runtime.so",
    srcs = ["@llvm-project//mlir:lib/ExecutionEngine/CudaRuntimeWrappers.cpp"],
    copts = ["-fvisibility=default"],
    linkopts = select({
        "@xla//xla/python:use_jax_cuda_pip_rpaths": [
            "-Wl,-rpath,$$ORIGIN/../../../nvidia/cuda_runtime/lib",
        ],
        "//conditions:default": [],
    }),
    linkshared = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "@llvm-project//mlir:mlir_c_runner_utils_hdrs",
        "@xla//xla/tsl/cuda:cudart",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)
