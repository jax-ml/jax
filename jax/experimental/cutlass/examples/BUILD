# Copyright 2025 The JAX Authors.
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
load("//jaxlib:jax.bzl", "jax_multiplatform_test", "py_deps")

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:cutlass_dsl_gpu_users"],
)

exports_files(
    srcs = [
        "blackwell_dense_gemm.py",
        "blackwell_dense_gemm.py",
    ],
    visibility = ["//jax:internal"],
)

py_library(
    name = "blackwell_dense_gemm",
    srcs = ["blackwell_dense_gemm.py"],
    deps = [
        "//jax",
        "//jax:cutlass_dsl",
    ],
)

py_library(
    name = "blackwell_grouped_gemm",
    srcs = ["blackwell_grouped_gemm.py"],
    deps = [
        "//jax",
        "//jax:cutlass_dsl",
    ],
)

jax_multiplatform_test(
    name = "blackwell_dense_gemm",
    srcs = ["blackwell_dense_gemm.py"],
    enable_backends = [],
    enable_configs = ["gpu_b200"],
    main = "blackwell_dense_gemm.py",
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//jax:cutlass_dsl",
    ] + py_deps("numpy"),
)

jax_multiplatform_test(
    name = "blackwell_grouped_gemm",
    srcs = ["blackwell_grouped_gemm.py"],
    enable_backends = [],
    enable_configs = ["gpu_b200"],
    main = "blackwell_grouped_gemm.py",
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//jax:cutlass_dsl",
    ] + py_deps("numpy"),
)
