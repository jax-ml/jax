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

load(
    "//jaxlib:jax.bzl",
    "jax_generate_backend_suites",
    "jax_multiplatform_test",
    "jax_py_test",
    "py_deps",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:private"],
)

jax_generate_backend_suites()

jax_multiplatform_test(
    name = "gpu_test",
    srcs = ["gpu_test.py"],
    enable_backends = [],
    enable_configs = [
        "gpu_h100",
        "gpu_h100x2",
    ],
    shard_count = 8,
    tags = ["multiaccelerator"],
    deps = [
        "//jax:mosaic_gpu",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_py_test(
    name = "gpu_dialect_test",
    srcs = ["gpu_dialect_test.py"],
    deps = [
        "//jax",
        "//jax:mosaic_gpu",
        "//jax:test_util",
    ] + py_deps("absl/testing"),
)

jax_py_test(
    name = "gpu_layout_inference_test",
    srcs = ["gpu_layout_inference_test.py"],
    deps = [
        "//jax",
        "//jax:mosaic_gpu",
        "//jax:test_util",
    ] + py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "matmul_test",
    srcs = ["matmul_test.py"],
    enable_backends = [],
    enable_configs = ["gpu_h100"],
    shard_count = 5,
    deps = [
        "//jax:mosaic_gpu",
        "//jax/experimental/mosaic/gpu/examples:matmul",
    ] + py_deps("absl/testing") + py_deps("numpy") + py_deps("hypothesis"),
)

jax_multiplatform_test(
    name = "flash_attention",
    srcs = ["//jax/experimental/mosaic/gpu/examples:flash_attention.py"],
    enable_backends = [],
    enable_configs = ["gpu_h100"],
    main = "//jax/experimental/mosaic/gpu/examples:flash_attention.py",
    tags = ["notap"],
    deps = [
        "//jax:mosaic_gpu",
    ] + py_deps("numpy"),
)

jax_multiplatform_test(
    name = "flash_attention_test",
    srcs = ["flash_attention_test.py"],
    enable_backends = [],
    enable_configs = ["gpu_h100"],
    deps = [
        "//jax:mosaic_gpu",
        "//jax/experimental/mosaic/gpu/examples:flash_attention",
    ] + py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "profiler_cupti_test",
    srcs = ["profiler_cupti_test.py"],
    enable_backends = [],
    enable_configs = ["gpu_h100"],
    tags = [
        "noasan",  # CUPTI leaks memory
        "nomsan",
    ],
    deps = [
        "//jax:mosaic_gpu",
    ] + py_deps("absl/testing"),
)
