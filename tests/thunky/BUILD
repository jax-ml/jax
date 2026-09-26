# Copyright 2026 The JAX Authors.
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
    "py_deps",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:private"],
)

jax_generate_backend_suites()

jax_multiplatform_test(
    name = "thunky_test",
    srcs = ["thunky_test.py"],
    enable_backends = ["gpu"],
    enable_configs = ["gpu_h100x2_full"],
    shard_count = 8,
    tags = ["multiaccelerator"],
    deps = [
        "//jax/experimental:mosaic_gpu",
        "//jax/experimental:pallas_experimental_gpu_ops",
        "//jax/experimental:pallas_mosaic_gpu",
        "//jax/experimental:thunky",
        "//jaxlib/thunky:test_custom_calls",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "collective_matmul_test",
    srcs = ["collective_matmul_test.py"],
    enable_backends = ["gpu"],
    enable_configs = ["gpu_h100x2_full"],
    env = {"XLA_FLAGS": "--xla_gpu_enable_command_buffer=+COLLECTIVES"},
    tags = ["multiaccelerator"],
    deps = [
        "//jax/experimental:mosaic_gpu",
        "//jax/experimental:pallas_experimental_gpu_ops",
        "//jax/experimental:pallas_mosaic_gpu",
        "//jax/experimental:thunky",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)
