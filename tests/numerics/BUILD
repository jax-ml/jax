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
    "py_library",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:private"],
)

jax_generate_backend_suites()

py_library(
    name = "numerics_test_util",
    testonly = True,
    srcs = ["numerics_test_util.py"],
    deps = [
        "//jax",
        "//jax/_src:api",
        "//jax/_src:test_util",
    ] + py_deps([
        "absl/flags",
        "mpmath",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "elementary_test",
    srcs = ["elementary_test.py"],
    disable_configs = [
        "cpu_x32",
        "gpu_a100_x32",
        "gpu_h100_x32",
        "gpu_p100_x32",
        "gpu_v100_x32",
    ],
    shard_count = 10,
    tags = ["cpu:4"],
    deps = [
        ":numerics_test_util",
    ] + py_deps([
        "absl/testing",
        "mpmath",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "numerics_test_util_test",
    srcs = ["numerics_test_util_test.py"],
    deps = [
        ":numerics_test_util",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)
