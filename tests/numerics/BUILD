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

DISABLED_CONFIGS = [
    "cpu_x32",
    "gpu_a100",
    "gpu_a100_x32",
    "gpu_h100_x32",
    "gpu_p100",
    "gpu_p100_x32",
    "gpu_v100",
    "gpu_v100_x32",
    "tpu_v4_compilation_cache",
]

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
        "absl/testing",
        "google_benchmark",
        "mpmath",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "bessel_test",
    srcs = ["bessel_test.py"],
    disable_configs = DISABLED_CONFIGS,
    shard_count = 2,
    tags = ["cpu:4"],
    deps = [
        ":numerics_test_util",
    ] + py_deps([
        "absl/testing",
        "mpmath",
        "numpy",
        "scipy",
    ]),
)

jax_multiplatform_test(
    name = "erf_test",
    srcs = ["erf_test.py"],
    disable_configs = DISABLED_CONFIGS,
    shard_count = 5,
    tags = ["cpu:4"],
    deps = [
        ":numerics_test_util",
    ] + py_deps([
        "absl/testing",
        "mpmath",
        "numpy",
        "scipy",
    ]),
)

jax_multiplatform_test(
    name = "exp_test",
    srcs = ["exp_test.py"],
    disable_configs = DISABLED_CONFIGS,
    shard_count = 4,
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
    name = "gamma_test",
    srcs = ["gamma_test.py"],
    disable_configs = DISABLED_CONFIGS,
    shard_count = 2,
    tags = ["cpu:4"],
    deps = [
        ":numerics_test_util",
    ] + py_deps([
        "absl/testing",
        "mpmath",
        "numpy",
        "scipy",
    ]),
)

jax_multiplatform_test(
    name = "hyperbolic_test",
    srcs = ["hyperbolic_test.py"],
    disable_configs = DISABLED_CONFIGS,
    shard_count = 6,
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
    name = "log_test",
    srcs = ["log_test.py"],
    disable_configs = DISABLED_CONFIGS,
    shard_count = 4,
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
    name = "power_test",
    srcs = ["power_test.py"],
    disable_configs = DISABLED_CONFIGS,
    shard_count = 5,
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
    name = "trig_test",
    srcs = ["trig_test.py"],
    disable_configs = DISABLED_CONFIGS,
    shard_count = 7,
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
    disable_configs = DISABLED_CONFIGS,
    deps = [
        ":numerics_test_util",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)
