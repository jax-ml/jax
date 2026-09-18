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

ELEMENTARY_OPS = [
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "cbrt",
    "cos",
    "cosh",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log10",
    "log1p",
    "log2",
    "logistic",
    "reciprocal",
    "rsqrt",
    "sin",
    "sinc",
    "sinh",
    "sqrt",
    "square",
    "tan",
    "tanh",
]

[
    jax_multiplatform_test(
        name = "elementary_test_" + op,
        srcs = ["elementary_test.py"],
        args = ["--test_targets=ElementaryTest.test_" + op + "_accuracy_"],
        disable_configs = [
            "cpu_x32",
            "gpu_a100",
            "gpu_a100_x32",
            "gpu_h100_x32",
            "gpu_p100",
            "gpu_p100_x32",
            "gpu_v100",
            "gpu_v100_x32",
            "tpu_v4_compilation_cache",
        ],
        tags = ["cpu:4"],
        deps = [
            ":numerics_test_util",
        ] + py_deps([
            "absl/testing",
            "mpmath",
            "numpy",
        ]),
    )
    for op in ELEMENTARY_OPS
]

test_suite(
    name = "elementary_test",
    tests = [":elementary_test_" + op for op in ELEMENTARY_OPS],
)

SPECIAL_OPS = [
    "bessel_i0e",
    "bessel_i1e",
    "digamma",
    "erf",
    "erfc",
    "erfinv",
    "lgamma",
]

[
    jax_multiplatform_test(
        name = "special_test_" + op,
        srcs = ["special_test.py"],
        args = ["--test_targets=SpecialTest.test_" + op + "_accuracy_"],
        disable_configs = [
            "cpu_x32",
            "gpu_a100",
            "gpu_a100_x32",
            "gpu_h100_x32",
            "gpu_p100",
            "gpu_p100_x32",
            "gpu_v100",
            "gpu_v100_x32",
            "tpu_v4_compilation_cache",
        ],
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
    for op in SPECIAL_OPS
]

test_suite(
    name = "special_test",
    tests = [":special_test_" + op for op in SPECIAL_OPS],
)

jax_multiplatform_test(
    name = "numerics_test_util_test",
    srcs = ["numerics_test_util_test.py"],
    disable_configs = [
        "cpu_x32",
        "gpu_a100",
        "gpu_a100_x32",
        "gpu_h100_x32",
        "gpu_p100",
        "gpu_p100_x32",
        "gpu_v100",
        "gpu_v100_x32",
        "tpu_v4_compilation_cache",
    ],
    deps = [
        ":numerics_test_util",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)
