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

load(
    "//jaxlib:jax.bzl",
    "nanobind_extension",
    "py_deps",
    "py_strict_library",
    "py_strict_test",
    "pytype_strict_library",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:internal"],
)

package_group(
    name = "xla_python",
    includes = [
        "//jax:internal",
    ],
)

pytype_strict_library(
    name = "xla_client",
    srcs = ["xla_client.py"],
    pytype_srcs = ["xla_client.pyi"],
    visibility = [":xla_python"],
    deps = py_deps([
        "numpy",
        "ml_dtypes",
    ]) + ["@xla//xla/python:xla_extension"],
)

py_strict_test(
    name = "xla_client_backend_independent_test",
    srcs = ["xla_client_backend_independent_test.py"],
    deps = [
        ":xla_client",
    ] + py_deps([
        "absl/testing",
        "numpy",
        "portpicker",
    ]),
)

py_strict_library(
    name = "xla_client_test",
    testonly = 1,
    srcs = ["xla_client_test.py"],
    visibility = [":xla_python"],
    deps = [
        ":xla_client",
        "//jax",
        "//jax:test_util",
        "//jaxlib",
    ] + py_deps([
        "absl/flags",
        "absl/logging",
        "absl/testing",
        "ml_dtypes",
        "numpy",
    ]),
)

nanobind_extension(
    name = "custom_calls_testlib",
    testonly = 1,
    srcs = ["custom_calls_testlib.cc"],
    deps = [
        "@com_google_absl//absl/status",
        "@nanobind",
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/ffi/api:ffi",
    ],
)

py_strict_test(
    name = "xla_client_test_cpu",
    srcs = ["xla_client_test.py"],
    args = ["--backend=cpu"],
    env = {
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
    },
    main = "xla_client_test.py",
    deps = [
        ":custom_calls_testlib",
        ":xla_client",
        "//jax",
        "//jax:test_util",
        "//jaxlib",
    ] + py_deps([
        "absl/flags",
        "absl/logging",
        "absl/testing",
        "ml_dtypes",
        "numpy",
    ]),
)

py_strict_test(
    name = "weakref_lru_cache_test",
    srcs = ["weakref_lru_cache_test.py"],
    deps = [
        ":xla_client",
    ] + py_deps([
        "absl/flags",
        "absl/logging",
        "absl/testing",
    ]),
)

py_strict_test(
    name = "pytree_test",
    srcs = ["pytree_test.py"],
    deps = [
        ":xla_client",
    ] + py_deps([
        "absl/flags",
        "absl/logging",
        "absl/testing",
    ]),
)

py_strict_test(
    name = "config_test",
    srcs = ["config_test.py"],
    deps = [
        ":xla_client",
    ] + py_deps([
        "absl/flags",
        "absl/logging",
        "absl/testing",
    ]),
)

py_strict_test(
    name = "jax_jit_test",
    srcs = ["jax_jit_test.py"],
    deps = [
        ":xla_client",
    ] + py_deps([
        "absl/flags",
        "absl/logging",
        "absl/testing",
        "numpy",
    ]),
)
