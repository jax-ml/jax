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

load("@rules_python//python:py_library.bzl", "py_library")
load(
    "//jaxlib:jax.bzl",
    "jax_multiprocess_generate_backend_suites",
    "jax_multiprocess_test",
    "py_deps",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:__subpackages__"],
)

py_library(
    name = "multiprocess_test",
    # testonly = 1,
    srcs = ["multiprocess_test.py"],
    deps = py_deps([
        "portpicker",
        "absl-all",
    ]),
)

jax_multiprocess_generate_backend_suites()

jax_multiprocess_test(
    name = "axis_index_test",
    srcs = ["axis_index_test.py"],
    main = "axis_index_test.py",
    deps = [
        "//third_party/py/absl/testing:parameterized",
    ],
)
