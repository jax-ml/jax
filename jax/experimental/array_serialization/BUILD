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
    "py_deps",
    "pytype_library",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
)

pytype_library(
    name = "serialization",
    srcs = [
        "__init__.py",
        "serialization.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//jax",
    ] + py_deps([
        "numpy",
        "tensorstore",
        "absl/logging",
    ]),
)
