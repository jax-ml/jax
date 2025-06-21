# Copyright 2023 The JAX Authors.
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

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax/experimental/jax2tf:__subpackages__"],
)

py_library(
    name = "back_compat_testdata",
    srcs = glob(["*.py"]),
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/typing_extensions",
    ],
)
