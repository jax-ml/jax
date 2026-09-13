# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//jaxlib:jax.bzl", "py_library")

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:__subpackages__"],
)

py_library(
    name = "thunky_dialect",
    srcs = [
        "thunky.py",
        "//jaxlib/thunky/dialect:_thunky_ops_gen.py",
    ],
    strict_deps = False,
    deps = [
        "//jaxlib/mlir",
        "//jaxlib/mlir/_mlir_libs:_thunky_ext",
    ],
)
