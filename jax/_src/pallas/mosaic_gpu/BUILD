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

# Package for Mosaic-specific Pallas extensions

load("//jaxlib:jax.bzl", "pytype_strict_library")
load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = [],
    default_visibility = [
        "//third_party/py/jax:internal",
    ],
)

py_library(
    name = "mosaic_gpu",
    srcs = ["__init__.py"],
    deps = [
        ":pallas_call_registration",
    ],
)

pytype_strict_library(
    name = "pallas_call_registration",
    srcs = ["pallas_call_registration.py"],
    deps = [
        ":lowering",
        "//third_party/py/jax",
        "//third_party/py/jax:mlir",
        "//third_party/py/jax:mosaic_gpu",
        "//third_party/py/jax/_src/pallas",
    ],
)

pytype_strict_library(
    name = "lowering",
    srcs = ["lowering.py"],
    deps = [
        "//third_party/py/jax",
        "//third_party/py/jax:core",
        "//third_party/py/jax:mlir",
        "//third_party/py/jax:mosaic_gpu",
        "//third_party/py/jax:util",
        "//third_party/py/jax/_src/lib",
        "//third_party/py/jax/_src/pallas",
        "//third_party/py/numpy",
    ],
)
