# Copyright 2022 The JAX Authors.
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

# Note: these examples were imported May 26, 2022 and may be out of sync.
licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = [
        "//jax/experimental/jax2tf:__subpackages__",
        "//third_party/tensorflow/lite/experimental/mlir/testing/jax/test_models:__subpackages__",
    ],
)

py_library(
    name = "flax_models",
    srcs = glob(["*.py"]),
    deps = [
        "//jax",
        "//third_party/py/flax:core",
        "//third_party/py/jraph",
        "//third_party/py/numpy",
        "//third_party/py/typing_extensions",
    ],
)
