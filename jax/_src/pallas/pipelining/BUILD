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
    "pytype_strict_library",
)

package(
    default_applicable_licenses = [],
    default_visibility = [
        "//jax:internal",
    ],
)

pytype_strict_library(
    name = "internal",
    srcs = ["internal.py"],
    deps = [
        "//jax",
        "//jax/_src:core",
        "//jax/_src:state_types",
    ],
)

pytype_strict_library(
    name = "schedule_api",
    srcs = ["schedule_api.py"],
    deps = [
        ":internal",
        ":schedulers",
        "//jax",
        "//jax/_src:api_util",
        "//jax/_src:core",
        "//jax/_src:partial_eval",
        "//jax/_src:state_types",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "schedulers",
    srcs = ["schedulers.py"],
    deps = [
        ":internal",
        "//jax",
        "//jax/_src:core",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "pipeline_test_util",
    testonly = True,
    srcs = ["pipeline_test_util.py"],
    deps = [
        "//jax/_src:debugging",
        "//jax/_src/pallas/pipelining:internal",
        "//jax/_src/pallas/pipelining:schedulers",
    ] + py_deps([
        "absl/testing",
    ]),
)
