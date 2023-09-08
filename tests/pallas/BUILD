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

load("@rules_python//python:defs.bzl", "py_test")
load(
    "//jaxlib:jax.bzl",
    "jax_test",
    "py_deps",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:private"],
)

jax_test(
    name = "pallas_test",
    srcs = [
        "pallas_test.py",
    ],
    config_tags_overrides = {
        "gpu_x32": {
            "ondemand": False,  # Include in presubmit.
        },
    },
    disable_backends = [
        "cpu",
        "tpu",
    ],
    disable_configs = [
        "gpu",
        "gpu_a100",
        "gpu_p100",
    ],
    enable_configs = [
        "gpu_x32",
        "gpu_a100_x32",
    ],
    shard_count = 4,
    deps = [
        "//third_party/py/jax:pallas_gpu",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

py_test(
    name = "indexing_test",
    srcs = [
        "indexing_test.py",
    ],
    deps = [
        "//third_party/py/jax:pallas",
    ] + py_deps("absl/testing") + py_deps("hypothesis") + py_deps("numpy"),
)
