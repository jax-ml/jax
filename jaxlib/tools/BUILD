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

load("@rules_python//python:py_test.bzl", "py_test")
load("//jaxlib:jax.bzl", "PLATFORM_TAGS_DICT", "pytype_strict_library")

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:private"],
)

py_test(
    name = "wheel_size_test",
    srcs = ["wheel_size_test.py"],
    tags = [
        "manual",
        "notap",
    ],
)

genrule(
    name = "platform_tags_py",
    srcs = [],
    outs = ["platform_tags.py"],
    cmd = "echo 'PLATFORM_TAGS_DICT = %s' > $@" % PLATFORM_TAGS_DICT,
)

pytype_strict_library(
    name = "build_utils",
    srcs = [
        "build_utils.py",
        ":platform_tags_py",
    ],
    tags = ["notap"],
)

pytype_strict_library(
    name = "build_gpu_kernels_wheel",
    srcs = ["build_gpu_kernels_wheel.py"],
    tags = ["notap"],
    deps = [
        ":build_utils",
        "@rules_python//python/runfiles",
    ],
)

pytype_strict_library(
    name = "build_gpu_plugin_wheel",
    srcs = ["build_gpu_plugin_wheel.py"],
    tags = ["notap"],
    deps = [
        ":build_utils",
        "@rules_python//python/runfiles",
    ],
)

pytype_strict_library(
    name = "build_mosaic_wheel",
    srcs = ["build_mosaic_wheel.py"],
    tags = ["notap"],
    deps = [
        ":build_utils",
        "@rules_python//python/runfiles",
    ],
)

pytype_strict_library(
    name = "build_wheel",
    srcs = ["build_wheel.py"],
    tags = ["notap"],
    deps = [
        ":build_utils",
        "@rules_python//python/runfiles",
    ],
)
