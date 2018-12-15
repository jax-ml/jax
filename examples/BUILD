# Copyright 2018 Google LLC
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

licenses(["notice"])  # Apache 2

load("//tests:build_defs.bzl", "jax_test")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "libjax_depend",
    values = {"define": "libjax_depend=true"},
)

py_library(
    name = "datasets",
    srcs = ["datasets.py"],
)

py_binary(
    name = "mnist_classifier_fromscratch",
    srcs = ["mnist_classifier_fromscratch.py"],
    deps = [
        ":datasets",
    ],
)

py_binary(
    name = "mnist_classifier",
    srcs = ["mnist_classifier.py"],
    deps = [
        ":datasets",
    ],
)

py_binary(
    name = "mnist_vae",
    srcs = ["mnist_vae.py"],
    deps = [
        ":datasets",
    ],
)

py_binary(
    name = "resnet50",
    srcs = ["resnet50.py"],
    deps = [
        ":datasets",
    ],
)

py_binary(
    name = "kernel_lsq",
    srcs = ["kernel_lsq.py"],
)

jax_test(
    name = "examples_test",
    srcs = ["examples_test.py"],
)
