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

# JAX is Autograd and XLA

load("@org_tensorflow//tensorflow/core:platform/default/build_config.bzl", "pyx_library")

licenses(["notice"])  # Apache 2

package(default_visibility = ["//visibility:public"])

pyx_library(
    name = "lapack",
    srcs = ["lapack.pyx"],
    py_deps = [
        "@org_tensorflow//third_party/py/numpy",
    ],
)
