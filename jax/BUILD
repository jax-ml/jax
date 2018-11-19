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

licenses(["notice"])  # Apache 2

package(default_visibility = ["//visibility:public"])

# top-level EF placeholder

py_library(
    name = "libjax",
    srcs = glob(
        [
            "*.py",
            "lib/*.py",
            "interpreters/*.py",
            "numpy/*.py",
            "scipy/*.py",
        ],
        exclude = [
            "*_test.py",
            "**/*_test.py",
        ],
    ),
    deps = ["@org_tensorflow//tensorflow/compiler/xla/python:xla_client"],
)

py_library(
    name = "stax",
    srcs = ["experimental/stax.py"],
    deps = [":libjax"],
)

py_library(
    name = "minmax",
    srcs = ["experimental/minmax.py"],
    deps = [":libjax"],
)

py_library(
    name = "lapax",
    srcs = ["experimental/lapax.py"],
    deps = [":libjax"],
)

# this is a dummy target for building purposes
py_binary(
    name = "build_jax",
    srcs = ["__init__.py"],
    main = "__init__.py",
    deps = [":libjax"],
)
