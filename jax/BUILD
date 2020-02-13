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

pytype_library = py_library

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

# top-level EF placeholder

pytype_library(
    name = "jax",
    srcs = glob(
        [
            "*.py",
            "lib/*.py",
            "interpreters/*.py",
            "lax/*.py",
            "numpy/*.py",
            "ops/*.py",
            "scipy/*.py",
            "scipy/stats/*.py",
            "third_party/*.py",
            "third_party/numpy/*.py",
            "nn/*.py",
        ],
        exclude = [
            "*_test.py",
            "**/*_test.py",
        ],
    ),
    srcs_version = "PY3",
    deps = ["@org_tensorflow//tensorflow/compiler/xla/python:xla_client"],
)

pytype_library(
    name = "stax",
    srcs = ["experimental/stax.py"],
    srcs_version = "PY3",
    deps = [":jax"],
)

pytype_library(
    name = "optimizers",
    srcs = ["experimental/optimizers.py"],
    srcs_version = "PY3",
    deps = [":jax"],
)

pytype_library(
    name = "optix",
    srcs = ["experimental/optix.py"],
    srcs_version = "PY3",
    deps = [":jax"],
)

pytype_library(
    name = "ode",
    srcs = ["experimental/ode.py"],
    srcs_version = "PY3",
    deps = [":jax"],
)

pytype_library(
    name = "vectorize",
    srcs = ["experimental/vectorize.py"],
    srcs_version = "PY3",
    deps = [":jax"],
)

pytype_library(
    name = "loops",
    srcs = ["experimental/loops.py"],
    srcs_version = "PY3",
    deps = [":jax"],
)
