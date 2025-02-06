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

load("//jax:py_deps.bzl", "transitive_py_deps")
load(
    "//jaxlib:jax.bzl",
    "jax_wheel",
)

transitive_py_deps(
    name = "transitive_py_deps",
    deps = [
        "//jax",
        "//jax:compilation_cache",
        "//jax:experimental",
        "//jax:experimental_colocated_python",
        "//jax:experimental_sparse",
        "//jax:lax_reference",
        "//jax:pallas_gpu_ops",
        "//jax:pallas_mosaic_gpu",
        "//jax:pallas_tpu_ops",
        "//jax:pallas_triton",
        "//jax:source_mapper",
        "//jax/_src/lib",
        "//jax/_src/pallas/mosaic_gpu",
        "//jax/experimental/jax2tf",
        "//jax/extend",
        "//jax/extend:ifrt_programs",
        "//jax/tools:jax_to_ir",
    ],
)

py_binary(
    name = "build_wheel",
    srcs = ["build_wheel.py"],
    deps = [
        "//jaxlib/tools:build_utils",
        "@pypi_build//:pkg",
        "@pypi_setuptools//:pkg",
        "@pypi_wheel//:pkg",
    ],
)

jax_wheel(
    name = "jax_wheel",
    no_abi = True,
    no_platform = True,
    source_files = [
        ":transitive_py_deps",
        "//jax:py.typed",
        "//jax:numpy/__init__.pyi",
        "//jax:_src/basearray.pyi",
        "AUTHORS",
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "setup.py",
    ],
    wheel_binary = ":build_wheel",
    wheel_name = "jax",
)
