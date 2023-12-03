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

load(
    "//jaxlib:jax.bzl",
    "jax_visibility",
    "py_library_providing_imports_info",
    "pytype_strict_library",
)

package(default_applicable_licenses = ["//jax:license"])

py_library_providing_imports_info(
    name = "lib",
    srcs = [
        "__init__.py",
        "mlir/__init__.py",
        "mlir/dialects/__init__.py",
    ],
    lib_rule = pytype_strict_library,
    visibility = ["//jax:internal"] + jax_visibility("lib"),
    deps = [
        "//jax:version",
    ] + select({
        "//jax:enable_jaxlib_build": [
            "//jaxlib",
            "//jaxlib:cpu_feature_guard",
            "//jaxlib:utils",
            "//jaxlib/mlir:arithmetic_dialect",
            "//jaxlib/mlir:builtin_dialect",
            "//jaxlib/mlir:chlo_dialect",
            "//jaxlib/mlir:func_dialect",
            "//jaxlib/mlir:ir",
            "//jaxlib/mlir:math_dialect",
            "//jaxlib/mlir:memref_dialect",
            "//jaxlib/mlir:mhlo_dialect",
            "//jaxlib/mlir:pass_manager",
            "//jaxlib/mlir:scf_dialect",
            "//jaxlib/mlir:sparse_tensor_dialect",
            "//jaxlib/mlir:stablehlo_dialect",
            "//jaxlib/mlir:vector_dialect",
            # xla_client
        ],
        "//conditions:default": [],
    }),
)
