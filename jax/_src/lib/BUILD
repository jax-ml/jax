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
    "//jax/_src/lib:jax.bzl",
    "if_windows",
    "jax_visibility",
    "py_library_providing_imports_info",
    "pybind_extension",
    "pytype_library",
    "pytype_strict_library",
)
load("//jax/_src/lib:symlink_files.bzl", "symlink_files")

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)

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
            ":cpu_feature_guard",
            ":jaxlib",
            ":utils",
            "//jax/_src/lib/mlir:arithmetic_dialect",
            "//jax/_src/lib/mlir:builtin_dialect",
            "//jax/_src/lib/mlir:chlo_dialect",
            "//jax/_src/lib/mlir:func_dialect",
            "//jax/_src/lib/mlir:ir",
            "//jax/_src/lib/mlir:math_dialect",
            "//jax/_src/lib/mlir:memref_dialect",
            "//jax/_src/lib/mlir:mhlo_dialect",
            "//jax/_src/lib/mlir:ml_program_dialect",
            "//jax/_src/lib/mlir:pass_manager",
            "//jax/_src/lib/mlir:scf_dialect",
            "//jax/_src/lib/mlir:sparse_tensor_dialect",
            "//jax/_src/lib/mlir:stablehlo_dialect",
            "//jax/_src/lib/mlir:vector_dialect",
            # xla_client
        ],
        "//conditions:default": [],
    }),
)

py_library_providing_imports_info(
    name = "jaxlib",
    srcs = [
        "ducc_fft.py",
        "gpu_common_utils.py",
        "gpu_linalg.py",
        "gpu_prng.py",
        "gpu_rnn.py",
        "gpu_solver.py",
        "gpu_sparse.py",
        "gpu_triton.py",
        "hlo_helpers.py",
        "lapack.py",
        "tpu_mosaic.py",
        ":version",
        ":xla_client",
    ],
    data = [":xla_extension"],
    lib_rule = pytype_library,
    deps = [
        ":cpu_feature_guard",
        ":utils",
        "//jax/_src/lib/cpu:_ducc_fft",
        "//jax/_src/lib/cpu:_lapack",
        "//jax/_src/lib/mlir",
        "//jax/_src/lib/mlir:builtin_dialect",
        "//jax/_src/lib/mlir:chlo_dialect",
        "//jax/_src/lib/mlir:func_dialect",
        "//jax/_src/lib/mlir:ir",
        "//jax/_src/lib/mlir:memref_dialect",
        "//jax/_src/lib/mlir:mhlo_dialect",
        "//jax/_src/lib/mlir:ml_program_dialect",
        "//jax/_src/lib/mlir:pass_manager",
        "//jax/_src/lib/mlir:scf_dialect",
        "//jax/_src/lib/mlir:sparse_tensor_dialect",
        "//jax/_src/lib/mlir:stablehlo_dialect",
        "//jax/_src/lib/mosaic",
        # xla_client
    ],
)

symlink_files(
    name = "version",
    srcs = ["//jax:version.py"],
    dst = ".",
    flatten = True,
)

symlink_files(
    name = "xla_client",
    srcs = ["@xla//xla/python:xla_client"],
    dst = ".",
    flatten = True,
)

symlink_files(
    name = "xla_extension",
    srcs = if_windows(
        ["@xla//xla/python:xla_extension.pyd"],
        ["@xla//xla/python:xla_extension.so"],
    ),
    dst = ".",
    flatten = True,
)

exports_files([
    "README.md",
    "setup.py",
])

cc_library(
    name = "absl_status_casters",
    hdrs = ["absl_status_casters.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "kernel_nanobind_helpers",
    hdrs = ["kernel_nanobind_helpers.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":kernel_helpers",
        "@tsl//tsl/python/lib/core:numpy",
        "@com_google_absl//absl/base",
        "@nanobind",
    ],
)

cc_library(
    name = "kernel_helpers",
    hdrs = ["kernel_helpers.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "handle_pool",
    hdrs = ["handle_pool.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

# This isn't a CPU kernel. This exists to catch cases where jaxlib is built for the wrong
# target architecture.
pybind_extension(
    name = "cpu_feature_guard",
    srcs = ["cpu_feature_guard.c"],
    module_name = "cpu_feature_guard",
    deps = [
        "@xla//third_party/python_runtime:headers",
    ],
)

pybind_extension(
    name = "utils",
    srcs = ["utils.cc"],
    module_name = "utils",
    deps = [
        "@xla//third_party/python_runtime:headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:inlined_vector",
        "@nanobind",
    ],
)

pybind_extension(
    name = "cuda_plugin_extension",
    srcs = ["cuda_plugin_extension.cc"],
    module_name = "cuda_plugin_extension",
    deps = [
        "@nanobind",
        ":kernel_nanobind_helpers",
        "@xla//third_party/python_runtime:headers",
        "@xla//xla:status",
        "@xla//xla:util",
        "@xla//xla/pjrt/c:pjrt_c_api_gpu_extension_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_helpers",
        # TODO(jieying): move to jaxlib after py_client_gpu is separated from py_client
        "@xla//xla/python:py_client_gpu",
        "@xla//xla/python:status_casters",
        "@tsl//tsl/cuda:cublas",
        "@tsl//tsl/cuda:cudart",
    ],
)

# CPU kernels

# TODO(phawkins): Remove this forwarding target.
cc_library(
    name = "cpu_kernels",
    visibility = ["//visibility:public"],
    deps = [
        "//jax/_src/lib/cpu:cpu_kernels",
    ],
    alwayslink = 1,
)

# TODO(phawkins): Remove this forwarding target.
cc_library(
    name = "gpu_kernels",
    visibility = ["//visibility:public"],
    deps = [
        "//jax/_src/lib/cuda:cuda_gpu_kernels",
    ],
    alwayslink = 1,
)
