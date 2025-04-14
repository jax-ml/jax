# Copyright 2018 The JAX Authors.
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

load(
    "//jaxlib:jax.bzl",
    "nanobind_extension",
    "py_deps",
    "py_library_providing_imports_info",
    "py_strict_test",
    "pytype_library",
)
load(
    "//jaxlib:pywrap.bzl",
    "nanobind_pywrap_extension",
    "pywrap_binaries",
    "pywrap_library",
)
load("//jaxlib:symlink_files.bzl", "symlink_files")

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:internal"],
)

py_library_providing_imports_info(
    name = "jaxlib",
    srcs = [
        "gpu_common_utils.py",
        "gpu_linalg.py",
        "gpu_prng.py",
        "gpu_rnn.py",
        "gpu_solver.py",
        "gpu_sparse.py",
        "gpu_triton.py",
        "hlo_helpers.py",
        "init.py",
        "lapack.py",
        "plugin_support.py",
        "xla_client.py",
        "xla_extension.py",
        ":version",
    ],
    data = [":ffi_headers"],
    lib_rule = pytype_library,
    deps = [
        ":cpu_feature_guard",
        ":jax",
        ":utils",
        ":weakref_lru_cache",
        "//jaxlib/cpu:_lapack",
        "//jaxlib/mlir",
        "//jaxlib/mlir:arithmetic_dialect",
        "//jaxlib/mlir:builtin_dialect",
        "//jaxlib/mlir:chlo_dialect",
        "//jaxlib/mlir:func_dialect",
        "//jaxlib/mlir:gpu_dialect",
        "//jaxlib/mlir:ir",
        "//jaxlib/mlir:llvm_dialect",
        "//jaxlib/mlir:math_dialect",
        "//jaxlib/mlir:memref_dialect",
        "//jaxlib/mlir:mhlo_dialect",
        "//jaxlib/mlir:nvgpu_dialect",
        "//jaxlib/mlir:nvvm_dialect",
        "//jaxlib/mlir:pass_manager",
        "//jaxlib/mlir:scf_dialect",
        "//jaxlib/mlir:sdy_dialect",
        "//jaxlib/mlir:sparse_tensor_dialect",
        "//jaxlib/mlir:stablehlo_dialect",
        "//jaxlib/mlir:vector_dialect",
        "//jaxlib/mosaic",
        "//jaxlib/triton",
        "//jaxlib/xla:xla_client",
        "//jaxlib/xla:xla_extension",
    ],
)

symlink_files(
    name = "version",
    srcs = ["//jax:version.py"],
    dst = ".",
    flatten = True,
)

symlink_files(
    name = "ffi_headers",
    srcs = ["@xla//xla/ffi/api:all_headers"],
    dst = "include/xla/ffi/api",
    flatten = True,
)

exports_files([
    "README.md",
    "setup.py",
])

pywrap_library(
    name = "jax",
    common_lib_def_files_or_filters = {
        "jaxlib/jax_common": "jax_common.json",
    },
    common_lib_version_scripts = {
        "jaxlib/jax_common": select({
            "@bazel_tools//src/conditions:windows": None,
            "@bazel_tools//src/conditions:darwin": "libjax_common_darwin.lds",
            "//conditions:default": "libjax_common.lds",
        }),
    },
    deps = [
        ":utils",
        ":weakref_lru_cache",
        "//jaxlib/mlir/_mlir_libs:_chlo",
        "//jaxlib/mlir/_mlir_libs:_mlir",
        "//jaxlib/mlir/_mlir_libs:_mlirDialectsGPU",
        "//jaxlib/mlir/_mlir_libs:_mlirDialectsLLVM",
        "//jaxlib/mlir/_mlir_libs:_mlirDialectsNVGPU",
        "//jaxlib/mlir/_mlir_libs:_mlirDialectsSparseTensor",
        "//jaxlib/mlir/_mlir_libs:_mlirGPUPasses",
        "//jaxlib/mlir/_mlir_libs:_mlirHlo",
        "//jaxlib/mlir/_mlir_libs:_mlirSparseTensorPasses",
        "//jaxlib/mlir/_mlir_libs:_mosaic_gpu_ext",
        "//jaxlib/mlir/_mlir_libs:_sdy",
        "//jaxlib/mlir/_mlir_libs:_stablehlo",
        "//jaxlib/mlir/_mlir_libs:_tpu_ext",
        "//jaxlib/mlir/_mlir_libs:_triton_ext",
        "//jaxlib/mlir/_mlir_libs:register_jax_dialects",
        "//jaxlib/xla:xla_extension",
    ],
)

pywrap_binaries(
    name = "jaxlib_binaries",
    dep = ":jax",
)

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
    name = "ffi_helpers",
    hdrs = ["ffi_helpers.h"],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/ffi/api:ffi",
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
        "@com_google_absl//absl/base",
        "@nanobind",
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/tsl/python/lib/core:numpy",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

# This isn't a CPU kernel. This exists to catch cases where jaxlib is built for the wrong
# target architecture.
nanobind_extension(
    name = "cpu_feature_guard",
    srcs = ["cpu_feature_guard.c"],
    module_name = "cpu_feature_guard",
    deps = [
        "@xla//third_party/python_runtime:headers",
    ],
)

nanobind_pywrap_extension(
    name = "weakref_lru_cache",
    srcs = ["weakref_lru_cache.cc"],
    pytype_srcs = ["weakref_lru_cache.pyi"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@nanobind",
        "@xla//third_party/python_runtime:headers",
        "@xla//xla/pjrt:lru_cache",
        "@xla//xla/tsl/platform:logging",
    ],
)

py_strict_test(
    name = "weakref_lru_cache_test",
    srcs = ["weakref_lru_cache_test.py"],
    deps = [
        ":weakref_lru_cache",
    ] + py_deps([
        "absl/flags",
        "absl/logging",
        "absl/testing",
    ]),
)

nanobind_pywrap_extension(
    name = "utils",
    srcs = ["utils.cc"],
    deps = [
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/synchronization",
        "@nanobind",
        "@xla//third_party/python_runtime:headers",
    ],
)
