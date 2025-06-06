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
    "cc_proto_library",
    "if_oss",
    "jax_visibility",
    "nanobind_extension",
    "proto_library",
    "py_deps",
    "py_strict_test",
    "pytype_library",
    "pytype_strict_library",
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

package_group(
    name = "xla_python",
    includes = [
        "//jax:internal",
    ],
)

pytype_strict_library(
    name = "jaxlib",
    data = [":ffi_headers"],
    deps = [
        ":_jax",
        ":cpu_feature_guard",
        ":jax",
        ":jaxlib_files",
        ":utils",
        ":weakref_lru_cache",
        ":xla_client",
        "//jaxlib/cpu:_lapack",
        "//jaxlib/cpu:_sparse",
        "//jaxlib/mlir",
        "//jaxlib/mlir:arithmetic_dialect",
        "//jaxlib/mlir:builtin_dialect",
        "//jaxlib/mlir:chlo_dialect",
        "//jaxlib/mlir:control_flow_dialect",
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
        "//jaxlib/mlir/_mlir_libs:register_jax_dialects",
        "//jaxlib/mosaic",
        "//jaxlib/mosaic/python:gpu_dialect",
        "//jaxlib/mosaic/python:tpu_dialect",
        "//jaxlib/triton",
        "@xla//xla/python:_profiler",
    ],
)

pytype_library(
    name = "jaxlib_files",
    srcs = [
        "cpu_sparse.py",
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
        ":version",
    ],
    deps = [
        ":_jax",
        "//jaxlib/cpu:_lapack",
        "//jaxlib/cpu:_sparse",
        "//jaxlib/mlir:ir",
        "//jaxlib/mlir:stablehlo_dialect",
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
        ":_jax",
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
        "@xla//xla/python:_profiler",
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

nanobind_pywrap_extension(
    name = "_jax",
    srcs = ["xla.cc"],
    pytype_deps = py_deps(["numpy"]),
    pytype_srcs = glob(["_jax/*.pyi"]),
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":custom_call_sharding",
        ":dlpack",
        ":ffi",
        ":guard_lib",
        ":ifrt_proxy",
        ":jax_jit",
        ":mlir",
        ":nb_class_ptr",
        ":pjit",
        ":pmap_lib",
        ":py_client",
        ":python_ref_manager",
        ":pytree",
        ":sdy",
        ":traceback",
        ":util",
        ":xla_compiler",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@nanobind",
        "@tsl//tsl/platform",
        "@xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@xla//xla:literal",
        "@xla//xla:shape_util",
        "@xla//xla:types",
        "@xla//xla:util",
        "@xla//xla/backends/cpu/collectives:cpu_collectives",
        "@xla//xla/ffi:ffi_api",
        "@xla//xla/hlo/builder/lib:approx_topk_shape",
        "@xla//xla/pjrt:exceptions",
        "@xla//xla/pjrt:mlir_to_hlo",
        "@xla//xla/pjrt:pjrt_api",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_common",
        "@xla//xla/pjrt:pjrt_compiler",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_layout",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_external",
        "@xla//xla/pjrt/distributed",
        "@xla//xla/pjrt/distributed:client",
        "@xla//xla/pjrt/distributed:key_value_store_interface",
        "@xla//xla/pjrt/distributed:protocol_proto_cc",
        "@xla//xla/pjrt/distributed:service",
        "@xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/python:logging",
        "@xla//xla/python:nb_absl_flat_hash_map",
        "@xla//xla/python:nb_absl_span",
        "@xla//xla/python:pprof_profile_builder",
        "@xla//xla/python:refine_polymorphic_shapes",
        "@xla//xla/python:types",
        "@xla//xla/python:version",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt:plugin_program",
        "@xla//xla/python/ifrt:plugin_program_serdes",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/python/pjrt_ifrt:pjrt_attribute_map_util",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/distributed_runtime/preemption:preemption_sync_manager",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:status",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform/cloud:gcs_file_system",
        "@xla//xla/tsl/python/lib/core:numpy",
    ] + select({
        # gloo tcp transport only builds on linux
        "@xla//xla/tsl:macos": [
            "@gloo//:transport_uv",
            "@xla//xla/backends/cpu/collectives:gloo_collectives",
            "@xla//xla/backends/cpu/collectives:gloo_kv_store",
        ],
        "@xla//xla/tsl:windows": [],
        "//conditions:default": [
            ":py_socket_transfer",
            "@gloo//:transport_tcp",
            "@xla//xla/backends/cpu/collectives:gloo_collectives",
            "@xla//xla/backends/cpu/collectives:gloo_kv_store",
        ],
    }) + select({
        # mpitrampoline does not build on windows
        "@xla//xla/tsl:windows": [],
        # we support MPI collectives only in OSS builds
        "//conditions:default": if_oss(["@xla//xla/backends/cpu/collectives:mpi_collectives"]),
    }),
)

cc_library(
    name = "callback",
    srcs = [
        "callback.cc",
    ],
    hdrs = [
        "callback.h",
    ],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":python_ref_manager",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@nanobind",
        "@xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@xla//xla:comparison_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt:host_callback",
        "@xla//xla/pjrt:transpose",
        "@xla//xla/python:nb_numpy",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/python/lib/core:numpy",
    ],
)

cc_library(
    name = "config",
    srcs = ["config.cc"],
    hdrs = ["config.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":python_ref_manager",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@nanobind",
        "@xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@xla//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "custom_call_sharding",
    srcs = ["custom_call_sharding.cc"],
    hdrs = ["custom_call_sharding.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@nanobind",
        "@xla//third_party/python_runtime:headers",
        "@xla//xla:shape_util",
        "@xla//xla:util",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/hlo/utils:hlo_sharding_util",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/pjrt/c:pjrt_c_api_custom_partitioner_extension_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_helpers",
        "@xla//xla/python:custom_call_batch_partitioner",
        "@xla//xla/python:custom_partition_callback",
        "@xla//xla/python:debug_callback_partitioner",
        "@xla//xla/python:inspect_sharding",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "dlpack_support",
    srcs = ["dlpack_support.cc"],
    hdrs = ["dlpack_support.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@dlpack",
        "@xla//xla:util",
        "@xla//xla:xla_data_proto_cc",
    ],
)

cc_library(
    name = "dlpack",
    srcs = ["dlpack.cc"],
    hdrs = ["dlpack.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":dlpack_support",
        ":nb_class_ptr",
        ":py_client",
        ":python_ref_manager",
        ":traceback",
        ":util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@dlpack",
        "@llvm-project//llvm:Support",
        "@nanobind",
        "@xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@xla//xla:shape_util",
        "@xla//xla:status_macros",
        "@xla//xla:util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt:exceptions",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_common",
        "@xla//xla/pjrt:pjrt_compiler",
        "@xla//xla/python:types",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "ffi",
    srcs = ["ffi.cc"],
    hdrs = ["ffi.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":dlpack_support",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@dlpack",
        "@nanobind",
        "@xla//third_party/python_runtime:headers",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/ffi:ffi_api",
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/pjrt:host_callback",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/python:nb_numpy",
        "@xla//xla/python:types",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "guard_lib",
    srcs = ["guard_lib.cc"],
    hdrs = ["guard_lib.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@nanobind",
        "@xla//xla:util",
    ],
)

cc_library(
    name = "ifrt_proxy",
    srcs = ["ifrt_proxy.cc"],
    hdrs = ["ifrt_proxy.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":nb_class_ptr",
        ":py_client",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:log_entry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@nanobind",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt:attribute_map",
        "@xla//xla/python/ifrt_proxy/client:grpc_client",
        "@xla//xla/python/ifrt_proxy/client:registry",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "jax_jit",
    srcs = ["jax_jit.cc"],
    hdrs = ["jax_jit.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":py_client",
        ":python_ref_manager",
        ":pytree",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@nanobind",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//third_party/python_runtime:headers",  # build_cleaner: keep
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_layout",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/python:nb_absl_inlined_vector",
        "@xla//xla/python:nb_absl_span",
        "@xla//xla/python:types",
        "@xla//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "mlir",
    srcs = ["mlir.cc"],
    hdrs = ["mlir.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:Support",
        "@nanobind",
        "@stablehlo//:stablehlo_serialization",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/hlo/translate:stablehlo",
        "@xla//xla/mlir_hlo:mhlo_passes",
        "@xla//xla/pjrt:mlir_to_hlo",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/python:refine_polymorphic_shapes",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "nb_class_ptr",
    hdrs = ["nb_class_ptr.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = jax_visibility("jaxlib/nb_class_ptr"),
    deps = ["@nanobind"],
)

cc_library(
    name = "pjit",
    srcs = ["pjit.cc"],
    hdrs = ["pjit.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":config",
        ":guard_lib",
        ":jax_jit",
        ":nb_class_ptr",
        ":py_client",
        ":python_ref_manager",
        ":pytree",
        ":traceback",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@nanobind",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@xla//xla:shape_util",
        "@xla//xla:util",
        "@xla//xla/pjrt:exceptions",
        "@xla//xla/pjrt:lru_cache",
        "@xla//xla/python:nb_helpers",
        "@xla//xla/python:nb_numpy",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "pmap_lib",
    srcs = ["pmap_lib.cc"],
    hdrs = ["pmap_lib.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":config",
        ":jax_jit",
        ":nb_class_ptr",
        ":py_client",
        ":python_ref_manager",
        ":pytree",
        ":traceback",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@nanobind",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@xla//xla:status_macros",
        "@xla//xla:util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt:exceptions",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/python:nb_helpers",
        "@xla//xla/python:nb_numpy",
        "@xla//xla/python:safe_static_init",
        "@xla//xla/python:types",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/python/lib/core:numpy",
    ],
)

cc_library(
    name = "py_client",
    srcs = [
        "partition_spec.cc",
        "py_array.cc",
        "py_client.cc",
        "py_compile_only_client.cc",
        "py_device.cc",
        "py_device_list.cc",
        "py_executable.cc",
        "py_memory_space.cc",
        "py_program.cc",
        "py_values.cc",
        "sharding.cc",
        "to_ifrt_sharding.cc",
    ],
    hdrs = [
        "partition_spec.h",
        "py_array.h",
        "py_client.h",
        "py_compile_only_client.h",
        "py_device.h",
        "py_device_list.h",
        "py_executable.h",
        "py_memory_space.h",
        "py_program.h",
        "py_values.h",
        "sharded_device_array.h",
        "sharding.h",
        "to_ifrt_sharding.h",
    ],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = jax_visibility("jaxlib/py_client"),
    deps = [
        ":guard_lib",
        ":nb_class_ptr",
        ":py_client_cpu",
        ":py_host_callback",
        ":python_ref_manager",
        ":traceback",
        ":util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@nanobind",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:stablehlo_ops",
        "@tsl//tsl/platform:fingerprint",
        "@tsl//tsl/platform:ml_dtypes",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@xla//xla:literal",
        "@xla//xla:shape_util",
        "@xla//xla:status_macros",
        "@xla//xla:types",
        "@xla//xla:util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/pjrt:exceptions",
        "@xla//xla/pjrt:lru_cache",
        "@xla//xla/pjrt:mlir_to_hlo",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_compiler",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:pjrt_future",
        "@xla//xla/pjrt:pjrt_layout",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/python:nb_absl_span",
        "@xla//xla/python:nb_helpers",
        "@xla//xla/python:nb_numpy",
        "@xla//xla/python:pprof_profile_builder",
        "@xla//xla/python:safe_static_init",
        "@xla//xla/python:types",
        "@xla//xla/python:version",
        "@xla//xla/python/compile_only_ifrt:client",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt:attribute_map",
        "@xla//xla/python/ifrt:custom_call_program",
        "@xla//xla/python/ifrt:plugin_program",
        "@xla//xla/python/ifrt:plugin_program_serdes",
        "@xla//xla/python/ifrt:user_context",
        "@xla//xla/python/ifrt/hlo:hlo_program",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/python/pjrt_ifrt:pjrt_dtype",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/service:platform_util",
        "@xla//xla/service/spmd/shardy:constants",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/framework:allocator",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:status",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/python/lib/core:numpy",
    ],
)

cc_library(
    name = "py_client_cpu",
    srcs = ["py_client_cpu.cc"],
    hdrs = ["py_client_cpu.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":ffi",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@dlpack",
        "@nanobind",
        "@xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@xla//xla:shape_util",
        "@xla//xla:util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/ffi:ffi_api",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/pjrt:host_callback",
        "@xla//xla/pjrt:transpose",
        "@xla//xla/python:nb_numpy",
        "@xla//xla/python:types",
    ],
    alwayslink = 1,
)

cc_library(
    name = "py_host_callback",
    srcs = ["py_host_callback.cc"],
    hdrs = ["py_host_callback.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":callback",
        ":py_host_callback_cc_proto",
        ":python_ref_manager",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@nanobind",
        "@xla//xla:shape_util",
        "@xla//xla:status_macros",
        "@xla//xla:util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt:host_callback",
        "@xla//xla/python:types",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/python/pjrt_ifrt:xla_host_callback_proto_cc",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/platform:statusor",
    ],
)

proto_library(
    name = "py_host_callback_proto",
    srcs = ["py_host_callback.proto"],
)

cc_proto_library(
    name = "py_host_callback_cc_proto",
    visibility = jax_visibility("jaxlib/py_host_callback_cc_proto"),
    deps = [":py_host_callback_proto"],
)

cc_library(
    name = "py_socket_transfer",
    srcs = ["py_socket_transfer.cc"],
    hdrs = ["py_socket_transfer.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":nb_class_ptr",
        ":py_client",
        ":traceback",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Support",
        "@nanobind",
        "@tsl//tsl/platform:casts",
        "@xla//xla:util",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/python:nb_numpy",
        "@xla//xla/python:types",
        "@xla//xla/python:version",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/python/pjrt_ifrt:pjrt_dtype",
        "@xla//xla/python/transfer:event_loop",
        "@xla//xla/python/transfer:socket-server",
        "@xla//xla/python/transfer:socket_bulk_transport",
        "@xla//xla/python/transfer:streaming",
        "@xla//xla/python/transfer:streaming_ifrt",
        "@xla//xla/python/transfer:transfer_socket_proto_cc",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "python_ref_manager",
    srcs = ["python_ref_manager.cc"],
    hdrs = ["python_ref_manager.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = jax_visibility("jaxlib/python_ref_manager"),
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@nanobind",
        "@xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

proto_library(
    name = "pytree_proto",
    srcs = ["pytree.proto"],
)

cc_proto_library(
    name = "pytree_cc_proto",
    deps = [":pytree_proto"],
)

cc_library(
    name = "pytree",
    srcs = ["pytree.cc"],
    hdrs = ["pytree.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = jax_visibility("jaxlib/pytree"),
    deps = [
        ":nb_class_ptr",
        ":pytree_cc_proto",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@nanobind",
        "@xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@xla//xla/pjrt:exceptions",
        "@xla//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "sdy",
    srcs = ["sdy.cc"],
    hdrs = ["sdy.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@nanobind",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@xla//xla/hlo/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
        "@xla//xla/mlir_hlo:all_passes",
        "@xla//xla/pjrt:mlir_to_hlo",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/service/spmd/shardy:constants",
        "@xla//xla/service/spmd/shardy:utils",
        "@xla//xla/service/spmd/shardy/sdy_round_trip:import_shardy_attrs",
        "@xla//xla/service/spmd/shardy/sdy_round_trip:pipelines",
        "@xla//xla/tsl/framework/mlir:status_scoped_diagnostic_handler",
    ],
)

cc_library(
    name = "traceback",
    srcs = ["traceback.cc"],
    hdrs = ["traceback.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = jax_visibility("jaxlib/traceback"),
    deps = [
        ":nb_class_ptr",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@nanobind",
        "@tsl//tsl/platform",
        "@xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@xla//xla/pjrt:exceptions",
        "@xla//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@nanobind",
        "@xla//xla:util",
        "@xla//xla/pjrt:pjrt_future",
        "@xla//xla/python:version",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/concurrency:async_value",
        "@xla//xla/tsl/concurrency:ref_count",
    ],
)

cc_library(
    name = "xla_compiler",
    srcs = ["xla_compiler.cc"],
    hdrs = ["xla_compiler.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":dlpack",
        ":py_client",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:Support",
        "@nanobind",
        "@xla//xla:array",
        "@xla//xla:debug_options_flags",
        "@xla//xla:literal",
        "@xla//xla:shape_util",
        "@xla//xla:util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla:xla_proto_cc",
        "@xla//xla/client:executable_build_options",
        "@xla//xla/ffi",
        "@xla//xla/ffi:ffi_api",
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/hlo/parser:hlo_parser",
        "@xla//xla/pjrt:exceptions",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
        "@xla//xla/python:nb_absl_span",
        "@xla//xla/python:nb_helpers",
        "@xla//xla/python:nb_numpy",
        "@xla//xla/python:types",
        "@xla//xla/service:computation_placer",
        "@xla//xla/service:custom_call_target_registry",
        "@xla//xla/service:hlo_graph_dumper",
        "@xla//xla/service:hlo_module_config",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/service:name_uniquer",
        "@xla//xla/service/spmd/shardy/stablehlo_round_trip:stablehlo_import",
        "@xla//xla/tsl/lib/strings:proto_serialization",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:statusor",
    ],
)

pytype_strict_library(
    name = "xla_client",
    srcs = ["xla_client.py"],
    pytype_srcs = ["xla_client.pyi"],
    visibility = [":xla_python"],
    deps = py_deps([
        "numpy",
        "ml_dtypes",
    ]) + [":_jax"],
)

py_strict_test(
    name = "pytree_test",
    srcs = ["pytree_test.py"],
    deps = [
        ":xla_client",
    ] + py_deps([
        "absl/flags",
        "absl/logging",
        "absl/testing",
    ]),
)

py_strict_test(
    name = "config_test",
    srcs = ["config_test.py"],
    deps = [
        ":xla_client",
    ] + py_deps([
        "absl/flags",
        "absl/logging",
        "absl/testing",
    ]),
)

py_strict_test(
    name = "jax_jit_test",
    srcs = ["jax_jit_test.py"],
    deps = [
        ":xla_client",
    ] + py_deps([
        "absl/flags",
        "absl/logging",
        "absl/testing",
        "numpy",
    ]),
)
