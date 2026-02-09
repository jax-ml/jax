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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_python//python:defs.bzl", "py_library")
load("//jaxlib:jax.bzl", "if_oss", "nanobind_extension")

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax/experimental:mosaic_gpu_users"],
)

py_library(
    name = "mosaic_gpu",
    data = [":libmosaic_gpu_runtime.so"],
    deps = [":_mosaic_gpu_ext"],
)

cc_library(
    name = "mosaic_gpu_support",
    deps = [
        ":custom_call",
        ":runtime",
    ],
)

cc_library(
    name = "target",
    srcs = ["target.cc"],
    hdrs = ["target.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_library(
    name = "dump",
    srcs = ["dump.cc"],
    hdrs = ["dump.h"],
    deps = [
        ":library_paths",  # buildcleaner: keep
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@tsl//tsl/platform:path",
    ],
)

cc_test(
    name = "dump_test",
    srcs = ["dump_test.cc"],
    deps = [
        ":dump",
        "//testing/base/public:gunit_main",
        "@llvm-project//mlir:IR",
    ],
)

cc_test(
    name = "gpu_module_to_assembly_test",
    srcs = ["gpu_module_to_assembly_test.cc"],
    deps = [
        ":passes",
        "//jaxlib/mosaic/dialect/gpu:mosaic_gpu",
        "//testing/base/public:gunit_main",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:BufferizationInterfaces",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:ComplexToLLVM",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToLLVMIRTranslation",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:IndexToLLVM",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:NVVMTarget",
        "@llvm-project//mlir:NVVMToLLVM",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLDialect",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:UBToLLVM",
        "@llvm-project//mlir:VectorDialect",
        "@xla//xla/service/gpu/llvm_gpu_backend:nvptx_libdevice_path",
    ],
)

cc_library(
    name = "serde",
    srcs = ["serde.cc"],
    hdrs = ["serde.h"],
    deps = [
        "//jaxlib/mosaic:pass_boilerplate",
        "//jaxlib/mosaic:serde",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:VectorDialect",
    ],
)

cc_library(
    name = "passes",
    srcs = [
        "assembly_to_binary.cc",
        "gpu_module_to_assembly.cc",
        "launch_lowering.cc",
        "passes.cc",
    ],
    hdrs = [
        "assembly_to_binary.h",
        "gpu_module_to_assembly.h",
        "launch_lowering.h",
        "passes.h",
    ],
    features = ["asan_runtime_closure"],  # Details: https://github.com/google-ml-infra/rules_ml_toolchain/tree/main/cc/sanitizers#troubleshooting
    deps = [
        ":dump",
        "//jaxlib/mosaic:pass_boilerplate",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToGPURuntimeTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TargetLLVM",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:VectorDialect",
        "@xla//xla/service/gpu/llvm_gpu_backend:load_ir_module",
        "@xla//xla/stream_executor/cuda:compilation_options",
        "@xla//xla/stream_executor/cuda:compilation_provider",
        "@xla//xla/stream_executor/cuda:cuda_compute_capability",
        "@xla//xla/tsl/platform:statusor",
    ],
)

CAPI_SOURCES = [
    "integrations/c/passes.cc",
]

CAPI_HEADERS = [
    "integrations/c/passes.h",
]

# `mlir_capi` should be used only to wrap hardware-agnostic passes that need to be called directly
# from Python.
cc_library(
    name = "mlir_capi",
    srcs = CAPI_SOURCES,
    hdrs = CAPI_HEADERS,
    deps = [
        ":serde",
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
)

# Header-only target, used when using the C API from a separate shared library.
cc_library(
    name = "mlir_capi_headers",
    hdrs = CAPI_HEADERS,
    deps = [
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
)

# Alwayslink target, used when exporting the C API from a shared library.
cc_library(
    name = "mlir_capi_objects",
    srcs = CAPI_SOURCES,
    hdrs = CAPI_HEADERS,
    deps = [
        ":passes",
        "@llvm-project//mlir:CAPIIRObjects",
    ],
    alwayslink = True,
)

cc_library(
    name = "runtime",
    srcs = ["runtime.cc"],
    # Linker may prune these symbols if they are not explicitly exported.
    linkopts = [
        "-Wl,--export-dynamic-symbol='mosaic_gpu_*'",
        "-Wl,--export-dynamic-symbol='nvshmem_my_pe'",
        "-Wl,--export-dynamic-symbol='nvshmem_ptr'",
        "-Wl,--export-dynamic-symbol='nvshmemx_barrier_all_on_stream'",
        "-Wl,--export-dynamic-symbol='nvshmemx_cumodule_finalize'",
        "-Wl,--export-dynamic-symbol='nvshmemx_cumodule_init'",
        "-Wl,--export-dynamic-symbol='nvshmemx_init_status'",
        "-Wl,--export-dynamic-symbol='nvshmemx_mc_ptr'",
    ],
    deps = [
        ":nvshmem",
        "@local_config_cuda//cuda:cuda_headers",
    ],
    alwayslink = True,
)

cc_library(
    name = "nvshmem",
    hdrs = ["nvshmem.h"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@xla//xla/tsl/cuda:cudart",
    ],
)

cc_library(
    name = "custom_call",
    srcs = ["custom_call.cc"],
    linkopts = if_oss(
        [],  # We use a version script to ensure symbol visibility in the OSS build.
        [
            "-Wl,--export-dynamic-symbol='MosaicGpuCompile'",
            "-Wl,--export-dynamic-symbol='MosaicGpuUnload'",
        ],
    ),
    deps = [
        ":dump",
        ":library_paths",  # buildcleaner: keep
        ":nvshmem",
        ":passes",
        ":serde",
        ":target",
        "//jaxlib/cuda:cuda_vendor",
        "//jaxlib/mosaic/dialect/gpu:mosaic_gpu",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:ComplexToLLVM",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToLLVMIRTranslation",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:IndexToLLVM",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMIRTransforms",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:NVGPUDialect",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:NVVMTarget",
        "@llvm-project//mlir:NVVMToLLVM",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:UBToLLVM",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:VectorToLLVM",
        "@local_config_cuda//cuda:cuda_headers",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla:executable_run_options",
        "@xla//xla:status_macros",
        "@xla//xla/backends/gpu:ffi",
        "@xla//xla/backends/gpu/collectives:gpu_clique_key",
        "@xla//xla/backends/gpu/runtime:collective_clique_requests",
        "@xla//xla/backends/gpu/runtime:collective_cliques",
        "@xla//xla/backends/gpu/runtime:collective_execution",
        "@xla//xla/backends/gpu/runtime:collective_kernel_utils",
        "@xla//xla/backends/gpu/runtime:collective_metadata_thunk",
        "@xla//xla/backends/gpu/runtime:collective_params",
        "@xla//xla/core/collectives:rank_id",
        "@xla//xla/ffi",
        "@xla//xla/service:custom_call_status",
        "@xla//xla/service:custom_call_target_registry",
        "@xla//xla/service/gpu/llvm_gpu_backend:nvptx_libdevice_path",
        "@xla//xla/service/llvm_ir:llvm_command_line_options",
        "@xla//xla/stream_executor:device_description",
        "@xla//xla/stream_executor/cuda:assemble_compilation_provider",
        "@xla//xla/stream_executor/cuda:compilation_provider",
        "@xla//xla/stream_executor/cuda:compilation_provider_options",
        "@xla//xla/stream_executor/cuda:cuda_compute_capability",
        "@xla//xla/stream_executor/cuda:ptx_compiler_support",
        "@xla//xla/stream_executor/gpu:collective_kernel_metadata",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
    alwayslink = True,
)

cc_test(
    name = "custom_call_test",
    srcs = ["custom_call_test.cc"],
    tags = ["requires-gpu-sm90"],
    deps = [
        ":mosaic_gpu_support",
        "//testing/base/public:gunit_main",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/log:scoped_mock_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/hlo/parser:hlo_parser",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt/plugin/xla_gpu:xla_gpu_pjrt_client",
        "@xla//xla/service:gpu_plugin",
        "@xla//xla/stream_executor/cuda:cuda_platform",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:statusor",
    ],
)

nanobind_extension(
    name = "_mosaic_gpu_ext",
    srcs = ["mosaic_gpu_ext.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    deps = [
        "//jaxlib:kernel_nanobind_helpers",
        "//jaxlib/cuda:cuda_vendor",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/strings",
        "@nanobind",
        "@xla//xla/tsl/cuda:cudart",
    ],
)

cc_binary(
    name = "libmosaic_gpu_runtime.so",
    srcs = ["runtime.cc"],
    copts = ["-fvisibility=default"],
    linkshared = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":nvshmem",
        "@local_config_cuda//cuda:cuda_headers",
        "@xla//xla/tsl/cuda:cudart",
    ],
)

cc_library(
    name = "library_paths",
    hdrs = ["library_paths.h"],
)

cc_library(
    name = "tiled_layout",
    srcs = ["tiled_layout.cc"],
    hdrs = ["tiled_layout.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_test(
    name = "tiled_layout_test",
    srcs = ["tiled_layout_test.cc"],
    deps = [
        ":tiled_layout",
        "//testing/base/public:gunit_main",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":utils",
        "//testing/base/public:gunit_main",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Support",
    ],
)
