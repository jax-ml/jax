# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//jaxlib:jax.bzl", "py_library")

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:__subpackages__"],
)

py_library(
    name = "thunky_gpu_support",
    deps = [
        ":thunky_support",
    ],
)

cc_library(
    name = "thunk_proto_util",
    hdrs = ["thunk_proto_util.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@xla//xla/backends/gpu/runtime:collective_kernel_thunk_proto_cc",
        "@xla//xla/backends/gpu/runtime:collective_thunk_proto_cc",
        "@xla//xla/backends/gpu/runtime:convolution_filter_thunk_proto_cc",
        "@xla//xla/backends/gpu/runtime:copy_thunk_proto_cc",
        "@xla//xla/backends/gpu/runtime:thunk_proto_cc",
        "@xla//xla/service:buffer_assignment_proto_cc",
        "@xla//xla/service:shaped_slice_proto_cc",
    ],
)

cc_library(
    name = "mlir_to_thunks",
    srcs = ["mlir_to_thunks.cc"],
    hdrs = ["mlir_to_thunks.h"],
    deps = [
        ":thunk_proto_util",
        "//jaxlib/thunky/dialect:thunky_dialect",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@tsl//tsl/platform:protobuf",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla:xla_proto_cc",
        "@xla//xla/backends/cpu:target_machine_options",
        "@xla//xla/backends/gpu/codegen/kernels:custom_kernel",
        "@xla//xla/backends/gpu/codegen/kernels:custom_kernel_proto_cc",
        "@xla//xla/backends/gpu/runtime:all_gather_thunk",
        "@xla//xla/backends/gpu/runtime:all_reduce_thunk",
        "@xla//xla/backends/gpu/runtime:all_to_all_thunk",
        "@xla//xla/backends/gpu/runtime:async_execution",
        "@xla//xla/backends/gpu/runtime:async_thunk",
        "@xla//xla/backends/gpu/runtime:collective_group_thunk",
        "@xla//xla/backends/gpu/runtime:collective_permute_thunk",
        "@xla//xla/backends/gpu/runtime:collective_thunk",
        "@xla//xla/backends/gpu/runtime:command",
        "@xla//xla/backends/gpu/runtime:conditional_thunk",
        "@xla//xla/backends/gpu/runtime:custom_call_thunk",
        "@xla//xla/backends/gpu/runtime:custom_kernel_thunk",
        "@xla//xla/backends/gpu/runtime:device_to_device_copy_thunk",
        "@xla//xla/backends/gpu/runtime:execution_stream_id",
        "@xla//xla/backends/gpu/runtime:memset_thunk",
        "@xla//xla/backends/gpu/runtime:p2p_thunk_common",
        "@xla//xla/backends/gpu/runtime:sequential_thunk",
        "@xla//xla/backends/gpu/runtime:thunk",
        "@xla//xla/backends/gpu/runtime:thunk_proto_cc",
        "@xla//xla/backends/gpu/runtime:thunk_proto_deserialization",
        "@xla//xla/backends/gpu/runtime:while_thunk",
        "@xla//xla/codegen/emitters:kernel_arguments",
        "@xla//xla/core/collectives:reduction_kind",
        "@xla//xla/ffi",
        "@xla//xla/ffi:attribute_map",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/mlir/utils:type_util",
        "@xla//xla/service:buffer_assignment",
        "@xla//xla/service:buffer_assignment_proto_cc",
        "@xla//xla/service:shaped_slice",
        "@xla//xla/service:shaped_slice_proto_cc",
        "@xla//xla/service/gpu:buffer_allocations",
        "@xla//xla/service/gpu:gpu_executable",
        "@xla//xla/service/gpu:gpu_executable_proto_cc",
        "@xla//xla/service/llvm_ir:buffer_assignment_util",
        "@xla//xla/stream_executor:command_buffer",
        "@xla//xla/stream_executor:device_address",
        "@xla//xla/stream_executor:device_description",
        "@xla//xla/stream_executor:kernel_spec",
        "@xla//xla/stream_executor:launch_dim",
        "@xla//xla/stream_executor:module_spec",
        "@xla//xla/stream_executor:platform",
        "@xla//xla/stream_executor:platform_manager",
        "@xla//xla/stream_executor:scoped_module_handle",
        "@xla//xla/stream_executor:stream",
        "@xla//xla/stream_executor:stream_executor_h",
    ],
)

cc_library(
    name = "thunky_support",
    srcs = ["custom_call.cc"],
    deps = [
        ":mlir_to_thunks",
        "//jaxlib/thunky/dialect:thunky_dialect",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@xla//xla:shape_util",
        "@xla//xla/backends/gpu/libraries/native_custom_call_thunks:native_custom_call_emitter_context",
        "@xla//xla/backends/gpu/libraries/native_custom_call_thunks:native_custom_call_handler_registration",
        "@xla//xla/backends/gpu/runtime:thunk",
        "@xla//xla/ffi",
        "@xla//xla/ffi:attribute_map",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service:buffer_assignment",
    ],
    alwayslink = True,
)

cc_library(
    name = "test_custom_calls",
    testonly = True,
    srcs = ["test_custom_calls.cc"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@xla//xla/backends/gpu:ffi",
        "@xla//xla/ffi",
        "@xla//xla/stream_executor:device_memory",
        "@xla//xla/stream_executor:stream",
    ],
    alwayslink = True,
)

cc_test(
    name = "mlir_to_thunks_test",
    srcs = ["mlir_to_thunks_test.cc"],
    deps = [
        ":mlir_to_thunks",
        "//jaxlib/thunky/dialect:thunky_dialect",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@xla//xla:shape_util",
        "@xla//xla/backends/gpu/codegen/kernels:custom_kernel",
        "@xla//xla/backends/gpu/runtime:async_thunk",
        "@xla//xla/backends/gpu/runtime:collective_group_thunk",
        "@xla//xla/backends/gpu/runtime:conditional_thunk",
        "@xla//xla/backends/gpu/runtime:custom_call_thunk",
        "@xla//xla/backends/gpu/runtime:custom_kernel_thunk",
        "@xla//xla/backends/gpu/runtime:device_to_device_copy_thunk",
        "@xla//xla/backends/gpu/runtime:memset_thunk",
        "@xla//xla/backends/gpu/runtime:thunk",
        "@xla//xla/backends/gpu/runtime:thunk_proto_cc",
        "@xla//xla/backends/gpu/runtime:while_thunk",
        "@xla//xla/ffi",
        "@xla//xla/service:buffer_assignment",
    ],
)
