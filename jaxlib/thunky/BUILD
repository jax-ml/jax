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

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:internal"],
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
    name = "mlir_to_proto",
    srcs = ["mlir_to_proto.cc"],
    hdrs = ["mlir_to_proto.h"],
    deps = [
        ":thunk_proto_util",
        "//jaxlib/thunky/dialect:thunky_dialect",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@riegeli//riegeli/bytes:string_reader",
        "@tsl//tsl/platform:protobuf",
        "@xla//xla:literal",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla:xla_proto_cc",
        "@xla//xla/backends/gpu/codegen/kernels:custom_kernel_proto_cc",
        "@xla//xla/backends/gpu/runtime:collective_thunk_proto_cc",
        "@xla//xla/backends/gpu/runtime:copy_thunk_proto_cc",
        "@xla//xla/backends/gpu/runtime:p2p_thunk_common",
        "@xla//xla/backends/gpu/runtime:thunk",
        "@xla//xla/backends/gpu/runtime:thunk_proto_cc",
        "@xla//xla/core/collectives:reduction_kind_proto_cc",
        "@xla//xla/ffi:attribute_map",
        "@xla//xla/ffi:execution_state_proto_cc",
        "@xla//xla/mlir/utils:type_util",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
        "@xla//xla/python/pjrt_ifrt:executable_metadata_proto_cc",
        "@xla//xla/service:buffer_assignment_proto_cc",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/service:shaped_slice_proto_cc",
        "@xla//xla/service/gpu:dense_data_intermediate",
        "@xla//xla/service/gpu:dense_data_intermediate_proto_cc",
        "@xla//xla/service/gpu:gpu_executable_proto_cc",
        "@xla//xla/service/gpu:launch_dimensions_proto_cc",
        "@xla//xla/stream_executor:kernel_spec_proto_cc",
        "@xla//xla/stream_executor:launch_dim",
        "@xla//xla/stream_executor:launch_dim_proto_cc",
        "@xla//xla/util/split_proto:split_proto_reader",
    ],
)

cc_library(
    name = "proto_to_thunks",
    srcs = ["proto_to_thunks.cc"],
    hdrs = ["proto_to_thunks.h"],
    deps = [
        ":thunk_proto_util",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@xla//xla/backends/gpu/runtime:thunk",
        "@xla//xla/backends/gpu/runtime:thunk_proto_cc",
        "@xla//xla/backends/gpu/runtime:thunk_proto_deserialization",
        "@xla//xla/service:buffer_assignment",
        "@xla//xla/service:buffer_assignment_proto_cc",
        "@xla//xla/service/gpu:gpu_executable_proto_cc",
        "@xla//xla/stream_executor:device_description",
    ],
)

cc_library(
    name = "thunky_support",
    srcs = ["custom_call.cc"],
    deps = [
        ":proto_to_thunks",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@xla//xla:shape_util",
        "@xla//xla/backends/gpu/libraries/native_custom_call_thunks:native_custom_call_emitter_context",
        "@xla//xla/backends/gpu/libraries/native_custom_call_thunks:native_custom_call_handler_registration",
        "@xla//xla/backends/gpu/runtime:thunk",
        "@xla//xla/ffi",
        "@xla//xla/ffi:attributes",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service:buffer_assignment",
        "@xla//xla/service/gpu:gpu_executable_proto_cc",
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
        ":mlir_to_proto",
        ":proto_to_thunks",
        "//jaxlib/thunky/dialect:thunky_dialect",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
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
        "@xla//xla/service/gpu:gpu_executable_proto_cc",
        "@xla//xla/stream_executor:device_description",
    ],
)
