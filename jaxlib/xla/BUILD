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

load(
    "//jaxlib:jax.bzl",
    "if_oss",
    "nanobind_extension",
    "py_deps",
    "py_strict_library",
    "py_strict_test",
    "pytype_strict_library",
)

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

nanobind_extension(
    name = "xla_extension",
    srcs = ["xla.cc"],
    pytype_deps = py_deps(["numpy"]),
    pytype_srcs = glob(["xla_extension/*.pyi"]),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
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
        "@xla//xla/pjrt/distributed",
        "@xla//xla/pjrt/distributed:client",
        "@xla//xla/pjrt/distributed:key_value_store_interface",
        "@xla//xla/pjrt/distributed:protocol_proto_cc",
        "@xla//xla/pjrt/distributed:service",
        "@xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/python:config",
        "@xla//xla/python:custom_call_sharding",
        "@xla//xla/python:dlpack",
        "@xla//xla/python:guard_lib",
        "@xla//xla/python:jax_jit",
        "@xla//xla/python:logging",
        "@xla//xla/python:mlir",
        "@xla//xla/python:nb_absl_flat_hash_map",
        "@xla//xla/python:nb_absl_span",
        "@xla//xla/python:nb_class_ptr",
        "@xla//xla/python:ops",
        "@xla//xla/python:pjit",
        "@xla//xla/python:pmap_lib",
        "@xla//xla/python:pprof_profile_builder",
        "@xla//xla/python:profiler",
        "@xla//xla/python:py_client",
        "@xla//xla/python:python_ref_manager",
        "@xla//xla/python:pytree",
        "@xla//xla/python:refine_polymorphic_shapes",
        "@xla//xla/python:sdy",
        "@xla//xla/python:traceback",
        "@xla//xla/python:types",
        "@xla//xla/python:util",
        "@xla//xla/python:weakref_lru_cache",
        "@xla//xla/python:xla_compiler",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt:plugin_program",
        "@xla//xla/python/ifrt:plugin_program_serdes",
        "@xla//xla/python/ifrt_proxy/client:py_module",
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
            "@gloo//:transport_tcp",
            "@xla//xla/backends/cpu/collectives:gloo_collectives",
            "@xla//xla/backends/cpu/collectives:gloo_kv_store",
            "@xla//xla/python/transfer:py_socket_transfer",
        ],
    }) + select({
        # mpitrampoline does not build on windows
        "@xla//xla/tsl:windows": [],
        # we support MPI collectives only in OSS builds
        "//conditions:default": if_oss(["@xla//xla/backends/cpu/collectives:mpi_collectives"]),
    }),
)

pytype_strict_library(
    name = "xla_client",
    srcs = ["xla_client.py"],
    pytype_srcs = ["xla_client.pyi"],
    visibility = [":xla_python"],
    deps = py_deps([
        "numpy",
        "ml_dtypes",
    ]) + [":xla_extension"],
)

py_strict_test(
    name = "xla_client_backend_independent_test",
    srcs = ["xla_client_backend_independent_test.py"],
    deps = [
        ":xla_client",
    ] + py_deps([
        "absl/testing",
        "numpy",
        "portpicker",
    ]),
)

py_strict_library(
    name = "xla_client_test",
    testonly = 1,
    srcs = ["xla_client_test.py"],
    visibility = [":xla_python"],
    deps = [
        ":xla_client",
        "//jax",
        "//jax:test_util",
        "//jaxlib",
    ] + py_deps([
        "absl/flags",
        "absl/logging",
        "absl/testing",
        "ml_dtypes",
        "numpy",
    ]),
)

nanobind_extension(
    name = "custom_calls_testlib",
    testonly = 1,
    srcs = ["custom_calls_testlib.cc"],
    deps = [
        "@com_google_absl//absl/status",
        "@nanobind",
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/ffi/api:ffi",
    ],
)

py_strict_test(
    name = "xla_client_test_cpu",
    srcs = ["xla_client_test.py"],
    args = ["--backend=cpu"],
    env = {
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
    },
    main = "xla_client_test.py",
    deps = [
        ":custom_calls_testlib",
        ":xla_client",
        "//jax",
        "//jax:test_util",
        "//jaxlib",
    ] + py_deps([
        "absl/flags",
        "absl/logging",
        "absl/testing",
        "ml_dtypes",
        "numpy",
    ]),
)

py_strict_test(
    name = "weakref_lru_cache_test",
    srcs = ["weakref_lru_cache_test.py"],
    deps = [
        ":xla_client",
    ] + py_deps([
        "absl/flags",
        "absl/logging",
        "absl/testing",
    ]),
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
