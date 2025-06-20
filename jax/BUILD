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

load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load(
    "//jaxlib:jax.bzl",
    "buffer_callback_internal_users",
    "jax_extend_internal_users",
    "jax_extra_deps",
    "jax_internal_packages",
    "jax_visibility",
    "mosaic_gpu_internal_users",
    "mosaic_internal_users",
    "pallas_fuser_users",
    "pallas_gpu_internal_users",
    "pallas_sc_internal_users",
    "pallas_tpu_internal_users",
    "py_deps",
    "py_library_providing_imports_info",
    "pytype_library",
    "pytype_strict_library",
    "serialize_executable_internal_users",
)

package(
    default_applicable_licenses = [],
    default_visibility = [":internal"],
)

licenses(["notice"])

# The flag controls whether jaxlib should be built by Bazel.
# If ":build_jaxlib=true", then jaxlib will be built.
# If ":build_jaxlib=false", then jaxlib is not built. It is assumed that the pre-built jaxlib wheel
# is available in the "dist" folder.
# If ":build_jaxlib=wheel", then jaxlib wheel will be built as a py_import rule attribute.
# The py_import rule unpacks the wheel and provides its content as a py_library.
string_flag(
    name = "build_jaxlib",
    build_setting_default = "true",
    values = [
        "true",
        "false",
        "wheel",
    ],
)

config_setting(
    name = "config_build_jaxlib_true",
    flag_values = {
        ":build_jaxlib": "true",
    },
)

config_setting(
    name = "config_build_jaxlib_false",
    flag_values = {
        ":build_jaxlib": "false",
    },
)

config_setting(
    name = "config_build_jaxlib_wheel",
    flag_values = {
        ":build_jaxlib": "wheel",
    },
)

# The flag controls whether jax should be built by Bazel.
# If ":build_jax=true", then jax will be built.
# If ":build_jax=false", then jax is not built. It is assumed that the pre-built jax wheel
# is available in the "dist" folder.
# If ":build_jax=wheel", then jax wheel will be built as a py_import rule attribute.
# The py_import rule unpacks the wheel and provides its content as a py_library.
string_flag(
    name = "build_jax",
    build_setting_default = "true",
    values = [
        "true",
        "false",
        "wheel",
    ],
)

config_setting(
    name = "config_build_jax_true",
    flag_values = {
        ":build_jax": "true",
    },
)

config_setting(
    name = "config_build_jax_false",
    flag_values = {
        ":build_jax": "false",
    },
)

config_setting(
    name = "config_build_jax_wheel",
    flag_values = {
        ":build_jax": "wheel",
    },
)

exports_files([
    "LICENSE",
    "version.py",
    "py.typed",
])

# Packages that have access to JAX-internal implementation details.
package_group(
    name = "internal",
    packages = [
        "//...",
    ] + jax_internal_packages,
)

package_group(
    name = "jax_extend_users",
    includes = [":internal"],
    packages = [
        # Intentionally avoid jax dependencies on jax.extend.
        # See https://docs.jax.dev/en/latest/jep/15856-jex.html
        "//tests/...",
    ] + jax_extend_internal_users,
)

package_group(
    name = "mosaic_users",
    includes = [":internal"],
    packages = mosaic_internal_users,
)

package_group(
    name = "pallas_gpu_users",
    includes = [":internal"],
    packages = pallas_gpu_internal_users,
)

package_group(
    name = "pallas_tpu_users",
    includes = [":internal"],
    packages = pallas_tpu_internal_users,
)

package_group(
    name = "pallas_sc_users",
    includes = [":internal"],
    packages = pallas_sc_internal_users,
)

package_group(
    name = "pallas_fuser_users",
    includes = [":internal"],
    packages = pallas_fuser_users,
)

package_group(
    name = "mosaic_gpu_users",
    includes = [":internal"],
    packages = mosaic_gpu_internal_users,
)

package_group(
    name = "serialize_executable_users",
    includes = [":internal"],
    packages = serialize_executable_internal_users,
)

package_group(
    name = "buffer_callback_users",
    includes = [":internal"],
    packages = buffer_callback_internal_users,
)

py_library_providing_imports_info(
    name = "jax",
    srcs = glob(
        [
            "*.py",
            "experimental/key_reuse/**/*.py",
            "experimental/roofline/**/*.py",
            "image/**/*.py",
            "interpreters/**/*.py",
            "lax/**/*.py",
            "lib/**/*.py",
            "nn/**/*.py",
            "numpy/**/*.py",
            "ops/**/*.py",
            "scipy/**/*.py",
            "third_party/**/*.py",
        ],
        exclude = [
            "*_test.py",
            "**/*_test.py",
        ],
    ) + [
        "experimental/pjit.py",
        "experimental/multihost_utils.py",
        "experimental/shard_map.py",
        # until checkify is moved out of experimental
        "experimental/checkify.py",
        "experimental/compilation_cache/compilation_cache.py",
    ],
    lib_rule = pytype_library,
    pytype_srcs = glob(
        [
            "nn/*.pyi",
            "numpy/*.pyi",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":version",
        "//jax/_src:abstract_arrays",
        "//jax/_src:ad",
        "//jax/_src:ad_util",
        "//jax/_src:api",
        "//jax/_src:api_util",
        "//jax/_src:basearray",
        "//jax/_src:batching",
        "//jax/_src:blocked_sampler",
        "//jax/_src:buffer_callback",
        "//jax/_src:callback",
        "//jax/_src:checkify",
        "//jax/_src:cloud_tpu_init",
        "//jax/_src:compilation_cache_internal",
        "//jax/_src:compiler",
        "//jax/_src:compute_on",
        "//jax/_src:config",
        "//jax/_src:core",
        "//jax/_src:cudnn",
        "//jax/_src:custom_api_util",
        "//jax/_src:custom_batching",
        "//jax/_src:custom_dce",
        "//jax/_src:custom_derivatives",
        "//jax/_src:custom_partitioning",
        "//jax/_src:custom_partitioning_sharding_rule",
        "//jax/_src:custom_transpose",
        "//jax/_src:debugger",
        "//jax/_src:debugging",
        "//jax/_src:deprecations",
        "//jax/_src:dlpack",
        "//jax/_src:dtypes",
        "//jax/_src:earray",
        "//jax/_src:effects",
        "//jax/_src:environment_info",
        "//jax/_src:error_check",
        "//jax/_src:export",
        "//jax/_src:ffi",
        "//jax/_src:flatten_util",
        "//jax/_src:hashable_array",
        "//jax/_src:image",
        "//jax/_src:init",
        "//jax/_src:internal_mesh_utils",
        "//jax/_src:jaxpr_util",
        "//jax/_src:lax",
        "//jax/_src:layout",
        "//jax/_src:lazy_loader",
        "//jax/_src:mesh",
        "//jax/_src:mlir",
        "//jax/_src:monitoring",
        "//jax/_src:named_sharding",
        "//jax/_src:nn",
        "//jax/_src:numpy",
        "//jax/_src:op_shardings",
        "//jax/_src:partial_eval",
        "//jax/_src:partition_spec",
        "//jax/_src:path",
        "//jax/_src:pickle_util",
        "//jax/_src:pretty_printer",
        "//jax/_src:profiler",
        "//jax/_src:public_test_util",
        "//jax/_src:random",
        "//jax/_src:scipy",
        "//jax/_src:shard_alike",
        "//jax/_src:shard_map",
        "//jax/_src:sharding",
        "//jax/_src:sharding_impls",
        "//jax/_src:sharding_specs",
        "//jax/_src:source_info_util",
        "//jax/_src:sourcemap",
        "//jax/_src:stages",
        "//jax/_src:tpu",
        "//jax/_src:traceback_util",
        "//jax/_src:tree",
        "//jax/_src:tree_util",
        "//jax/_src:typing",
        "//jax/_src:util",
        "//jax/_src:xla",
        "//jax/_src:xla_bridge",
        "//jax/_src:xla_metadata",
        "//jax/_src:xla_metadata_lib",
        "//jax/_src/lib",
    ] + py_deps("numpy") + py_deps("scipy") + py_deps("opt_einsum") + py_deps("flatbuffers") + jax_extra_deps,
)

pytype_strict_library(
    name = "experimental_serialize_executable",
    srcs = ["experimental/serialize_executable.py"],
    visibility = [":serialize_executable_users"],
    deps = [
        ":jax",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "source_mapper",
    srcs = glob(include = ["experimental/source_mapper/**/*.py"]),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":jax",
        "//jax/_src:config",
        "//jax/_src:core",
        "//jax/_src:source_info_util",
        "//jax/_src:sourcemap",
    ] + py_deps("absl/flags"),
)

pytype_strict_library(
    name = "pallas",
    srcs = glob(
        [
            "experimental/pallas/**/*.py",
        ],
        exclude = [
            "experimental/pallas/mosaic_gpu.py",
            "experimental/pallas/ops/gpu/**/*.py",
            "experimental/pallas/ops/tpu/**/*.py",
            "experimental/pallas/tpu.py",
            "experimental/pallas/fuser.py",
            "experimental/pallas/triton.py",
        ],
    ),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":jax",
        "//jax/_src:deprecations",
        "//jax/_src:lax",
        "//jax/_src:source_info_util",
        "//jax/_src:state_types",
        "//jax/_src/pallas",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "pallas_tpu",
    srcs = ["experimental/pallas/tpu.py"],
    visibility = [
        ":pallas_tpu_users",
    ],
    deps = [
        ":pallas",  # build_cleaner: keep
        "//jax/_src:deprecations",
        "//jax/_src/pallas",
        "//jax/_src/pallas/mosaic:core",
        "//jax/_src/pallas/mosaic:helpers",
        "//jax/_src/pallas/mosaic:interpret",
        "//jax/_src/pallas/mosaic:lowering",
        "//jax/_src/pallas/mosaic:pallas_call_registration",  # build_cleaner: keep
        "//jax/_src/pallas/mosaic:pipeline",
        "//jax/_src/pallas/mosaic:primitives",
        "//jax/_src/pallas/mosaic:random",
        "//jax/_src/pallas/mosaic:verification",
    ],
)

pytype_strict_library(
    name = "pallas_fuser",
    srcs = ["experimental/pallas/fuser.py"],
    visibility = [
        ":pallas_fuser_users",
    ],
    deps = [
        ":pallas",  # build_cleaner: keep
        "//jax/_src/pallas/fuser:block_spec",
        "//jax/_src/pallas/fuser:custom_evaluate",
        "//jax/_src/pallas/fuser:fusible",
        "//jax/_src/pallas/fuser:fusion",
        "//jax/_src/pallas/fuser:jaxpr_fusion",
    ],
)

pytype_strict_library(
    name = "pallas_gpu_ops",
    srcs = ["//jax/experimental/pallas/ops/gpu:triton_ops"],
    visibility = [
        ":pallas_gpu_users",
    ],
    deps = [
        ":jax",
        ":pallas",
        ":pallas_gpu",
        "//jax/_src:lax",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "pallas_experimental_gpu_ops",
    srcs = ["//jax/experimental/pallas/ops/gpu:mgpu_ops"],
    visibility = [
        ":mosaic_gpu_users",
    ],
    deps = [
        ":jax",
        ":mosaic_gpu",
        ":pallas",
        ":pallas_mosaic_gpu",
        "//jax/_src:test_util",  # This is only to make them runnable as jax_multiplatform_test...
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "pallas_tpu_ops",
    srcs = glob(["experimental/pallas/ops/tpu/**/*.py"]),
    visibility = [
        ":pallas_tpu_users",
    ],
    deps = [
        ":jax",
        ":pallas",
        ":pallas_tpu",
        "//jax/_src:dtypes",
        "//jax/_src:random",
        "//jax/_src:shard_map",
        "//jax/_src:util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "pallas_gpu",
    visibility = [
        ":pallas_gpu_users",
    ],
    deps = [
        ":pallas_triton",
        # TODO(slebedev): Add :pallas_mosaic_gpu once it is ready.
    ],
)

pytype_strict_library(
    name = "pallas_triton",
    srcs = [
        "experimental/pallas/triton.py",
    ],
    visibility = [
        ":pallas_gpu_users",
    ],
    deps = [
        "//jax/_src:deprecations",
        "//jax/_src/pallas",
        "//jax/_src/pallas/triton:core",
        "//jax/_src/pallas/triton:pallas_call_registration",  # build_cleaner: keep
        "//jax/_src/pallas/triton:primitives",
    ],
)

pytype_strict_library(
    name = "pallas_mosaic_gpu",
    srcs = ["experimental/pallas/mosaic_gpu.py"],
    visibility = [
        ":mosaic_gpu_users",
    ],
    deps = [
        ":mosaic_gpu",
        "//jax/_src/pallas/mosaic_gpu:core",
        "//jax/_src/pallas/mosaic_gpu:helpers",
        "//jax/_src/pallas/mosaic_gpu:pallas_call_registration",  # build_cleaner: keep
        "//jax/_src/pallas/mosaic_gpu:pipeline",
        "//jax/_src/pallas/mosaic_gpu:primitives",
    ],
)

# This target only supports sm_90 GPUs.
py_library_providing_imports_info(
    name = "mosaic_gpu",
    srcs = glob(["experimental/mosaic/gpu/*.py"]),
    data = [
        "@cuda_nvcc//:nvdisasm",
        "@cuda_nvcc//:nvvm",
        "@cuda_nvcc//:ptxas",
        "@nvidia_nvshmem//:libnvshmem_device",
    ],
    lib_rule = pytype_strict_library,
    visibility = [
        ":mosaic_gpu_users",
    ],
    deps = [
        ":jax",
        "//jax/_src:config",
        "//jax/_src:core",
        "//jax/_src:mlir",
        "//jax/_src:sharding_impls",
        "//jax/_src:stages",
        "//jax/_src:util",
        "//jax/_src/lib",
        "//jaxlib/mlir:arithmetic_dialect",
        "//jaxlib/mlir:builtin_dialect",
        "//jaxlib/mlir:control_flow_dialect",
        "//jaxlib/mlir:func_dialect",
        "//jaxlib/mlir:gpu_dialect",
        "//jaxlib/mlir:ir",
        "//jaxlib/mlir:llvm_dialect",
        "//jaxlib/mlir:math_dialect",
        "//jaxlib/mlir:memref_dialect",
        "//jaxlib/mlir:nvgpu_dialect",
        "//jaxlib/mlir:nvvm_dialect",
        "//jaxlib/mlir:pass_manager",
        "//jaxlib/mlir:scf_dialect",
        "//jaxlib/mlir:vector_dialect",
        "//jaxlib/mosaic/python:gpu_dialect",
    ] + py_deps("absl/flags") + py_deps("numpy"),
)

pytype_strict_library(
    name = "experimental_compute_on",
    srcs = ["experimental/compute_on.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":compute_on",
    ],
)

pytype_strict_library(
    name = "experimental_mesh_utils",
    srcs = ["experimental/mesh_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":internal_mesh_utils",
    ],
)

pytype_strict_library(
    name = "experimental_multihost_utils",
    srcs = ["experimental/multihost_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":api",
        ":jax",
        ":sharding_impls",
        "//jax/_src:ad",
        "//jax/_src:batching",
        "//jax/_src:core",
        "//jax/_src:mlir",
        "//jax/_src:util",
        "//jax/_src:xla_bridge",
        "//jax/_src/lib",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "experimental_pjit",
    srcs = ["experimental/pjit.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":api",
        ":sharding_impls",
    ],
)

pytype_strict_library(
    name = "experimental_profiler",
    srcs = ["experimental/profiler.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "experimental_topologies",
    srcs = ["experimental/topologies.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":experimental_mesh_utils",
        ":jax",
        "//jax/_src:xla_bridge",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "experimental_transfer",
    srcs = ["experimental/transfer.py"],
    deps = [
        ":jax",
        "//jax/_src:util",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "version",
    srcs = ["version.py"],
)

# Public JAX libraries below this point.

py_library_providing_imports_info(
    name = "experimental",
    srcs = glob(
        [
            "experimental/*.py",
            "example_libraries/*.py",
        ],
        exclude = [
            "experimental/buffer_callback.py",
            "experimental/mosaic/gpu/*.py",
            "experimental/serialize_executable.py",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":jax",
        "//jax/_src:buffer_callback",
    ] + py_deps("absl/logging") + py_deps("numpy"),
)

pytype_strict_library(
    name = "stax",
    srcs = [
        "example_libraries/stax.py",
    ],
    visibility = ["//visibility:public"],
    deps = [":jax"],
)

pytype_strict_library(
    name = "experimental_sparse",
    srcs = glob(
        [
            "experimental/sparse/*.py",
        ],
        exclude = ["experimental/sparse/test_util.py"],
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":jax",
        "//jax/_src:ad",
        "//jax/_src:api",
        "//jax/_src:api_util",
        "//jax/_src:batching",
        "//jax/_src:config",
        "//jax/_src:core",
        "//jax/_src:custom_derivatives",
        "//jax/_src:dtypes",
        "//jax/_src:ffi",
        "//jax/_src:lax",
        "//jax/_src:mlir",
        "//jax/_src:numpy",
        "//jax/_src:partial_eval",
        "//jax/_src:sharding_impls",
        "//jax/_src:traceback_util",
        "//jax/_src:typing",
        "//jax/_src:util",
        "//jax/_src/lib",
    ] + py_deps("numpy") + py_deps("scipy"),
)

pytype_strict_library(
    name = "sparse_test_util",
    srcs = [
        "experimental/sparse/test_util.py",
    ],
    visibility = [":internal"],
    deps = [
        ":experimental_sparse",
        ":jax",
        "//jax/_src:lax",
        "//jax/_src:test_util",
        "//jax/_src:typing",
        "//jax/_src:util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "optimizers",
    srcs = [
        "example_libraries/optimizers.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":jax",
        "//jax/_src:util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "ode",
    srcs = ["experimental/ode.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":jax",
        "//jax/_src:core",
        "//jax/_src:numpy",
        "//jax/_src:util",
    ],
)

pytype_strict_library(
    name = "jet",
    srcs = ["experimental/jet.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":jax",
        "//jax/_src:ad_util",
        "//jax/_src:api",
        "//jax/_src:core",
        "//jax/_src:lax",
        "//jax/_src:partial_eval",
        "//jax/_src:sharding_impls",
        "//jax/_src:util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "experimental_host_callback",
    srcs = [
        "experimental/__init__.py",  # To support JAX_HOST_CALLBACK_LEGACY=False
        "experimental/host_callback.py",
        "experimental/x64_context.py",  # To support JAX_HOST_CALLBACK_LEGACY=False
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//jax/_src:api",
        "//jax/_src:callback",
        "//jax/_src:config",
        "//jax/_src:core",
        "//jax/_src:dtypes",
        "//jax/_src:earray",
    ],
)

pytype_strict_library(
    name = "compilation_cache",
    srcs = [
        "experimental/compilation_cache/__init__.py",
        "experimental/compilation_cache/compilation_cache.py",
    ],
    visibility = ["//visibility:public"],
    deps = ["//jax/_src:compilation_cache_internal"],
)

pytype_strict_library(
    name = "mesh_utils",
    srcs = ["experimental/mesh_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//jax/_src:internal_mesh_utils",
    ],
)

# TODO(phawkins): remove this target in favor of the finer-grained targets in jax/extend/...
pytype_strict_library(
    name = "extend",
    visibility = [":jax_extend_users"],
    deps = [
        "//jax/extend",
        "//jax/extend:backend",
        "//jax/extend:core",
        "//jax/extend:linear_util",
        "//jax/extend:random",
        "//jax/extend:source_info_util",
    ],
)

pytype_strict_library(
    name = "mosaic",
    srcs = [
        "experimental/mosaic/__init__.py",
        "experimental/mosaic/dialects.py",
    ],
    visibility = [":mosaic_users"],
    deps = [
        "//jax/_src:tpu_custom_call",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "rnn",
    srcs = ["experimental/rnn.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":jax",
        "//jax/_src:core",
        "//jax/_src:custom_derivatives",
        "//jax/_src:lax",
        "//jax/_src:typing",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "experimental_colocated_python",
    srcs = [
        "experimental/colocated_python/__init__.py",
        "experimental/colocated_python/api.py",
        "experimental/colocated_python/func.py",
        "experimental/colocated_python/func_backend.py",
        "experimental/colocated_python/obj.py",
        "experimental/colocated_python/obj_backend.py",
        "experimental/colocated_python/serialization.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":jax",
        "//jax/_src:api",
        "//jax/_src:api_util",
        "//jax/_src:traceback_util",
        "//jax/_src:tree_util",
        "//jax/_src:util",
        "//jax/_src:xla_bridge",
        "//jax/_src/lib",
        "//jax/extend:ifrt_programs",
    ] + py_deps("numpy") + py_deps("cloudpickle"),
)

pytype_strict_library(
    name = "experimental_buffer_callback",
    srcs = [
        "experimental/buffer_callback.py",
    ],
    visibility = [":buffer_callback_users"],
    deps = [
        "//jax/_src:buffer_callback",
    ],
)

# Aliases of _src targets.
# TODO(vanderplas): remove these aliases.

alias(
    name = "test_util",
    actual = "//jax/_src:test_util",
)

alias(
    name = "internal_test_util",
    actual = "//jax/_src:internal_test_util",
)

alias(
    name = "internal_test_harnesses",
    actual = "//jax/_src:internal_test_harnesses",
)

alias(
    name = "test_multiprocess",
    actual = "//jax/_src:test_multiprocess",
)

alias(
    name = "internal_export_back_compat_test_util",
    actual = "//jax/_src:internal_export_back_compat_test_util",
)

alias(
    name = "internal_export_back_compat_test_data",
    actual = "//jax/_src:internal_export_back_compat_test_data",
)

alias(
    name = "abstract_arrays",
    actual = "//jax/_src:abstract_arrays",
)

alias(
    name = "ad_util",
    actual = "//jax/_src:ad_util",
)

alias(
    name = "api",
    actual = "//jax/_src:api",
    visibility = [":internal"] + jax_visibility("api"),
)

alias(
    name = "api_util",
    actual = "//jax/_src:api_util",
)

alias(
    name = "basearray",
    actual = "//jax/_src:basearray",
)

alias(
    name = "blocked_sampler",
    actual = "//jax/_src:blocked_sampler",
)

alias(
    name = "buffer_callback",
    actual = "//jax/_src:buffer_callback",
)

alias(
    name = "callback",
    actual = "//jax/_src:callback",
)

alias(
    name = "checkify",
    actual = "//jax/_src:checkify",
    visibility = [":internal"] + jax_visibility("checkify"),
)

alias(
    name = "cloud_tpu_init",
    actual = "//jax/_src:cloud_tpu_init",
)

alias(
    name = "compilation_cache_internal",
    actual = "//jax/_src:compilation_cache_internal",
    visibility = [":internal"] + jax_visibility("compilation_cache"),
)

alias(
    name = "cache_key",
    actual = "//jax/_src:cache_key",
    visibility = [":internal"] + jax_visibility("compilation_cache"),
)

alias(
    name = "compilation_cache_interface",
    actual = "//jax/_src:compilation_cache_interface",
)

alias(
    name = "lax",
    actual = "//jax/_src:lax",
    visibility = [":internal"] + jax_visibility("lax"),
)

alias(
    name = "lru_cache",
    actual = "//jax/_src:lru_cache",
)

alias(
    name = "config",
    actual = "//jax/_src:config",
)

alias(
    name = "logging_config",
    actual = "//jax/_src:logging_config",
)

alias(
    name = "compiler",
    actual = "//jax/_src:compiler",
    visibility = [":internal"] + jax_visibility("compiler"),
)

alias(
    name = "core",
    actual = "//jax/_src:core",
)

alias(
    name = "cudnn",
    actual = "//jax/_src:cudnn",
)

alias(
    name = "custom_api_util",
    actual = "//jax/_src:custom_api_util",
)

alias(
    name = "custom_batching",
    actual = "//jax/_src:custom_batching",
)

alias(
    name = "custom_dce",
    actual = "//jax/_src:custom_dce",
)

alias(
    name = "custom_derivatives",
    actual = "//jax/_src:custom_derivatives",
)

alias(
    name = "custom_partitioning",
    actual = "//jax/_src:custom_partitioning",
)

alias(
    name = "custom_partitioning_sharding_rule",
    actual = "//jax/_src:custom_partitioning_sharding_rule",
)

alias(
    name = "custom_transpose",
    actual = "//jax/_src:custom_transpose",
)

alias(
    name = "debugger",
    actual = "//jax/_src:debugger",
)

alias(
    name = "debugging",
    actual = "//jax/_src:debugging",
)

alias(
    name = "deprecations",
    actual = "//jax/_src:deprecations",
)

alias(
    name = "dlpack",
    actual = "//jax/_src:dlpack",
)

alias(
    name = "dtypes",
    actual = "//jax/_src:dtypes",
)

alias(
    name = "earray",
    actual = "//jax/_src:earray",
)

alias(
    name = "effects",
    actual = "//jax/_src:effects",
)

alias(
    name = "environment_info",
    actual = "//jax/_src:environment_info",
)

alias(
    name = "error_check",
    actual = "//jax/_src:error_check",
)

alias(
    name = "export",
    actual = "//jax/_src:export",
    visibility = [":internal"] + jax_visibility("export"),
)

alias(
    name = "extend_src",
    actual = "//jax/_src:extend_src",
)

alias(
    name = "ffi",
    actual = "//jax/_src:ffi",
)

alias(
    name = "flatten_util",
    actual = "//jax/_src:flatten_util",
)

alias(
    name = "frozen_dict",
    actual = "//jax/_src:frozen_dict",
)

alias(
    name = "hardware_utils",
    actual = "//jax/_src:hardware_utils",
)

alias(
    name = "hashable_array",
    actual = "//jax/_src:hashable_array",
)

alias(
    name = "image",
    actual = "//jax/_src:image",
    visibility = [":internal"] + jax_visibility("image"),
)

alias(
    name = "lax_reference",
    actual = "//jax/_src:lax_reference",
    visibility = [":internal"] + jax_visibility("lax_reference"),
)

alias(
    name = "lazy_loader",
    actual = "//jax/_src:lazy_loader",
)

alias(
    name = "jaxpr_util",
    actual = "//jax/_src:jaxpr_util",
)

alias(
    name = "mesh",
    actual = "//jax/_src:mesh",
)

alias(
    name = "ad",
    actual = "//jax/_src:ad",
)

alias(
    name = "batching",
    actual = "//jax/_src:batching",
)

alias(
    name = "mlir",
    actual = "//jax/_src:mlir",
)

alias(
    name = "monitoring",
    actual = "//jax/_src:monitoring",
)

alias(
    name = "op_shardings",
    actual = "//jax/_src:op_shardings",
)

alias(
    name = "scipy",
    actual = "//jax/_src:scipy",
)

alias(
    name = "sourcemap",
    actual = "//jax/_src:sourcemap",
)

alias(
    name = "partial_eval",
    actual = "//jax/_src:partial_eval",
)

alias(
    name = "partition_spec",
    actual = "//jax/_src:partition_spec",
)

alias(
    name = "path",
    actual = "//jax/_src:path",
)

alias(
    name = "pickle_util",
    actual = "//jax/_src:pickle_util",
)

alias(
    name = "pretty_printer",
    actual = "//jax/_src:pretty_printer",
    visibility = [":internal"] + jax_visibility("pretty_printer"),
)

alias(
    name = "profiler",
    actual = "//jax/_src:profiler",
)

alias(
    name = "public_test_util",
    actual = "//jax/_src:public_test_util",
)

alias(
    name = "random",
    actual = "//jax/_src:random",
    visibility = [":internal"] + jax_visibility("random"),
)

alias(
    name = "sharding",
    actual = "//jax/_src:sharding",
)

alias(
    name = "shard_alike",
    actual = "//jax/_src:shard_alike",
)

alias(
    name = "shard_map",
    actual = "//jax/_src:shard_map",
)

alias(
    name = "stages",
    actual = "//jax/_src:stages",
    visibility = [":internal"] + jax_visibility("stages"),
)

alias(
    name = "compute_on",
    actual = "//jax/_src:compute_on",
)

alias(
    name = "xla_metadata",
    actual = "//jax/_src:xla_metadata",
)

alias(
    name = "xla_metadata_lib",
    actual = "//jax/_src:xla_metadata_lib",
)

alias(
    name = "layout",
    actual = "//jax/_src:layout",
)

alias(
    name = "sharding_impls",
    actual = "//jax/_src:sharding_impls",
    visibility = [":internal"] + jax_visibility("sharding_impls"),
)

alias(
    name = "named_sharding",
    actual = "//jax/_src:named_sharding",
)

alias(
    name = "nn",
    actual = "//jax/_src:nn",
)

alias(
    name = "numpy",
    actual = "//jax/_src:numpy",
)

alias(
    name = "sharding_specs",
    actual = "//jax/_src:sharding_specs",
)

alias(
    name = "internal_mesh_utils",
    actual = "//jax/_src:internal_mesh_utils",
)

alias(
    name = "source_info_util",
    actual = "//jax/_src:source_info_util",
    visibility = [":internal"] + jax_visibility("source_info_util"),
)

alias(
    name = "state_types",
    actual = "//jax/_src:state_types",
    visibility = [":internal"] + jax_visibility("state_types"),
)

alias(
    name = "tpu",
    actual = "//jax/_src:tpu",
)

alias(
    name = "tree",
    actual = "//jax/_src:tree",
)

alias(
    name = "tree_util",
    actual = "//jax/_src:tree_util",
    visibility = [":internal"] + jax_visibility("tree_util"),
)

alias(
    name = "traceback_util",
    actual = "//jax/_src:traceback_util",
    visibility = [":internal"] + jax_visibility("traceback_util"),
)

alias(
    name = "typing",
    actual = "//jax/_src:typing",
)

alias(
    name = "tpu_custom_call",
    actual = "//jax/_src:tpu_custom_call",
)

alias(
    name = "util",
    actual = "//jax/_src:util",
)

alias(
    name = "xla",
    actual = "//jax/_src:xla",
)

alias(
    name = "xla_bridge",
    actual = "//jax/_src:xla_bridge",
    visibility = [":internal"] + jax_visibility("xla_bridge"),
)
