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
    "jax_extend_internal_users",
    "jax_extra_deps",
    "jax_internal_packages",
    "py_deps",
    "py_library_providing_imports_info",
    "pytype_library",
    "pytype_strict_library",
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

py_library_providing_imports_info(
    name = "jax",
    srcs = glob(
        [
            "*.py",
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
        # TODO(dsuo): Consider moving these files out of experimental if they're in the public API.
    ) + ["//jax/experimental:jax_public"],
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
        "//jax/_src:hijax",
        "//jax/_src:image",
        "//jax/_src:init",
        "//jax/_src:internal_mesh_utils",
        "//jax/_src:jaxpr_util",
        "//jax/_src:lax",
        "//jax/_src:layout",
        "//jax/_src:lazy_loader",
        "//jax/_src:memory",
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
        "//jax/_src:pmap",
        "//jax/_src:pretty_printer",
        "//jax/_src:profiler",
        "//jax/_src:public_test_util",
        "//jax/_src:random",
        "//jax/_src:ref",
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
        "//jax/_src:xla_bridge",
        "//jax/_src:xla_metadata",
        "//jax/_src:xla_metadata_lib",
        "//jax/_src/lib",
    ] + py_deps("numpy") + py_deps("opt_einsum") + py_deps("flatbuffers") + jax_extra_deps,
)

pytype_strict_library(
    name = "version",
    srcs = ["version.py"],
)

# Public JAX libraries below this point.

# Aliases of experimental targets.
# TODO(dsuo): remove these aliases/targets.
pytype_strict_library(
    name = "experimental",
    visibility = ["//visibility:public"],
    deps = [
        ":jax",
        "//jax/example_libraries:optimizers",
        "//jax/example_libraries:stax",
        "//jax/experimental",
        "//jax/experimental:checkify",
        "//jax/experimental:compute_on",
        "//jax/experimental:custom_dce",
        "//jax/experimental:custom_partitioning",
        "//jax/experimental:fused",
        "//jax/experimental:hijax",
        "//jax/experimental:jet",
        "//jax/experimental:layout",
        "//jax/experimental:mesh_utils",
        "//jax/experimental:multihost_utils",
        "//jax/experimental:ode",
        "//jax/experimental:pjit",
        "//jax/experimental:profiler",
        "//jax/experimental:rnn",
        "//jax/experimental:scheduling_groups",
        "//jax/experimental:shard_alike",
        "//jax/experimental:shard_map",
        "//jax/experimental:topologies",
        "//jax/experimental:transfer",
        "//jax/experimental:xla_metadata",
    ],
)

alias(
    name = "experimental_buffer_callback",
    actual = "//jax/experimental:buffer_callback",
    visibility = ["//jax/experimental:buffer_callback_users"],
)

alias(
    name = "experimental_colocated_python",
    actual = "//jax/experimental:colocated_python",
    visibility = ["//visibility:public"],
)

alias(
    name = "experimental_compute_on",
    actual = "//jax/experimental:compute_on",
    visibility = ["//visibility:public"],
)

alias(
    name = "compilation_cache",
    actual = "//jax/experimental:compilation_cache",
    visibility = ["//visibility:public"],
)

alias(
    name = "jet",
    actual = "//jax/experimental:jet",
    visibility = ["//visibility:public"],
)

alias(
    name = "mesh_utils",
    actual = "//jax/experimental:mesh_utils",
    visibility = ["//visibility:public"],
)

alias(
    name = "experimental_mesh_utils",
    actual = "//jax/experimental:mesh_utils",
    visibility = ["//visibility:public"],
)

alias(
    name = "mosaic",
    actual = "//jax/experimental:mosaic",
    visibility = ["//jax/experimental:mosaic_users"],
)

alias(
    name = "mosaic_gpu",
    actual = "//jax/experimental:mosaic_gpu",
    visibility = ["//jax/experimental:mosaic_gpu_users"],
)

alias(
    name = "experimental_multihost_utils",
    actual = "//jax/experimental:multihost_utils",
    visibility = ["//visibility:public"],
)

alias(
    name = "ode",
    actual = "//jax/experimental:ode",
    visibility = ["//visibility:public"],
)

alias(
    name = "pallas",
    actual = "//jax/experimental:pallas",
    visibility = ["//visibility:public"],
)

alias(
    name = "pallas_fuser",
    actual = "//jax/experimental:pallas_fuser",
    visibility = ["//jax/experimental:pallas_fuser_users"],
)

alias(
    name = "pallas_gpu",
    actual = "//jax/experimental:pallas_gpu",
    visibility = ["//jax/experimental:pallas_gpu_users"],
)

alias(
    name = "pallas_gpu_ops",
    actual = "//jax/experimental:pallas_gpu_ops",
    visibility = ["//jax/experimental:pallas_gpu_users"],
)

alias(
    name = "pallas_mosaic_gpu",
    actual = "//jax/experimental:pallas_mosaic_gpu",
    visibility = ["//jax/experimental:mosaic_gpu_users"],
)

alias(
    name = "pallas_tpu",
    actual = "//jax/experimental:pallas_tpu",
    visibility = ["//visibility:public"],
)

alias(
    name = "pallas_tpu_ops",
    actual = "//jax/experimental:pallas_tpu_ops",
    visibility = ["//visibility:public"],
)

alias(
    name = "pallas_triton",
    actual = "//jax/experimental:pallas_triton",
    visibility = ["//jax/experimental:pallas_gpu_users"],
)

alias(
    name = "pallas_experimental_gpu_ops",
    actual = "//jax/experimental:pallas_experimental_gpu_ops",
    visibility = ["//jax/experimental:mosaic_gpu_users"],
)

alias(
    name = "experimental_profiler",
    actual = "//jax/experimental:profiler",
    visibility = ["//visibility:public"],
)

alias(
    name = "experimental_pjit",
    actual = "//jax/experimental:pjit",
    visibility = ["//visibility:public"],
)

alias(
    name = "rnn",
    actual = "//jax/experimental:rnn",
    visibility = ["//visibility:public"],
)

alias(
    name = "experimental_serialize_executable",
    actual = "//jax/experimental:serialize_executable",
    visibility = ["//jax/experimental:serialize_executable_users"],
)

alias(
    name = "source_mapper",
    actual = "//jax/experimental:source_mapper",
    visibility = ["//visibility:public"],
)

alias(
    name = "experimental_sparse",
    actual = "//jax/experimental:sparse",
    visibility = ["//visibility:public"],
)

alias(
    name = "sparse_test_util",
    actual = "//jax/experimental:sparse_test_util",
    visibility = [":internal"],
)

alias(
    name = "experimental_topologies",
    actual = "//jax/experimental:topologies",
    visibility = ["//visibility:public"],
)

alias(
    name = "experimental_transfer",
    actual = "//jax/experimental:transfer",
    visibility = [":internal"],
)

# Aliases of example_library targets.
# TODO(dsuo): remove these aliases.
alias(
    name = "optimizers",
    actual = "//jax/example_libraries:optimizers",
    visibility = ["//visibility:public"],
)

alias(
    name = "stax",
    actual = "//jax/example_libraries:stax",
    visibility = ["//visibility:public"],
)
