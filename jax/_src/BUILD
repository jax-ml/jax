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
    "if_building_jaxlib",
    "jax_export_file_visibility",
    "jax_internal_export_back_compat_test_util_visibility",
    "jax_internal_test_harnesses_visibility",
    "jax_test_util_visibility",
    "jax_visibility",
    "py_deps",
    "py_library_providing_imports_info",
    "pytype_strict_library",
)

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:internal"],
)

exports_files(
    ["export/serialization.fbs"],
    visibility = jax_export_file_visibility,
)

pytype_strict_library(
    name = "init",
    srcs = [
        "__init__.py",
        "interpreters/__init__.py",
    ],
    deps = [":traceback_util"],
)

# JAX-private test utilities.
pytype_strict_library(
    # This build target is required in order to use private test utilities in jax._src.test_util,
    # and its visibility is intentionally restricted to discourage its use outside JAX itself.
    # JAX does provide some public test utilities (see jax/test_util.py);
    # these are available in jax.test_util via the standard :jax target.
    name = "test_util",
    srcs = [
        "test_loader.py",
        "test_util.py",
        "test_warning_util.py",
    ],
    visibility = [
        "//jax:internal",
    ] + jax_test_util_visibility,
    deps = [
        ":api",
        ":cloud_tpu_init",
        ":compilation_cache_internal",
        ":config",
        ":core",
        ":deprecations",
        ":dtypes",
        ":lax",
        ":mesh",
        ":mlir",
        ":monitoring",
        ":numpy",
        ":public_test_util",
        ":sharding_impls",
        ":tree_util",
        ":typing",
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

# TODO(necula): break the internal_test_util into smaller build targets.
pytype_strict_library(
    name = "internal_test_util",
    srcs = [
        "internal_test_util/__init__.py",
        "internal_test_util/deprecation_module.py",
        "internal_test_util/lax_test_util.py",
    ] + glob(
        [
            "internal_test_util/lazy_loader_module/*.py",
        ],
    ),
    visibility = ["//jax:internal"],
    deps = if_building_jaxlib(
        if_building = [
            ":api",
            ":config",
            ":core",
            ":dtypes",
            ":deprecations",
            ":lax",
            ":lazy_loader",
            ":random",
            ":test_util",
            ":tree_util",
            ":typing",
            ":util",
            ":xla_bridge",
        ],
        if_not_building = [],
    ) + py_deps("numpy"),
)

pytype_strict_library(
    name = "internal_test_harnesses",
    srcs = ["internal_test_util/test_harnesses.py"],
    visibility = ["//jax:internal"] + jax_internal_test_harnesses_visibility,
    deps = if_building_jaxlib(
        if_building = [
            ":ad_util",
            ":api",
            ":config",
            ":dtypes",
            ":lax",
            ":numpy",
            ":random",
            ":test_util",
            ":typing",
            ":xla_bridge",
            "//jax/_src/lib",
        ],
        if_not_building = [],
    ) + py_deps("numpy") + py_deps("absl/testing"),
)

pytype_strict_library(
    name = "test_multiprocess",
    srcs = ["test_multiprocess.py"],
    visibility = ["//jax:internal"],
    deps = if_building_jaxlib(
        if_building = [
            ":config",
            ":test_util",
            ":xla_bridge",
            "//jax/_src/lib",
        ],
        if_not_building = [],
    ) + py_deps("absl-all") + py_deps("portpicker"),
)

pytype_strict_library(
    name = "internal_export_back_compat_test_util",
    srcs = ["internal_test_util/export_back_compat_test_util.py"],
    visibility = [
        "//jax:internal",
    ] + jax_internal_export_back_compat_test_util_visibility,
    deps = if_building_jaxlib(
        if_building = [
            ":api",
            ":core",
            ":stages",
            ":export",
            ":test_util",
            ":tree_util",
            ":typing",
            ":xla_bridge",
        ],
        if_not_building = [],
    ) + py_deps("numpy") + py_deps("absl/logging"),
)

pytype_strict_library(
    name = "internal_export_back_compat_test_data",
    srcs = glob([
        "internal_test_util/export_back_compat_test_data/*.py",
        "internal_test_util/export_back_compat_test_data/pallas/*.py",
    ]),
    visibility = [
        "//jax:internal",
    ],
    deps = py_deps("numpy"),
)

pytype_strict_library(
    name = "abstract_arrays",
    srcs = ["abstract_arrays.py"],
    deps = [
        ":ad_util",
        ":config",
        ":core",
        ":dtypes",
        ":literals",
        ":traceback_util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "ad_util",
    srcs = ["ad_util.py"],
    deps = [
        ":core",
        ":traceback_util",
        ":tree_util",
        ":typing",
        ":util",
    ],
)

pytype_strict_library(
    name = "api",
    srcs = [
        "api.py",
        "array.py",
        "dispatch.py",
        "interpreters/pxla.py",
        "pjit.py",
    ],
    visibility = ["//jax:internal"] + jax_visibility("api"),
    deps = [
        ":abstract_arrays",
        ":ad",
        ":api_util",
        ":basearray",
        ":batching",
        ":compiler",
        ":config",
        ":core",
        ":deprecations",
        ":dtypes",
        ":effects",
        ":jaxpr_util",
        ":layout",
        ":literals",
        ":mesh",
        ":mlir",
        ":monitoring",
        ":op_shardings",
        ":partial_eval",
        ":partition_spec",
        ":profiler",
        ":sharding",
        ":sharding_impls",
        ":sharding_specs",
        ":source_info_util",
        ":stages",
        ":state_types",
        ":traceback_util",
        ":tree_util",
        ":typing",
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "api_util",
    srcs = ["api_util.py"],
    deps = [
        ":abstract_arrays",
        ":config",
        ":core",
        ":dtypes",
        ":state_types",
        ":traceback_util",
        ":tree_util",
        ":util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "basearray",
    srcs = ["basearray.py"],
    pytype_srcs = ["basearray.pyi"],
    deps = [
        ":literals",
        ":named_sharding",
        ":partition_spec",
        ":sharding",
        ":util",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "blocked_sampler",
    srcs = ["blocked_sampler.py"],
    deps = [
        ":numpy",
        ":random",
        ":typing",
    ],
)

pytype_strict_library(
    name = "buffer_callback",
    srcs = ["buffer_callback.py"],
    deps = [
        ":ad",
        ":api",
        ":batching",
        ":core",
        ":effects",
        ":ffi",
        ":mlir",
        ":tree_util",
        ":util",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "memory",
    srcs = ["memory.py"],
)

pytype_strict_library(
    name = "callback",
    srcs = ["callback.py"],
    deps = [
        ":ad",
        ":api",
        ":batching",
        ":config",
        ":core",
        ":dtypes",
        ":effects",
        ":ffi",
        ":mlir",
        ":pickle_util",
        ":sharding",
        ":sharding_impls",
        ":tree_util",
        ":typing",
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "checkify",
    srcs = ["checkify.py"],
    visibility = ["//jax:internal"] + jax_visibility("checkify"),
    deps = [
        ":ad",
        ":ad_util",
        ":api",
        ":api_util",
        ":batching",
        ":callback",
        ":config",
        ":core",
        ":custom_derivatives",
        ":dtypes",
        ":effects",
        ":lax",
        ":mesh",
        ":mlir",
        ":numpy",
        ":partial_eval",
        ":partition_spec",
        ":shard_map",
        ":sharding_impls",
        ":source_info_util",
        ":traceback_util",
        ":tree_util",
        ":typing",
        ":util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "cloud_tpu_init",
    srcs = ["cloud_tpu_init.py"],
    deps = [
        ":config",
        ":hardware_utils",
    ],
)

pytype_strict_library(
    name = "compilation_cache_internal",
    srcs = ["compilation_cache.py"],
    visibility = ["//jax:internal"] + jax_visibility("compilation_cache"),
    deps = [
        ":cache_key",
        ":compilation_cache_interface",
        ":config",
        ":lru_cache",
        ":monitoring",
        ":path",
        "//jax/_src/lib",
    ] + py_deps("numpy") + py_deps("zstandard"),
)

pytype_strict_library(
    name = "cache_key",
    srcs = ["cache_key.py"],
    visibility = ["//jax:internal"] + jax_visibility("compilation_cache"),
    deps = [
        ":config",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "compilation_cache_interface",
    srcs = ["compilation_cache_interface.py"],
    deps = [":util"],
)

py_library_providing_imports_info(
    name = "lax",
    srcs = glob(
        [
            "lax/**/*.py",
            "state/**/*.py",
        ],
    ) + [
        "ad_checkpoint.py",
    ],
    visibility = ["//jax:internal"] + jax_visibility("lax"),
    deps = [
        ":abstract_arrays",
        ":ad",
        ":ad_util",
        ":api",
        ":api_util",
        ":batching",
        ":callback",
        ":config",
        ":core",
        ":custom_derivatives",
        ":custom_partitioning_sharding_rule",
        ":deprecations",
        ":dtypes",
        ":effects",
        ":ffi",
        ":literals",
        ":mesh",
        ":mlir",
        ":named_sharding",
        ":partial_eval",
        ":partition_spec",
        ":pretty_printer",
        ":sharding",
        ":sharding_impls",
        ":source_info_util",
        ":state_types",
        ":traceback_util",
        ":tree_util",
        ":typing",
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "lru_cache",
    srcs = ["lru_cache.py"],
    deps = [
        ":compilation_cache_interface",
        ":path",
    ] + py_deps("filelock"),
)

pytype_strict_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
        ":deprecations",
        ":logging_config",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "logging_config",
    srcs = ["logging_config.py"],
    deps = ["//jax/_src/lib"],
)

pytype_strict_library(
    name = "compiler",
    srcs = ["compiler.py"],
    visibility = ["//jax:internal"] + jax_visibility("compiler"),
    deps = [
        ":cache_key",
        ":compilation_cache_internal",
        ":config",
        ":mlir",
        ":monitoring",
        ":path",
        ":profiler",
        ":traceback_util",
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "core",
    srcs = [
        "core.py",
        "errors.py",
        "linear_util.py",
    ],
    deps = [
        ":config",
        ":dtypes",
        ":effects",
        ":layout",
        ":memory",
        ":mesh",
        ":named_sharding",
        ":partition_spec",
        ":pretty_printer",
        ":sharding",
        ":source_info_util",
        ":traceback_util",
        ":tree_util",
        ":typing",
        ":util",
        ":xla_metadata_lib",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "literals",
    srcs = ["literals.py"],
    deps = [
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "custom_api_util",
    srcs = ["custom_api_util.py"],
)

pytype_strict_library(
    name = "custom_batching",
    srcs = ["custom_batching.py"],
    deps = [
        ":ad",
        ":api",
        ":api_util",
        ":batching",
        ":core",
        ":custom_api_util",
        ":mlir",
        ":partial_eval",
        ":source_info_util",
        ":traceback_util",
        ":tree_util",
        ":util",
    ],
)

pytype_strict_library(
    name = "custom_dce",
    srcs = ["custom_dce.py"],
    deps = [
        ":ad",
        ":api_util",
        ":batching",
        ":core",
        ":custom_api_util",
        ":mlir",
        ":partial_eval",
        ":source_info_util",
        ":traceback_util",
        ":tree_util",
        ":util",
    ],
)

pytype_strict_library(
    name = "custom_derivatives",
    srcs = ["custom_derivatives.py"],
    deps = [
        ":ad",
        ":ad_util",
        ":api",
        ":api_util",
        ":batching",
        ":config",
        ":core",
        ":custom_api_util",
        ":custom_transpose",
        ":dtypes",
        ":effects",
        ":mlir",
        ":partial_eval",
        ":state_types",
        ":traceback_util",
        ":tree_util",
        ":util",
    ],
)

pytype_strict_library(
    name = "custom_partitioning",
    srcs = ["custom_partitioning.py"],
    deps = [
        ":api",
        ":api_util",
        ":config",
        ":core",
        ":custom_api_util",
        ":custom_partitioning_sharding_rule",
        ":mesh",
        ":mlir",
        ":partial_eval",
        ":sharding",
        ":sharding_impls",
        ":tree_util",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "custom_partitioning_sharding_rule",
    srcs = ["custom_partitioning_sharding_rule.py"],
    deps = [
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "custom_transpose",
    srcs = ["custom_transpose.py"],
    deps = [
        ":ad",
        ":ad_util",
        ":api",
        ":api_util",
        ":core",
        ":custom_api_util",
        ":mlir",
        ":partial_eval",
        ":source_info_util",
        ":traceback_util",
        ":tree_util",
        ":util",
    ],
)

pytype_strict_library(
    name = "debugger",
    srcs = glob(["debugger/**/*.py"]),
    deps = [
        ":callback",
        ":core",
        ":debugging",
        ":lax",
        ":traceback_util",
        ":tree_util",
        ":util",
    ],
)

pytype_strict_library(
    name = "debugging",
    srcs = [
        "debugging.py",
    ],
    deps = [
        ":ad",
        ":api",
        ":batching",
        ":callback",
        ":config",
        ":core",
        ":effects",
        ":lax",
        ":mesh",
        ":mlir",
        ":numpy",
        ":partial_eval",
        ":shard_map",
        ":sharding",
        ":sharding_impls",
        ":source_info_util",
        ":tree_util",
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "deprecations",
    srcs = ["deprecations.py"],
)

pytype_strict_library(
    name = "dlpack",
    srcs = ["dlpack.py"],
    deps = [
        ":api",
        ":deprecations",
        ":dtypes",
        ":lax",
        ":numpy",
        ":sharding",
        ":typing",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "dtypes",
    srcs = [
        "dtypes.py",
    ],
    deps = [
        ":config",
        ":literals",
        ":traceback_util",
        ":typing",
        ":util",
        "//jax/_src/lib",
    ] + py_deps("ml_dtypes") + py_deps("numpy"),
)

pytype_strict_library(
    name = "earray",
    srcs = ["earray.py"],
    deps = [
        ":api",
        ":basearray",
        ":core",
        ":dtypes",
        ":sharding_impls",
        ":tree_util",
        ":util",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "effects",
    srcs = ["effects.py"],
)

pytype_strict_library(
    name = "environment_info",
    srcs = ["environment_info.py"],
    deps = [
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "error_check",
    srcs = ["error_check.py"],
    deps = [
        ":core",
        ":export",
        ":lax",
        ":mesh",
        ":shard_map",
        ":sharding_impls",
        ":source_info_util",
        ":traceback_util",
        ":tree_util",
        ":typing",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "export",
    srcs = glob([
        "export/**/*.py",
    ]),
    visibility = ["//jax:internal"] + jax_visibility("export"),
    deps = [
        ":ad_util",
        ":api",
        ":config",
        ":core",
        ":custom_derivatives",
        ":dtypes",
        ":effects",
        ":mesh",
        ":mlir",
        ":sharding",
        ":sharding_impls",
        ":source_info_util",
        ":stages",
        ":traceback_util",
        ":tree_util",
        ":typing",
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("flatbuffers") + py_deps("numpy") + py_deps("opt_einsum"),
)

pytype_strict_library(
    name = "ffi",
    srcs = ["ffi.py"],
    deps = [
        ":ad",
        ":api",
        ":batching",
        ":core",
        ":effects",
        ":frozen_dict",
        ":hashable_array",
        ":layout",
        ":mlir",
        ":typing",
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "flatten_util",
    srcs = [
        "flatten_util.py",
    ],
    deps = [
        ":dtypes",
        ":lax",
        ":tree_util",
        ":typing",
        ":util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "frozen_dict",
    srcs = ["frozen_dict.py"],
)

pytype_strict_library(
    name = "hardware_utils",
    srcs = ["hardware_utils.py"],
)

pytype_strict_library(
    name = "hashable_array",
    srcs = ["hashable_array.py"],
    deps = py_deps("numpy"),
)

pytype_strict_library(
    name = "image",
    srcs = glob([
        "image/**/*.py",
    ]),
    visibility = ["//jax:internal"] + jax_visibility("image"),  # buildifier: disable=visibility-as-string-list
    deps = [
        ":api",
        ":core",
        ":dtypes",
        ":lax",
        ":numpy",
        ":util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "lax_reference",
    srcs = ["lax_reference.py"],
    visibility = ["//jax:internal"] + jax_visibility("lax_reference"),
    deps = [
        ":core",
        ":dtypes",
        ":util",
    ] + py_deps("numpy") + py_deps("scipy") + py_deps("opt_einsum"),
)

pytype_strict_library(
    name = "lazy_loader",
    srcs = ["lazy_loader.py"],
)

pytype_strict_library(
    name = "jaxpr_util",
    srcs = ["jaxpr_util.py"],
    deps = [
        ":config",
        ":core",
        ":path",
        ":source_info_util",
        ":util",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "mesh",
    srcs = ["mesh.py"],
    deps = [
        ":config",
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "ad",
    srcs = ["interpreters/ad.py"],
    deps = [
        ":ad_util",
        ":api_util",
        ":config",
        ":core",
        ":dtypes",
        ":mesh",
        ":partial_eval",
        ":source_info_util",
        ":state_types",
        ":tree_util",
        ":util",
    ],
)

pytype_strict_library(
    name = "batching",
    srcs = ["interpreters/batching.py"],
    deps = [
        ":ad_util",
        ":config",
        ":core",
        ":mesh",
        ":partial_eval",
        ":partition_spec",
        ":sharding_impls",
        ":source_info_util",
        ":tree_util",
        ":typing",
        ":util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "mlir",
    srcs = ["interpreters/mlir.py"],
    deps = [
        ":ad_util",
        ":api_util",
        ":config",
        ":core",
        ":dtypes",
        ":effects",
        ":frozen_dict",
        ":hashable_array",
        ":jaxpr_util",
        ":layout",
        ":literals",
        ":mesh",
        ":op_shardings",
        ":partial_eval",
        ":partition_spec",
        ":path",
        ":pickle_util",
        ":sharding",
        ":sharding_impls",
        ":source_info_util",
        ":state_types",
        ":typing",
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "monitoring",
    srcs = ["monitoring.py"],
)

pytype_strict_library(
    name = "op_shardings",
    srcs = ["op_shardings.py"],
    deps = [
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "scipy",
    srcs = glob([
        "scipy/**/*.py",
        "third_party/**/*.py",
    ]),
    deps = [
        ":api",
        ":api_util",
        ":config",
        ":core",
        ":custom_derivatives",
        ":deprecations",
        ":dtypes",
        ":lax",
        ":nn",
        ":numpy",
        ":random",
        ":tpu",
        ":tree_util",
        ":typing",
        ":util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "sourcemap",
    srcs = ["sourcemap.py"],
)

pytype_strict_library(
    name = "partial_eval",
    srcs = ["interpreters/partial_eval.py"],
    visibility = ["//jax:internal"] + jax_visibility("partial_eval"),
    deps = [
        ":ad_util",
        ":api_util",
        ":config",
        ":core",
        ":dtypes",
        ":effects",
        ":profiler",
        ":source_info_util",
        ":state_types",
        ":tree_util",
        ":util",
        ":xla_metadata_lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "partition_spec",
    srcs = ["partition_spec.py"],
    deps = [
        ":util",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "path",
    srcs = ["path.py"],
    deps = py_deps("epath"),
)

pytype_strict_library(
    name = "pickle_util",
    srcs = ["pickle_util.py"],
    deps = [":profiler"] + py_deps("cloudpickle"),
)

pytype_strict_library(
    name = "pretty_printer",
    srcs = ["pretty_printer.py"],
    visibility = ["//jax:internal"] + jax_visibility("pretty_printer"),
    deps = [
        ":config",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "profiler",
    srcs = ["profiler.py"],
    deps = [
        ":traceback_util",
        ":xla_bridge",
        "//jax/_src/lib",
        "//jaxlib:_jax",
    ],
)

pytype_strict_library(
    name = "pmap",
    srcs = ["pmap.py"],
    deps = [
        ":api",
        ":core",
        ":lax",
        ":mesh",
        ":shard_map",
        ":stages",
        ":traceback_util",
        ":tree_util",
        ":util",
        ":xla_bridge",
    ],
)

pytype_strict_library(
    name = "public_test_util",
    srcs = [
        "public_test_util.py",
    ],
    deps = [
        ":api",
        ":config",
        ":dtypes",
        ":tree_util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "sharding",
    srcs = ["sharding.py"],
    deps = [
        ":op_shardings",
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "shard_alike",
    srcs = [
        "shard_alike.py",
    ],
    deps = [
        ":ad",
        ":api",
        ":batching",
        ":config",
        ":core",
        ":mlir",
        ":tree_util",
        ":util",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "shard_map",
    srcs = ["shard_map.py"],
    deps = [
        ":ad",
        ":ad_util",
        ":api",
        ":api_util",
        ":batching",
        ":config",
        ":core",
        ":dtypes",
        ":effects",
        ":lax",
        ":layout",
        ":mesh",
        ":mlir",
        ":partial_eval",
        ":sharding",
        ":sharding_impls",
        ":source_info_util",
        ":stages",
        ":traceback_util",
        ":tree_util",
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "hijax",
    srcs = ["hijax.py"],
    deps = [
        ":ad",
        ":ad_util",
        ":batching",
        ":core",
        ":dtypes",
        ":effects",
        ":tree_util",
        ":util",
    ],
)

pytype_strict_library(
    name = "stages",
    srcs = ["stages.py"],
    visibility = ["//jax:internal"] + jax_visibility("stages"),
    deps = [
        ":config",
        ":core",
        ":layout",
        ":mlir",
        ":sharding",
        ":sharding_impls",
        ":source_info_util",
        ":traceback_util",
        ":tree_util",
        ":typing",
        ":util",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "compute_on",
    srcs = ["compute_on.py"],
    deps = [
        ":ad",
        ":api",
        ":api_util",
        ":batching",
        ":config",
        ":core",
        ":mlir",
        ":partial_eval",
        ":tree_util",
        ":util",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "xla_metadata",
    srcs = ["xla_metadata.py"],
    deps = [
        ":ad",
        ":api",
        ":batching",
        ":config",
        ":core",
        ":mlir",
        ":tree_util",
        ":xla_metadata_lib",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "xla_metadata_lib",
    srcs = ["xla_metadata_lib.py"],
    deps = [
        ":config",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "layout",
    srcs = ["layout.py"],
    deps = [
        ":dtypes",
        ":named_sharding",
        ":sharding",
        ":util",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "sharding_impls",
    srcs = ["sharding_impls.py"],
    visibility = ["//jax:internal"],
    deps = [
        ":config",
        ":core",
        ":deprecations",
        ":internal_mesh_utils",
        ":mesh",
        ":named_sharding",
        ":op_shardings",
        ":partition_spec",
        ":sharding",
        ":sharding_specs",
        ":source_info_util",
        ":tree_util",
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "named_sharding",
    srcs = ["named_sharding.py"],
    deps = [
        ":config",
        ":mesh",
        ":partition_spec",
        ":sharding",
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "nn",
    srcs = glob([
        "nn/**/*.py",
    ]),
    deps = [
        ":api",
        ":config",
        ":core",
        ":cudnn",
        ":custom_derivatives",
        ":deprecations",
        ":dtypes",
        ":lax",
        ":named_sharding",
        ":numpy",
        ":partition_spec",
        ":random",
        ":sharding_impls",
        ":typing",
        ":util",
    ] + py_deps("numpy"),
)

py_library_providing_imports_info(
    name = "numpy",
    srcs = glob([
        "numpy/**/*.py",
        "ops/**/*.py",
    ]),
    deps = [
        ":api",
        ":api_util",
        ":config",
        ":core",
        ":custom_derivatives",
        ":deprecations",
        ":dtypes",
        ":error_check",
        ":export",
        ":lax",
        ":literals",
        ":mesh",
        ":sharding",
        ":sharding_impls",
        ":tree_util",
        ":typing",
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy") + py_deps("opt_einsum"),
)

pytype_strict_library(
    name = "sharding_specs",
    srcs = ["sharding_specs.py"],
    deps = [
        ":config",
        ":op_shardings",
        ":util",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "internal_mesh_utils",
    srcs = ["mesh_utils.py"],
    deps = [
        ":xla_bridge",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "source_info_util",
    srcs = ["source_info_util.py"],
    visibility = ["//jax:internal"] + jax_visibility("source_info_util"),
    deps = [
        ":traceback_util",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "state_types",
    srcs = [
        "state/__init__.py",
        "state/indexing.py",
        "state/types.py",
    ],
    visibility = ["//jax:internal"] + jax_visibility("state_types"),
    deps = [
        ":core",
        ":dtypes",
        ":effects",
        ":pretty_printer",
        ":traceback_util",
        ":tree_util",
        ":typing",
        ":util",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "tpu",
    srcs = glob([
        "tpu/**/*.py",
    ]),
    deps = [
        ":api",
        ":config",
        ":core",
        ":dtypes",
        ":lax",
        ":mlir",
        ":numpy",
        ":traceback_util",
        ":tree_util",
        ":typing",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "tree",
    srcs = ["tree.py"],
    deps = [":tree_util"],
)

pytype_strict_library(
    name = "tree_util",
    srcs = ["tree_util.py"],
    visibility = ["//jax:internal"] + jax_visibility("tree_util"),
    deps = [
        ":traceback_util",
        ":util",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "traceback_util",
    srcs = ["traceback_util.py"],
    visibility = ["//jax:internal"] + jax_visibility("traceback_util"),
    deps = [
        ":config",
        ":util",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "typing",
    srcs = [
        "typing.py",
    ],
    deps = [":basearray"] + py_deps("numpy"),
)

pytype_strict_library(
    name = "tpu_custom_call",
    srcs = ["tpu_custom_call.py"],
    visibility = ["//jax:internal"],
    deps = [
        ":api",
        ":batching",
        ":cloud_tpu_init",
        ":config",
        ":core",
        ":frozen_dict",
        ":mlir",
        ":sharding_impls",
        "//jax/_src/lib",
        "//jax/_src/pallas",
    ] + if_building_jaxlib([
        "//jaxlib/mlir:ir",
        "//jaxlib/mlir:mhlo_dialect",
        "//jaxlib/mlir:pass_manager",
        "//jaxlib/mlir:stablehlo_dialect",
    ]) + py_deps("numpy") + py_deps("absl/flags"),
)

pytype_strict_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        ":config",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

# TODO(phawkins): break up this SCC.
pytype_strict_library(
    name = "xla_bridge",
    srcs = [
        "clusters/__init__.py",
        "clusters/cloud_tpu_cluster.py",
        "clusters/cluster.py",
        "clusters/k8s_cluster.py",
        "clusters/mpi4py_cluster.py",
        "clusters/ompi_cluster.py",
        "clusters/slurm_cluster.py",
        "distributed.py",
        "xla_bridge.py",
    ],
    visibility = ["//jax:internal"] + jax_visibility("xla_bridge"),
    deps = [
        ":cloud_tpu_init",
        ":config",
        ":hardware_utils",
        ":traceback_util",
        ":util",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "cudnn",
    srcs = glob(["cudnn/**/*.py"]),
    deps = [
        ":api",
        ":batching",
        ":core",
        ":custom_derivatives",
        ":custom_partitioning",
        ":custom_partitioning_sharding_rule",
        ":dtypes",
        ":lax",
        ":mlir",
        ":numpy",
        ":sharding_impls",
        ":tree_util",
        ":typing",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "extend_src",
    srcs = glob(include = ["extend/**/*.py"]),
    deps = [
        ":random",
        ":typing",
    ],
)

pytype_strict_library(
    name = "random",
    srcs = [
        "prng.py",
        "random.py",
    ],
    visibility = ["//jax:internal"] + jax_visibility("random"),
    deps = [
        ":ad",
        ":api",
        ":batching",
        ":config",
        ":core",
        ":dtypes",
        ":ffi",
        ":lax",
        ":literals",
        ":mesh",
        ":mlir",
        ":numpy",
        ":pretty_printer",
        ":sharding_impls",
        ":source_info_util",
        ":tree_util",
        ":typing",
        ":util",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_strict_library(
    name = "ref",
    srcs = ["ref.py"],
    deps = [":core"],
)
