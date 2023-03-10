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

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load(
    "//jaxlib:jax.bzl",
    "global_device_array_visibility",
    "jax_extra_deps",
    "jax_internal_packages",
    "jax_test_util_visibility",
    "jax_visibility",
    "py_deps",
    "py_library_providing_imports_info",
    "pytype_library",
    "pytype_strict_library",
)

package(default_visibility = [":internal"])

licenses(["notice"])

bool_flag(
    name = "build_jaxlib",
    build_setting_default = True,
)

config_setting(
    name = "enable_jaxlib_build",
    flag_values = {
        ":build_jaxlib": "True",
    },
)

exports_files([
    "LICENSE",
    "version.py",
])

# Packages that have access to JAX-internal implementation details.
package_group(
    name = "internal",
    packages = [
        "//...",
    ] + jax_internal_packages,
)

# JAX-private test utilities.
py_library(
    # This build target is required in order to use private test utilities in jax._src.test_util,
    # and its visibility is intentionally restricted to discourage its use outside JAX itself.
    # JAX does provide some public test utilities (see jax/test_util.py);
    # these are available in jax.test_util via the standard :jax target.
    name = "test_util",
    testonly = 1,
    srcs = [
        "_src/test_util.py",
    ],
    visibility = [
        ":internal",
    ] + jax_test_util_visibility,
    deps = [
        ":jax",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

py_library(
    name = "internal_test_util",
    testonly = 1,
    srcs = glob(["_src/internal_test_util/**/*.py"]),
    visibility = [":internal"],
    deps = [
        ":jax",
    ] + py_deps("numpy"),
)

py_library_providing_imports_info(
    name = "jax",
    srcs = [
        "_src/abstract_arrays.py",
        "_src/ad_checkpoint.py",
        "_src/ad_util.py",
        "_src/api.py",
        "_src/api_util.py",
        "_src/array.py",
        "_src/basearray.py",
        "_src/callback.py",
        "_src/checkify.py",
        "_src/core.py",
        "_src/custom_batching.py",
        "_src/custom_derivatives.py",
        "_src/custom_transpose.py",
        "_src/debugging.py",
        "_src/device_array.py",
        "_src/dispatch.py",
        "_src/dlpack.py",
        "_src/dtypes.py",
        "_src/errors.py",
        "_src/flatten_util.py",
        "_src/global_device_array.py",
        "_src/__init__.py",
        "_src/lax_reference.py",
        "_src/linear_util.py",
        "_src/maps.py",
        "_src/mesh.py",
        "_src/pjit.py",
        "_src/prng.py",
        "_src/public_test_util.py",
        "_src/random.py",
        "_src/sharding.py",
        "_src/stages.py",
        "_src/typing.py",
    ] + glob(
        [
            "*.py",
            "_src/debugger/**/*.py",
            "_src/image/**/*.py",
            "_src/interpreters/**/*.py",
            "_src/lax/**/*.py",
            "_src/nn/**/*.py",
            "_src/numpy/**/*.py",
            "_src/ops/**/*.py",
            "_src/scipy/**/*.py",
            "_src/state/**/*.py",
            "_src/third_party/**/*.py",
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
            "_src/internal_test_util/**",
        ],
    ) + [
        # until new parallelism APIs are moved out of experimental
        "experimental/maps.py",
        "experimental/pjit.py",
        "experimental/multihost_utils.py",
        "experimental/shard_map.py",
        # until checkify is moved out of experimental
        "experimental/checkify.py",
        # to avoid circular dependencies
        "experimental/compilation_cache/compilation_cache.py",
        "experimental/compilation_cache/gfile_cache.py",
        "experimental/compilation_cache/cache_interface.py",
    ],
    lib_rule = pytype_library,
    pytype_srcs = glob(["_src/**/*.pyi"]),
    visibility = ["//visibility:public"],
    deps = [
        ":cloud_tpu_init",
        ":custom_api_util",
        ":config",
        ":deprecations",
        ":effects",
        ":environment_info",
        ":lazy_loader",
        ":monitoring",
        ":path",
        ":pretty_printer",
        ":profiler",
        ":source_info_util",
        ":traceback_util",
        ":tree_util",
        ":util",
        ":version",
        ":xla_bridge",
        "//jax/_src/lib",
    ] + py_deps("numpy") + py_deps("scipy") + jax_extra_deps,
)

pytype_library(
    name = "cloud_tpu_init",
    srcs = ["_src/cloud_tpu_init.py"],
)

pytype_library(
    name = "config",
    srcs = ["_src/config.py"],
    deps = [
        "//jax/_src/lib",
    ],
)

pytype_library(
    name = "custom_api_util",
    srcs = ["_src/custom_api_util.py"],
)

pytype_library(
    name = "deprecations",
    srcs = ["_src/deprecations.py"],
)

pytype_library(
    name = "effects",
    srcs = ["_src/effects.py"],
)

pytype_library(
    name = "environment_info",
    srcs = ["_src/environment_info.py"],
    deps = [
        ":xla_bridge",
        ":version",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_library(
    name = "iree",
    srcs = ["_src/iree.py"],
    deps = [
        ":config",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
)

pytype_library(
    name = "lazy_loader",
    srcs = ["_src/lazy_loader.py"],
)

pytype_library(
    name = "monitoring",
    srcs = ["_src/monitoring.py"],
)

pytype_library(
    name = "path",
    srcs = ["_src/path.py"],
    deps = py_deps("epath"),
)

pytype_library(
    name = "pretty_printer",
    srcs = ["_src/pretty_printer.py"],
    deps = [":config"] + py_deps("colorama"),
)

pytype_library(
    name = "profiler",
    srcs = ["_src/profiler.py"],
    deps = [
        ":traceback_util",
        ":xla_bridge",
        "//jax/_src/lib",
    ],
)

pytype_library(
    name = "source_info_util",
    srcs = ["_src/source_info_util.py"],
    visibility = [":internal"] + jax_visibility("source_info_util"),
    deps = [
        ":traceback_util",
        ":version",
        "//jax/_src/lib",
    ],
)

pytype_library(
    name = "tree_util",
    srcs = ["_src/tree_util.py"],
    visibility = [":internal"] + jax_visibility("tree_util"),
    deps = [
        ":traceback_util",
        ":util",
        "//jax/_src/lib",
    ],
)

pytype_library(
    name = "traceback_util",
    srcs = ["_src/traceback_util.py"],
    visibility = [":internal"] + jax_visibility("traceback_util"),
    deps = [
        ":config",
        ":util",
        "//jax/_src/lib",
    ],
)

pytype_library(
    name = "util",
    srcs = ["_src/util.py"],
    deps = [
        ":config",
        "//jax/_src/lib",
    ],
)

pytype_strict_library(
    name = "version",
    srcs = ["version.py"],
)

# TODO(phawkins): break up this SCC.
pytype_library(
    name = "xla_bridge",
    srcs = [
        "_src/clusters/__init__.py",
        "_src/clusters/cloud_tpu_cluster.py",
        "_src/clusters/cluster.py",
        "_src/clusters/ompi_cluster.py",
        "_src/clusters/slurm_cluster.py",
        "_src/distributed.py",
        "_src/xla_bridge.py",
    ],
    visibility = [":internal"] + jax_visibility("xla_bridge"),
    deps = [
        ":cloud_tpu_init",
        ":config",
        ":iree",
        ":util",
        ":traceback_util",
        "//jax/_src/lib",
    ] + py_deps("numpy"),
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
            "experimental/global_device_array.py",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":jax",
    ] + py_deps("absl/logging") + py_deps("numpy"),
)

pytype_library(
    name = "stax",
    srcs = [
        "example_libraries/stax.py",
    ],
    visibility = ["//visibility:public"],
    deps = [":jax"],
)

pytype_library(
    name = "experimental_sparse",
    srcs = glob(
        [
            "experimental/sparse/*.py",
        ],
        exclude = ["experimental/sparse/test_util.py"],
    ),
    visibility = ["//visibility:public"],
    deps = [":jax"],
)

pytype_library(
    name = "sparse_test_util",
    testonly = 1,
    srcs = [
        "experimental/sparse/test_util.py",
    ],
    visibility = [":internal"],
    deps = [
        ":experimental_sparse",
        ":jax",
        ":test_util",
    ] + py_deps("numpy"),
)

pytype_library(
    name = "optimizers",
    srcs = [
        "example_libraries/optimizers.py",
    ],
    visibility = ["//visibility:public"],
    deps = [":jax"],
)

pytype_library(
    name = "ode",
    srcs = ["experimental/ode.py"],
    visibility = ["//visibility:public"],
    deps = [":jax"],
)

# TODO(apaszke): Remove this target
pytype_library(
    name = "maps",
    srcs = ["experimental/maps.py"],
    visibility = ["//visibility:public"],
    deps = [":jax"],
)

# TODO(apaszke): Remove this target
pytype_library(
    name = "pjit",
    srcs = ["experimental/pjit.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":experimental",
        ":jax",
    ],
)

pytype_library(
    name = "jet",
    srcs = ["experimental/jet.py"],
    visibility = ["//visibility:public"],
    deps = [":jax"],
)

pytype_library(
    name = "experimental_host_callback",
    srcs = ["experimental/host_callback.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":jax",
    ],
)

pytype_library(
    name = "compilation_cache",
    srcs = [
        "experimental/compilation_cache/compilation_cache.py",
        "experimental/compilation_cache/gfile_cache.py",
    ],
    visibility = ["//visibility:public"],
    deps = [":jax"],
)

pytype_library(
    name = "mesh_utils",
    srcs = ["experimental/mesh_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":experimental",
        ":jax",
    ],
)

pytype_library(
    name = "rnn",
    srcs = ["experimental/rnn.py"],
    visibility = ["//visibility:public"],
    deps = [":jax"],
)

pytype_library(
    name = "global_device_array",
    srcs = ["experimental/global_device_array.py"],
    visibility = [":internal"] + global_device_array_visibility,
    deps = [":jax"],
)
