# Copyright 2022 The JAX Authors.
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

load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("//jaxlib:jax.bzl", "if_oss", "jax_generate_backend_suites", "jax_multiplatform_test", "jax_visibility", "py_deps")

licenses(["notice"])

package(default_applicable_licenses = [])

jax_generate_backend_suites()

py_library(
    name = "jax2tf_tests",
    srcs = [
        "__init__.py",
    ],
    visibility = ["//jax:internal"],
)

py_library(
    name = "jax2tf_limitations",
    srcs = [
        "jax2tf_limitations.py",
    ],
    visibility = jax_visibility("jax2tf_limitations"),
    deps = [
        "//jax/_src:internal_test_harnesses",
        "//jax/_src:test_util",
    ],
)

py_library(
    name = "tf_test_util",
    srcs = [
        "tf_test_util.py",
    ],
    # //:wheel_additives needs visibility to this target
    visibility = jax_visibility("tf_test_util") + if_oss(["//:__pkg__"]),
    deps = [
        ":jax2tf_limitations",
        "//jax/_src:internal_test_harnesses",
        "//jax/_src:test_util",
        "//jax/experimental/jax2tf",
    ],
)

py_library(
    name = "models_util",
    srcs = [
        "converters.py",
        "model_harness.py",
    ],
    deps = [
        "//jax",
        "//jax/experimental/jax2tf/tests/flax_models",
        "//third_party/py/jraph",
        "//third_party/py/numpy",
        "//third_party/py/tensorflowjs",
    ],
)

jax_multiplatform_test(
    name = "jax2tf_test",
    srcs = ["jax2tf_test.py"],
    enable_configs = [
        "tpu_v3_x4",
    ],
    # On GPU, this test uses both TF and JAX and might OOM, see:
    # https://docs.jax.dev/en/latest/gpu_memory_allocation.html
    env = {
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
    },
    tags = ["jax2tf"],
    deps = [
        ":tf_test_util",
        "//jax/experimental/jax2tf",
    ] + py_deps("tensorflow") + py_deps([
        "absl/testing",
        "absl/logging",
    ]),
)

jax_multiplatform_test(
    name = "control_flow_ops_test",
    srcs = ["control_flow_ops_test.py"],
    config_tags_overrides = {
        "tpu_v3": {"ondemand": False},
    },
    disable_configs = [
        # TODO(b/172667839): Compilation times out with -c dbg.
        "tpu_v4i",
    ],
    # On GPU, this test uses both TF and JAX and might OOM, see:
    # https://docs.jax.dev/en/latest/gpu_memory_allocation.html
    env = {
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
    },
    tags = ["jax2tf"],
    deps = [
        ":tf_test_util",
        "//jax/experimental/jax2tf",
    ],
)

jax_multiplatform_test(
    name = "shape_poly_test",
    size = "large",
    srcs = ["shape_poly_test.py"],
    config_tags_overrides = {
        "tpu_v3": {"ondemand": False},
    },
    disable_configs = [
        "tpu_v4i",  # TODO(b/172667839): Compilation times out with -c dbg
    ],
    enable_configs = [
        "cpu",
        "cpu_x32",
    ],
    # On GPU, this test uses both TF and JAX and might OOM, see:
    # https://docs.jax.dev/en/latest/gpu_memory_allocation.html
    env = {
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
    },
    shard_count = {
        "cpu": 4,
        "gpu": 4,
        "tpu": 4,
    },
    tags = [
        "jax2tf",
    ],
    deps = [
        ":tf_test_util",
        "//jax/_src:internal_test_harnesses",
        "//jax/experimental/jax2tf",
    ],
)

jax_multiplatform_test(
    name = "primitives_test",
    size = "large",
    srcs = ["primitives_test.py"],
    backend_variant_args = {
        "tpu_v4i": ["--xla_tpu_allow_in_cmem_copy=true --xla_jf_convolution_performance_target=0.1"],  # TODO(b/172584898): slow compilation.
    },
    config_tags_overrides = {
        "tpu_v3": {"ondemand": False},
    },
    # On GPU, this test uses both TF and JAX and might OOM, see:
    # https://docs.jax.dev/en/latest/gpu_memory_allocation.html
    env = {
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
    },
    shard_count = {
        "cpu": 40,
        "gpu": 20,
        "tpu": 20,
    },
    tags = [
        "jax2tf",
        "nodebug",  # Times out.
    ],
    deps = [
        ":tf_test_util",
        "//jax/_src:internal_test_harnesses",
        "//jax/experimental/jax2tf",
    ],
)

jax_multiplatform_test(
    name = "jax_primitives_coverage_test",
    srcs = ["jax_primitives_coverage_test.py"],
    backend_variant_args = {
        "tpu_v4i": ["--xla_jf_convolution_performance_target=0.1"],  # TODO(b/172584898): slow compilation.
    },
    shard_count = {
        "cpu": 8,
        "gpu": 8,
        "tpu": 8,
    },
    tags = [
        "jax2tf",
        "nodebug",  # Times out.
    ],
    deps = [
        ":tf_test_util",
        "//jax/_src:internal_test_harnesses",
        "//jax/experimental/jax2tf",
    ],
)

jax_multiplatform_test(
    name = "savedmodel_test",
    srcs = ["savedmodel_test.py"],
    # On GPU, this test uses both TF and JAX and might OOM, see:
    # https://docs.jax.dev/en/latest/gpu_memory_allocation.html
    env = {
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
    },
    tags = ["jax2tf"],
    deps = [
        ":tf_test_util",
        "//jax/experimental/jax2tf",
    ],
)

jax_multiplatform_test(
    name = "sharding_test",
    srcs = ["sharding_test.py"],
    disable_configs = [
        "gpu_h100",
        "tpu_v4i",
    ],
    enable_configs = [
        "tpu_v3_x4",
    ],
    tags = [
        "jax2tf",
        "requires-net:external",  # For running with xprof
    ],
    deps = [
        ":tf_test_util",
        "//jax/_src:compiler",
        "//jax/experimental/jax2tf",
        "//third_party/py/absl:app",
    ],
)

jax_multiplatform_test(
    name = "call_tf_test",
    srcs = ["call_tf_test.py"],
    # On GPU, this test uses both TF and JAX and might OOM, see:
    # https://docs.jax.dev/en/latest/gpu_memory_allocation.html
    env = {
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
    },
    tags = ["jax2tf"],
    deps = [
        ":tf_test_util",
    ] + py_deps("tpu_ops"),
)

jax_multiplatform_test(
    name = "back_compat_tf_test",
    srcs = ["back_compat_tf_test.py"],
    # On GPU, this test uses both TF and JAX and might OOM, see:
    # https://docs.jax.dev/en/latest/gpu_memory_allocation.html
    env = {
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
    },
    tags = ["jax2tf"],
    deps = [
        "//jax/_src:internal_export_back_compat_test_util",
        "//jax/experimental/jax2tf",
        "//jax/experimental/jax2tf/tests/back_compat_testdata",
    ],
)

# This filegroup specifies the set of tests which should be run on forge, for
# the purposes of the verify_tests_in_build.
# If a test is external only, add the filename to the `exclude` list.
filegroup(
    name = "forge_tests",
    srcs = glob(
        include = [
            "*_test.py",
        ],
        exclude = [],
    ) + ["BUILD"],
    visibility = jax_visibility("forge_tests"),
)

py_binary(
    name = "models_test_main",
    srcs = ["models_test_main.py"],
    data = [
        "//jax/experimental/jax2tf/g3doc:convert_models_results",
    ],
    deps = [
        ":models_util",
        "//jax",
        "//pyglib:resources",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)
