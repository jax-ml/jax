# Copyright 2023 The JAX Authors.
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
    "jax_generate_backend_suites",
    "jax_multiplatform_test",
    "jax_py_test",
    "py_deps",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:private"],
)

jax_generate_backend_suites()

jax_multiplatform_test(
    name = "pallas_test",
    srcs = [
        "pallas_test.py",
    ],
    enable_backends = [
        "cpu",
        "tpu",
    ],
    enable_configs = [
        "gpu_a100",
        "gpu_h100",
    ],
    shard_count = {
        "cpu": 8,
        "gpu": 4,
        "tpu": 4,
    },
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",
        "//jax:pallas_gpu_ops",
        "//jax:pallas_tpu",
        "//jax:pallas_tpu_ops",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_multiplatform_test(
    name = "pallas_cost_estimate_test",
    srcs = [
        "pallas_cost_estimate_test.py",
    ],
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",
        "//jax:pallas_gpu_ops",
        "//jax:pallas_tpu",
        "//jax:pallas_tpu_ops",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_multiplatform_test(
    name = "pallas_jumble_test",
    srcs = [
        "pallas_jumble_test.py",
    ],
    disable_configs = [
        "gpu_v100",
        "gpu_x32",
        "gpu_a100",
        "gpu_p100",
        "gpu_p100_x32",
        "gpu_h100",
    ],
    deps = [
        "//jax:pallas",
        "//jax:pallas_tpu",
        "//jax:pallas_tpu_ops",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_multiplatform_test(
    name = "ops_test",
    srcs = [
        "ops_test.py",
    ],
    disable_configs = [
        "gpu_v100",
        "gpu_x32",
        "gpu_p100",
        "gpu_p100_x32",
    ],
    enable_configs = [
        "gpu_a100",
        "gpu_a100_x32",
        "gpu_h100",
        "gpu_h100_x32",
    ],
    shard_count = {
        "cpu": 16,
        "gpu": 16,
        "tpu": 16,
    },
    tags = [
        "noasan",  # Times out.
        "nomsan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",  # build_cleaner: keep
        "//jax:pallas_tpu",
        "//jax:pallas_tpu_ops",
    ] + py_deps("absl/testing") + py_deps("hypothesis") + py_deps("numpy"),
)

jax_multiplatform_test(
    name = "indexing_test",
    srcs = [
        "indexing_test.py",
    ],
    enable_backends = [
        "cpu",
        "tpu",
    ],
    tags = [
        "noasan",  # Times out.
        "nomsan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax:pallas",
        "//jax:pallas_tpu",
    ] + py_deps("absl/testing") + py_deps("hypothesis") + py_deps("numpy"),
)

jax_multiplatform_test(
    name = "pallas_vmap_test",
    srcs = [
        "pallas_vmap_test.py",
    ],
    enable_backends = ["cpu"],
    enable_configs = [
        "gpu_a100_x32",
        "gpu_h100_x32",
    ],
    shard_count = 4,
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",
        "//jax:pallas_gpu_ops",
        "//jax:pallas_tpu",
        "//jax:pallas_tpu_ops",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_multiplatform_test(
    name = "mosaic_gpu_test",
    srcs = [
        "mosaic_gpu_test.py",
    ],
    enable_backends = [],
    enable_configs = [
        "gpu_h100_x32",
    ],
    env = {
        "JAX_PALLAS_USE_MOSAIC_GPU": "1",
        "JAX_PALLAS_VERBOSE_ERRORS": "0",
    },
    deps = [
        "//jax:pallas",
        "//jax:pallas_mosaic_gpu",  # build_cleaner: keep
        "//jax/_src/pallas/mosaic_gpu",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_multiplatform_test(
    name = "export_back_compat_pallas_test",
    srcs = ["export_back_compat_pallas_test.py"],
    enable_backends = ["cpu"],
    enable_configs = [
        "gpu_a100_x32",
        "gpu_h100_x32",
    ],
    tags = [],
    deps = [
        "//jax:internal_export_back_compat_test_data",
        "//jax:internal_export_back_compat_test_util",
        "//jax:pallas",
        "//jax:pallas_gpu",  # build_cleaner: keep
        "//jax:pallas_tpu_ops",  # build_cleaner: keep
    ],
)

jax_multiplatform_test(
    name = "export_pallas_test",
    srcs = ["export_pallas_test.py"],
    enable_backends = ["cpu"],
    enable_configs = [
        "gpu_a100_x32",
    ],
    tags = [],
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",  # build_cleaner: keep
        "//jax:pallas_tpu",  # build_cleaner: keep
    ],
)

jax_multiplatform_test(
    name = "pallas_shape_poly_test",
    srcs = ["pallas_shape_poly_test.py"],
    disable_configs = [
        "gpu_x32",
        "gpu_h100",
        "gpu_p100",
        "gpu_p100_x32",
        "gpu_pjrt_c_api",
    ],
    enable_configs = [
        "gpu_a100_x32",
    ],
    tags = [],
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",  # build_cleaner: keep
        "//jax:pallas_tpu",  # build_cleaner: keep
    ],
)

jax_multiplatform_test(
    name = "pallas_error_handling_test",
    srcs = [
        "pallas_error_handling_test.py",
    ],
    enable_backends = ["tpu"],
    deps = [
        "//jax:pallas",
        "//jax:pallas_tpu",
        "//jax/_src/pallas/mosaic:random",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_multiplatform_test(
    name = "tpu_all_gather_test",
    srcs = [
        "tpu_all_gather_test.py",
    ],
    enable_backends = [],
    enable_configs = [
        "tpu_v5e_4x2",
    ],
    deps = [
        "//jax:pallas_tpu_ops",
    ] + py_deps("absl/testing") + py_deps("numpy") + py_deps("hypothesis"),
)

jax_multiplatform_test(
    name = "tpu_gmm_test",
    srcs = [
        "tpu_gmm_test.py",
    ],
    enable_backends = ["tpu"],
    shard_count = 50,
    tags = [
        "noasan",  # Times out.
        "nomsan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax:pallas_tpu_ops",
    ] + py_deps([
        "absl/testing",
        "absl/flags",
        "numpy",
        "hypothesis",
    ]),
)

jax_multiplatform_test(
    name = "tpu_pallas_test",
    srcs = ["tpu_pallas_test.py"],
    # The flag is necessary for ``pl.debug_print`` tests to work on TPU.
    args = ["--logtostderr"],
    enable_backends = ["tpu"],
    enable_configs = [
        "tpu_v5e",
        "tpu_v5p_1x1",
    ],
    deps = [
        "//jax:extend",
        "//jax:pallas_tpu",
        "//jax:pallas_tpu_ops",
    ],
)

jax_multiplatform_test(
    name = "tpu_ops_test",
    srcs = [
        "tpu_ops_test.py",
    ],
    enable_backends = [
        "cpu",
        "tpu",
    ],
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",  # build_cleaner: keep
        "//jax:pallas_tpu",
        "//jax:pallas_tpu_ops",
    ] + py_deps("absl/testing") + py_deps("hypothesis") + py_deps("numpy"),
)

jax_multiplatform_test(
    name = "tpu_pallas_distributed_test",
    srcs = ["tpu_pallas_distributed_test.py"],
    enable_backends = ["tpu"],
    enable_configs = [
        "tpu_v5e_4x2",
        "tpu_v5p_2x2",
        "tpu_v4_2x2",
        "tpu_v3_2x2",
    ],
    deps = [
        "//jax:extend",
        "//jax:pallas_tpu",
        "//jax:pallas_tpu_ops",
    ],
)

jax_multiplatform_test(
    name = "tpu_pallas_pipeline_test",
    srcs = ["tpu_pallas_pipeline_test.py"],
    enable_backends = ["tpu"],
    enable_configs = [
        "tpu_v5e_4x2",
        "tpu_v5p_1x1",
    ],
    shard_count = 5,
    tags = [
        "noasan",  # Times out.
        "nomsan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax:extend",
        "//jax:pallas_tpu",
        "//jax:pallas_tpu_ops",
    ] + py_deps("hypothesis"),
)

jax_multiplatform_test(
    name = "tpu_pallas_async_test",
    srcs = ["tpu_pallas_async_test.py"],
    enable_backends = ["tpu"],
    enable_configs = [
        "tpu_v5e_4x2",
        "tpu_v5p_1x1",
    ],
    deps = [
        "//jax:pallas_tpu",
    ],
)

jax_multiplatform_test(
    name = "tpu_pallas_state_test",
    srcs = ["tpu_pallas_state_test.py"],
    enable_backends = ["tpu"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        "//jax:extend",
        "//jax:pallas_tpu",
    ],
)

jax_multiplatform_test(
    name = "tpu_pallas_random_test",
    srcs = [
        "tpu_pallas_random_test.py",
    ],
    enable_backends = ["tpu"],
    enable_configs = [
        "tpu_v5p_2x2",
    ],
    deps = [
        "//jax:pallas",
        "//jax:pallas_tpu",
        "//jax:pallas_tpu_ops",
        "//jax/_src/pallas/mosaic:random",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_multiplatform_test(
    name = "tpu_paged_attention_kernel_test",
    srcs = ["tpu_paged_attention_kernel_test.py"],
    disable_configs = [
        "tpu_v5p_1x1",
    ],
    enable_backends = ["tpu"],
    shard_count = 5,
    tags = [
        "noasan",  # Times out.
        "nomsan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax:pallas_tpu_ops",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_multiplatform_test(
    name = "tpu_splash_attention_kernel_test",
    srcs = [
        "tpu_splash_attention_kernel_test.py",
    ],
    enable_backends = ["tpu"],
    shard_count = 24,
    tags = [
        "noasan",  # Times out.
        "nomsan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax:pallas_tpu_ops",
    ] + py_deps("absl/testing") + py_deps("numpy") + py_deps("hypothesis"),
)

# This test doesn't need a TPU; it only tests numpy-using helpers.
jax_py_test(
    name = "tpu_splash_attention_mask_test",
    srcs = [
        "tpu_splash_attention_mask_test.py",
    ],
    deps = [
        "//jax",
        "//jax:pallas_tpu_ops",
        "//jax:test_util",
    ] + py_deps("absl/testing") + py_deps("numpy") + py_deps("hypothesis"),
)

jax_multiplatform_test(
    name = "gpu_attention_test",
    srcs = [
        "gpu_attention_test.py",
    ],
    enable_backends = ["cpu"],
    enable_configs = [
        "gpu_a100_x32",
        "gpu_h100_x32",
    ],
    shard_count = 1,
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",  # build_cleaner: keep
        "//jax:pallas_gpu_ops",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_multiplatform_test(
    name = "gpu_ops_test",
    srcs = [
        "gpu_ops_test.py",
    ],
    enable_backends = ["cpu"],
    enable_configs = [
        "gpu_a100_x32",
        "gpu_h100_x32",
    ],
    shard_count = 6,
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",
        "//jax:pallas_gpu_ops",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_multiplatform_test(
    name = "mgpu_attention_run",
    srcs = ["//jax/experimental/pallas/ops/gpu:attention_mgpu.py"],
    enable_backends = [],
    enable_configs = ["gpu_h100_x32"],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0"},
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//jax:pallas",
        "//jax:pallas_mosaic_gpu",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_multiplatform_test(
    name = "mgpu_attention_test",
    srcs = ["mgpu_attention_test.py"],
    enable_backends = [],
    enable_configs = ["gpu_h100_x32"],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0"},
    deps = [
        "//jax:pallas",
        "//jax:pallas_experimental_gpu_ops",
        "//jax:pallas_mosaic_gpu",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)
