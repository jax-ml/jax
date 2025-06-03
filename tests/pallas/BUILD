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
    "jax_gpu_support_deps",
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
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
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
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "pallas_jumble_test",
    srcs = [
        "pallas_jumble_test.py",
    ],
    disable_configs = [
        "gpu_v100",
        "gpu_v100_x32",
        "gpu_a100",
        "gpu_p100",
        "gpu_p100_x32",
        "gpu_h100",
    ],
    deps = [
        "//jax:pallas",
        "//jax:pallas_tpu",
        "//jax:pallas_tpu_ops",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "ops_test",
    srcs = [
        "ops_test.py",
    ],
    disable_configs = [
        "gpu_v100",
        "gpu_v100_x32",
        "gpu_p100",
        "gpu_p100_x32",
    ],
    enable_configs = [
        "gpu_a100",
        "gpu_a100_x32",
        "gpu_h100",
        "gpu_h100_x32",
        "tpu_v6e",
    ],
    shard_count = {
        "cpu": 16,
        "gpu": 32,
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
    ] + py_deps([
        "absl/testing",
        "hypothesis",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "ops_test_mgpu",
    srcs = [
        "ops_test.py",
    ],
    disable_configs = [
        "gpu_v100",
        "gpu_v100_x32",
        "gpu_p100",
        "gpu_p100_x32",
        "gpu_a100",
        "gpu_a100_x32",
    ],
    enable_backends = [
        "gpu",
    ],
    enable_configs = [
        "gpu_h100",
        "gpu_h100_x32",
    ],
    env = {
        "JAX_PALLAS_USE_MOSAIC_GPU": "1",
    },
    shard_count = 16,
    tags = [
        "noasan",  # Times out.
        "nomsan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",  # build_cleaner: keep
        "//jax:pallas_mosaic_gpu",  # build_cleaner: keep
        "//jax:pallas_tpu",
    ] + py_deps([
        "absl/testing",
        "hypothesis",
        "numpy",
    ]),
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
    ] + py_deps([
        "absl/testing",
        "hypothesis",
        "numpy",
    ]),
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
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "mosaic_gpu_test",
    srcs = [
        "mosaic_gpu_test.py",
    ],
    enable_backends = [],
    enable_configs = [
        "gpu_h100_x32",
        "gpu_h100",
    ],
    deps = [
        "//jax:pallas",
        "//jax:pallas_mosaic_gpu",  # build_cleaner: keep
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
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
        "//jax:pallas_mosaic_gpu",  # build_cleaner: keep
        "//jax:pallas_tpu_ops",  # build_cleaner: keep
    ] + py_deps("absl/testing"),
)

jax_py_test(
    name = "export_pallas_test_cpu_only",
    srcs = ["export_pallas_test.py"],
    args = ["--jax_test_dut=cpu"],
    main = "export_pallas_test.py",
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",  # build_cleaner: keep
        "//jax:pallas_tpu",  # build_cleaner: keep
        "//jax:test_util",
    ] + jax_gpu_support_deps + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "export_pallas_test",
    srcs = ["export_pallas_test.py"],
    # Cross-compilation on CPU is tested separately.
    disable_configs = [
        "cpu",
        "cpu_shardy",
        "cpu_x32",
    ],
    enable_configs = [
        "gpu_a100_x32",
    ],
    tags = [],
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",  # build_cleaner: keep
        "//jax:pallas_tpu",  # build_cleaner: keep
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "pallas_shape_poly_test",
    srcs = ["pallas_shape_poly_test.py"],
    disable_configs = [
        "gpu_h100",
        "gpu_p100",
        "gpu_p100_x32",
        "gpu_v100_x32",
        "gpu_p100_pjrt_c_api",
    ],
    enable_configs = [
        "gpu_a100_x32",
    ],
    tags = [],
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",  # build_cleaner: keep
        "//jax:pallas_tpu",  # build_cleaner: keep
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
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
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_all_gather_test",
    srcs = [
        "tpu_all_gather_test.py",
    ],
    enable_backends = [],
    enable_configs = [
        "tpu_v5e_x8",
    ],
    deps = [
        "//jax:pallas_tpu_ops",
    ] + py_deps([
        "absl/testing",
        "numpy",
        "hypothesis",
    ]),
)

jax_multiplatform_test(
    name = "tpu_gmm_test",
    srcs = [
        "tpu_gmm_test.py",
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
        "tpu_v5p",
    ],
    deps = [
        "//jax:extend",
        "//jax:pallas_tpu",
        "//jax:pallas_tpu_ops",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "gpu_pallas_distributed_test",
    srcs = ["gpu_pallas_distributed_test.py"],
    args = [
        "--num_processes=2",
        "--gpus_per_process=1",
    ],
    enable_backends = [],
    enable_configs = ["gpu_h100x2"],
    env = {
        "JAX_PALLAS_USE_MOSAIC_GPU": "1",
        "XLA_FLAGS": "--xla_gpu_experimental_enable_nvshmem=true",
    },
    tags = ["multiaccelerator"],
    deps = [
        "//jax:extend",
        "//jax:pallas_mosaic_gpu",
        "//jax:test_multiprocess",
    ] + py_deps([
        "portpicker",
        "absl/testing",
        "numpy",
    ]),
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
    ] + py_deps([
        "absl/testing",
        "hypothesis",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_pallas_distributed_test",
    srcs = ["tpu_pallas_distributed_test.py"],
    enable_backends = ["tpu"],
    enable_configs = [
        "tpu_v5e_x8",
        "tpu_v5p_x4",
        "tpu_v4_x4",
        "tpu_v3_x4",
    ],
    deps = [
        "//jax:extend",
        "//jax:pallas_tpu",
        "//jax:pallas_tpu_ops",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_pallas_pipeline_test",
    srcs = ["tpu_pallas_pipeline_test.py"],
    enable_backends = ["tpu"],
    enable_configs = [
        "tpu_v5e_x8",
        "tpu_v5p",
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
    ] + py_deps([
        "absl/testing",
        "numpy",
        "hypothesis",
    ]),
)

jax_multiplatform_test(
    name = "tpu_pallas_async_test",
    srcs = ["tpu_pallas_async_test.py"],
    enable_backends = ["tpu"],
    enable_configs = [
        "tpu_v5e_x8",
        "tpu_v5p",
    ],
    deps = [
        "//jax:pallas_tpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
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
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_pallas_random_test",
    srcs = [
        "tpu_pallas_random_test.py",
    ],
    enable_backends = ["tpu"],
    enable_configs = [
        "tpu_v5p_x4",
    ],
    deps = [
        "//jax:pallas",
        "//jax:pallas_tpu",
        "//jax:pallas_tpu_ops",
        "//jax/_src/pallas/mosaic:random",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_pallas_interpret_test",
    srcs = [
        "tpu_pallas_interpret_test.py",
    ],
    disable_configs = ["cpu_shardy"],
    enable_backends = ["cpu"],
    deps = [
        "//jax:experimental",
        "//jax:pallas",
        "//jax:pallas_tpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_pallas_interpret_distributed_test",
    srcs = [
        "tpu_pallas_interpret_distributed_test.py",
    ],
    disable_configs = ["cpu_shardy"],
    enable_backends = ["cpu"],
    deps = [
        "//jax:pallas",
        "//jax:pallas_tpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_paged_attention_kernel_test",
    srcs = ["tpu_paged_attention_kernel_test.py"],
    disable_configs = [
        "tpu_v5p",
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
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_ragged_paged_attention_test",
    srcs = ["tpu_ragged_paged_attention_test.py"],
    disable_configs = [
        "tpu_v5p",
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
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
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
    ] + py_deps([
        "absl/testing",
        "numpy",
        "hypothesis",
    ]),
)

jax_multiplatform_test(
    name = "tpu_splash_attention_kernel_sharded_test",
    srcs = ["tpu_splash_attention_kernel_sharded_test.py"],
    enable_configs = [
        "tpu_v5e_x8",
        "tpu_v5p_x4",
    ],
    shard_count = 5,
    deps = [
        "//jax:extend",
        "//jax:pallas_tpu",
        "//jax:pallas_tpu_ops",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
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
    ] + py_deps([
        "absl/testing",
        "numpy",
        "hypothesis",
    ]),
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
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
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
    shard_count = 20,
    tags = [
        "noasan",  # Times out.
        "nomsan",  # Times out.
        "notsan",  # Times out.
    ],
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",
        "//jax:pallas_gpu_ops",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "gpu_paged_attention_test",
    srcs = [
        "gpu_paged_attention_test.py",
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
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "triton_pallas_test",
    srcs = [
        "triton_pallas_test.py",
    ],
    enable_backends = ["cpu"],
    enable_configs = [
        "gpu_h100_x32",
    ],
    shard_count = 1,
    deps = [
        "//jax:pallas",
        "//jax:pallas_gpu",
    ] + py_deps([
        "absl/testing",
    ]),
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
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "mgpu_attention_test",
    srcs = ["mgpu_attention_test.py"],
    enable_backends = [],
    enable_configs = [
        "gpu_h100_x32",
        "gpu_h100",
    ],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0"},
    shard_count = 8,
    deps = [
        "//jax:pallas",
        "//jax:pallas_experimental_gpu_ops",
        "//jax:pallas_mosaic_gpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "mgpu_matmul_test",
    srcs = ["mgpu_matmul_test.py"],
    enable_backends = [],
    enable_configs = ["gpu_b200"],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0"},
    shard_count = 8,
    tags = [
        # TODO(b/330364373): Remove when B200 is fully supported.
        "notap",
    ],
    deps = [
        "//jax:pallas",
        "//jax:pallas_experimental_gpu_ops",
        "//jax:pallas_mosaic_gpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "blackwell_matmul_mgpu_run",
    srcs = ["//jax/experimental/pallas/ops/gpu:blackwell_matmul_mgpu.py"],
    enable_backends = [],
    enable_configs = ["gpu_b200"],
    env = {"XLA_FLAGS": "--xla_gpu_autotune_level=0"},
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//jax:pallas",
        "//jax:pallas_mosaic_gpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "mgpu_ragged_dot_run",
    srcs = ["//jax/experimental/pallas/ops/gpu:ragged_dot_mgpu.py"],
    enable_backends = [],
    enable_configs = [
        "gpu_h100_x32",
        "gpu_h100",
    ],
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
    name = "mgpu_ragged_dot_test",
    srcs = ["mgpu_ragged_dot_test.py"],
    enable_backends = [],
    enable_configs = [
        "gpu_h100",
    ],
    shard_count = 12,
    tags = [
        "noasan",  # Times out.
    ],
    deps = [
        "//jax:pallas",
        "//jax:pallas_experimental_gpu_ops",
        "//jax:pallas_mosaic_gpu",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "mgpu_collective_matmul_test",
    srcs = ["mgpu_collective_matmul_test.py"],
    args = [
        "--num_processes=2",
        "--gpus_per_process=1",
    ],
    enable_backends = [],
    enable_configs = [
        "gpu_h100x2",
    ],
    env = {
        "XLA_FLAGS": "--xla_gpu_experimental_enable_nvshmem=true",
        "JAX_PALLAS_USE_MOSAIC_GPU": "1",
    },
    shard_count = 4,
    tags = [
        "manual",
        "multiaccelerator",
        "notap",
    ],
    deps = [
        "//jax:experimental",
        "//jax:pallas",
        "//jax:pallas_experimental_gpu_ops",
        "//jax:pallas_mosaic_gpu",
        "//jax:test_multiprocess",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)

jax_multiplatform_test(
    name = "fuser_block_spec_test",
    srcs = [
        "fuser_block_spec_test.py",
    ],
    disable_configs = [
        "cpu",
        "cpu_shardy",
    ],
    enable_backends = ["cpu"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        "//jax:pallas",
        "//jax/_src/pallas/fuser",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "fusion_test",
    srcs = [
        "fusion_test.py",
    ],
    disable_configs = [
        "cpu",
        "cpu_shardy",
    ],
    enable_backends = ["cpu"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        "//jax:pallas",
        "//jax:pallas_fuser",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "tpu_fusible_matmul_test",
    srcs = ["tpu_fusible_matmul_test.py"],
    disable_configs = [
        "tpu_v3",
        "tpu_pjrt_c_api",
        "gpu_v100",
        "gpu_v100_x32",
        "gpu_a100",
        "gpu_p100",
        "gpu_p100_x32",
        "gpu_h100",
        "cpu",
        "cpu_x32",
        "cpu_shardy",
    ],
    enable_backends = ["tpu"],
    enable_configs = [
        "tpu_v4",
        "tpu_v5e",
        "tpu_v5p",
        "tpu_v6e",
    ],
    shard_count = 4,
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        "//jax:pallas_tpu",
        "//jax:pallas_tpu_ops",
        "//jax/_src/pallas/fuser",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)
