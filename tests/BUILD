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

load(
    "//jaxlib:jax.bzl",
    "jax_generate_backend_suites",
    "jax_multiplatform_test",
    "jax_py_test",
    "jax_test_file_visibility",
    "py_deps",
    "pytype_test",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:private"],
)

jax_generate_backend_suites()

jax_multiplatform_test(
    name = "api_test",
    srcs = ["api_test.py"],
    enable_configs = ["tpu_v3_x4"],
    shard_count = 10,
    deps = [
        "//jax:experimental",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "custom_api_test",
    srcs = ["custom_api_test.py"],
    shard_count = 10,
    deps = [
        "//jax:custom_derivatives",
        "//jax:experimental",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "debug_info_test",
    srcs = ["debug_info_test.py"],
    enable_configs = ["tpu_v3_x4"],
    deps = [
        "//jax:custom_transpose",
        "//jax:experimental",
        "//jax:pallas",
        "//jax:pallas_gpu",
        "//jax:pallas_gpu_ops",
        "//jax:pallas_tpu",
        "//jax:pallas_tpu_ops",
    ] + py_deps([
        "numpy",
        "absl/testing",
    ]),
)

jax_multiplatform_test(
    name = "device_test",
    srcs = ["device_test.py"],
    deps = py_deps("absl/testing"),
)

jax_py_test(
    name = "dynamic_api_test",
    srcs = ["dynamic_api_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "api_util_test",
    srcs = ["api_util_test.py"],
    deps = py_deps("absl/testing"),
)

jax_py_test(
    name = "array_api_test",
    srcs = ["array_api_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ] + py_deps("absl/testing"),
)

jax_py_test(
    name = "array_extensibility_test",
    srcs = ["array_extensibility_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "array_interoperability_test",
    srcs = ["array_interoperability_test.py"],
    disable_configs = [
        "gpu_h100_tfrt",  # TODO(b/411472145): Re-enable once fixed.
        "gpu_h100x2_tfrt",
    ],
    enable_backends = [
        "cpu",
        "gpu",
    ],
    enable_configs = [
        "gpu_h100x2",
    ],
    tags = [
        "multiaccelerator",
    ],
    deps = py_deps([
        "absl/testing",
        "numpy",
        "tensorflow_core",
    ]),
)

jax_multiplatform_test(
    name = "batching_test",
    srcs = ["batching_test.py"],
    shard_count = {
        "gpu": 5,
    },
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "buffer_callback_test",
    srcs = ["buffer_callback_test.py"],
    enable_backends = [
        "cpu",
        "gpu",
    ],
    deps = [
        "//jax:experimental_buffer_callback",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_py_test(
    name = "config_test",
    srcs = ["config_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ] + py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "core_test",
    srcs = ["core_test.py"],
    shard_count = {
        "cpu": 5,
        "gpu": 10,
    },
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "debug_nans_test",
    srcs = ["debug_nans_test.py"],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_py_test(
    name = "distributed_initialize_test",
    srcs = ["distributed_initialize_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ] + py_deps([
        "portpicker",
        "absl/testing",
    ]),
)

jax_multiplatform_test(
    name = "distributed_test",
    srcs = ["distributed_test.py"],
    enable_backends = ["gpu"],
    deps = py_deps([
        "portpicker",
        "absl/testing",
    ]),
)

jax_py_test(
    name = "multiprocess_gpu_test",
    srcs = ["multiprocess_gpu_test.py"],
    args = [
        "--exclude_test_targets=MultiProcessGpuTest",
    ],
    tags = ["manual"],
    deps = [
        "//jax",
        "//jax:test_util",
    ] + py_deps([
        "portpicker",
        "absl/testing",
    ]),
)

jax_multiplatform_test(
    name = "dtypes_test",
    srcs = ["dtypes_test.py"],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "errors_test",
    srcs = ["errors_test.py"],
    # No need to test all other configs.
    enable_configs = [
        "cpu",
    ],
    deps = py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "extend_test",
    srcs = ["extend_test.py"],
    deps = ["//jax:extend"] + py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "ffi_test",
    srcs = ["ffi_test.py"],
    enable_configs = [
        "gpu_h100x2",
    ],
    # TODO(dfm): Remove after removal of jex.ffi imports.
    deps = ["//jax:extend"] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "fft_test",
    srcs = ["fft_test.py"],
    backend_tags = {
        "tpu": [
            "noasan",
            "notsan",
        ],  # Times out on TPU with asan/tsan.
    },
    shard_count = {
        "tpu": 10,
        "cpu": 20,
        "gpu": 10,
    },
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "generated_fun_test",
    srcs = ["generated_fun_test.py"],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "gpu_memory_flags_test_no_preallocation",
    srcs = ["gpu_memory_flags_test.py"],
    enable_backends = ["gpu"],
    env = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "0",
    },
    main = "gpu_memory_flags_test.py",
    deps = py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "gpu_memory_flags_test",
    srcs = ["gpu_memory_flags_test.py"],
    enable_backends = ["gpu"],
    env = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "1",
    },
    deps = py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "lobpcg_test",
    srcs = ["lobpcg_test.py"],
    # Set LOBPCG_EMIT_DEBUG_PLOTS=1 to debug
    # checkLobpcgMonotonicity and checkApproxEigs tests
    # using matplotlib plots
    # env = {"LOBPCG_EMIT_DEBUG_PLOTS": "1"},
    shard_count = {
        "cpu": 8,
    },
    deps = [
        "//jax:experimental_sparse",
    ] + py_deps([
        "matplotlib",
        "absl/testing",
        "numpy",
        "scipy",
    ]),
)

jax_multiplatform_test(
    name = "svd_test",
    srcs = ["svd_test.py"],
    shard_count = {
        "cpu": 20,
        "gpu": 10,
        "tpu": 15,
    },
    deps = py_deps([
        "absl/testing",
        "numpy",
        "scipy",
    ]),
)

jax_py_test(
    name = "xla_interpreter_test",
    srcs = ["xla_interpreter_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ] + py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "memories_test",
    srcs = ["memories_test.py"],
    enable_configs = [
        "cpu",
        "gpu_h100x2",
        "tpu_v3_x4",
        "tpu_v4_x4",
        "tpu_v5p_x4",
        "tpu_v5e_x8",
        "gpu_p100x2_shardy",
        "tpu_v5e_x8_shardy",
    ],
    deps = [
        "//jax:experimental",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "pjit_test",
    srcs = ["pjit_test.py"],
    backend_tags = {
        "tpu": ["notsan"],  # Times out under tsan.
        "gpu": ["noasan"],  # Memory leaks in NCCL, see https://github.com/NVIDIA/nccl/pull/1143
    },
    enable_configs = [
        "gpu_p100x2_shardy",
        "tpu_v3_x4_shardy",
        "tpu_v3_x4",
        "gpu_h100x2",
    ],
    shard_count = {
        "cpu": 5,
        "gpu": 5,
        "tpu": 5,
    },
    tags = ["multiaccelerator"],
    deps = [
        "//jax:experimental",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "layout_test",
    srcs = ["layout_test.py"],
    backend_tags = {
        "tpu": ["requires-mem:16g"],  # Under tsan on 2x2 this test exceeds the default 12G memory limit.
    },
    enable_configs = [
        "tpu_v3_x4_shardy",
        "tpu_v3_x4",
    ],
    tags = ["multiaccelerator"],
    deps = [
        "//jax:experimental",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "shard_alike_test",
    srcs = ["shard_alike_test.py"],
    enable_configs = [
        "tpu_v3_x4",
        "tpu_v5e_x8",
        "tpu_v4_x4",
        "tpu_v3_x4_shardy",
    ],
    deps = [
        "//jax:experimental",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "pgle_test",
    srcs = ["pgle_test.py"],
    backend_tags = {
        "gpu": ["noasan"],  # Memory leaks in NCCL, see https://github.com/NVIDIA/nccl/pull/1143
    },
    disable_configs = [
        "gpu_h100x2_tfrt",  # TODO(b/419192167): Doesn't work
    ],
    enable_backends = ["gpu"],
    tags = [
        "config-cuda-only",
        "multiaccelerator",
    ],
    deps = [
        "//jax:experimental",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "mock_gpu_test",
    srcs = ["mock_gpu_test.py"],
    enable_backends = ["gpu"],
    enable_configs = [
        "gpu_p100x2_shardy",
    ],
    tags = [
        "config-cuda-only",
    ],
    deps = [
        "//jax:experimental",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "mock_gpu_topology_test",
    srcs = ["mock_gpu_topology_test.py"],
    enable_backends = ["gpu"],
    enable_configs = [
        "gpu_h100",
        "gpu_h100_shardy",
    ],
    tags = [
        "config-cuda-only",
    ],
    deps = [
        "//jax:experimental",
    ] + py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "array_test",
    srcs = ["array_test.py"],
    backend_tags = {
        "tpu": ["requires-mem:16g"],  # Under tsan on 2x2 this test exceeds the default 12G memory limit.
    },
    enable_configs = [
        "tpu_v3_x4",
    ],
    tags = ["multiaccelerator"],
    deps = [
        "//jax:experimental",
        "//jax:internal_test_util",
    ] + py_deps([
        "numpy",
        "absl/testing",
    ]),
)

jax_multiplatform_test(
    name = "aot_test",
    srcs = ["aot_test.py"],
    tags = ["multiaccelerator"],
    deps = [
        "//jax:experimental",
    ] + py_deps([
        "numpy",
        "absl/testing",
    ]),
)

jax_multiplatform_test(
    name = "image_test",
    srcs = ["image_test.py"],
    shard_count = {
        "cpu": 10,
        "gpu": 10,
        "tpu": 8,
    },
    tags = ["noasan"],  # Linking TF causes a linker OOM.
    deps = py_deps([
        "pil",
        "tensorflow_core",
        "numpy",
        "absl/testing",
    ]),
)

jax_multiplatform_test(
    name = "infeed_test",
    srcs = ["infeed_test.py"],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "jax_jit_test",
    srcs = ["jax_jit_test.py"],
    main = "jax_jit_test.py",
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_py_test(
    name = "jax_to_ir_test",
    srcs = ["jax_to_ir_test.py"],
    deps = [
        "//jax:test_util",
        "//jax/experimental/jax2tf",
        "//jax/tools:jax_to_ir",
    ] + py_deps([
        "tensorflow_core",
        "absl/testing",
    ]),
)

jax_py_test(
    name = "jaxpr_util_test",
    srcs = ["jaxpr_util_test.py"],
    deps = [
        "//jax",
        "//jax:jaxpr_util",
        "//jax:test_util",
    ] + py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "jet_test",
    srcs = ["jet_test.py"],
    shard_count = {
        "cpu": 10,
        "gpu": 4,
    },
    deps = [
        "//jax:jet",
        "//jax:stax",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "lax_control_flow_test",
    srcs = ["lax_control_flow_test.py"],
    shard_count = {
        "cpu": 30,
        "gpu": 30,
        "tpu": 20,
    },
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "custom_root_test",
    srcs = ["custom_root_test.py"],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "custom_linear_solve_test",
    srcs = ["custom_linear_solve_test.py"],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "lax_numpy_test",
    srcs = ["lax_numpy_test.py"],
    backend_tags = {
        "tpu": ["notsan"],  # Test times out.
        "cpu": ["notsan"],  # Test times out.
    },
    shard_count = {
        "cpu": 40,
        "gpu": 40,
        "tpu": 50,
    },
    tags = [
        "noasan",  # Test times out on all backends
        "test_cpu_thunks",
    ],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "lax_numpy_operators_test",
    srcs = ["lax_numpy_operators_test.py"],
    shard_count = {
        "cpu": 30,
        "gpu": 30,
        "tpu": 40,
    },
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "lax_numpy_reducers_test",
    srcs = ["lax_numpy_reducers_test.py"],
    shard_count = {
        "cpu": 20,
        "gpu": 20,
        "tpu": 20,
    },
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "lax_numpy_indexing_test",
    srcs = ["lax_numpy_indexing_test.py"],
    shard_count = {
        "cpu": 10,
        "gpu": 10,
        "tpu": 10,
    },
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "lax_numpy_einsum_test",
    srcs = ["lax_numpy_einsum_test.py"],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "lax_numpy_ufuncs_test",
    srcs = ["lax_numpy_ufuncs_test.py"],
    shard_count = {
        "cpu": 10,
        "gpu": 5,
        "tpu": 5,
    },
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "lax_numpy_vectorize_test",
    srcs = ["lax_numpy_vectorize_test.py"],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "lax_scipy_test",
    srcs = ["lax_scipy_test.py"],
    shard_count = {
        "cpu": 30,
        "gpu": 20,
        "tpu": 8,
    },
    deps = py_deps([
        "numpy",
        "scipy",
        "absl/testing",
    ]),
)

jax_multiplatform_test(
    name = "lax_scipy_sparse_test",
    srcs = ["lax_scipy_sparse_test.py"],
    backend_tags = {
        "cpu": ["nomsan"],  # Test fails under msan because of fortran code inside scipy.
    },
    shard_count = {
        "cpu": 10,
        "gpu": 5,
        "tpu": 5,
    },
    deps = py_deps([
        "numpy",
        "scipy",
        "absl/testing",
    ]),
)

jax_multiplatform_test(
    name = "lax_scipy_special_functions_test",
    srcs = ["lax_scipy_special_functions_test.py"],
    backend_tags = {
        "cpu": [
            "nomsan",  # Times out.
            "notsan",  # Times out.
        ],
    },
    shard_count = {
        "cpu": 20,
        "gpu": 30,
        "tpu": 20,
    },
    tags = ["noasan"],  # Times out under asan.
    deps = py_deps([
        "numpy",
        "scipy",
        "absl/testing",
    ]),
)

jax_multiplatform_test(
    name = "lax_scipy_spectral_dac_test",
    srcs = ["lax_scipy_spectral_dac_test.py"],
    shard_count = {
        "cpu": 20,
        "gpu": 8,
        "tpu": 8,
    },
    deps = [
        "//jax:internal_test_util",
    ] + py_deps([
        "numpy",
        "absl/testing",
    ]),
)

jax_multiplatform_test(
    name = "lax_test",
    srcs = ["lax_test.py"],
    backend_tags = {
        "cpu": ["not_run:arm"],  # Numerical issues, including https://github.com/jax-ml/jax/issues/24787
        "tpu": ["noasan"],  # Times out.
    },
    shard_count = {
        "cpu": 40,
        "gpu": 40,
        "tpu": 40,
    },
    deps = [
        "//jax:internal_test_util",
        "//jax:lax_reference",
    ] + py_deps([
        "numpy",
        "absl/testing",
        "mpmath",
    ]),
)

jax_multiplatform_test(
    name = "lax_metal_test",
    srcs = ["lax_metal_test.py"],
    enable_backends = ["metal"],
    tags = ["notap"],
    deps = [
        "//jax:internal_test_util",
        "//jax:lax_reference",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "lax_autodiff_test",
    srcs = ["lax_autodiff_test.py"],
    shard_count = {
        "cpu": 40,
        "gpu": 30,
        "tpu": 20,
    },
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "lax_vmap_test",
    srcs = ["lax_vmap_test.py"],
    shard_count = {
        "cpu": 40,
        "gpu": 40,
        "tpu": 40,
    },
    deps = ["//jax:internal_test_util"] + py_deps([
        "numpy",
        "absl/testing",
    ]),
)

jax_multiplatform_test(
    name = "lax_vmap_op_test",
    srcs = ["lax_vmap_op_test.py"],
    shard_count = {
        "cpu": 40,
        "gpu": 40,
        "tpu": 40,
    },
    deps = ["//jax:internal_test_util"] + py_deps([
        "numpy",
        "absl/testing",
    ]),
)

jax_py_test(
    name = "lazy_loader_test",
    srcs = [
        "lazy_loader_test.py",
    ],
    deps = [
        "//jax:internal_test_util",
        "//jax:test_util",
    ] + py_deps("absl/testing"),
)

jax_py_test(
    name = "deprecation_test",
    srcs = [
        "deprecation_test.py",
    ],
    deps = [
        "//jax:internal_test_util",
        "//jax:test_util",
    ] + py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "linalg_test",
    srcs = ["linalg_test.py"],
    backend_tags = {
        "tpu": [
            "cpu:8",
            "noasan",  # Times out.
            "nomsan",  # Times out.
            "nodebug",  # Times out.
            "notsan",  # Times out.
        ],
    },
    shard_count = {
        "cpu": 40,
        "gpu": 40,
        "tpu": 40,
    },
    deps = py_deps([
        "absl/testing",
        "numpy",
        "scipy",
    ]),
)

jax_multiplatform_test(
    name = "linalg_sharding_test",
    srcs = ["linalg_sharding_test.py"],
    enable_backends = [
        "cpu",
    ],
    enable_configs = [
        "gpu_h100x2",
        "gpu_p100x2_shardy",
        "gpu_p100x2_pjrt_c_api",
    ],
    shard_count = {
        "cpu": 10,
        "gpu": 10,
    },
    tags = [
        "multiaccelerator",
    ],
    deps = py_deps([
        "absl/testing",
    ]),
)

jax_multiplatform_test(
    name = "magma_linalg_test",
    srcs = ["magma_linalg_test.py"],
    enable_backends = ["gpu"],
    deps = py_deps([
        "magma",
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "cholesky_update_test",
    srcs = ["cholesky_update_test.py"],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "metadata_test",
    srcs = ["metadata_test.py"],
    enable_backends = ["cpu"],
    deps = py_deps("absl/testing"),
)

jax_py_test(
    name = "monitoring_test",
    srcs = ["monitoring_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ] + py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "multibackend_test",
    srcs = ["multibackend_test.py"],
    enable_configs = [
        "tpu_v3_x4",
        "gpu_h100x2",
    ],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "multi_device_test",
    srcs = ["multi_device_test.py"],
    enable_backends = ["cpu"],
    deps = py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "nn_test",
    srcs = ["nn_test.py"],
    backend_tags = {
        "gpu": [
            "noasan",  # Times out under asan.
        ],
        "tpu": [
            "noasan",  # Times out under asan.
        ],
    },
    shard_count = {
        "cpu": 10,
        "tpu": 10,
        "gpu": 10,
    },
    deps = py_deps([
        "absl/testing",
        "numpy",
        "scipy",
    ]),
)

jax_multiplatform_test(
    name = "optimizers_test",
    srcs = ["optimizers_test.py"],
    deps = ["//jax:optimizers"] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "pickle_test",
    srcs = ["pickle_test.py"],
    deps = [
        "//jax:experimental",
    ] + py_deps([
        "cloudpickle",
        "numpy",
        "absl/testing",
    ]),
)

jax_multiplatform_test(
    name = "pmap_test",
    srcs = ["pmap_test.py"],
    backend_tags = {
        "tpu": [
            "noasan",  # Times out under asan.
            "requires-mem:16g",  # Under tsan on 2x2 this test exceeds the default 12G memory limit.
        ],
    },
    enable_configs = [
        "gpu_v100",
        "tpu_v3_x4",
    ],
    shard_count = {
        "cpu": 30,
        "gpu": 10,
        "tpu": 30,
    },
    tags = ["multiaccelerator"],
    deps = [
        "//jax:internal_test_util",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "polynomial_test",
    srcs = ["polynomial_test.py"],
    # No implementation of nonsymmetric Eigendecomposition.
    enable_backends = ["cpu"],
    shard_count = {
        "cpu": 5,
    },
    # This test ends up calling Fortran code that initializes some memory and
    # passes it to C code. MSan is not able to detect that the memory was
    # initialized by Fortran, and it makes the test fail. This can usually be
    # fixed by annotating the memory with `ANNOTATE_MEMORY_IS_INITIALIZED`, but
    # in this case there's not a good place to do it, see b/197635968#comment19
    # for details.
    tags = ["nomsan"],
    deps = py_deps([
        "absl/testing",
        "numpy",
        "scipy",
    ]),
)

jax_multiplatform_test(
    name = "heap_profiler_test",
    srcs = ["heap_profiler_test.py"],
    enable_backends = ["cpu"],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "profiler_test",
    srcs = ["profiler_test.py"],
    backend_tags = {
        "gpu": [
            # disable suspicious leaking in cupti/cuda,
            # TODO: remove this once b/372714955 is resolved.
            "noasan",
        ],
    },
    enable_backends = [
        "cpu",
        "gpu",
        "tpu",
    ],
    deps = [
        "//jax:profiler",
    ] + py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "pytorch_interoperability_test",
    srcs = ["pytorch_interoperability_test.py"],
    disable_configs = [
        "gpu_h100_tfrt",  # TODO(b/411472145): Re-enable once fixed.
        "gpu_h100x2_tfrt",
    ],
    enable_backends = [
        "cpu",
        "gpu",
    ],
    tags = [
        "noasan",  # TODO(b/392599624): torch fails to build.
        "nomsan",  # TODO(b/355237462): msan false-positives in torch?
        "not_build:arm",
    ],
    deps = py_deps([
        "torch",
        "absl/testing",
    ]),
)

jax_multiplatform_test(
    name = "qdwh_test",
    srcs = ["qdwh_test.py"],
    backend_tags = {
        "tpu": [
            "noasan",  # Times out
            "nomsan",  # Times out
            "notsan",  # Times out
        ],
    },
    shard_count = 8,
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "random_test",
    srcs = ["random_test.py"],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "random_lax_test",
    srcs = ["random_lax_test.py"],
    backend_tags = {
        "cpu": [
            "notsan",  # Times out
            "nomsan",  # Times out
        ],
        "tpu": [
            "optonly",
            "nomsan",  # Times out
            "notsan",  # Times out
        ],
    },
    backend_variant_args = {
        "gpu": ["--jax_num_generated_cases=40"],
    },
    shard_count = {
        "cpu": 40,
        "gpu": 50,
        "tpu": 40,
    },
    tags = ["noasan"],  # Times out
    deps = py_deps([
        "absl/testing",
        "numpy",
        "scipy",
    ]),
)

# TODO(b/199564969): remove once we always enable_custom_prng
jax_multiplatform_test(
    name = "random_test_with_custom_prng",
    srcs = ["random_test.py"],
    args = ["--jax_enable_custom_prng=true"],
    main = "random_test.py",
    deps = py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "scipy_fft_test",
    srcs = ["scipy_fft_test.py"],
    backend_tags = {
        "tpu": [
            "noasan",
            "notsan",
            "nomsan",
        ],  # Times out on TPU with asan/tsan/msan.
    },
    shard_count = 12,
    deps = py_deps([
        "absl/testing",
        "numpy",
        "scipy",
    ]),
)

jax_multiplatform_test(
    name = "scipy_interpolate_test",
    srcs = ["scipy_interpolate_test.py"],
    deps = py_deps([
        "absl/testing",
        "numpy",
        "scipy",
    ]),
)

jax_multiplatform_test(
    name = "scipy_ndimage_test",
    srcs = ["scipy_ndimage_test.py"],
    deps = py_deps([
        "absl/testing",
        "numpy",
        "scipy",
    ]),
)

jax_multiplatform_test(
    name = "scipy_optimize_test",
    srcs = ["scipy_optimize_test.py"],
    deps = py_deps([
        "absl/testing",
        "numpy",
        "scipy",
    ]),
)

jax_multiplatform_test(
    name = "scipy_signal_test",
    srcs = ["scipy_signal_test.py"],
    backend_tags = {
        "cpu": [
            "noasan",  # Test times out under asan.
        ],
        # TPU test times out under asan/msan/tsan (b/260710050)
        "tpu": [
            "noasan",
            "nomsan",
            "notsan",
            "optonly",
        ],
    },
    disable_configs = [
        "gpu_h100",  # TODO(phawkins): numerical failure on h100
    ],
    shard_count = {
        "cpu": 40,
        "gpu": 40,
        "tpu": 50,
    },
    deps = py_deps([
        "absl/testing",
        "numpy",
        "scipy",
    ]),
)

jax_multiplatform_test(
    name = "scipy_spatial_test",
    srcs = ["scipy_spatial_test.py"],
    shard_count = {
        "cpu": 4,
        "gpu": 4,
    },
    deps = py_deps([
        "absl/testing",
        "numpy",
        "scipy",
    ]),
)

jax_multiplatform_test(
    name = "scipy_stats_test",
    srcs = ["scipy_stats_test.py"],
    backend_tags = {
        "tpu": ["nomsan"],  # Times out
    },
    shard_count = {
        "cpu": 50,
        "gpu": 50,
        "tpu": 50,
    },
    tags = [
        "noasan",
        "notsan",
    ],  # Times out
    deps = py_deps([
        "absl/testing",
        "numpy",
        "scipy",
    ]),
)

jax_multiplatform_test(
    name = "sparse_test",
    srcs = ["sparse_test.py"],
    args = ["--jax_bcoo_cusparse_lowering=true"],
    backend_tags = {
        "cpu": [
            "nomsan",  # Times out
            "notsan",  # Times out
        ],
        "tpu": ["optonly"],
    },
    # Use fewer cases to prevent timeouts.
    backend_variant_args = {
        "cpu": ["--jax_num_generated_cases=40"],
        "cpu_x32": ["--jax_num_generated_cases=40"],
        "gpu": ["--jax_num_generated_cases=40"],
    },
    shard_count = {
        "cpu": 50,
        "gpu": 30,
        "tpu": 20,
    },
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],  # Test times out under asan/msan/tsan.
    deps = [
        "//jax:experimental_sparse",
        "//jax:sparse_test_util",
    ] + py_deps([
        "scipy",
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "sparse_bcoo_bcsr_test",
    srcs = ["sparse_bcoo_bcsr_test.py"],
    args = ["--jax_bcoo_cusparse_lowering=true"],
    backend_tags = {
        "cpu": [
            "nomsan",  # Times out
            "notsan",  # Times out
        ],
        "tpu": ["optonly"],
    },
    # Use fewer cases to prevent timeouts.
    backend_variant_args = {
        "cpu": ["--jax_num_generated_cases=40"],
        "cpu_x32": ["--jax_num_generated_cases=40"],
        "gpu": ["--jax_num_generated_cases=40"],
        "tpu": ["--jax_num_generated_cases=40"],
    },
    disable_configs = [
        "cpu_shardy",  # TODO(b/376475853): array values mismatch, need to fix and re-enable.
    ],
    shard_count = {
        "cpu": 50,
        "gpu": 50,
        "tpu": 50,
    },
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],  # Test times out under asan/msan/tsan.
    deps = [
        "//jax:experimental_sparse",
        "//jax:sparse_test_util",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "sparsify_test",
    srcs = ["sparsify_test.py"],
    args = ["--jax_bcoo_cusparse_lowering=true"],
    backend_tags = {
        "cpu": [
            "noasan",  # Times out under asan
            "notsan",  # Times out under tsan
        ],
        "tpu": [
            "noasan",  # Times out under asan
            "notsan",  # Times out under tsan
        ],
    },
    shard_count = {
        "cpu": 5,
        "gpu": 20,
        "tpu": 10,
    },
    deps = [
        "//jax:experimental_sparse",
        "//jax:sparse_test_util",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "stack_test",
    srcs = ["stack_test.py"],
    deps = py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "checkify_test",
    srcs = ["checkify_test.py"],
    enable_configs = ["tpu_v3_x4"],
    shard_count = {
        "gpu": 2,
        "tpu": 4,
    },
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "error_check_test",
    srcs = ["error_check_test.py"],
    deps = py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "jax_numpy_error_test",
    srcs = ["jax_numpy_error_test.py"],
    deps = py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "stax_test",
    srcs = ["stax_test.py"],
    deps = ["//jax:stax"] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "linear_search_test",
    srcs = ["third_party/scipy/line_search_test.py"],
    main = "third_party/scipy/line_search_test.py",
    deps = py_deps([
        "absl/testing",
        "scipy",
    ]),
)

jax_multiplatform_test(
    name = "blocked_sampler_test",
    srcs = ["blocked_sampler_test.py"],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_py_test(
    name = "tree_util_test",
    srcs = ["tree_util_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ] + py_deps([
        "absl/testing",
        "numpy",
        "cloudpickle",
    ]),
)

pytype_test(
    name = "typing_test",
    srcs = ["typing_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
        "//jax:typing",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ] + py_deps("absl/testing"),
)

jax_py_test(
    name = "version_test",
    srcs = ["version_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ] + py_deps("absl/testing"),
)

jax_py_test(
    name = "warnings_util_test",
    srcs = ["warnings_util_test.py"],
    deps = [
        "//jax:test_util",
    ] + py_deps("absl/testing"),
)

jax_py_test(
    name = "xla_bridge_test",
    srcs = ["xla_bridge_test.py"],
    data = ["testdata/example_pjrt_plugin_config.json"],
    deps = [
        "//jax",
        "//jax:compiler",
        "//jax:test_util",
    ] + py_deps([
        "absl/logging",
    ]),
)

jax_py_test(
    name = "lru_cache_test",
    srcs = ["lru_cache_test.py"],
    deps = [
        "//jax",
        "//jax:lru_cache",
        "//jax:test_util",
    ] + py_deps([
        "filelock",
        "absl/logging",
    ]),
)

jax_multiplatform_test(
    name = "compilation_cache_test",
    srcs = ["compilation_cache_test.py"],
    deps = [
        "//jax:compilation_cache_internal",
        "//jax:compiler",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "cache_key_test",
    srcs = ["cache_key_test.py"],
    deps = [
        "//jax:cache_key",
        "//jax:compiler",
    ] + py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "ode_test",
    srcs = ["ode_test.py"],
    shard_count = {
        "cpu": 10,
    },
    deps = ["//jax:ode"] + py_deps([
        "absl/testing",
        "numpy",
        "scipy",
    ]),
)

jax_multiplatform_test(
    name = "key_reuse_test",
    srcs = ["key_reuse_test.py"],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "roofline_test",
    srcs = ["roofline_test.py"],
    enable_backends = ["cpu"],
    deps = py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "x64_context_test",
    srcs = ["x64_context_test.py"],
    deps = [
        "//jax:experimental",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "ann_test",
    srcs = ["ann_test.py"],
    shard_count = {
        "cpu": 5,
        "gpu": 5,
        "tpu": 10,
    },
    deps = py_deps([
        "numpy",
        "absl/testing",
    ]),
)

jax_py_test(
    name = "mesh_utils_test",
    srcs = ["mesh_utils_test.py"],
    deps = [
        "//jax",
        "//jax:mesh_utils",
        "//jax:test_util",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "transfer_guard_test",
    srcs = ["transfer_guard_test.py"],
    deps = py_deps([
        "absl/testing",
        "numpy",
        "cloudpickle",
    ]),
)

jax_multiplatform_test(
    name = "garbage_collection_guard_test",
    srcs = ["garbage_collection_guard_test.py"],
    deps = py_deps("absl/testing"),
)

jax_py_test(
    name = "name_stack_test",
    srcs = ["name_stack_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ] + py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "jaxpr_effects_test",
    srcs = ["jaxpr_effects_test.py"],
    backend_tags = {
        "gpu": ["noasan"],  # Memory leaks in NCCL, see https://github.com/NVIDIA/nccl/pull/1143
    },
    enable_configs = [
        "cpu",
        "gpu_h100",
        "tpu_v2",
        "tpu_v3_x4",
        "tpu_v4_x4",
    ],
    tags = ["multiaccelerator"],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "debugging_primitives_test",
    srcs = ["debugging_primitives_test.py"],
    enable_configs = [
        "cpu",
        "gpu_h100",
        "tpu_v2",
        "tpu_v3_x4",
        "tpu_v4_x4",
        "gpu_h100_shardy",
        "tpu_v3_x4_shardy",
    ],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "python_callback_test",
    srcs = ["python_callback_test.py"],
    backend_tags = {
        "gpu": ["noasan"],  # Memory leaks in NCCL, see https://github.com/NVIDIA/nccl/pull/1143
    },
    enable_configs = [
        "tpu_v2",
        "tpu_v3_x4",
        "tpu_v4_x4",
        "tpu_v3_x4_shardy",
        "gpu_p100x2_shardy",
    ],
    tags = ["multiaccelerator"],
    deps = [
        "//jax:experimental",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "debugger_test",
    srcs = ["debugger_test.py"],
    disable_configs = [
        "cpu_shardy",  # TODO(b/364547005): enable once pure callbacks are supported.
    ],
    enable_configs = [
        "cpu",
        "gpu_h100",
        "tpu_v2",
        "tpu_v3_x4",
        "tpu_v4_x4",
    ],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "state_test",
    srcs = ["state_test.py"],
    # Use fewer cases to prevent timeouts.
    args = [
        "--jax_num_generated_cases=5",
    ],
    backend_variant_args = {
        "tpu_pjrt_c_api": ["--jax_num_generated_cases=1"],
    },
    enable_configs = [
        "gpu_h100",
        "cpu",
    ],
    shard_count = {
        "cpu": 2,
        "gpu": 2,
        "tpu": 2,
    },
    deps = py_deps([
        "absl/testing",
        "numpy",
        "hypothesis",
    ]),
)

jax_multiplatform_test(
    name = "mutable_array_test",
    srcs = ["mutable_array_test.py"],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "for_loop_test",
    srcs = ["for_loop_test.py"],
    shard_count = {
        "tpu": 20,
    },
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "ragged_collective_test",
    srcs = ["ragged_collective_test.py"],
    disable_configs = [
        "tpu_pjrt_c_api",
    ],
    enable_backends = [
        "gpu",
        "tpu",
    ],
    enable_configs = [
        "gpu_p100x2_shardy",
    ],
    tags = [
        "multiaccelerator",
    ],
    deps = [
        "//jax:experimental",
    ] + py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "shard_map_test",
    srcs = ["shard_map_test.py"],
    disable_configs = [
        "gpu_h100x2_tfrt",  # TODO(b/419192167): Doesn't work
    ],
    enable_configs = [
        "gpu_p100x2_shardy",
        "tpu_v3_x4_shardy",
    ],
    shard_count = {
        "cpu": 50,
        "gpu": 20,
        "tpu": 50,
    },
    tags = [
        "multiaccelerator",
        "noasan",
        "nomsan",
        "notsan",
    ],  # Times out under *SAN.
    deps = [
        "//jax:experimental",
        "//jax:tree_util",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "clear_backends_test",
    srcs = ["clear_backends_test.py"],
    deps = py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "attrs_test",
    srcs = ["attrs_test.py"],
    deps = [
        "//jax:experimental",
    ] + py_deps([
        "numpy",
        "absl/testing",
    ]),
)

jax_multiplatform_test(
    name = "hijax_test",
    srcs = ["hijax_test.py"],
    deps = [
        "//jax:experimental",
    ] + py_deps([
        "numpy",
        "absl/testing",
    ]),
)

jax_multiplatform_test(
    name = "colocated_python_test",
    srcs = ["colocated_python_test.py"],
    deps = [
        "//jax:experimental_colocated_python",
        "//jax/extend:ifrt_programs",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "experimental_rnn_test",
    srcs = ["experimental_rnn_test.py"],
    disable_configs = [
        "gpu_a100",  # Numerical precision problems.
    ],
    enable_backends = ["gpu"],
    shard_count = 15,
    deps = [
        "//jax:rnn",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_py_test(
    name = "mosaic_test",
    srcs = ["mosaic_test.py"],
    deps = [
        "//jax",
        "//jax:mosaic",
        "//jax:test_util",
    ] + py_deps("absl/testing"),
)

jax_py_test(
    name = "source_info_test",
    srcs = ["source_info_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ] + py_deps("absl/testing"),
)

jax_py_test(
    name = "package_structure_test",
    srcs = ["package_structure_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ] + py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "logging_test",
    srcs = ["logging_test.py"],
    deps = py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "export_test",
    srcs = ["export_test.py"],
    enable_configs = [
        "cpu_shardy",
        "gpu_p100x2_shardy",
        "tpu_v3_x4_shardy",
        "tpu_v3_x4",
    ],
    tags = [],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "shape_poly_test",
    srcs = ["shape_poly_test.py"],
    disable_configs = [
        "gpu_a100",  # TODO(b/269593297): matmul precision issues
    ],
    enable_configs = [
        "cpu",
        "cpu_x32",
    ],
    shard_count = {
        "cpu": 30,
        "gpu": 20,
        "tpu": 25,
    },
    tags = [
        "noasan",  # Times out
        "nomsan",  # Times out
        "notsan",  # Times out
    ],
    deps = [
        "//jax:internal_test_harnesses",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "export_harnesses_multi_platform_test",
    srcs = ["export_harnesses_multi_platform_test.py"],
    disable_configs = [
        "gpu_a100",  # TODO(b/269593297): matmul precision issues
        "gpu_h100",  # Scarce resources.
        "cpu_shardy",  # TODO(b/355263220): enable once export is supported.
    ],
    shard_count = {
        "cpu": 40,
        "gpu": 30,
        "tpu": 20,
    },
    tags = [
        "noasan",  # Times out
        "nodebug",  # Times out.
    ],
    deps = [
        "//jax:internal_test_harnesses",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "export_back_compat_test",
    srcs = ["export_back_compat_test.py"],
    # TODO(b/415285434): enable once we have backwards compatibility support with GSPMD checkpoints.
    # enable_configs = [
    #     "tpu_v3_x4_shardy",
    # ],
    tags = [],
    deps = [
        "//jax:internal_export_back_compat_test_data",
        "//jax:internal_export_back_compat_test_util",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "fused_attention_stablehlo_test",
    srcs = ["fused_attention_stablehlo_test.py"],
    enable_backends = ["gpu"],
    tags = ["multiaccelerator"],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "xla_metadata_test",
    srcs = ["xla_metadata_test.py"],
    deps = ["//jax:experimental"] + py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "unary_ops_accuracy_test",
    srcs = ["unary_ops_accuracy_test.py"],
    disable_configs = [
        "tpu_pjrt_c_api",
    ],
    enable_backends = [
        "tpu",
    ],
    deps = [
        "//jax:experimental",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_py_test(
    name = "pretty_printer_test",
    srcs = ["pretty_printer_test.py"],
    deps = [
        "//jax",
        "//jax:test_util",
    ] + py_deps("absl/testing"),
)

jax_py_test(
    name = "source_mapper_test",
    srcs = ["source_mapper_test.py"],
    deps = [
        "//jax",
        "//jax:source_mapper",
        "//jax:test_util",
    ] + py_deps("absl/testing"),
)

jax_py_test(
    name = "sourcemap_test",
    srcs = ["sourcemap_test.py"],
    deps = [
        "//jax",
        "//jax:sourcemap",
        "//jax:test_util",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "string_array_test",
    srcs = ["string_array_test.py"],
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "cudnn_fusion_test",
    srcs = ["cudnn_fusion_test.py"],
    enable_backends = [],
    enable_configs = [
        "gpu_a100",
        "gpu_h100",
    ],
    tags = ["multiaccelerator"],
    deps = py_deps("absl/testing"),
)

jax_multiplatform_test(
    name = "scaled_matmul_stablehlo_test",
    srcs = ["scaled_matmul_stablehlo_test.py"],
    enable_backends = ["gpu"],
    shard_count = {
        "gpu": 4,
    },
    deps = py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_py_test(
    name = "custom_partitioning_sharding_rule_test",
    srcs = ["custom_partitioning_sharding_rule_test.py"],
    deps = [
        "//jax",
        "//jax:experimental",
        "//jax:test_util",
    ] + py_deps("absl/testing"),
)

exports_files(
    [
        "api_test.py",
        "custom_api_test.py",
        "array_test.py",
        "cache_key_test.py",
        "colocated_python_test.py",
        "compilation_cache_test.py",
        "memories_test.py",
        "pmap_test.py",
        "pjit_test.py",
        "python_callback_test.py",
        "shard_map_test.py",
        "transfer_guard_test.py",
        "layout_test.py",
        "string_array_test.py",
    ],
    visibility = jax_test_file_visibility,
)

# This filegroup specifies the set of tests known to Bazel, used for a test that
# verifies every test has a Bazel test rule.
# If a test isn't meant to be tested with Bazel, add it to the exclude list.
filegroup(
    name = "all_tests",
    srcs = glob(
        include = [
            "*_test.py",
            "third_party/*/*_test.py",
        ],
        exclude = [],
    ) + ["BUILD"],
    visibility = [
        "//jax:internal",
    ],
)
