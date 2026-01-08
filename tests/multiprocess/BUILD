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
    "jax_multiprocess_generate_backend_suites",
    "jax_multiprocess_test",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
)

jax_multiprocess_generate_backend_suites()

jax_multiprocess_test(
    name = "all_reduce_test",
    srcs = ["all_reduce_test.py"],
    backend_tags = {
        "gpu": ["noasan"],  # Memory leaks in NCCL, see https://github.com/NVIDIA/nccl/pull/1143
    },
    enable_configs = ["cpu_megascale"],
    main = "all_reduce_test.py",
    deps = ["//jax/_src:test_multiprocess"],
)

jax_multiprocess_test(
    name = "all_gather_test",
    srcs = ["all_gather_test.py"],
    backend_tags = {
        "gpu": ["noasan"],  # Memory leaks in NCCL, see https://github.com/NVIDIA/nccl/pull/1143
    },
    enable_configs = ["cpu_megascale"],
    main = "all_gather_test.py",
    deps = ["//jax/_src:test_multiprocess"],
)

jax_multiprocess_test(
    name = "all_to_all_test",
    srcs = ["all_to_all_test.py"],
    backend_tags = {
        "gpu": ["noasan"],  # Memory leaks in NCCL, see https://github.com/NVIDIA/nccl/pull/1143
    },
    enable_configs = ["cpu_megascale"],
    main = "all_to_all_test.py",
    deps = ["//jax/_src:test_multiprocess"],
)

jax_multiprocess_test(
    name = "axis_index_test",
    srcs = ["axis_index_test.py"],
    main = "axis_index_test.py",
    deps = [
        "//jax/_src:test_multiprocess",
    ],
)

jax_multiprocess_test(
    name = "array_test",
    srcs = ["array_test.py"],
    main = "array_test.py",
    deps = [
        "//jax/_src:test_multiprocess",
    ],
)

jax_multiprocess_test(
    name = "colocated_python_test",
    srcs = ["colocated_python_test.py"],
    disable_configs = [
        # This config has two cores per chip, and JAX distributed does not get
        # the correct number of logical devices per host.
        "tpu_v3_x4",
    ],
    main = "colocated_python_test.py",
    deps = [
        "//jax/_src:test_multiprocess",
        "//jax/experimental:colocated_python",
    ],
)

jax_multiprocess_test(
    name = "device_id_test",
    srcs = ["device_id_test.py"],
    main = "device_id_test.py",
    deps = [
        "//jax/_src:test_multiprocess",
    ],
)

jax_multiprocess_test(
    name = "host_callback_test",
    srcs = ["host_callback_test.py"],
    main = "host_callback_test.py",
    deps = [
        "//jax/_src:test_multiprocess",
        "//jax/experimental",
        "//jax/experimental:multihost_utils",
    ],
)

jax_multiprocess_test(
    name = "key_value_store_test",
    srcs = ["key_value_store_test.py"],
    main = "key_value_store_test.py",
    deps = [
        "//jax/_src:test_multiprocess",
    ],
)

jax_multiprocess_test(
    name = "multihost_utils_test",
    srcs = ["multihost_utils_test.py"],
    backend_tags = {
        "gpu": ["noasan"],  # Memory leaks in NCCL, see https://github.com/NVIDIA/nccl/pull/1143
    },
    enable_backends = if_oss(
        [
            "cpu",
            "tpu",
        ],
        None,
    ),  # b/453057226
    main = "multihost_utils_test.py",
    deps = [
        "//jax/_src:test_multiprocess",
    ],
)

jax_multiprocess_test(
    name = "pjit_test",
    srcs = ["pjit_test.py"],
    main = "pjit_test.py",
    deps = [
        "//jax/_src:test_multiprocess",
    ],
)

jax_multiprocess_test(
    name = "pgle_test",
    srcs = ["pgle_test.py"],
    backend_tags = {
        "gpu": ["noasan"],  # Memory leaks in NCCL.
    },
    enable_backends = ["gpu"],
    main = "pgle_test.py",
    deps = [
        "//jax/_src:test_multiprocess",
    ],
)

jax_multiprocess_test(
    name = "pmap_test",
    srcs = ["pmap_test.py"],
    backend_tags = {
        "gpu": ["noasan"],  # Memory leaks in NCCL, see https://github.com/NVIDIA/nccl/pull/1143
    },
    main = "pmap_test.py",
    deps = [
        "//jax/_src:test_multiprocess",
    ],
)

jax_multiprocess_test(
    name = "thread_guard_test",
    srcs = ["thread_guard_test.py"],
    main = "thread_guard_test.py",
    deps = [
        "//jax/_src:test_multiprocess",
    ],
)

jax_multiprocess_test(
    name = "tpu_device_test",
    srcs = ["tpu_device_test.py"],
    enable_backends = ["tpu"],
    main = "tpu_device_test.py",
    deps = [
        "//jax/_src:test_multiprocess",
    ],
)

jax_multiprocess_test(
    name = "wait_barrier_test",
    srcs = ["wait_barrier_test.py"],
    enable_backends = [
        "cpu",
        "gpu",
    ],
    main = "wait_barrier_test.py",
    deps = [
        "//jax/_src:test_multiprocess",
    ],
)
