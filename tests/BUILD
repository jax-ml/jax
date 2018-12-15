# Copyright 2018 Google LLC
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

licenses(["notice"])  # Apache 2

load(":build_defs.bzl", "jax_test")

config_setting(
    name = "libjax_depend",
    values = {"define": "libjax_depend=true"},
)

jax_test(
    name = "core_test",
    srcs = ["core_test.py"],
    shard_count = {
        "cpu": 5,
    },
)

jax_test(
    name = "lax_test",
    srcs = ["lax_test.py"],
    disable = ["tpu"],
    shard_count = {
        "cpu": 40,
        "gpu": 40,
    },
)

jax_test(
    name = "lax_numpy_test",
    srcs = ["lax_numpy_test.py"],
    disable = ["tpu"],
    shard_count = {
        "cpu": 40,
        "gpu": 40,
    },
)

jax_test(
    name = "lax_numpy_indexing_test",
    srcs = ["lax_numpy_indexing_test.py"],
    shard_count = {
        "cpu": 10,
        "gpu": 4,
    },
)

jax_test(
    name = "linalg_test",
    srcs = ["linalg_test.py"],
    shard_count = {
        "cpu": 2,
    },
)

jax_test(
    name = "lax_scipy_test",
    srcs = ["lax_scipy_test.py"],
    shard_count = {
        "cpu": 10,
        "gpu": 4,
    },
)

jax_test(
    name = "random_test",
    srcs = ["random_test.py"],
)

jax_test(
    name = "api_test",
    srcs = ["api_test.py"],
)

jax_test(
    name = "batching_test",
    srcs = ["batching_test.py"],
)

jax_test(
    name = "stax_test",
    srcs = ["stax_test.py"],
)

jax_test(
    name = "minmax_test",
    srcs = ["minmax_test.py"],
)

jax_test(
    name = "lapax_test",
    srcs = ["lapax_test.py"],
)

jax_test(
    name = "generated_fun_test",
    srcs = ["generated_fun_test.py"],
)
