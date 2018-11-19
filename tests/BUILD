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

load("//third_party/py/jax/tests:build_defs.bzl", "jax_test")

jax_test(
    name = "core_test",
    srcs = ["tests/core_test.py"],
    shard_count = {
        "cpu": 5,
    },
)

jax_test(
    name = "lax_test",
    srcs = ["tests/lax_test.py"],
    shard_count = {
        "cpu": 40,
        "gpu": 20,
    },
)

jax_test(
    name = "lax_numpy_test",
    srcs = ["tests/lax_numpy_test.py"],
    shard_count = {
        "cpu": 40,
        "gpu": 20,
    },
)

jax_test(
    name = "lax_numpy_indexing_test",
    srcs = ["tests/lax_numpy_indexing_test.py"],
    shard_count = {
        "cpu": 10,
        "gpu": 2,
    },
)

jax_test(
    name = "lax_scipy_test",
    srcs = ["tests/lax_scipy_test.py"],
    shard_count = {
        "cpu": 10,
        "gpu": 2,
    },
)

jax_test(
    name = "random_test",
    srcs = ["tests/random_test.py"],
)

jax_test(
    name = "api_test",
    srcs = ["tests/api_test.py"],
)

jax_test(
    name = "batching_test",
    srcs = ["tests/batching_test.py"],
)

jax_test(
    name = "stax_test",
    srcs = ["tests/stax_test.py"],
    deps = [":stax"],
)

jax_test(
    name = "minmax_test",
    srcs = ["tests/minmax_test.py"],
    deps = [":minmax"],
)

jax_test(
    name = "lapax_test",
    srcs = ["tests/lapax_test.py"],
    deps = [":lapax"],
)
