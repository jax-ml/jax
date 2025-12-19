# Copyright 2024 The JAX Authors.
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
    "py_deps",
)

licenses(["notice"])

package(
    default_applicable_licenses = ["//jax:license"],
    default_visibility = ["//visibility:private"],
)

jax_generate_backend_suites()

jax_multiplatform_test(
    name = "matmul_bench",
    srcs = ["matmul_bench.py"],
    enable_backends = [],
    enable_configs = ["gpu_h100"],
    tags = ["notap"],
    deps = [
        "//jax/experimental:mosaic_gpu",
        "//jax/experimental/mosaic/gpu/examples:matmul",
        "//third_party/py/google_benchmark",
    ] + py_deps("absl/testing") + py_deps("numpy"),
)
