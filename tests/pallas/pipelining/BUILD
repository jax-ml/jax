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
    "jax_generate_backend_suites",
    "jax_multiplatform_test",
    "py_deps",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:private"],
)

jax_generate_backend_suites(backends = ["cpu"])

jax_multiplatform_test(
    name = "schedule_api_test",
    srcs = ["schedule_api_test.py"],
    enable_backends = ["cpu"],
    enable_configs = ["cpu"],
    deps = [
        "//jax/_src:core",
        "//jax/_src:state_types",
        "//jax/_src/pallas/pipelining:pipeline_test_util",
        "//jax/_src/pallas/pipelining:schedule_api",
    ] + py_deps([
        "absl/testing",
        "numpy",
    ]),
)

jax_multiplatform_test(
    name = "schedulers_test",
    srcs = ["schedulers_test.py"],
    enable_backends = ["cpu"],
    enable_configs = ["cpu"],
    deps = [
        "//jax/_src/pallas/pipelining:internal",
        "//jax/_src/pallas/pipelining:pipeline_test_util",
        "//jax/_src/pallas/pipelining:schedulers",
    ] + py_deps([
        "absl/testing",
    ]),
)
