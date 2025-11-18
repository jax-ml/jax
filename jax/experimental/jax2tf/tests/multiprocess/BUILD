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

load("//jaxlib:jax.bzl", "jax_multiprocess_generate_backend_suites", "jax_multiprocess_test", "py_deps")

licenses(["notice"])

package(default_applicable_licenses = [])

jax_multiprocess_generate_backend_suites()

jax_multiprocess_test(
    name = "jax2tf_multiprocess_test",
    srcs = ["jax2tf_multiprocess_test.py"],
    main = "jax2tf_multiprocess_test.py",
    tags = ["jax2tf"],
    deps = [
        "//jax/_src:test_multiprocess",
        "//jax/experimental/jax2tf",
        "//jax/experimental/jax2tf/tests:tf_test_util",
    ] + py_deps("tensorflow"),
)
