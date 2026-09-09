# Copyright 2026 The JAX Authors.
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
# ==============================================================================

load("//jaxlib:jax.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:__subpackages__"],
)

sh_binary(
    name = "process_test_results",
    srcs = ["process_test_results.sh"],
    data = [":xml2json.py"],
)

py_library(
    name = "xml2json",
    srcs = ["xml2json.py"],
)

py_test(
    name = "xml2json_test",
    srcs = ["xml2json_test.py"],
    deps = [":xml2json"],
)
