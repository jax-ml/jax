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

# Public API of Pallas, a JAX extension for custom kernels.

load(
    "//jaxlib:jax.bzl",
    "pytype_strict_library",
)

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:internal"],
)

pytype_strict_library(
    name = "pallas",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//jax",
        "//jax/_src:core",
        "//jax/_src:lax",
        "//jax/_src:source_info_util",
        "//jax/_src:state_types",
        "//jax/_src/pallas",
    ],
)

pytype_strict_library(
    name = "tpu",
    srcs = ["tpu.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":pallas",  # build_cleaner: keep
        "//jax/_src/pallas",
        "//jax/_src/pallas/mosaic:core",
        "//jax/_src/pallas/mosaic:helpers",
        "//jax/_src/pallas/mosaic:pallas_call_registration",  # build_cleaner: keep
        "//jax/_src/pallas/mosaic:pipeline",
        "//jax/_src/pallas/mosaic:primitives",
        "//jax/_src/pallas/mosaic:random",
        "//jax/_src/pallas/mosaic:tpu_info",
        "//jax/_src/pallas/mosaic/interpret:interpret_pallas_call",
        "//jax/_src/pallas/mosaic/interpret:params",
    ],
)
