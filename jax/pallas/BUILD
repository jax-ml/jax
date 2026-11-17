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

# Public API of Pallas, a JAX kernel language.

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
        "//jax/_src:core",
        "//jax/_src:lax",
        "//jax/_src:state_types",
        "//jax/_src/pallas",
    ],
)

pytype_strict_library(
    name = "fuser",
    srcs = ["fuser.py"],
    visibility = ["//jax/experimental:pallas_fuser_users"],
    deps = [
        "//jax/_src/pallas/fuser:block_spec",
        "//jax/_src/pallas/fuser:custom_evaluate",
        "//jax/_src/pallas/fuser:custom_fusion",
        "//jax/_src/pallas/fuser:fuser_utils",
        "//jax/_src/pallas/fuser:fusible",
        "//jax/_src/pallas/fuser:fusion",
        "//jax/_src/pallas/fuser:jaxpr_fusion",
    ],
)

pytype_strict_library(
    name = "mosaic_gpu",
    srcs = ["mosaic_gpu.py"],
    visibility = ["//jax/experimental:mosaic_gpu_users"],
    deps = [
        "//jax/_src:lax",
        "//jax/_src:state_types",
        "//jax/_src/pallas/mosaic_gpu:core",
        "//jax/_src/pallas/mosaic_gpu:helpers",
        "//jax/_src/pallas/mosaic_gpu:layout_rules",  # build_cleaner: keep
        "//jax/_src/pallas/mosaic_gpu:pallas_call_registration",  # build_cleaner: keep
        "//jax/_src/pallas/mosaic_gpu:pipeline",
        "//jax/_src/pallas/mosaic_gpu:primitives",
        "//jax/_src/pallas/mosaic_gpu:torch",
        "//jax/_src/pallas/mosaic_gpu/interpret:interpret_pallas_call",  # build_cleaner: keep
        "//jax/_src/pallas/mosaic_gpu/interpret:params",
        "//jax/experimental:mosaic_gpu",
    ],
)

pytype_strict_library(
    name = "tpu",
    srcs = ["tpu.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//jax/_src/pallas",
        "//jax/_src/pallas/mosaic:core",
        "//jax/_src/pallas/mosaic:helpers",
        "//jax/_src/pallas/mosaic:lowering",
        "//jax/_src/pallas/mosaic:pallas_call_registration",  # build_cleaner: keep
        "//jax/_src/pallas/mosaic:pipeline",
        "//jax/_src/pallas/mosaic:prefetch",
        "//jax/_src/pallas/mosaic:primitives",
        "//jax/_src/pallas/mosaic:random",
        "//jax/_src/pallas/mosaic:tpu_info",
        "//jax/_src/pallas/mosaic/interpret:interpret_pallas_call",
        "//jax/_src/pallas/mosaic/interpret:params",
    ],
)

pytype_strict_library(
    name = "tpu_sc",
    srcs = ["tpu_sc.py"],
    visibility = ["//jax/experimental:pallas_sc_users"],
    deps = [
        "//jax/_src/pallas/mosaic:sc_core",
        "//jax/_src/pallas/mosaic:sc_primitives",
    ],
)
