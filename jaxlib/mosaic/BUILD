load("@llvm-project//mlir:tblgen.bzl", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Copyright 2023 The JAX Authors.
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
load("//jaxlib:jax.bzl", "py_library")

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = [
        "//jax/experimental:mosaic_users",
    ],
)

py_library(
    name = "mosaic",
    deps = [
        "//jaxlib/mosaic/python:gpu_dialect",
        "//jaxlib/mosaic/python:tpu_dialect",
    ],
)

################################################################################
# TPU dialect

cc_library(
    name = "tpu_dialect",
    hdrs = [
        "dialect/tpu/tpu_dialect.h",
        "dialect/tpu/util.h",
    ],
    # compatible with libtpu
    deps = [
        "@xla//xla/mosaic:tpu_dialect",
    ],
)

cc_library(
    name = "tpu_serde_pass",
    hdrs = ["dialect/tpu/transforms/serde.h"],
    # compatible with libtpu
    deps = [
        "@xla//xla/mosaic:tpu_serde_pass",
    ],
)

td_library(
    name = "tpu_td_files",
    srcs = [
        "dialect/tpu/tpu.td",
        "dialect/tpu/tpu_enums.td",
        "dialect/tpu/tpu_ops.td",
        "dialect/tpu/tpu_types.td",
    ],
    # compatible with libtpu
    includes = ["/external/xla"],  # needed for OSS TODO(@willfroom) move to xla.
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@xla//xla/mosaic:tpu_td_files",
    ],
)

# C API targets

TPU_CAPI_HEADERS = [
    "dialect/tpu/integrations/c/tpu_dialect.h",
]

cc_library(
    name = "tpu_dialect_capi",
    hdrs = TPU_CAPI_HEADERS,
    deps = [
        "@xla//xla/mosaic:tpu_dialect_capi",
    ],
)

# Header-only target, used when using the C API from a separate shared library.
cc_library(
    name = "tpu_dialect_capi_headers",
    hdrs = TPU_CAPI_HEADERS,
    deps = [
        "@xla//xla/mosaic:tpu_dialect_capi_headers",
    ],
)

# Alwayslink target, used when exporting the C API from a shared library.
cc_library(
    name = "tpu_dialect_capi_objects",
    hdrs = TPU_CAPI_HEADERS,
    deps = [
        "@xla//xla/mosaic:tpu_dialect_capi_objects",
    ],
    alwayslink = True,
)

alias(
    name = "tiled_layout_test",
    actual = "@xla//xla/mosaic:tiled_layout_test",
)

alias(
    name = "divisibility_check_test",
    actual = "@xla//xla/mosaic:divisibility_check_test",
)

alias(
    name = "tpu_ops_verification_test",
    actual = "@xla//xla/mosaic:tpu_ops_verification_test",
)

cc_library(
    name = "pass_boilerplate",
    hdrs = ["pass_boilerplate.h"],
    # compatible with libtpu
    deps = [
        "@xla//xla/mosaic:pass_boilerplate",
    ],
)

cc_library(
    name = "serde",
    hdrs = ["serde.h"],
    # compatible with libtpu
    deps = [
        "@xla//xla/mosaic:serde",
    ],
)

cc_library(
    name = "stringify_util",
    hdrs = ["dialect/tpu/stringify_util.h"],
    # compatible with libtpu
    deps = [
        "@xla//xla/mosaic:stringify_util",
    ],
)
