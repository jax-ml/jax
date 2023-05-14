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

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)

gentbl_cc_library(
    name = "jax_pass_inc_gen",
    strip_include_prefix = ".",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=Jax",
            ],
            "jax_passes.h.inc",
        ),
        (
            [
                "-gen-pass-capi-header",
                "--prefix=Jax",
            ],
            "jax_passes.capi.h.inc",
        ),
        (
            [
                "-gen-pass-capi-impl",
                "--prefix=Jax",
            ],
            "jax_passes.capi.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "jax_passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "jax_passes",
    srcs = [
        "jax_passes.h.inc",
        "jax_strip_locations_pass.cc",
    ],
    hdrs = ["jax_passes.h"],
    deps = [
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "capi_jax_passes",
    srcs = [
        "capi_jax_passes.cc",
    ],
    hdrs = [
        "capi_jax_passes.h",
    ],
    deps = [
        ":jax_pass_inc_gen",
        ":jax_passes",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = True,
)

cc_library(
    name = "capi_jax_passes_headers",
    hdrs = [
        "capi_jax_passes.h",
        "jax_passes.capi.h.inc",
    ],
    deps = [
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
)
