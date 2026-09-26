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

load(
    "@llvm-project//mlir:tblgen.bzl",
    "gentbl_cc_library",
    "gentbl_filegroup",
    "td_library",
)
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:__subpackages__"],
)

td_library(
    name = "thunky_td_files",
    srcs = [
        "thunky_dialect.td",
        "thunky_ops.td",
    ],
    includes = ["."],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "thunky_inc_gen",
    tbl_outs = {
        "thunky_dialect.h.inc": [
            "-gen-dialect-decls",
            "-dialect=thunky",
        ],
        "thunky_dialect.cc.inc": [
            "-gen-dialect-defs",
            "-dialect=thunky",
        ],
        "thunky_ops.h.inc": ["-gen-op-decls"],
        "thunky_ops.cc.inc": ["-gen-op-defs"],
        "thunky_types.h.inc": [
            "-gen-typedef-decls",
            "--typedefs-dialect=thunky",
        ],
        "thunky_types.cc.inc": [
            "-gen-typedef-defs",
            "--typedefs-dialect=thunky",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "thunky_ops.td",
    deps = [
        ":thunky_td_files",
    ],
)

gentbl_filegroup(
    name = "thunky_python_gen_raw",
    tbl_outs = [
        (
            [
                "-gen-python-op-bindings",
                "-bind-dialect=thunky",
            ],
            "_thunky_ops_gen_raw.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "thunky_ops.td",
    deps = [
        ":thunky_td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

genrule(
    name = "thunky_python_gen_ops",
    srcs = ["_thunky_ops_gen_raw.py"],
    outs = ["_thunky_ops_gen.py"],
    cmd = "cat $(execpath _thunky_ops_gen_raw.py) | sed -e 's/^from \\./from jaxlib\\.mlir\\.dialects\\./g' > $@",
)

cc_library(
    name = "thunky_dialect",
    srcs = ["thunky_dialect.cc"],
    hdrs = ["thunky_dialect.h"],
    deps = [
        ":thunky_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "thunky_dialect_capi",
    srcs = ["capi.cc"],
    hdrs = ["capi.h"],
    deps = [
        ":thunky_dialect",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:IR",
    ],
)
