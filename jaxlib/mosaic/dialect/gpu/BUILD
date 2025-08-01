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
    "@llvm-project//mlir:tblgen.bzl",
    "gentbl_cc_library",
    "gentbl_filegroup",
    "td_library",
)

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax/experimental:mosaic_gpu_users"],
)

td_library(
    name = "mosaic_gpu_td_files",
    srcs = ["mosaic_gpu.td"],
    includes = ["."],
    deps = [
        "@llvm-project//mlir:BasicPtxBuilderIntTdFiles",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:LLVMOpsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "mosaic_gpu_inc_gen",
    tbl_outs = {
        "mosaic_gpu_dialect.h.inc": [
            "-gen-dialect-decls",
            "-dialect=mosaic_gpu",
        ],
        "mosaic_gpu_dialect.cc.inc": [
            "-gen-dialect-defs",
            "-dialect=mosaic_gpu",
        ],
        "mosaic_gpu_ops.h.inc": ["-gen-op-decls"],
        "mosaic_gpu_ops.cc.inc": ["-gen-op-defs"],
        "mosaic_gpu_types.h.inc": [
            "-gen-typedef-decls",
            "--typedefs-dialect=mosaic_gpu",
        ],
        "mosaic_gpu_types.cc.inc": [
            "-gen-typedef-defs",
            "--typedefs-dialect=mosaic_gpu",
        ],
        "mosaic_gpu_enums.h.inc": ["-gen-enum-decls"],
        "mosaic_gpu_enums.cc.inc": ["-gen-enum-defs"],
        "mosaic_gpu_attrdefs.h.inc": [
            "-gen-attrdef-decls",
            "--attrdefs-dialect=mosaic_gpu",
        ],
        "mosaic_gpu_attrdefs.cc.inc": [
            "-gen-attrdef-defs",
            "--attrdefs-dialect=mosaic_gpu",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "mosaic_gpu.td",
    deps = [
        ":mosaic_gpu_td_files",
    ],
)

cc_library(
    name = "mosaic_gpu",
    srcs = ["mosaic_gpu.cc"],
    hdrs = ["mosaic_gpu.h"],
    deps = [
        ":mosaic_gpu_inc_gen",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefUtils",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:Support",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_test(
    name = "mosaic_gpu_test",
    srcs = ["mosaic_gpu_test.cc"],
    deps = [
        ":mosaic_gpu",
        "//testing/base/public:gunit_main",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BufferizationInterfaces",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:Support",
        "@xla//xla/tsl/platform:errors",
    ],
)

gentbl_filegroup(
    name = "mosaic_gpu_python_gen_raw",
    tbl_outs = [
        (
            [
                "-gen-python-enum-bindings",
                "-bind-dialect=mosaic_gpu",
            ],
            "_mosaic_gpu_gen_enums_raw.py",
        ),
        (
            [
                "-gen-python-op-bindings",
                "-bind-dialect=mosaic_gpu",
            ],
            "_mosaic_gpu_gen_ops_raw.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = ":mosaic_gpu.td",
    deps = [
        ":mosaic_gpu_td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

genrule(
    name = "mosaic_gpu_python_gen_enums",
    srcs = ["_mosaic_gpu_gen_enums_raw.py"],
    outs = ["_mosaic_gpu_gen_enums.py"],
    cmd = """
      cat $(location _mosaic_gpu_gen_enums_raw.py) | \
      sed -e 's/^from \\.\\.ir/from jaxlib\\.mlir\\.ir/g; s/^from \\./from jaxlib\\.mlir\\.dialects\\./g' > $@""",
)

genrule(
    name = "mosaic_gpu_python_gen_ops",
    srcs = ["_mosaic_gpu_gen_ops_raw.py"],
    outs = ["_mosaic_gpu_gen_ops.py"],
    cmd = "cat $(location _mosaic_gpu_gen_ops_raw.py) | sed -e 's/^from \\./from jaxlib\\.mlir\\.dialects\\./g' > $@",
)

DIALECT_CAPI_SOURCES = [
    ":integrations/c/attributes.cc",
    ":integrations/c/gpu_dialect.cc",
]

DIALECT_CAPI_HEADERS = [
    ":integrations/c/attributes.h",
    ":integrations/c/gpu_dialect.h",
]

cc_library(
    name = "gpu_dialect_capi",
    srcs = DIALECT_CAPI_SOURCES,
    hdrs = DIALECT_CAPI_HEADERS,
    deps = [
        ":mosaic_gpu",
        ":mosaic_gpu_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

# Header-only target, used when using the C API from a separate shared library.
cc_library(
    name = "gpu_dialect_capi_headers",
    hdrs = DIALECT_CAPI_HEADERS,
    deps = [
        ":mosaic_gpu_inc_gen",
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
)

# Alwayslink target, used when exporting the C API from a shared library.
cc_library(
    name = "gpu_dialect_capi_objects",
    srcs = DIALECT_CAPI_SOURCES,
    hdrs = DIALECT_CAPI_HEADERS,
    deps = [
        ":mosaic_gpu",
        ":mosaic_gpu_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIRObjects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = True,
)
