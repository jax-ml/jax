load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_python//python:defs.bzl", "py_library")

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
load("//jaxlib:jax.bzl", "mosaic_extension_deps")

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
    srcs = [
        "dialect/tpu/array_util.cc",
        "dialect/tpu/layout.cc",
        "dialect/tpu/tpu_dialect.cc",
        "dialect/tpu/tpu_ops.cc",
        "dialect/tpu/util.cc",
        "dialect/tpu/vreg_util.cc",
        ":extension_srcs",
    ] + glob([
        "dialect/tpu/transforms/*.cc",
    ]),
    hdrs = [
        "dialect/tpu/array_util.h",
        "dialect/tpu/layout.h",
        "dialect/tpu/tpu_dialect.h",
        "dialect/tpu/util.h",
        "dialect/tpu/vreg_util.h",
    ] + glob([
        "dialect/tpu/transforms/*.h",
    ]),
    # compatible with libtpu
    deps = [
        ":pass_boilerplate",
        ":serde",
        ":tpu_inc_gen",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:CommonFolders",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:VectorTransforms",
        "@tsl//tsl/platform:statusor",
        "@xla//xla:array",
        "@xla//xla:shape_util",
        "@xla//xla:util",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ] + mosaic_extension_deps,
)

gentbl_cc_library(
    name = "tpu_inc_gen",
    # compatible with libtpu
    tbl_outs = {
        "dialect/tpu/tpu_ops.h.inc": ["-gen-op-decls"],
        "dialect/tpu/tpu_ops.cc.inc": ["-gen-op-defs"],
        "dialect/tpu/tpu_dialect.h.inc": ["-gen-dialect-decls"],
        "dialect/tpu/tpu_dialect.cc.inc": ["-gen-dialect-defs"],
        "dialect/tpu/tpu_enums.h.inc": ["-gen-enum-decls"],
        "dialect/tpu/tpu_enums.cc.inc": ["-gen-enum-defs"],
        "dialect/tpu/tpu_attr_defs.h.inc": ["-gen-attrdef-decls"],
        "dialect/tpu/tpu_attr_defs.cc.inc": ["-gen-attrdef-defs"],
        "dialect/tpu/tpu_type_defs.h.inc": ["-gen-typedef-decls"],
        "dialect/tpu/tpu_type_defs.cc.inc": ["-gen-typedef-defs"],
        "dialect/tpu/tpu_passes.h.inc": [
            "-gen-pass-decls",
            "-name=TPU",
        ],
        "dialect/tpu/integrations/c/tpu_passes.capi.h.inc": [
            "-gen-pass-capi-header",
            "--prefix=TPU",
        ],
        "dialect/tpu/integrations/c/tpu_passes.capi.cc.inc": [
            "-gen-pass-capi-impl",
            "--prefix=TPU",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/tpu/tpu.td",
    deps = [":tpu_td_files"],
)

td_library(
    name = "tpu_td_files",
    srcs = [
        "dialect/tpu/tpu.td",
    ],
    # compatible with libtpu
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

# C API targets

TPU_CAPI_SOURCES = [
    "dialect/tpu/integrations/c/tpu_dialect.cc",
    "dialect/tpu/integrations/c/tpu_passes.capi.cc.inc",
]

TPU_CAPI_HEADERS = [
    "dialect/tpu/integrations/c/tpu_dialect.h",
    "dialect/tpu/integrations/c/tpu_passes.capi.h.inc",
]

cc_library(
    name = "tpu_dialect_capi",
    srcs = TPU_CAPI_SOURCES,
    hdrs = TPU_CAPI_HEADERS,
    deps = [
        ":tpu_dialect",
        ":tpu_inc_gen",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@xla//xla:array",
    ],
)

# Header-only target, used when using the C API from a separate shared library.
cc_library(
    name = "tpu_dialect_capi_headers",
    hdrs = TPU_CAPI_HEADERS,
    deps = [
        ":tpu_inc_gen",
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
)

# Alwayslink target, used when exporting the C API from a shared library.
cc_library(
    name = "tpu_dialect_capi_objects",
    srcs = TPU_CAPI_SOURCES,
    hdrs = TPU_CAPI_HEADERS,
    deps = [
        ":tpu_dialect",
        ":tpu_inc_gen",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIRObjects",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@xla//xla:array",
    ],
    alwayslink = True,
)

cc_test(
    name = "vreg_util_test",
    srcs = ["dialect/tpu/vreg_util_test.cc"],
    deps = [
        ":tpu_dialect",
        "//testing/base/public:gunit_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:VectorDialect",
    ],
)

cc_test(
    name = "array_util_test",
    srcs = ["dialect/tpu/array_util_test.cc"],
    deps = [
        ":tpu_dialect",
        "//testing/base/public:gunit_main",
        "@llvm-project//mlir:Support",
        "@xla//xla:array",
    ],
)

cc_test(
    name = "tpu_ops_verification_test",
    srcs = ["dialect/tpu/tpu_ops_verification_test.cc"],
    deps = [
        ":tpu_dialect",
        "//testing/base/public:gunit_main",
        "@com_google_absl//absl/status",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Support",
        "@xla//xla/mlir/utils:error_util",
    ],
)

filegroup(
    name = "extension_srcs",
    srcs = [
        "dialect/tpu/transforms/extensions/apply_vector_layout_extensions.cc",
        "dialect/tpu/transforms/extensions/infer_vector_layout_extensions.cc",
    ],
    # compatible with libtpu
)

cc_library(
    name = "pass_boilerplate",
    hdrs = ["pass_boilerplate.h"],
    # compatible with libtpu
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "serde",
    srcs = ["serde.cc"],
    hdrs = ["serde.h"],
    # compatible with libtpu
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
