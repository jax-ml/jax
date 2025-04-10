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

load("@llvm-project//mlir:tblgen.bzl", "gentbl_filegroup")
load("//jaxlib:jax.bzl", "if_windows", "pytype_strict_library")

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:internal"],
)

pytype_strict_library(
    name = "triton",
    srcs = [
        "__init__.py",
        "dialect.py",
        ":_triton_gen",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//jaxlib/mlir:core",
        "//jaxlib/mlir:ir",
    ] + if_windows(
        [],
        [
            "//jaxlib/mlir/_mlir_libs:_triton_ext",
        ],
    ),
)

genrule(
    name = "_triton_gen",
    srcs = [
        "_triton_ops_gen_raw.py",
        "_triton_enum_gen_raw.py",
    ],
    outs = [
        "_triton_ops_gen.py",
        "_triton_enum_gen.py",
    ],
    # Use $(RULEDIR) to avoid an implicit dependency on whether inputs are in bin or genfiles.
    cmd = """
    for src in $(SRCS); do
      base=$$(basename $$src)
      out=$(RULEDIR)/$${base//_raw/}
      echo '# pytype: skip-file' > $${out} && \
          cat $${src} |
          sed -e 's/^from \\.\\./from jaxlib\\.mlir\\./g' |
          sed -e 's/^from \\./from jaxlib\\.mlir\\.dialects\\./g' >> $${out}
    done
    """,
)

gentbl_filegroup(
    name = "_triton_gen_raw",
    tbl_outs = [
        (
            [
                "-gen-python-enum-bindings",
                "-bind-dialect=tt",
            ],
            "_triton_enum_gen_raw.py",
        ),
        (
            [
                "-gen-python-op-bindings",
                "-bind-dialect=tt",
            ],
            "_triton_ops_gen_raw.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "triton.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@triton//:td_files",
    ],
)

cc_library(
    name = "triton_dialect_capi",
    srcs = ["triton_dialect_capi.cc"],
    hdrs = ["triton_dialect_capi.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:IR",
        "@triton//:TritonDialects",
    ],
)

# Header-only target, used when using the C API from a separate shared library.
cc_library(
    name = "triton_dialect_capi_headers",
    hdrs = ["triton_dialect_capi.h"],
    deps = [
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
)

# Alwayslink target, used when exporting the C API from a shared library.
cc_library(
    name = "triton_dialect_capi_objects",
    srcs = ["triton_dialect_capi.cc"],
    hdrs = ["triton_dialect_capi.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIRObjects",
        "@llvm-project//mlir:IR",
        "@triton//:TritonDialects",
    ],
    alwayslink = True,
)
