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

load("//jaxlib:jax.bzl", "pybind_extension", "pytype_strict_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_filegroup")

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
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
        ":_triton_ext",
        "//jaxlib/mlir:core",
        "//jaxlib/mlir:ir",
    ],
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
    cmd = """
    for src in $(SRCS); do
      out=$${src//_raw/}
      echo '# pytype: skip-file' > $${out} && \
          cat $${src} |
          sed -e 's/^from \\.\\./from mlir\\./g' |
          sed -e 's/^from \\./from mlir\\.dialects\\./g' >> $${out}
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

pybind_extension(
    name = "_triton_ext",
    srcs = ["_triton_ext.cc"],
    pytype_deps = [
        "//jaxlib/mlir:ir",
    ],
    pytype_srcs = ["_triton_ext.pyi"],
    visibility = ["//visibility:private"],
    deps = [
        ":triton_dialect_capi",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:MLIRBindingsPythonHeadersAndDeps",
        "@pybind11",
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
