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

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:internal"],
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
    td_file = "@xla//xla/mosaic/dialect/gpu:mosaic_gpu.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@xla//xla/mosaic/dialect/gpu:mosaic_gpu_td_files",
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
