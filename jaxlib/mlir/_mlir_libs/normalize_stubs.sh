#!/bin/bash
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

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <FILES...>"
  exit 1
fi

# Replace all occurrences of `mlir._mlir_libs._mlir.ir` with `mlir.ir`.
sed -i 's/mlir\._mlir_libs\._mlir\.ir/mlir.ir/g' "$@"
# Replace `import mlir.ir` with `from mlir import ir`.
sed -i 's/import mlir\.ir/from mlir import ir/g' "$@"
# Deduplicate `from mlir import ir` lines, keeping only the first one.
sed -i -e '0,/^[[:space:]]*from mlir import ir[[:space:]]*$/b' \
  -e '/^[[:space:]]*from mlir import ir[[:space:]]*$/d' "$@"
# Replace `mlir.ir.<NAME>` with `ir.<NAME>`.
sed -i -E 's/mlir\.ir\.([a-zA-Z0-9_]+)/ir.\1/g' "$@"

# TODO(slebedev): Remove once https://github.com/llvm/llvm-project/pull/183021
# is merged and available internally.
# Replace bare `TypeID` with `ir.TypeID`.
sed -i -E 's/: TypeID/: ir.TypeID/g' "$@"
