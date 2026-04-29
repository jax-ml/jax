#!/bin/bash
# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Mirrors NCCL's bindings/ir/Makefile pipeline:
#   clang -emit-llvm → opt (internalize) → llvm-dis (strip ftz) → llvm-as
#
# Args (passed by Bazel rule):
#   --clang, --opt, --llvm-dis, --llvm-as: tool paths
#   --cuda-nvcc-path: path containing nvvm/ directory
#   --cuda-inc-dirs: space-separated CUDA include directories
#   --resource-dir: clang builtin headers
#   --gpu-arch: e.g. sm_90
#   --nccl-device-header: path to nccl_device.h
#   --nccl-inc-flags: -I flags for NCCL headers
#   --src: wrapper impl source file
#   --out: output .bc file
set -euo pipefail

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clang) CLANG="$2"; shift 2;;
    --opt) OPT="$2"; shift 2;;
    --llvm-dis) LLVM_DIS="$2"; shift 2;;
    --llvm-as) LLVM_AS="$2"; shift 2;;
    --cuda-nvcc-path) CUDA_NVCC_PATH="$2"; shift 2;;
    --cuda-inc-dirs) CUDA_INC_DIRS="$2"; shift 2;;
    --resource-dir) RESOURCE_DIR="$2"; shift 2;;
    --gpu-arch) GPU_ARCH="$2"; shift 2;;
    --nccl-device-header) NCCL_DEVICE_HEADER="$2"; shift 2;;
    --nccl-inc-flags) NCCL_INC_FLAGS="$2"; shift 2;;
    --src) SRC="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

OBJDIR=$(dirname "$OUT")

# Hermetic CUDA splits headers across Bazel packages but clang --cuda-path
# expects a single root with nvvm/ and include/. Build a merged tree.
CUDA_MERGED=$(mktemp -d)
trap 'rm -rf "$CUDA_MERGED"' EXIT
ln -s "$(pwd)/$CUDA_NVCC_PATH/nvvm" "$CUDA_MERGED/nvvm"
mkdir -p "$CUDA_MERGED/bin" "$CUDA_MERGED/include"
for dir in $CUDA_INC_DIRS; do
  if [ -d "$dir" ]; then
    for f in "$dir"/*; do
      base=$(basename "$f")
      [ ! -e "$CUDA_MERGED/include/$base" ] && ln -s "$(pwd)/$f" "$CUDA_MERGED/include/$base"
    done
  fi
done

# Step 1: clang -emit-llvm (same as NCCL Makefile CLANG_FLAGS)
$CLANG -c -emit-llvm -O1 \
  -std=gnu++17 -x cuda \
  --cuda-path="$CUDA_MERGED" \
  --cuda-device-only -nocudalib \
  --cuda-gpu-arch="$GPU_ARCH" \
  -resource-dir "$RESOURCE_DIR" \
  -D__clang_llvm_bitcode_lib__ \
  -D_NV_RSQRT_SPECIFIER= \
  -Wno-unknown-cuda-version \
  -include "$NCCL_DEVICE_HEADER" \
  $NCCL_INC_FLAGS \
  "$SRC" \
  -o "$OBJDIR/unopt.bc"

# Step 2: opt (internalize non-nccl symbols, inline, DCE)
$OPT \
  --passes='internalize,inline,globaldce' \
  -internalize-public-api-list='*nccl*' \
  "$OBJDIR/unopt.bc" -o "$OBJDIR/opt.bc"

# Step 3: Strip nvvm-reflect-ftz metadata (same as NCCL Makefile)
$LLVM_DIS "$OBJDIR/opt.bc" -o "$OBJDIR/opt.ll"
FTZ_ID=$(grep -oP '!(\d+) = !\{[^"]*"nvvm-reflect-ftz"' "$OBJDIR/opt.ll" | head -1 | cut -d' ' -f1 || true)
if [ -n "$FTZ_ID" ]; then
  awk '!/nvvm-reflect-ftz/' "$OBJDIR/opt.ll" | \
    sed "/^!llvm\.module\.flags/s/$FTZ_ID, //" | \
    sed "/^!llvm\.module\.flags/s/, $FTZ_ID//" > "$OBJDIR/clean.ll"
else
  cp "$OBJDIR/opt.ll" "$OBJDIR/clean.ll"
fi

# Step 4: llvm-as → final bitcode
$LLVM_AS "$OBJDIR/clean.ll" -o "$OUT"
rm -f "$OBJDIR/unopt.bc" "$OBJDIR/opt.bc" "$OBJDIR/opt.ll" "$OBJDIR/clean.ll"
