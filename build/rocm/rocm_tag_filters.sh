#!/usr/bin/env bash
# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Helper for ROCm CI scripts to compute Bazel tag filters and pytest args.
#
# Intended usage:
#   source build/rocm/rocm_tag_filters.sh "$@"
#
# Exports:
#   ROCM_TAG_FILTERS: Bazel tag filter string.
#   ROCM_PYTEST_MARKER_EXPR: Optional pytest marker expression to pass via `-m`.
#     When unset/empty, pytest will collect all tests and rely on `conftest.py`
#     to mark Mosaic GPU tests and skip them on ROCm (so CI reports explicit
#     SKIPPED instead of deselection).
#   ROCM_LOCAL_TEST_JOBS: Optional Bazel concurrency cap for tests. If unset,
#     the caller can pass Bazel flags like --test_jobs=N / --local_test_jobs=N
#     explicitly. For multiaccelerator runs we default to 1 unless overridden.
set -euo pipefail

: "${ROCM_PYTEST_MARKER_EXPR:=}"
: "${ROCM_LOCAL_TEST_JOBS:=}"

ROCM_TAG_FILTERS="jax_test_gpu,-config-cuda-only,-manual"

_saw_multi=0
_saw_single=0
for arg in "$@"; do
  case "$arg" in
    --config=multi_gpu|--config=rocm_multi_gpu)
      _saw_multi=1
      ;;
    --config=single_gpu|--config=rocm_single_gpu)
      _saw_single=1
      ;;
  esac
done

if [[ $_saw_multi -eq 1 ]]; then
  ROCM_TAG_FILTERS="${ROCM_TAG_FILTERS},multiaccelerator"
elif [[ $_saw_single -eq 1 ]]; then
  ROCM_TAG_FILTERS="${ROCM_TAG_FILTERS},gpu,-multiaccelerator"
fi

if [[ -z "$ROCM_LOCAL_TEST_JOBS" && $_saw_multi -eq 1 ]]; then
  # Multiaccelerator tests typically consume all GPUs on the machine.
  ROCM_LOCAL_TEST_JOBS=1
fi

export ROCM_TAG_FILTERS
export ROCM_PYTEST_MARKER_EXPR
export ROCM_LOCAL_TEST_JOBS

