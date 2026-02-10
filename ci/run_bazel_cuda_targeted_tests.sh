#!/bin/bash
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
# Runs targeted Bazel CUDA tests used by B200/H100 workflows.
#
# Required environment variables:
#   JAXCI_BAZEL_TARGETS: newline-separated Bazel targets.
#
# Optional environment variables:
#   JAXCI_TEST_TAG_FILTERS: value for --test_tag_filters
#   JAXCI_USE_PARALLEL_ACCELERATOR_RUNNER: "1" to use parallel_accelerator_execute.sh
#   JAXCI_LOCAL_TEST_JOBS: value for --local_test_jobs (default: 8)
#   JAXCI_EXCLUDE_TEST_TARGETS: value for JAX_EXCLUDE_TEST_TARGETS test env
#   JAXCI_TEST_TIMEOUT: value for --test_timeout
#   JAXCI_ACCELERATOR_COUNT: value for JAX_ACCELERATOR_COUNT when run_under is enabled (default: 1)
#   JAXCI_TESTS_PER_ACCELERATOR: value for JAX_TESTS_PER_ACCELERATOR when run_under is enabled (default: 8)
#   JAXCI_HERMETIC_PYTHON_VERSION: Hermetic Python version (default: 3.14 in workflow)
#   JAXCI_XLA_TRACK: XLA source to use, "pinned" or "head" (default: pinned)

set -eo pipefail

source ci/envs/default.env

if [[ -z "${JAXCI_BAZEL_TARGETS:-}" ]]; then
  echo 'JAXCI_BAZEL_TARGETS must be set.'
  exit 1
fi

hermetic_python_version="${JAXCI_HERMETIC_PYTHON_VERSION:-3.14}"
if [[ ! "${hermetic_python_version}" =~ ^3\.[0-9]+$ ]]; then
  echo "Invalid JAXCI_HERMETIC_PYTHON_VERSION: ${hermetic_python_version}. Expected <major>.<minor> (for example, 3.14)."
  exit 1
fi

xla_track="${JAXCI_XLA_TRACK:-pinned}"
if [[ "${xla_track}" != 'pinned' && "${xla_track}" != 'head' ]]; then
  echo "Invalid JAXCI_XLA_TRACK value: ${xla_track}. Expected 'pinned' or 'head'."
  exit 1
fi

if [[ "${xla_track}" == 'pinned' ]]; then
  export JAXCI_CLONE_MAIN_XLA=0
  unset JAXCI_XLA_GIT_DIR
else
  # Reuse an explicit local XLA checkout if set. Otherwise clone XLA HEAD.
  if [[ -z "${JAXCI_XLA_GIT_DIR:-}" ]]; then
    export JAXCI_CLONE_MAIN_XLA=1
  fi
fi

source ci/utilities/setup_build_environment.sh

nvidia-smi

local_test_jobs="${JAXCI_LOCAL_TEST_JOBS:-8}"

bazel_args=(
  --config=ci_linux_x86_64_cuda
  --config=ci_rbe_cache
  --config=hermetic_cuda_umd
  --repo_env=HERMETIC_PYTHON_VERSION="${hermetic_python_version}"
  --repo_env=HERMETIC_CUDNN_VERSION=9.11.0
  --repo_env=HERMETIC_CUDA_UMD_VERSION=13.0.0
  --test_env=XLA_PYTHON_CLIENT_ALLOCATOR=platform
  --test_output=errors
  --strategy=TestRunner=local
  --local_test_jobs="${local_test_jobs}"
  --test_env=TF_CPP_MIN_LOG_LEVEL=0
  --test_env=JAX_SKIP_SLOW_TESTS=true
  --action_env=JAX_ENABLE_X64=1
  --action_env=NCCL_DEBUG=WARN
  --flaky_test_attempts=1
  --color=yes
)

if [[ "${xla_track}" == 'head' ]]; then
  if [[ -z "${JAXCI_XLA_GIT_DIR:-}" ]]; then
    echo 'JAXCI_XLA_GIT_DIR is not set for XLA track "head".'
    exit 1
  fi
  bazel_args+=(--override_repository=xla="${JAXCI_XLA_GIT_DIR}")
fi

if [[ -n "${JAXCI_TEST_TAG_FILTERS:-}" ]]; then
  bazel_args+=(--test_tag_filters="${JAXCI_TEST_TAG_FILTERS}")
fi

if [[ "${JAXCI_USE_PARALLEL_ACCELERATOR_RUNNER:-0}" == '1' ]]; then
  bazel_args+=(--run_under "$(pwd)/build/parallel_accelerator_execute.sh")
  bazel_args+=(--test_env=JAX_ACCELERATOR_COUNT="${JAXCI_ACCELERATOR_COUNT:-1}")
  bazel_args+=(--test_env=JAX_TESTS_PER_ACCELERATOR="${JAXCI_TESTS_PER_ACCELERATOR:-8}")
fi

if [[ -n "${JAXCI_EXCLUDE_TEST_TARGETS:-}" ]]; then
  bazel_args+=(--test_env=JAX_EXCLUDE_TEST_TARGETS="${JAXCI_EXCLUDE_TEST_TARGETS}")
fi

if [[ -n "${JAXCI_TEST_TIMEOUT:-}" ]]; then
  bazel_args+=(--test_timeout="${JAXCI_TEST_TIMEOUT}")
fi

targets=()
while IFS= read -r target; do
  if [[ -n "${target}" ]]; then
    targets+=("${target}")
  fi
done <<< "${JAXCI_BAZEL_TARGETS}"

if [[ ${#targets[@]} -eq 0 ]]; then
  echo 'No Bazel targets were provided.'
  exit 1
fi

bazel test "${bazel_args[@]}" "${targets[@]}"
