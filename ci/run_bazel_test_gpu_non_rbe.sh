#!/bin/bash
# Copyright 2024 The JAX Authors.
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
# Run Bazel GPU tests without RBE. This runs two commands: single accelerator
# tests with one GPU a piece, multiaccelerator tests with all GPUS.
# Requires that jaxlib, jax-cuda-plugin, and jax-cuda-pjrt wheels are stored
# inside the ../dist folder
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport

# Source default JAXCI environment variables.
source ci/envs/default.env

# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

# Run Bazel GPU tests (single accelerator and multiaccelerator tests) directly
# on the VM without RBE.
nvidia-smi
echo "Running single accelerator tests (without RBE)..."

# Set up test environment variables.
export gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export num_test_jobs=$((gpu_count * JAXCI_MAX_TESTS_PER_GPU))
export num_cpu_cores=$(nproc)

# tests_jobs = max(gpu_count * max_tests_per_gpu, num_cpu_cores)
if [[ $num_test_jobs -gt $num_cpu_cores ]]; then
  num_test_jobs=$num_cpu_cores
fi
# End of test environment variables setup.

# Runs single accelerator tests with one GPU apiece.
# It appears --run_under needs an absolute path.
# The product of the `JAX_ACCELERATOR_COUNT`` and `JAX_TESTS_PER_ACCELERATOR`
# should match the VM's CPU core count (set in `--local_test_jobs`).
bazel test --config=ci_linux_x86_64_cuda \
      --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
      --//jax:build_jaxlib=false \
      --test_env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
      --run_under "$(pwd)/build/parallel_accelerator_execute.sh" \
      --test_output=errors \
      --test_env=JAX_ACCELERATOR_COUNT=$gpu_count \
      --test_env=JAX_TESTS_PER_ACCELERATOR=$JAXCI_MAX_TESTS_PER_GPU \
      --local_test_jobs=$num_test_jobs \
      --test_env=JAX_EXCLUDE_TEST_TARGETS=PmapTest.testSizeOverflow \
      --test_tag_filters=-multiaccelerator \
      --test_env=TF_CPP_MIN_LOG_LEVEL=0 \
      --test_env=JAX_SKIP_SLOW_TESTS=true \
      --action_env=JAX_ENABLE_X64="$JAXCI_ENABLE_X64" \
      --action_env=NCCL_DEBUG=WARN \
      --color=yes \
      //tests:gpu_tests //tests:backend_independent_tests \
      //tests/pallas:gpu_tests //tests/pallas:backend_independent_tests

echo "Running multi-accelerator tests (without RBE)..."
# Runs multiaccelerator tests with all GPUs directly on the VM without RBE..
bazel test --config=ci_linux_x86_64_cuda \
      --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
      --//jax:build_jaxlib=false \
      --test_env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
      --test_output=errors \
      --jobs=8 \
      --test_tag_filters=multiaccelerator \
      --test_env=TF_CPP_MIN_LOG_LEVEL=0 \
      --test_env=JAX_SKIP_SLOW_TESTS=true \
      --action_env=JAX_ENABLE_X64="$JAXCI_ENABLE_X64" \
      --action_env=NCCL_DEBUG=WARN \
      --color=yes \
      //tests:gpu_tests //tests/pallas:gpu_tests