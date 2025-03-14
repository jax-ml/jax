#!/bin/bash
# Copyright 2025 The JAX Authors.
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
# Runs Bazel CPU/TPU tests. Requires a jaxlib wheel to be present
# inside $JAXCI_OUTPUT_DIR (../dist)
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

IGNORE_TESTS=""
tpu_version_suffix="tpu"

if [[ "$JAXCI_LIBTPU_VERSION_TYPE" == "nightly" ]]; then
  tpu_version_suffix="nightlytpu"
  bazel run //build:nightly_tpu_requirements.update \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION-$tpu_version_suffix"
elif [[ "$JAXCI_LIBTPU_VERSION_TYPE" == "oldest_supported_libtpu" ]]; then
  tpu_version_suffix="oldesttpu"
  bazel run //build:oldest_tpu_requirements.update \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION-$tpu_version_suffix"
  # We're deselecting all Pallas TPU tests in the oldest libtpu build. Mosaic
  # TPU does not guarantee anything about forward compatibility (unless
  # jax.export is used) and the 12 week compatibility window accumulates way
  # too many failures.  
  IGNORE_TESTS="-//tests/pallas/..."
else
  bazel run //build:tpu_requirements.update \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION-$tpu_version_suffix"
fi

echo "Running TPU tests..."

if [[ "$JAXCI_RUN_FULL_TPU_TEST_SUITE" == "1" ]]; then
  # Run single-accelerator tests in parallel
  bazel test \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION-$tpu_version_suffix" \
    --config=ci_linux_x86_64 \
    --//jax:build_jaxlib=false \
    --//jax:with_libtpu_dependency=true \
    --test_env=JAX_TEST_NUM_THREADS=16  \
    --run_under="$(pwd)/build/parallel_accelerator_execute.sh" \
    --test_env=JAX_ACCELERATOR_COUNT=8 \
    --test_env=JAX_TESTS_PER_ACCELERATOR=1 \
    --test_env=ALLOW_MULTIPLE_LIBTPU_LOAD=true \
    --local_test_jobs="$JAXCI_TPU_CORES" \
    --test_env=JAX_SKIP_SLOW_TESTS=1 \
    --test_env=JAX_ENABLE_TPU_XDIST=1 \
    --test_env=TPU_TOPOLOGY \
    --test_env=TPU_WORKER_ID \
    --test_env=TPU_SKIP_MDS_QUERY \
    --test_env=TPU_TOPOLOGY_WRAP \
    --test_env=TPU_CHIPS_PER_HOST_BOUNDS \
    --test_env=TPU_ACCELERATOR_TYPE \
    --test_env=TPU_RUNTIME_METRICS_PORTS \
    --test_env=TPU_TOPOLOGY_ALT \
    --test_env=TPU_HOST_BOUNDS \
    --test_env=TPU_WORKER_HOSTNAMES \
    --test_tag_filters=-multiaccelerator \
    --test_filter="^(?!PallasCallPrintTest$).*" \
    --verbose_failures \
    --test_output=errors \
    -- \
    //tests:cpu_tests \
    //tests:tpu_tests \
    //tests:backend_independent_tests \
    //tests/pallas:cpu_tests \
    //tests/pallas:tpu_tests \
    //tests/pallas:backend_independent_tests \
    //examples:cpu_tests \
    //examples:tpu_tests \
    //examples:backend_independent_tests \
    $IGNORE_TESTS

  # Run multi-accelerator across all chips
  bazel test \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION-$tpu_version_suffix" \
    --config=ci_linux_x86_64 \
    --//jax:build_jaxlib=false \
    --//jax:with_libtpu_dependency=true \
    --test_env=JAX_TEST_NUM_THREADS=16  \
    --run_under="$(pwd)/build/parallel_accelerator_execute.sh" \
    --test_env=JAX_ACCELERATOR_COUNT=8 \
    --test_env=JAX_TESTS_PER_ACCELERATOR=1 \
    --test_env=ALLOW_MULTIPLE_LIBTPU_LOAD=true \
    --local_test_jobs="$JAXCI_TPU_CORES" \
    --test_env=JAX_SKIP_SLOW_TESTS=1 \
    --test_env=TPU_TOPOLOGY \
    --test_env=TPU_WORKER_ID \
    --test_env=TPU_SKIP_MDS_QUERY \
    --test_env=TPU_TOPOLOGY_WRAP \
    --test_env=TPU_CHIPS_PER_HOST_BOUNDS \
    --test_env=TPU_ACCELERATOR_TYPE \
    --test_env=TPU_RUNTIME_METRICS_PORTS \
    --test_env=TPU_TOPOLOGY_ALT \
    --test_env=TPU_HOST_BOUNDS \
    --test_env=TPU_WORKER_HOSTNAMES \
    --test_tag_filters=multiaccelerator \
    --verbose_failures \
    --test_output=errors \
    -- \
    //tests:cpu_tests \
    //tests:tpu_tests \
    //tests:backend_independent_tests \
    //tests/pallas:cpu_tests \
    //tests/pallas:tpu_tests \
    //tests/pallas:backend_independent_tests \
    $IGNORE_TESTS
else

  # Run single-accelerator tests in parallel
  bazel test \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION-$tpu_version_suffix" \
    --config=ci_linux_x86_64 \
    --//jax:build_jaxlib=false \
    --//jax:with_libtpu_dependency=true \
    --test_env=JAX_TEST_NUM_THREADS=16  \
    --run_under="$(pwd)/build/parallel_accelerator_execute.sh" \
    --test_env=JAX_ACCELERATOR_COUNT=8 \
    --test_env=JAX_TESTS_PER_ACCELERATOR=1 \
    --test_env=ALLOW_MULTIPLE_LIBTPU_LOAD=true \
    --local_test_jobs="$JAXCI_TPU_CORES" \
    --test_env=JAX_SKIP_SLOW_TESTS=1 \
    --test_env=JAX_ENABLE_TPU_XDIST=1 \
    --test_env=TPU_TOPOLOGY \
    --test_env=TPU_WORKER_ID \
    --test_env=TPU_SKIP_MDS_QUERY \
    --test_env=TPU_TOPOLOGY_WRAP \
    --test_env=TPU_CHIPS_PER_HOST_BOUNDS \
    --test_env=TPU_ACCELERATOR_TYPE \
    --test_env=TPU_RUNTIME_METRICS_PORTS \
    --test_env=TPU_TOPOLOGY_ALT \
    --test_env=TPU_HOST_BOUNDS \
    --test_env=TPU_WORKER_HOSTNAMES \
    --test_tag_filters=-multiaccelerator \
    --test_filter="^(?!PallasCallPrintTest$).*" \
    --verbose_failures \
    --test_output=errors \
    -- \
    //tests/pallas:ops_test_cpu \
    //tests/pallas:ops_test_tpu \
    //tests/pallas:export_back_compat_pallas_test_cpu \
    //tests/pallas:export_pallas_test_cpu \
    //tests/pallas:export_pallas_test_tpu \
    //tests/pallas:tpu_ops_test_cpu \
    //tests/pallas:tpu_ops_test_tpu \
    //tests/pallas:tpu_pallas_random_test_tpu \
    //tests/pallas:tpu_pallas_async_test_tpu \
    //tests/pallas:tpu_pallas_state_test_tpu \
    $IGNORE_TESTS

  # Run multi-accelerator across all chips
  bazel test \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION-$tpu_version_suffix" \
    --config=ci_linux_x86_64 \
    --//jax:build_jaxlib=false \
    --//jax:with_libtpu_dependency=true \
    --test_env=JAX_TEST_NUM_THREADS=16  \
    --run_under="$(pwd)/build/parallel_accelerator_execute.sh" \
    --test_env=JAX_ACCELERATOR_COUNT=8 \
    --test_env=JAX_TESTS_PER_ACCELERATOR=1 \
    --test_env=ALLOW_MULTIPLE_LIBTPU_LOAD=true \
    --local_test_jobs="$JAXCI_TPU_CORES" \
    --test_env=JAX_SKIP_SLOW_TESTS=1 \
    --test_env=TPU_TOPOLOGY \
    --test_env=TPU_WORKER_ID \
    --test_env=TPU_SKIP_MDS_QUERY \
    --test_env=TPU_TOPOLOGY_WRAP \
    --test_env=TPU_CHIPS_PER_HOST_BOUNDS \
    --test_env=TPU_ACCELERATOR_TYPE \
    --test_env=TPU_RUNTIME_METRICS_PORTS \
    --test_env=TPU_TOPOLOGY_ALT \
    --test_env=TPU_HOST_BOUNDS \
    --test_env=TPU_WORKER_HOSTNAMES \
    --test_tag_filters=multiaccelerator \
    --verbose_failures \
    --test_output=errors \
    -- \
    //tests:pjit_test_tpu \
    //tests:pjit_test_cpu \
    //tests/pallas:tpu_pallas_distributed_test_tpu \
    $IGNORE_TESTS
fi

# Run Pallas printing tests, which need to run with I/O capturing disabled.
bazel test \
  --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION-$tpu_version_suffix" \
  --config=ci_linux_x86_64 \
  --//jax:build_jaxlib=false \
  --//jax:with_libtpu_dependency=true \
  --test_env=JAX_TEST_NUM_THREADS=16  \
  --run_under="$(pwd)/build/parallel_accelerator_execute.sh" \
  --test_env=JAX_ACCELERATOR_COUNT=8 \
  --test_env=JAX_TESTS_PER_ACCELERATOR=1 \
  --test_env=ALLOW_MULTIPLE_LIBTPU_LOAD=true \
  --local_test_jobs="$JAXCI_TPU_CORES" \
  --test_env=JAX_SKIP_SLOW_TESTS=1 \
  --test_env=TPU_TOPOLOGY \
  --test_env=TPU_WORKER_ID \
  --test_env=TPU_SKIP_MDS_QUERY \
  --test_env=TPU_TOPOLOGY_WRAP \
  --test_env=TPU_CHIPS_PER_HOST_BOUNDS \
  --test_env=TPU_ACCELERATOR_TYPE \
  --test_env=TPU_RUNTIME_METRICS_PORTS \
  --test_env=TPU_TOPOLOGY_ALT \
  --test_env=TPU_HOST_BOUNDS \
  --test_env=TPU_WORKER_HOSTNAMES \
  --test_env=TPU_STDERR_LOG_LEVEL=0 \
  --test_filter=PallasCallPrintTest \
  --verbose_failures \
  --test_output=all \
  -- \
  //tests/pallas:tpu_pallas_test_tpu \
  $IGNORE_TESTS
