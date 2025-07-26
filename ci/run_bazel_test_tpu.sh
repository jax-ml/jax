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
# Runs Bazel CPU/TPU tests. Requires jax and jaxlib wheels to be present
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

if [[ "$JAXCI_HERMETIC_PYTHON_VERSION" == *"-nogil" ]]; then
  JAXCI_HERMETIC_PYTHON_VERSION=${JAXCI_HERMETIC_PYTHON_VERSION%-nogil}-ft
  FREETHREADED_FLAG_VALUE="yes"
else
  FREETHREADED_FLAG_VALUE="no"
fi

# TODO(ybaturina): Bazel cache shouldn't be invalidated when
# `VBAR_CONTROL_SERVICE_URL` changes.
COMMON_TPU_TEST_ENV_VARS="--test_env=TPU_TOPOLOGY \
 --test_env=TPU_WORKER_ID \
 --test_env=TPU_SKIP_MDS_QUERY=true \
 --test_env=TPU_TOPOLOGY_WRAP \
 --test_env=TPU_CHIPS_PER_HOST_BOUNDS \
 --test_env=TPU_ACCELERATOR_TYPE \
 --test_env=TPU_RUNTIME_METRICS_PORTS \
 --test_env=TPU_TOPOLOGY_ALT \
 --test_env=TPU_HOST_BOUNDS \
 --test_env=TPU_WORKER_HOSTNAMES \
 --test_env=CHIPS_PER_HOST_BOUNDS \
 --test_env=HOST_BOUNDS \
 --test_env=ALT=false \
 --test_env=WRAP \
 --test_env=VBAR_CONTROL_SERVICE_URL"
echo "COMMON_TPU_TEST_ENV_VARS: $COMMON_TPU_TEST_ENV_VARS"

echo "Running Bazel TPU tests..."

# Don't abort the script if one command fails to ensure we run both test
# commands below.
set +e

if [[ "$JAXCI_RUN_FULL_TPU_TEST_SUITE" == "1" ]]; then
  # We're deselecting all Pallas TPU tests in the oldest libtpu build. Mosaic
  # TPU does not guarantee anything about forward compatibility (unless
  # jax.export is used) and the 12 week compatibility window accumulates way
  # too many failures.
  IGNORE_TESTS=""
  if [ "${libtpu_version_type:-""}" == "oldest_supported_libtpu" ]; then
    IGNORE_TESTS="-//tests/pallas/..."
  fi

  # Run single-accelerator tests in parallel
  bazel test \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
    --@rules_python//python/config_settings:py_freethreaded="$FREETHREADED_FLAG_VALUE" \
    --config=ci_linux_x86_64 \
    --config=ci_rbe_cache \
    --//jax:build_jaxlib=false \
    --//jax:build_jax=false \
    --test_env=JAX_TEST_NUM_THREADS=16  \
    --run_under="$(pwd)/build/parallel_accelerator_execute.sh" \
    --test_sharding_strategy=disabled \
    --test_env=JAX_ACCELERATOR_COUNT="$JAXCI_TPU_CORES" \
    --test_env=JAX_TESTS_PER_ACCELERATOR=1 \
    --test_env=ALLOW_MULTIPLE_LIBTPU_LOAD=true \
    --local_test_jobs="$JAXCI_TPU_CORES" \
    --test_env=JAX_SKIP_SLOW_TESTS=1 \
    --test_env=JAX_ENABLE_TPU_XDIST=1 \
    $COMMON_TPU_TEST_ENV_VARS \
    --test_tag_filters=-multiaccelerator \
    --test_filter="^(?!.*(PallasCallPrintTest|InterpretTest::test_thread_map))" \
    --verbose_failures \
    --test_output=summary \
    -- \
    //tests:cpu_tests \
    //tests:tpu_tests \
    //tests:backend_independent_tests \
    //tests/pallas:cpu_tests \
    //tests/pallas:tpu_tests \
    //tests/pallas:backend_independent_tests \
    //tests/pallas:tpu_pallas_test_tpu \
    //tests/mosaic:backend_independent_tests \
    $IGNORE_TESTS

  # Store the return value of the first bazel command.
  first_bazel_cmd_retval=$?

  # Run multi-accelerator across all chips
  bazel test \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
    --@rules_python//python/config_settings:py_freethreaded="$FREETHREADED_FLAG_VALUE" \
    --config=ci_linux_x86_64 \
    --config=ci_rbe_cache \
    --//jax:build_jaxlib=false \
    --//jax:build_jax=false \
    --test_env=JAX_TEST_NUM_THREADS=16  \
    --run_under="$(pwd)/build/parallel_accelerator_execute.sh" \
    --test_sharding_strategy=disabled \
    --test_env=JAX_ACCELERATOR_COUNT="$JAXCI_TPU_CORES" \
    --test_env=JAX_TESTS_PER_ACCELERATOR=1 \
    --test_env=ALLOW_MULTIPLE_LIBTPU_LOAD=true \
    --local_test_jobs="$JAXCI_TPU_CORES" \
    --test_env=JAX_SKIP_SLOW_TESTS=1 \
    $COMMON_TPU_TEST_ENV_VARS \
    --test_tag_filters=multiaccelerator \
    --test_filter="^(?!PallasCallPrintTest$).*" \
    --verbose_failures \
    --test_output=summary \
    -- \
    //tests:cpu_tests \
    //tests:tpu_tests \
    //tests:backend_independent_tests \
    //tests/pallas:cpu_tests \
    //tests/pallas:tpu_tests \
    //tests/pallas:backend_independent_tests \
    //tests/pallas:tpu_pallas_test_tpu

  # Store the return value of the second bazel command.
  second_bazel_cmd_retval=$?
else

  # Run single-accelerator tests in parallel
  bazel test \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
    --@rules_python//python/config_settings:py_freethreaded="$FREETHREADED_FLAG_VALUE" \
    --config=ci_linux_x86_64 \
    --config=ci_rbe_cache \
    --//jax:build_jaxlib=false \
    --//jax:build_jax=false \
    --test_env=JAX_TEST_NUM_THREADS=16  \
    --run_under="$(pwd)/build/parallel_accelerator_execute.sh" \
    --test_sharding_strategy=disabled \
    --test_env=JAX_ACCELERATOR_COUNT="$JAXCI_TPU_CORES" \
    --test_env=JAX_TESTS_PER_ACCELERATOR=1 \
    --test_env=ALLOW_MULTIPLE_LIBTPU_LOAD=true \
    --local_test_jobs="$JAXCI_TPU_CORES" \
    --test_env=JAX_SKIP_SLOW_TESTS=1 \
    --test_env=JAX_ENABLE_TPU_XDIST=1 \
    $COMMON_TPU_TEST_ENV_VARS \
    --test_tag_filters=-multiaccelerator \
    --test_filter="^(?!PallasCallPrintTest$).*" \
    --verbose_failures \
    --test_output=summary \
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
    //tests/pallas:tpu_pallas_test_tpu

  # Store the return value of the first bazel command.
  first_bazel_cmd_retval=$?

  # Run multi-accelerator across all chips
  bazel test \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
    --@rules_python//python/config_settings:py_freethreaded="$FREETHREADED_FLAG_VALUE" \
    --config=ci_linux_x86_64 \
    --config=ci_rbe_cache \
    --//jax:build_jaxlib=false \
    --//jax:build_jax=false \
    --test_env=JAX_TEST_NUM_THREADS=16  \
    --run_under="$(pwd)/build/parallel_accelerator_execute.sh" \
    --test_sharding_strategy=disabled \
    --test_env=JAX_ACCELERATOR_COUNT="$JAXCI_TPU_CORES" \
    --test_env=JAX_TESTS_PER_ACCELERATOR=1 \
    --test_env=ALLOW_MULTIPLE_LIBTPU_LOAD=true \
    --local_test_jobs="$JAXCI_TPU_CORES" \
    --test_env=JAX_SKIP_SLOW_TESTS=1 \
    $COMMON_TPU_TEST_ENV_VARS \
    --test_tag_filters=multiaccelerator \
    --verbose_failures \
    --test_output=summary \
    -- \
    //tests:pjit_test_tpu \
    //tests:pjit_test_cpu \
    //tests/pallas:tpu_pallas_distributed_test_tpu

  # Store the return value of the second bazel command.
  second_bazel_cmd_retval=$?
fi

# Run Pallas printing tests, which need to run with I/O capturing disabled.
bazel test \
  --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
  --@rules_python//python/config_settings:py_freethreaded="$FREETHREADED_FLAG_VALUE" \
  --config=ci_linux_x86_64 \
  --config=ci_rbe_cache \
  --//jax:build_jaxlib=false \
  --//jax:build_jax=false \
  --test_env=JAX_TEST_NUM_THREADS=16  \
  --run_under="$(pwd)/build/parallel_accelerator_execute.sh" \
  --test_sharding_strategy=disabled \
  --test_env=JAX_ACCELERATOR_COUNT="$JAXCI_TPU_CORES" \
  --test_env=JAX_TESTS_PER_ACCELERATOR=1 \
  --test_env=ALLOW_MULTIPLE_LIBTPU_LOAD=true \
  --local_test_jobs="$JAXCI_TPU_CORES" \
  --test_env=JAX_SKIP_SLOW_TESTS=1 \
  $COMMON_TPU_TEST_ENV_VARS \
  --test_env=TPU_STDERR_LOG_LEVEL=0 \
  --test_filter=PallasCallPrintTest \
  --verbose_failures \
  --test_output=all \
  -- \
  //tests/pallas:tpu_pallas_test_tpu

# Store the return value of the third bazel command.
third_bazel_cmd_retval=$?

# Exit with failure if either command fails.
if [[ $first_bazel_cmd_retval -ne 0 ]]; then
  exit $first_bazel_cmd_retval
elif [[ $second_bazel_cmd_retval -ne 0 ]]; then
  exit $second_bazel_cmd_retval
elif [[ $third_bazel_cmd_retval -ne 0 ]]; then
  exit $third_bazel_cmd_retval
else
  exit 0
fi