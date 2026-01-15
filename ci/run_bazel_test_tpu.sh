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
# Runs Bazel TPU tests. If $JAXCI_BUILD_JAXLIB=false and $JAXCI_BUILD_JAX=false,
# the job requires that jax and jaxlib wheels are stored inside the ../dist
# folder.
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport

# Source default JAXCI environment variables.
source ci/envs/default.env

# Clone XLA at HEAD if path to local XLA is not provided
if [[ -z "$JAXCI_XLA_GIT_DIR" ]]; then
    export JAXCI_CLONE_MAIN_XLA=1
fi

# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

if [[ "$JAXCI_HERMETIC_PYTHON_VERSION" == *"-nogil" ]]; then
  JAXCI_HERMETIC_PYTHON_VERSION=${JAXCI_HERMETIC_PYTHON_VERSION%-nogil}-ft
  FREETHREADED_FLAG_VALUE="yes"
else
  FREETHREADED_FLAG_VALUE="no"
fi

OVERRIDE_XLA_REPO=""
if [[ "$JAXCI_CLONE_MAIN_XLA" == 1 ]]; then
  OVERRIDE_XLA_REPO="--override_repository=xla=${JAXCI_XLA_GIT_DIR}"
fi

NB_TPUS=$JAXCI_TPU_CORES
JOBS_PER_ACC=1
J=$((NB_TPUS * JOBS_PER_ACC))

# TODO(ybaturina): Bazel cache shouldn't be invalidated when
# `VBAR_CONTROL_SERVICE_URL` changes.
COMMON_TPU_TEST_ENV_VARS="--test_env=TPU_SKIP_MDS_QUERY=true \
 --test_env=TPU_TOPOLOGY \
 --test_env=TPU_WORKER_ID \
 --test_env=TPU_TOPOLOGY_WRAP \
 --test_env=TPU_CHIPS_PER_HOST_BOUNDS \
 --test_env=TPU_ACCELERATOR_TYPE \
 --test_env=TPU_RUNTIME_METRICS_PORTS \
 --test_env=TPU_TOPOLOGY_ALT \
 --test_env=TPU_HOST_BOUNDS \
 --test_env=TPU_WORKER_HOSTNAMES \
 --test_env=CHIPS_PER_HOST_BOUNDS \
 --test_env=HOST_BOUNDS \
 --test_env=VBAR_CONTROL_SERVICE_URL"

echo "Running Bazel TPU tests..."

# Don't abort the script if one command fails to ensure we run both test
# commands below.
set +e

# TODO(emilyaf): Debug and re-enable this test.
IGNORE_TESTS_MULTIACCELERATOR="-//tests/multiprocess:array_test_tpu"

if [[ "$JAXCI_RUN_FULL_TPU_TEST_SUITE" == "1" ]]; then
  # We're deselecting all Pallas TPU tests in the oldest libtpu build. Mosaic
  # TPU does not guarantee anything about forward compatibility (unless
  # jax.export is used) and the 12 week compatibility window accumulates way
  # too many failures.
  IGNORE_TESTS=""
  if [ "${libtpu_version_type:-""}" == "oldest_supported_libtpu" ]; then
    IGNORE_TESTS="-//tests/pallas/..."
  else
    IGNORE_TESTS="-//tests/pallas:tpu_pallas_interpret_thread_map_test_tpu"
  fi

  # Run single-accelerator tests in parallel
  bazel test \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
    --@rules_python//python/config_settings:py_freethreaded="$FREETHREADED_FLAG_VALUE" \
    $OVERRIDE_XLA_REPO \
    --config=ci_linux_x86_64 \
    --config=ci_rbe_cache \
    --//jax:build_jaxlib=$JAXCI_BUILD_JAXLIB \
    --//jax:build_jax=$JAXCI_BUILD_JAX \
    --run_under="$(pwd)/build/parallel_accelerator_execute.sh" \
    --test_env=JAX_ACCELERATOR_COUNT=${NB_TPUS} \
    --test_env=JAX_TESTS_PER_ACCELERATOR=${JOBS_PER_ACC} \
    --strategy=TestRunner=local \
    --local_test_jobs=$J \
    --test_env=JAX_TEST_NUM_THREADS=$J \
    --test_env=ALLOW_MULTIPLE_LIBTPU_LOAD=true \
    --test_env=JAX_SKIP_SLOW_TESTS=1 \
    --test_env=JAX_ENABLE_TPU_XDIST=1 \
    --test_env=JAX_PLATFORMS=tpu,cpu \
    --repo_env=USE_MINIMAL_SHARD_COUNT=True \
    $COMMON_TPU_TEST_ENV_VARS \
    --test_tag_filters=-multiaccelerator \
    --verbose_failures \
    --test_output=errors \
    -- \
    //tests:tpu_tests \
    //tests/pallas:tpu_tests \
    $IGNORE_TESTS

  # Store the return value of the first bazel command.
  first_bazel_cmd_retval=$?

  # Run multi-accelerator across all chips
  bazel test \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
    --@rules_python//python/config_settings:py_freethreaded="$FREETHREADED_FLAG_VALUE" \
    $OVERRIDE_XLA_REPO \
    --config=ci_linux_x86_64 \
    --config=ci_rbe_cache \
    --//jax:build_jaxlib=$JAXCI_BUILD_JAXLIB \
    --//jax:build_jax=$JAXCI_BUILD_JAXLIB \
    --test_env=ALLOW_MULTIPLE_LIBTPU_LOAD=true \
    --strategy=TestRunner=local \
    --local_test_jobs=1 \
    --repo_env=USE_MINIMAL_SHARD_COUNT=True \
    --test_env=JAX_SKIP_SLOW_TESTS=1 \
    --test_env=JAX_PLATFORMS=tpu,cpu \
    $COMMON_TPU_TEST_ENV_VARS \
    --test_tag_filters=multiaccelerator \
    --verbose_failures \
    --test_output=errors \
    -- \
    //tests:tpu_tests \
    //tests/pallas:tpu_tests \
    //tests/multiprocess:tpu_tests \
    $IGNORE_TESTS_MULTIACCELERATOR

  # Store the return value of the second bazel command.
  second_bazel_cmd_retval=$?
else

  # Run single-accelerator tests in parallel
  bazel test \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
    --@rules_python//python/config_settings:py_freethreaded="$FREETHREADED_FLAG_VALUE" \
    $OVERRIDE_XLA_REPO \
    --config=ci_linux_x86_64 \
    --config=ci_rbe_cache \
    --//jax:build_jaxlib=$JAXCI_BUILD_JAXLIB \
    --//jax:build_jax=$JAXCI_BUILD_JAXLIB \
    --run_under="$(pwd)/build/parallel_accelerator_execute.sh" \
    --test_env=JAX_ACCELERATOR_COUNT=${NB_TPUS} \
    --test_env=JAX_TESTS_PER_ACCELERATOR=${JOBS_PER_ACC} \
    --strategy=TestRunner=local \
    --local_test_jobs=$J \
    --test_env=JAX_TEST_NUM_THREADS=$J \
    --test_env=ALLOW_MULTIPLE_LIBTPU_LOAD=true \
    --test_env=JAX_SKIP_SLOW_TESTS=1 \
    --test_env=JAX_ENABLE_TPU_XDIST=1 \
    --test_env=JAX_PLATFORMS=tpu,cpu \
    --repo_env=USE_MINIMAL_SHARD_COUNT=True \
    $COMMON_TPU_TEST_ENV_VARS \
    --test_tag_filters=-multiaccelerator \
    --verbose_failures \
    --test_output=errors \
    -- \
    //jaxlib/tools:check_tpu_wheel_sources_test \
    //tests/pallas:ops_test_tpu \
    //tests/pallas:export_back_compat_pallas_test_tpu \
    //tests/pallas:export_pallas_test_tpu \
    //tests/pallas:tpu_ops_test_tpu \
    //tests/pallas:tpu_pallas_random_test_tpu \
    //tests/pallas:tpu_pallas_async_test_tpu \
    //tests/pallas:tpu_pallas_state_test_tpu \
    //tests/pallas:tpu_pallas_test_tpu \
    //tests/pallas:tpu_pallas_call_print_test_tpu \
    //tests/pallas:indexing_test_tpu \
    //tests/pallas:pallas_error_handling_test_tpu \
    //tests/pallas:pallas_shape_poly_test_tpu \
    //tests/pallas:tpu_all_gather_test_tpu \
    //tests/pallas:tpu_fusible_matmul_test_tpu \
    //tests/pallas:tpu_pallas_distributed_test_tpu \
    //tests/pallas:tpu_pallas_memory_space_test_tpu \
    //tests/pallas:tpu_splash_attention_kernel_sharded_test_tpu \
    //tests/pallas:tpu_sparsecore_pallas_test_tpu

  # Store the return value of the first bazel command.
  first_bazel_cmd_retval=$?

  # Run multi-accelerator across all chips
  bazel test \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
    --@rules_python//python/config_settings:py_freethreaded="$FREETHREADED_FLAG_VALUE" \
    $OVERRIDE_XLA_REPO \
    --config=ci_linux_x86_64 \
    --config=ci_rbe_cache \
    --//jax:build_jaxlib=$JAXCI_BUILD_JAXLIB \
    --//jax:build_jax=$JAXCI_BUILD_JAXLIB \
    --test_env=ALLOW_MULTIPLE_LIBTPU_LOAD=true \
    --strategy=TestRunner=local \
    --local_test_jobs=1 \
    --test_env=JAX_ACCELERATOR_COUNT=${NB_TPUS} \
    --repo_env=USE_MINIMAL_SHARD_COUNT=True \
    --test_env=JAX_SKIP_SLOW_TESTS=1 \
    --test_env=JAX_PLATFORMS=tpu,cpu \
    $COMMON_TPU_TEST_ENV_VARS \
    --test_tag_filters=multiaccelerator \
    --verbose_failures \
    --test_output=errors \
    -- \
    //tests:aot_test_tpu \
    //tests:array_test_tpu \
    //tests:jaxpr_effects_test_tpu \
    //tests:layout_test_tpu \
    //tests:pjit_test_tpu \
    //tests:python_callback_test_tpu \
    //tests:ragged_collective_test_tpu

  # Store the return value of the second bazel command.
  second_bazel_cmd_retval=$?
fi

# Exit with failure if either command fails.
if [[ $first_bazel_cmd_retval -ne 0 ]]; then
  exit $first_bazel_cmd_retval
elif [[ $second_bazel_cmd_retval -ne 0 ]]; then
  exit $second_bazel_cmd_retval
else
  exit 0
fi
