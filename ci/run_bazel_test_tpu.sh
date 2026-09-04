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

echo "::group::Setup Environment" >&2
# Source default JAXCI environment variables.
source ci/envs/default.env

# Clone XLA at HEAD if path to local XLA is not provided
if [[ -z "$JAXCI_XLA_GIT_DIR" && -z "$JAXCI_CLONE_MAIN_XLA" ]]; then
    export JAXCI_CLONE_MAIN_XLA=1
fi

# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

if [[ "$JAXCI_HERMETIC_PYTHON_VERSION" == *t || "$JAXCI_HERMETIC_PYTHON_VERSION" == *-ft || "$JAXCI_HERMETIC_PYTHON_VERSION" == *-nogil ]]; then
  JAXCI_HERMETIC_PYTHON_VERSION=${JAXCI_HERMETIC_PYTHON_VERSION%t}
  JAXCI_HERMETIC_PYTHON_VERSION=${JAXCI_HERMETIC_PYTHON_VERSION%-ft}
  JAXCI_HERMETIC_PYTHON_VERSION=${JAXCI_HERMETIC_PYTHON_VERSION%-nogil}
  FREETHREADED_FLAG_VALUE="yes"
else
  FREETHREADED_FLAG_VALUE="no"
fi

OVERRIDE_XLA_REPO=""
if [[ "$JAXCI_CLONE_MAIN_XLA" == 1 ]]; then
  OVERRIDE_XLA_REPO="--override_repository=xla=${JAXCI_XLA_GIT_DIR} --override_module=xla=${JAXCI_XLA_GIT_DIR}"
fi

NB_TPUS=$JAXCI_TPU_CORES
JOBS_PER_ACC=1
J=$((NB_TPUS * JOBS_PER_ACC))

# TODO(ybaturina): Bazel cache shouldn't be invalidated when
# `VBAR_CONTROL_SERVICE_URL` changes.
COMMON_TPU_TEST_ENV_VARS="--test_env=PORTSERVER_ADDRESS=@unittest-portserver \
 --test_env=TPU_SKIP_MDS_QUERY=true \
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

# Only TPU v6e runners, used for presubmits, are configured with enough /dev/shm space for Bazel output_base.
if [[ "$JAXCI_RUN_FULL_TPU_TEST_SUITE" != "1" ]]; then
  BAZEL_STARTUP_ARGS=("--output_base=/dev/shm/bazel")
else
  BAZEL_STARTUP_ARGS=()
fi

echo "Running Bazel TPU tests..."

# Don't abort the script if one command fails to ensure we run all test
# commands below.
set +e

# TODO(emilyaf): Debug and re-enable this test.
IGNORE_TESTS_MULTIPROCESS="-//tests/multiprocess:array_test_tpu"

multiprocess_bazel_cmd_retval=0

echo "::endgroup::" >&2

PYTHON_BIN="$JAXCI_PYTHON" source ci/utilities/setup_portserver.sh

if [[ "$JAXCI_RUN_FULL_TPU_TEST_SUITE" == "1" ]]; then
  IGNORE_TESTS="-//tests/pallas:tpu_pallas_interpret_thread_map_test_tpu"

  # Run single-accelerator tests in parallel
  TEST_ARTIFACTS_DIR="${JAXCI_TEST_ARTIFACT_DIR}-single"
  mkdir -p "$TEST_ARTIFACTS_DIR"

  echo "::group::Bazel TPU single-accelerator tests (full)" >&2
  INVOCATION_ID_SINGLE=$(python3 ci/utilities/generate_invocation_id.py)

  bazel "${BAZEL_STARTUP_ARGS[@]}" test \
    --invocation_id="$INVOCATION_ID_SINGLE" \
    --profile="$TEST_ARTIFACTS_DIR/bazel_profile.json.gz" \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
    $OVERRIDE_XLA_REPO \
    --@rules_python//python/config_settings:py_freethreaded="$FREETHREADED_FLAG_VALUE" \
    --config=ci_linux_x86_64 \
    --config=ci_rbe_cache \
    --//jax:build_jaxlib=$JAXCI_BUILD_JAXLIB \
    --//jax:build_jax=$JAXCI_BUILD_JAX \
    --run_under="$(pwd)/build/parallel_accelerator_execute.sh" \
    --test_env=JAX_ACCELERATOR_COUNT=${NB_TPUS} \
    --test_env=JAX_TESTS_PER_ACCELERATOR=${JOBS_PER_ACC} \
    --strategy=TestRunner=local \
    --local_test_jobs=$J \
    --test_env=JAX_TEST_NUM_THREADS=32 \
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
  echo "::endgroup::" >&2
  python3 ci/utilities/report_resultstore_link.py "TPU single-accelerator tests (full)" "$INVOCATION_ID_SINGLE" "${first_bazel_cmd_retval:-0}"
  ci/utilities/collect_bazel_test_xmls.sh "$TEST_ARTIFACTS_DIR"

  # Run non-multiprocess multi-accelerator tests across all chips
  TEST_ARTIFACTS_DIR="${JAXCI_TEST_ARTIFACT_DIR}-multi"
  mkdir -p "$TEST_ARTIFACTS_DIR"

  echo "::group::Bazel TPU multi-accelerator tests (full)" >&2
  INVOCATION_ID_MULTI=$(python3 ci/utilities/generate_invocation_id.py)

  bazel "${BAZEL_STARTUP_ARGS[@]}" test \
    --invocation_id="$INVOCATION_ID_MULTI" \
    --profile="$TEST_ARTIFACTS_DIR/bazel_profile.json.gz" \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
    $OVERRIDE_XLA_REPO \
    --@rules_python//python/config_settings:py_freethreaded="$FREETHREADED_FLAG_VALUE" \
    --config=ci_linux_x86_64 \
    --config=ci_rbe_cache \
    --//jax:build_jaxlib=$JAXCI_BUILD_JAXLIB \
    --//jax:build_jax=$JAXCI_BUILD_JAXLIB \
    --test_env=ALLOW_MULTIPLE_LIBTPU_LOAD=true \
    --strategy=TestRunner=local \
    --local_test_jobs=1 \
    --repo_env=USE_MINIMAL_SHARD_COUNT=True \
    --test_env=JAX_SKIP_SLOW_TESTS=1 \
    --test_env=JAX_TEST_NUM_THREADS=32 \
    --test_env=JAX_PLATFORMS=tpu,cpu \
    $COMMON_TPU_TEST_ENV_VARS \
    --test_tag_filters=multiaccelerator \
    --verbose_failures \
    --test_output=errors \
    -- \
    //tests:tpu_tests \
    //tests/pallas:tpu_tests

  # Store the return value of the second bazel command.
  second_bazel_cmd_retval=$?
  echo "::endgroup::" >&2
  python3 ci/utilities/report_resultstore_link.py "TPU multi-accelerator tests (full)" "$INVOCATION_ID_MULTI" "${second_bazel_cmd_retval:-0}"
  ci/utilities/collect_bazel_test_xmls.sh "$TEST_ARTIFACTS_DIR"

  # Run multiprocess targets one at a time. Their workers must execute each test
  # together, so disable test-level threading.
  TEST_ARTIFACTS_DIR="${JAXCI_TEST_ARTIFACT_DIR}-multiprocess"
  mkdir -p "$TEST_ARTIFACTS_DIR"

  echo "::group::Bazel TPU multiprocess tests (full)" >&2
  INVOCATION_ID_MULTIPROCESS=$(python3 ci/utilities/generate_invocation_id.py)

  bazel "${BAZEL_STARTUP_ARGS[@]}" test \
    --invocation_id="$INVOCATION_ID_MULTIPROCESS" \
    --profile="$TEST_ARTIFACTS_DIR/bazel_profile.json.gz" \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
    $OVERRIDE_XLA_REPO \
    --@rules_python//python/config_settings:py_freethreaded="$FREETHREADED_FLAG_VALUE" \
    --config=ci_linux_x86_64 \
    --config=ci_rbe_cache \
    --//jax:build_jaxlib=$JAXCI_BUILD_JAXLIB \
    --//jax:build_jax=$JAXCI_BUILD_JAXLIB \
    --test_env=ALLOW_MULTIPLE_LIBTPU_LOAD=true \
    --strategy=TestRunner=local \
    --local_test_jobs=1 \
    --repo_env=USE_MINIMAL_SHARD_COUNT=True \
    --test_env=JAX_SKIP_SLOW_TESTS=1 \
    --test_env=JAX_TEST_NUM_THREADS=0 \
    --test_env=JAX_PLATFORMS=tpu,cpu \
    $COMMON_TPU_TEST_ENV_VARS \
    --test_tag_filters=multiaccelerator \
    --verbose_failures \
    --test_output=errors \
    -- \
    //tests/multiprocess:tpu_tests \
    $IGNORE_TESTS_MULTIPROCESS

  multiprocess_bazel_cmd_retval=$?
  echo "::endgroup::" >&2
  python3 ci/utilities/report_resultstore_link.py "TPU multiprocess tests (full)" "$INVOCATION_ID_MULTIPROCESS" "${multiprocess_bazel_cmd_retval:-0}"
  ci/utilities/collect_bazel_test_xmls.sh "$TEST_ARTIFACTS_DIR"
else

  # Run single-accelerator tests in parallel
  TEST_ARTIFACTS_DIR="${JAXCI_TEST_ARTIFACT_DIR}-single"
  mkdir -p "$TEST_ARTIFACTS_DIR"

  echo "::group::Bazel TPU single-accelerator tests" >&2
  INVOCATION_ID_SINGLE=$(python3 ci/utilities/generate_invocation_id.py)

  bazel "${BAZEL_STARTUP_ARGS[@]}" test \
    --invocation_id="$INVOCATION_ID_SINGLE" \
    --profile="$TEST_ARTIFACTS_DIR/bazel_profile.json.gz" \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
    $OVERRIDE_XLA_REPO \
    --@rules_python//python/config_settings:py_freethreaded="$FREETHREADED_FLAG_VALUE" \
    --config=ci_linux_x86_64 \
    --config=ci_rbe_cache \
    --//jax:build_jaxlib=$JAXCI_BUILD_JAXLIB \
    --//jax:build_jax=$JAXCI_BUILD_JAXLIB \
    --run_under="$(pwd)/build/parallel_accelerator_execute.sh" \
    --test_env=JAX_ACCELERATOR_COUNT=${NB_TPUS} \
    --test_env=JAX_TESTS_PER_ACCELERATOR=${JOBS_PER_ACC} \
    --strategy=TestRunner=local \
    --local_test_jobs=$J \
    --test_env=JAX_TEST_NUM_THREADS=32 \
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
    //tests:tpu_tests \
    //tests/pallas:ops_test_tpu \
    //tests/pallas:export_back_compat_pallas_test_tpu \
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
  echo "::endgroup::" >&2
  python3 ci/utilities/report_resultstore_link.py "TPU single-accelerator tests" "$INVOCATION_ID_SINGLE" "${first_bazel_cmd_retval:-0}"
  ci/utilities/collect_bazel_test_xmls.sh "$TEST_ARTIFACTS_DIR"

  # Run multi-accelerator across all chips
  TEST_ARTIFACTS_DIR="${JAXCI_TEST_ARTIFACT_DIR}-multi"
  mkdir -p "$TEST_ARTIFACTS_DIR"

  echo "::group::Bazel TPU multi-accelerator tests" >&2
  INVOCATION_ID_MULTI=$(python3 ci/utilities/generate_invocation_id.py)

  bazel "${BAZEL_STARTUP_ARGS[@]}" test \
    --invocation_id="$INVOCATION_ID_MULTI" \
    --profile="$TEST_ARTIFACTS_DIR/bazel_profile.json.gz" \
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
    --test_env=JAX_TEST_NUM_THREADS=32 \
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
  echo "::endgroup::" >&2
  python3 ci/utilities/report_resultstore_link.py "TPU multi-accelerator tests" "$INVOCATION_ID_MULTI" "${second_bazel_cmd_retval:-0}"
  ci/utilities/collect_bazel_test_xmls.sh "$TEST_ARTIFACTS_DIR"
fi

echo "::group::Cleanup" >&2
# Merge results with prefixes to avoid overwriting
{ set +x; } 2>/dev/null
mkdir -p "$JAXCI_TEST_ARTIFACT_DIR"
if [[ -d "${JAXCI_TEST_ARTIFACT_DIR}-single" ]]; then
  for f in "${JAXCI_TEST_ARTIFACT_DIR}-single"/*; do
    [[ -e "$f" ]] || continue
    cp "$f" "${JAXCI_TEST_ARTIFACT_DIR}/single_$(basename "$f")"
  done
fi
if [[ -d "${JAXCI_TEST_ARTIFACT_DIR}-multi" ]]; then
  for f in "${JAXCI_TEST_ARTIFACT_DIR}-multi"/*; do
    [[ -e "$f" ]] || continue
    cp "$f" "${JAXCI_TEST_ARTIFACT_DIR}/multi_$(basename "$f")"
  done
fi
if [[ -d "${JAXCI_TEST_ARTIFACT_DIR}-multiprocess" ]]; then
  for f in "${JAXCI_TEST_ARTIFACT_DIR}-multiprocess"/*; do
    [[ -e "$f" ]] || continue
    cp "$f" "${JAXCI_TEST_ARTIFACT_DIR}/multiprocess_$(basename "$f")"
  done
fi
set -x
echo "::endgroup::" >&2

# Exit with failure if any command fails.
if [[ $first_bazel_cmd_retval -ne 0 ]]; then
  exit $first_bazel_cmd_retval
elif [[ $second_bazel_cmd_retval -ne 0 ]]; then
  exit $second_bazel_cmd_retval
elif [[ $multiprocess_bazel_cmd_retval -ne 0 ]]; then
  exit $multiprocess_bazel_cmd_retval
else
  exit 0
fi
