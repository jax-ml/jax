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
# Runs Pyest CPU tests. Requires a jaxlib wheel to be present
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

# Install jaxlib wheel inside the $JAXCI_OUTPUT_DIR directory on the system.
echo "Installing wheels locally..."
source ./ci/utilities/install_wheels_locally.sh

# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

# Print all the installed packages
echo "Installed packages:"
"$JAXCI_PYTHON" -m uv pip freeze

"$JAXCI_PYTHON" -c 'import sys; print("python version:", sys.version)'
"$JAXCI_PYTHON" -c 'import jax; print("jax version:", jax.__version__)'
"$JAXCI_PYTHON" -c 'import jaxlib; print("jaxlib version:", jaxlib.__version__)'
if [[ -z "${JAXCI_TPU_PYTEST_PROBE_NODE:-}" ]]; then
  "$JAXCI_PYTHON" -c "import jax; print(jax.default_backend()); print(jax.devices()); print(len(jax.devices()))"
  "$JAXCI_PYTHON" -c 'import jax.extend; print("libtpu version:",jax.extend.backend.get_backend().platform_version)'
fi

# Set up all common test environment variables
export PY_COLORS=1
export JAX_PLATFORMS=tpu,cpu
export JAX_SKIP_SLOW_TESTS=true
# End of common test environment variable setup

echo "Running TPU tests..."
mkdir -p test-artifacts

# Don't abort the script if one command fails to ensure we run both test
# commands below.
set +e

first_cmd_retval=0
second_cmd_retval=0

run_tpu_health_probe() {
  local phase="$1"
  local run_number="$2"
  local log_file="test-artifacts/health-${run_number}-${phase}.log"

  echo "::group::TPU health probe ${run_number} (${phase})" >&2
  timeout --signal=TERM --kill-after=30s 120s "$JAXCI_PYTHON" -c '
import jax

devices = jax.devices()
value = jax.random.normal(jax.random.key(0), (1024, 1024))
value.block_until_ready()
print("devices:", devices)
print("random normal shape:", value.shape)
' 2>&1 | tee "$log_file"
  local probe_status=${PIPESTATUS[0]}
  echo "::endgroup::" >&2
  return "$probe_status"
}

if [[ -n "${JAXCI_TPU_PYTEST_PROBE_NODE:-}" ]]; then
  if ! [[ "$JAXCI_TPU_PYTEST_PROBE_RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid TPU pytest probe run count: $JAXCI_TPU_PYTEST_PROBE_RUNS" >&2
    exit 2
  fi

  printf 'run\tpre_health\tpytest\tpost_health\n' > test-artifacts/probe-status.tsv

  for ((probe_run = 1; probe_run <= JAXCI_TPU_PYTEST_PROBE_RUNS; probe_run++)); do
    run_tpu_health_probe pre "$probe_run"
    pre_health_status=$?

    echo "::group::Pytest TPU focused probe ${probe_run}: $JAXCI_TPU_PYTEST_PROBE_NODE" >&2
    JAX_ENABLE_TPU_XDIST=true timeout --signal=TERM --kill-after=30s 300s \
      "$JAXCI_PYTHON" -m pytest -n="$JAXCI_TPU_CORES" --dist=loadfile \
      --tb=short --maxfail=1 -vv \
      --junitxml="test-artifacts/junit-probe-${probe_run}.xml" \
      "$JAXCI_TPU_PYTEST_PROBE_NODE"
    pytest_status=$?
    echo "::endgroup::" >&2

    run_tpu_health_probe post "$probe_run"
    post_health_status=$?

    printf '%s\t%s\t%s\t%s\n' \
      "$probe_run" "$pre_health_status" "$pytest_status" "$post_health_status" \
      >> test-artifacts/probe-status.tsv

    for status in "$pre_health_status" "$pytest_status" "$post_health_status"; do
      if [[ $first_cmd_retval -eq 0 && $status -ne 0 ]]; then
        first_cmd_retval=$status
      fi
    done
  done
elif [[ "$JAXCI_RUN_FULL_TPU_TEST_SUITE" == "1" ]]; then
  # Run single-accelerator tests in parallel
  JAX_ENABLE_TPU_XDIST=true "$JAXCI_PYTHON" -m pytest -n="$JAXCI_TPU_CORES" --tb=short \
    --junitxml=test-artifacts/junit-single.xml \
    --deselect=tests/pallas/tpu_pallas_interpret_thread_map_test.py::InterpretThreadMapTest::test_thread_map \
    --dist=loadfile --maxfail=20 -m "not multiaccelerator" tests examples

  # Store the return value of the first command.
  first_cmd_retval=$?

  # Run multi-accelerator across all chips
  "$JAXCI_PYTHON" -m pytest --tb=short --maxfail=20 \
    --junitxml=test-artifacts/junit-multi.xml \
    -m "multiaccelerator" tests

  # Store the return value of the second command.
  second_cmd_retval=$?
else
  # Run single-accelerator tests in parallel
  JAX_ENABLE_TPU_XDIST=true "$JAXCI_PYTHON" -m pytest -n="$JAXCI_TPU_CORES" --tb=short \
    --junitxml=test-artifacts/junit-single.xml \
    --maxfail=20 -m "not multiaccelerator" \
    tests/pallas/ops_test.py \
    tests/pallas/export_back_compat_pallas_test.py \
    tests/pallas/export_pallas_test.py \
    tests/pallas/tpu_ops_test.py \
    tests/pallas/tpu_pallas_test.py \
    tests/pallas/tpu_pallas_random_test.py \
    tests/pallas/tpu_pallas_async_test.py \
    tests/pallas/tpu_pallas_state_test.py

  # Store the return value of the first command.
  first_cmd_retval=$?

  # Run multi-accelerator across all chips
  "$JAXCI_PYTHON" -m pytest --tb=short --maxfail=20 \
    --junitxml=test-artifacts/junit-multi.xml \
    -m "multiaccelerator" \
    tests/pjit_test.py \
    tests/pallas/tpu_pallas_distributed_test.py

  # Store the return value of the second command.
  second_cmd_retval=$?
fi

# Exit with failure if either command fails.
if [[ $first_cmd_retval -ne 0 ]]; then
  exit $first_cmd_retval
elif [[ $second_cmd_retval -ne 0 ]]; then
  exit $second_cmd_retval
else
  exit 0
fi
