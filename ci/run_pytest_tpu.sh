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

"$JAXCI_PYTHON" -c "import jax; print(jax.default_backend()); print(jax.devices()); print(len(jax.devices()))"
"$JAXCI_PYTHON" -c 'import sys; print("python version:", sys.version)'
"$JAXCI_PYTHON" -c 'import jax; print("jax version:", jax.__version__)'
"$JAXCI_PYTHON" -c 'import jaxlib; print("jaxlib version:", jaxlib.__version__)'
# Free-threaded builds use "-nogil" as the suffix for the binary and "t" for its
# dist-packages path
strings /usr/local/lib/"${JAXCI_PYTHON//-nogil/t}"/dist-packages/libtpu/libtpu.so | grep 'Built on'
"$JAXCI_PYTHON" -c 'import jax.extend; print("libtpu version:",jax.extend.backend.get_backend().platform_version)'

# Set up all common test environment variables
export PY_COLORS=1
export JAX_PLATFORMS=tpu,cpu
export JAX_SKIP_SLOW_TESTS=true
# End of common test environment variable setup

case "${JAXCI_TPU_PARALLELISM_MODE:-chip}" in
  core)
    TPU_XDIST_VISIBILITY_MODE="devices"
    ;;
  chip)
    TPU_XDIST_VISIBILITY_MODE="chips"
    ;;
  *)
    echo "Unknown JAXCI_TPU_PARALLELISM_MODE: ${JAXCI_TPU_PARALLELISM_MODE}"
    exit 1
    ;;
esac

echo "Running TPU tests..."
mkdir -p test-artifacts
source ci/utilities/tpu_visibility_smoke_check.sh
run_tpu_visibility_smoke_check \
  "$TPU_XDIST_VISIBILITY_MODE" "$JAXCI_TPU_CORES" "$JAXCI_PYTHON"

# Don't abort the script if one command fails to ensure we run both test
# commands below.
set +e

if [[ "$JAXCI_RUN_FULL_TPU_TEST_SUITE" == "1" ]]; then
  # We're deselecting all Pallas TPU tests in the oldest libtpu build. Mosaic
  # TPU does not guarantee anything about forward compatibility (unless
  # jax.export is used) and the 12 week compatibility window accumulates way
  # too many failures.
  IGNORE_FLAGS=""
  if [ "${libtpu_version_type:-""}" == "oldest_supported_libtpu" ]; then
    IGNORE_FLAGS="--ignore=tests/pallas"
  fi

  if [[ "$TPU_XDIST_VISIBILITY_MODE" == "devices" ]]; then
    # TODO: Restore the full TPU pytest suite before merging the
    # TPU7x core-splitting experiment. The two narrowed Pallas files are
    # currently noisy, so the core-split signal comes from the smoke check above.
    echo "Skipping narrowed Pallas pytest targets for TPU core-splitting diagnostic."
    first_cmd_retval=0
  else
    # TODO: Restore the full TPU pytest suite before merging the
    # TPU7x core-splitting experiment. This is temporarily narrowed to files seen
    # in the last scheduled continuous TPU7x run to keep diagnostic logs readable.
    JAX_ENABLE_TPU_XDIST=true JAX_TPU_XDIST_DEBUG=1 \
      JAX_TPU_XDIST_VISIBILITY_MODE="$TPU_XDIST_VISIBILITY_MODE" \
      "$JAXCI_PYTHON" -m pytest -n="$JAXCI_TPU_CORES" --tb=short \
      --junitxml=test-artifacts/junit-single.xml \
      --deselect=tests/pallas/tpu_pallas_call_print_test.py::PallasCallPrintTest \
      --deselect=tests/pallas/tpu_sparsecore_pallas_test.py::DebugPrintTest \
      --deselect=tests/pallas/tpu_pallas_interpret_thread_map_test.py::InterpretThreadMapTest::test_thread_map \
      --dist=loadfile --maxfail=5 -m "not multiaccelerator" $IGNORE_FLAGS \
      tests/pallas/tpu_pallas_test.py \
      tests/pallas/ops_test.py

    # Store the return value of the first command.
    first_cmd_retval=$?
  fi

  # TODO: Re-enable multi-accelerator pytest before merging. It is
  # disabled while debugging TPU7x single-worker visibility so unrelated
  # multi-accelerator failures do not muddy the logs.
  second_cmd_retval=0
else
  # Run single-accelerator tests in parallel
  JAX_ENABLE_TPU_XDIST=true JAX_TPU_XDIST_DEBUG=1 \
    JAX_TPU_XDIST_VISIBILITY_MODE="$TPU_XDIST_VISIBILITY_MODE" \
    "$JAXCI_PYTHON" -m pytest -n="$JAXCI_TPU_CORES" --tb=short \
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

  # TODO: Re-enable multi-accelerator pytest before merging. It is
  # disabled while debugging TPU7x single-worker visibility so unrelated
  # multi-accelerator failures do not muddy the logs.
  second_cmd_retval=0
fi

# Run Pallas printing tests, which need to run with I/O capturing disabled.
TPU_STDERR_LOG_LEVEL=0 "$JAXCI_PYTHON" -m pytest \
  --junitxml=test-artifacts/junit-print.xml \
  -s tests/pallas/tpu_pallas_call_print_test.py::PallasCallPrintTest \
  -s tests/pallas/tpu_sparsecore_pallas_test.py::DebugPrintTest

# Store the return value of the third command.
third_cmd_retval=$?

# Exit with failure if either command fails.
if [[ $first_cmd_retval -ne 0 ]]; then
  exit $first_cmd_retval
elif [[ $second_cmd_retval -ne 0 ]]; then
  exit $second_cmd_retval
elif [[ $third_cmd_retval -ne 0 ]]; then
  exit $third_cmd_retval
else
  exit 0
fi
