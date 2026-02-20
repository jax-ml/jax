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
# Runs Pytest ROCm tests (with ROCm pytest-abort retry wrapper).
# Requires the jaxlib and ROCm plugin wheels to be present inside $JAXCI_OUTPUT_DIR (../dist)
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport

source ci/envs/default.env

# Install jaxlib and ROCm plugin wheels inside the $JAXCI_OUTPUT_DIR directory
echo "Installing wheels locally..."
source ./ci/utilities/install_wheels_locally.sh

# Print all the installed packages
echo "Installed packages:"
"$JAXCI_PYTHON" -m uv pip freeze

"$JAXCI_PYTHON" -c "import jax; print(jax.default_backend()); print(jax.devices()); print(len(jax.devices()))"

rocm-smi

# ==============================================================================
# Set up the generic test environment variables
# ==============================================================================
export PY_COLORS=1
export JAX_SKIP_SLOW_TESTS=true
export NCCL_DEBUG=WARN
export TF_CPP_MIN_LOG_LEVEL=0
export JAX_ENABLE_X64="$JAXCI_ENABLE_X64"

# ==============================================================================
# Calculate the optimal number of parallel processes for pytest
# This will be the minimum of: GPU capacity, CPU core count, and a system RAM limit.
# ==============================================================================

export gpu_count=$(rocminfo | egrep -c "Device Type:\\s+GPU")
echo "Number of GPUs detected: $gpu_count"

# Query GPU 0 memory using rocm-smi
export memory_per_gpu_mib=$(rocm-smi -d 0 --showmeminfo vram | grep -i "vram total" | awk '{print int($NF/1024/1024)}' | head -1)
echo "Reported memory per GPU: $memory_per_gpu_mib MiB"

# Convert effective memory from MiB to GiB.
export memory_per_gpu_gib=$((memory_per_gpu_mib / 1024))
echo "Effective memory per GPU: $memory_per_gpu_gib GiB"

# Allow 2 GiB of GPU RAM per test.
export max_tests_per_gpu=$((memory_per_gpu_gib / 2))
echo "Max tests per GPU (assuming 2GiB/test): $max_tests_per_gpu"

export num_processes=$((gpu_count * max_tests_per_gpu))
echo "Initial number of processes based on GPU capacity: $num_processes"

export num_cpu_cores=$(nproc)
echo "Number of CPU cores available: $num_cpu_cores"

# Reads total memory from /proc/meminfo (in KiB) and converts to GiB.
export total_ram_gib=$(awk '/MemTotal/ {printf \"%.0f\", $2/1048576}' /proc/meminfo)
echo "Total system RAM: $total_ram_gib GiB"

# Set a safety limit for system RAM usage, e.g., 1/6th of total.
export host_memory_limit=$((total_ram_gib / 6))
echo "Host memory process limit (1/6th of total RAM): $host_memory_limit"

if [[ $num_cpu_cores -lt $num_processes ]]; then
  num_processes=$num_cpu_cores
  echo "Adjusting num_processes to match CPU core count: $num_processes"
fi

if [[ $host_memory_limit -lt $num_processes ]]; then
  num_processes=$host_memory_limit
  echo "Adjusting num_processes to match host memory limit: $num_processes"
fi

if [[ 16 -lt $num_processes ]]; then
  num_processes=16
  echo "Reducing num_processes to $num_processes"
fi

echo "Final number of processes to run: $num_processes"

export JAX_ENABLE_ROCM_XDIST="$gpu_count"
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_command_buffer="

# Disable core dumps just in case
ulimit -c 0

# Keep deselected tests in one place for the abort wrapper.
ROCM_PYTEST_DESELECT_ARGS=(
  --deselect=tests/multi_device_test.py::MultiDeviceTest::test_computation_follows_data
  --deselect=tests/multiprocess_gpu_test.py::MultiProcessGpuTest::test_distributed_jax_visible_devices
  --deselect=tests/compilation_cache_test.py::CompilationCacheTest::test_task_using_cache_metric
)

# --max-runs: retry the entire pytest run up to N times on abort/crash.
# --max-worker-restart: restart crashed xdist workers up to N times.
# --maxfail: stop the run after N test failures.
rocm_test_cmd() {
  local abort_flag="${1:-0}"
  shift
  if [[ "$abort_flag" == "1" ]]; then
    pytest-abort-retry --max-runs 3 --clear-crash-log -- "$@"
  else
    "$@"
  fi
}

rocm_log_tail_on_failure() {
  local logfile="$1"
  local status="$2"
  if [[ "$status" -ne 0 ]]; then
    echo "Pytest failed (exit=$status). Showing last 200 lines of $logfile:"
    tail -n 200 "$logfile" || true
  else
    echo "Pytest output saved to $logfile (uploaded as artifact)."
  fi
}

rocm_install_extra_requirements() {
  if [[ -n "${GITHUB_WORKSPACE:-}" ]]; then
    cd "$GITHUB_WORKSPACE"
  fi

  # Install extra requirements.
  "$JAXCI_PYTHON" -m uv pip install pytest-timeout pytest-html pytest-csv pytest-json-report pytest-abort
}

rocm_install_extra_requirements

echo "Running ROCm tests (with abort/retry wrapper)..."
mkdir -p logs_abort
logfile="logs_abort/jax_ToT_UT_abort.log"

# pytest-abort output directories (must be set before running pytest).
export PYTEST_ABORT_LAST_RUNNING_DIR="logs_abort/last_running"
export PYTEST_ABORT_CRASHED_TESTS_LOG="logs_abort/crashed_tests.jsonl"
mkdir -p "$PYTEST_ABORT_LAST_RUNNING_DIR"

set +e
rocm_test_cmd 1 "$JAXCI_PYTHON" -m pytest -n "$num_processes" --max-worker-restart=200 --tb=short --timeout=1200 --timeout-method=thread tests \
  "${ROCM_PYTEST_DESELECT_ARGS[@]}" \
  --json-report \
  --json-report-file=logs_abort/tests-report-abort.json \
  --csv=logs_abort/tests-report-abort.csv \
  --html=logs_abort/tests-report-abort.html \
  --self-contained-html \
  >"$logfile" 2>&1
pytest_status=$?
set -e
rocm_log_tail_on_failure "$logfile" "$pytest_status"

echo "Postprocessing reports with crashed tests..."
pytest-abort-postprocess \
  --crash-log "$PYTEST_ABORT_CRASHED_TESTS_LOG" \
  --json-report logs_abort/tests-report-abort.json \
  --html-report logs_abort/tests-report-abort.html \
  --csv-report logs_abort/tests-report-abort.csv \
  >>"$logfile" 2>&1

exit "$pytest_status"
