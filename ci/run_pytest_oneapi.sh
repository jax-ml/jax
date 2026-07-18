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
# Runs Pytest ONEAPI tests. Requires the jaxlib, jax-oneapi-plugin, and jax-oneapi-pjrt
# wheels to be present inside $JAXCI_OUTPUT_DIR (../dist)
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport

# Source default JAXCI environment variables.
source ci/envs/default.env


# Install jaxlib, jax-oneapi-plugin, and jax-oneapi-pjrt wheels inside the
# $JAXCI_OUTPUT_DIR directory on the system.
echo "Installing wheels locally..."
source ./ci/utilities/install_wheels_locally.sh

# Install oneapi runtime dependencies from the locally installed plugin wheel.
# This avoids version pinning conflicts that occur when using jax[oneapi] extras.
oneapi_plugin_whl=$(ls "$JAXCI_OUTPUT_DIR"/*oneapi*plugin*.whl 2>/dev/null | head -1)
if [[ -n "$oneapi_plugin_whl" ]]; then
  "$JAXCI_PYTHON" -m uv pip install "${oneapi_plugin_whl}[with-oneapi]" --prerelease=allow
fi

# Test-time deps not provided by install_wheels_locally.sh. pytest-xdist is
# required for the `-n $num_processes` flag used below.
"$JAXCI_PYTHON" -m uv pip install pytest-xdist

# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

# Validate OneAPI runtime is available
if ! command -v xpu-smi &> /dev/null; then
  echo "ERROR: OneAPI runtime not detected (xpu-smi not found)"
  echo "Please ensure xpu-smi is installed"
  exit 1
fi

# Verify wheels exist
if ! ls "$JAXCI_OUTPUT_DIR"/*oneapi*.whl 1> /dev/null 2>&1; then
  echo "ERROR: No OneAPI wheels found in $JAXCI_OUTPUT_DIR"
  echo "Available files:"
  ls -la "$JAXCI_OUTPUT_DIR" || echo "(directory not accessible)"
  exit 1
fi

# Print all the installed packages
echo "Installed packages:"
"$JAXCI_PYTHON" -m uv pip freeze

"$JAXCI_PYTHON" -c "import jax; print(jax.default_backend()); print(jax.devices()); print(len(jax.devices()))"

# ==============================================================================
# Set up the generic test environment variables
# ==============================================================================
export PY_COLORS=1
export JAX_SKIP_SLOW_TESTS=true
export TF_CPP_MIN_LOG_LEVEL=0
export JAX_ENABLE_X64="$JAXCI_ENABLE_X64"


# TODO(Intel-tf): Revisit this.
# Some process-instance fails to find the tests/ direcotory in multi-process run.
# add a path to $pwd to PYTHONPATH to ensure that the tests/ directory is found.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# ==============================================================================
# Calculate the optimal number of parallel processes for pytest
# This will be the minimum of: GPU capacity, CPU core count, and a system RAM limit.
# ==============================================================================

# Detect OneAPI GPU count
detect_oneapi_gpus() {
  if command -v xpu-smi &> /dev/null; then
    local count=$(xpu-smi discovery 2>/dev/null | grep -c "Device Name")
    echo $count
  else
    echo 0
  fi
}

export gpu_count=$(detect_oneapi_gpus)
if [[ $gpu_count -eq 0 ]]; then
  echo "ERROR: No OneAPI GPUs detected, exiting"
  exit 1
fi
echo "Number of GPUs detected: $gpu_count"

# Calculate max tests per GPU based on available memory
export max_tests_per_gpu=8
if command -v xpu-smi &> /dev/null; then
  gpu_memory_gb=$(xpu-smi discovery --dump 16 | grep -v "Memory\|^$" | awk '{print int($1/1024+0.5)}' | head -1)
  gpu_memory_gb="${gpu_memory_gb:-16}"
  export max_tests_per_gpu=$((gpu_memory_gb / 2))
  [[ $max_tests_per_gpu -lt 4 ]] && export max_tests_per_gpu=4
  [[ $max_tests_per_gpu -gt 16 ]] && export max_tests_per_gpu=16
fi
echo "Max tests per GPU (assuming 2GiB/test): $max_tests_per_gpu"

export num_processes=$((gpu_count * max_tests_per_gpu))
echo "Initial number of processes based on GPU capacity: $num_processes"

export num_cpu_cores=$(nproc)
echo "Number of CPU cores available: $num_cpu_cores"

# Reads total memory from /proc/meminfo (in KiB) and converts to GiB.
export total_ram_gib=$(awk '/MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo)
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

echo "Final number of processes to run: $num_processes"

export JAX_ENABLE_ONEAPI_XDIST="$gpu_count"
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# ==============================================================================
# Run tests
# ==============================================================================
echo "Running oneAPI tests..."
mkdir -p test-artifacts
"$JAXCI_PYTHON" -m pytest -n $num_processes --tb=short --maxfail=20 \
--junitxml=test-artifacts/junit.xml \
--ignore=jax_plugins \
--ignore=tests/pallas \
--ignore=tests/mosaic \
tests examples \
--deselect=tests/multi_device_test.py::MultiDeviceTest::test_computation_follows_data \
--deselect=tests/multiprocess_gpu_test.py::MultiProcessGpuTest::test_distributed_jax_visible_devices \
--deselect=tests/compilation_cache_test.py::CompilationCacheTest::test_task_using_cache_metric
