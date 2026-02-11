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
# Runs Pytest ROCm tests. Requires the jaxlib and ROCm plugin wheels to be
# present inside $JAXCI_OUTPUT_DIR (../dist)
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport

# Source default JAXCI environment variables.
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

export gpu_count=$(rocminfo | egrep -c "Device Type:\s+GPU")
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

if [[ 16 -lt $num_processes ]]; then
  num_processes=16
  echo "Reducing num_processes to $num_processes"
fi

echo "Final number of processes to run: $num_processes"

export JAX_ENABLE_CUDA_XDIST="$gpu_count"
export JAX_ENABLE_ROCM_XDIST="$gpu_count"
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_command_buffer="

# ==============================================================================
# Run tests
# ==============================================================================

# Disable core dumps just in case
ulimit -c 0

echo "Running ROCm tests..."
# TODO: Add examples directory to test suite (CUDA tests both: tests examples)
# TODO: Verify if CSV/HTML report generation should be kept (unique to ROCm)
# TODO: Verify if log file output should be kept (unique to ROCm)
export NPROC=32
"$JAXCI_PYTHON" -m pytest -n $num_processes --tb=short \
tests \
--deselect=tests/multi_device_test.py::MultiDeviceTest::test_computation_follows_data \
--deselect=tests/multiprocess_gpu_test.py::MultiProcessGpuTest::test_distributed_jax_visible_devices \
--deselect=tests/compilation_cache_test.py::CompilationCacheTest::test_task_using_cache_metric
