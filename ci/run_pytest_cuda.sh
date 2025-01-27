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
# Runs Pyest CUDA tests. Requires the jaxlib, jax-cuda-plugin, and jax-cuda-pjrt
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

# Install jaxlib, jax-cuda-plugin, and jax-cuda-pjrt wheels inside the
# $JAXCI_OUTPUT_DIR directory on the system.
echo "Installing wheels locally..."
source ./ci/utilities/install_wheels_locally.sh

# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

"$JAXCI_PYTHON" -c "import jax; print(jax.default_backend()); print(jax.devices()); print(len(jax.devices()))"

nvidia-smi

# Set up all test environment variables
export PY_COLORS=1
export JAX_SKIP_SLOW_TESTS=true
export NCCL_DEBUG=WARN
export TF_CPP_MIN_LOG_LEVEL=0
export JAX_ENABLE_64="$JAXCI_ENABLE_X64"

# Set the number of processes to run to be 4x the number of GPUs.
export gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export num_processes=`expr 4 \* $gpu_count`

export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1
# End of test environment variable setup

echo "Running CUDA tests..."
"$JAXCI_PYTHON" -m pytest -n $num_processes --tb=short --maxfail=20 \
tests examples \
--deselect=tests/multi_device_test.py::MultiDeviceTest::test_computation_follows_data \
--deselect=tests/multiprocess_gpu_test.py::MultiProcessGpuTest::test_distributed_jax_visible_devices \
--deselect=tests/compilation_cache_test.py::CompilationCacheTest::test_task_using_cache_metric
