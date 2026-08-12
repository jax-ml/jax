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
#
# The environment the ROCm pytest runs share. Sourced by
# ci/utilities/prepare_rocm_tests.sh, and on its own by anything that needs to
# reproduce the environment of a run without starting one. Requires
# ci/envs/default.env to have been sourced already.

# ==============================================================================
# Set up the generic test environment variables
# ==============================================================================
export PY_COLORS=1
export JAX_SKIP_SLOW_TESTS=true
export NCCL_DEBUG=WARN
export TF_CPP_MIN_LOG_LEVEL=0
export JAX_ENABLE_X64="$JAXCI_ENABLE_X64"

# ==============================================================================
# Number of parallel processes for pytest: 4 test workers per GPU.
# ==============================================================================

export gpu_count=$(rocminfo | egrep -c "Device Type:\s+GPU")
echo "Number of GPUs detected: $gpu_count"

export num_processes=$((gpu_count * 4))
echo "Number of processes to run: $num_processes"

export JAX_ENABLE_ROCM_XDIST="$gpu_count"
export XLA_PYTHON_CLIENT_ALLOCATOR=address
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_command_buffer="

# Deselected by both subsets, listed here so the two runs cannot drift apart.
# Expanded unquoted by the callers, which is why no entry may contain a space.
export rocm_pytest_deselect="\
--deselect=tests/multi_device_test.py::MultiDeviceTest::test_computation_follows_data \
--deselect=tests/multiprocess_gpu_test.py::MultiProcessGpuTest::test_distributed_jax_visible_devices \
--deselect=tests/compilation_cache_test.py::CompilationCacheTest::test_task_using_cache_metric"
