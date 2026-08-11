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
# Runs the multi-accelerator ROCm tests, the only subset that needs more than
# one GPU. Requires the jaxlib and ROCm plugin wheels to be present inside
# $JAXCI_OUTPUT_DIR (../dist)
#
# ci/run_pytest_rocm.sh calls this for the second half of a full run, and CI
# that puts this subset on a multi-GPU runner calls it on its own.
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport

source ./ci/utilities/prepare_rocm_tests.sh

# Running this script is a request for these tests, so a host that cannot run
# them fails instead of reporting success without having run anything.
if [[ $gpu_count -le 1 ]]; then
  echo "Multi-accelerator tests need more than one GPU, found $gpu_count" >&2
  exit 1
fi

echo "Running multi-accelerator ROCm tests..."

# Run multi-accelerator tests across all GPUs without xdist.
unset JAX_ENABLE_ROCM_XDIST

"$JAXCI_PYTHON" -m pytest --tb=short \
--json-report --json-report-file=${LOGS_DIR}/pytest_results_multi.json \
--junitxml=test-artifacts/junit-multi.xml \
-m "multiaccelerator" \
$rocm_pytest_deselect \
tests
