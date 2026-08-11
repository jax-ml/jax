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
# Everything a ROCm pytest run needs before it can call pytest: the JAXCI
# defaults, the wheels under test, and the shared test environment. Sourced by
# ci/run_pytest_rocm.sh and by ci/run_pytest_rocm_multi.sh, either of which may
# be the one a caller starts, so doing this twice in a run is a no-op.

if [[ -n "${rocm_tests_prepared:-}" ]]; then
  return 0
fi
rocm_tests_prepared=1

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

source ./ci/utilities/rocm_test_env.sh

# Disable core dumps just in case
ulimit -c 0

LOGS_DIR="logs"
mkdir -p "${LOGS_DIR}"
mkdir -p test-artifacts
