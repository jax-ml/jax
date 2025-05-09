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
# Runs Bazel CPU tests with py_import on RBE.
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport

# Source default JAXCI environment variables.
source ci/envs/default.env

# Clone XLA at HEAD if path to local XLA is not provided
if [[ -z "$JAXCI_XLA_GIT_DIR" ]]; then
    export JAXCI_CLONE_MAIN_XLA=1
fi

# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

# Run Bazel CPU tests with RBE.
os=$(uname -s | awk '{print tolower($0)}')
arch=$(uname -m)

echo "Running CPU tests..."
# When running on Mac or Linux Aarch64, we build the test targets on RBE
# and run the tests locally. These platforms do not have native RBE support so
# we RBE cross-compile them on remote Linux x86 machines.
if [[ $os == "darwin" ]] || ( [[ $os == "linux" ]] && [[ $arch == "aarch64" ]] ); then
      bazel test --config=rbe_cross_compile_${os}_${arch} \
            --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
            --override_repository=xla="${JAXCI_XLA_GIT_DIR}" \
            --test_env=JAX_NUM_GENERATED_CASES=25 \
            --test_env=JAX_SKIP_SLOW_TESTS=true \
            --action_env=JAX_ENABLE_X64="$JAXCI_ENABLE_X64" \
            --test_output=errors \
            --color=yes \
            --test_timeout=500 \
            --strategy=TestRunner=local \
            --//jax:build_jaxlib=wheel \
            --//jax:build_jax=wheel \
            //tests:cpu_tests //tests:backend_independent_tests
else
      bazel test --config=rbe_${os}_${arch} \
            --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
            --override_repository=xla="${JAXCI_XLA_GIT_DIR}" \
            --test_env=JAX_NUM_GENERATED_CASES=25 \
            --test_env=JAX_SKIP_SLOW_TESTS=true \
            --action_env=JAX_ENABLE_X64="$JAXCI_ENABLE_X64" \
            --test_output=errors \
            --color=yes \
            --//jax:build_jaxlib=wheel \
            --//jax:build_jax=wheel \
            //tests:cpu_tests //tests:backend_independent_tests
fi