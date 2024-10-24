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
# Source "JAXCI_" environment variables.
source "ci/utilities/source_jaxci_envs.sh" "$1"
# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

export JAX_SKIP_SLOW_TESTS=true
export JAX_ENABLE_X64=0

# Run Bazel CPU tests with RBE.
if [[ $JAXCI_RUN_BAZEL_TEST_CPU_RBE == 1 ]]; then
      os=$(uname -s | awk '{print tolower($0)}')
      arch=$(uname -m)

      # When running on Mac or Linux Aarch64, we only build the test targets and
      # not run them. These platforms do not have native RBE support so we
      # RBE cross-compile them on remote Linux x86 machines. As the tests still
      # need to be run on the host machine and because running the tests on a
      # single machine can take a long time, we skip running them on these
      # platforms.
      if [[ $os == "darwin" ]] || ( [[ $os == "linux" ]] && [[ $arch == "aarch64" ]] ); then
            echo "Building RBE CPU tests..."
            bazel build --config=rbe_cross_compile_${os}_${arch} \
                  --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
                  --override_repository=xla="${JAXCI_XLA_GIT_DIR}" \
                  --test_env=JAX_NUM_GENERATED_CASES=25 \
                  --test_output=errors \
                  //tests:cpu_tests //tests:backend_independent_tests
      else
            echo "Running RBE CPU tests..."
            bazel test --config=rbe_${os}_${arch} \
                  --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
                  --override_repository=xla="${JAXCI_XLA_GIT_DIR}" \
                  --test_env=JAX_NUM_GENERATED_CASES=25 \
                  --test_output=errors \
                  //tests:cpu_tests //tests:backend_independent_tests
      fi
fi