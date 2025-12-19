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
# Runs Bazel GPU tests with RBE. This runs single accelerator tests with one
# GPU apiece on RBE.
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

if [[ "$JAXCI_BUILD_JAXLIB" == "false" ]]; then
  WHEEL_SIZE_TESTS=""
else
  WHEEL_SIZE_TESTS="//jaxlib/tools:jax_cuda_plugin_wheel_size_test \
      //jaxlib/tools:jax_cuda_pjrt_wheel_size_test \
      //jaxlib/tools:jaxlib_wheel_size_test"
fi

if [[ "$JAXCI_BUILD_JAX" != "false" ]]; then
  WHEEL_SIZE_TESTS="$WHEEL_SIZE_TESTS //:jax_wheel_size_test"
fi

if [[ "$JAXCI_BUILD_JAXLIB" != "true" ]]; then
  cuda_libs_flag="--config=cuda_libraries_from_stubs"
else
  cuda_libs_flag="--@local_config_cuda//cuda:override_include_cuda_libs=true"
fi

# Run Bazel GPU tests with RBE (single accelerator tests with one GPU apiece).
echo "Running RBE GPU tests..."

bazel test --config=rbe_linux_x86_64_cuda${JAXCI_CUDA_VERSION} \
      --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
      --override_repository=xla="${JAXCI_XLA_GIT_DIR}" \
      --test_env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
      --test_output=errors \
      --test_env=TF_CPP_MIN_LOG_LEVEL=0 \
      --test_env=JAX_EXCLUDE_TEST_TARGETS=PmapTest.testSizeOverflow \
      --test_tag_filters=-multiaccelerator \
      --test_env=JAX_SKIP_SLOW_TESTS=true \
      --action_env=JAX_ENABLE_X64="$JAXCI_ENABLE_X64" \
      --color=yes \
      $cuda_libs_flag \
      --config=hermetic_cuda_umd \
      --//jax:build_jaxlib=$JAXCI_BUILD_JAXLIB \
      --//jax:build_jax=$JAXCI_BUILD_JAX \
      //tests:gpu_tests //tests:backend_independent_tests \
      //tests/pallas:gpu_tests //tests/pallas:backend_independent_tests \
      //jaxlib/tools:check_gpu_wheel_sources_test \
      $WHEEL_SIZE_TESTS
