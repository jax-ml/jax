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
if [[ -z "$JAXCI_XLA_GIT_DIR" && -z "$JAXCI_CLONE_MAIN_XLA" ]]; then
    export JAXCI_CLONE_MAIN_XLA=1
fi

# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

OVERRIDE_XLA_REPO=""
if [[ "$JAXCI_CLONE_MAIN_XLA" == 1 ]]; then
  OVERRIDE_XLA_REPO="--override_repository=xla=${JAXCI_XLA_GIT_DIR}"
fi

if [[ "$JAXCI_BUILD_JAXLIB" != "true" ]]; then
  cuda_libs_flag="--config=cuda_libraries_from_stubs"
else
  cuda_libs_flag="--@local_config_cuda//cuda:override_include_cuda_libs=true"
fi

# Run Bazel GPU tests with RBE (single accelerator tests with one GPU apiece).
echo "Running RBE GPU tests..."

if [[ "$JAXCI_HERMETIC_PYTHON_VERSION" == *"-nogil" ]]; then
  JAXCI_HERMETIC_PYTHON_VERSION=${JAXCI_HERMETIC_PYTHON_VERSION%-nogil}-ft
  FREETHREADED_FLAG_VALUE="yes"
else
  FREETHREADED_FLAG_VALUE="no"
fi

PIPSTAR_ARG=""
if [[ "${JAXCI_ENABLE_BZLMOD:-0}" == "0" ]]; then
  PIPSTAR_ARG="--repo_env=RULES_PYTHON_ENABLE_PIPSTAR=0"
fi

bazel test --config=rbe_linux_x86_64_cuda${JAXCI_CUDA_VERSION} \
      --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
      $PIPSTAR_ARG \
      --@rules_python//python/config_settings:py_freethreaded="$FREETHREADED_FLAG_VALUE" \
      $OVERRIDE_XLA_REPO \
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
      //jaxlib/tools:check_gpu_wheel_sources_test
