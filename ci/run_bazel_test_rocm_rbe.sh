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

# Run Bazel GPU tests with RBE (single accelerator tests with one GPU apiece).
echo "Running RBE GPU tests..."

TAG_FILTERS="jax_test_gpu,-config-cuda-only,-manual"

for arg in "$@"; do
    if [[ "$arg" == "--config=rocm_multi_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},multiaccelerator"
    fi
    if [[ "$arg" == "--config=rocm_single_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},gpu,-multiaccelerator"
    fi
done

bazel test --config=rocm_rbe \
    --config=rocm \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
    --test_env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    --test_output=errors \
    --test_env=TF_CPP_MIN_LOG_LEVEL=0 \
    --test_env=JAX_EXCLUDE_TEST_TARGETS=PmapTest.testSizeOverflow \
    --build_tag_filters=${TAG_FILTERS} \
    --test_tag_filters=${TAG_FILTERS} \
    --remote_download_outputs=minimal \
    --test_env=JAX_SKIP_SLOW_TESTS=true \
    --action_env=JAX_ENABLE_X64="$JAXCI_ENABLE_X64" \
    --repo_env=TF_AMD_GPU_TARGETS="gfx908,gfx90a,gfx942" \
    --color=yes \
    --//jax:build_jaxlib=$JAXCI_BUILD_JAXLIB \
    --//jax:build_jax=$JAXCI_BUILD_JAX \
    $@ \
    //tests:gpu_tests \
    //tests:backend_independent_tests \
    //tests/pallas:gpu_tests \
    //tests/pallas:backend_independent_tests \
    //jaxlib/tools:check_gpu_wheel_sources_test
