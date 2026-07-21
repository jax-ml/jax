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

echo "::group::Setup Environment" >&2
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
    OVERRIDE_XLA_REPO="--override_repository=xla=${JAXCI_XLA_GIT_DIR} --override_module=xla=${JAXCI_XLA_GIT_DIR}"
fi

# Run Bazel GPU tests with RBE (single accelerator tests with one GPU apiece).
echo "Running RBE GPU tests..."

TAG_FILTERS="jax_test_gpu,-config-cuda-only,-manual"

# JAXCI_GATE_TARGETS_FILE selects which Bazel target pattern file to use.
# Defaults to the full CI suite; set to build/rocm/ci_blocking_test_targets.txt
# for the PR blocking gate.
TARGETS_FILE="${JAXCI_GATE_TARGETS_FILE:-build/rocm/ci_test_targets.txt}"

for arg in "$@"; do
    if [[ "$arg" == "--config=multi_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},multiaccelerator"
    fi
    if [[ "$arg" == "--config=single_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},gpu,-multiaccelerator"
    fi
done

TEST_ARTIFACTS_DIR="test-artifacts"
mkdir -p "$TEST_ARTIFACTS_DIR"
echo "::endgroup::" >&2

echo "::group::Bazel ROCm RBE tests" >&2
bazel --bazelrc=build/rocm/rocm.bazelrc test \
    --profile="$TEST_ARTIFACTS_DIR/bazel_profile.json.gz" \
    --config=rocm_clang_hermetic \
    --config=rocm_rbe_dynamic \
    $OVERRIDE_XLA_REPO \
    --test_env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    --test_output=errors \
    --test_env=TF_CPP_MIN_LOG_LEVEL=0 \
    --test_env=JAX_EXCLUDE_TEST_TARGETS=PmapTest.testSizeOverflow \
    --build_tag_filters=${TAG_FILTERS} \
    --test_tag_filters=${TAG_FILTERS} \
    --remote_download_regex='.*test\.xml$' \
    --test_env=JAX_SKIP_SLOW_TESTS=true \
    --repo_env=TF_ROCM_AMDGPU_TARGETS="gfx908,gfx90a,gfx942,gfx950" \
    --color=yes \
    $@ \
    --spawn_strategy=local \
    --target_pattern_file="${TARGETS_FILE}" || bazel_retval=$?
echo "::endgroup::" >&2

echo "::group::Cleanup" >&2
ci/utilities/collect_bazel_test_xmls.sh "$TEST_ARTIFACTS_DIR"
echo "::endgroup::" >&2
exit "${bazel_retval:-0}"
