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
# Build ROCm JAX artifacts.
# Usage: ./ci/build_rocm_artifacts.sh "<artifact>"
# Supported artifact values are: jax-rocm-plugin, jax-rocm-pjrt
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport

artifact="$1"

# Source default JAXCI environment variables.
source ci/envs/default.env

# Clone XLA at HEAD if path to local XLA is not provided
if [[ -z "$JAXCI_XLA_GIT_DIR" && -z "$JAXCI_CLONE_MAIN_XLA" ]]; then
    export JAXCI_CLONE_MAIN_XLA=1
fi

# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

allowed_artifacts=("jax-rocm-plugin" "jax-rocm-pjrt")

if [[ ! " ${allowed_artifacts[*]} " =~ " ${artifact} " ]]; then
  echo "Error: Invalid artifact: $artifact. Allowed values are: ${allowed_artifacts[*]}"
  exit 1
fi

# Determine the artifact tag flags based on the artifact type, mirroring
# ci/build_artifacts.sh.
if [[ "$JAXCI_ARTIFACT_TYPE" == "release" ]]; then
  artifact_tag_flags="--bazel_options=--repo_env=ML_WHEEL_TYPE=release --bazel_options=--//jaxlib/tools:jaxlib_git_hash=$(git rev-parse HEAD)"
elif [[ "$JAXCI_ARTIFACT_TYPE" == "nightly" ]]; then
  current_date=$(date +%Y%m%d)
  artifact_tag_flags="--bazel_options=--repo_env=ML_WHEEL_BUILD_DATE=${current_date} --bazel_options=--repo_env=ML_WHEEL_TYPE=nightly --bazel_options=--//jaxlib/tools:jaxlib_git_hash=$(git rev-parse HEAD)"
elif [[ "$JAXCI_ARTIFACT_TYPE" == "default" ]]; then
  artifact_tag_flags="--bazel_options=--repo_env=ML_WHEEL_TYPE=custom --bazel_options=--repo_env=ML_WHEEL_BUILD_DATE=$(git show -s --format=%as HEAD) --bazel_options=--repo_env=ML_WHEEL_GIT_HASH=$(git rev-parse HEAD) --bazel_options=--//jaxlib/tools:jaxlib_git_hash=$(git rev-parse HEAD)"
else
  echo "Error: Invalid artifact type: $JAXCI_ARTIFACT_TYPE. Allowed values are: release, nightly, default"
  exit 1
fi

override_xla_repo=""
if [[ "$JAXCI_CLONE_MAIN_XLA" == 1 ]]; then
  override_xla_repo="--bazel_options=--override_repository=xla=${JAXCI_XLA_GIT_DIR}"
fi

wheel_version_suffix_flag=""
if [[ -n "${JAXCI_WHEEL_VERSION_SUFFIX:-}" ]]; then
  wheel_version_suffix_flag="--bazel_options=--repo_env=ML_WHEEL_VERSION_SUFFIX=${JAXCI_WHEEL_VERSION_SUFFIX}"
fi

bazel_startup_options=""
if [[ -n "${JAXCI_BAZEL_OUTPUT_BASE}" ]]; then
  bazel_startup_options="--bazel_startup_options=--output_base=${JAXCI_BAZEL_OUTPUT_BASE}"
fi

echo "Building $artifact..."

python build/build.py build --wheels="$artifact" \
  --bazel_startup_options="--bazelrc=build/rocm/rocm.bazelrc" \
  $bazel_startup_options \
  --bazel_options=--config=rocm_release_wheel \
  --bazel_options=--config=rocm_rbe \
  --python_version=$JAXCI_HERMETIC_PYTHON_VERSION \
  --verbose --detailed_timestamped_log \
  --output_path="$JAXCI_OUTPUT_DIR" \
  $artifact_tag_flags \
  $override_xla_repo \
  $wheel_version_suffix_flag

# Verify manylinux compliance.
./ci/utilities/run_auditwheel.sh
