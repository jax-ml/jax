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
# Computes the Bazel flags that stamp a wheel's version, for every
# ci/build_*_artifacts.sh script.
#
# Reads JAXCI_ARTIFACT_TYPE and JAXCI_WHEEL_VERSION_SUFFIX, sets
# artifact_tag_flags. Callers expand it unquoted: it must be word split.
#
# Sourced, not executed, so the "exit 1" calls below fail the calling build.

# Determine the artifact tag flags based on the artifact type. A release
# wheel is tagged with the release version (e.g. 0.5.1), a nightly wheel is
# tagged with the release version and a nightly suffix that contains the
# current date (e.g. 0.5.2.dev20250227), and a default wheel is tagged with
# the git commit hash of the HEAD of the current branch and the date of the
# commit (e.g. 0.5.1.dev20250128+3e75e20c7).
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

# The only place that emits ML_WHEEL_VERSION_SUFFIX, so no build gets it twice.
if [[ -n "$JAXCI_WHEEL_VERSION_SUFFIX" ]]; then
  artifact_tag_flags="${artifact_tag_flags} --bazel_options=--repo_env=ML_WHEEL_VERSION_SUFFIX=${JAXCI_WHEEL_VERSION_SUFFIX}"
fi
