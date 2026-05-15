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
# Set up the build environment for JAX CI jobs. This script depends on the
# "JAXCI_" environment variables set or sourced in the build script.

readonly JAXCI_CUDA_OOM_TEST_XLA_COMMIT="4c1b00509e646d13a9cf443cd10c866810ed923d"
readonly JAXCI_CUDA_OOM_TEST_XLA_REVERT_COMMIT="d3a17722ef0781dd17c28f53f80f541a5f8259ed"

# Preemptively mark the JAX git directory as safe. This is necessary for JAX CI
# jobs running on Linux runners in GitHub Actions. Without this, git complains
# that the directory has dubious ownership and refuses to run any commands.
# Avoid running on Windows runners as git runs into issues with not being able
# to lock the config file. Other git commands seem to work on the Windows
# runners so we can skip this step for Windows.
# TODO(b/375073267): Remove this once we understand why git repositories are
# being marked as unsafe inside the self-hosted runners.
if [[ ! $(uname -s) =~ "MSYS_NT" ]]; then
  git config --global --add safe.directory "$JAXCI_JAX_GIT_DIR"
  git config --global --add safe.directory '*'
  git config --global user.name 'JAX CI'
  git config --global user.email 'jax-ci@google.com'
fi

function clone_main_xla() {
  echo "Cloning XLA at HEAD to $(pwd)/xla"
  git clone --depth=256 https://github.com/openxla/xla.git $(pwd)/xla
  cd $(pwd)/xla
  echo "XLA commit: $(git log -1 --format=%H)"
  cd ..
  export JAXCI_XLA_GIT_DIR=$(pwd)/xla
}

if [[ "$JAXCI_CLONE_MAIN_XLA" == 1 ]]; then
  export JAXCI_XLA_COMMIT="$JAXCI_CUDA_OOM_TEST_XLA_COMMIT"
fi

# Clone XLA at HEAD if required.
if [[ "$JAXCI_CLONE_MAIN_XLA" == 1 ]]; then
  # Clone only if $(pwd)/xla does not exist to avoid failure on re-runs.
  if [[ ! -d $(pwd)/xla ]]; then
    clone_main_xla
  else
    echo "JAXCI_CLONE_MAIN_XLA set but local XLA folder already exists: $(pwd)/xla so using that instead."
    # Set JAXCI_XLA_GIT_DIR if local XLA already exists
    export JAXCI_XLA_GIT_DIR=$(pwd)/xla
  fi
fi

# If a XLA commit is provided, check out XLA at that commit.
if [[ -n "${JAXCI_XLA_COMMIT:-}" ]]; then
  # Clone XLA at HEAD if a path to local XLA is not provided.
  if [[ -z "${JAXCI_XLA_GIT_DIR:-}" ]]; then
    clone_main_xla
  fi
  pushd "$JAXCI_XLA_GIT_DIR"

  git fetch --depth=256 origin "$JAXCI_XLA_COMMIT"
  echo "JAXCI_XLA_COMMIT is set. Checking out XLA at $JAXCI_XLA_COMMIT"
  git checkout "$JAXCI_XLA_COMMIT"

  if [[ "$JAXCI_XLA_COMMIT" == "$JAXCI_CUDA_OOM_TEST_XLA_COMMIT" ]]; then
    if [[ "${JAXCI_REVERT_CUDA_OOM_COMMIT:-1}" == "1" ]]; then
      echo "Reverting suspected CUDA OOM XLA commit $JAXCI_CUDA_OOM_TEST_XLA_REVERT_COMMIT"
      git revert --no-edit "$JAXCI_CUDA_OOM_TEST_XLA_REVERT_COMMIT"
      echo "XLA commit after suspect revert: $(git log -1 --format=%H)"
    else
      echo "Leaving suspected CUDA OOM XLA commit $JAXCI_CUDA_OOM_TEST_XLA_REVERT_COMMIT in place"
    fi
  fi

  popd
fi

if [[ -n "${JAXCI_XLA_GIT_DIR:-}" ]]; then
  echo "INFO: Overriding XLA to be read from $JAXCI_XLA_GIT_DIR instead of the"
  echo "pinned version in the WORKSPACE."
  echo "If you would like to revert this behavior, unset JAXCI_CLONE_MAIN_XLA"
  echo "and JAXCI_XLA_COMMIT in your environment. Note that the Bazel RBE test"
  echo "commands overrides the XLA repository and thus require a local copy of"
  echo "XLA to run."
fi

# On Windows, convert MSYS Linux-like paths to Windows paths.
if [[ $(uname -s) =~ "MSYS_NT" ]]; then
  echo 'Converting MSYS Linux-like paths to Windows paths (for Bazel, Python, etc.)'
  # Convert all "JAXCI.*DIR" variables
  source <(python3 ./ci/utilities/convert_msys_paths_to_win_paths.py --convert $(env | grep "JAXCI.*DIR" | awk -F= '{print $1}'))
fi

function retry {
  local cmd="$1"
  local max_attempts=3
  local attempt=1
  local delay=10

  while [[ $attempt -le $max_attempts ]] ; do
    if eval "$cmd"; then
      return 0
    fi
    echo "Attempt $attempt failed. Retrying in $delay seconds..."
    sleep $delay # Prevent overloading

    attempt=$((attempt + 1))
  done
  echo "$cmd failed after $max_attempts attempts."
  exit 1
}

# Retry "bazel --version" 3 times to avoid flakiness when downloading bazel.
retry "bazel --version"

# Create the output directory if it doesn't exist.
mkdir -p "$JAXCI_OUTPUT_DIR"
