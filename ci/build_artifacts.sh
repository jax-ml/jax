#!/bin/bash
# Copyright 2024 The JAX Authors.
##
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
# Build JAX artifacts.
# Usage: ./ci/build_artifacts.sh "<artifact>"
# Supported artifact values are: jax, jaxlib, jax-cuda-plugin, jax-cuda-pjrt
# E.g: ./ci/build_artifacts.sh "jax" or ./ci/build_artifacts.sh "jaxlib"
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

# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

allowed_artifacts=("jax" "jaxlib" "jax-cuda-plugin" "jax-cuda-pjrt")

os=$(uname -s | awk '{print tolower($0)}')
arch=$(uname -m)

# Adjust the values when running on Windows x86 to match the config in
# .bazelrc
if [[ $os =~ "msys_nt" ]] && [[ $arch == "x86_64" ]]; then
  os="windows"
  arch="amd64"
fi

if [[ " ${allowed_artifacts[@]} " =~ " ${artifact} " ]]; then

  # Build the jax artifact
  if [[ "$artifact" == "jax" ]]; then
    python -m build --outdir $JAXCI_OUTPUT_DIR
  else

    # For bazel builds, use the "rbe_" config for Linux x86/Windows and "ci_" for other platforms
    bazelrc_config="${os}_${arch}"
    if ( [[ "$os" == "linux" ]] && [[ "$arch" == "x86_64" ]] ) || [[ "$os" == "windows" ]]; then
      bazelrc_config="rbe_$bazelrc_config"
    else
      bazelrc_config="ci_$bazelrc_config"
    fi

    # Build the jaxlib CPU artifact
    if [[ "$artifact" == "jaxlib" ]]; then
      python build/build.py build --wheels="jaxlib" --bazel_options=--config="$bazelrc_config" --python_version=$JAXCI_HERMETIC_PYTHON_VERSION --verbose
    fi

    # Build the jax-cuda-plugin artifact
    if [[ "$artifact" == "jax-cuda-plugin" ]]; then
      python build/build.py build --wheels="jax-cuda-plugin" --bazel_options=--config="${bazelrc_config}_cuda" --python_version=$JAXCI_HERMETIC_PYTHON_VERSION --verbose
    fi

    # Build the jax-cuda-pjrt artifact
    if [[ "$artifact" == "jax-cuda-pjrt" ]]; then
      python build/build.py build --wheels="jax-cuda-pjrt" --bazel_options=--config="${bazelrc_config}_cuda" --verbose
    fi

    # If building `jaxlib` or `jax-cuda-plugin` or `jax-cuda-pjrt` for Linux, we
    # run `auditwheel show` to verify manylinux compliance.
    if  [[ "$os" == "linux" ]]; then
      ./ci/utilities/run_auditwheel.sh
    fi

  fi

else
  echo "Error: Invalid artifact: $artifact. Allowed values are: ${allowed_artifacts[@]}"
  exit 1
fi