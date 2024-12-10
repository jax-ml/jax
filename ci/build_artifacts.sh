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
if [[ $os =~ "msys_nt"  && $arch == "x86_64" ]]; then
  os="windows"
  arch="amd64"
fi

if [[ "${allowed_artifacts[@]}" =~ "${artifact}" ]]; then

  # Build the jax artifact
  if [[ "$artifact" == "jax" ]]; then
    python -m build --outdir $JAXCI_OUTPUT_DIR
  else

    # Figure out the bazelrc config to use. We will use one of the "rbe_"/"ci_"
    # flags in the .bazelrc depending upon the platform we are building for.
    bazelrc_config="${os}_${arch}"

    # TODO(b/379903748): Add remote cache options for Linux and Windows.
    if [[ "$JAXCI_BUILD_ARTIFACT_WITH_RBE" == 1 ]]; then
      bazelrc_config="rbe_${bazelrc_config}"
    else
      bazelrc_config="ci_${bazelrc_config}"
    fi

    # Use the "_cuda" configs when building the CUDA artifacts.
    if [[ ("$artifact" == "jax-cuda-plugin") || ("$artifact" == "jax-cuda-pjrt") ]]; then
      bazelrc_config="${bazelrc_config}_cuda"
    fi

    # Build the artifact.
    python build/build.py build --wheels="$artifact" --bazel_options=--config="$bazelrc_config" --python_version=$JAXCI_HERMETIC_PYTHON_VERSION --verbose --detailed_timestamped_log

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