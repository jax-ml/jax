#!/bin/bash
# Copyright 2024 JAX Authors. All Rights Reserved.
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
# Common setup for all JAX scripts.
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o pipefail: entire command fails if pipe fails. watch out for yes | ...
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -euxo pipefail -o history -o allexport

# Tempoary way to source build configs. In the final version of the CL, these
# will be moved to the Kokoro build configs and the jobs will automatically
# source before running the script.
BUILD_CONFIG_FILE=""
if [[ -n "$BUILD_CONFIG_FILE" ]]; then
  echo "Please set a config file to $BUILD_CONFIG_FILE"
  exit 1
fi
source "$BUILD_CONFIG_FILE"

# Decide whether to use the release tag. JAX CI jobs build from the main
# branch by default. 
if [[ -n "$JAXCI_RELEASE_TAG" ]]; then
  git checkout tags/"$JAXCI_RELEASE_TAG"
fi

# Setup jaxrun, a helper function for executing steps that can either be run
# locally or run under Docker. setup_docker.sh, below, redefines it as "docker
# exec".
# Important: "jaxrun foo | bar" is "( jaxrun foo ) | bar", not "jaxrun (foo | bar)".
# Therefore, "jaxrun" commands cannot include pipes -- which is
# probably for the better. If a pipe is necessary for something, it is probably
# complex. Write a well-documented script under utilities/ to encapsulate the
# functionality instead.
jaxrun() { "$@"; }

# All builds except for Mac run under Docker.
if [[ "$(uname -s)" != "Darwin" ]]; then
  source ./ci/utilities/setup_docker.sh
fi

# Set Bazel configs. Temporary; in the final version of the CL, after the build
# CLI has been reworked, these will be removed. The build CLI will handle
# setting the Bazel configs.
if [[ "$(uname -s)" == "Linux" && $(uname -m) == "x86_64" ]]; then
  if [[ "$JAXCI_BUILD_JAXLIB_ENABLE" == 1 ]]; then
    export BAZEL_CONFIG_CPU=rbe_linux_x86_64_cpu
  else
    export BAZEL_CONFIG_CUDA=rbe_linux_x86_64_cuda
  fi
elif [[ "$(uname -s)" == "Linux" && $(uname -m) == "aarch64" ]]; then
  if [[ "$JAXCI_BUILD_JAXLIB_ENABLE" == 1 ]]; then
    export BAZEL_CONFIG_CPU=ci_linux_aarch64_cpu
  else
    export BAZEL_CONFIG_CUDA=ci_linux_aarch64_cuda
  fi
elif [[ "$(uname -s)" =~ "MSYS_NT" && $(uname -m) == "x86_64" ]]; then
  export BAZEL_CONFIG_CPU=ci_windows_x86_64_cpu
elif [[ "$(uname -s)" == "Darwin" && $(uname -m) == "x86_64" ]]; then
  export BAZEL_CONFIG_CPU=ci_darwin_x86_64_cpu
elif [[ "$(uname -s)" == "Darwin" && $(uname -m) == "arm64" ]]; then
  export BAZEL_CONFIG_CPU=ci_darwin_arm64_cpu
else
  echo "Unsupported platform: $(uname -s) $(uname -m)"
  exit 1
fi

# TODO: cleanup steps