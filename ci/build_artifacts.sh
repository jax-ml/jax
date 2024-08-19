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
source "ci/utilities/setup.sh"

# Build the jax artifact
if [[ "$JAXCI_BUILD_JAX_ENABLE" == 1 ]]; then
  jaxrun python -m build --outdir $JAXCI_OUTPUT_DIR
fi

# Tempoary; "bazel build" commands will wrapped by the build CLI in the final
# version of the CL.
# Build the jaxlib CPU artifact
if [[ "$JAXCI_BUILD_JAXLIB_ENABLE" == 1 ]]; then
  jaxrun bazel --bazelrc=ci/.bazelrc build --config="$BAZEL_CONFIG_CPU" --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" -- //jaxlib/tools:build_wheel
  jaxrun bazel-bin/jaxlib/tools/build_wheel --output_path=$JAXCI_OUTPUT_DIR --cpu=$(uname -m) --jaxlib_git_hash=$(git rev-parse HEAD)
fi

# Build the jax-cuda-plugin artifact
if [[ "$JAXCI_BUILD_PLUGIN_ENABLE" == 1 ]]; then
  jaxrun bazel --bazelrc=ci/.bazelrc build --config="$BAZEL_CONFIG_CUDA" --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" -- //jaxlib/tools:build_gpu_kernels_wheel
  jaxrun bazel-bin/jaxlib/tools/build_gpu_kernels_wheel --output_path=$JAXCI_OUTPUT_DIR --cpu=$(uname -m) --jaxlib_git_hash=$(git rev-parse HEAD) --enable-cuda=True --platform_version=12
fi

# Build the jax-cuda-pjrt artifact
if [[ "$JAXCI_BUILD_PJRT_ENABLE" == 1 ]]; then
  jaxrun bazel --bazelrc=ci/.bazelrc build --config="$BAZEL_CONFIG_CUDA" --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" -- //jaxlib/tools:build_gpu_plugin_wheel
  jaxrun bazel-bin/jaxlib/tools/build_gpu_plugin_wheel --output_path=$JAXCI_OUTPUT_DIR --cpu=$(uname -m) --jaxlib_git_hash=$(git rev-parse HEAD) --enable-cuda=True --platform_version=12
fi

# After building `jaxlib`, `jaxcuda-plugin`, and `jax-cuda-pjrt`, we run
# `auditwheel show` to ensure manylinux compliance.
if  [[ "$JAXCI_WHEEL_AUDIT_ENABLE" == 1 ]]; then
  jaxrun ./ci/utilities/run_auditwheel.sh
fi
