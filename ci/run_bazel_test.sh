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

jaxrun nvidia-smi

jaxrun bash -c "export TF_CPP_MIN_LOG_LEVEL=0"
jaxrun bash -c "export NCCL_DEBUG=WARN"

jaxrun "$JAXCI_PYTHON" -c "import jax; print(jax.default_backend()); print(jax.devices()); print(len(jax.devices()))"

# Runs non-multiaccelerator tests with one GPU apiece.
# It appears --run_under needs an absolute path.
jaxrun bazel --bazelrc=ci/.bazelrc test --config="$BAZEL_CONFIG_CUDA" --config=non_multiaccelerator \
  --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
  --run_under "$(pwd)/build/parallel_accelerator_execute.sh" \
  //tests:gpu_tests //tests:backend_independent_tests //tests/pallas:gpu_tests //tests/pallas:backend_independent_tests

# Runs multiaccelerator tests with all GPUs.
# (Below if $1="python3.12" then ${1:6}="3.12")
# bazel test \
#   --repo_env=HERMETIC_PYTHON_VERSION="${1:6}" \
#   --//jax:build_jaxlib=false \
#   --config=nvcc_clang \
#   //tests:gpu_tests \
#   //tests/pallas:gpu_tests \
#   --test_env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
#   --test_output=errors \
#   --jobs=8 \
#   --test_env=JAX_SKIP_SLOW_TESTS=1 \
#   --test_tag_filters=multiaccelerator