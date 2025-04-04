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

# Clone XLA at HEAD if path to local XLA is not provided
if [[ -z "$JAXCI_XLA_GIT_DIR" ]]; then
    export JAXCI_CLONE_MAIN_XLA=1
fi

# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

# Run Bazel GPU tests with RBE (single accelerator tests with one GPU apiece).
echo "Running RBE GPU tests..."

distArray=("cuda_cupti/linux-x86_64/cuda_cupti-linux-x86_64-12.8.57-archive.tar.xz" "cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.57-archive.tar.xz" "libcublas/linux-x86_64/libcublas-linux-x86_64-12.8.3.14-archive.tar.xz" "libcusolver/linux-x86_64/libcusolver-linux-x86_64-11.7.2.55-archive.tar.xz" "libcurand/linux-x86_64/libcurand-linux-x86_64-10.3.9.55-archive.tar.xz" "libcufft/linux-x86_64/libcufft-linux-x86_64-11.3.3.41-archive.tar.xz" "libcusparse/linux-x86_64/libcusparse-linux-x86_64-12.5.7.53-archive.tar.xz" "cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.61-archive.tar.xz" "cuda_nvrtc/linux-x86_64/cuda_nvrtc-linux-x86_64-12.8.61-archive.tar.xz" "libnvjitlink/linux-x86_64/libnvjitlink-linux-x86_64-12.8.61-archive.tar.xz" "cuda_nvml_dev/linux-x86_64/cuda_nvml_dev-linux-x86_64-12.8.55-archive.tar.xz" "cuda_nvtx/linux-x86_64/cuda_nvtx-linux-x86_64-12.8.55-archive.tar.xz" "cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.55-archive.tar.xz" "nvidia_driver/linux-x86_64/nvidia_driver-linux-x86_64-570.86.10-archive.tar.xz")
for str in ${distArray[@]}; do
  timeBefore=$(date +%s)
  arch="https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cuda/redist/$str"
  wget $arch
  timeAfter=$(date +%s)
  diff=$((timeAfter-timeBefore))
  printf "Downloaded $str in %d seconds\n" $diff
  timeBefore=$(date +%s)
  tar -xvf ${arch/*\//}
  timeAfter=$(date +%s)
  diff=$((timeAfter-timeBefore))
  printf "Unzipped $str in %d seconds\n" $diff
done
timeBefore=$(date +%s)
wget "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.8.0.87_cuda12-archive.tar.xz"
timeAfter=$(date +%s)
diff=$((timeAfter-timeBefore))
printf "Downloaded cudnn in %d seconds\n" $diff
timeBefore=$(date +%s)
tar -xvf "cudnn-linux-x86_64-9.8.0.87_cuda12-archive.tar.xz"
timeAfter=$(date +%s)
diff=$((timeAfter-timeBefore))
printf "Unzipped cudnn in %d seconds\n" $diff

bazel test --config=rbe_linux_x86_64_cuda \
      --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
      --override_repository=xla="${JAXCI_XLA_GIT_DIR}" \
      --test_env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
      --test_output=errors \
      --test_env=TF_CPP_MIN_LOG_LEVEL=0 \
      --test_env=JAX_EXCLUDE_TEST_TARGETS=PmapTest.testSizeOverflow \
      --test_tag_filters=-multiaccelerator \
      --test_env=JAX_SKIP_SLOW_TESTS=true \
      --action_env=JAX_ENABLE_X64="$JAXCI_ENABLE_X64" \
      --color=yes \
      --@local_config_cuda//cuda:override_include_cuda_libs=true \
      //tests:gpu_tests //tests:backend_independent_tests \
      //tests/pallas:gpu_tests //tests/pallas:backend_independent_tests \
      //jaxlib/tools:jax_cuda_plugin_wheel_size_test \
      //jaxlib/tools:jax_cuda_pjrt_wheel_size_test \
      //jaxlib/tools:jaxlib_wheel_size_test \
      //:jax_wheel_size_test