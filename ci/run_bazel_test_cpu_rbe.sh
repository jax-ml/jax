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
# Runs Bazel CPU tests with RBE.
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

# Run Bazel CPU tests with RBE.
os=$(uname -s | awk '{print tolower($0)}')
arch=$(uname -m)

bazel_output_base=""
# Adjust os and arch for Windows
if [[  $os  =~ "msys_nt" ]] && [[ $arch =~ "x86_64" ]]; then
  os="windows"
  arch="amd64"
  bazel_output_base="--output_base=C:\actions-runner\_work\bazel_output_base"
fi

if [[ "$JAXCI_BUILD_JAXLIB" == "false" ]]; then
  WHEEL_SIZE_TESTS=""
else
  WHEEL_SIZE_TESTS="//jaxlib/tools:jaxlib_wheel_size_test"
fi

if [[ "$JAXCI_BUILD_JAX" != "false" ]]; then
  WHEEL_SIZE_TESTS="$WHEEL_SIZE_TESTS //:jax_wheel_size_test"
fi

if [[ "$JAXCI_HERMETIC_PYTHON_VERSION" == *"-nogil" ]]; then
  JAXCI_HERMETIC_PYTHON_VERSION=${JAXCI_HERMETIC_PYTHON_VERSION%-nogil}-ft
  FREETHREADED_FLAG_VALUE="yes"
else
  FREETHREADED_FLAG_VALUE="no"
fi

 # TODO(b/446172564): Remove this condition when the test is fixed on all
 # platforms.
if [[ $os == "linux" ]] && [[ $arch == "x86_64" ]]; then
  IGNORE_TESTS=""
else
  IGNORE_TESTS="-//tests/multiprocess:array_test_cpu"
fi

if [[ "$JAXCI_BAZEL_CPU_RBE_MODE" == 'build' ]]; then
    echo "Building RBE CPU tests..."
else
    echo "Running RBE CPU tests..."
fi

test_strategy=""
# When running on Mac or Linux Aarch64, we only build the test targets and
# not run them. These platforms do not have native RBE support so we
# RBE cross-compile them on remote Linux x86 machines. As the tests still
# need to be run on the host machine and because running the tests on a
# single machine can take a long time, we skip running them on these
# platforms in the presubmit jobs.
if [[ $os == "darwin" ]] || ( [[ $os == "linux" ]] && [[ $arch == "aarch64" ]] ); then
    rbe_config=rbe_cross_compile_${os}_${arch}
    if [[ "$JAXCI_BAZEL_CPU_RBE_MODE" == 'test' ]]; then
        test_strategy="--strategy=TestRunner=local"
    fi
else
    rbe_config=rbe_${os}_${arch}
fi

bazel $bazel_output_base $JAXCI_BAZEL_CPU_RBE_MODE \
    --build_runfile_links=false \
    --config=$rbe_config \
    --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
    --@rules_python//python/config_settings:py_freethreaded="$FREETHREADED_FLAG_VALUE" \
    --override_repository=xla="${JAXCI_XLA_GIT_DIR}" \
    --//jax:build_jaxlib=$JAXCI_BUILD_JAXLIB \
    --//jax:build_jax=$JAXCI_BUILD_JAX \
    $test_strategy \
    --test_env=JAX_NUM_GENERATED_CASES=25 \
    --test_env=JAX_SKIP_SLOW_TESTS=true \
    --action_env=JAX_ENABLE_X64="$JAXCI_ENABLE_X64" \
    --test_output=errors \
    --color=yes \
    -- \
    $WHEEL_SIZE_TESTS \
    //tests:cpu_tests //tests:backend_independent_tests \
    //jax/experimental/jax2tf/tests:jax2tf_test_cpu \
    //tests/multiprocess:cpu_tests \
    //jax/experimental/jax2tf/tests/multiprocess:cpu_tests \
    //jaxlib/tools:check_cpu_wheel_sources_test \
    $IGNORE_TESTS