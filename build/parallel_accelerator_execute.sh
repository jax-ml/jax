#!/usr/bin/env bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# Coordinates access to accelerators (GPUs/TPUs) between concurrent Bazel tests.
#
# Assigns each test an accelerator and ensures that tests are evenly distributed
# between accelerators.
#
# Example usage:
# bazel test --run_under=/path/to/build/parallel_accelerator_execute.sh //tests/...
# 
#
# Environment variables:
#     JAX_ACCELERATOR_COUNT = Number of accelerators (GPUs/TPUs) available.
#     JAX_TESTS_PER_ACCELERATOR = Number of accelerators (GPUs/TPUs) available.
#     JAX_TPU_XDIST_VISIBILITY_MODE = "chips" or "devices" for TPU assignment.

JAX_ACCELERATOR_COUNT=${JAX_ACCELERATOR_COUNT:-4}
JAX_TESTS_PER_ACCELERATOR=${JAX_TESTS_PER_ACCELERATOR:-8}
JAX_TPU_XDIST_VISIBILITY_MODE=${JAX_TPU_XDIST_VISIBILITY_MODE:-chips}

export TF_PER_DEVICE_MEMORY_LIMIT_MB=${TF_PER_DEVICE_MEMORY_LIMIT_MB:-2048}

# This function is used below in rlocation to check that a path is absolute
function is_absolute {
  [[ "$1" = /* ]] || [[ "$1" =~ ^[a-zA-Z]:[/\\].* ]]
}

# *******************************************************************
#         This section of the script is needed to
#         make things work on windows under msys.
# *******************************************************************
RUNFILES_MANIFEST_FILE="${TEST_SRCDIR}/MANIFEST"
function rlocation() {
  if is_absolute "$1" ; then
    # If the file path is already fully specified, simply return it.
    echo "$1"
  elif [[ -e "$TEST_SRCDIR/$1" ]]; then
    # If the file exists in the $TEST_SRCDIR then just use it.
    echo "$TEST_SRCDIR/$1"
  elif [[ -e "$RUNFILES_MANIFEST_FILE" ]]; then
    # If a runfiles manifest file exists then use it.
    echo "$(grep "^$1 " "$RUNFILES_MANIFEST_FILE" | sed 's/[^ ]* //')"
  fi
}

TEST_BINARY="$(rlocation $TEST_WORKSPACE/${1#./})"
shift
# *******************************************************************

mkdir -p /var/lock
# Try to acquire any of the JAX_ACCELERATOR_COUNT * JAX_TESTS_PER_ACCELERATOR
# slots to run a test at.
#
# Prefer to allocate 1 test per accelerator over 4 tests on 1 accelerator
# So, we iterate over JAX_TESTS_PER_ACCELERATOR first.
for j in `seq 0 $((JAX_TESTS_PER_ACCELERATOR-1))`; do
  for i in `seq 0 $((JAX_ACCELERATOR_COUNT-1))`; do
    exec {lock_fd}>/var/lock/jax_accelerator_lock_${i}_${j} || exit 1
    if flock -n "$lock_fd";
    then
      (
        # This export only works within the brackets, so it is isolated to one
        # single command.
        case "$JAX_TPU_XDIST_VISIBILITY_MODE" in
          devices)
            unset TPU_VISIBLE_CHIPS
            export TPU_VISIBLE_DEVICES=$i
            export TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1,1
            export TPU_PROCESS_BOUNDS=1,1,1,1
            ;;
          chips)
            unset TPU_VISIBLE_DEVICES
            unset TPU_CHIPS_PER_PROCESS_BOUNDS
            unset TPU_PROCESS_BOUNDS
            export TPU_VISIBLE_CHIPS=$i
            ;;
          *)
            echo "Unknown JAX_TPU_XDIST_VISIBILITY_MODE: $JAX_TPU_XDIST_VISIBILITY_MODE"
            exit 1
            ;;
        esac
        export CUDA_VISIBLE_DEVICES=$i
        export ROCR_VISIBLE_DEVICES=$i
        echo "Running test $TEST_BINARY $* on accelerator $i lock_slot=${i}_${j} visibility_mode=${JAX_TPU_XDIST_VISIBILITY_MODE} JAX_TEST_NUM_THREADS=${JAX_TEST_NUM_THREADS:-unset}"
        echo "TPU assignment: TPU_VISIBLE_DEVICES=${TPU_VISIBLE_DEVICES:-unset} TPU_VISIBLE_CHIPS=${TPU_VISIBLE_CHIPS:-unset} TPU_CHIPS_PER_PROCESS_BOUNDS=${TPU_CHIPS_PER_PROCESS_BOUNDS:-unset} TPU_PROCESS_BOUNDS=${TPU_PROCESS_BOUNDS:-unset}"
        "$TEST_BINARY" $@
      )
      return_code=$?
      # flock locks are automatically released when the FD is closed.
      exit $return_code
    fi
  done
done

echo "Cannot find a free accelerator to run the test $* on, exiting with failure..."
exit 1
