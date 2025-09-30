#!/bin/bash
# Copyright 2025 The JAX Authors.
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
# Runs targets one at a time logging memory usage of each target
#
# Example usage:
# bazel test --run_under=/path/to/build/cuda_target_memory_data.sh //tests/...
# 
#
# Environment variables:
#     CUDA_MEMORY_LOG_DIR=path to put memory logs
#     SAMPLE_TIME=how often to sample, defaults to 1

echo "Merging cuda target memory data"
SAMPLE_TIME=${SAMPLE_TIME:-1}
JAX_ACCELERATOR_COUNT=${JAX_ACCELERATOR_COUNT:-1}


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

# We go one accelerator at a time.  Multiple tests per accelerator would mix memory usage
for i in `seq 0 $((JAX_ACCELERATOR_COUNT-1))`; do
  exec {lock_fd}>/var/lock/jax_accelerator_lock_${i} || exit 1
  if flock -n "$lock_fd";
  then
    (
      # This export only works within the brackets, so it is isolated to one
      # single command.
      export TPU_VISIBLE_CHIPS=$i
      export CUDA_VISIBLE_DEVICES=$i
      export HIP_VISIBLE_DEVICES=$i
      NVIDIA_SMI_DUMP_PATH="$TEST_UNDECLARED_OUTPUTS_DIR/memory_log.csv"
      echo "Logging memory info to $NVIDIA_SMI_DUMP_PATH"
      echo "Running test $TEST_BINARY $* on accelerator $i"
      # Have nvidia-smi log data on the cadance of $SAMPLE_TIME to memory_log.csv
      nvidia-smi -i $i --query-gpu=timestamp,name,utilization.memory,memory.used,memory.total,pci.bus_id,driver_version,utilization.gpu --format=csv -l $SAMPLE_TIME -f $NVIDIA_SMI_DUMP_PATH &
      smi_pid=$!
      "$TEST_BINARY" $@
      command_exit=$?

      # Kill nvida smi and wait for exit to ensure file is fully written
      kill $smi_pid && wait $smi_pid

      # Find the max memory in the log file, awk considers the first value it encounters when getting the number so the "MiB" gets removed
      start_memory=$(awk -F',' 'NR==2 {print $4}' $NVIDIA_SMI_DUMP_PATH)
      max_memory=$(awk -F", " '($4+0 > max_value) {max_value = $4+0; max_record = $4} END {print max_record}' $NVIDIA_SMI_DUMP_PATH)
      echo "Max GPU memory: $max_memory, Starting Memory: $start_memory"
      
      # Print out the max to its own file that can be gathered at the end of a run
      echo "$TEST_TARGET,$TEST_SHARD_INDEX,$max_memory,$start_memory" >> $TEST_UNDECLARED_OUTPUTS_DIR/max_memory.csv

      # The exit code is that of the bazel command.  Memory log errors are suppresed. 
      exit $command_exit
    )
    return_code=$?
    # flock locks are automatically released when the FD is closed.
    exit $return_code
  fi
done


echo "Cannot find a free accelerator to run the test $* on, exiting with failure..."
exit 1