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
# Gathers the logs of the previous bazel invocations cuda memory usage and 
# Puts it into a single file
#
# Example usage:
# bazel test --run_under=/path/to/build/cuda_target_memory_data.sh //tests/...
# 


# Set the input directory from the first command-line argument ($1)
LOG_FILES_DIR="./bazel-out/k8-opt/testlogs/tests/"
OUTPUT_FILE="combined_max_memory.csv"
TARGET_FILENAME="max_memory.csv"

# --- 1. Validate Input ---
if [ -z "$LOG_FILES_DIR" ]; then
    echo "Error: Please provide the log directory path."
    echo "Usage: $0 /path/to/log_files_directory"
    exit 1
fi

if [ ! -d "$LOG_FILES_DIR" ]; then
    echo "Error: Directory '$LOG_FILES_DIR' not found."
    exit 1
fi

echo "--- Starting Log Consolidation ---"
echo "Searching recursively in: $LOG_FILES_DIR"

# Count files before the operation for the final message
FILE_COUNT=$(find "$LOG_FILES_DIR" -type f -name "$TARGET_FILENAME" | wc -l)

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "No files named '$TARGET_FILENAME' found in '$LOG_FILES_DIR'. Exiting."
    exit 0
fi

# --- 2. Concatenate All Files (Including All Headers) ---
echo "Concatenating the full content of all $FILE_COUNT files into: $OUTPUT_FILE"
# Find all matching files recursively and use -exec cat {} + to combine their full contents.
# The output is redirected (>) to create the new output file.
find "$LOG_FILES_DIR" -type f -name "$TARGET_FILENAME" -exec cat {} + > "$OUTPUT_FILE"

# --- 3. Completion Message ---
if [ -s "$OUTPUT_FILE" ]; then
    echo "--- Consolidation Complete! ---"
    echo "Successfully combined $FILE_COUNT files into '$OUTPUT_FILE'."
else
    echo "An unexpected error occurred or the files were empty. $OUTPUT_FILE is empty."
    rm -f "$OUTPUT_FILE"
    exit 1
fi
