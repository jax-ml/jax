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

# Script that can be run at the end of jobs to copy the logs or pytest results to a transient CI bucket

GCS_BUCKET="gs://general-ml-ci-transient/jax-github-actions/logs/$GITHUB_REPOSITORY/$GITHUB_RUN_ID/$GITHUB_RUN_ATTEMPT"

# Check if we are in a Bazel workspace.
if [ ! -f "WORKSPACE" ] && [ ! -f "WORKSPACE.bazel" ]; then
    echo "ERROR: No WORKSPACE file found. Please run this script from the root of your Bazel workspace."
    exit 1
fi

echo "Starting Bazel test log upload to GCS..."
echo "Destination Bucket: $GCS_BUCKET"

# Get the path to the bazel-out directory. We use XDG_CACHE_HOME if set as it doesn't require starting bazel and is much quicker
if [ -n "$XDG_CACHE_HOME" ]; then
    MD5=($(echo -n $(pwd) | md5sum))
    BAZEL_OUT_DIR="$XDG_CACHE_HOME/bazel/_bazel_root/$MD5/execroot/__main__/bazel-out"
else
    BAZEL_OUT_DIR=$(bazel info output_path)
fi

if [ ! -d "$BAZEL_OUT_DIR" ]; then
    echo "ERROR: Could not find the Bazel output directory at '$BAZEL_OUT_DIR'."
    echo "Have you built or tested any targets yet?"
    exit 1
fi

echo "Searching for 'testlogs' directories under: $BAZEL_OUT_DIR"

# Use 'find' to locate all directories named 'testlogs'.
found_logs=0
find "$BAZEL_OUT_DIR" -type d -name "testlogs" | while read -r testlogs_path; do
    found_logs=1
    echo "======================================================================"
    echo "Found testlogs directory: $testlogs_path"

    # To avoid naming collisions in GCS, we create a descriptive path from the
    # log file's location relative to the bazel-out directory.
    # e.g., 'k8-fastbuild/testlogs' becomes 'k8-fastbuild-testlogs'
    relative_path=${testlogs_path#"$BAZEL_OUT_DIR/"}
    gcs_prefix=$(echo "$relative_path" | tr '/' '-')

    # Define the final destination path in the GCS bucket.
    GCS_DESTINATION_PATH="${GCS_BUCKET}/${gcs_prefix}/"

    echo "Uploading contents to: $GCS_DESTINATION_PATH"

    # Use gsutil to copy the entire directory's contents recursively.
    # The '-m' flag enables parallel (multi-threaded/multi-processing) uploads.
    # The '-r' flag copies directories recursively.
    # The trailing '*' ensures the *contents* of the directory are copied.
    # gsutil -m cp -r "${testlogs_path}/*" "$GCS_DESTINATION_PATH"

    echo "Upload complete for $testlogs_path"
done

echo "Found logs $found_logs"
# --- Final Check ---
if [[ "$found_logs" -eq 0 ]]; then
    echo "Log: No 'testlogs' directories were found."
fi

echo "======================================================================"
echo "Script finished."

