#!/bin/bash
# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This script processes test XML files from GCS, converts them to JSON,
# and loads them into BigQuery.
#
# Usage: process_test_results.sh <gcs_upload_uri>

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <gcs_upload_uri> [work_dir]"
  exit 1
fi

GCS_UPLOAD_URI="$1"
if [[ "$GCS_UPLOAD_URI" == /bigstore/* ]]; then
  GCS_UPLOAD_URI="gs://${GCS_UPLOAD_URI#/bigstore/}"
fi
WORK_DIR="${2:-.}"

echo "Processing test results for: $GCS_UPLOAD_URI"
echo "Working directory: $WORK_DIR"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# 1. Download GHA workflow metadata (jobs.json and workflow_run.json) from GCS to extract the run timestamp for fallback.
echo "Downloading workflow metadata..."
gcloud storage cp "${GCS_UPLOAD_URI}/jobs.json" . || true
gcloud storage cp "${GCS_UPLOAD_URI}/workflow_run.json" . || true

# Extract the GHA workflow run start timestamp from jobs.json using lightweight python inline parser.
run_timestamp=""
if [[ -f jobs.json ]]; then
  run_timestamp=$(python3 -c 'import json,sys; jobs=json.load(sys.stdin).get("jobs", []); print(jobs[0].get("created_at", "") if jobs else "")' < jobs.json)
fi
echo "Run timestamp: $run_timestamp"

# 2. Download all test artifacts from GCS.
mkdir -p artifacts
echo "Downloading test artifacts from GCS..."
# List all test-artifacts.zip on GCS and download up to 10 in parallel (-P 10).
# We launch a sub-shell (sh -c) for each file to extract its parent directory name
# and save the zip file into a corresponding local folder under artifacts/ to avoid name collisions.
gcloud storage ls "${GCS_UPLOAD_URI}/*/test-artifacts.zip" | xargs -P 10 -I {} sh -c '
  line="$1"
  dir_name=$(basename "$(dirname "$line")")
  mkdir -p "artifacts/$dir_name"
  gcloud storage cp "$line" "artifacts/$dir_name/"
' -- {} || true

# 3. Extract all XMLs into separate directories to avoid collision in parallel.
mkdir -p extracted
echo "Extracting artifacts in parallel..."
# Find all downloaded zip files and extract them concurrently (-P 10) under extracted/.
# The zip path is safely passed as "$1" to the sub-shell to prevent quoting issues.
# unzip uses quiet mode (-q) to suppress excessive logs and overwrite mode (-o).
find artifacts -name "test-artifacts.zip" | xargs -P 10 -I {} sh -c '
  z="$1"
  dir_name=$(basename "$(dirname "$z")")
  mkdir -p "extracted/$dir_name"
  unzip -q -o "$z" -d "extracted/$dir_name/"
' -- {}

# 4. Run xml2json.py in parallel across directories, passing job_id.
: > all_tests.json
echo "Processing XML files in parallel..."
# Find all extracted platform directories and process up to 10 of them in parallel (-P 10).
# For each directory:
#   1. Read and parse job_id from metadata.json using an inline Python command.
#   2. Find all .xml files in the directory and process up to 5 in parallel (-P 5).
#   3. Execute xml2json.py, redirecting stdout to a corresponding .xml.json file.
# Variables are cleanly passed as arguments to avoid escaping errors ($1=XML path, $2=job_id, $3=SCRIPT_DIR).
find extracted -mindepth 1 -maxdepth 1 -type d | xargs -P 10 -I {} sh -c '
  dir="$1"
  script_dir="$2"
  run_ts="$3"
  if [[ -d "$dir" && -f "$dir/metadata.json" ]]; then
    # Extract job_id from metadata.json using lightweight Python json parser.
    job_id=$(python3 -c "import json,sys; print(json.load(sys.stdin).get(\"job_id\", \"\"))" < "$dir/metadata.json")
    if [[ -n "$job_id" ]]; then
      # Parse each XML file to JSON concurrently (-P 5).
      # We pass run_ts down as "$4" to provide the GHA workflow run timestamp as a fallback for missing XML timestamps.
      find "$dir" -name "*.xml" | xargs -P 5 -I [] sh -c '\''python3 "$3/xml2json.py" "$1" "$2" "$4" > "$1.json"'\'' -- [] "$job_id" "$script_dir" "$run_ts"
    fi
  fi
' -- {} "$SCRIPT_DIR" "$run_timestamp"

# 5. Concatenate all generated JSON files.
find extracted -name "*.xml.json" -exec cat {} + >> all_tests.json

# 6. Concatenate all metadata.json files into all_metadata.json (newline-delimited).
echo "Concatenating job metadata..."
find extracted -name "metadata.json" -exec sh -c 'cat "$1"; echo' -- {} + >> all_metadata.json

# 7. Convert jobs.json to newline-delimited JSON jobs_nd.json.
echo "Converting GHA jobs metadata..."
if [[ -f jobs.json ]]; then
  python3 -c 'import json,sys; print("\n".join(json.dumps(j) for j in json.load(sys.stdin).get("jobs", [])))' < jobs.json > jobs_nd.json
fi

# 8. Load all tables to BigQuery (using --autodetect to automatically create tables/columns).
echo "Loading results to BigQuery..."
if [[ -s all_tests.json ]]; then
  echo "Loading tests table..."
  bq load --source_format=NEWLINE_DELIMITED_JSON --autodetect --ignore_unknown_values --schema_update_option=ALLOW_FIELD_ADDITION jax-dev:jax_ci.tests all_tests.json
fi

if [[ -s all_metadata.json ]]; then
  echo "Loading job_metadata table..."
  bq load --source_format=NEWLINE_DELIMITED_JSON --autodetect --ignore_unknown_values --schema_update_option=ALLOW_FIELD_ADDITION jax-dev:jax_ci.job_metadata all_metadata.json
fi

if [[ -s jobs_nd.json ]]; then
  echo "Loading jobs table..."
  bq load --source_format=NEWLINE_DELIMITED_JSON --autodetect --ignore_unknown_values --schema_update_option=ALLOW_FIELD_ADDITION jax-dev:jax_ci.jobs jobs_nd.json
fi

if [[ -s workflow_run.json ]]; then
  echo "Converting workflow_run.json to single-line JSON..."
  python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)))' < workflow_run.json > workflow_run_nd.json
  echo "Loading workflow_runs table..."
  bq load --source_format=NEWLINE_DELIMITED_JSON --autodetect --ignore_unknown_values --schema_update_option=ALLOW_FIELD_ADDITION jax-dev:jax_ci.workflow_runs workflow_run_nd.json
fi

# 9. Cleanup temporary files.
echo "Cleaning up temporary files..."
rm -rf artifacts extracted all_tests.json all_metadata.json jobs_nd.json jobs.json workflow_run.json workflow_run_nd.json

echo "Done!"
