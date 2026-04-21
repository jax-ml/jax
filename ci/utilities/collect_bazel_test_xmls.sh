#!/bin/bash
# Copyright 2026 The JAX Authors.
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
# Collects test.xml files from bazel-testlogs/ and copies them to a target
# directory with path-based naming to avoid collisions.
#
# Bazel generates a separate test.xml per target, scattered across a deep
# directory tree under bazel-testlogs/ (e.g.,
# bazel-testlogs/tests/cpu_tests/test.xml). Pytest, by contrast, writes a
# single XML file to a path you specify via --junitxml. This script normalizes
# bazel's output into a flat directory (test-artifacts/) so the
# upload-test-artifacts action can process both pytest and bazel results
# uniformly.
#
# Usage: collect_bazel_test_xmls.sh [--verbose|-v] [output_dir]
#   --verbose, -v: Print diagnostics about bazel-testlogs discovery.
#   output_dir: Directory to copy XML files to (default: test-artifacts)
#
# Set JAXCI_COLLECT_BAZEL_TEST_XMLS_VERBOSE=1 to enable verbose diagnostics
# without changing call sites.

set -euo pipefail

VERBOSE="${JAXCI_COLLECT_BAZEL_TEST_XMLS_VERBOSE:-0}"
OUTPUT_DIR="test-artifacts"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -v|--verbose)
      VERBOSE=1
      shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      OUTPUT_DIR="$1"
      shift
      ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

TESTLOGS_DIR="bazel-testlogs"

if [[ ! -d "$TESTLOGS_DIR" ]]; then
  echo "No bazel-testlogs directory found. Skipping XML collection."
  if [[ "$VERBOSE" == "1" ]]; then
    echo "DEBUG: pwd=$(pwd)"
    echo "DEBUG: ls -la bazel-*:"
    ls -la bazel-* 2>/dev/null || echo "  (no bazel-* entries)"
  fi
  exit 0
fi

if [[ "$VERBOSE" == "1" ]]; then
  echo "DEBUG: TESTLOGS_DIR=$TESTLOGS_DIR"
  echo "DEBUG: ls -la $TESTLOGS_DIR:"
  ls -la "$TESTLOGS_DIR" 2>/dev/null | head -20 || true
  echo "DEBUG: readlink bazel-testlogs:"
  readlink -f "$TESTLOGS_DIR" 2>/dev/null || echo "  (not a symlink)"
  echo "DEBUG: find $TESTLOGS_DIR -name 'test.xml' | head -10:"
  find -L "$TESTLOGS_DIR" -name "test.xml" 2>/dev/null | head -10 || true
  echo "DEBUG: find $TESTLOGS_DIR -type f | head -20:"
  find -L "$TESTLOGS_DIR" -type f 2>/dev/null | head -20 || true
fi

count=0
while IFS= read -r -d '' xml_file; do
  relative_path="${xml_file#${TESTLOGS_DIR}/}"
  mangled_name="${relative_path//\//__}"
  cp "$xml_file" "$OUTPUT_DIR/$mangled_name"
  count=$((count + 1))
done < <(find -L "$TESTLOGS_DIR" -name "test.xml" -print0 2>/dev/null || true)

echo "Collected $count test XML file(s) into $OUTPUT_DIR/"
