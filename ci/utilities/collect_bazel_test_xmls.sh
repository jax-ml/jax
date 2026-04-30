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
# without changing call sites. Set JAXCI_COLLECT_BAZEL_TEST_XMLS_TRACE=1 to
# preserve inherited xtrace output while debugging this script.

if [[ "${JAXCI_COLLECT_BAZEL_TEST_XMLS_TRACE:-0}" != "1" ]]; then
  # Many CI entrypoints run with `set -x`; disable it unless explicitly
  # debugging this helper so copying many files stays quiet.
  { set +x; } 2>/dev/null
fi
set -euo pipefail

VERBOSE="${JAXCI_COLLECT_BAZEL_TEST_XMLS_VERBOSE:-0}"
LOG_LIMIT="${JAXCI_COLLECT_BAZEL_TEST_XMLS_LOG_LIMIT:-200}"
OUTPUT_DIR="test-artifacts"

if [[ ! "$LOG_LIMIT" =~ ^[0-9]+$ ]]; then
  LOG_LIMIT=200
fi
LOG_LIMIT=$((10#$LOG_LIMIT))

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

GROUP_OPENED=0
if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
  echo "::group::Collect Bazel test XML artifacts"
  GROUP_OPENED=1
fi

# `trap ... EXIT` runs on success and failure, which keeps GitHub log groups
# balanced even when a command exits early.
finish_group() {
  if [[ "$GROUP_OPENED" == "1" ]]; then
    echo "::endgroup::"
  fi
}
trap finish_group EXIT

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

manifest_tmp="$(mktemp)"
manifest_file="$OUTPUT_DIR/bazel-test-artifacts-manifest.tsv"
printf 'source\tdestination\n' > "$manifest_tmp"

count=0
# `find -print0` plus `read -d ''` preserves file names with spaces or other
# shell-sensitive characters.
while IFS= read -r -d '' xml_file; do
  relative_path="${xml_file#${TESTLOGS_DIR}/}"
  mangled_name="${relative_path//\//__}"
  cp "$xml_file" "$OUTPUT_DIR/$mangled_name"
  printf '%s\t%s\n' "$relative_path" "$mangled_name" >> "$manifest_tmp"
  count=$((count + 1))
done < <(find -L "$TESTLOGS_DIR" -name "test.xml" -print0 2>/dev/null || true)

if [[ "$count" -gt 0 ]]; then
  # Publish the manifest only after successful copies, so partial failures do
  # not leave behind a misleading artifact list.
  mv "$manifest_tmp" "$manifest_file"
else
  rm -f "$manifest_tmp"
fi

echo "Collected $count test XML file(s) into $OUTPUT_DIR/"
if [[ "$count" -gt 0 ]]; then
  echo "Wrote manifest: $manifest_file"
  if [[ "$LOG_LIMIT" -gt 0 ]]; then
    echo "Copied Bazel XML files:"
    awk -F '\t' -v limit="$LOG_LIMIT" '
      NR > 1 && printed < limit {
        printf "  %s -> %s\n", $1, $2
        printed++
      }
    ' "$manifest_file"
  fi
  if [[ "$count" -gt "$LOG_LIMIT" ]]; then
    echo "  ... and $((count - LOG_LIMIT)) more file(s); see $manifest_file"
  fi
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" && "$count" -gt 0 ]]; then
  {
    echo "### Bazel test XML collection"
    echo
    echo "- output directory: \`$OUTPUT_DIR\`"
    echo "- XML files copied: $count"
    echo "- manifest: \`$manifest_file\`"
    echo
    summary_label="Copied files"
    if [[ "$count" -gt "$LOG_LIMIT" ]]; then
      summary_label="$summary_label (first $LOG_LIMIT)"
    fi
    echo "<details><summary>$summary_label</summary>"
    echo
    echo '```text'
    awk -F '\t' -v limit="$LOG_LIMIT" '
      NR > 1 && printed < limit {
        printf "%s -> %s\n", $1, $2
        printed++
      }
    ' "$manifest_file"
    if [[ "$count" -gt "$LOG_LIMIT" ]]; then
      echo "... and $((count - LOG_LIMIT)) more file(s)"
    fi
    echo '```'
    echo
    echo "</details>"
    echo
  } >> "$GITHUB_STEP_SUMMARY"
fi
