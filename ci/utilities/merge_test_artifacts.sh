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
# Merges files from multiple artifact directories into one directory with
# filename prefixes to avoid collisions.
#
# Usage: merge_test_artifacts.sh output_dir prefix=source_dir [...]
#
# Set JAXCI_MERGE_TEST_ARTIFACTS_TRACE=1 to preserve inherited xtrace output
# while debugging this script.

if [[ "${JAXCI_MERGE_TEST_ARTIFACTS_TRACE:-0}" != "1" ]]; then
  # Many CI entrypoints run with `set -x`; disable it unless explicitly
  # debugging this helper so copying many files stays quiet.
  { set +x; } 2>/dev/null
fi
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 output_dir prefix=source_dir [...]" >&2
  exit 2
fi

OUTPUT_DIR="$1"
shift
LOG_LIMIT="${JAXCI_MERGE_TEST_ARTIFACTS_LOG_LIMIT:-200}"

if [[ ! "$LOG_LIMIT" =~ ^[0-9]+$ ]]; then
  LOG_LIMIT=200
fi
LOG_LIMIT=$((10#$LOG_LIMIT))

GROUP_OPENED=0
if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
  echo "::group::Merge test artifact directories"
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
manifest_tmp="$(mktemp)"
manifest_file="$OUTPUT_DIR/merged-test-artifacts-manifest.tsv"
printf 'source_dir\tsource_file\tdestination\n' > "$manifest_tmp"

total_count=0
for mapping in "$@"; do
  if [[ "$mapping" != *=* ]]; then
    echo "Invalid mapping '$mapping'. Expected prefix=source_dir." >&2
    exit 2
  fi
  prefix="${mapping%%=*}"
  source_dir="${mapping#*=}"
  if [[ -z "$prefix" || -z "$source_dir" ]]; then
    echo "Invalid mapping '$mapping'. Expected non-empty prefix and source_dir." >&2
    exit 2
  fi

  if [[ ! -d "$source_dir" ]]; then
    echo "No $source_dir/ directory found. Skipping."
    continue
  fi

  source_count=0
  # Only merge the flat files created by collect_bazel_test_xmls.sh. Use a
  # NUL-delimited stream so unusual file names are handled correctly.
  while IFS= read -r -d '' artifact_file; do
    basename="$(basename "$artifact_file")"
    destination_name="${prefix}${basename}"
    cp "$artifact_file" "$OUTPUT_DIR/$destination_name"
    printf '%s\t%s\t%s\n' "$source_dir" "$basename" "$destination_name" >> "$manifest_tmp"
    source_count=$((source_count + 1))
    total_count=$((total_count + 1))
  done < <(find "$source_dir" -maxdepth 1 -type f -print0 2>/dev/null || true)

  echo "Merged $source_count file(s) from $source_dir/ with prefix '$prefix'."
done

if [[ "$total_count" -gt 0 ]]; then
  # Publish the manifest only after successful copies, so partial failures do
  # not leave behind a misleading artifact list.
  mv "$manifest_tmp" "$manifest_file"
else
  rm -f "$manifest_tmp"
fi

echo "Merged $total_count test artifact file(s) into $OUTPUT_DIR/"
if [[ "$total_count" -gt 0 ]]; then
  echo "Wrote manifest: $manifest_file"
  if [[ "$LOG_LIMIT" -gt 0 ]]; then
    echo "Merged files:"
    awk -F '\t' -v limit="$LOG_LIMIT" '
      NR > 1 && printed < limit {
        printf "  %s/%s -> %s\n", $1, $2, $3
        printed++
      }
    ' "$manifest_file"
  fi
  if [[ "$total_count" -gt "$LOG_LIMIT" ]]; then
    echo "  ... and $((total_count - LOG_LIMIT)) more file(s); see $manifest_file"
  fi
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" && "$total_count" -gt 0 ]]; then
  {
    echo "### Test artifact merge"
    echo
    echo "- output directory: \`$OUTPUT_DIR\`"
    echo "- files merged: $total_count"
    echo "- manifest: \`$manifest_file\`"
    echo
    summary_label="Merged files"
    if [[ "$total_count" -gt "$LOG_LIMIT" ]]; then
      summary_label="$summary_label (first $LOG_LIMIT)"
    fi
    echo "<details><summary>$summary_label</summary>"
    echo
    echo '```text'
    awk -F '\t' -v limit="$LOG_LIMIT" '
      NR > 1 && printed < limit {
        printf "%s/%s -> %s\n", $1, $2, $3
        printed++
      }
    ' "$manifest_file"
    if [[ "$total_count" -gt "$LOG_LIMIT" ]]; then
      echo "... and $((total_count - LOG_LIMIT)) more file(s)"
    fi
    echo '```'
    echo
    echo "</details>"
    echo
  } >> "$GITHUB_STEP_SUMMARY"
fi
