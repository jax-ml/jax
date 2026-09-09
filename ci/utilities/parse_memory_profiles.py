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

"""Aggregates per-test memory_profile.json files from bazel-testlogs into a summary report for log output.

Usage:
    bazel test --run_under="python3 ci/utilities/memory_wrapper.py" //lib:all
    python3 ci/utilities/parse_memory_profiles.py [--testlogs-dir bazel-testlogs] [--threshold-mb 500]
"""

import argparse
import json
import os
import zipfile


def _find_reports(testlogs_dirs):
  reports = []
  for testlogs_dir in testlogs_dirs:
    for root, _dirs, files in os.walk(testlogs_dir):
      if "outputs.zip" not in files:
        continue
      zip_path = os.path.join(root, "outputs.zip")
      try:
        with zipfile.ZipFile(zip_path) as zf:
          if "memory_profile.json" not in zf.namelist():
            continue
          with zf.open("memory_profile.json") as f:
            reports.append(json.load(f))
      except Exception:
        continue
  return reports


def _render_report(reports, threshold_mb=500.0):
  reports = sorted(
      reports, key=lambda r: r.get("peak_rss_bytes", 0), reverse=True
  )
  total_count = len(reports)
  filtered_reports = [
      r
      for r in reports
      if (r.get("peak_rss_bytes", 0) / (1024 * 1024)) >= threshold_mb
  ]
  shown_count = len(filtered_reports)

  max_target_len = max(
      (len(str(r.get("target", "unknown"))) for r in filtered_reports),
      default=0,
  )
  target_width = max(54, max_target_len)
  table_width = max(80, target_width + 26)

  lines = [
      "=" * table_width,
      f"JAX Test Memory Report (>= {threshold_mb:.0f} MB)",
      f"Showing {shown_count} of {total_count} tests.",
      "=" * table_width,
  ]
  if shown_count > 0:
    header = (
        f"{'Test'.ljust(target_width)}  {'Peak (MB)'.rjust(10)}  "
        f"{'Duration (s)'.rjust(12)}"
    )
    lines.extend([
        header,
        "-" * table_width,
    ])
    for r in filtered_reports:
      peak_mb = (r.get("peak_rss_bytes") or 0) / (1024 * 1024)
      duration = r.get("duration_s") or 0.0
      target = str(r.get("target") or "unknown")
      lines.append(
          f"{target.ljust(target_width)}  {peak_mb:10.1f}  {duration:12.2f}"
      )
    lines.append("=" * table_width)
  else:
    lines.extend([
        f"No tests exceeded {threshold_mb:.0f} MB.",
        "=" * table_width,
    ])
  return "\n".join(lines) + "\n"


def _write_json(path, data):
  """Writes JSON data to path, creating parent directories if needed."""
  parent_dir = os.path.dirname(path)
  if parent_dir:
    os.makedirs(parent_dir, exist_ok=True)
  with open(path, "w") as f:
    json.dump(data, f, indent=2)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--testlogs-dir",
      nargs="+",
      default=["bazel-testlogs", "test-artifacts"],
      help="Directories to search for outputs.zip files.",
  )
  parser.add_argument(
      "--out-json",
      default="test-artifacts/memory_report.json",
      help="Optional file path to write aggregated JSON report.",
  )
  parser.add_argument(
      "--threshold-mb",
      type=float,
      default=500.0,
      help="Only include tests with peak memory >= threshold in MB (default: 500).",
  )
  args = parser.parse_args()

  reports = _find_reports(args.testlogs_dir)
  if not reports:
    raise SystemExit(
        "No memory_profile.json files found under {}. "
        "Did you run `bazel test --run_under=\"python3 ci/utilities/memory_wrapper.py\" //...`?".format(
            ", ".join(args.testlogs_dir)
        )
    )

  report = _render_report(reports, threshold_mb=args.threshold_mb)
  print(report, end="")

  if args.out_json:
    _write_json(args.out_json, reports)


if __name__ == "__main__":
  main()
