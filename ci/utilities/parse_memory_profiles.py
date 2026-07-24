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

"""Aggregates per-test memory_profile.json files from bazel-testlogs into a Markdown summary report.

Usage:
    bazel test --config=memprofile //lib:all
    python3 ci/utilities/parse_memory_profiles.py [--testlogs-dir bazel-testlogs] [--out memory_report.md]
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


def _render_markdown(reports):
  reports = sorted(reports, key=lambda r: r.get("peak_rss_bytes", 0), reverse=True)
  lines = [
      "## :bar_chart: JAX Test Memory Report",
      "",
      "| Test | Peak memory (MB) | Duration (s) |",
      "| --- | --- | --- |",
  ]
  for r in reports:
    peak_mb = r.get("peak_rss_bytes", 0) / (1024 * 1024)
    lines.append(
        "| {target} | {peak:.1f} | {duration:.2f} |".format(
            target=r.get("target", "unknown"),
            peak=peak_mb,
            duration=r.get("duration_s", 0.0),
        )
    )
  return "\n".join(lines) + "\n"


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--testlogs-dir",
      nargs="+",
      default=["bazel-testlogs", "test-artifacts"],
      help="Directories to search for outputs.zip files.",
  )
  parser.add_argument("--out", default="memory_report.md")
  args = parser.parse_args()

  reports = _find_reports(args.testlogs_dir)
  if not reports:
    raise SystemExit(
        "No memory_profile.json files found under {}. "
        "Did you run `bazel test --config=memprofile //...`?".format(
            ", ".join(args.testlogs_dir)
        )
    )

  markdown = _render_markdown(reports)
  with open(args.out, "w") as f:
    f.write(markdown)
  print(markdown, end="")

if __name__ == "__main__":
  main()
