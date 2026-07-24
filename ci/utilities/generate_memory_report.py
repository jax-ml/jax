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
import csv
import os
import re
import sys


def main():
  testlogs_dir = "bazel-testlogs"
  if not os.path.exists(testlogs_dir):
    print(f"Directory {testlogs_dir} not found.")
    sys.exit(1)

  report = []

  # Regex to capture time -v memory output
  # e.g., "Maximum resident set size (kbytes): 123456"
  rss_pat = re.compile(r"Maximum resident set size \(kbytes\):\s+(\d+)")

  for root, _, files in os.walk(testlogs_dir):
    if "test.log" in files:
      log_path = os.path.join(root, "test.log")

      # Reconstruct Bazel target from the logs directory
      # bazel-testlogs/tests/core_test/test.log -> //tests:core_test
      rel_path = os.path.relpath(root, testlogs_dir)
      if rel_path == ".":
        continue

      # E.g. tests/core_test -> //tests:core_test
      parts = rel_path.rsplit("/", 1)
      if len(parts) == 2:
        test_target = f"//{parts[0]}:{parts[1]}"
      else:
        test_target = f"//:{parts[0]}"

      try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
          content = f.read()

        m = rss_pat.search(content)
        if m:
          rss_kb = int(m.group(1))
          rss_mb = rss_kb / 1024.0
          report.append((rss_mb, test_target))
      except Exception as e:
        print(f"Error reading {log_path}: {e}")

  if not report:
    print(
        "No memory footprint data found in test logs. Make sure JAXCI_TRACK_TEST_MEMORY=1 was set."
    )
    sys.exit(0)

  report.sort(reverse=True, key=lambda x: x[0])

  out_file = "memory_footprint_report.csv"
  with open(out_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Memory (MB)", "Test Target"])
    for r in report:
      writer.writerow([f"{r[0]:.2f}", r[1]])

  print(f"Memory footprint report generated: {out_file}")
  print("Top tests by memory usage:")
  for r in report[:10]:
    print(f"{r[0]:.2f} MB - {r[1]}")


if __name__ == "__main__":
  main()
