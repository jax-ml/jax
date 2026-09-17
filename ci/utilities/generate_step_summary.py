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

"""Generates a GitHub Step Summary from JUnit XML test results and ResultStore links."""

import argparse
import dataclasses
import glob
import os
import sys

# Add postprocess directory to sys.path so we can import xml2json
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "postprocess"))
import xml2json  # pylint: disable=g-import-not-at-top


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Generates a GitHub Step Summary from JUnit XMLs."
  )
  parser.add_argument(
      "description",
      type=str,
      help="Description of the Bazel invocation (e.g. 'CPU RBE tests').",
  )
  parser.add_argument(
      "artifacts_dir",
      type=str,
      help="Directory containing collected test.xml files.",
  )
  parser.add_argument(
      "invocation_id",
      type=str,
      help="The Bazel invocation ID (as comma-separated id:name pairs).",
  )
  parser.add_argument(
      "bazel_exit_code",
      type=int,
      help="The exit code of the Bazel command.",
  )
  parser.add_argument(
      "action_type",
      type=str,
      nargs="?",
      default="test",
      help="The action type ('test' or 'build').",
  )
  return parser.parse_args()


def extract_test_name(record: dict[str, str]) -> str:
  """Formats the human-readable test name, stripping `__main__`."""
  classname = record.get("classname") or ""
  if classname.startswith("__main__."):
    classname = classname[9:]
  elif classname == "__main__":
    classname = ""

  name = record.get("name") or "UnknownTest"
  file_path = record.get("file_path") or ""
  prefix = f"{file_path}::" if file_path else ""
  return f"{prefix}{classname}.{name}" if classname else f"{prefix}{name}"


@dataclasses.dataclass
class TestStats:
  total: int = 0
  passed: int = 0
  failed: int = 0
  skipped: int = 0
  failures: list[tuple[str, str]] = dataclasses.field(default_factory=list)


def parse_junit_xmls(artifacts_dir: str) -> TestStats:
  """Processes directory of JUnit XMLs and returns aggregate totals and failures."""
  stats = TestStats()

  for xml_file in glob.glob(os.path.join(artifacts_dir, "*.xml")):
    for record in xml2json.iter_xml_records(xml_file):
      stats.total += 1
      status = record.get("status", "PASSED")

      if status in ("FAILED", "ERROR"):
        stats.failed += 1
        test_name = extract_test_name(record)
        message = (
            record.get("detail")
            or record.get("message")
            or record.get("system_err")
            or "No failure details provided."
        )
        if len(message) > 500:
          message = message[:500] + "\n... (truncated)"
        stats.failures.append((test_name, message.strip()))
      elif status == "SKIPPED":
        stats.skipped += 1
      else:
        stats.passed += 1

  return stats


def generate_markdown(args, stats=None) -> str:
  lines = []
  status_emoji = "✅" if args.bazel_exit_code == 0 else "❌"
  lines.extend([
      f"## {status_emoji} {args.description}",
      "",
  ])

  lines.append("**ResultStore Link(s):** *(visible to Googlers only)*")
  for inv in args.invocation_id.split(","):
    parts = inv.split(":", 1)
    id_val = parts[0]
    name = parts[1] if len(parts) > 1 else args.description
    url = f"https://source.cloud.google.com/results/invocations/{id_val}"
    lines.append(f"* **{name}:** [{url}]({url})")
  lines.append("")

  if args.action_type == "build":
    if args.bazel_exit_code == 0:
      lines.append("**Build completed successfully.**")
    else:
      lines.extend([
          "> [!WARNING]",
          "> **Build failed.** Please check the workflow logs or ResultStore link for build errors.",
      ])
    return "\n".join(lines) + "\n"

  # Handle test results
  if stats and stats.total == 0 and args.bazel_exit_code != 0:
    lines.extend([
        "> [!WARNING]",
        "> **Build or setup failed before tests could finish running.** Please check the workflow logs or ResultStore link for build errors.",
    ])
  elif stats and stats.total > 0:
    lines.extend([
        "### 📊 Test Results",
        "| Total | Passed | Failed | Skipped |",
        "| :-: | :-: | :-: | :-: |",
        f"| {stats.total} | {stats.passed} | {stats.failed} | {stats.skipped} |",
        "",
    ])

  if stats and stats.failures:
    lines.append(f"### ❌ Failing Tests ({len(stats.failures)})")
    for test_name, message in stats.failures[:20]:
      lines.extend([
          f"<details><summary><code>{test_name}</code></summary>",
          "",
          "```",
          message,
          "```",
          "</details>",
          "",
      ])
    if len(stats.failures) > 20:
      lines.extend([
          f"*... and {len(stats.failures) - 20} more failing test(s).*",
          "",
      ])

  return "\n".join(lines) + "\n"


def main():
  args = parse_args()
  stats = None

  if args.action_type == "test":
    stats = parse_junit_xmls(args.artifacts_dir)

  summary_markdown = generate_markdown(args, stats)

  step_summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
  if step_summary_file:
    try:
      with open(step_summary_file, "a", encoding="utf-8") as f:
        f.write(summary_markdown)
      print(f"Step summary written to {step_summary_file}", file=sys.stderr)
    except Exception as e:
      print(f"Error writing to GITHUB_STEP_SUMMARY: {e}", file=sys.stderr)
  else:
    print(
        "GITHUB_STEP_SUMMARY environment variable not set. Printing summary to"
        " stdout:",
        file=sys.stderr,
    )
    print(summary_markdown)


if __name__ == "__main__":
  main()
