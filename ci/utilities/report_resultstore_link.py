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

"""Reports a ResultStore link to GitHub Actions."""

import argparse
import sys


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Reports a ResultStore link to GitHub Actions."
  )
  parser.add_argument(
      "description",
      type=str,
      help="Description of the Bazel invocation (e.g. 'CPU tests').",
  )
  parser.add_argument(
      "invocation_id",
      type=str,
      help="The Bazel invocation ID.",
  )
  parser.add_argument(
      "bazel_exit_code",
      type=int,
      help="The exit code of the Bazel command.",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  url = f"https://source.cloud.google.com/results/invocations/{args.invocation_id}"
  title = f"ResultStore Link: {args.description}"

  if args.bazel_exit_code != 0:
    # Report as ERROR (red, auto-expanded)
    # We use %0A to represent newlines in GHA workflow commands
    message = (
        f"{args.description} failed (ResultStore: {url} visible to Googlers only)%0A%0A"
        f"To view the Bazel output, expand the 'Bazel {args.description}' log group."
    )
    print(f"::error title={title}::{message}", file=sys.stderr)
  else:
    # Report as NOTICE (blue, collapsed by default)
    message = f"{args.description} passed (ResultStore: {url} visible to Googlers only)"
    print(f"::notice title={title}::{message}", file=sys.stderr)


if __name__ == "__main__":
  main()
