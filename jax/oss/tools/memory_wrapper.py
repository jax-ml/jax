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

"""Wraps a test binary to record its peak memory usage.

Used via `--run_under=//tools:memory_wrapper` (see the `memprofile` Bazel
config in .bazelrc). Runs the wrapped command as a subprocess and records its
peak RSS to `$TEST_UNDECLARED_OUTPUTS_DIR/memory_profile.json`, which Bazel
zips into `bazel-testlogs/**/test.outputs/outputs.zip` for later collection.
"""

import json
import os
import resource
import subprocess
import sys
import time


def _peak_rss_bytes():
  """Peak RSS of terminated child processes, normalized to bytes.

  ru_maxrss is reported in KB on Linux but bytes on macOS.
  """
  maxrss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
  if sys.platform == "darwin":
    return maxrss
  return maxrss * 1024


def main(argv):
  if len(argv) < 2:
    sys.exit("usage: memory_wrapper.py <command> [args...]")

  command = argv[1:]
  start = time.time()
  result = subprocess.run(command)
  duration = time.time() - start

  outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
  if outputs_dir:
    report = {
        "target": os.environ.get("TEST_TARGET", command[0]),
        "peak_rss_bytes": _peak_rss_bytes(),
        "duration_s": duration,
    }
    report_path = os.path.join(outputs_dir, "memory_profile.json")
    with open(report_path, "w") as f:
      json.dump(report, f)

  return result.returncode


if __name__ == "__main__":
  sys.exit(main(sys.argv))
