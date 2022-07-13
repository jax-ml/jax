# Copyright 2022 Google LLC
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
"""Script used in the upstream-dev workflow in order to parse pytest logs."""


import argparse
import json

from pytest import TestReport


MSG_FORMAT = """\
<details><summary>Summary of Failures</summary>

```
{summary}
```

</details>
"""


def main(logfile, outfile):
  failures = []

  with open(logfile, 'r') as f:
    for line in f:
      parsed = json.loads(line)
      report_type = parsed['$report_type']
      if report_type == "TestReport":
        parsed = TestReport._from_json(parsed)
        if parsed.outcome == "failed":
          failures.append(parsed)

  summary = "\n".join(f"{f.nodeid}: {f.longrepr.chain[0][1].message}"
                      for f in failures)
  print(f"writing to {outfile}")
  with open(outfile, 'w') as f:
    f.write(MSG_FORMAT.format(summary=summary))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("logfile", help="The path to the input logfile")
  parser.add_argument("--outfile", help="The path to the parsed output file to be created.",
                      default="parsed_logs.txt")
  args = parser.parse_args()
  main(logfile=args.logfile, outfile=args.outfile)
