# Copyright 2022 The JAX Authors.
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
import logging
import traceback

from pytest import TestReport


MSG_FORMAT = """\
<details><summary>Summary of Failures</summary>

```
{summary}
```

</details>
"""


class DefaultReport:
  outcome : str = "none"


def parse_line(line):
  # TODO(jakevdp): should we parse other report types?
  parsed = json.loads(line)
  if parsed.get("$report_type") == "TestReport":
    return TestReport._from_json(parsed)
  return DefaultReport()


def main(logfile, outfile):
  logging.info("Parsing %s", logfile)
  try:
    with open(logfile) as f:
      reports = (parse_line(line) for line in f)
      failures = (r for r in reports if r.outcome == "failed")
      summary = "\n".join(f"{f.nodeid}: {f.longrepr.chain[0][1].message}"
                          for f in failures)
    logging.info("Parsed summary:\n%s", summary)
  except Exception:
    err_info = traceback.format_exc()
    logging.info("Parsing failed:\n%s", err_info)
    summary = f"Log parsing failed; traceback:\n\n{err_info}"
  logging.info("Writing result to %s", outfile)
  with open(outfile, 'w') as f:
    f.write(MSG_FORMAT.format(summary=summary))


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument("logfile", help="The path to the input logfile")
  parser.add_argument("--outfile", help="The path to the parsed output file to be created.",
                      default="parsed_logs.txt")
  args = parser.parse_args()
  main(logfile=args.logfile, outfile=args.outfile)
