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
"""Script used in the nightly-ci-multiprocess-gpu workflow to process logs."""

import argparse
import os

ISSUE_FORMAT = """\
<details><summary>Failure summary {name}</summary>

```
{content}
```

</details>
"""

def main(logfiles: list[str], outfile: str):
  print(f"extracting content of {logfiles}")
  print(f"and writing to {outfile}")
  with open(outfile, 'w') as f:
    for logfile in logfiles:
      content = open(logfile).read()
      f.write(ISSUE_FORMAT.format(name=os.path.basename(logfile), content=content))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("logfiles", nargs="+", help="The path to the input logfiles")
  parser.add_argument("--outfile", help="The path to the parsed output file to be created.",
                      default="parsed_logs.txt")
  args = parser.parse_args()
  main(logfiles=args.logfiles, outfile=args.outfile)
