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
import re
import json
import datetime
import pandas as pd

stats_pat = r".*collection=train .*(timing/seconds=[\d.]+), (timing/seqs=[\d.]+), (timing/seqs_per_second=[\d.]+), (timing/seqs_per_second_per_core=[\d.]+), (timing/steps_per_second=[\d.]+), (timing/target_tokens_per_second=[\d.]+), (timing/target_tokens_per_second_per_core=[\d.]+).*"

def main(logfile: str, outmd: str, outjson: str, name: str):
    print(f"Extracting content of {logfile}")
    print(f"and writing to {outmd} and {outjson}")

    with open(logfile, 'r') as fp:
        lines = fp.read()
        stats = re.findall(stats_pat,lines)

    data_parsed = [
        # Extract `metric` and `value` from `timings/metric=value`
        {re.split('=|/',s)[1] : float(re.split('=|/',s)[2]) for s in stat}
        for stat in stats
    ]
    df = pd.DataFrame(data_parsed).reset_index(drop=True)
    df.to_markdown(outmd, index=False)

    data = {
        'name': name,
        'date': datetime.datetime.now(tz=None).isoformat(),
        'data': data_parsed,
        'github': {k:v for (k,v) in os.environ.items() if k.startswith('GITHUB')}
    }

    with open(outjson, "w") as ofile:
        ofile.write(json.dumps(data, indent=4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", help="The path to the input logfile")
    parser.add_argument("--outmd", help="The path to the parsed output markdown file to be created.",
                        default="metrics_report.md")
    parser.add_argument("--outjson", help="The path to the parsed output json file to be created.",
                        default="metrics_report.json")
    parser.add_argument("--name", help="Name of the benchmark to be added to the JSON.")
    args = parser.parse_args()
    main(logfile=args.logfile, outmd=args.outmd, outjson=args.outjson, name=args.name)
