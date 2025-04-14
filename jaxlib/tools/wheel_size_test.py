# Copyright 2025 The JAX Authors.
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

import argparse
import logging
import os


def parse_args():
  """Arguments parser."""
  parser = argparse.ArgumentParser(
      description="Helper for the wheel size verification",
      fromfile_prefix_chars="@",
  )
  parser.add_argument(
      "--wheel-path", required=True, help="Path of the wheel, mandatory"
  )
  parser.add_argument(
      "--max-size-mib",
      required=True,
      help="Maximum size of the wheel in MiB",
  )
  return parser.parse_args()


def verify_wheel_size(args):
  wheel_size_mib = os.path.getsize(args.wheel_path) >> 20
  wheel_name = os.path.basename(args.wheel_path)
  if wheel_size_mib > int(args.max_size_mib):
    raise RuntimeError(
        "The {name} size is {size} MiB, which is larger than the maximum size"
        " {max_size} MiB".format(
            name=wheel_name,
            size=wheel_size_mib,
            max_size=args.max_size_mb,
        )
    )
  else:
    logging.info(
        "The %s size is %s MiB, which is less than the maximum size"
        " %s MB", wheel_name, wheel_size_mib, args.max_size_mib)


if __name__ == "__main__":
  verify_wheel_size(parse_args())
