# Copyright 2026 The JAX Authors.
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

"""Checks that files have the correct copyright header."""

# pygrep was unnecessarily slow, so we use a custom script.

import sys
import re2


def check_copyright(filename):
  with open(filename, encoding="utf-8") as f:
    content = f.read(2048)
    return bool(
        re2.search(
            r"(?s)Copyright \d{4} The JAX Authors.*?Licensed under the Apache"
            r" License, Version 2.0",
            content,
        )
    )


def main():
  failed_files = []
  for filename in sys.argv[1:]:
    if not check_copyright(filename):
      failed_files.append(filename)

  if failed_files:
    print("Missing or incorrect copyright notice in the following files:")
    for f in failed_files:
      print(f"  {f}")
    sys.exit(1)


if __name__ == "__main__":
  main()
