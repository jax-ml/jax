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

"""Backports generated .pyi stubs to support older Python versions."""

import argparse
import re

_TYPES_CAPSULE_TYPE = re.compile(r"\btypes\.CapsuleType\b")
_TYPES_USAGE = re.compile(r"\btypes\.")
_IMPORT_TYPES = re.compile(r"^import types\n", re.MULTILINE)
_CAPSULE_TYPE_IMPORT = re.compile(
    r"^from typing_extensions import\b.*\bCapsuleType\b", re.MULTILINE
)


def backport(content: str) -> str:
  content, n = _TYPES_CAPSULE_TYPE.subn("CapsuleType", content)

  if n and not _CAPSULE_TYPE_IMPORT.search(content):
    content = _IMPORT_TYPES.sub(
        r"from typing_extensions import CapsuleType\n\g<0>", content
    )

  if not _TYPES_USAGE.search(content):
    content = _IMPORT_TYPES.sub("", content)

  return content


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("files", nargs="+", metavar="FILE")
  args = parser.parse_args()

  for path in args.files:
    with open(path) as f:
      content = f.read()
    content = backport(content)
    with open(path, "w") as f:
      f.write(content)


if __name__ == "__main__":
  main()
