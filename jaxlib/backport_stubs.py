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

_REPLACEMENTS = [
    (re.compile(r"\btypes\.CapsuleType\b"), "typing_extensions.CapsuleType"),
]

_TYPES_USAGE = re.compile(r"\btypes\.")
_IMPORT_TYPES = re.compile(r"^import types\n", re.MULTILINE)
_TYPING_EXTENSIONS_USAGE = re.compile(r"\btyping_extensions\.")
_IMPORT_TYPING_EXTENSIONS = re.compile(
    r"^import typing_extensions\b", re.MULTILINE
)
_DOCSTRING_END = re.compile(r'^""".*?"""\n', re.MULTILINE | re.DOTALL)


def backport(content: str) -> str:
  for pattern, replacement in _REPLACEMENTS:
    content = pattern.sub(replacement, content)

  if not _TYPES_USAGE.search(content):
    content = _IMPORT_TYPES.sub("", content)

  if _TYPING_EXTENSIONS_USAGE.search(
      content
  ) and not _IMPORT_TYPING_EXTENSIONS.search(content):
    content = _DOCSTRING_END.sub(
        r"\g<0>\nimport typing_extensions\n", content, count=1
    )

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
