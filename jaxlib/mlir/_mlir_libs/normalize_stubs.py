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

"""Normalizes mlir.ir imports in generated .pyi stubs."""

import argparse
import re


def normalize(content: str, *, jaxlib_build: bool = False) -> str:
  if jaxlib_build:
    # If we are building jaxlib, normalize `mlir.ir` to `jaxlib.mlir.ir`.
    content = re.sub(r"\bmlir\.ir", "jaxlib.mlir.ir", content)

  # Replace internal module paths with public ones.
  content = re.sub(r"mlir\._mlir_libs\._mlir\.ir", "mlir.ir", content)

  # Rewrite `import mlir.ir` to `from mlir import ir`.
  content = re.sub(
      r"import (jaxlib\.)?mlir\.ir", r"from \1mlir import ir", content
  )

  # Deduplicate consecutive `from mlir import ir` lines.
  content = re.sub(
      r"(^\s*from (?:jaxlib\.)?mlir import ir\s*\n)"
      r"(?:\s*from (?:jaxlib\.)?mlir import ir\s*\n)+",
      r"\1",
      content,
      flags=re.MULTILINE,
  )

  # Shorten `mlir.ir.<NAME>` to `ir.<NAME>`.
  content = re.sub(r"mlir\.ir\.([a-zA-Z0-9_]+)", r"ir.\1", content)
  return content


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("files", nargs="+", metavar="FILE")
  parser.add_argument(
      "--jaxlib-build",
      action="store_true",
      help="Normalize `mlir.ir` to `jaxlib.mlir.ir`.",
  )
  args = parser.parse_args()

  for path in args.files:
    with open(path) as f:
      content = f.read()
    content = normalize(content, jaxlib_build=args.jaxlib_build)
    with open(path, "w") as f:
      f.write(content)


if __name__ == "__main__":
  main()
