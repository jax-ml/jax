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

"""Utility script for parsing metadata from built JAX wheels.

This script scans a directory of .whl files to extract the expected version
(derived from the jaxlib wheel) and determines the complete list of expected
JAX packages, including hardware-specific plugins (CUDA, ROCm).
"""

import argparse
from pathlib import Path
import re
import sys
from typing import Optional


def get_version(wheel_name: str) -> Optional[str]:
  """Extracts the version from a wheel filename.

  Assumes the filename is PEP 427-compliant, meaning the distribution name
  and version themselves do not contain hyphens. This allows the first two
  hyphens to act as reliable delimiters for the version string.

  E.g., jaxlib-0.4.30.dev20240101-cp312-... -> 0.4.30.dev20240101
  """
  match = re.match(r"^[^-]+-([^-]+)-", wheel_name)
  if match:
    return match.group(1)
  return None


def get_expected_packages(wheels: list[Path]) -> list[str]:
  """Determines which plugins to expect based on the downloaded wheels."""
  packages = ["jax", "jaxlib"]

  # This keeps the logic generic across CUDA, ROCm, CPU, and TPU
  if any(
      w.name.startswith(("jax_cuda12_plugin", "jax-cuda12-plugin"))
      for w in wheels
  ):
    packages.extend(["jax-cuda12-plugin", "jax-cuda12-pjrt"])
  elif any(
      w.name.startswith(("jax_cuda13_plugin", "jax-cuda13-plugin"))
      for w in wheels
  ):
    packages.extend(["jax-cuda13-plugin", "jax-cuda13-pjrt"])
  elif any(w.name.startswith(("jax_rocm", "jax-rocm")) for w in wheels):
    packages.extend(["jax-rocm-plugin", "jax-rocm-pjrt"])

  return packages


def parse_metadata(wheel_dir: Path) -> tuple[Optional[str], list[str]]:
  """Parses metadata from the given directory of wheels.

  Returns:
    A tuple of (version, packages_list).
    Returns (None, []) if the directory doesn't exist or has no wheels.
  """
  if not wheel_dir.exists():
    return None, []

  wheels = list(wheel_dir.glob("*.whl"))
  if not wheels:
    return None, []

  # Use just the jaxlib wheel as the canonical anchor to determine the version.
  jaxlib_wheel = next(
      (w.name for w in wheels if w.name.startswith("jaxlib-")), None
  )
  if not jaxlib_wheel:
    raise RuntimeError("jaxlib wheel not found in the specified directory")

  version = get_version(jaxlib_wheel)
  if not version:
    raise RuntimeError(
        f"Could not extract version from anchor wheel {jaxlib_wheel}"
    )

  packages = get_expected_packages(wheels)
  return version, packages


def parse_args() -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--wheel-dir",
      type=Path,
      default=Path("dist"),
      help="Directory containing downloaded .whl files",
  )
  group = parser.add_mutually_exclusive_group()
  group.add_argument(
      "--get-version",
      action="store_true",
      help="Print the extracted version",
  )
  group.add_argument(
      "--get-packages",
      action="store_true",
      help="Print the expected packages",
  )

  return parser.parse_args()


def main():
  args = parse_args()

  version, packages = parse_metadata(args.wheel_dir)

  # Gracefully exit with no output if no wheels were found.
  if not version:
    sys.exit(0)

  if args.get_version:
    print(version)
  elif args.get_packages:
    print(",".join(packages))
  else:
    print(f"Extracted Version: {version}")
    print(f"Expected Packages: {','.join(packages)}")


if __name__ == "__main__":
  main()
