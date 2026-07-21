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

"""Parses JAX-owned wheel metadata from a directory of local wheels."""

import argparse
import json
from pathlib import Path


_SUPPORTED_JAX_WHEEL_PACKAGES = frozenset([
    "jax",
    "jaxlib",
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
    "jax-cuda13-plugin",
    "jax-cuda13-pjrt",
    "jax-rocm-plugin",
    "jax-rocm-pjrt",
])


def _extract_wheel_package_name(wheel_name: str) -> str | None:
  # Assumes the wheel filename is PEP 427-compliant, so the distribution name
  # and version do not contain hyphens. That makes the first two hyphens
  # reliable delimiters for extracting both fields:
  # https://peps.python.org/pep-0427/
  parts = wheel_name.split("-", 2)
  if len(parts) < 3:
    return None
  return parts[0].replace("_", "-").lower()


def _extract_wheel_version(wheel_name: str) -> str | None:
  parts = wheel_name.split("-", 2)
  if len(parts) < 3:
    return None
  return parts[1]


def parse_expected_wheel_versions(wheel_dir: Path) -> dict[str, str]:
  """Returns a package-to-version map for JAX-owned wheels in `wheel_dir`."""
  if not wheel_dir.exists():
    raise RuntimeError(f"Wheel directory does not exist: {wheel_dir}")

  expected_versions = {}
  for wheel_path in sorted(wheel_dir.glob("*.whl")):
    package_name = _extract_wheel_package_name(wheel_path.name)
    if package_name not in _SUPPORTED_JAX_WHEEL_PACKAGES:
      continue

    version = _extract_wheel_version(wheel_path.name)
    if not version:
      raise RuntimeError(
          f"Could not extract a wheel version from {wheel_path.name}"
      )

    previous_version = expected_versions.get(package_name)
    if previous_version and previous_version != version:
      raise RuntimeError(
          "Found conflicting versions for package "
          f"{package_name}: {previous_version} vs {version}"
      )
    expected_versions[package_name] = version

  if not expected_versions:
    raise RuntimeError(
        f"No supported JAX wheels were found in {wheel_dir}"
    )

  return expected_versions


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--wheel-dir",
      type=Path,
      default=Path("dist"),
      help="Directory containing wheel files.",
  )
  parser.add_argument(
      "--pretty",
      action="store_true",
      help="Pretty-print the parsed JSON.",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  expected_versions = parse_expected_wheel_versions(args.wheel_dir)
  if args.pretty:
    print(json.dumps(expected_versions, indent=2, sort_keys=True))
  else:
    print(json.dumps(expected_versions, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
  main()
