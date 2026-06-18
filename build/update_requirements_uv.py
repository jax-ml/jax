#!/usr/bin/env python3
#
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
#
"""Updates JAX requirements_lock.txt files using uv (without Bazel)."""

import argparse
import os
import shlex
import subprocess
import sys

# Define the mapping of Python versions to their lock files and configurations
PYTHON_VERSIONS = {
    "3.12": {"ft": False, "dest": "build/requirements_lock_3_12.txt"},
    "3.13": {"ft": False, "dest": "build/requirements_lock_3_13.txt"},
    "3.13-ft": {"ft": True, "dest": "build/requirements_lock_3_13_ft.txt"},
    "3.14": {"ft": False, "dest": "build/requirements_lock_3_14.txt"},
    "3.14-ft": {"ft": True, "dest": "build/requirements_lock_3_14_ft.txt"},
}

COMMON_SRCS = [
    "build/requirements.in",
    "build/test-requirements.txt",
    "build/nvidia-requirements.txt",
]

def update_requirements(py_ver, nightly=False, upgrade=True, dry_run=False):
    if py_ver not in PYTHON_VERSIONS:
        print(f"Error: Unsupported Python version {py_ver}")
        sys.exit(1)

    config = PYTHON_VERSIONS[py_ver]
    dest = config["dest"]
    is_ft = config["ft"]

    # 1. Determine input files
    srcs = list(COMMON_SRCS)
    if is_ft:
        srcs.append("build/freethreading-requirements.txt")
    else:
        srcs.append("build/nonfreethreading-requirements.txt")

    # Verify inputs exist
    for src in srcs:
        if not os.path.exists(src):
            print(f"Error: Input file {src} does not exist.")
            sys.exit(1)

    # 2. Construct uv pip compile command
    cmd = ["uv", "pip", "compile", "--no-strip-extras", "--universal"]

    # Add inputs and output
    cmd.extend(srcs)
    cmd.extend(["-o", dest])

    # Target Python version for resolution (e.g., "3.13-ft" -> "3.13")
    uv_py_ver = py_ver.split("-")[0]
    cmd.extend(["--python-version", uv_py_ver])

    # 3. Apply flag logic matching BUILD.bazel
    if nightly:
        cmd.extend([
            "--extra-index-url",
            "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple",
            "--pre",
            "--upgrade",
        ])
    else:
        cmd.append("--generate-hashes")
        if upgrade:
            cmd.append("--upgrade")

    # Add a header to the output file indicating how it was generated
    cmd.extend([
        "--custom-compile-command",
        "python build/update_requirements_uv.py",
    ])

    # 4. Execute
    print(f"Running: {shlex.join(cmd)}")
    if not dry_run:
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully updated {dest}")
        except subprocess.CalledProcessError as e:
            print(f"Error running uv: {e}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Update JAX requirements_lock files using uv"
    )
    parser.add_argument(
        "--nightly",
        action="store_true",
        help="Update using nightly/pre-release wheels.",
    )
    parser.add_argument(
        "--no-upgrade",
        action="store_false",
        dest="upgrade",
        help="Do not upgrade packages; only re-resolve constraints.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be run without executing it.",
    )
    parser.set_defaults(upgrade=True)

    args = parser.parse_args()

    # Ensure we are run from the repo root
    if not os.path.exists("build/requirements.in"):
        print("Error: This script must be run from the root of the JAX repository.")
        sys.exit(1)

    # Check if uv is installed
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "Error: 'uv' binary not found. Please install uv first"
            " (https://github.com/astral-sh/uv)."
        )
        sys.exit(1)

    versions_to_update = list(PYTHON_VERSIONS.keys())

    for ver in versions_to_update:
        print(f"\n--- Updating Python {ver} ---")
        update_requirements(
            ver, nightly=args.nightly, upgrade=args.upgrade, dry_run=args.dry_run
        )

if __name__ == "__main__":
    main()
