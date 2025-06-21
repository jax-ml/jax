# !/usr/bin/env python3
#
# Copyright 2024 The JAX Authors.
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

# NOTE(ruturaj4): This script automates the build process for JAX and XLA on ROCm,
# allowing for optional uninstallation of existing packages, and custom paths for ROCm and XLA repositories.

import argparse
import os
import shutil
import subprocess
import sys


def get_rocm_version():
    try:
        version = subprocess.check_output(
            "cat /opt/rocm/.info/version | cut -d '-' -f 1", shell=True
        )
        return version.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        print(f"Error fetching ROCm version: {e}")
        return None


def get_rocm_target():
    try:
        target_info = subprocess.check_output(
            "rocminfo | grep gfx | head -n 1", shell=True
        )
        target = target_info.decode("utf-8").split()[1]
        return target
    except subprocess.CalledProcessError as e:
        print(f"Error fetching ROCm target: {e}")
        return None


def uninstall_existing_packages(packages):
    cmd = ["python3", "-m", "pip", "uninstall", "-y"]
    cmd.extend(packages)

    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully uninstalled {packages}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to uninstall {packages}: {e}")


def clean_dist_directory():
    try:
        shutil.rmtree("dist")
        print("Cleaned dist directory.")
    except FileNotFoundError:
        print("dist directory not found, skipping cleanup.")
    except Exception as e:
        print(f"Failed to clean dist directory: {e}")
        sys.exit(1)


def build_jax_xla(xla_path, rocm_version, rocm_target, use_clang, clang_path):
    bazel_options = (
        f"--bazel_options=--override_repository=xla={xla_path}" if xla_path else ""
    )
    clang_option = f"--clang_path={clang_path}" if clang_path else ""
    build_command = [
        "python3",
        "./build/build.py",
        "build"
        f"--use_clang={str(use_clang).lower()}",
        "--wheels=jaxlib,jax-rocm-plugin,jax-rocm-pjrt"
        "--rocm_path=%/opt/rocm-{rocm_version}/",
        "--rocm_version=60",
        f"--rocm_amdgpu_targets={rocm_target}",
        bazel_options,
        "--verbose"
    ]

    if clang_option:
        build_command.append(clang_option)

    print("Executing build command:")
    print(" ".join(build_command))

    try:
        subprocess.run(build_command, check=True)
        print("Build completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        sys.exit(1)


def install_wheel():
    try:
        subprocess.run(
            ["python3", "-m", "pip", "install", "dist/*.whl"], check=True, shell=True
        )
        print("Packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Script to build JAX and XLA on ROCm.")
    parser.add_argument(
        "--clang-path", type=str, default="", help="Specify the Clang compiler path"
    )
    parser.add_argument(
        "--skip-uninstall",
        action="store_true",
        help="Skip uninstall of old versions during package install",
    )
    parser.add_argument(
        "--use-clang", default="false", help="Use Clang compiler if set"
    )
    parser.add_argument(
        "--xla-path", type=str, default="", help="Specify the XLA repository path"
    )

    args = parser.parse_args()

    if args.xla_path:
        args.xla_path = os.path.abspath(args.xla_path)
        print(f"Converted XLA path to absolute: {args.xla_path}")

    rocm_version = get_rocm_version()
    if not rocm_version:
        print("Could not determine ROCm version. Exiting.")
        sys.exit(1)

    rocm_target = get_rocm_target()
    if not rocm_target:
        print("Could not determine ROCm target. Exiting.")
        sys.exit(1)

    if not args.skip_uninstall:
        print("Uninstalling existing packages...")
        packages = ["jax", "jaxlib", "jax-rocm60-pjrt", "jax-rocm60-plugin"]
        uninstall_existing_packages(packages)

    clean_dist_directory()

    print(
        f"Building JAX and XLA with ROCm version: {rocm_version}, Target: {rocm_target}"
    )
    build_jax_xla(
        args.xla_path, rocm_version, rocm_target, args.use_clang, args.clang_path
    )

    install_wheel()


if __name__ == "__main__":
    main()
