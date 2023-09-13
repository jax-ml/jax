# Copyright 2023 The JAX Authors.
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

"""Utilities for the building JAX related python packages."""

import os
import platform
import shutil
import sys
import subprocess
import glob


def is_windows() -> bool:
  return sys.platform.startswith("win32")


def copy_file(
    src_file: str,
    dst_dir: str,
    dst_filename=None,
    from_runfiles=True,
    runfiles=None,
) -> None:
  if from_runfiles:
    src_file = runfiles.Rlocation(src_file)
  src_filename = os.path.basename(src_file)
  dst_file = os.path.join(dst_dir, dst_filename or src_filename)
  if is_windows():
    shutil.copyfile(src_file, dst_file)
  else:
    shutil.copy(src_file, dst_file)


def platform_tag(cpu: str) -> str:
  platform_name, cpu_name = {
    ("Linux", "x86_64"): ("manylinux2014", "x86_64"),
    ("Linux", "aarch64"): ("manylinux2014", "aarch64"),
    ("Linux", "ppc64le"): ("manylinux2014", "ppc64le"),
    ("Darwin", "x86_64"): ("macosx_10_14", "x86_64"),
    ("Darwin", "arm64"): ("macosx_11_0", "arm64"),
    ("Windows", "AMD64"): ("win", "amd64"),
  }[(platform.system(), cpu)]
  return f"{platform_name}_{cpu_name}"


def build_wheel(sources_path: str, output_path: str, package_name: str) -> None:
  """Builds a wheel in `output_path` using the source tree in `sources_path`."""
  subprocess.run([sys.executable, "-m", "build", "-n", "-w"],
                 check=True, cwd=sources_path)
  for wheel in glob.glob(os.path.join(sources_path, "dist", "*.whl")):
    output_file = os.path.join(output_path, os.path.basename(wheel))
    sys.stderr.write(f"Output wheel: {output_file}\n\n")
    sys.stderr.write(f"To install the newly-built {package_name} wheel, run:\n")
    sys.stderr.write(f"  pip install {output_file} --force-reinstall\n\n")
    shutil.copy(wheel, output_path)


def build_editable(
    sources_path: str, output_path: str, package_name: str
) -> None:
  sys.stderr.write(
    f"To install the editable {package_name} build, run:\n\n"
    f"  pip install -e {output_path}\n\n"
  )
  shutil.rmtree(output_path, ignore_errors=True)
  shutil.copytree(sources_path, output_path)
